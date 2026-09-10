from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io_utils import (
    git_revision,
    git_status,
    iter_jsonl,
    load_torch,
    normalise_question,
    prediction_answers,
    retrieval_semantic_sha256,
    runtime_package_versions,
    save_torch,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)


CORE_FILES = (
    "retrieve/inference.py",
    "retrieve/eval.py",
    "retrieve/emb.py",
    "reason/main.py",
    "reason/preprocess/prepare_data.py",
    "reason/preprocess/prepare_prompts.py",
    "reason/llm_utils.py",
    "reason/prompts.py",
    "reason/metrics/evaluate_results.py",
    "reason/metrics/evaluate_results_corrected.py",
)


def _triple(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"Malformed triple: {value!r}")
    return str(value[0]), str(value[1]), str(value[2])


def prompt_triples(sample: dict[str, Any], budget: int) -> list[tuple[str, str, str]]:
    """Mirror reason/preprocess/prepare_prompts.py scored-K semantics."""
    seen: set[tuple[str, str, str]] = set()
    result = []
    for value in sample.get("scored_triples", sample.get("scored_triplets", [])):
        triple = _triple(value)
        if triple not in seen:
            seen.add(triple)
            result.append(triple)
        if len(result) >= budget:
            break
    return result


def _serialise_sample(sample_id: str, row_index: int, sample: dict[str, Any], prompt_budget: int) -> dict[str, Any]:
    scored = sample.get("scored_triples", sample.get("scored_triplets", [])) or []
    serialised = []
    for rank, value in enumerate(scored, start=1):
        head, relation, tail = _triple(value)
        score = None
        if len(value) >= 4:
            try:
                score = float(value[3])
            except (TypeError, ValueError):
                pass
        serialised.append({
            "rank": rank,
            "head": head,
            "relation": relation,
            "tail": tail,
            "score": score,
        })
    prompt_topk = prompt_triples(sample, prompt_budget)
    triple_values = [(item["head"], item["relation"], item["tail"]) for item in serialised]
    return {
        "id": sample_id,
        "row_index": row_index,
        "question": sample.get("question", ""),
        "q_entity": sample.get("q_entity", []),
        "q_entity_in_graph": sample.get("q_entity_in_graph", []),
        "a_entity": sample.get("a_entity", []),
        "a_entity_in_graph": sample.get("a_entity_in_graph", []),
        "max_path_length": sample.get("max_path_length"),
        "target_relevant_triples": [list(_triple(item)) for item in sample.get("target_relevant_triples", [])],
        "stored_pool_size": len(serialised),
        "candidate_graph_ordered_sha256": sha256_json(triple_values),
        "candidate_graph_set_sha256": sha256_json(sorted(set(triple_values))),
        "prompt_budget": prompt_budget,
        "effective_prompt_k": len(prompt_topk),
        "prompt_topk": [list(item) for item in prompt_topk],
        "scored_triples": serialised,
    }


def validate_retrieval(
    retrieval: dict[str, dict[str, Any]],
    prompt_budget: int,
    gpt_triples: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(retrieval, dict):
        raise ValueError("Retrieval result must be a dict keyed by sample ID")
    rows = []
    problems: list[str] = []
    warnings: Counter[str] = Counter()
    pool_sizes: Counter[int] = Counter()
    target_numerator = target_denominator = target_samples = 0
    answer_numerator = answer_denominator = answer_samples = 0
    gpt_numerator = gpt_denominator = gpt_samples = 0
    raw_target_numerator = raw_target_denominator = raw_target_samples = 0
    raw_answer_numerator = raw_answer_denominator = raw_answer_samples = 0
    raw_gpt_numerator = raw_gpt_denominator = raw_gpt_samples = 0
    target_sample_recalls: list[float] = []
    answer_sample_recalls: list[float] = []
    gpt_sample_recalls: list[float] = []
    raw_target_sample_recalls: list[float] = []
    raw_answer_sample_recalls: list[float] = []
    raw_gpt_sample_recalls: list[float] = []
    for index, (sample_id_raw, sample) in enumerate(retrieval.items()):
        sample_id = str(sample_id_raw)
        if not isinstance(sample, dict):
            problems.append(f"{sample_id}: sample is not a dict")
            continue
        if not sample.get("question"):
            warnings["missing_question"] += 1
        scored = sample.get("scored_triples", sample.get("scored_triplets"))
        if scored is None:
            problems.append(f"{sample_id}: missing scored_triples")
            continue
        if not scored:
            warnings["empty_retrieval"] += 1
        scores = []
        triples = []
        for rank, value in enumerate(scored):
            try:
                triples.append(_triple(value))
            except ValueError as exc:
                problems.append(f"{sample_id} rank {rank + 1}: {exc}")
                continue
            if len(value) >= 4:
                try:
                    score = float(value[3])
                except (TypeError, ValueError):
                    problems.append(f"{sample_id} rank {rank + 1}: non-numeric score")
                    continue
                if not math.isfinite(score):
                    problems.append(f"{sample_id} rank {rank + 1}: non-finite score")
                scores.append(score)
        if scores and any(left < right for left, right in zip(scores, scores[1:])):
            problems.append(f"{sample_id}: scores are not non-increasing")
        if scores and any(score < 0 or score > 1 for score in scores):
            warnings["score_outside_probability_range"] += 1
        duplicates = len(triples) - len(set(triples))
        if duplicates:
            warnings["samples_with_duplicate_triples"] += 1
            warnings["duplicate_triple_count"] += duplicates
        prompt_k = len(prompt_triples(sample, prompt_budget))
        if prompt_k < min(prompt_budget, len(triples)):
            warnings["effective_prompt_k_reduced_by_duplicates"] += 1
        topics = {str(item) for item in sample.get("q_entity", [])}
        topic_in_graph = {str(item) for item in sample.get("q_entity_in_graph", [])}
        if topics and not topic_in_graph:
            warnings["topic_entity_not_in_graph"] += 1
        answers = {str(item) for item in sample.get("a_entity_in_graph", [])}
        if not answers:
            warnings["answer_not_in_graph"] += 1
        effective_triples = prompt_triples(sample, prompt_budget)
        effective_triple_set = set(effective_triples)
        entities = {entity for triple in effective_triples for entity in (triple[0], triple[2])}
        raw_triple_set = set(triples[:prompt_budget])
        raw_entities = {
            entity for triple in triples[:prompt_budget] for entity in (triple[0], triple[2])
        }
        if answers:
            answer_samples += 1
            matched_answers = len(answers & entities)
            answer_numerator += matched_answers
            answer_denominator += len(answers)
            answer_sample_recalls.append(matched_answers / len(answers))
            raw_answer_samples += 1
            raw_matched_answers = len(answers & raw_entities)
            raw_answer_numerator += raw_matched_answers
            raw_answer_denominator += len(answers)
            raw_answer_sample_recalls.append(raw_matched_answers / len(answers))
        targets = {_triple(item) for item in sample.get("target_relevant_triples", [])}
        if targets:
            target_samples += 1
            matched_targets = len(targets & effective_triple_set)
            target_numerator += matched_targets
            target_denominator += len(targets)
            target_sample_recalls.append(matched_targets / len(targets))
            raw_target_samples += 1
            raw_matched_targets = len(targets & raw_triple_set)
            raw_target_numerator += raw_matched_targets
            raw_target_denominator += len(targets)
            raw_target_sample_recalls.append(raw_matched_targets / len(targets))
        gpt_targets = {
            _triple(item) for item in (gpt_triples or {}).get(sample_id, [])
        }
        if gpt_targets:
            gpt_samples += 1
            matched_gpt = len(gpt_targets & effective_triple_set)
            gpt_numerator += matched_gpt
            gpt_denominator += len(gpt_targets)
            gpt_sample_recalls.append(matched_gpt / len(gpt_targets))
            raw_gpt_samples += 1
            raw_matched_gpt = len(gpt_targets & raw_triple_set)
            raw_gpt_numerator += raw_matched_gpt
            raw_gpt_denominator += len(gpt_targets)
            raw_gpt_sample_recalls.append(raw_matched_gpt / len(gpt_targets))
        pool_sizes[len(triples)] += 1
        rows.append(_serialise_sample(sample_id, index, sample, prompt_budget))

    if problems:
        preview = "\n".join(problems[:20])
        raise ValueError(f"Retrieval validation failed with {len(problems)} error(s):\n{preview}")
    return rows, {
        "sample_count": len(rows),
        "sample_id_order_sha256": sha256_json([row["id"] for row in rows]),
        "questions_sha256": sha256_json([(row["id"], normalise_question(row["question"])) for row in rows]),
        "ordered_candidate_pools_sha256": sha256_json([(row["id"], row["candidate_graph_ordered_sha256"]) for row in rows]),
        "pool_size_distribution": {str(key): value for key, value in sorted(pool_sizes.items())},
        "warning_counts": dict(sorted(warnings.items())),
        "topk_metrics": {
            "budget": prompt_budget,
            "shortest_path_triple_recall": (
                sum(target_sample_recalls) / len(target_sample_recalls) if target_sample_recalls else None
            ),
            "shortest_path_triple_recall_matched_triples": target_numerator,
            "shortest_path_triple_recall_total_triples": target_denominator,
            "shortest_path_triple_recall_sample_denominator": target_samples,
            "answer_entity_recall": (
                sum(answer_sample_recalls) / len(answer_sample_recalls) if answer_sample_recalls else None
            ),
            "answer_entity_recall_numerator": answer_numerator,
            "answer_entity_recall_denominator": answer_denominator,
            "answer_entity_recall_sample_denominator": answer_samples,
            "gpt_triple_recall": (
                sum(gpt_sample_recalls) / len(gpt_sample_recalls) if gpt_sample_recalls else None
            ),
            "gpt_triple_recall_matched_triples": gpt_numerator,
            "gpt_triple_recall_total_triples": gpt_denominator,
            "gpt_triple_recall_sample_denominator": gpt_samples,
        },
        "repo_raw_topk_metrics": {
            "budget": prompt_budget,
            "shortest_path_triple_recall": (
                sum(raw_target_sample_recalls) / len(raw_target_sample_recalls)
                if raw_target_sample_recalls else None
            ),
            "shortest_path_triple_recall_matched_triples": raw_target_numerator,
            "shortest_path_triple_recall_total_triples": raw_target_denominator,
            "shortest_path_triple_recall_sample_denominator": raw_target_samples,
            "answer_entity_recall": (
                sum(raw_answer_sample_recalls) / len(raw_answer_sample_recalls)
                if raw_answer_sample_recalls else None
            ),
            "answer_entity_recall_numerator": raw_answer_numerator,
            "answer_entity_recall_denominator": raw_answer_denominator,
            "answer_entity_recall_sample_denominator": raw_answer_samples,
            "gpt_triple_recall": (
                sum(raw_gpt_sample_recalls) / len(raw_gpt_sample_recalls)
                if raw_gpt_sample_recalls else None
            ),
            "gpt_triple_recall_matched_triples": raw_gpt_numerator,
            "gpt_triple_recall_total_triples": raw_gpt_denominator,
            "gpt_triple_recall_sample_denominator": raw_gpt_samples,
        },
    }


def compare_retrieval_replicates(
    baseline: dict[str, dict[str, Any]],
    specs: Iterable[tuple[str, str]],
    score_tolerance: float = 1e-6,
) -> dict[str, Any]:
    baseline = {str(key): value for key, value in baseline.items()}
    baseline_ids = list(baseline)
    results = {}
    for label, path in specs:
        candidate = load_torch(path)
        if not isinstance(candidate, dict):
            raise ValueError(f"Retrieval replicate {label!r} is not a dictionary")
        candidate = {str(key): value for key, value in candidate.items()}
        if list(candidate) != baseline_ids:
            raise ValueError(f"Retrieval replicate {label!r} changed sample IDs or order")
        identity_mismatches = score_mismatches = metadata_mismatches = 0
        maximum_score_delta = 0.0
        mismatch_examples = []
        for sample_id in baseline_ids:
            left, right = baseline[sample_id], candidate[sample_id]
            left_values = left.get("scored_triples", left.get("scored_triplets", []))
            right_values = right.get("scored_triples", right.get("scored_triplets", []))
            left_triples = [_triple(value) for value in left_values]
            right_triples = [_triple(value) for value in right_values]
            if left_triples != right_triples:
                identity_mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(sample_id)
            for left_value, right_value in zip(left_values, right_values):
                if len(left_value) >= 4 and len(right_value) >= 4:
                    delta = abs(float(left_value[3]) - float(right_value[3]))
                    maximum_score_delta = max(maximum_score_delta, delta)
                    score_mismatches += delta > score_tolerance
            for key in (
                "question", "q_entity", "q_entity_in_graph", "a_entity",
                "a_entity_in_graph", "max_path_length", "target_relevant_triples",
            ):
                if left.get(key) != right.get(key):
                    metadata_mismatches += 1
                    break
        passed = not identity_mismatches and not score_mismatches and not metadata_mismatches
        results[label] = {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "passed": passed,
            "sample_identity_mismatch_count": identity_mismatches,
            "score_value_mismatch_count": score_mismatches,
            "metadata_mismatch_count": metadata_mismatches,
            "maximum_absolute_score_delta": maximum_score_delta,
            "score_tolerance": score_tolerance,
            "mismatch_examples": mismatch_examples,
        }
        if not passed:
            raise ValueError(f"Retrieval replicate {label!r} is unstable: {results[label]}")
    return {
        "status": "passed" if results else "not_measured",
        "replicates": results,
    }


def load_predictions(specs: Iterable[tuple[str, str]], expected_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    expected = set(expected_ids)
    for label, path in specs:
        rows = list(iter_jsonl(path))
        by_id: dict[str, dict[str, Any]] = {}
        duplicates = []
        for row in rows:
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"Prediction {label!r} contains a row without id")
            if sample_id in by_id:
                duplicates.append(sample_id)
            by_id[sample_id] = row
        if duplicates:
            raise ValueError(f"Prediction {label!r} has duplicate IDs: {duplicates[:10]}")
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        if extra:
            raise ValueError(
                f"Prediction {label!r} contains IDs outside the frozen retrieval cohort: "
                f"missing={len(missing)}, extra={len(extra)}; "
                f"examples missing={missing[:5]}, extra={extra[:5]}"
            )
        question_mismatches = []
        variants = {
            row.get("_qforacle", {}).get("variant")
            for row in rows if row.get("_qforacle", {}).get("variant")
        }
        run_fingerprints = {
            row.get("_qforacle", {}).get("run_fingerprint")
            for row in rows if row.get("_qforacle", {}).get("run_fingerprint")
        }
        protocol_fingerprints = {
            row.get("_qforacle", {}).get("fixed_reasoning_protocol_sha256")
            for row in rows
            if row.get("_qforacle", {}).get("fixed_reasoning_protocol_sha256")
        }
        if variants and variants != {"baseline"}:
            raise ValueError(f"Phase-0 prediction {label!r} is not a baseline run: {variants}")
        if len(run_fingerprints) > 1:
            raise ValueError(f"Prediction {label!r} mixes multiple run fingerprints")
        if len(protocol_fingerprints) > 1:
            raise ValueError(f"Prediction {label!r} mixes multiple reasoning protocols")
        # The retrieval artifact defines the cohort and order.  Question
        # mismatch is checked by the caller after loading; this field remains
        # here for a stable run summary schema.
        loaded[label] = by_id
        summary[label] = {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "sample_count": len(by_id),
            "retrieval_only_sample_count": len(missing),
            "retrieval_only_sample_examples": missing[:20],
            "id_order_sha256": sha256_json([str(row["id"]) for row in rows]),
            "question_mismatch_count": len(question_mismatches),
            "variant": next(iter(variants), None),
            "run_fingerprint": next(iter(run_fingerprints), None),
            "fixed_reasoning_protocol_sha256": next(iter(protocol_fingerprints), None),
        }
    labels = list(loaded)
    if labels:
        reference_ids = list(loaded[labels[0]])
        for label in labels[1:]:
            if list(loaded[label]) != reference_ids:
                raise ValueError(
                    f"Reasoning replicate {label!r} changed the QA cohort or its order"
                )
        protocol_values = {
            summary[label].get("fixed_reasoning_protocol_sha256") for label in labels
        }
        if None not in protocol_values and len(protocol_values) != 1:
            raise ValueError("Reasoning replicas do not share one fixed protocol")
    else:
        reference_ids = []
    pairwise = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            raw_equal = answer_equal = 0
            for sample_id in reference_ids:
                lvalue = str(loaded[left][sample_id].get("prediction", ""))
                rvalue = str(loaded[right][sample_id].get("prediction", ""))
                raw_equal += lvalue == rvalue
                answer_equal += prediction_answers(lvalue) == prediction_answers(rvalue)
            key = f"{left}__vs__{right}"
            pairwise[key] = {
                "sample_count": len(reference_ids),
                "raw_output_exact_agreement": raw_equal / len(reference_ids) if reference_ids else None,
                "parsed_answer_exact_agreement": answer_equal / len(reference_ids) if reference_ids else None,
                "raw_output_disagreement_count": len(reference_ids) - raw_equal,
                "parsed_answer_disagreement_count": len(reference_ids) - answer_equal,
            }
    return loaded, {
        "runs": summary,
        "qa_cohort_sample_count": len(reference_ids),
        "qa_cohort_id_order_sha256": sha256_json(reference_ids) if reference_ids else None,
        "retrieval_cohort_sample_count": len(expected_ids),
        "pairwise_stability": pairwise,
    }


def environment_snapshot() -> dict[str, Any]:
    packages = runtime_package_versions()
    cuda = {"available": None, "version": None, "device_count": None, "devices": []}
    try:
        import torch
        cuda["available"] = torch.cuda.is_available()
        cuda["version"] = torch.version.cuda
        cuda["device_count"] = torch.cuda.device_count()
        cuda["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception as exc:  # environment capture must not block a CPU-only audit
        cuda["error"] = f"{type(exc).__name__}: {exc}"

    def command_output(command: list[str]) -> str | None:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = result.stdout.strip() or result.stderr.strip()
        return output or None

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }


def compare_expected_metrics(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    rows = {}
    passed = True
    for metric, specification in expected.items():
        if metric in {"source", "note"}:
            continue
        if isinstance(specification, dict):
            target = specification.get("value")
            tolerance = specification.get("tolerance", 0.0)
        else:
            target = specification
            tolerance = 0.0
        value = actual.get(metric)
        difference = None if value is None or target is None else float(value) - float(target)
        within = difference is not None and abs(difference) <= float(tolerance)
        passed = passed and within
        rows[metric] = {
            "actual": value,
            "expected": target,
            "difference": difference,
            "tolerance": tolerance,
            "within_tolerance": within,
        }
    return {
        "passed": passed,
        "source": expected.get("source"),
        "metrics": rows,
    }


def audit_retriever_checkpoint(
    checkpoint_path: str | os.PathLike[str] | None,
    declared: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path:
        return {"status": "not_measured", "declared": declared}
    checkpoint = load_torch(checkpoint_path)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Retriever checkpoint must contain a config dictionary")
    config = checkpoint["config"]
    dataset_config = config.get("dataset", {})
    retriever_config = config.get("retriever", {})
    dde_config = retriever_config.get("DDE_kwargs", {})
    env_config = config.get("env", {})
    actual = {
        "text_encoder": dataset_config.get("text_encoder_name"),
        "topic_positional_encoding": retriever_config.get("topic_pe"),
        "forward_rounds": dde_config.get("num_rounds"),
        "reverse_rounds": dde_config.get("num_reverse_rounds"),
        "training_seed": env_config.get("seed"),
    }
    comparisons = {}
    for key, actual_value in actual.items():
        declared_value = declared.get(key)
        if key == "text_encoder" and actual_value is not None and declared_value is not None:
            matched = str(actual_value).split("/")[-1] == str(declared_value).split("/")[-1]
        else:
            matched = actual_value == declared_value
        comparisons[key] = {
            "declared": declared_value,
            "checkpoint": actual_value,
            "matched": matched,
        }
    passed = all(value["matched"] for value in comparisons.values())
    result = {
        "status": "passed" if passed else "failed",
        "path": str(Path(checkpoint_path).resolve()),
        "sha256": sha256_file(checkpoint_path),
        "checkpoint_config": config,
        "comparisons": comparisons,
    }
    if not passed:
        raise ValueError(f"Retriever checkpoint/config mismatch: {comparisons}")
    return result


def core_hashes(repo: Path) -> dict[str, str | None]:
    return {
        relative: sha256_file(repo / relative) if (repo / relative).is_file() else None
        for relative in CORE_FILES
    }


def experiment_hashes(repo: Path) -> dict[str, str]:
    root = repo / "experiments" / "query_factor_oracle"
    allowed = {".py", ".json", ".sh", ".md"}
    return {
        str(path.relative_to(repo)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in allowed and "outputs" not in path.parts
    }


def audit_hf_datasets(
    sources: Iterable[dict[str, Any]],
    retrieval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare frozen retrieval cohort with one or more Hugging Face graph sources."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the repository's datasets dependency for --audit-datasets") from exc
    retrieval = {str(key): value for key, value in retrieval.items()}
    expected_ids = list(retrieval)
    result = {}
    for source in sources:
        name = source["name"]
        repo_id = source["repo_id"]
        split = source.get("split", "test")
        revision = source.get("revision")
        kwargs = {"path": repo_id, "split": split}
        if revision:
            kwargs["revision"] = revision
        dataset = load_dataset(**kwargs)
        seen_ids = []
        expected_set = set(expected_ids)
        question_mismatches = []
        topic_mismatches = []
        answer_mismatches = []
        answer_in_graph_mismatches = []
        missing_candidates = []
        graph_hash_rows = []
        metadata_hash_rows = []
        for row in dataset:
            sample_id = str(row.get("id", row.get("ID", "")))
            if not sample_id:
                continue
            seen_ids.append(sample_id)
            if sample_id not in expected_set:
                continue
            sample = retrieval[sample_id]
            graph = [tuple(map(str, triple[:3])) for triple in row.get("graph", [])]
            graph_set = set(graph)
            source_question = str(row.get("question", ""))
            if source_question and normalise_question(source_question) != normalise_question(sample.get("question", "")):
                question_mismatches.append(sample_id)
            source_topics = sorted(map(str, row.get("q_entity", [])))
            retrieval_topics = sorted(map(str, sample.get("q_entity", [])))
            if "q_entity" in row and source_topics != retrieval_topics:
                topic_mismatches.append(sample_id)
            source_answers = sorted(map(str, row.get("a_entity", [])))
            retrieval_answers = sorted(map(str, sample.get("a_entity", [])))
            if "a_entity" in row and source_answers != retrieval_answers:
                answer_mismatches.append(sample_id)
            graph_entities = {entity for triple in graph for entity in (triple[0], triple[2])}
            source_answers_in_graph = sorted(set(source_answers) & graph_entities)
            retrieval_answers_in_graph = sorted(map(str, sample.get("a_entity_in_graph", [])))
            if "a_entity" in row and source_answers_in_graph != retrieval_answers_in_graph:
                answer_in_graph_mismatches.append(sample_id)
            triples = {_triple(value) for value in sample.get("scored_triples", [])}
            absent = triples - graph_set
            if absent:
                missing_candidates.append({"id": sample_id, "count": len(absent)})
            graph_hash_rows.append((sample_id, sha256_json(sorted(graph_set))))
            metadata_hash_rows.append((
                sample_id, source_question, source_topics,
                source_answers, source_answers_in_graph,
            ))
        observed_set = set(seen_ids)
        retrieval_missing = sorted(expected_set - observed_set)
        fatal_issues = []
        if retrieval_missing and source.get("require_all_retrieval_ids", True):
            fatal_issues.append(f"{len(retrieval_missing)} retrieval IDs absent from dataset")
        if question_mismatches:
            fatal_issues.append(f"{len(question_mismatches)} question mismatches")
        if source.get("validate_entity_metadata", False):
            if topic_mismatches:
                fatal_issues.append(f"{len(topic_mismatches)} topic-entity mismatches")
            if answer_mismatches:
                fatal_issues.append(f"{len(answer_mismatches)} answer-entity mismatches")
            if answer_in_graph_mismatches:
                fatal_issues.append(
                    f"{len(answer_in_graph_mismatches)} answer-in-graph mismatches"
                )
        if missing_candidates and source.get("validate_retrieval_membership", True):
            fatal_issues.append(
                f"{len(missing_candidates)} samples contain retrieval triples outside the candidate graph"
            )
        result[name] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "validate_retrieval_membership": source.get("validate_retrieval_membership", True),
            "require_all_retrieval_ids": source.get("require_all_retrieval_ids", True),
            "validate_entity_metadata": source.get("validate_entity_metadata", False),
            "split": split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "dataset_sample_count": len(dataset),
            "retrieval_ids_missing_from_dataset": retrieval_missing,
            "dataset_ids_not_in_retrieval": sorted(observed_set - expected_set),
            "question_mismatch_count": len(question_mismatches),
            "question_mismatch_examples": question_mismatches[:20],
            "topic_entity_mismatch_count": len(topic_mismatches),
            "topic_entity_mismatch_examples": topic_mismatches[:20],
            "answer_entity_mismatch_count": len(answer_mismatches),
            "answer_entity_mismatch_examples": answer_mismatches[:20],
            "answer_in_graph_mismatch_count": len(answer_in_graph_mismatches),
            "answer_in_graph_mismatch_examples": answer_in_graph_mismatches[:20],
            "samples_with_retrieval_triples_outside_graph": len(missing_candidates),
            "outside_graph_examples": missing_candidates[:20],
            "graph_set_hashes_sha256": sha256_json(graph_hash_rows),
            "question_topic_answer_metadata_sha256": sha256_json(metadata_hash_rows),
            "fatal_issues": fatal_issues,
            "passed": not fatal_issues,
        }
    return result


def build_lock(
    config: dict[str, Any],
    retrieval_path: Path,
    validation: dict[str, Any],
    source_hashes: dict[str, str | None],
    experiment_source_hashes: dict[str, str],
    prediction_summary: dict[str, Any],
    dataset_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    stable_config = {key: value for key, value in config.items() if not key.startswith("_")}
    gpt_triples_path = config.get("data", {}).get("gpt_triples_file")
    return {
        "schema_version": 1,
        "dataset": config["dataset"],
        "split": config["split"],
        "git_commit": git_revision(config["_repo_root"]),
        "git_status": git_status(config["_repo_root"]),
        "config_sha256": sha256_json(stable_config),
        "retrieval_file": str(retrieval_path),
        "retrieval_bytes": retrieval_path.stat().st_size,
        "retrieval_sha256": sha256_file(retrieval_path),
        "retrieval_semantic_sha256": validation["retrieval_semantic_sha256"],
        "gpt_triples_sha256": (
            sha256_file(gpt_triples_path)
            if gpt_triples_path and Path(gpt_triples_path).is_file()
            else None
        ),
        "sample_id_order_sha256": validation["sample_id_order_sha256"],
        "questions_sha256": validation["questions_sha256"],
        "ordered_candidate_pools_sha256": validation["ordered_candidate_pools_sha256"],
        "core_file_sha256": source_hashes,
        "experiment_file_sha256": experiment_source_hashes,
        "prediction_runs": prediction_summary.get("runs", {}),
        "dataset_audit_fingerprint": sha256_json(dataset_audit) if dataset_audit is not None else None,
        "fixed_protocol": config.get("protocol", {}),
        "retriever": config.get("retriever", {}),
        "retriever_checkpoint_audit": validation.get("retriever_checkpoint_audit"),
        "llm": config.get("llm", {}),
        "evaluator": config.get("evaluator", {}),
    }


def verify_lock(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    ignored = {"prediction_runs"}
    mismatches = []
    for key, expected_value in expected.items():
        if key in ignored:
            continue
        if actual.get(key) != expected_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Baseline lock mismatch in: {', '.join(mismatches)}")


def freeze_baseline(
    config: dict[str, Any],
    output_dir: str | os.PathLike[str],
    predictions: Iterable[tuple[str, str]] = (),
    qa_summaries: Iterable[tuple[str, str]] = (),
    retrieval_replicates: Iterable[tuple[str, str]] = (),
    retriever_checkpoint: str | os.PathLike[str] | None = None,
    audit_datasets: bool = False,
    copy_retrieval: bool = True,
    verify_lock_path: str | os.PathLike[str] | None = None,
    require_confirmed: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    retrieval_path = Path(config["retrieval_result"])
    retrieval = load_torch(retrieval_path)
    prompt_budget = int(config.get("protocol", {}).get("prompt_top_k", 100))
    gpt_path_value = config.get("data", {}).get("gpt_triples_file")
    gpt_triples = load_torch(gpt_path_value) if gpt_path_value and Path(gpt_path_value).is_file() else None
    if isinstance(gpt_triples, dict):
        gpt_triples = {str(key): value for key, value in gpt_triples.items()}
    samples, validation = validate_retrieval(retrieval, prompt_budget, gpt_triples)
    validation["retrieval_semantic_sha256"] = retrieval_semantic_sha256(retrieval)
    expected_retrieval = config.get("expected_baseline", {}).get("retrieval")
    validation["expected_baseline_comparison"] = (
        compare_expected_metrics(validation["repo_raw_topk_metrics"], expected_retrieval)
        if expected_retrieval else None
    )
    retrieval_stability = compare_retrieval_replicates(retrieval, retrieval_replicates)
    retriever_checkpoint_audit = audit_retriever_checkpoint(
        retriever_checkpoint, config.get("retriever", {}),
    )
    validation["retriever_checkpoint_audit"] = retriever_checkpoint_audit
    declared_pool_value = config.get("retriever", {}).get("stored_pool_budget")
    declared_pool_budget = int(declared_pool_value) if declared_pool_value is not None else None
    observed_pool_max = max(
        (row["stored_pool_size"] for row in samples), default=0
    )
    if declared_pool_budget is not None and observed_pool_max > declared_pool_budget:
        raise ValueError(
            f"Stored retrieval pool exceeds declared budget: {observed_pool_max} > {declared_pool_budget}"
        )
    retrieval_budget_status = (
        "passed" if declared_pool_budget is not None and observed_pool_max == declared_pool_budget
        else "not_measured"
    )
    validation["retrieval_budget_audit"] = {
        "declared_stored_pool_budget": declared_pool_budget,
        "maximum_observed_pool_size": observed_pool_max,
        "status": retrieval_budget_status,
        "note": (
            "Exact max_K is confirmed when at least one candidate graph reaches the cap; "
            "otherwise only the upper bound is observed."
        ),
    }
    expected_ids = [row["id"] for row in samples]
    prediction_data, stability = load_predictions(predictions, expected_ids)
    phase0_config = config.get("phase0", {})
    raw_threshold = float(phase0_config.get("minimum_raw_output_agreement", 0.99))
    answer_threshold = float(phase0_config.get("minimum_parsed_answer_agreement", 1.0))
    stability_violations = []
    for comparison, values in stability["pairwise_stability"].items():
        if (values.get("raw_output_exact_agreement") or 0.0) < raw_threshold:
            stability_violations.append(
                f"{comparison}: raw agreement {values.get('raw_output_exact_agreement')} < {raw_threshold}"
            )
        if (values.get("parsed_answer_exact_agreement") or 0.0) < answer_threshold:
            stability_violations.append(
                f"{comparison}: parsed-answer agreement "
                f"{values.get('parsed_answer_exact_agreement')} < {answer_threshold}"
            )
    stability["criteria"] = {
        "minimum_raw_output_agreement": raw_threshold,
        "minimum_parsed_answer_agreement": answer_threshold,
    }
    stability["status"] = (
        "not_measured" if len(prediction_data) < 2
        else "failed" if stability_violations else "passed"
    )
    stability["violations"] = stability_violations
    if stability_violations:
        write_json(output / "reasoning_stability_failed.json", stability)
        raise ValueError(
            "Baseline reasoning stability check failed; inspect "
            f"{output / 'reasoning_stability_failed.json'}"
        )
    qa_validation: dict[str, Any] = {}
    for label, path in qa_summaries:
        from .io_utils import read_json
        value = read_json(path)
        comparisons = value.get("expected_baseline_comparison")
        scope_failures = [
            scope for scope, comparison in (comparisons or {}).items()
            if not comparison.get("passed")
        ]
        qa_validation[label] = {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "variant": value.get("variant"),
            "qa_cohort_sample_count": value.get("qa_cohort_sample_count"),
            "qa_cohort_id_order_sha256": value.get("qa_cohort_id_order_sha256"),
            "expected_baseline_comparison": comparisons,
            "failed_scopes": scope_failures,
            "passed": (
                value.get("variant") == "baseline"
                and bool(comparisons)
                and not scope_failures
                and value.get("qa_cohort_id_order_sha256")
                == stability.get("qa_cohort_id_order_sha256")
            ),
        }
    qa_metric_status = (
        "not_measured" if not qa_validation
        else "passed" if all(value["passed"] for value in qa_validation.values())
        else "failed"
    )
    sample_by_id = {row["id"]: row for row in samples}
    for label, rows in prediction_data.items():
        mismatches = [
            sample_id
            for sample_id in rows
            if rows[sample_id].get("question")
            and normalise_question(rows[sample_id]["question"])
            != normalise_question(sample_by_id[sample_id]["question"])
        ]
        stability["runs"][label]["question_mismatch_count"] = len(mismatches)
        stability["runs"][label]["question_mismatch_examples"] = mismatches[:20]
        if mismatches:
            raise ValueError(
                f"Prediction {label!r} has {len(mismatches)} question mismatch(es): {mismatches[:10]}"
            )
    for sample in samples:
        if prediction_data:
            sample["reasoning_predictions"] = {
                label: {
                    "prediction": rows[sample["id"]].get("prediction"),
                    "ground_truth": rows[sample["id"]].get("ground_truth"),
                }
                for label, rows in prediction_data.items()
                if sample["id"] in rows
            }
    dataset_audit = None
    if audit_datasets:
        dataset_audit = audit_hf_datasets(config.get("data", {}).get("dataset_sources", []), retrieval)
        failed_sources = {
            name: audit["fatal_issues"]
            for name, audit in dataset_audit.items()
            if audit.get("fatal_issues")
        }
        if failed_sources:
            write_json(output / "dataset_audit_failed.json", {
                "status": "failed",
                "sources": failed_sources,
                "full_audit": dataset_audit,
            })
            raise ValueError(
                "Dataset audit failed; inspect "
                f"{output / 'dataset_audit_failed.json'}: {failed_sources}"
            )
    repo = Path(config["_repo_root"])
    source_hashes = core_hashes(repo)
    experiment_source_hashes = experiment_hashes(repo)
    lock = build_lock(
        config, retrieval_path, validation, source_hashes,
        experiment_source_hashes, stability, dataset_audit,
    )
    if verify_lock_path:
        from .io_utils import read_json
        verify_lock(read_json(verify_lock_path), lock)

    expected_comparison = validation.get("expected_baseline_comparison")
    components = {
        "retrieval_artifact_validation": "passed",
        "retriever_checkpoint_provenance": retriever_checkpoint_audit["status"],
        "retrieval_budget": retrieval_budget_status,
        "retrieval_paper_metric_sanity": (
            "passed" if expected_comparison and expected_comparison.get("passed")
            else "failed" if expected_comparison else "not_measured"
        ),
        "qa_paper_metric_sanity": qa_metric_status,
        "retrieval_rerun_stability": retrieval_stability["status"],
        "reasoning_rerun_stability": stability["status"],
        "dataset_audit": (
            "passed" if dataset_audit is not None and all(
                value.get("passed") for value in dataset_audit.values()
            ) else "failed" if dataset_audit is not None else "not_measured"
        ),
    }
    if any(value == "failed" for value in components.values()):
        phase0_status = "failed"
    elif all(value == "passed" for value in components.values()):
        phase0_status = "confirmed"
    else:
        phase0_status = "not_measured"
    validation["phase0_status"] = phase0_status
    validation["phase0_components"] = components
    validation["dataset"] = config["dataset"]
    validation["split"] = config["split"]

    if require_confirmed:
        # Keep a failed confirmation attempt from replacing a previously
        # confirmed lock/validation pair.  The dedicated attempt artifact is
        # diagnostic; the canonical Phase-0 artifacts below are committed only
        # after every gate has passed.
        confirmation_attempt = {
            "phase0_status": phase0_status,
            "phase0_components": components,
            "expected_baseline_comparison": validation.get("expected_baseline_comparison"),
            "retrieval_stability": retrieval_stability,
            "reasoning_stability": stability,
            "qa_baseline_validation": qa_validation,
            "dataset_audit": dataset_audit,
        }
        write_json(output / "last_confirmation_attempt.json", confirmation_attempt)
        if phase0_status != "confirmed":
            raise ValueError(
                f"Phase 0 is {phase0_status}, not confirmed; inspect "
                f"{output / 'last_confirmation_attempt.json'}"
            )

    write_jsonl(output / "retrieval" / "ranked_samples.jsonl.gz", samples)
    if copy_retrieval:
        save_torch(retrieval, output / "retrieval" / "baseline_retrieval.pth")
    write_json(output / "retrieval" / "metrics.json", {
        "prompt_effective_topk": validation["topk_metrics"],
        "repo_raw_topk": validation["repo_raw_topk_metrics"],
        "expected_paper_comparison_uses": "repo_raw_topk",
        "expected_baseline_comparison": validation["expected_baseline_comparison"],
    })
    write_json(output / "validation.json", {
        **validation,
        "dataset_audit": dataset_audit,
        "retrieval_stability": retrieval_stability,
        "stability": stability,
        "qa_baseline_validation": qa_validation,
    })
    write_json(output / "environment.json", environment_snapshot())
    write_json(output / "baseline.lock.json", lock)
    manifest = {
        "schema_version": 1,
        "phase0_status": phase0_status,
        "phase0_components": components,
        "dataset": config["dataset"],
        "split": config["split"],
        "cohort_definition": "ordered keys of frozen retrieval PTH",
        "oracle_candidate_universe": "stored scored_triples pool only",
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "lock_file": str(output / "baseline.lock.json"),
        "artifacts": {
            "ranked_samples": "retrieval/ranked_samples.jsonl.gz",
            "retrieval_copy": "retrieval/baseline_retrieval.pth" if copy_retrieval else None,
            "metrics": "retrieval/metrics.json",
            "validation": "validation.json",
        },
        "gpt_triples_sha256": sha256_file(gpt_path_value) if gpt_triples is not None else None,
        "limitations": [
            "The frozen retrieval PTH stores at most the retriever's max_K pool, not every edge in the candidate graph.",
            "Local vLLM does not consume reason/main.py's CLI seed; stability is established by repeated output comparison.",
            "Paper Score_h is not reused because the original evaluator hard-codes the author baseline retrieval path.",
            *(
                ["Retrieval rerun stability was not measured; pass --retrieval-replicate LABEL=PATH after rerunning inference."]
                if retrieval_stability["status"] == "not_measured" else []
            ),
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
