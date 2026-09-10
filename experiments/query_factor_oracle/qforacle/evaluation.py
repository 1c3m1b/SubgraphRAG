from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import (
    iter_jsonl,
    load_torch,
    retrieval_semantic_sha256,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .baseline import compare_expected_metrics


def _core_metrics(repo: Path):
    reason_dir = repo / "reason"
    if str(reason_dir) not in sys.path:
        sys.path.insert(0, str(reason_dir))
    # Reuse the exact answer extraction/matching functions used by the repo,
    # without calling eval_results (which hard-codes the author retrieval file).
    from metrics.evaluate_results_corrected import (
        eval_f1,
        eval_hit,
        eval_precision,
        eval_recall,
        get_pred,
        remove_duplicates,
    )
    from metrics.evaluate_results import eval_hit as eval_hit_original
    return get_pred, remove_duplicates, eval_precision, eval_recall, eval_f1, eval_hit, eval_hit_original


def _load_factors(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {str(row["id"]): row for row in iter_jsonl(path)}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "hit": None,
            "hit_at_1": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "micro_f1": None,
            "micro_precision": None,
            "micro_recall": None,
            "exact_match": None,
            "totally_wrong": None,
            "no_answer_count": 0,
            "no_answer_ratio": None,
        }
    total_pred = sum(row["num_pred"] for row in rows)
    total_answer = sum(row["num_answer"] for row in rows)
    total_match = sum(row["matched"] for row in rows)
    micro_precision = total_match / total_pred if total_pred else 0.0
    micro_recall = total_match / total_answer if total_answer else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    percent = lambda key: sum(row[key] for row in rows) * 100.0 / len(rows)
    no_answer_count = sum(row["no_answer"] for row in rows)
    return {
        "sample_count": len(rows),
        "hit": percent("hit"),
        "hit_at_1": percent("hit_at_1"),
        "macro_f1": percent("f1"),
        "macro_precision": percent("precision"),
        "macro_recall": percent("recall"),
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "exact_match": percent("exact_match"),
        "totally_wrong": percent("totally_wrong"),
        "no_answer_count": no_answer_count,
        "no_answer_ratio": no_answer_count / len(rows),
    }


def evaluate_predictions(
    repo_root: str,
    prediction_path: str,
    retrieval_path: str,
    output_dir: str,
    factors_path: str | None = None,
    allow_incomplete: bool = False,
    expected_baseline: dict[str, Any] | None = None,
    expected_qa_path: str | None = None,
    allow_retrieval_only_samples: bool = False,
) -> dict[str, Any]:
    retrieval = load_torch(retrieval_path)
    retrieval = {str(key): value for key, value in retrieval.items()}
    predictions = list(iter_jsonl(prediction_path))
    factors = _load_factors(factors_path)
    expected_ids = list(map(str, retrieval))
    prediction_ids = [str(row.get("id", "")) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Prediction file contains duplicate IDs")
    expected_qa_ids: list[str] | None = None
    if expected_qa_path:
        qa_rows = list(iter_jsonl(expected_qa_path))
        expected_qa_ids = [str(row.get("id", "")) for row in qa_rows]
        if not all(expected_qa_ids):
            raise ValueError("Expected QA cohort contains a row without id")
        if len(expected_qa_ids) != len(set(expected_qa_ids)):
            raise ValueError("Expected QA cohort contains duplicate IDs")
        qa_outside_retrieval = sorted(set(expected_qa_ids) - set(expected_ids))
        if qa_outside_retrieval:
            raise ValueError(
                "Expected QA cohort contains IDs outside retrieval: "
                f"{qa_outside_retrieval[:10]}"
            )
        if not allow_incomplete and prediction_ids != expected_qa_ids:
            raise ValueError(
                "Prediction IDs/order differ from the fixed QA cohort: "
                f"predictions={len(prediction_ids)}, expected={len(expected_qa_ids)}"
            )
    missing = sorted(set(expected_ids) - set(prediction_ids))
    extra = sorted(set(prediction_ids) - set(expected_ids))
    permitted_retrieval_only = (
        set(expected_ids) - set(expected_qa_ids or [])
        if allow_retrieval_only_samples and expected_qa_ids is not None else set()
    )
    unpermitted_missing = sorted(set(missing) - permitted_retrieval_only)
    if extra or (unpermitted_missing and not allow_incomplete):
        raise ValueError(
            f"Prediction/retrieval ID mismatch: missing={len(missing)}, "
            f"unpermitted_missing={len(unpermitted_missing)}, extra={len(extra)}; "
            f"examples missing={unpermitted_missing[:5]}, extra={extra[:5]}"
        )
    prediction_by_id = {str(row["id"]): row for row in predictions}
    functions = _core_metrics(Path(repo_root).resolve())
    get_pred, remove_duplicates, eval_precision, eval_recall, eval_f1, eval_hit, eval_hit_original = functions
    sample_rows = []
    for sample_id in expected_ids:
        if sample_id not in prediction_by_id:
            continue
        data = prediction_by_id[sample_id]
        prediction_text = data.get("prediction", "")
        if isinstance(prediction_text, list):
            prediction_text = "\n".join(map(str, prediction_text))
        else:
            prediction_text = str(prediction_text)
        answers = sorted(remove_duplicates([str(item) for item in data.get("ground_truth", [])]), key=len, reverse=True)
        question = str(data.get("question", retrieval[sample_id].get("question", "")))
        if "when" in question.lower() or "what year" in question.lower():
            answers = [
                answer.split("-", 1)[0]
                if "-" in answer and answer.split("-", 1)[0].isdigit()
                else answer
                for answer in answers
            ]
        double_check = any(keyword in question.lower() for keyword in (
            "when", "what year", "which year", "where", "sport", "what countr",
            "language", "nba finals", "world series",
        ))
        parsed = get_pred(prediction_text, None)
        if answers:
            precision, matched_precision, num_pred = eval_precision(parsed, answers, double_check)
            recall, matched_recall, num_answer = eval_recall(parsed, answers, double_check)
            if matched_precision != matched_recall:
                raise AssertionError(f"Core evaluator match-count disagreement for {sample_id}")
            f1 = eval_f1(precision, recall)
            hit_at_1 = eval_hit(parsed, answers, double_check)
            hit = eval_hit_original(prediction_text, answers, double_check)
            matched = matched_precision
        else:
            precision = recall = f1 = hit_at_1 = hit = 0.0
            matched = num_pred = num_answer = 0
        no_answer = (
            not parsed
            or "ans:" not in prediction_text
            or "ans: not available" in prediction_text.lower()
            or "ans: no information available" in prediction_text.lower()
        )
        factor = factors.get(sample_id, {})
        sample_rows.append({
            "id": sample_id,
            "query_family": factor.get("query_family", "unparsed"),
            "parse_status": factor.get("parse_status", "missing"),
            "max_path_length": retrieval[sample_id].get("max_path_length"),
            "answer_in_candidate_graph": bool(retrieval[sample_id].get("a_entity_in_graph")),
            "prediction": parsed,
            "ground_truth": answers,
            "hit": float(hit),
            "hit_at_1": float(hit_at_1),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "exact_match": float(f1 == 1.0),
            "totally_wrong": float(recall == 0.0),
            "no_answer": bool(no_answer),
            "matched": matched,
            "num_pred": num_pred,
            "num_answer": num_answer,
        })

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hop_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        family_rows[row["query_family"]].append(row)
        status_rows[row["parse_status"]].append(row)
        hop = row["max_path_length"]
        hop_rows["3+" if isinstance(hop, (int, float)) and hop >= 3 else str(hop)].append(row)
    subset_rows = [row for row in sample_rows if row["answer_in_candidate_graph"]]
    variants = {
        row.get("_qforacle", {}).get("variant")
        for row in predictions
        if row.get("_qforacle", {}).get("variant")
    }
    run_fingerprints = {
        row.get("_qforacle", {}).get("run_fingerprint")
        for row in predictions
        if row.get("_qforacle", {}).get("run_fingerprint")
    }
    protocol_fingerprints = {
        row.get("_qforacle", {}).get("fixed_reasoning_protocol_sha256")
        for row in predictions
        if row.get("_qforacle", {}).get("fixed_reasoning_protocol_sha256")
    }
    if len(variants) > 1:
        raise ValueError(f"Prediction file mixes oracle variants: {sorted(variants)}")
    if len(run_fingerprints) > 1:
        raise ValueError("Prediction file mixes multiple reasoning run fingerprints")
    if len(protocol_fingerprints) > 1:
        raise ValueError("Prediction file mixes multiple fixed reasoning protocols")
    summary = {
        "schema_version": 1,
        "prediction_path": str(Path(prediction_path).resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "retrieval_path": str(Path(retrieval_path).resolve()),
        "retrieval_sha256": sha256_file(retrieval_path),
        "retrieval_semantic_sha256": retrieval_semantic_sha256(retrieval),
        "factors_path": str(Path(factors_path).resolve()) if factors_path else None,
        "factors_sha256": sha256_file(factors_path) if factors_path else None,
        "missing_prediction_count": len(missing),
        "retrieval_cohort_sample_count": len(expected_ids),
        "retrieval_cohort_id_order_sha256": sha256_json(expected_ids),
        "qa_cohort_sample_count": len(prediction_ids),
        "qa_cohort_id_order_sha256": sha256_json(prediction_ids),
        "qa_cohort_id_set_sha256": sha256_json(sorted(prediction_ids)),
        "qa_question_collection_sha256": sha256_json([
            (row["id"], row.get("question", "")) for row in sample_rows
        ]),
        "qa_ground_truth_collection_sha256": sha256_json([
            (row["id"], row.get("ground_truth", [])) for row in sample_rows
        ]),
        "expected_qa_cohort_path": str(Path(expected_qa_path).resolve()) if expected_qa_path else None,
        "expected_qa_cohort_sha256": sha256_file(expected_qa_path) if expected_qa_path else None,
        "retrieval_only_sample_count": len(permitted_retrieval_only),
        "retrieval_only_sample_examples": sorted(permitted_retrieval_only)[:20],
        "cohort_note": (
            "full denotes the complete fixed final-QA cohort; it may be an explicit ID subset "
            "of the retrieval cohort (WebQSP: RoG reasoning split versus retrieval split)."
        ),
        "variant": next(iter(variants), None),
        "run_fingerprint": next(iter(run_fingerprints), None),
        "fixed_reasoning_protocol_sha256": next(iter(protocol_fingerprints), None),
        "score_h": None,
        "score_h_note": (
            "Not reported: the repository evaluator hard-codes the author baseline retrieval. "
            "Using it for a reranked variant would ground Score_h in the wrong evidence."
        ),
        "full": _aggregate(sample_rows),
        "answer_in_graph_subset": _aggregate(subset_rows),
        "by_query_family": {key: _aggregate(value) for key, value in sorted(family_rows.items())},
        "by_parse_status": {key: _aggregate(value) for key, value in sorted(status_rows.items())},
        "by_max_path_length": {key: _aggregate(value) for key, value in sorted(hop_rows.items())},
        "core_functions_reused": [
            "reason.metrics.evaluate_results_corrected.get_pred",
            "reason.metrics.evaluate_results_corrected.eval_precision",
            "reason.metrics.evaluate_results_corrected.eval_recall",
            "reason.metrics.evaluate_results_corrected.eval_f1",
            "reason.metrics.evaluate_results_corrected.eval_hit",
            "reason.metrics.evaluate_results.eval_hit",
        ],
    }
    if expected_baseline and variants == {"baseline"}:
        summary["expected_baseline_comparison"] = {
            scope: compare_expected_metrics(summary[scope], expected_metrics)
            for scope, expected_metrics in expected_baseline.items()
            if scope in summary and isinstance(expected_metrics, dict)
        }
    else:
        summary["expected_baseline_comparison"] = None
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_metrics_path = output / "sample_metrics.jsonl.gz"
    write_jsonl(sample_metrics_path, sample_rows)
    summary["sample_metrics_path"] = str(sample_metrics_path)
    summary["sample_metrics_sha256"] = sha256_file(sample_metrics_path)
    write_json(output / "qa_summary.json", summary)
    return summary
