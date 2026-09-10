from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import (
    iter_jsonl,
    load_torch,
    normalise_question,
    retrieval_semantic_sha256,
    save_torch,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .oracle import (
    VARIANTS,
    VARIANT_DEFINITIONS,
    aggregate_metrics,
    rerank_sample,
    sample_retrieval_metrics,
)


def _factor_map(path: str) -> dict[str, dict[str, Any]]:
    rows = list(iter_jsonl(path))
    result = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError("Factor row is missing id")
        if sample_id in result:
            raise ValueError(f"Duplicate factor ID: {sample_id}")
        result[sample_id] = row
    return result


def _delta(current: Any, baseline: Any) -> float | None:
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)


def _metric_deltas(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(baseline.get(key), dict):
            output[key] = _metric_deltas(value, baseline[key])
        elif isinstance(value, (int, float)) and not key.endswith(("count", "denominator")):
            output[key] = _delta(value, baseline.get(key))
    return output


def run_oracle_experiment(
    baseline_path: str,
    factors_path: str,
    output_dir: str,
    budgets: Iterable[int],
    greedy_budget: int,
    variants: Iterable[str] = VARIANTS,
    gpt_triples_path: str | None = None,
    qa_cohort_path: str | None = None,
    diagnostics_top_k: int = 100,
    match_state_limit: int = 100_000,
) -> dict[str, Any]:
    baseline = load_torch(baseline_path)
    factors = _factor_map(factors_path)
    gpt_triples = load_torch(gpt_triples_path) if gpt_triples_path else {}
    gpt_triples = {str(key): value for key, value in gpt_triples.items()}
    if not isinstance(baseline, dict):
        raise ValueError("Baseline retrieval must be a dict keyed by sample ID")
    baseline = {str(key): value for key, value in baseline.items()}
    baseline_ids = list(baseline)
    baseline_semantic_sha256 = retrieval_semantic_sha256(baseline)
    qa_ids: list[str] = []
    if qa_cohort_path:
        qa_rows = list(iter_jsonl(qa_cohort_path))
        qa_ids = [str(row.get("id", "")) for row in qa_rows]
        if not all(qa_ids):
            raise ValueError("QA cohort source contains a row without id")
        if len(qa_ids) != len(set(qa_ids)):
            raise ValueError("QA cohort source contains duplicate IDs")
        qa_outside_retrieval = sorted(set(qa_ids) - set(baseline_ids))
        if qa_outside_retrieval:
            raise ValueError(
                "QA cohort contains IDs outside the frozen retrieval cohort: "
                f"{qa_outside_retrieval[:10]}"
            )
    qa_id_set = set(qa_ids)
    missing = sorted(set(baseline_ids) - set(factors))
    extra = sorted(set(factors) - set(baseline_ids))
    if missing or extra:
        raise ValueError(
            f"Factor/retrieval ID mismatch: missing={len(missing)}, extra={len(extra)}; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )
    mismatched_questions = [
        sample_id for sample_id in baseline_ids
        if normalise_question(factors[sample_id].get("question", ""))
        != normalise_question(baseline[sample_id].get("question", ""))
    ]
    if mismatched_questions:
        raise ValueError(
            f"Factor/retrieval question mismatch for {len(mismatched_questions)} samples: "
            f"{mismatched_questions[:10]}"
        )
    factor_baseline_hashes = {
        row.get("baseline_retrieval_semantic_sha256") for row in factors.values()
    }
    if factor_baseline_hashes != {baseline_semantic_sha256}:
        raise ValueError(
            "Factor artifact was not built from this frozen retrieval artifact: "
            f"factor hashes={factor_baseline_hashes}, retrieval={baseline_semantic_sha256}"
        )
    factor_datasets = {row.get("dataset") for row in factors.values()}
    factor_splits = {row.get("split") for row in factors.values()}
    if len(factor_datasets) != 1 or len(factor_splits) != 1:
        raise ValueError("Factor artifact mixes datasets or splits")
    budgets = sorted(set(int(value) for value in budgets))
    if not budgets or min(budgets) <= 0:
        raise ValueError("At least one positive evaluation budget is required")
    if diagnostics_top_k <= 0:
        raise ValueError("diagnostics_top_k must be positive")
    if match_state_limit <= 0:
        raise ValueError("match_state_limit must be positive")
    variants = list(dict.fromkeys(variants))
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if "baseline" not in variants:
        variants.insert(0, "baseline")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    per_sample_by_variant: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for variant in variants:
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        reranked: dict[str, dict[str, Any]] = {}
        metric_rows_by_budget: dict[int, list[dict[str, Any]]] = {
            budget: [] for budget in budgets
        }
        rerank_truncation = {"count": 0}

        def process_samples():
            for sample_id, sample in baseline.items():
                reranked_sample, annotated_rows = rerank_sample(
                    sample, factors[str(sample_id)], variant, greedy_budget,
                    match_state_limit,
                )
                reranked[str(sample_id)] = reranked_sample
                sample_rerank_truncated = any(
                    candidate.get("dependency_match_truncated", False)
                    for candidate in annotated_rows
                )
                rerank_truncation["count"] += int(sample_rerank_truncated)
                for budget in budgets:
                    row = sample_retrieval_metrics(
                        sample,
                        factors[str(sample_id)],
                        annotated_rows,
                        budget,
                        gpt_triples.get(str(sample_id), []),
                        match_state_limit,
                    )
                    row["id"] = str(sample_id)
                    row["variant"] = variant
                    metric_rows_by_budget[budget].append(row)
                yield {
                    "id": str(sample_id),
                    "parse_status": factors[str(sample_id)].get("parse_status"),
                    "oracle_eligible": factors[str(sample_id)].get("oracle_eligible", False),
                    "query_family": factors[str(sample_id)].get("query_family", "unparsed"),
                    "diagnostics_top_k": diagnostics_top_k,
                    "dependency_match_truncated": sample_rerank_truncated,
                    "ranked_candidates": [
                        {
                            "new_rank": new_rank,
                            "original_rank": candidate["original_rank"] + 1,
                            "triple": [candidate["head"], candidate["relation_raw"], candidate["tail"]],
                            "original_score": candidate["original_score"],
                            "relation_match": candidate["relation_match"],
                            "structure_match": candidate["structure_match"],
                            "direction_match": candidate["direction_match"],
                            "slot_matches": candidate["slot_matches"],
                            "slot_direction_matches": candidate["slot_direction_matches"],
                            "branch_matches": candidate["branch_matches"],
                            "all_factor_slot_matches": candidate["all_factor_slot_matches"],
                            "all_factor_slot_direction_matches": candidate[
                                "all_factor_slot_direction_matches"
                            ],
                            "all_factor_branch_matches": candidate["all_factor_branch_matches"],
                            "anonymous_graph_slots": candidate["anonymous_graph_slots"],
                            "dependency_graph_slots": candidate["dependency_graph_slots"],
                            "directed_dependency_graph_slots": candidate[
                                "directed_dependency_graph_slots"
                            ],
                            "anonymous_graph_branches": candidate[
                                "anonymous_graph_branches"
                            ],
                            "dependency_graph_branches": candidate[
                                "dependency_graph_branches"
                            ],
                            "directed_dependency_graph_branches": candidate[
                                "directed_dependency_graph_branches"
                            ],
                            "dependency_match_truncated": candidate[
                                "dependency_match_truncated"
                            ],
                        }
                        for new_rank, candidate in enumerate(
                            annotated_rows[:diagnostics_top_k], start=1
                        )
                    ],
                }

        # Streaming JSONL avoids retaining every candidate annotation for CWQ.
        write_jsonl(variant_dir / "rerank_diagnostics.jsonl.gz", process_samples())
        if list(reranked) != baseline_ids:
            raise AssertionError("Reranking changed sample order or membership")
        save_torch(reranked, variant_dir / "retrieval_result.pth")
        budget_summaries = {}
        sample_metric_artifacts = {}
        for budget in budgets:
            metric_rows = metric_rows_by_budget[budget]
            per_sample_by_variant[variant][budget] = metric_rows
            aggregate = aggregate_metrics(metric_rows)
            if qa_ids:
                qa_metric_rows = [row for row in metric_rows if row["id"] in qa_id_set]
                qa_aggregate = aggregate_metrics(qa_metric_rows)
                aggregate["qa_cohort"] = qa_aggregate["overall"]
                aggregate["qa_cohort_parsed_only"] = qa_aggregate["parsed_only"]
                aggregate["qa_cohort_oracle_eligible_only"] = qa_aggregate["oracle_eligible_only"]
                aggregate["qa_cohort_by_family"] = qa_aggregate["by_family"]
                aggregate["qa_cohort_by_parse_status"] = qa_aggregate["by_parse_status"]
            budget_summaries[str(budget)] = aggregate
            sample_metric_path = variant_dir / f"sample_metrics_at_{budget}.jsonl.gz"
            write_jsonl(sample_metric_path, metric_rows)
            sample_metric_artifacts[str(budget)] = {
                "path": str(sample_metric_path.resolve()),
                "sha256": sha256_file(sample_metric_path),
            }
        summaries[variant] = {
            "retrieval_result": str(variant_dir / "retrieval_result.pth"),
            "retrieval_sha256": sha256_file(variant_dir / "retrieval_result.pth"),
            "retrieval_semantic_sha256": retrieval_semantic_sha256(reranked),
            "rerank_match_truncated_sample_count": rerank_truncation["count"],
            "rerank_match_truncated_rate": (
                rerank_truncation["count"] / len(baseline_ids)
                if baseline_ids else None
            ),
            "sample_metric_artifacts": sample_metric_artifacts,
            "budgets": budget_summaries,
        }

    baseline_summary = summaries["baseline"]["budgets"]
    for variant in variants:
        summaries[variant]["delta_vs_baseline"] = {
            budget: _metric_deltas(summary, baseline_summary[budget])
            for budget, summary in summaries[variant]["budgets"].items()
        }

    # Paired transition counts make it possible to distinguish recoverable
    # improvements from sample-composition effects without dropping null rows.
    transitions: dict[str, Any] = {}
    for variant in variants:
        if variant == "baseline":
            continue
        transitions[variant] = {}
        for budget in budgets:
            base_rows = {row["id"]: row for row in per_sample_by_variant["baseline"][budget]}
            variant_rows = {row["id"]: row for row in per_sample_by_variant[variant][budget]}
            counts = CounterTransitions()
            for sample_id in baseline_ids:
                counts.add(base_rows[sample_id], variant_rows[sample_id])
            transitions[variant][str(budget)] = counts.to_dict()

    manifest = {
        "schema_version": 1,
        "protocol": {
            "candidate_universe": "frozen stored scored_triples pool",
            "operation": "deterministic reranking only; no triple injection",
            "unparseable_policy": "exact baseline ordering",
            "prompt_budget_semantics": "ordered triple deduplication, then truncate",
            "score_policy": "preserve original retriever score; list order defines reranked rank",
            "greedy_budget": greedy_budget,
            "diagnostics_top_k": diagnostics_top_k,
            "match_state_limit": match_state_limit,
            "budgets": budgets,
            "variants": variants,
            "variant_definitions": {
                variant: VARIANT_DEFINITIONS[variant] for variant in variants
            },
        },
        "inputs": {
            "baseline_path": str(Path(baseline_path).resolve()),
            "baseline_sha256": sha256_file(baseline_path),
            "baseline_semantic_sha256": baseline_semantic_sha256,
            "factor_dataset": next(iter(factor_datasets)),
            "factor_split": next(iter(factor_splits)),
            "factors_path": str(Path(factors_path).resolve()),
            "factors_sha256": sha256_file(factors_path),
            "gpt_triples_path": str(Path(gpt_triples_path).resolve()) if gpt_triples_path else None,
            "gpt_triples_sha256": sha256_file(gpt_triples_path) if gpt_triples_path else None,
            "sample_id_order_sha256": sha256_json(baseline_ids),
            "retrieval_cohort_sample_count": len(baseline_ids),
            "qa_cohort_path": str(Path(qa_cohort_path).resolve()) if qa_cohort_path else None,
            "qa_cohort_sha256": sha256_file(qa_cohort_path) if qa_cohort_path else None,
            "qa_cohort_sample_count": len(qa_ids) if qa_cohort_path else None,
            "qa_cohort_id_order_sha256": sha256_json(qa_ids) if qa_ids else None,
            "retrieval_only_sample_count": len(baseline_ids) - len(qa_ids) if qa_cohort_path else None,
            "retrieval_only_sample_examples": (
                [sample_id for sample_id in baseline_ids if sample_id not in qa_id_set][:20]
                if qa_cohort_path else []
            ),
        },
        "variants": summaries,
        "paired_transitions": transitions,
    }
    write_json(output / "retrieval_summary.json", manifest)
    return manifest


class CounterTransitions:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, baseline: dict[str, Any], current: dict[str, Any]) -> None:
        for metric in ("triple_recall", "answer_recall", "answer_hit", "gold_relation_coverage",
                       "slot_coverage", "slot_direction_coverage", "consistent_slot_coverage",
                       "branch_complete_coverage", "full_structure_hit"):
            left, right = baseline.get(metric), current.get(metric)
            if left is None or right is None:
                self.values[metric]["ineligible"] += 1
            elif right > left:
                self.values[metric]["improved"] += 1
            elif right < left:
                self.values[metric]["regressed"] += 1
            else:
                self.values[metric]["unchanged"] += 1

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {metric: dict(sorted(values.items())) for metric, values in sorted(self.values.items())}
