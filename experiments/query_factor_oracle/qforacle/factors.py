from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io_utils import (
    load_torch,
    retrieval_semantic_sha256,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .logical_form import extract_factor_record, load_logical_form_records, normalize_relation


PARSED_STATUSES = {"parsed", "partial"}


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _candidate_relations(sample: dict[str, Any]) -> set[str]:
    result = set()
    for triple in sample.get("scored_triples", sample.get("scored_triplets", [])):
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            result.add(normalize_relation(str(triple[1])))
    return result


def summarise_factors(
    factors: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parsed = [row for row in factors if row.get("parse_status") in PARSED_STATUSES]
    relations = Counter()
    operators = Counter()
    directions = Counter()
    slot_roles = Counter()
    operator_relations = Counter()
    slot_counts = Counter()
    branch_counts = Counter()
    hop_counts = Counter()
    alternative_counts = Counter()
    source_families = Counter()
    logical_form_types = Counter()
    parse_error_types = Counter()
    pool_relation_numerator = pool_relation_denominator = 0
    fully_pool_covered = 0
    relation_oov = Counter()
    duplicate_relation_slot_samples = 0
    for row in parsed:
        relations.update(row.get("relations", []))
        operator_relations.update(row.get("operator_relations", []))
        operators.update(row.get("operators", []))
        slot_counts[row.get("slot_count", 0)] += 1
        branch_counts[row.get("branch_count", 0)] += 1
        hop_counts[row.get("max_hop", 0)] += 1
        alternative_counts[row.get("alternative_count", 0)] += 1
        source_families[str(row.get("source_family") or "unknown")] += 1
        logical_form_types[str(row.get("logical_form_type") or "unknown")] += 1
        slot_relations = [slot.get("relation") for slot in row.get("relation_slots", [])]
        if len(slot_relations) != len(set(slot_relations)):
            duplicate_relation_slot_samples += 1
        for slot in row.get("relation_slots", []):
            directions[slot.get("topic_direction", "unknown")] += 1
            slot_roles[slot.get("role", "relation")] += 1
        pool_relations = _candidate_relations(baseline[row["id"]])
        gold = set(row.get("relations", []))
        overlap = gold & pool_relations
        pool_relation_numerator += len(overlap)
        pool_relation_denominator += len(gold)
        fully_pool_covered += bool(gold) and overlap == gold
        relation_oov.update(gold - pool_relations)
    status_counts = Counter(row.get("parse_status", "missing") for row in factors)
    for row in factors:
        for error in row.get("parse_errors", []):
            parse_error_types[error.split(":", 2)[0]] += 1
    family_counts = Counter(row.get("query_family", "unparsed") for row in parsed)
    question_matches = Counter(str(row.get("alignment", {}).get("question_match")) for row in factors)
    alternative_equivalence = Counter()
    for row in parsed:
        signatures = row.get("alternative_semantic_signatures", [])
        if row.get("alternative_count", 0) > 1:
            alternative_equivalence["multiple_gold_forms"] += 1
            if len(signatures) == 1:
                alternative_equivalence["semantic_factor_equivalent"] += 1
            else:
                alternative_equivalence["semantic_factor_distinct"] += 1
            if not row.get("cross_form_consistent", True):
                alternative_equivalence["semantic_factor_conflict"] += 1
    return {
        "schema_version": 1,
        "cohort_sample_count": len(factors),
        "source_present_count": sum(row.get("alignment", {}).get("source_present", False) for row in factors),
        "parseable_count": len(parsed),
        "parseable_ratio": len(parsed) / len(factors) if factors else None,
        "oracle_eligible_count": sum(bool(row.get("oracle_eligible")) for row in factors),
        "oracle_eligible_ratio": (
            sum(bool(row.get("oracle_eligible")) for row in factors) / len(factors)
            if factors else None
        ),
        "parse_status_distribution": dict(sorted(status_counts.items())),
        "question_alignment_distribution": dict(sorted(question_matches.items())),
        "query_family_distribution": dict(sorted(family_counts.items())),
        "operator_distribution": dict(sorted(operators.items())),
        "source_family_distribution": dict(sorted(source_families.items())),
        "logical_form_type_distribution": dict(sorted(logical_form_types.items())),
        "parse_error_type_distribution": dict(sorted(parse_error_types.items())),
        "slot_count_distribution": {str(key): value for key, value in sorted(slot_counts.items())},
        "branch_count_distribution": {str(key): value for key, value in sorted(branch_counts.items())},
        "max_hop_distribution": {str(key): value for key, value in sorted(hop_counts.items())},
        "direction_distribution": dict(sorted(directions.items())),
        "slot_role_distribution": dict(sorted(slot_roles.items())),
        "operator_relation_frequency": dict(operator_relations.most_common()),
        "answer_type_available_count": sum(bool(row.get("answer_types")) for row in parsed),
        "topic_entity_in_graph_sample_count": sum(
            bool(baseline[row["id"]].get("q_entity_in_graph")) for row in factors
        ),
        "duplicate_relation_slot_sample_count": duplicate_relation_slot_samples,
        "alternative_count_distribution": {str(key): value for key, value in sorted(alternative_counts.items())},
        "alternative_equivalence": dict(sorted(alternative_equivalence.items())),
        "unique_relation_count": len(relations),
        "relation_frequency": dict(relations.most_common()),
        "pool_gold_relation_coverage": (
            pool_relation_numerator / pool_relation_denominator if pool_relation_denominator else None
        ),
        "pool_gold_relation_coverage_numerator": pool_relation_numerator,
        "pool_gold_relation_coverage_denominator": pool_relation_denominator,
        "fully_relation_covered_sample_count": fully_pool_covered,
        "relations_absent_from_stored_pool": dict(relation_oov.most_common()),
    }


def build_factors(
    baseline_path: str,
    logical_form_paths: Iterable[str],
    output_dir: str,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    baseline = load_torch(baseline_path)
    if not isinstance(baseline, dict):
        raise ValueError("Baseline retrieval must be a dict keyed by sample ID")
    baseline = {str(key): value for key, value in baseline.items()}
    baseline_semantic_sha256 = retrieval_semantic_sha256(baseline)
    sources = load_logical_form_records(logical_form_paths)
    factors = [
        {
            "schema_version": 1,
            "dataset": dataset,
            "split": split,
            "baseline_retrieval_semantic_sha256": baseline_semantic_sha256,
            **extract_factor_record(str(sample_id), sample, sources.get(str(sample_id))),
        }
        for sample_id, sample in baseline.items()
    ]
    summary = summarise_factors(factors, baseline)
    summary.update({
        "dataset": dataset,
        "split": split,
        "baseline_path": str(Path(baseline_path).resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "baseline_semantic_sha256": baseline_semantic_sha256,
        "logical_form_sources": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in logical_form_paths
        ],
        "cohort_id_order_sha256": sha256_json([row["id"] for row in factors]),
        "unmatched_source_record_count": len(set(sources) - set(map(str, baseline))),
    })
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "query_factors.jsonl.gz", factors)
    write_json(output / "summary.json", summary)
    return summary
