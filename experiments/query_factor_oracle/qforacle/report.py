from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, read_json, sha256_file, write_json


def _get_metric(summary: dict[str, Any], variant: str, budget: int, scope: str, metric: str) -> float | None:
    return summary["variants"][variant]["budgets"][str(budget)][scope].get(metric)


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _fmt(value: float | None, percent: bool = True) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}" if percent else f"{value:.2f}"


def _load_hashed_jsonl(metadata: dict[str, Any], label: str) -> list[dict[str, Any]]:
    path = metadata.get("path")
    expected_hash = metadata.get("sha256")
    if not path or not expected_hash:
        raise ValueError(f"{label} is missing its per-sample artifact provenance")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} per-sample artifact hash mismatch")
    return list(iter_jsonl(path))


def _paired_sample_diagnostic(
    baseline_retrieval: list[dict[str, Any]],
    current_retrieval: list[dict[str, Any]],
    baseline_qa: list[dict[str, Any]],
    current_qa: list[dict[str, Any]],
) -> dict[str, Any]:
    collections = [
        {str(row["id"]): row for row in rows}
        for rows in (
            baseline_retrieval, current_retrieval, baseline_qa, current_qa,
        )
    ]
    id_sets = [set(value) for value in collections]
    if id_sets[0] != id_sets[1] or id_sets[2] != id_sets[3]:
        raise ValueError("Paired per-sample artifacts differ within retrieval or QA")
    if not id_sets[2] <= id_sets[0]:
        raise ValueError("Per-sample QA cohort is not a subset of retrieval")
    base_retrieval, new_retrieval, base_qa, new_qa = collections
    counts = {
        "sample_count": len(id_sets[2]),
        "answer_recall_comparable_count": 0,
        "answer_recall_improved_count": 0,
        "answer_recall_improved_hit_0_to_1_count": 0,
        "answer_recall_improved_f1_improved_count": 0,
        "answer_recall_improved_f1_unchanged_count": 0,
        "answer_recall_improved_f1_regressed_count": 0,
        "overall_hit_1_to_0_count": 0,
        "overall_f1_regressed_count": 0,
    }
    for sample_id in sorted(id_sets[2]):
        left_retrieval = base_retrieval[sample_id].get("answer_recall")
        right_retrieval = new_retrieval[sample_id].get("answer_recall")
        left_hit = float(base_qa[sample_id].get("hit", 0.0))
        right_hit = float(new_qa[sample_id].get("hit", 0.0))
        left_f1 = float(base_qa[sample_id].get("f1", 0.0))
        right_f1 = float(new_qa[sample_id].get("f1", 0.0))
        if right_hit < left_hit:
            counts["overall_hit_1_to_0_count"] += 1
        if right_f1 < left_f1 - 1e-12:
            counts["overall_f1_regressed_count"] += 1
        if left_retrieval is None or right_retrieval is None:
            continue
        counts["answer_recall_comparable_count"] += 1
        if float(right_retrieval) <= float(left_retrieval):
            continue
        counts["answer_recall_improved_count"] += 1
        if right_hit > left_hit:
            counts["answer_recall_improved_hit_0_to_1_count"] += 1
        if right_f1 > left_f1 + 1e-12:
            counts["answer_recall_improved_f1_improved_count"] += 1
        elif right_f1 < left_f1 - 1e-12:
            counts["answer_recall_improved_f1_regressed_count"] += 1
        else:
            counts["answer_recall_improved_f1_unchanged_count"] += 1
    improved = counts["answer_recall_improved_count"]
    counts["answer_recall_improved_to_hit_gain_rate"] = (
        counts["answer_recall_improved_hit_0_to_1_count"] / improved
        if improved else None
    )
    counts["answer_recall_improved_but_f1_unchanged_rate"] = (
        counts["answer_recall_improved_f1_unchanged_count"] / improved
        if improved else None
    )
    return counts


def build_report(
    retrieval_summary_path: str,
    qa_summary_paths: dict[str, str],
    output_dir: str,
    budget: int,
    phase0_validation_path: str | None = None,
) -> dict[str, Any]:
    retrieval = read_json(retrieval_summary_path)
    qa = {variant: read_json(path) for variant, path in qa_summary_paths.items()}
    phase0_validation = read_json(phase0_validation_path) if phase0_validation_path else None
    retrieval_inputs = retrieval.get("inputs", {})
    dataset = retrieval_inputs.get("factor_dataset")
    split = retrieval_inputs.get("factor_split")
    required_input_hashes = {
        "baseline_semantic_sha256": retrieval_inputs.get("baseline_semantic_sha256"),
        "factors_sha256": retrieval_inputs.get("factors_sha256"),
        "sample_id_order_sha256": retrieval_inputs.get("sample_id_order_sha256"),
    }
    if not dataset or not split:
        raise ValueError("Retrieval summary is missing factor dataset/split provenance")
    missing_input_hashes = sorted(
        key for key, value in required_input_hashes.items() if not value
    )
    if missing_input_hashes:
        raise ValueError(
            "Retrieval summary is missing required input fingerprints: "
            f"{missing_input_hashes}"
        )
    phase0_status = (
        phase0_validation.get("phase0_status") if phase0_validation else "not_provided"
    )
    if phase0_validation:
        if (
            phase0_validation.get("retrieval_semantic_sha256")
            != retrieval_inputs.get("baseline_semantic_sha256")
            or phase0_validation.get("sample_id_order_sha256")
            != retrieval_inputs.get("sample_id_order_sha256")
            or phase0_validation.get("dataset") != retrieval_inputs.get("factor_dataset")
            or phase0_validation.get("split") != retrieval_inputs.get("factor_split")
        ):
            raise ValueError("Phase-0 validation does not belong to this retrieval/factor cohort")
    variants = list(retrieval["variants"])
    extra_qa_variants = sorted(set(qa) - set(variants))
    if extra_qa_variants:
        raise ValueError(f"QA summaries contain unknown variants: {extra_qa_variants}")
    if str(budget) not in retrieval["variants"]["baseline"]["budgets"]:
        raise ValueError(f"Budget {budget} is absent from retrieval summary")
    baseline_budget = retrieval["variants"]["baseline"]["budgets"][str(budget)]
    primary_scope = "qa_cohort" if "qa_cohort" in baseline_budget else "overall"
    baseline_answer = _get_metric(retrieval, "baseline", budget, primary_scope, "answer_recall")
    baseline_triple = _get_metric(retrieval, "baseline", budget, primary_scope, "triple_recall")
    all_answer = _get_metric(retrieval, "all_factors", budget, primary_scope, "answer_recall")
    all_triple = _get_metric(retrieval, "all_factors", budget, primary_scope, "triple_recall")
    relation_answer = _get_metric(retrieval, "relation_set", budget, primary_scope, "answer_recall")
    slot_answer = _get_metric(retrieval, "relation_slot_branch", budget, primary_scope, "answer_recall")
    direction_answer = _get_metric(retrieval, "direction", budget, primary_scope, "answer_recall")
    relation_gain = _difference(relation_answer, baseline_answer)
    joint_gain = _difference(all_answer, baseline_answer)
    structure_increment = _difference(slot_answer, relation_answer)
    direction_gain = _difference(direction_answer, baseline_answer)
    direction_increment = _difference(direction_answer, slot_answer)
    joint_increment_over_slot = _difference(all_answer, slot_answer)
    joint_increment_over_direction = _difference(all_answer, direction_answer)
    relation_share = (
        relation_gain / joint_gain
        if relation_gain is not None and joint_gain is not None and joint_gain > 0
        else None
    )
    available_budgets = sorted(
        int(value) for value in retrieval["variants"]["baseline"]["budgets"]
    )
    pool_budget = available_budgets[-1]
    pool_scope = retrieval["variants"]["baseline"]["budgets"][str(pool_budget)][primary_scope]
    pool_answer_ceiling = pool_scope.get("answer_recall")
    recoverable_denominator = _difference(pool_answer_ceiling, baseline_answer)
    recoverable_captured = (
        joint_gain / recoverable_denominator
        if joint_gain is not None and recoverable_denominator is not None and recoverable_denominator > 0
        else None
    )
    family_gains = []
    family_scope = "qa_cohort_by_family" if primary_scope == "qa_cohort" else "by_family"
    base_families = retrieval["variants"]["baseline"]["budgets"][str(budget)][family_scope]
    all_families = retrieval["variants"]["all_factors"]["budgets"][str(budget)][family_scope]
    for family in sorted(set(base_families) & set(all_families)):
        base_value = base_families[family].get("answer_recall")
        all_value = all_families[family].get("answer_recall")
        family_gains.append({
            "query_family": family,
            "sample_count": base_families[family]["sample_count"],
            "baseline_answer_recall": base_value,
            "all_factors_answer_recall": all_value,
            "delta": _difference(all_value, base_value),
        })
    family_gains.sort(
        key=lambda row: (
            row["delta"] is not None,
            row["delta"] if row["delta"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    structure_match_truncation = {
        variant: _get_metric(
            retrieval, variant, budget, primary_scope,
            "structure_match_truncated",
        )
        for variant in variants
    }
    rerank_match_truncation = {
        variant: retrieval["variants"][variant].get(
            "rerank_match_truncated_rate"
        )
        for variant in variants
    }
    maximum_structure_truncation = max(
        (
            value for value in (
                *structure_match_truncation.values(),
                *rerank_match_truncation.values(),
            )
            if value is not None
        ),
        default=0.0,
    )

    if qa:
        for label, value in qa.items():
            if value.get("variant") != label:
                raise ValueError(
                    f"QA summary labelled {label!r} was produced for variant "
                    f"{value.get('variant')!r}"
                )
            expected_retrieval_hash = retrieval["variants"].get(label, {}).get(
                "retrieval_semantic_sha256"
            )
            if (
                expected_retrieval_hash
                and value.get("retrieval_semantic_sha256") != expected_retrieval_hash
            ):
                raise ValueError(
                    f"QA summary {label!r} was evaluated against a different retrieval artifact"
                )
        protocol_hashes = {
            value.get("fixed_reasoning_protocol_sha256") for value in qa.values()
        }
        if None in protocol_hashes or len(protocol_hashes) != 1:
            raise ValueError("QA summaries do not share one fixed reasoning protocol")
        factor_hashes = {value.get("factors_sha256") for value in qa.values()}
        if None in factor_hashes or len(factor_hashes) != 1:
            raise ValueError("QA summaries do not share one query-factor artifact")
        if next(iter(factor_hashes)) != retrieval_inputs.get("factors_sha256"):
            raise ValueError(
                "QA summaries were not evaluated with the query-factor artifact "
                "used for retrieval reranking"
            )
        retrieval_cohort_hashes = {
            value.get("retrieval_cohort_id_order_sha256") for value in qa.values()
        }
        if None in retrieval_cohort_hashes or len(retrieval_cohort_hashes) != 1:
            raise ValueError("QA summaries do not share one retrieval cohort")
        if next(iter(retrieval_cohort_hashes)) != retrieval_inputs.get("sample_id_order_sha256"):
            raise ValueError("QA summaries do not belong to the reranked retrieval cohort")
        cohort_keys = {
            (
                value.get("qa_cohort_sample_count"),
                value.get("qa_cohort_id_order_sha256"),
            )
            for value in qa.values()
        }
        if len(cohort_keys) != 1:
            raise ValueError("QA summaries do not use the same sample IDs/order")
        question_hashes = {value.get("qa_question_collection_sha256") for value in qa.values()}
        answer_hashes = {value.get("qa_ground_truth_collection_sha256") for value in qa.values()}
        if None in question_hashes or len(question_hashes) != 1:
            raise ValueError("QA summaries do not use the same question collection")
        if None in answer_hashes or len(answer_hashes) != 1:
            raise ValueError("QA summaries do not use the same ground-truth answers")
        retrieval_qa_hash = retrieval.get("inputs", {}).get("qa_cohort_id_order_sha256")
        qa_hash = next(iter(cohort_keys))[1]
        if retrieval_qa_hash and qa_hash and retrieval_qa_hash != qa_hash:
            raise ValueError("Retrieval and final-QA summaries use different QA cohorts")
    missing_qa_variants = sorted(set(variants) - set(qa))
    qa_available = not missing_qa_variants and bool(qa)
    qa_headroom = None
    qa_hit_headroom = None
    best_qa_variant = None
    qa_family_gains: list[dict[str, Any]] = []
    paired_sample_diagnostics: dict[str, dict[str, Any]] = {}
    if qa_available:
        baseline_retrieval_rows = _load_hashed_jsonl(
            retrieval["variants"]["baseline"]
            .get("sample_metric_artifacts", {})
            .get(str(budget), {}),
            "baseline retrieval",
        )
        baseline_qa_rows = _load_hashed_jsonl({
            "path": qa["baseline"].get("sample_metrics_path"),
            "sha256": qa["baseline"].get("sample_metrics_sha256"),
        }, "baseline QA")
        for variant in variants:
            if variant == "baseline":
                continue
            current_retrieval_rows = _load_hashed_jsonl(
                retrieval["variants"][variant]
                .get("sample_metric_artifacts", {})
                .get(str(budget), {}),
                f"{variant} retrieval",
            )
            current_qa_rows = _load_hashed_jsonl({
                "path": qa[variant].get("sample_metrics_path"),
                "sha256": qa[variant].get("sample_metrics_sha256"),
            }, f"{variant} QA")
            paired_sample_diagnostics[variant] = _paired_sample_diagnostic(
                baseline_retrieval_rows,
                current_retrieval_rows,
                baseline_qa_rows,
                current_qa_rows,
            )
        baseline_f1 = qa["baseline"]["full"]["macro_f1"]
        comparable = {
            variant: value["full"]["macro_f1"]
            for variant, value in qa.items()
            if value.get("full", {}).get("macro_f1") is not None
        }
        if comparable:
            best_qa_variant = max(comparable, key=comparable.get)
        qa_headroom = qa["all_factors"]["full"]["macro_f1"] - baseline_f1
        qa_hit_headroom = qa["all_factors"]["full"]["hit"] - qa["baseline"]["full"]["hit"]
        baseline_qa_families = qa["baseline"].get("by_query_family", {})
        for variant in variants:
            if variant == "baseline":
                continue
            variant_families = qa[variant].get("by_query_family", {})
            for family in sorted(set(baseline_qa_families) & set(variant_families)):
                base_family = baseline_qa_families[family]
                current_family = variant_families[family]
                qa_family_gains.append({
                    "variant": variant,
                    "query_family": family,
                    "sample_count": base_family["sample_count"],
                    "macro_f1_delta_points": _difference(
                        current_family.get("macro_f1"), base_family.get("macro_f1")
                    ),
                    "hit_delta_points": _difference(
                        current_family.get("hit"), base_family.get("hit")
                    ),
                })
        qa_family_gains.sort(
            key=lambda row: (
                row["macro_f1_delta_points"] is not None,
                row["macro_f1_delta_points"]
                if row["macro_f1_delta_points"] is not None else float("-inf"),
            ),
            reverse=True,
        )
    retrieval_material = joint_gain is not None and joint_gain >= 0.01
    structure_material = structure_increment is not None and structure_increment >= 0.005
    if phase0_status != "confirmed":
        recommendation = "do not interpret oracle headroom until Phase 0 is confirmed"
    elif maximum_structure_truncation > 0:
        recommendation = (
            "increase phase2.match_state_limit and rerun until structural-match "
            "truncation is zero"
        )
    elif not qa_available:
        recommendation = "run every fixed-LLM variant before deciding on predicted factors"
    elif joint_gain is None:
        recommendation = "blocked: retrieval metrics are incomplete"
    elif joint_gain < 0.005:
        recommendation = "pause factor-predictor work: gold factors show little retrieval headroom"
    elif qa_available and retrieval_material and (qa_headroom is None or qa_headroom < 0.5):
        recommendation = "prioritize sharing query state with reasoning: retrieval improves but final QA does not"
    elif structure_material:
        recommendation = "proceed to predicted structured factors; slot/branch adds value beyond relation identity"
    elif relation_share is not None and relation_share >= 0.8:
        recommendation = "prioritize relation grounding; relation identity captures most oracle gain"
    else:
        recommendation = "proceed cautiously to predicted factors and validate on both datasets"

    def retrieval_value_for_sort(variant: str) -> float:
        value = _get_metric(retrieval, variant, budget, primary_scope, "answer_recall")
        return float("-inf") if value is None else float(value)

    report = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "budget": budget,
        "inputs": {
            "retrieval_summary_path": str(Path(retrieval_summary_path).resolve()),
            "retrieval_summary_sha256": sha256_file(retrieval_summary_path),
            "phase0_validation_path": (
                str(Path(phase0_validation_path).resolve()) if phase0_validation_path else None
            ),
            "phase0_validation_sha256": (
                sha256_file(phase0_validation_path) if phase0_validation_path else None
            ),
            **required_input_hashes,
            "qa_cohort_id_order_sha256": retrieval_inputs.get(
                "qa_cohort_id_order_sha256"
            ),
            "qa_summaries": {
                variant: {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                }
                for variant, path in sorted(qa_summary_paths.items())
            },
        },
        "retrieval_scope": primary_scope,
        "phase0_status": phase0_status,
        "phase0_components": (
            phase0_validation.get("phase0_components") if phase0_validation else None
        ),
        "qa_complete": qa_available,
        "missing_qa_variants": missing_qa_variants,
        "maximum_structure_match_truncation_rate": maximum_structure_truncation,
        "retrieval_headroom": {
            "baseline_answer_recall": baseline_answer,
            "all_factors_answer_recall": all_answer,
            "answer_recall_delta": joint_gain,
            "baseline_triple_recall": baseline_triple,
            "all_factors_triple_recall": all_triple,
            "triple_recall_delta": _difference(all_triple, baseline_triple),
            "stored_pool_budget": pool_budget,
            "stored_pool_answer_recall_ceiling": pool_answer_ceiling,
            "stored_pool_triple_recall_ceiling": pool_scope.get("triple_recall"),
            "stored_pool_relation_coverage_ceiling": pool_scope.get("gold_relation_coverage"),
            "stored_pool_slot_coverage_ceiling": pool_scope.get("slot_coverage"),
            "stored_pool_branch_complete_ceiling": pool_scope.get("branch_complete_coverage"),
            "stored_pool_full_structure_hit_ceiling": pool_scope.get("full_structure_hit"),
            "recoverable_answer_headroom_captured": recoverable_captured,
            "structure_match_truncation_rate_by_variant": structure_match_truncation,
            "rerank_match_truncation_rate_by_variant": rerank_match_truncation,
        },
        "factor_value": {
            "relation_set_answer_recall_delta": relation_gain,
            "slot_branch_increment_over_relation_set": structure_increment,
            "direction_answer_recall_delta": direction_gain,
            "direction_increment_over_slot_branch": direction_increment,
            "joint_increment_over_slot_branch": joint_increment_over_slot,
            "joint_increment_over_direction": joint_increment_over_direction,
            "relation_share_of_joint_gain": relation_share,
            "most_valuable_retrieval_variant": max(
                variants, key=retrieval_value_for_sort,
            ),
        },
        "qa_headroom_macro_f1_points": qa_headroom,
        "qa_headroom_hit_points": qa_hit_headroom,
        "best_qa_variant": best_qa_variant,
        "family_gains": family_gains,
        "qa_family_gains": qa_family_gains,
        "paired_retrieval_to_qa_diagnostics": paired_sample_diagnostics,
        "paired_retrieval_transitions_at_budget": {
            variant: values.get(str(budget), {})
            for variant, values in retrieval.get("paired_transitions", {}).items()
        },
        "recommendation": recommendation,
        "decision_thresholds": {
            "little_retrieval_headroom": 0.005,
            "material_retrieval_gain": 0.01,
            "material_structure_increment": 0.005,
            "flat_qa_macro_f1_points": 0.5,
            "relation_dominance_share": 0.8,
        },
        "caveats": [
            "Oracle reranking is limited to the frozen stored candidate pool (normally Top-500).",
            "Unparseable or misaligned logical forms retain exact baseline order and remain in full-cohort QA metrics.",
            "The direction arm is SRD (relation-slot/branch plus KG direction), so direction value is measured by direction minus relation_slot_branch.",
            "The recommendation is provisional until both WebQSP and CWQ fixed-LLM runs are complete.",
            "Retrieval/QA comparisons use the fixed QA cohort when it is a subset of the retrieval cohort.",
            "Structural graph matching is exact only when structure_match_truncated is zero; the report blocks structural conclusions otherwise.",
        ],
    }
    lines = [
        "# Phase 0–2 Gold Query-Factor Oracle Report",
        "",
        f"Phase-0 baseline status: `{phase0_status}`.",
        "",
        f"Evaluation budget: prompt-effective Top-{budget}. Candidate universe: frozen stored retrieval pool. "
        f"Primary comparison scope: `{primary_scope}`.",
        f"Structural match truncation (maximum across variants): "
        f"{_fmt(maximum_structure_truncation)}%.",
        "",
        "## Retrieval summary",
        "",
        f"Stored-pool ceiling is measured at effective Top-{pool_budget}; answer-recall ceiling: "
        f"{_fmt(pool_answer_ceiling)}%.",
        "",
        "| Variant | Ans recall | Ans hit | ΔAnsR | Path triple R | GPT triple R | Relation cov. | Slot cov. | Directed slot cov. | Complete branch cov. | Full structure hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        scope = retrieval["variants"][variant]["budgets"][str(budget)][primary_scope]
        answer = scope.get("answer_recall")
        lines.append(
            f"| {variant} | {_fmt(answer)} | {_fmt(scope.get('answer_hit'))} | "
            f"{_fmt(_difference(answer, baseline_answer))} | "
            f"{_fmt(scope.get('triple_recall'))} | {_fmt(scope.get('gpt_triple_recall'))} | "
            f"{_fmt(scope.get('gold_relation_coverage'))} | {_fmt(scope.get('slot_coverage'))} | "
            f"{_fmt(scope.get('slot_direction_coverage'))} | "
            f"{_fmt(scope.get('branch_complete_coverage'))} | "
            f"{_fmt(scope.get('full_structure_hit'))} |"
        )
    lines += ["", "## Final QA", ""]
    if not qa:
        lines.append("Pending remote LLM runs; no QA result has been fabricated.")
    else:
        lines += [
            "| Variant | Hit (%) | ΔHit (pp) | Hit@1 (%) | Macro F1 (%) | ΔF1 (pp) | Micro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        baseline_qa = qa.get("baseline", {}).get("full", {})
        for variant, value in sorted(qa.items()):
            full = value["full"]
            lines.append(
                f"| {variant} | {_fmt(full.get('hit'), percent=False)} | "
                f"{_fmt(_difference(full.get('hit'), baseline_qa.get('hit')), percent=False)} | "
                f"{_fmt(full.get('hit_at_1'), percent=False)} | "
                f"{_fmt(full.get('macro_f1'), percent=False)} | "
                f"{_fmt(_difference(full.get('macro_f1'), baseline_qa.get('macro_f1')), percent=False)} | "
                f"{_fmt(full.get('micro_f1'), percent=False)} |"
            )
        if missing_qa_variants:
            lines += ["", f"Pending QA variants: {', '.join(missing_qa_variants)}."]
        if qa_family_gains:
            lines += [
                "",
                "### Final-QA gains by query family",
                "",
                "| Variant | Family | N | ΔHit (pp) | ΔMacro-F1 (pp) |",
                "|---|---|---:|---:|---:|",
            ]
            for row in qa_family_gains:
                lines.append(
                    f"| {row['variant']} | {row['query_family']} | {row['sample_count']} | "
                    f"{_fmt(row['hit_delta_points'], percent=False)} | "
                    f"{_fmt(row['macro_f1_delta_points'], percent=False)} |"
                )
        if paired_sample_diagnostics:
            lines += [
                "",
                "### Retrieval-to-QA paired diagnostics",
                "",
                "| Variant | AnsR improved N | Hit 0→1 | F1 improved | F1 unchanged | Hit 1→0 (all) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for variant, values in paired_sample_diagnostics.items():
                lines.append(
                    f"| {variant} | {values['answer_recall_improved_count']} | "
                    f"{values['answer_recall_improved_hit_0_to_1_count']} | "
                    f"{values['answer_recall_improved_f1_improved_count']} | "
                    f"{values['answer_recall_improved_f1_unchanged_count']} | "
                    f"{values['overall_hit_1_to_0_count']} |"
                )
    lines += [
        "",
        "## Conclusions",
        "",
        f"1. Most valuable retrieval variant: `{report['factor_value']['most_valuable_retrieval_variant']}`.",
        f"2. Joint oracle answer-recall headroom: {_fmt(joint_gain)} percentage points; final QA Macro-F1 headroom: "
        f"{_fmt(qa_headroom, percent=False)} points.",
        f"3. Largest answer-recall gain by family: `{family_gains[0]['query_family'] if family_gains else 'N/A'}` "
        f"({_fmt(family_gains[0]['delta']) if family_gains else 'N/A'} pp).",
        f"4. Recommendation: {recommendation}.",
        "",
        "The thresholds and all denominators are stored in `final_report.json`; interpret family slices together with sample counts.",
    ]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "final_report.json", report)
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
