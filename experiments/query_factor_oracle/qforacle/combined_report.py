from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file, write_json


def _fmt(value: float | None, scale: float = 1.0) -> str:
    return "N/A" if value is None else f"{value * scale:.2f}"


def build_combined_report(
    report_paths: dict[str, str],
    output_dir: str,
) -> dict[str, Any]:
    if set(report_paths) != {"webqsp", "cwq"}:
        raise ValueError("Combined conclusion requires exactly webqsp and cwq reports")
    resolved_paths = {
        dataset: str(Path(path).resolve()) for dataset, path in report_paths.items()
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise ValueError("Combined conclusion cannot reuse the same report for both datasets")
    reports = {dataset: read_json(path) for dataset, path in report_paths.items()}
    for dataset, report in reports.items():
        if report.get("dataset") != dataset:
            raise ValueError(
                f"Report labelled {dataset!r} belongs to dataset "
                f"{report.get('dataset')!r}"
            )
        if report.get("split") != "test":
            raise ValueError(
                f"Combined Phase 0-2 conclusion requires the test split; "
                f"{dataset!r} report uses {report.get('split')!r}"
            )
        source_inputs = report.get("inputs", {})
        for key in (
            "retrieval_summary_sha256", "baseline_semantic_sha256",
            "factors_sha256", "sample_id_order_sha256",
        ):
            if not source_inputs.get(key):
                raise ValueError(
                    f"{dataset!r} report is missing required input fingerprint {key!r}"
                )
    incomplete = {
        dataset: {
            "phase0_status": report.get("phase0_status"),
            "qa_complete": report.get("qa_complete"),
        }
        for dataset, report in reports.items()
        if report.get("phase0_status") != "confirmed" or not report.get("qa_complete")
    }
    rows = {}
    for dataset, report in reports.items():
        retrieval = report["retrieval_headroom"]
        factor = report["factor_value"]
        best_retrieval_family = next(iter(report.get("family_gains", [])), None)
        joint_qa_rows = [
            row for row in report.get("qa_family_gains", [])
            if row.get("variant") == "all_factors"
        ]
        best_qa_family = max(
            joint_qa_rows,
            key=lambda row: row.get("macro_f1_delta_points")
            if row.get("macro_f1_delta_points") is not None else float("-inf"),
            default=None,
        )
        rows[dataset] = {
            "relation_set_answer_recall_delta": factor.get("relation_set_answer_recall_delta"),
            "slot_branch_increment_over_relation_set": factor.get(
                "slot_branch_increment_over_relation_set"
            ),
            "direction_increment_over_slot_branch": factor.get(
                "direction_increment_over_slot_branch"
            ),
            "joint_increment_over_direction": factor.get(
                "joint_increment_over_direction"
            ),
            "relation_share_of_joint_gain": factor.get("relation_share_of_joint_gain"),
            "joint_answer_recall_delta": retrieval.get("answer_recall_delta"),
            "joint_triple_recall_delta": retrieval.get("triple_recall_delta"),
            "joint_qa_macro_f1_delta_points": report.get("qa_headroom_macro_f1_points"),
            "joint_qa_hit_delta_points": report.get("qa_headroom_hit_points"),
            "maximum_structure_match_truncation_rate": report.get(
                "maximum_structure_match_truncation_rate"
            ),
            "best_retrieval_family": best_retrieval_family,
            "best_joint_qa_family": best_qa_family,
        }

    required_decision_metrics = (
        "relation_set_answer_recall_delta",
        "slot_branch_increment_over_relation_set",
        "direction_increment_over_slot_branch",
        "joint_increment_over_direction",
        "joint_answer_recall_delta",
        "joint_qa_macro_f1_delta_points",
        "joint_qa_hit_delta_points",
        "maximum_structure_match_truncation_rate",
    )
    for dataset, row in rows.items():
        missing_metrics = [
            metric for metric in required_decision_metrics if row.get(metric) is None
        ]
        if missing_metrics:
            incomplete.setdefault(dataset, {
                "phase0_status": reports[dataset].get("phase0_status"),
                "qa_complete": reports[dataset].get("qa_complete"),
            })["missing_decision_metrics"] = missing_metrics
        truncation_rate = row.get("maximum_structure_match_truncation_rate")
        if truncation_rate is not None and float(truncation_rate) > 0:
            incomplete.setdefault(dataset, {
                "phase0_status": reports[dataset].get("phase0_status"),
                "qa_complete": reports[dataset].get("qa_complete"),
            })["maximum_structure_match_truncation_rate"] = float(truncation_rate)

    if incomplete:
        recommendation = (
            "blocked: resolve incomplete inputs and structural-match truncation "
            "before drawing a cross-dataset conclusion"
        )
    else:
        joint = [float(rows[name]["joint_answer_recall_delta"]) for name in rows]
        structure = [
            float(rows[name]["slot_branch_increment_over_relation_set"])
            for name in rows
        ]
        qa_f1 = [
            abs(float(rows[name]["joint_qa_macro_f1_delta_points"])) for name in rows
        ]
        relation_shares = [
            rows[name]["relation_share_of_joint_gain"]
            for name in rows if rows[name]["relation_share_of_joint_gain"] is not None
        ]
        if max(joint) < 0.005:
            recommendation = "pause predicted-factor development: joint gold factors have little retrieval headroom"
        elif min(joint) >= 0.01 and max(qa_f1) < 0.5:
            recommendation = "prioritize sharing query state with reasoning: retrieval gains do not transfer to QA"
        elif max(structure) >= 0.005:
            recommendation = "proceed to predicted structured factors: slot/branch adds value beyond relation identity"
        elif relation_shares and min(relation_shares) >= 0.8:
            recommendation = "prioritize relation grounding: relation identity explains most joint oracle gain"
        else:
            recommendation = "proceed cautiously; validate predicted factors and shared reasoning on both datasets"

    combined = {
        "schema_version": 1,
        "datasets": rows,
        "input_reports": {
            dataset: {
                "path": resolved_paths[dataset],
                "sha256": sha256_file(report_paths[dataset]),
                "source_inputs": reports[dataset]["inputs"],
            }
            for dataset in report_paths
        },
        "incomplete": incomplete,
        "recommendation": recommendation,
        "answers": {
            "most_valuable_factor": {
                dataset: reports[dataset]["factor_value"]["most_valuable_retrieval_variant"]
                for dataset in reports
            },
            "oracle_headroom": {
                dataset: {
                    "answer_recall_delta": rows[dataset]["joint_answer_recall_delta"],
                    "macro_f1_delta_points": rows[dataset]["joint_qa_macro_f1_delta_points"],
                    "hit_delta_points": rows[dataset]["joint_qa_hit_delta_points"],
                }
                for dataset in reports
            },
            "gain_concentration": {
                dataset: {
                    "retrieval": rows[dataset]["best_retrieval_family"],
                    "final_qa": rows[dataset]["best_joint_qa_family"],
                }
                for dataset in reports
            },
            "next_stage": recommendation,
        },
    }
    lines = [
        "# WebQSP + CWQ Phase 0–2 Combined Conclusion",
        "",
        f"Decision: **{recommendation}**",
        "",
        "| Dataset | Relation ΔAnsR (pp) | Slot/branch + (pp) | Direction + (pp) | Type/operator + (pp) | Joint ΔAnsR (pp) | Joint ΔF1 (pp) | Joint ΔHit (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("webqsp", "cwq"):
        row = rows[dataset]
        lines.append(
            f"| {dataset} | {_fmt(row['relation_set_answer_recall_delta'], 100)} | "
            f"{_fmt(row['slot_branch_increment_over_relation_set'], 100)} | "
            f"{_fmt(row['direction_increment_over_slot_branch'], 100)} | "
            f"{_fmt(row['joint_increment_over_direction'], 100)} | "
            f"{_fmt(row['joint_answer_recall_delta'], 100)} | "
            f"{_fmt(row['joint_qa_macro_f1_delta_points'])} | "
            f"{_fmt(row['joint_qa_hit_delta_points'])} |"
        )
    lines += ["", "## Four requested answers", ""]
    for index, label in enumerate((
        "Most valuable factor: see each dataset's selected retrieval variant and relation/structure increments above.",
        "Oracle headroom: joint retrieval, Macro-F1 and Hit deltas are reported in the table.",
        "Gain concentration: the JSON records the largest retrieval and joint-QA family slice with N.",
        f"Proceed decision: {recommendation}.",
    ), start=1):
        lines.append(f"{index}. {label}")
    if incomplete:
        lines += ["", f"Incomplete inputs: {incomplete}. No scientific conclusion should be drawn."]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "combined_report.json", combined)
    (output / "combined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined
