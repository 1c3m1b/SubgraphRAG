from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.query_factor_oracle.qforacle.combined_report import (
    build_combined_report,
)
from experiments.query_factor_oracle.qforacle.report import (
    _paired_sample_diagnostic,
    build_report,
)
from experiments.query_factor_oracle.qforacle.io_utils import sha256_file


def _retrieval_summary(answer_values: dict[str, float | None]) -> dict:
    variants = {}
    for variant in (
        "baseline", "family_structure", "relation_set",
        "relation_slot_branch", "direction", "all_factors",
    ):
        answer = answer_values.get(variant)
        metrics = {
            "sample_count": 1,
            "answer_recall": answer,
            "answer_hit": answer,
            "triple_recall": answer,
            "gpt_triple_recall": answer,
            "gold_relation_coverage": answer,
            "slot_coverage": answer,
            "slot_direction_coverage": answer,
            "branch_complete_coverage": answer,
            "full_structure_hit": answer,
            "structure_match_truncated": 0.0,
        }
        variants[variant] = {
            "retrieval_semantic_sha256": f"retrieval-{variant}",
            "budgets": {
                "100": {
                    "overall": metrics,
                    "by_family": {},
                }
            },
        }
    return {
        "inputs": {
            "factor_dataset": "webqsp",
            "factor_split": "test",
            "baseline_semantic_sha256": "baseline-semantic",
            "factors_sha256": "factor-sha",
            "sample_id_order_sha256": "retrieval-ids",
            "qa_cohort_id_order_sha256": None,
        },
        "variants": variants,
        "paired_transitions": {},
    }


def _final_report(dataset: str, **overrides) -> dict:
    value = {
        "dataset": dataset,
        "split": "test",
        "inputs": {
            "retrieval_summary_sha256": f"summary-{dataset}",
            "baseline_semantic_sha256": f"baseline-{dataset}",
            "factors_sha256": f"factors-{dataset}",
            "sample_id_order_sha256": f"ids-{dataset}",
        },
        "phase0_status": "confirmed",
        "qa_complete": True,
        "maximum_structure_match_truncation_rate": 0.0,
        "retrieval_headroom": {
            "answer_recall_delta": 0.02,
            "triple_recall_delta": 0.01,
        },
        "factor_value": {
            "relation_set_answer_recall_delta": 0.015,
            "slot_branch_increment_over_relation_set": 0.005,
            "direction_increment_over_slot_branch": 0.002,
            "joint_increment_over_direction": 0.003,
            "relation_share_of_joint_gain": 0.75,
            "most_valuable_retrieval_variant": "all_factors",
        },
        "qa_headroom_macro_f1_points": 0.6,
        "qa_headroom_hit_points": 0.5,
        "family_gains": [],
        "qa_family_gains": [],
    }
    value.update(overrides)
    return value


class ReportProvenanceTest(unittest.TestCase):
    def test_complete_report_joins_hashed_retrieval_and_qa_samples(self) -> None:
        variants = (
            "baseline", "family_structure", "relation_set",
            "relation_slot_branch", "direction", "all_factors",
        )
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            summary = _retrieval_summary({
                variant: (0.0 if variant == "baseline" else 1.0)
                for variant in variants
            })
            summary["inputs"]["qa_cohort_id_order_sha256"] = "qa-ids"
            qa_paths = {}
            for variant in variants:
                retrieval_rows = temp / f"retrieval-{variant}.jsonl"
                retrieval_rows.write_text(json.dumps({
                    "id": "q1",
                    "answer_recall": 0.0 if variant == "baseline" else 1.0,
                }) + "\n", encoding="utf-8")
                summary["variants"][variant]["sample_metric_artifacts"] = {
                    "100": {
                        "path": str(retrieval_rows),
                        "sha256": sha256_file(retrieval_rows),
                    }
                }
                qa_rows = temp / f"qa-{variant}.jsonl"
                qa_rows.write_text(json.dumps({
                    "id": "q1",
                    "hit": 0.0 if variant == "baseline" else 1.0,
                    "f1": 0.0 if variant == "baseline" else 1.0,
                }) + "\n", encoding="utf-8")
                qa_summary = {
                    "variant": variant,
                    "retrieval_semantic_sha256": f"retrieval-{variant}",
                    "fixed_reasoning_protocol_sha256": "protocol",
                    "factors_sha256": "factor-sha",
                    "retrieval_cohort_id_order_sha256": "retrieval-ids",
                    "qa_cohort_sample_count": 1,
                    "qa_cohort_id_order_sha256": "qa-ids",
                    "qa_question_collection_sha256": "questions",
                    "qa_ground_truth_collection_sha256": "answers",
                    "sample_metrics_path": str(qa_rows),
                    "sample_metrics_sha256": sha256_file(qa_rows),
                    "full": {
                        "macro_f1": 0.0 if variant == "baseline" else 100.0,
                        "hit": 0.0 if variant == "baseline" else 100.0,
                        "hit_at_1": 0.0,
                        "micro_f1": 0.0,
                    },
                    "by_query_family": {},
                }
                qa_path = temp / f"qa-summary-{variant}.json"
                qa_path.write_text(json.dumps(qa_summary), encoding="utf-8")
                qa_paths[variant] = str(qa_path)
            summary_path = temp / "retrieval.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = build_report(
                str(summary_path), qa_paths, str(temp / "report"), 100
            )
            self.assertTrue(result["qa_complete"])
            self.assertEqual(
                result["paired_retrieval_to_qa_diagnostics"]["all_factors"][
                    "answer_recall_improved_hit_0_to_1_count"
                ],
                1,
            )

    def test_paired_diagnostic_uses_qa_subset_and_tracks_non_transfer(self) -> None:
        baseline_retrieval = [
            {"id": "q1", "answer_recall": 0.0},
            {"id": "q2", "answer_recall": 0.5},
            {"id": "retrieval-only", "answer_recall": 0.0},
        ]
        current_retrieval = [
            {"id": "q1", "answer_recall": 1.0},
            {"id": "q2", "answer_recall": 1.0},
            {"id": "retrieval-only", "answer_recall": 1.0},
        ]
        baseline_qa = [
            {"id": "q1", "hit": 0.0, "f1": 0.0},
            {"id": "q2", "hit": 1.0, "f1": 1.0},
        ]
        current_qa = [
            {"id": "q1", "hit": 0.0, "f1": 0.0},
            {"id": "q2", "hit": 0.0, "f1": 0.0},
        ]
        result = _paired_sample_diagnostic(
            baseline_retrieval, current_retrieval, baseline_qa, current_qa
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["answer_recall_improved_count"], 2)
        self.assertEqual(result["answer_recall_improved_f1_unchanged_count"], 1)
        self.assertEqual(result["overall_hit_1_to_0_count"], 1)

    def test_report_records_identity_and_distinguishes_zero_from_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            summary_path = temp / "retrieval.json"
            summary_path.write_text(json.dumps(_retrieval_summary({
                "baseline": None,
                "relation_set": 0.0,
            })), encoding="utf-8")
            report = build_report(str(summary_path), {}, str(temp / "out"), 100)
            self.assertEqual(report["dataset"], "webqsp")
            self.assertEqual(report["split"], "test")
            self.assertEqual(
                report["factor_value"]["most_valuable_retrieval_variant"],
                "relation_set",
            )
            self.assertEqual(report["inputs"]["factors_sha256"], "factor-sha")
            self.assertTrue(report["inputs"]["retrieval_summary_sha256"])

    def test_report_rejects_qa_factor_artifact_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            summary = _retrieval_summary({variant: 0.5 for variant in (
                "baseline", "family_structure", "relation_set",
                "relation_slot_branch", "direction", "all_factors",
            )})
            summary_path = temp / "retrieval.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            qa = {
                "variant": "baseline",
                "retrieval_semantic_sha256": "retrieval-baseline",
                "fixed_reasoning_protocol_sha256": "protocol",
                "factors_sha256": "wrong-factor",
                "retrieval_cohort_id_order_sha256": "retrieval-ids",
                "qa_cohort_sample_count": 1,
                "qa_cohort_id_order_sha256": "qa-ids",
                "qa_question_collection_sha256": "questions",
                "qa_ground_truth_collection_sha256": "answers",
                "full": {"macro_f1": 1.0, "hit": 1.0},
            }
            qa_path = temp / "qa.json"
            qa_path.write_text(json.dumps(qa), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query-factor artifact"):
                build_report(
                    str(summary_path), {"baseline": str(qa_path)},
                    str(temp / "out"), 100,
                )

    def test_combined_report_rejects_duplicate_and_swapped_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            web = temp / "web.json"
            cwq = temp / "cwq.json"
            web.write_text(json.dumps(_final_report("webqsp")), encoding="utf-8")
            cwq.write_text(json.dumps(_final_report("cwq")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reuse the same report"):
                build_combined_report(
                    {"webqsp": str(web), "cwq": str(web)}, str(temp / "same")
                )
            with self.assertRaisesRegex(ValueError, "belongs to dataset"):
                build_combined_report(
                    {"webqsp": str(cwq), "cwq": str(web)}, str(temp / "swapped")
                )

    def test_combined_report_blocks_on_missing_metric_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            web_value = _final_report("webqsp")
            web_value["retrieval_headroom"]["answer_recall_delta"] = None
            web = temp / "web.json"
            cwq = temp / "cwq.json"
            web.write_text(json.dumps(web_value), encoding="utf-8")
            cwq.write_text(json.dumps(_final_report("cwq")), encoding="utf-8")
            result = build_combined_report(
                {"webqsp": str(web), "cwq": str(cwq)}, str(temp / "out")
            )
            self.assertIn("blocked", result["recommendation"])
            self.assertIn(
                "joint_answer_recall_delta",
                result["incomplete"]["webqsp"]["missing_decision_metrics"],
            )

    def test_combined_report_blocks_on_structure_match_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            web_value = _final_report(
                "webqsp", maximum_structure_match_truncation_rate=0.01,
            )
            web = temp / "web.json"
            cwq = temp / "cwq.json"
            web.write_text(json.dumps(web_value), encoding="utf-8")
            cwq.write_text(json.dumps(_final_report("cwq")), encoding="utf-8")
            result = build_combined_report(
                {"webqsp": str(web), "cwq": str(cwq)}, str(temp / "out")
            )
            self.assertIn("blocked", result["recommendation"])
            self.assertEqual(
                result["incomplete"]["webqsp"][
                    "maximum_structure_match_truncation_rate"
                ],
                0.01,
            )


if __name__ == "__main__":
    unittest.main()
