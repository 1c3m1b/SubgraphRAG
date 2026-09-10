from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.query_factor_oracle.qforacle.baseline import freeze_baseline
from experiments.query_factor_oracle.qforacle.factors import build_factors
from experiments.query_factor_oracle.qforacle.io_utils import (
    load_torch,
    read_json,
    sha256_file,
    write_jsonl,
)
from experiments.query_factor_oracle.qforacle.phase2 import run_oracle_experiment
from experiments.query_factor_oracle.qforacle.evaluation import evaluate_predictions
from experiments.query_factor_oracle.qforacle.reasoning import (
    _initialize_core_llm,
    _load_pinned_reasoning_data,
    run_reasoning,
)
from experiments.query_factor_oracle.qforacle.report import build_report


class PipelineTest(unittest.TestCase):
    def test_gzip_jsonl_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            path = Path(temp_value) / "artifact.jsonl.gz"
            rows = [{"id": "q1", "value": 1}]
            write_jsonl(path, rows)
            first = sha256_file(path)
            write_jsonl(path, rows)
            self.assertEqual(first, sha256_file(path))

    def test_reasoning_dataset_revision_is_injected_and_restored(self) -> None:
        class TinyDataset(list):
            _fingerprint = "fixture-fingerprint"

        calls = []
        original_get_subgraphs = lambda *args: None
        prepare_data = types.SimpleNamespace()
        prepare_data.get_subgraphs = original_get_subgraphs
        prepare_data.load_dataset = lambda repo_id, **kwargs: (
            calls.append((repo_id, kwargs)) or TinyDataset([{"id": "q1"}])
        )

        def fake_get_data(dataset, pred, retrieval, split, prompt_mode):
            loaded = prepare_data.get_subgraphs(dataset, split)
            return list(loaded)

        prepare_data.get_data = fake_get_data
        config = {
            "dataset": "synthetic", "split": "test",
            "data": {"reasoning_dataset": {"repo_id": "owner/data", "revision": "abc123"}},
        }
        data, runtime = _load_pinned_reasoning_data(
            prepare_data, config, Path("pred.jsonl"), Path("retrieval.pth"), "scored_1"
        )
        self.assertEqual(data, [{"id": "q1"}])
        self.assertEqual(calls, [("owner/data", {"split": "test", "revision": "abc123"})])
        self.assertEqual(runtime["dataset_fingerprint"], "fixture-fingerprint")
        self.assertIs(prepare_data.get_subgraphs, original_get_subgraphs)

    def test_runtime_model_path_is_injected_without_changing_dispatch(self) -> None:
        captured = {}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        module = types.SimpleNamespace(LLM=FakeLLM)

        def fake_llm_init(model_name, *args):
            self = module.LLM(model=model_name)
            captured["configured_model"] = model_name
            return self

        module.llm_init = fake_llm_init
        original = module.LLM
        _initialize_core_llm(
            module, "meta-llama/model", "/snapshots/path-containing-gpt/model", {},
        )
        self.assertEqual(captured["configured_model"], "meta-llama/model")
        self.assertEqual(captured["kwargs"]["model"], "/snapshots/path-containing-gpt/model")
        self.assertIs(module.LLM, original)

    def test_phase0_to_phase2_cpu_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            baseline_path = temp / "baseline.pth"
            baseline = {
                "q1": {
                    "question": "Who is connected?",
                    "q_entity": ["topic"], "q_entity_in_graph": ["topic"],
                    "a_entity": ["answer"], "a_entity_in_graph": ["answer"],
                    "max_path_length": 1,
                    "target_relevant_triples": [("topic", "gold.r", "answer")],
                    "scored_triples": [
                        ("topic", "noise.r", "noise", 0.9),
                        ("topic", "gold.r", "answer", 0.8),
                    ],
                },
                "q2": {
                    "question": "Missing LF?",
                    "q_entity": ["t"], "q_entity_in_graph": ["t"],
                    "a_entity": [], "a_entity_in_graph": [],
                    "max_path_length": None, "target_relevant_triples": [],
                    "scored_triples": [("t", "r", "x", 0.6)],
                },
            }
            torch.save(baseline, baseline_path)
            logical_forms = temp / "lf.json"
            logical_forms.write_text(json.dumps([{
                "id": "q1", "question": "Who is connected?",
                "SExpr": "(JOIN (R gold.r) m.topic)"
            }]), encoding="utf-8")
            config = {
                "dataset": "synthetic", "split": "test",
                "retrieval_result": str(baseline_path),
                "protocol": {"prompt_top_k": 1},
                "data": {"dataset_sources": []},
                "_repo_root": str(Path(__file__).resolve().parents[3]),
            }
            phase0 = temp / "phase0"
            freeze_baseline(config, phase0, copy_retrieval=True)
            self.assertTrue((phase0 / "baseline.lock.json").is_file())
            phase1 = temp / "phase1"
            summary = build_factors(
                str(baseline_path), [str(logical_forms)], str(phase1), "synthetic", "test"
            )
            self.assertEqual(summary["cohort_sample_count"], 2)
            self.assertEqual(summary["parseable_count"], 1)
            phase2 = temp / "phase2"
            qa_cohort = temp / "qa.jsonl"
            qa_cohort.write_text('{"id":"q1"}\n', encoding="utf-8")
            result = run_oracle_experiment(
                str(baseline_path), str(phase1 / "query_factors.jsonl.gz"),
                str(phase2), [1, 2], 2, qa_cohort_path=str(qa_cohort),
            )
            self.assertEqual(set(result["variants"]), {
                "baseline", "family_structure", "relation_set",
                "relation_slot_branch", "direction", "all_factors",
            })
            self.assertEqual(result["inputs"]["qa_cohort_sample_count"], 1)
            self.assertEqual(
                result["variants"]["baseline"]["budgets"]["1"]["qa_cohort"]["sample_count"],
                1,
            )
            baseline_copy = load_torch(phase2 / "baseline" / "retrieval_result.pth")
            self.assertEqual(baseline_copy, baseline)
            # Missing LF q2 must remain bit-for-bit equivalent at the object level.
            all_factors = load_torch(phase2 / "all_factors" / "retrieval_result.pth")
            self.assertEqual(all_factors["q2"], baseline["q2"])
            report_dir = temp / "report"
            report = build_report(str(phase2 / "retrieval_summary.json"), {}, str(report_dir), 1)
            self.assertFalse(report["qa_complete"])
            self.assertEqual(report["retrieval_scope"], "qa_cohort")
            self.assertTrue((report_dir / "final_report.md").is_file())
            self.assertEqual(read_json(report_dir / "final_report.json")["budget"], 1)

    def test_failed_strict_confirmation_preserves_existing_phase0(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            baseline_path = temp / "baseline.pth"
            torch.save({
                "q1": {
                    "question": "Who?", "q_entity": ["topic"],
                    "q_entity_in_graph": ["topic"], "a_entity": ["answer"],
                    "a_entity_in_graph": ["answer"], "max_path_length": 1,
                    "target_relevant_triples": [("topic", "r", "answer")],
                    "scored_triples": [("topic", "r", "answer", 0.9)],
                },
            }, baseline_path)
            config = {
                "dataset": "synthetic", "split": "test",
                "retrieval_result": str(baseline_path),
                "protocol": {"prompt_top_k": 1},
                "data": {"dataset_sources": []},
                "_repo_root": str(Path(__file__).resolve().parents[3]),
            }
            output = temp / "phase0"
            freeze_baseline(config, output, copy_retrieval=True)
            environment = read_json(output / "environment.json")
            self.assertIn("huggingface_hub", environment["packages"])
            self.assertIn("openai", environment["packages"])
            self.assertIn("tqdm", environment["packages"])
            self.assertIn("hf_hub_offline", environment)
            protected = {
                relative: (output / relative).read_bytes()
                for relative in (
                    "baseline.lock.json", "validation.json", "manifest.json",
                    "retrieval/baseline_retrieval.pth",
                )
            }
            with self.assertRaisesRegex(ValueError, "not confirmed"):
                freeze_baseline(
                    config,
                    output,
                    copy_retrieval=True,
                    verify_lock_path=output / "baseline.lock.json",
                    require_confirmed=True,
                )
            for relative, expected in protected.items():
                self.assertEqual((output / relative).read_bytes(), expected, relative)
            attempt = read_json(output / "last_confirmation_attempt.json")
            self.assertEqual(attempt["phase0_status"], "not_measured")

    def test_reasoning_wrapper_resume_and_evaluator_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            temp = Path(temp_value)
            retrieval_path = temp / "baseline.pth"
            retrieval = {
                "q1": {
                    "question": "Who?", "q_entity": ["topic"],
                    "q_entity_in_graph": ["topic"], "a_entity": ["answer"],
                    "a_entity_in_graph": ["answer"], "max_path_length": 1,
                    "target_relevant_triples": [("topic", "r", "answer")],
                    "scored_triples": [("topic", "r", "answer", 0.9)],
                },
                "q2": {
                    "question": "Retrieval only?", "q_entity": ["other"],
                    "q_entity_in_graph": ["other"], "a_entity": [],
                    "a_entity_in_graph": [], "max_path_length": None,
                    "target_relevant_triples": [],
                    "scored_triples": [("other", "r", "x", 0.8)],
                },
            }
            torch.save(retrieval, retrieval_path)
            rog_path = temp / "rog.jsonl"
            rog_path.write_text('{"id":"q1"}\n', encoding="utf-8")
            model_path = temp / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"model_type":"synthetic"}', encoding="utf-8")
            repo = Path(__file__).resolve().parents[3]
            config = {
                "dataset": "synthetic", "split": "test", "_repo_root": str(repo),
                "data": {"rog_prediction_file": str(rog_path)},
                "protocol": {"prompt_mode": "scored_1", "prompt_top_k": 1, "threshold": 0.0},
                "llm": {
                    "model_name": str(model_path), "llm_mode": "sys_icl_dc_repro",
                    "resolved_revision": "synthetic-test-fixture",
                    "tensor_parallel_size": 1, "max_seq_len_to_capture": 128,
                    "max_tokens": 16, "seed": 0, "temperature": 0.0,
                    "frequency_penalty": 0.0,
                },
            }

            def fake_get_data(*args, **kwargs):
                return [{
                    "id": "q1", "question": "Who?", "ground_truth": ["answer"],
                    "graph": [("topic", "r", "answer")], "good_paths_rog": [],
                    "good_triplets_rog": [], "scored_triplets": [("topic", "r", "answer", 0.9)],
                }]

            def fake_prompts(data, *args):
                for row in data:
                    row.update({
                        "sys_query": "system", "user_query": "Triplets: topic,r,answer\nQuestion: Who?",
                        "all_query": "unused", "cot_query": "format",
                    })
                return data

            fake_prompt_module = types.SimpleNamespace(
                icl_sys_prompt="system", icl_cot_prompt="format",
                sys_prompt="system", cot_prompt="format",
                noevi_sys_prompt="system", noevi_cot_prompt="format",
                sys_prompt_gpt="system", cot_prompt_gpt="format",
            )
            fake_prepare_data = types.SimpleNamespace(get_data=fake_get_data)
            fake_llm_utils = types.SimpleNamespace(
                llm_init=lambda *args: object(),
                llm_inf_all=lambda *args: ["ans: answer", ""],
            )
            fake_core = (
                fake_prepare_data,
                fake_prompts,
                fake_llm_utils,
                fake_prompt_module,
            )
            output = temp / "reasoning"
            with patch(
                "experiments.query_factor_oracle.qforacle.reasoning._load_core",
                return_value=fake_core,
            ):
                first = run_reasoning(config, str(retrieval_path), str(output), "baseline", 0)
                second = run_reasoning(config, str(retrieval_path), str(output), "baseline", 0)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first["run_fingerprint"], second["run_fingerprint"])
            self.assertEqual(len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 1)

            # A drifted resume must fail without replacing the sidecars that
            # document the existing predictions.
            prompt_bytes = (output / "prompts.jsonl.gz").read_bytes()
            manifest_bytes = (output / "run_manifest.json").read_bytes()
            prediction_bytes = (output / "predictions.jsonl").read_bytes()

            def drifted_prompts(data, *args):
                for row in data:
                    row.update({
                        "sys_query": "changed system",
                        "user_query": "Triplets: changed\nQuestion: Who?",
                        "all_query": "unused",
                        "cot_query": "changed format",
                    })
                return data

            drifted_core = (
                fake_prepare_data,
                drifted_prompts,
                fake_llm_utils,
                fake_prompt_module,
            )
            with patch(
                "experiments.query_factor_oracle.qforacle.reasoning._load_core",
                return_value=drifted_core,
            ):
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    run_reasoning(config, str(retrieval_path), str(output), "baseline", 0)
            self.assertEqual((output / "prompts.jsonl.gz").read_bytes(), prompt_bytes)
            self.assertEqual((output / "run_manifest.json").read_bytes(), manifest_bytes)
            self.assertEqual((output / "predictions.jsonl").read_bytes(), prediction_bytes)

            # The upstream metric module imports datasets at module load even
            # though its pure answer helpers do not use it.  A tiny stub lets
            # this CPU test verify that the adapter calls those exact helpers.
            datasets_stub = types.ModuleType("datasets")
            datasets_stub.load_dataset = lambda *args, **kwargs: None
            with patch.dict(sys.modules, {"datasets": datasets_stub}):
                summary = evaluate_predictions(
                    str(repo), str(output / "predictions.jsonl"), str(retrieval_path),
                    str(output / "evaluation"), None,
                    expected_qa_path=str(rog_path),
                    allow_retrieval_only_samples=True,
                )
            self.assertEqual(summary["full"]["hit"], 100.0)
            self.assertEqual(summary["full"]["hit_at_1"], 100.0)
            self.assertEqual(summary["full"]["macro_f1"], 100.0)
            self.assertEqual(summary["retrieval_cohort_sample_count"], 2)
            self.assertEqual(summary["qa_cohort_sample_count"], 1)
            self.assertEqual(summary["retrieval_only_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
