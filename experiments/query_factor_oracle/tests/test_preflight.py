from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.query_factor_oracle.preflight import (
    REQUIRED_RUNTIME_PACKAGES,
    audit_runtime_dependencies,
    resolve_model_locally,
    validate_protocol,
)


class PreflightTest(unittest.TestCase):
    def test_local_snapshot_revision_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_value:
            snapshot = Path(temp_value) / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            result = resolve_model_locally(str(snapshot), "abc123")
            self.assertEqual(result["snapshot_commit"], "abc123")
            with self.assertRaisesRegex(ValueError, "disagree"):
                resolve_model_locally(str(snapshot), "different")

    def test_protocol_budget_is_consistent_and_present_in_phase2(self) -> None:
        result = validate_protocol({
            "protocol": {"prompt_mode": "scored_100", "prompt_top_k": 100},
            "phase2": {"budgets": [50, 100, 200]},
        })
        self.assertEqual(result["prompt_top_k"], 100)
        self.assertEqual(result["prompt_mode"], "scored_100")

        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            validate_protocol({
                "protocol": {"prompt_mode": "scored_50", "prompt_top_k": 100},
                "phase2": {"budgets": [50, 100]},
            })
        with self.assertRaisesRegex(ValueError, "absent from phase2"):
            validate_protocol({
                "protocol": {"prompt_mode": "scored_100", "prompt_top_k": 100},
                "phase2": {"budgets": [50, 200]},
            })
        with self.assertRaisesRegex(ValueError, "exact form scored_K"):
            validate_protocol({
                "protocol": {"prompt_mode": "scored_100_extra", "prompt_top_k": 100},
                "phase2": {"budgets": [100]},
            })

    def test_dependency_audit_is_metadata_only_and_fails_on_missing(self) -> None:
        installed = {package: "1.0-test" for package in REQUIRED_RUNTIME_PACKAGES}
        installed["networkx"] = None
        with patch(
            "experiments.query_factor_oracle.preflight.runtime_package_versions",
            return_value=installed,
        ):
            result = audit_runtime_dependencies()
        self.assertEqual(result["missing"], [])
        self.assertIn("metadata_only", result["check_mode"])
        self.assertEqual(result["versions"]["torch"], "1.0-test")

        missing = dict(installed)
        missing["datasets"] = None
        with patch(
            "experiments.query_factor_oracle.preflight.runtime_package_versions",
            return_value=missing,
        ):
            with self.assertRaisesRegex(RuntimeError, "datasets"):
                audit_runtime_dependencies()


if __name__ == "__main__":
    unittest.main()
