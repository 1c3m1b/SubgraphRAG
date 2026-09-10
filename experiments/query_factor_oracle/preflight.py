from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from .qforacle.config import load_config
from .qforacle.io_utils import (
    git_revision,
    runtime_package_versions,
    sha256_json,
    write_json,
)


REQUIRED_RUNTIME_PACKAGES = (
    "torch",
    "datasets",
    "transformers",
    "huggingface_hub",
    "vllm",
    "openai",
    "numpy",
    "tqdm",
)


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    if "prompt_top_k" not in protocol or "prompt_mode" not in protocol:
        raise ValueError("protocol must explicitly set prompt_mode and prompt_top_k")
    prompt_top_k = protocol["prompt_top_k"]
    if isinstance(prompt_top_k, bool) or not isinstance(prompt_top_k, int) or prompt_top_k <= 0:
        raise ValueError("protocol.prompt_top_k must be a positive integer")
    prompt_mode = str(protocol["prompt_mode"])
    match = re.fullmatch(r"scored_([1-9][0-9]*)", prompt_mode)
    if not match:
        raise ValueError("protocol.prompt_mode must have the exact form scored_K")
    mode_top_k = int(match.group(1))
    if mode_top_k != prompt_top_k:
        raise ValueError(
            "protocol budget mismatch: "
            f"prompt_mode={prompt_mode!r}, prompt_top_k={prompt_top_k}"
        )
    phase2 = config.get("phase2", {})
    if not isinstance(phase2, dict) or "budgets" not in phase2:
        raise ValueError("phase2 must explicitly set budgets")
    budgets = phase2["budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("phase2.budgets must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in budgets):
        raise ValueError("phase2.budgets must contain only positive integers")
    if prompt_top_k not in budgets:
        raise ValueError(
            f"Fixed prompt budget {prompt_top_k} is absent from phase2.budgets={budgets}"
        )
    return {
        "prompt_mode": prompt_mode,
        "prompt_top_k": prompt_top_k,
        "phase2_budgets": budgets,
    }


def audit_runtime_dependencies() -> dict[str, Any]:
    versions = runtime_package_versions()
    missing = [package for package in REQUIRED_RUNTIME_PACKAGES if versions.get(package) is None]
    if missing:
        raise RuntimeError(
            "Missing packages required by the Phase 0-2 remote workflow: "
            + ", ".join(missing)
        )
    return {
        "check_mode": "importlib.metadata_only_no_import_or_network",
        "required": list(REQUIRED_RUNTIME_PACKAGES),
        "versions": versions,
        "missing": missing,
    }


def resolve_model_locally(model_name: str, revision: str) -> dict[str, Any]:
    """Fail before vLLM if the exact configured snapshot is unavailable."""
    if "gpt" in model_name.lower():
        return {"mode": "api_model_identifier", "model_name": model_name}
    local = Path(model_name).expanduser()
    if local.is_dir():
        resolved = local.resolve()
        parts = resolved.parts
        snapshot_commit = None
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                snapshot_commit = parts[index + 1]
        declared_revision = revision if not revision.startswith("RECORD_THE_REMOTE_") else ""
        if not snapshot_commit and not declared_revision:
            raise ValueError(
                "A generic local model directory requires an exact llm.resolved_revision"
            )
        if snapshot_commit and declared_revision and snapshot_commit != declared_revision:
            raise ValueError(
                "Local model snapshot and llm.resolved_revision disagree: "
                f"{snapshot_commit!r} != {declared_revision!r}"
            )
        if not (resolved / "config.json").is_file():
            raise FileNotFoundError(f"Local model directory lacks config.json: {resolved}")
        return {
            "mode": "local_directory",
            "resolved_path": str(resolved),
            "snapshot_commit": snapshot_commit,
        }
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the model cache audit") from exc
    try:
        resolved = Path(snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=True,
        )).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"Pinned model snapshot is not complete in the local HF cache: "
            f"{model_name}@{revision}"
        ) from exc
    return {
        "mode": "huggingface_cached_snapshot",
        "resolved_path": str(resolved),
        "snapshot_commit": revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast remote input/config audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=("webqsp", "cwq"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("dataset") != args.dataset:
        raise ValueError(f"Config dataset is {config.get('dataset')!r}, expected {args.dataset!r}")
    protocol = validate_protocol(config)
    paths = {
        "retrieval_result": config.get("retrieval_result"),
        "rog_prediction_file": config.get("data", {}).get("rog_prediction_file"),
        "gpt_triples_file": config.get("data", {}).get("gpt_triples_file"),
    }
    for index, value in enumerate(config.get("data", {}).get("logical_form_files", [])):
        paths[f"logical_form_files[{index}]"] = value
    missing = {
        name: str(value) for name, value in paths.items()
        if not value or not Path(str(value)).is_file()
    }
    if missing:
        raise FileNotFoundError(f"Missing remote input files: {missing}")
    for source in config.get("data", {}).get("dataset_sources", []):
        if not source.get("repo_id") or not source.get("revision"):
            raise ValueError(f"Unpinned dataset source: {source}")
    reasoning_source = config.get("data", {}).get("reasoning_dataset", {})
    if not reasoning_source.get("repo_id") or not reasoning_source.get("revision"):
        raise ValueError("data.reasoning_dataset must be pinned")
    model_name = str(config.get("llm", {}).get("model_name", ""))
    model_revision = str(config.get("llm", {}).get("resolved_revision", ""))
    if not Path(model_name).is_dir() and (
        not model_revision or model_revision.startswith("RECORD_THE_REMOTE_")
    ):
        raise ValueError("Set an exact llm.resolved_revision or a local model snapshot directory")
    dependencies = audit_runtime_dependencies()
    model_resolution = resolve_model_locally(model_name, model_revision)
    chatkbqa_root = os.environ.get("CHATKBQA_ROOT")
    chatkbqa_commit = git_revision(chatkbqa_root) if chatkbqa_root else None
    if not chatkbqa_root or not chatkbqa_commit:
        raise ValueError("CHATKBQA_ROOT must be a pinned Git checkout with a readable commit")
    result = {
        "status": "passed",
        "dataset": args.dataset,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_json({key: value for key, value in config.items() if not key.startswith("_")}),
        "inputs": {
            name: {"path": str(Path(str(value)).resolve()), "bytes": Path(str(value)).stat().st_size}
            for name, value in paths.items()
        },
        "repo_commit": git_revision(config["_repo_root"]),
        "chatkbqa_root": str(Path(chatkbqa_root).resolve()) if chatkbqa_root else None,
        "chatkbqa_commit": chatkbqa_commit,
        "model_name": model_name,
        "model_revision": model_revision,
        "model_resolution": model_resolution,
        "protocol": protocol,
        "dependencies": dependencies,
        "offline_environment": {
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        },
    }
    write_json(args.output, result)
    print(f"Preflight passed: {args.output}")


if __name__ == "__main__":
    main()
