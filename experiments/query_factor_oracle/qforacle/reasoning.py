from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import core_hashes, environment_snapshot
from .io_utils import iter_jsonl, sha256_file, sha256_json, write_json, write_jsonl


def _load_core(repo: Path):
    reason_dir = repo / "reason"
    if str(reason_dir) not in sys.path:
        sys.path.insert(0, str(reason_dir))
    # Lazy imports keep phase-0/1/2 retrieval tooling usable on a CPU laptop
    # without importing vLLM.
    from preprocess import prepare_data
    from preprocess.prepare_prompts import get_prompts_for_data
    import llm_utils
    import prompts
    return prepare_data, get_prompts_for_data, llm_utils, prompts


def _load_pinned_reasoning_data(
    prepare_data: Any,
    config: dict[str, Any],
    rog_predictions: Path,
    retrieval: Path,
    prompt_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call the upstream loader while pinning only its dataset entry point."""
    dataset_spec = config.get("data", {}).get("reasoning_dataset")
    if not dataset_spec:
        return prepare_data.get_data(
            config["dataset"], str(rog_predictions), str(retrieval),
            config["split"], prompt_mode,
        ), {"pin_enforced": False, "reason": "reasoning_dataset is not configured"}
    repo_id = str(dataset_spec.get("repo_id", "")).strip()
    revision = str(dataset_spec.get("revision", "")).strip()
    if not repo_id or not revision:
        raise ValueError("data.reasoning_dataset must contain both repo_id and revision")
    if not hasattr(prepare_data, "get_subgraphs") or not hasattr(prepare_data, "load_dataset"):
        raise RuntimeError("Cannot pin the upstream reasoning dataset loader")

    original_get_subgraphs = prepare_data.get_subgraphs
    runtime: dict[str, Any] = {
        "pin_enforced": True,
        "repo_id": repo_id,
        "requested_revision": revision,
        "requested_split": config["split"],
    }

    def pinned_get_subgraphs(dataset_name: str, split: str) -> Any:
        if dataset_name != config["dataset"] or split != config["split"]:
            raise ValueError(
                "Upstream reasoning loader requested an unexpected dataset/split: "
                f"{dataset_name}/{split}"
            )
        dataset = prepare_data.load_dataset(repo_id, split=split, revision=revision)
        runtime["dataset_fingerprint"] = getattr(dataset, "_fingerprint", None)
        runtime["dataset_sample_count"] = len(dataset)
        return dataset

    prepare_data.get_subgraphs = pinned_get_subgraphs
    try:
        data = prepare_data.get_data(
            config["dataset"], str(rog_predictions), str(retrieval),
            config["split"], prompt_mode,
        )
    finally:
        prepare_data.get_subgraphs = original_get_subgraphs
    return data, runtime


def _model_metadata(path: Path) -> dict[str, Any]:
    metadata_files = (
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "special_tokens_map.json", "model.safetensors.index.json",
    )
    hashes = {
        name: sha256_file(path / name)
        for name in metadata_files
        if (path / name).is_file()
    }
    weight_files = sorted(
        (
            child.name,
            child.stat().st_size,
            child.resolve().name if child.is_symlink() else None,
        )
        for child in path.iterdir()
        if child.is_file() and child.suffix in {".safetensors", ".bin"}
    )
    parts = path.resolve().parts
    snapshot_commit = None
    if "snapshots" in parts:
        snapshot_index = parts.index("snapshots")
        if snapshot_index + 1 < len(parts):
            snapshot_commit = parts[snapshot_index + 1]
    return {
        "resolved_path": str(path.resolve()),
        "resolved_snapshot_commit": snapshot_commit,
        "metadata_sha256": hashes,
        "weight_artifacts_sha256": sha256_json(weight_files),
        "weight_file_count": len(weight_files),
    }


def _resolve_model_for_runtime(model_name: str, llm_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve an HF model revision to a local immutable snapshot before vLLM."""
    if "gpt" in model_name.lower():
        return model_name, {
            "mode": "api_model_identifier",
            "model_name": model_name,
            "note": "API providers do not expose a local weight snapshot to this runner.",
        }
    local_model = Path(model_name).expanduser()
    if local_model.is_dir():
        metadata = _model_metadata(local_model)
        declared_revision = str(llm_config.get("resolved_revision", "")).strip()
        if (
            not metadata.get("resolved_snapshot_commit")
            and (not declared_revision or declared_revision.startswith("RECORD_THE_REMOTE_"))
        ):
            raise ValueError(
                "A generic local model directory requires llm.resolved_revision; "
                "alternatively point model_name at an HF snapshots/<commit> directory."
            )
        snapshot_commit = metadata.get("resolved_snapshot_commit")
        if (
            snapshot_commit
            and declared_revision
            and not declared_revision.startswith("RECORD_THE_REMOTE_")
            and declared_revision != snapshot_commit
        ):
            raise ValueError(
                "Local model snapshot and llm.resolved_revision disagree: "
                f"{snapshot_commit!r} != {declared_revision!r}"
            )
        return str(local_model.resolve()), {
            "mode": "local_directory",
            "model_name": model_name,
            "declared_revision": declared_revision or None,
            **metadata,
        }

    revision = str(llm_config.get("resolved_revision", "")).strip()
    if not revision or revision.startswith("RECORD_THE_REMOTE_"):
        raise ValueError(
            "Remote Hugging Face model IDs require an exact llm.resolved_revision. "
            "Set it to a snapshot commit, or set llm.model_name to a local snapshot directory."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to resolve the pinned model snapshot") from exc
    snapshot_path = Path(snapshot_download(repo_id=model_name, revision=revision)).resolve()
    return str(snapshot_path), {
        "mode": "huggingface_snapshot",
        "model_name": model_name,
        "requested_revision": revision,
        **_model_metadata(snapshot_path),
    }


def _initialize_core_llm(
    llm_utils: Any,
    configured_model_name: str,
    runtime_model_name: str,
    llm_config: dict[str, Any],
) -> Any:
    arguments = (
        int(llm_config.get("tensor_parallel_size", 1)),
        int(llm_config.get("max_seq_len_to_capture", 16384)),
        int(llm_config.get("max_tokens", 4000)),
        int(llm_config.get("seed", 0)),
        float(llm_config.get("temperature", 0.0)),
        float(llm_config.get("frequency_penalty", 0.0)),
    )
    if "gpt" in configured_model_name.lower():
        return llm_utils.llm_init(configured_model_name, *arguments)
    # Let the original llm_init choose its local-vLLM branch using the original
    # model identifier, but replace only the model passed into vLLM with the
    # already resolved snapshot.  This avoids path substrings changing dispatch.
    if hasattr(llm_utils, "LLM"):
        original_llm_class = llm_utils.LLM

        def pinned_llm_class(*args: Any, **kwargs: Any) -> Any:
            if args:
                args = (runtime_model_name, *args[1:])
            else:
                kwargs["model"] = runtime_model_name
            return original_llm_class(*args, **kwargs)

        llm_utils.LLM = pinned_llm_class
        try:
            return llm_utils.llm_init(configured_model_name, *arguments)
        finally:
            llm_utils.LLM = original_llm_class
    # Synthetic adapters used by CPU tests do not expose the module-level LLM.
    return llm_utils.llm_init(runtime_model_name, *arguments)


def _defined_prompts(prompt_mode: str, model_name: str, llm_mode: str, prompts: Any) -> tuple[str, str]:
    # This is the same dispatch table as reason/main.py:get_defined_prompts,
    # but avoids importing main.py (and therefore wandb) just to select strings.
    if "gpt" in model_name or "gpt" in prompt_mode:
        if "gptLabel" in prompt_mode:
            return prompts.sys_prompt_gpt, prompts.cot_prompt_gpt
        return prompts.icl_sys_prompt, prompts.icl_cot_prompt
    if "noevi" in prompt_mode:
        return prompts.noevi_sys_prompt, prompts.noevi_cot_prompt
    if "icl" in llm_mode:
        return prompts.icl_sys_prompt, prompts.icl_cot_prompt
    return prompts.sys_prompt, prompts.cot_prompt


class TracingLLM:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[0] if args else None)
        self.calls.append({
            "conversation_sha256": sha256_json(messages),
            "message_count": len(messages) if isinstance(messages, list) else None,
        })
        return self.inner(*args, **kwargs)


def _resume_rows(path: Path, data: list[dict[str, Any]], fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = list(iter_jsonl(path))
    if len(rows) > len(data):
        raise ValueError("Checkpoint contains more rows than current input")
    seen = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", ""))
        if sample_id in seen:
            raise ValueError(f"Duplicate checkpoint ID: {sample_id}")
        seen.add(sample_id)
        expected_id = str(data[index].get("id", ""))
        if sample_id != expected_id:
            raise ValueError(
                f"Checkpoint prefix mismatch at row {index}: found {sample_id!r}, expected {expected_id!r}"
            )
        actual_fingerprint = row.get("_qforacle", {}).get("run_fingerprint")
        if actual_fingerprint != fingerprint:
            raise ValueError(
                f"Checkpoint input/config fingerprint mismatch at {sample_id}: "
                f"{actual_fingerprint!r} != {fingerprint!r}"
            )
    return rows


def run_reasoning(
    config: dict[str, Any],
    retrieval_path: str,
    output_dir: str,
    variant: str,
    replicate: int,
) -> dict[str, Any]:
    repo = Path(config["_repo_root"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_config = config.get("data", {})
    llm_config = config.get("llm", {})
    protocol = config.get("protocol", {})
    rog_predictions = Path(data_config["rog_prediction_file"])
    retrieval = Path(retrieval_path).resolve()
    prompt_mode = str(protocol.get("prompt_mode", f"scored_{protocol.get('prompt_top_k', 100)}"))
    if "gt" in prompt_mode:
        raise ValueError("Gold-triple injection prompt modes are forbidden in the oracle comparison")
    if not prompt_mode.startswith("scored_"):
        raise ValueError("Oracle reasoning must use a scored_K prompt mode")
    if float(protocol.get("threshold", 0.0)) != 0.0:
        raise ValueError("Phase 2 fixes threshold=0 so preserved baseline scores cannot change membership")
    model_name = str(llm_config.get("model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    llm_mode = str(llm_config.get("llm_mode", "sys_icl_dc_repro"))
    prepare_data, get_prompts_for_data, llm_utils, prompts = _load_core(repo)
    data, runtime_dataset = _load_pinned_reasoning_data(
        prepare_data, config, rog_predictions, retrieval, prompt_mode,
    )
    sys_prompt, cot_prompt = _defined_prompts(prompt_mode, model_name, llm_mode, prompts)
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, 0.0)
    prompt_rows = []
    prompt_budget = int(prompt_mode.split("_")[1])
    for item in data:
        evidence = []
        seen = set()
        for value in item.get("scored_triplets", []):
            triple = tuple(map(str, value[:3]))
            if triple not in seen:
                seen.add(triple)
                evidence.append(list(triple))
            if len(evidence) >= prompt_budget:
                break
        prompt_rows.append({
            "id": str(item["id"]),
            "question": item.get("question"),
            "sys_query": item.get("sys_query"),
            "user_query": item.get("user_query"),
            "cot_query": item.get("cot_query"),
            "effective_prompt_k": len(evidence),
            "evidence_triples": evidence,
            "prompt_sha256": sha256_json({
                "sys_query": item.get("sys_query"),
                "user_query": item.get("user_query"),
                "cot_query": item.get("cot_query"),
            }),
        })
    data_ids = [str(item["id"]) for item in data]
    if len(data_ids) != len(set(data_ids)):
        raise ValueError("Reasoning data contains duplicate IDs")
    prompt_collection_sha256 = sha256_json(
        [(row["id"], row["prompt_sha256"]) for row in prompt_rows]
    )
    question_collection_sha256 = sha256_json(
        [(row["id"], row["question"]) for row in prompt_rows]
    )
    prompt_template_sha256 = sha256_json({
        "sys_prompt": sys_prompt,
        "cot_prompt": cot_prompt,
        "prompt_mode": prompt_mode,
        "llm_mode": llm_mode,
    })
    runtime_model_name, runtime_model = _resolve_model_for_runtime(model_name, llm_config)
    fixed_reasoning_protocol = {
        "dataset": config["dataset"],
        "split": config["split"],
        "rog_predictions_sha256": sha256_file(rog_predictions),
        "reasoning_dataset": runtime_dataset,
        "sample_id_order_sha256": sha256_json(data_ids),
        "question_collection_sha256": question_collection_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "core_file_sha256": core_hashes(repo),
        "runner_sha256": sha256_file(__file__),
        "prompt_mode": prompt_mode,
        "threshold": 0.0,
        "model": llm_config,
        "runtime_model": runtime_model,
    }
    fixed_reasoning_protocol_sha256 = sha256_json(fixed_reasoning_protocol)
    fixed_inputs = {
        **fixed_reasoning_protocol,
        "variant": variant,
        "replicate": replicate,
        "retrieval_sha256": sha256_file(retrieval),
        "prompt_collection_sha256": prompt_collection_sha256,
    }
    fingerprint = sha256_json(fixed_inputs)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": fingerprint,
        "fixed_reasoning_protocol": fixed_reasoning_protocol,
        "fixed_reasoning_protocol_sha256": fixed_reasoning_protocol_sha256,
        "fixed_inputs": fixed_inputs,
        "sample_count": len(data),
        "sample_id_order_sha256": sha256_json(data_ids),
        "runtime_dataset": runtime_dataset,
        "runtime_model": runtime_model,
        "prompt_collection_sha256": prompt_collection_sha256,
        "question_collection_sha256": question_collection_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "core_reuse": [
            "reason.preprocess.prepare_data.get_data",
            "reason.preprocess.prepare_prompts.get_prompts_for_data",
            "reason.llm_utils.llm_init",
            "reason.llm_utils.llm_inf_all",
            "reason.prompts",
        ],
        "seed_note": (
            "For local vLLM, the upstream llm_init signature accepts seed but does not pass it to "
            "LLM or SamplingParams; use replicate agreement as the stability check."
        ),
        "environment": environment_snapshot(),
    }
    prediction_path = output / "predictions.jsonl"
    existing = _resume_rows(prediction_path, data, fingerprint)
    # Validate an existing checkpoint before replacing either sidecar.  If the
    # invocation drifted (config, prompt, model or retrieval), the old
    # predictions must remain paired with their original prompt/manifest
    # artifacts so the failed resume is still auditable.
    write_jsonl(output / "prompts.jsonl.gz", prompt_rows)
    if len(existing) == len(data):
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["prediction_sha256"] = sha256_file(prediction_path)
        manifest["llm_call_count"] = sum(
            row.get("_qforacle", {}).get("llm_call_count", 0) for row in existing
        )
        manifest["double_check_sample_count"] = sum(
            bool(row.get("_qforacle", {}).get("double_check_triggered")) for row in existing
        )
        manifest["resume_note"] = "complete checkpoint verified; vLLM initialization skipped"
        write_json(output / "run_manifest.json", manifest)
        return manifest
    write_json(output / "run_manifest.json", manifest)
    llm = _initialize_core_llm(
        llm_utils, model_name, runtime_model_name, llm_config,
    )
    tracing_llm = TracingLLM(llm)
    with prediction_path.open("a", encoding="utf-8") as handle:
        for index in range(len(existing), len(data)):
            each_qa = data[index]
            tracing_llm.reset()
            # Keep dispatch semantics based on the configured model identifier;
            # vLLM receives the resolved local snapshot path above.
            result = llm_utils.llm_inf_all(tracing_llm, each_qa, llm_mode, model_name)
            record = deepcopy(each_qa)
            for key in ("graph", "good_paths_rog", "good_triplets_rog", "scored_triplets"):
                record.pop(key, None)
            record["prediction"] = result[0]
            record["_qforacle"] = {
                "variant": variant,
                "replicate": replicate,
                "run_fingerprint": fingerprint,
                "fixed_reasoning_protocol_sha256": fixed_reasoning_protocol_sha256,
                "sample_index": index,
                "llm_call_count": len(tracing_llm.calls),
                "calls": tracing_llm.calls,
                "double_check_triggered": len(tracing_llm.calls) > 1,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    completed = list(iter_jsonl(prediction_path))
    if len(completed) != len(data):
        raise AssertionError("Reasoning run ended without a complete prediction cohort")
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["prediction_sha256"] = sha256_file(prediction_path)
    manifest["llm_call_count"] = sum(row.get("_qforacle", {}).get("llm_call_count", 0) for row in completed)
    manifest["double_check_sample_count"] = sum(
        bool(row.get("_qforacle", {}).get("double_check_triggered")) for row in completed
    )
    write_json(output / "run_manifest.json", manifest)
    return manifest
