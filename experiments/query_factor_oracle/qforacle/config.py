from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expand(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, root) for item in value]
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    # Only path-like values are resolved.  Hugging Face repo IDs and model IDs
    # intentionally remain unchanged.
    looks_like_path = (
        expanded.startswith((".", "..", "/", "\\"))
        or "${" in value
        or "%" in value
        or expanded.endswith((".pth", ".json", ".jsonl", ".gz", ".yaml", ".yml"))
    )
    path = Path(expanded)
    if looks_like_path and not path.is_absolute():
        return str((root / path).resolve())
    return expanded


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    root = repo_root()
    config = _expand(config, root)
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(root)
    return config


def require_keys(config: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")

