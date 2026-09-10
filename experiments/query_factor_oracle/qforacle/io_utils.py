from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


RUNTIME_PACKAGES = (
    "torch",
    "datasets",
    "transformers",
    "huggingface_hub",
    "vllm",
    "openai",
    "numpy",
    "tqdm",
    "networkx",
)


def runtime_package_versions() -> dict[str, str | None]:
    """Read installed distribution versions without importing runtime stacks."""
    versions: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@contextmanager
def open_text(path: str | os.PathLike[str], mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        if "w" in mode:
            # gzip.open embeds the wall-clock mtime.  Pin it so identical
            # factor/metric JSONL produces identical provenance hashes.
            with path.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="", fileobj=raw_handle, mode="wb", mtime=0
                ) as compressed_handle:
                    with io.TextIOWrapper(
                        compressed_handle, encoding="utf-8"
                    ) as text_handle:
                        yield text_handle
            return
        with gzip.open(path, mode, encoding="utf-8") as handle:
            yield handle
        return
    with path.open(mode, encoding="utf-8") as handle:
        yield handle


def read_json(path: str | os.PathLike[str]) -> Any:
    with open_text(path) as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            yield item


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def retrieval_semantic_sha256(retrieval: dict[Any, dict[str, Any]]) -> str:
    """Hash retrieval meaning, independent of torch.save container bytes."""
    digest = hashlib.sha256()
    for sample_id, sample in retrieval.items():
        scored = sample.get("scored_triples", sample.get("scored_triplets", [])) or []
        triples = []
        for value in scored:
            triple = [str(value[0]), str(value[1]), str(value[2])]
            if len(value) >= 4:
                triple.append(float(value[3]))
            triples.append(triple)
        payload = {
            "id": str(sample_id),
            "question": str(sample.get("question", "")),
            "q_entity": [str(value) for value in sample.get("q_entity", [])],
            "q_entity_in_graph": [str(value) for value in sample.get("q_entity_in_graph", [])],
            "a_entity": [str(value) for value in sample.get("a_entity", [])],
            "a_entity_in_graph": [str(value) for value in sample.get("a_entity_in_graph", [])],
            "max_path_length": sample.get("max_path_length"),
            "target_relevant_triples": [
                [str(value[0]), str(value[1]), str(value[2])]
                for value in sample.get("target_relevant_triples", [])
            ],
            "scored_triples": triples,
        }
        digest.update(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_revision(repo_root: str | os.PathLike[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_status(repo_root: str | os.PathLike[str]) -> list[str] | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_torch(path: str | os.PathLike[str]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read SubgraphRAG .pth files") from exc

    # weights_only became True by default in newer PyTorch releases.  Retrieval
    # result dictionaries contain arbitrary Python objects, so opt out explicitly.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_torch(value: Any, path: str | os.PathLike[str]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to write SubgraphRAG .pth files") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def normalise_question(text: str) -> str:
    return " ".join(str(text).casefold().split()).rstrip("?")


def prediction_answers(text: str) -> list[str]:
    answers: list[str] = []
    for line in str(text).splitlines():
        lower = line.casefold()
        if "ans:" not in lower:
            continue
        value = line[lower.index("ans:") + 4 :].strip()
        if value and value.casefold() not in {"not available", "no information available"}:
            answers.append(normalise_question(value))
    return list(dict.fromkeys(answers))
