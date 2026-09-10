from __future__ import annotations

import argparse

from .qforacle.config import load_config
from .qforacle.reasoning import run_reasoning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unchanged SubgraphRAG reasoning pipeline in an isolated output directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = run_reasoning(
        load_config(args.config),
        args.retrieval,
        args.output_dir,
        args.variant,
        args.replicate,
    )
    print(f"Reasoning complete: {args.output_dir}/predictions.jsonl")
    print(f"Run fingerprint: {manifest['run_fingerprint']}")


if __name__ == "__main__":
    main()
