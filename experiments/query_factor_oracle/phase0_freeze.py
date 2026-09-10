from __future__ import annotations

import argparse
from pathlib import Path

from .qforacle.baseline import freeze_baseline
from .qforacle.config import load_config


def _prediction(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("prediction must be LABEL=PATH")
    return label, str(Path(path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze a SubgraphRAG Phase-0 baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction", action="append", default=[], type=_prediction,
                        help="Repeatable reasoning run in LABEL=PATH form")
    parser.add_argument("--qa-summary", action="append", default=[], type=_prediction,
                        help="Repeatable evaluated baseline summary in LABEL=PATH form")
    parser.add_argument("--retrieval-replicate", action="append", default=[], type=_prediction,
                        help="Repeatable independent retrieval rerun in LABEL=PATH form")
    parser.add_argument("--retriever-checkpoint",
                        help="cpt.pth used by retrieve/inference.py; config and SHA are audited")
    parser.add_argument("--audit-datasets", action="store_true",
                        help="Load configured HF graph datasets and verify alignment/membership")
    parser.add_argument("--verify-lock")
    parser.add_argument("--no-copy-retrieval", action="store_true")
    parser.add_argument("--require-confirmed-baseline", action="store_true",
                        help="Fail unless metrics, dataset audit, retrieval rerun and LLM rerun all pass")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = freeze_baseline(
        config,
        args.output_dir,
        predictions=args.prediction,
        qa_summaries=args.qa_summary,
        retrieval_replicates=args.retrieval_replicate,
        retriever_checkpoint=args.retriever_checkpoint,
        audit_datasets=args.audit_datasets,
        copy_retrieval=not args.no_copy_retrieval,
        verify_lock_path=args.verify_lock,
        require_confirmed=args.require_confirmed_baseline,
    )
    print(f"Phase 0 baseline frozen: {manifest['lock_file']}")


if __name__ == "__main__":
    main()
