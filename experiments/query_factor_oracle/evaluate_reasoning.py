from __future__ import annotations

import argparse

from .qforacle.config import load_config
from .qforacle.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one isolated reasoning run with SubgraphRAG's answer matcher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--factors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = evaluate_predictions(
        repo_root=config["_repo_root"],
        prediction_path=args.predictions,
        retrieval_path=args.retrieval,
        output_dir=args.output_dir,
        factors_path=args.factors,
        allow_incomplete=args.allow_incomplete,
        expected_baseline=config.get("expected_baseline", {}).get("qa"),
        expected_qa_path=config.get("data", {}).get("rog_prediction_file"),
        allow_retrieval_only_samples=bool(
            config.get("data", {}).get("allow_retrieval_only_samples", False)
        ),
    )
    print(
        f"Evaluation complete: Hit={summary['full']['hit']:.3f}, "
        f"Hit@1={summary['full']['hit_at_1']:.3f}, Macro-F1={summary['full']['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
