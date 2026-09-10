from __future__ import annotations

import argparse

from .qforacle.config import load_config
from .qforacle.oracle import VARIANTS
from .qforacle.phase2 import run_oracle_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a frozen SubgraphRAG candidate pool with gold query factors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", help="Override config retrieval_result")
    parser.add_argument("--factors", required=True)
    parser.add_argument("--gpt-triples", help="Override optional GPT-labelled retrieval targets")
    parser.add_argument("--qa-cohort", help="Override RoG JSONL defining the fixed final-QA cohort")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, action="append", default=[],
                        help="Evaluation budget; repeatable (defaults to 50,100,200,500)")
    parser.add_argument("--greedy-budget", type=int,
                        help="Dynamic novelty cap; defaults to the fixed prompt budget")
    parser.add_argument("--diagnostics-top-k", type=int,
                        help="Candidate annotations retained per sample (default: config or 100)")
    parser.add_argument(
        "--match-state-limit", type=int,
        help="Deterministic query-graph backtracking cap per sample/budget",
    )
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    budgets = args.budget or config.get("phase2", {}).get("budgets", [50, 100, 200, 500])
    summary = run_oracle_experiment(
        args.baseline or config["retrieval_result"],
        args.factors,
        args.output_dir,
        budgets,
        args.greedy_budget or int(config.get("protocol", {}).get("prompt_top_k", 100)),
        args.variant or VARIANTS,
        args.gpt_triples or config.get("data", {}).get("gpt_triples_file"),
        args.qa_cohort or config.get("data", {}).get("rog_prediction_file"),
        args.diagnostics_top_k or int(config.get("phase2", {}).get("diagnostics_top_k", 100)),
        args.match_state_limit or int(
            config.get("phase2", {}).get("match_state_limit", 100000)
        ),
    )
    print(f"Phase 2 retrieval complete: {args.output_dir}/retrieval_summary.json")
    print(f"Variants: {', '.join(summary['variants'])}")


if __name__ == "__main__":
    main()
