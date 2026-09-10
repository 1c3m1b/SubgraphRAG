from __future__ import annotations

import argparse

from .qforacle.config import load_config
from .qforacle.factors import build_factors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical gold query factors for a frozen baseline cohort")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", help="Override config retrieval_result")
    parser.add_argument("--logical-form", action="append", default=[],
                        help="Override config logical-form sources; repeat for multiple sources")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    logical_forms = args.logical_form or config.get("data", {}).get("logical_form_files", [])
    if not logical_forms:
        raise ValueError("No logical-form files configured; pass --logical-form PATH")
    summary = build_factors(
        args.baseline or config["retrieval_result"],
        logical_forms,
        args.output_dir,
        config["dataset"],
        config["split"],
    )
    print(
        f"Phase 1 complete: {summary['parseable_count']}/{summary['cohort_sample_count']} "
        f"samples parseable; summary at {args.output_dir}"
    )


if __name__ == "__main__":
    main()
