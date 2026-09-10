from __future__ import annotations

import argparse

from .qforacle.report import build_report


def _qa(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("QA summary must be VARIANT=PATH")
    variant, path = value.split("=", 1)
    return variant, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 0–2 oracle headroom report")
    parser.add_argument("--retrieval-summary", required=True)
    parser.add_argument("--phase0-validation")
    parser.add_argument("--qa-summary", type=_qa, action="append", default=[])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    qa_summaries = dict(args.qa_summary)
    if len(qa_summaries) != len(args.qa_summary):
        raise ValueError("Duplicate variant labels in --qa-summary arguments")
    report = build_report(
        args.retrieval_summary,
        qa_summaries,
        args.output_dir,
        args.budget,
        args.phase0_validation,
    )
    print(f"Report written to {args.output_dir}/final_report.md")
    print(f"Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
