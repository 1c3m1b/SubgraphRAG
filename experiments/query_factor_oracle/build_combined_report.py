from __future__ import annotations

import argparse

from .qforacle.combined_report import build_combined_report


def _report(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must be DATASET=PATH")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine confirmed WebQSP and CWQ Phase 0–2 reports")
    parser.add_argument("--report", action="append", type=_report, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    reports = dict(args.report)
    if len(reports) != len(args.report):
        raise ValueError("Duplicate dataset labels in --report arguments")
    result = build_combined_report(reports, args.output_dir)
    print(f"Combined report written to {args.output_dir}/combined_report.md")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
