"""Command-line entry point for creating figures."""

from __future__ import annotations

import argparse

from network_oracle.figures.adaptive_feedback import figure as adaptive_feedback
from network_oracle.figures.baseline_reliability import figure as baseline
from network_oracle.figures.biased_trust import figure as biased_trust
from network_oracle.figures.fixed_testimony import figure as fixed_testimony
from network_oracle.figures.style import setup_style

FIGURES = {
    "baseline": baseline,
    "fixed-testimony": fixed_testimony,
    "adaptive-feedback": adaptive_feedback,
    "biased-trust": biased_trust,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures.")
    parser.add_argument(
        "--run",
        dest="run_name",
        default="all",
        metavar="{baseline,fixed-testimony,adaptive-feedback,biased-trust,all}",
        help="result block to plot",
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    setup_style()

    if args.run_name == "all":
        runs = list(FIGURES)
    elif args.run_name in FIGURES:
        runs = [args.run_name]
    else:
        choices = ", ".join([*FIGURES, "all"])
        parser.error(f"unknown figure block '{args.run_name}' (choose from {choices})")

    for run_name in runs:
        print(run_name.replace("-", " ").title())
        FIGURES[run_name](args.results, args.out)


if __name__ == "__main__":
    main()
