from __future__ import annotations

import argparse

from network_oracle.experiments.adaptive_feedback import run as adaptive_feedback
from network_oracle.experiments.baseline_reliability import run as baseline
from network_oracle.experiments.biased_trust import run as biased_trust
from network_oracle.experiments.fixed_testimony import run as fixed_testimony

RUNS = {
    "baseline": baseline,
    "fixed-testimony": fixed_testimony,
    "adaptive-feedback": adaptive_feedback,
    "biased-trust": biased_trust,
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run shared-oracle network-epistemology simulations."
    )
    parser.add_argument(
        "--run",
        dest="run_name",
        default="all",
        metavar="{baseline,fixed-testimony,adaptive-feedback,biased-trust,all}",
        help="simulation block to run",
    )
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--seeds",
        type=int,
        default=300,
        help="independent communities per parameter cell",
    )
    parser.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    parser.add_argument("--quick", action="store_true", help="tiny grids/seeds for a fast smoke test")
    args = parser.parse_args()
    if args.quick and args.seeds == 300:
        args.seeds = 60

    if args.run_name == "all":
        todo = list(RUNS)
    elif args.run_name in RUNS:
        todo = [args.run_name]
    else:
        choices = ", ".join([*RUNS, "all"])
        parser.error(f"unknown run '{args.run_name}' (choose from {choices})")

    for run_name in todo:
        RUNS[run_name](args.out, args.seeds, args.jobs, quick=args.quick)
        print()


if __name__ == "__main__":
    main()
