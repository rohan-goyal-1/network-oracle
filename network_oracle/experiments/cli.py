from __future__ import annotations

import argparse

from network_oracle.experiments.study0_baseline import study0
from network_oracle.experiments.study1_frozen_oracle import study1
from network_oracle.experiments.study2_live_oracle import study2
from network_oracle.experiments.study3_trust import study3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run shared-oracle network-epistemology studies."
    )
    parser.add_argument("--study", default="all", choices=["0", "1", "2", "3", "all"])
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

    studies = {"0": study0, "1": study1, "2": study2, "3": study3}
    todo = ["0", "1", "2", "3"] if args.study == "all" else [args.study]
    for study in todo:
        studies[study](args.out, args.seeds, args.jobs, quick=args.quick)
        print()


if __name__ == "__main__":
    main()
