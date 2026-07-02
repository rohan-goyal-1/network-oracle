"""Command-line entry point for creating figures."""

from __future__ import annotations

import argparse

from network_oracle.figures.style import setup_style
from network_oracle.figures.study0 import fig_study0
from network_oracle.figures.study1 import fig_study1
from network_oracle.figures.study2 import fig_study2
from network_oracle.figures.study3 import fig_study3


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures.")
    parser.add_argument("--study", default="all", choices=["0", "1", "2", "3", "all"])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    setup_style()

    functions = {
        "0": fig_study0,
        "1": fig_study1,
        "2": fig_study2,
        "3": fig_study3,
    }
    studies = ["0", "1", "2", "3"] if args.study == "all" else [args.study]

    for study in studies:
        print(f"Study {study}")
        functions[study](args.results, args.out)


if __name__ == "__main__":
    main()
