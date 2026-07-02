"""Small diagnostic runs for the core simulation engine."""

from __future__ import annotations

import numpy as np

from network_oracle.model import Params, run_simulation
from network_oracle.monte_carlo import summarize_runs


def main() -> None:
    """Run a compact set of model sanity checks and print summary metrics."""
    fast = dict(n_agents=10, epsilon=0.05, n_pulls=1, max_rounds=800)
    seeds = 300
    jobs = 1

    print("=== Zollman effect: reliability by topology (no oracle) ===")
    for topology in ("complete", "cycle", "wheel"):
        output = summarize_runs(Params(topology=topology, **fast), seeds=seeds, jobs=jobs)
        print(
            f"  {topology:9s} rel={output['reliability']:.3f}+/-{output['se']:.3f} "
            f"rounds={output['mean_rounds']:5.0f} wrong={output['frac_wrong']:.2f} "
            f"none={output['frac_none']:.2f}"
        )

    print("\n=== frozen oracle on COMPLETE graph (tau=1.0, full adoption) ===")
    base_c = summarize_runs(Params(topology="complete", **fast), seeds=seeds, jobs=jobs)[
        "reliability"
    ]
    print(f"  baseline (no oracle) rel={base_c:.3f}")
    for reliability in (0.2, 0.4, 0.6, 0.7, 0.9):
        output = summarize_runs(
            Params(
                topology="complete",
                oracle_kind="frozen",
                oracle_r=reliability,
                oracle_tau=1.0,
                **fast,
            ),
            seeds=seeds,
            jobs=jobs,
        )
        print(
            f"  frozen r={reliability:.2f} rel={output['reliability']:.3f} "
            f"wrong={output['frac_wrong']:.2f}"
        )

    print("\n=== shared vs independent at r=0.6 (the correlation penalty) ===")
    for shared in (True, False):
        output = summarize_runs(
            Params(
                topology="complete",
                oracle_kind="frozen",
                oracle_r=0.6,
                oracle_shared=shared,
                oracle_tau=1.0,
                **fast,
            ),
            seeds=seeds,
            jobs=jobs,
        )
        print(
            f"  shared={str(shared):5s} rel={output['reliability']:.3f} "
            f"wrong={output['frac_wrong']:.2f}"
        )

    print("\n=== does a SHARED oracle erase the cycle's diversity advantage? ===")
    base_cy = summarize_runs(Params(topology="cycle", **fast), seeds=seeds, jobs=jobs)[
        "reliability"
    ]
    orc_cy = summarize_runs(
        Params(
            topology="cycle",
            oracle_kind="frozen",
            oracle_r=0.6,
            oracle_tau=1.0,
            **fast,
        ),
        seeds=seeds,
        jobs=jobs,
    )["reliability"]
    print(f"  cycle no-oracle rel={base_cy:.3f}   cycle + shared r=0.6 rel={orc_cy:.3f}")

    print("\n=== live oracle aggregate + single-run history ===")
    output = summarize_runs(
        Params(topology="complete", oracle_kind="live", oracle_tau=1.0, **fast),
        seeds=seeds,
        jobs=jobs,
    )
    print(
        f"  live oracle rel={output['reliability']:.3f} wrong={output['frac_wrong']:.2f} "
        f"none={output['frac_none']:.2f}"
    )
    history = run_simulation(
        Params(
            topology="complete",
            oracle_kind="live",
            oracle_tau=1.0,
            no_early_stop=True,
            **fast,
        ),
        seed=3,
        record_history=True,
    )["history"]
    entropy = np.array(history["entropy"])
    print(
        f"  one run: entropy start={entropy[:5].mean():.3f} "
        f"end={entropy[-20:].mean():.3f}; "
        f"oracle value end={history['oracle_value'][-1]:.3f}; "
        f"frac_B end={history['frac_B'][-1]:.2f}"
    )


if __name__ == "__main__":
    main()
