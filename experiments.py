"""
experiments.py
==============
Runs the staged studies and writes results to disk (CSV grids + NPZ time series)
for `analysis.py` to plot.

Usage
-----
    python experiments.py --study 0 --out results --seeds 300 --jobs 4
    python experiments.py --study 1 --out results --seeds 300 --jobs 4
    python experiments.py --study 2 --out results --seeds 300 --jobs 4
    python experiments.py --study 3 --out results --seeds 300 --jobs 4
    python experiments.py --study all --quick           # fast smoke run

Studies
-------
  0  Replication & robustness map  : the Zollman effect across topology, problem
                                     difficulty, and community size (no oracle).
  1  Frozen oracle (core)          : reliability over (reliability r x adoption
                                     phi x topology); the shared-vs-independent
                                     correlation penalty; the "individually
                                     helpful but collectively harmful" band.
  2  Live oracle (feedback)        : reliability vs none/frozen/live, plus
                                     time series of exploration entropy and
                                     oracle confidence-vs-accuracy (monoculture
                                     collapse).
  3  Biased oracle & trust         : damage from a biased source, whether
                                     endogenous trust can discipline it, the
                                     network-structure inversion, and a
                                     robustness sweep of the correlation penalty.

Edit `config.BASE_PROBLEM` to change the base regime for every study.
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd

from config import BASE_PROBLEM
from model import Params
from runs import isolated_agent_reliability, run_grid, run_histories


def _save(df: pd.DataFrame, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


# --------------------------------------------------------------------------- #
#  Study 0 -- replication & robustness map
# --------------------------------------------------------------------------- #
def study0(out, seeds, jobs, quick=False):
    print("STUDY 0: Zollman effect across topology / difficulty / size")
    topos = ["complete", "cycle", "wheel", "er", "ws", "ba"]
    epsilons = [0.05] if quick else [0.02, 0.05, 0.10]
    sizes = [10] if quick else [10, 20]
    if quick:
        topos = ["complete", "cycle", "wheel"]
    cells = []
    for topo in topos:
        for eps in epsilons:
            for n in sizes:
                p = Params(**{**BASE_PROBLEM, "topology": topo, "epsilon": eps, "n_agents": n})
                cells.append(({"topology": topo, "epsilon": eps, "n_agents": n}, p))
    df = run_grid(cells, seeds, jobs)
    _save(df, out, "study0.csv")
    piv = df[df.n_agents == sizes[0]].pivot_table(index="topology",
                                                  columns="epsilon", values="reliability")
    print(piv.round(3).to_string())


# --------------------------------------------------------------------------- #
#  Study 1 -- frozen oracle: the core sweep
# --------------------------------------------------------------------------- #
def study1(out, seeds, jobs, quick=False):
    print("STUDY 1: frozen oracle -- reliability(r, phi, topology)")
    n_r = 6 if quick else 21
    r_grid = np.round(np.linspace(0.0, 1.0, n_r), 3)
    phis = [0.0, 0.5, 1.0] if quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    topos = ["complete", "cycle"]

    # main grid: r x phi x topology (shared oracle)
    cells = []
    for topo in topos:
        for phi in phis:
            for r_ in r_grid:
                kind = "none" if phi == 0.0 else "frozen"
                p = Params(**{**BASE_PROBLEM, "topology": topo, "oracle_kind": kind,
                              "oracle_r": float(r_), "oracle_tau": 1.0,
                              "oracle_shared": True, "adoption_fraction": float(phi)})
                cells.append(({"topology": topo, "adoption_fraction": phi,
                               "oracle_r": float(r_)}, p))
    df = run_grid(cells, seeds, jobs)
    _save(df, out, "study1_grid.csv")

    # shared vs independent at full adoption (the correlation penalty)
    cells = []
    for topo in topos:
        for shared in (True, False):
            for r_ in r_grid:
                p = Params(**{**BASE_PROBLEM, "topology": topo, "oracle_kind": "frozen",
                              "oracle_r": float(r_), "oracle_tau": 1.0,
                              "oracle_shared": shared, "adoption_fraction": 1.0})
                cells.append(({"topology": topo, "shared": shared,
                               "oracle_r": float(r_)}, p))
    df2 = run_grid(cells, seeds, jobs)
    _save(df2, out, "study1_sharedvsindep.csv")

    # lone-agent benchmark (to mark the individually-helpful threshold)
    solo_seeds = 1000 if quick else 4000
    solo = []
    for r_ in r_grid:
        p = Params(**{**BASE_PROBLEM, "oracle_kind": "frozen", "oracle_r": float(r_),
                      "oracle_tau": 1.0, "adoption_fraction": 1.0})
        solo.append({"oracle_r": float(r_),
                     "isolated_reliability": isolated_agent_reliability(p, seeds=solo_seeds)})
    _save(pd.DataFrame(solo), out, "study1_isolated.csv")


# --------------------------------------------------------------------------- #
#  Study 2 -- live oracle & feedback
# --------------------------------------------------------------------------- #
def study2(out, seeds, jobs, quick=False):
    print("STUDY 2: live oracle -- reliability comparison + collapse time series")
    phis = [0.5, 1.0] if quick else [0.25, 0.5, 0.75, 1.0]
    topos = ["complete", "cycle"]

    # reliability: none vs frozen(r=0.7) vs live
    cells = []
    for topo in topos:
        for phi in phis:
            specs = [("none", dict(oracle_kind="none")),
                     ("frozen0.7", dict(oracle_kind="frozen", oracle_r=0.7)),
                     ("live", dict(oracle_kind="live"))]
            for label, spec in specs:
                if label == "none" and phi != phis[0]:
                    continue
                p = Params(**{**BASE_PROBLEM, "topology": topo, "oracle_tau": 1.0,
                              "adoption_fraction": (1.0 if label == "none" else phi),
                              **spec})
                cells.append(({"topology": topo, "oracle": label,
                               "adoption_fraction": phi}, p))
    df = run_grid(cells, seeds, jobs)
    _save(df, out, "study2_reliability.csv")

    # time series: complete graph, live oracle, full adoption
    hist_seeds = 100 if quick else 300
    p = Params(**{**BASE_PROBLEM, "topology": "complete", "oracle_kind": "live",
                  "oracle_tau": 1.0, "adoption_fraction": 1.0})
    H, correct = run_histories(p, hist_seeds, jobs)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "study2_timeseries.npz"),
             correct=correct, **{k: H[k] for k in H})
    print(f"  wrote {os.path.join(out, 'study2_timeseries.npz')} "
          f"({hist_seeds} runs, {H['entropy'].shape[1]} rounds, "
          f"{correct.mean():.2f} correct)")


# --------------------------------------------------------------------------- #
#  Study 3 -- biased oracle, endogenous trust, robustness
# --------------------------------------------------------------------------- #
def study3(out, seeds, jobs, quick=False):
    print("STUDY 3: biased oracle, endogenous trust, network inversion, robustness")
    r_grid = [0.2, 0.35, 0.5, 0.65, 0.8] if not quick else [0.2, 0.5, 0.8]
    topos = ["complete", "cycle"]

    # endogenous trust on/off, across r and topology
    cells = []
    for topo in topos:
        for endo in (False, True):
            for r_ in r_grid:
                p = Params(**{**BASE_PROBLEM, "topology": topo, "oracle_kind": "frozen",
                              "oracle_r": float(r_), "oracle_tau": 2.0,
                              "adoption_fraction": 0.7,
                              "endogenous_trust": endo, "trust_learning_rate": 0.15})
                cells.append(({"topology": topo, "endogenous_trust": endo,
                               "oracle_r": float(r_)}, p))
    df = run_grid(cells, seeds, jobs)
    _save(df, out, "study3_endogenous.csv")

    # robustness sweep: shared-minus-independent correlation penalty across
    # difficulty, size, and trust (which parameters keep the effect alive)
    eps_grid = [0.05] if quick else [0.03, 0.05, 0.08]
    n_grid = [10] if quick else [10, 20]
    tau_grid = [1.0] if quick else [0.5, 1.0, 2.0]
    rows = []
    for eps in eps_grid:
        for n in n_grid:
            for tau in tau_grid:
                cells = []
                for shared in (True, False):
                    p = Params(**{**BASE_PROBLEM, "epsilon": eps, "n_agents": n,
                                  "topology": "complete", "oracle_kind": "frozen",
                                  "oracle_r": 0.6, "oracle_tau": tau,
                                  "oracle_shared": shared, "adoption_fraction": 1.0})
                    cells.append(({"shared": shared}, p))
                d = run_grid(cells, seeds, jobs)
                rel_s = float(d[d.shared].reliability.iloc[0])
                rel_i = float(d[~d.shared].reliability.iloc[0])
                rows.append({"epsilon": eps, "n_agents": n, "oracle_tau": tau,
                             "rel_shared": rel_s, "rel_independent": rel_i,
                             "penalty": rel_i - rel_s})
    _save(pd.DataFrame(rows), out, "study3_robustness.csv")

    # trust trajectories: biased vs reliable oracle, endogenous trust on
    hist_seeds = 80 if quick else 200
    traj = {}
    for label, r_ in [("biased_r0.3", 0.3), ("reliable_r0.8", 0.8)]:
        p = Params(**{**BASE_PROBLEM, "topology": "complete", "oracle_kind": "frozen",
                      "oracle_r": r_, "oracle_tau": 2.0, "adoption_fraction": 1.0,
                      "endogenous_trust": True, "trust_learning_rate": 0.15})
        H, _ = run_histories(p, hist_seeds, jobs)
        traj[label] = H["mean_trust"].mean(axis=0)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "study3_trust_traj.npz"), **traj)
    print(f"  wrote {os.path.join(out, 'study3_trust_traj.npz')}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run shared-oracle network-epistemology studies.")
    ap.add_argument("--study", default="all",
                    choices=["0", "1", "2", "3", "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=300,
                    help="independent communities per parameter cell")
    ap.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grids/seeds for a fast smoke test")
    args = ap.parse_args()
    if args.quick and args.seeds == 300:
        args.seeds = 60

    studies = {"0": study0, "1": study1, "2": study2, "3": study3}
    todo = ["0", "1", "2", "3"] if args.study == "all" else [args.study]
    for s in todo:
        studies[s](args.out, args.seeds, args.jobs, quick=args.quick)
        print()


if __name__ == "__main__":
    main()
