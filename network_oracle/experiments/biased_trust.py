from __future__ import annotations

import os

import numpy as np
import pandas as pd

from network_oracle.config import BASE_PROBLEM
from network_oracle.model import Params
from network_oracle.monte_carlo import run_grid, run_histories
from network_oracle.experiments.io import save_results

def run(out, seeds, jobs, quick=False):
    print("Biased testimony and adaptive trust: trust, inversion, robustness")
    r_grid = (
        [0.2, 0.5, 0.8]
        if quick
        else [
            0.10,
            0.20,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.80,
            0.90,
        ]
    )
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
    save_results(df, out, "adaptive_trust_under_bias.csv")

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
    save_results(pd.DataFrame(rows), out, "correlation_penalty_robustness.csv")

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
    np.savez(os.path.join(out, "adaptive_trust_trajectories.npz"), **traj)
    print(f"  wrote {os.path.join(out, 'adaptive_trust_trajectories.npz')}")
