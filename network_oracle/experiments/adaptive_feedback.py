from __future__ import annotations

import os

import numpy as np

from network_oracle.config import BASE_PROBLEM
from network_oracle.model import Params
from network_oracle.monte_carlo import run_grid, run_histories
from network_oracle.experiments.io import save_results

def run(out, seeds, jobs, quick=False):
    print("Adaptive feedback: reliability comparison and collapse trajectories")
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
    save_results(df, out, "adaptive_feedback_reliability.csv")

    # time series: complete graph, live oracle, full adoption
    hist_seeds = 100 if quick else 300
    p = Params(**{**BASE_PROBLEM, "topology": "complete", "oracle_kind": "live",
                  "oracle_tau": 1.0, "adoption_fraction": 1.0})
    H, correct = run_histories(p, hist_seeds, jobs)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "adaptive_feedback_timeseries.npz"),
             correct=correct, **{k: H[k] for k in H})
    print(f"  wrote {os.path.join(out, 'adaptive_feedback_timeseries.npz')} "
          f"({hist_seeds} runs, {H['entropy'].shape[1]} rounds, "
          f"{correct.mean():.2f} correct)")
