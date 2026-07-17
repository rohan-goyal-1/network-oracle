from __future__ import annotations

import math

import numpy as np
import pandas as pd

from network_oracle.config import BASE_PROBLEM
from network_oracle.model import Params
from network_oracle.monte_carlo import isolated_agent_reliability, run_grid
from network_oracle.experiments.io import save_results

def run(out, seeds, jobs, quick=False):
    print("Fixed shared testimony: reliability x adoption x topology")
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
    save_results(df, out, "fixed_testimony_grid.csv")

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
    save_results(df2, out, "shared_vs_independent_testimony.csv")

    # lone-agent benchmark (to mark the individually-helpful threshold)
    solo_seeds = 1000 if quick else 4000
    solo = []
    for r_ in r_grid:
        p = Params(**{**BASE_PROBLEM, "oracle_kind": "frozen", "oracle_r": float(r_),
                      "oracle_tau": 1.0, "adoption_fraction": 1.0})
        reliability = isolated_agent_reliability(p, seeds=solo_seeds)
        solo.append({
            "oracle_r": float(r_),
            "isolated_reliability": reliability,
            "isolated_se": math.sqrt(max(reliability * (1 - reliability), 0.0) / solo_seeds),
            "seeds": solo_seeds,
        })
    save_results(pd.DataFrame(solo), out, "single_agent_testimony.csv")
