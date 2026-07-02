from __future__ import annotations

from network_oracle.config import BASE_PROBLEM
from network_oracle.model import Params
from network_oracle.monte_carlo import run_grid
from network_oracle.experiments.io import save_results

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
    save_results(df, out, "study0.csv")
    piv = df[df.n_agents == sizes[0]].pivot_table(index="topology",
                                                  columns="epsilon", values="reliability")
    print(piv.round(3).to_string())
