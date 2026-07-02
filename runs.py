"""
Reusable Monte Carlo runners for the shared-oracle model.

The model module owns one simulation run. This module owns repeated runs,
parallel execution, and tidy tabular summaries used by experiments.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from model import Params, run_simulation


def _summary_worker(args):
    params, seed = args
    result = run_simulation(params, seed=seed, record_history=False)
    return (
        int(result["correct"]),
        result["rounds"],
        result["frac_believe_B"],
        1 if result["consensus"] == "none" else 0,
        1 if result["consensus"] == "incorrect" else 0,
    )


def _history_worker(args):
    params, seed = args
    result = run_simulation(params, seed=seed, record_history=True)
    return result["history"], int(result["correct"])


def summarize_runs(params: Params, seeds: int, jobs: int = 1, base_seed: int = 0) -> dict:
    """Run independent communities and return aggregate reliability metrics."""
    tasks = [(params, base_seed + seed) for seed in range(seeds)]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            rows = list(
                executor.map(
                    _summary_worker,
                    tasks,
                    chunksize=max(1, seeds // (jobs * 4)),
                )
            )
    else:
        rows = [_summary_worker(task) for task in tasks]

    data = np.asarray(rows, dtype=float)
    reliability = data[:, 0].mean()
    return {
        "reliability": reliability,
        "se": math.sqrt(max(reliability * (1 - reliability), 0.0) / seeds),
        "mean_rounds": data[:, 1].mean(),
        "frac_none": data[:, 3].mean(),
        "frac_wrong": data[:, 4].mean(),
        "seeds": seeds,
    }


def run_grid(cells, seeds: int, jobs: int = 1, base_seed: int = 0):
    """Run a list of ``(row_metadata, Params)`` cells and return one row per cell."""
    import pandas as pd

    tasks = []
    owner = []
    for cell_index, (_row, params) in enumerate(cells):
        for seed in range(seeds):
            tasks.append((params, base_seed + seed))
            owner.append(cell_index)

    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(
                executor.map(
                    _summary_worker,
                    tasks,
                    chunksize=max(1, len(tasks) // (jobs * 8)),
                )
            )
    else:
        results = [_summary_worker(task) for task in tasks]

    results = np.asarray(results, dtype=float)
    owner = np.asarray(owner)
    rows = []
    for cell_index, (row, _params) in enumerate(cells):
        subset = results[owner == cell_index]
        reliability = subset[:, 0].mean()
        rows.append(
            {
                **row,
                "reliability": reliability,
                "se": math.sqrt(max(reliability * (1 - reliability), 0.0) / seeds),
                "mean_rounds": subset[:, 1].mean(),
                "frac_none": subset[:, 3].mean(),
                "frac_wrong": subset[:, 4].mean(),
                "seeds": seeds,
            }
        )
    return pd.DataFrame(rows)


def run_histories(params: Params, seeds: int, jobs: int = 1, base_seed: int = 1000):
    """Run full-length histories and stack each recorded series into arrays."""
    params = Params(**{**asdict(params), "no_early_stop": True})
    tasks = [(params, base_seed + seed) for seed in range(seeds)]

    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            output = list(
                executor.map(
                    _history_worker,
                    tasks,
                    chunksize=max(1, seeds // (jobs * 4)),
                )
            )
    else:
        output = [_history_worker(task) for task in tasks]

    keys = output[0][0].keys()
    histories = {
        key: np.array([item[0][key] for item in output], dtype=float)
        for key in keys
    }
    correct = np.array([item[1] for item in output], dtype=bool)
    return histories, correct


def isolated_agent_reliability(params: Params, seeds: int = 3000, base_seed: int = 7) -> float:
    """Benchmark the current oracle settings for one isolated agent."""
    solo = Params(**{**asdict(params), "n_agents": 1, "topology": "complete"})
    hits = 0
    for seed in range(seeds):
        hits += run_simulation(solo, seed=base_seed + seed)["correct"]
    return hits / seeds
