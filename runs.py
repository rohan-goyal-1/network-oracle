"""Compatibility exports for Monte Carlo helpers.

The implementation lives in :mod:`network_oracle.monte_carlo`.
"""

from network_oracle.monte_carlo import (
    isolated_agent_reliability,
    run_grid,
    run_histories,
    summarize_runs,
)

__all__ = [
    "isolated_agent_reliability",
    "run_grid",
    "run_histories",
    "summarize_runs",
]
