"""
Shared defaults for the oracle studies.

Keep this file small: it is the one place to change the base problem regime
used by every experiment.
"""

BASE_PROBLEM = dict(
    epsilon=0.05,
    n_agents=10,
    n_pulls=1,
    max_rounds=800,
    prior_strength=4.0,
    stable_rounds=8,
)

