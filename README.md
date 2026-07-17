# Shared-oracle dynamics in network epistemology

Simulation code for asking what happens when a community of inquirers can all
consult the **same imperfect source**: an AI system, search engine, shared model,
or other common testimony channel. The model extends the Zollman / Bala-Goyal
networked two-armed-bandit framework with a shared-oracle layer and tests when
the source helps a community converge on the truth, when correlated error makes
it converge confidently on a falsehood, and when feedback suppresses the
exploration needed for correction.

The central claim this code lets you probe is: **reliability is an
individual-level virtue that does not automatically transfer to collective
inquiry.** A shared source can convert independent errors into correlated ones
and can reduce the transient diversity communities rely on to self-correct.

```
network_oracle/
  config.py             shared base problem regime
  model.py              one simulation run: agents, oracle, convergence
  networks.py           topology construction and neighbourhood matrices
  metrics.py            observables used by histories and summaries
  monte_carlo.py        repeated runs, parallel summaries, histories
  experiments/
    baseline_reliability.py
    fixed_testimony.py
    adaptive_feedback.py
    biased_trust.py
  figures/
    style.py            shared publication style
    baseline_reliability.py
    fixed_testimony.py
    adaptive_feedback.py
    biased_trust.py

experiments.py          compatibility launcher for simulations
analysis.py             compatibility launcher for figures
model.py                compatibility launcher for model diagnostics
requirements.txt        NumPy, NetworkX, pandas, matplotlib
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+ is enough. Everything is plain NumPy, NetworkX, pandas, and
matplotlib, and runs on a laptop CPU.

## Quick Start

A tiny end-to-end smoke test:

```bash
python experiments.py --run all --quick --jobs 4
python analysis.py --run all
```

A diagnostic check of the engine itself:

```bash
python model.py
```

## Publication Runs

Each simulation writes result files to `results/`; the matching figure command
reads them and writes high-resolution PNG and PDF files to `figures/`. Use more
seeds for publication-grade confidence intervals. `300` is a reasonable working
default; `1000+` gives tighter bands when runtime permits.

```bash
# Baseline network reliability
python experiments.py --run baseline --seeds 300 --jobs 4
python analysis.py --run baseline

# Fixed shared testimony
python experiments.py --run fixed-testimony --seeds 300 --jobs 4
python analysis.py --run fixed-testimony

# Adaptive feedback
python experiments.py --run adaptive-feedback --seeds 300 --jobs 4
python analysis.py --run adaptive-feedback

# Biased testimony and adaptive trust
python experiments.py --run biased-trust --seeds 300 --jobs 4
python analysis.py --run biased-trust
```

To regenerate everything:

```bash
python experiments.py --run all --seeds 300 --jobs 4
python analysis.py --run all
```

Change the shared problem regime for every simulation block by editing
`BASE_PROBLEM` in `network_oracle/config.py`.

## Simulation Blocks

**Baseline network reliability.** Reliability and speed across topology
(`complete`, `cycle`, `wheel`, Erdos-Renyi, small-world, scale-free), problem
difficulty `epsilon`, and community size. This reproduces the Zollman transient
diversity effect and maps where it holds.

Files: `baseline_reliability.csv` -> `baseline_network_reliability.png/.pdf`

**Fixed shared testimony.** A source points at the better arm with probability
`r`; `r < 0.5` is systematically biased. This sweeps reliability, adoption
fraction, and topology; contrasts shared with independent testimony; and compares
the community against a lone-agent benchmark.

Files: `fixed_testimony_grid.csv`, `shared_vs_independent_testimony.csv`,
`single_agent_testimony.csv` -> `fixed_testimony_heatmaps.png/.pdf`,
`shared_vs_independent_testimony.png/.pdf`,
`individual_help_collective_harm.png/.pdf`

**Adaptive feedback.** The oracle maintains its own posterior, updates each
round from the community's pooled data, and rebroadcasts it. This closes the
feedback loop and records exploration entropy, testing of the better arm, and
oracle belief over time.

Files: `adaptive_feedback_reliability.csv`,
`adaptive_feedback_timeseries.npz` -> `adaptive_feedback_reliability.png/.pdf`,
`exploration_entropy_collapse.png/.pdf`, `better_arm_exploration.png/.pdf`,
`adaptive_oracle_lock_in.png/.pdf`

**Biased testimony and adaptive trust.** A systematically biased shared source;
whether agents can discipline it by withdrawing trust; the network-structure
inversion; and a robustness sweep of the correlation penalty across difficulty,
community size, and testimony weight.

Files: `adaptive_trust_under_bias.csv`,
`correlation_penalty_robustness.csv`, `adaptive_trust_trajectories.npz` ->
`adaptive_trust_under_bias.png/.pdf`, `network_structure_inversion.png/.pdf`,
`correlation_penalty_robustness.png/.pdf`,
`adaptive_trust_trajectories.png/.pdf`

## Project Flow

1. `network_oracle/model.py` defines one community run.
2. `network_oracle/networks.py` owns graph construction.
3. `network_oracle/monte_carlo.py` repeats runs across seeds and summarizes
   outcomes.
4. `network_oracle/experiments/*.py` defines parameter grids and writes result
   files.
5. `network_oracle/figures/*.py` turns those result files into paper figures,
   using the common visual system in `network_oracle/figures/style.py`.

To add a new simulation block, add a module under `network_oracle/experiments/`,
register it in `network_oracle/experiments/cli.py`, and add a matching module
under `network_oracle/figures/` when it needs new plots.

## Modeling Choices

- **Two-armed bandit with a known safe arm.** Arm A has known rate `p_safe`
  (`0.5`); arm B's rate is `p_safe + epsilon` and is unknown but better.
- **The oracle as testimony folded into the posterior.** Each consultation adds
  `oracle_tau` pseudo-observations of B at an implied rate encoding the oracle's
  endorsement.
- **Fixed vs adaptive.** Fixed testimony has exogenous reliability. Adaptive
  feedback updates the oracle from community data and rebroadcasts it.
- **Shared vs independent.** With `oracle_shared=True`, all consulters get the
  same endorsement within a round. With `False`, each draws independently.
- **Convergence is judged on belief.** A run ends when all agents sit on the same
  side of `p_safe` for `stable_rounds` consecutive rounds, or at `max_rounds`.
- **Adaptive trust.** Consulters scale `oracle_tau` by an EMA of how often the
  oracle agrees with realized evidence. It only works where exploration survives
  long enough to expose bias.

## Robustness

Results in this family of models are parameter-sensitive. Treat a single
configuration as a how-possibly demonstration, not a prediction. Before claiming
an effect, sweep `epsilon`, `n_agents`, `n_pulls`, `prior_strength`, and
`oracle_tau`, and report where the effect holds or reverses. The
`biased-trust` block includes a compact robustness template.

## Key Parameters

| Parameter | Meaning |
|---|---|
| `n_agents`, `topology`, `topo_param` | community size and network structure |
| `epsilon` | how much better arm B is |
| `n_pulls` | Bernoulli trials per agent per round |
| `prior_strength` | concentration of agents' Beta priors |
| `oracle_kind` | `none` / `frozen` / `live` |
| `oracle_r` | fixed reliability; `< 0.5` means biased |
| `oracle_tau` | testimony weight in pseudo-observations |
| `oracle_h` | endorsement strength |
| `oracle_shared` | shared endorsement or independent endorsements |
| `adoption_fraction` | fraction of agents who consult the oracle |
| `consult_prob` | per-round consultation probability |
| `endogenous_trust`, `trust_learning_rate` | adaptive-trust mechanism |
| `max_rounds`, `stable_rounds`, `consensus_threshold` | stopping and classification |
