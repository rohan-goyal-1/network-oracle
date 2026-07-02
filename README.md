# The shared-oracle problem in network epistemology

Simulation code for studying what happens to a community of inquirers when they
can all consult the **same imperfect source** (an "oracle" — a stand-in for an
AI system, search engine, or shared model that everyone queries). It extends the
Zollman / Bala–Goyal networked two-armed-bandit model of collective inquiry with
a shared-oracle layer and asks: does the oracle help the community converge on
the truth, or does the correlated error it injects — plus the exploration it
discourages — make the community converge faster on a confident falsehood?

The central thesis the code lets you test: **reliability is an individual-level
virtue that does not automatically transfer to collective inquiry.** A shared
source converts independent errors into correlated ones and can suppress the
transient diversity that communities rely on to self-correct.

```
network_oracle/
  config.py             shared base problem regime used by every study
  model.py              one simulation run: agents, oracle, convergence
  networks.py           topology construction and neighbourhood matrices
  metrics.py            small observables used by histories and summaries
  monte_carlo.py        repeated runs, parallel summaries, histories
  experiments/
    study0_baseline.py
    study1_frozen_oracle.py
    study2_live_oracle.py
    study3_trust.py
  figures/
    style.py            shared publication style
    study0.py ...       one plotting module per study

experiments.py          compatibility launcher for running studies
analysis.py             compatibility launcher for generating figures
model.py                compatibility launcher for model diagnostics
requirements.txt        NumPy, NetworkX, pandas, matplotlib
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+ . Everything is plain NumPy + NetworkX and runs on a laptop CPU.

## Quick start

A tiny end-to-end smoke test (small grids, few seeds, ~1–2 minutes):

```bash
python experiments.py --study all --quick --jobs 4
python analysis.py     --study all
```

A sanity check of the engine itself (reproduces the Zollman effect and the
core oracle effects, printing numbers to the terminal):

```bash
python model.py
```

## Running the studies

Each study writes result files to `results/`; the matching `analysis.py` call
reads them and writes figures to `figures/`. Use more seeds for publication-grade
confidence intervals (300 is a reasonable default; 1000+ tightens the bands).

```bash
# Study 0 — replication & robustness map
python experiments.py --study 0 --seeds 300 --jobs 4
python analysis.py     --study 0

# Study 1 — frozen oracle: the core sweep
python experiments.py --study 1 --seeds 300 --jobs 4
python analysis.py     --study 1

# Study 2 — live oracle: feedback & collapse
python experiments.py --study 2 --seeds 300 --jobs 4
python analysis.py     --study 2

# Study 3 — biased oracle, endogenous trust, robustness
python experiments.py --study 3 --seeds 300 --jobs 4
python analysis.py     --study 3
```

To change the underlying problem regime for **every** study (difficulty,
community size, trials per round, etc.), edit `BASE_PROBLEM` in
`network_oracle/config.py`.

## Project flow

The code is organized around the research workflow:

1. `network_oracle/model.py` defines the mechanism for one community.
2. `network_oracle/networks.py` owns graph construction.
3. `network_oracle/monte_carlo.py` repeats runs across seeds and summarizes
   outcomes.
4. `network_oracle/experiments/study*.py` defines parameter grids and writes
   result files.
5. `network_oracle/figures/study*.py` turns those result files into figures,
   using the common visual system in `network_oracle/figures/style.py`.

To add a new study, create a `network_oracle/experiments/study4_*.py` module,
register it in `network_oracle/experiments/cli.py`, and add a matching
`network_oracle/figures/study4.py` only if it needs new plots. To change agent
or oracle behavior, work in `network_oracle/model.py`; to add a topology, work
in `network_oracle/networks.py`; to change replication, parallelism, or output
summaries, work in `network_oracle/monte_carlo.py`.

### What each study does

**Study 0 — replication & robustness map.** Reliability and speed across network
topology (complete, cycle, wheel, Erdős–Rényi, small-world, scale-free), problem
difficulty `epsilon`, and community size. Reproduces the Zollman "transient
diversity" effect (sparser networks converge on the truth more often) and shows
where it holds. *Files:* `study0.csv` → `study0_reliability.png`.

**Study 1 — frozen oracle (the core).** A source that points at the better arm
with probability `r` (its reliability; `r < 0.5` = biased). Sweeps reliability ×
adoption fraction × topology, plus a shared-vs-independent contrast and a
lone-agent benchmark. Produces the help/harm map, the **correlation penalty**
(shared does worse than independent at equal reliability), and the
**individually-helpful-but-collectively-harmful band**. *Files:*
`study1_grid.csv`, `study1_sharedvsindep.csv`, `study1_isolated.csv` →
`study1_heatmaps.png`, `study1_phase_and_band.png`,
`study1_shared_vs_independent.png`.

**Study 2 — live oracle (feedback).** The oracle maintains its own posterior,
updated each round from the community's pooled data, and rebroadcasts it —
closing a feedback loop. Compares reliability to none/frozen and records time
series of exploration entropy and the oracle's confidence-vs-accuracy. Shows the
**monoculture collapse**: exploration entropy falls to zero and the oracle's
belief can freeze on a falsehood. *Files:* `study2_reliability.csv`,
`study2_timeseries.npz` → `study2_reliability.png`,
`study2_entropy_collapse.png`, `study2_oracle_freeze.png`,
`study2_exploration.png`.

**Study 3 — biased oracle, endogenous trust, robustness.** A systematically
biased source; whether agents can "discipline" it by withdrawing trust as it
disagrees with the evidence; the network-structure inversion; and a robustness
sweep of the correlation penalty across difficulty, size, and trust. *Files:*
`study3_endogenous.csv`, `study3_robustness.csv`, `study3_trust_traj.npz` →
`study3_endogenous_trust.png`, `study3_network_inversion.png`,
`study3_robustness.png`, `study3_trust_trajectories.png`.

## What the code reproduces

With the default regime (`epsilon = 0.05`, `n = 10`, 1 trial/round), 300 seeds:

- **Zollman effect.** Complete graph ≈ 0.62 correct-consensus rate vs cycle
  ≈ 0.86; the denser network converges on *falsehood* far more often.
- **Frozen oracle is monotone in reliability.** A biased oracle (`r = 0.2`)
  drags reliability *below* baseline (≈ 0.45); a reliable one (`r = 0.9`) lifts
  it to ≈ 1.0.
- **Correlation penalty.** At identical reliability `r = 0.6`, the shared
  (correlated) oracle does measurably worse than independent draws.
- **Live-oracle feedback collapse.** In runs that end wrong, the oracle's belief
  crashes below the indifference line and freezes there for the rest of the run —
  permanent lock-in on the worse arm.
- **Network-structure inversion.** A biased *shared* oracle devastates the sparse
  cycle (reliability → ~0) far more than the dense complete graph: density helps
  wash out a correlated bias, inverting the usual "sparser is better" ordering.

These are illustrative, not universal — see the robustness note below.

## Modeling choices (documented)

- **Two-armed bandit with a known safe arm.** Arm A has known rate `p_safe`
  (0.5); arm B's rate `p_safe + epsilon` is unknown and better. Agents hold a
  Beta posterior over B and pull the arm they currently judge better; pulling A
  is uninformative about B, so a community can prematurely abandon B. Agents share
  experimental **data** (not opinions) with network neighbours.
- **The oracle as testimony folded into the posterior.** Each consultation adds
  `oracle_tau` pseudo-observations of B at an "implied rate" encoding what the
  oracle currently says. Belief and action are therefore the *same* quantity —
  there is no decoupling between what an agent believes and how it acts. (An
  alternative "decision-weight" oracle, which blends the oracle into the action
  rule while leaving the evidence-based posterior untouched, was prototyped and
  set aside because it decouples belief from action and complicates the outcome
  measure; it is a reasonable **extension** if you want to separate "advice that
  changes behaviour" from "testimony that changes belief.")
- **Frozen vs live.** "Frozen" fixes the oracle's reliability over time (a
  knowledge-cutoff analogue). "Live" updates the oracle from the community's data
  and rebroadcasts (a continual-learning / retrieval analogue), creating the
  feedback loop.
- **Shared vs independent.** With `oracle_shared=True` all consulters get the
  same endorsement within a round (the monoculture/correlation mechanism); with
  `False` each draws independently (the contrast that isolates correlation from
  mere noise).
- **Convergence is judged on belief.** A run ends when all agents sit on the same
  side of `p_safe` for `stable_rounds` consecutive rounds (or `max_rounds`), and
  is classified `correct` / `incorrect` / `none` by the final beliefs.
- **Endogenous trust** (Study 3) scales each consulter's `oracle_tau` by an EMA of
  how often the oracle's endorsement has agreed with realized evidence; trust
  falls toward zero for a clearly biased source — but only where exploration
  survives to expose it. The effect is deliberately *not* tuned to look dramatic;
  its weakness is part of the finding (a "suppression trap": a biased source that
  stops the community exploring also stops it gathering the evidence that would
  reveal the bias).

## A note on robustness (please read before drawing conclusions)

Results from this family of models are known to be **parameter-sensitive**
(Rosenstock, Bruner & O'Connor 2017 showed the original "sparser is better"
result is not robust across all parameter settings). Treat any single
configuration as a *how-possibly* demonstration, not a prediction. Before
claiming an effect, sweep `epsilon`, `n_agents`, `n_pulls`, `prior_strength`,
and `oracle_tau` and report where the effect holds and where it reverses. The
robustness sweep in Study 3 is a template for exactly this.

## Key parameters (`model.Params`)

| Parameter | Meaning |
|---|---|
| `n_agents`, `topology`, `topo_param` | community size and network structure |
| `epsilon` | how much better arm B is (smaller = harder problem) |
| `n_pulls` | Bernoulli trials per agent per round |
| `prior_strength` | concentration of agents' Beta priors |
| `oracle_kind` | `none` / `frozen` / `live` |
| `oracle_r` | frozen reliability (P endorse the better arm; < 0.5 = biased) |
| `oracle_tau` | trust = pseudo-observations added per consultation |
| `oracle_h` | endorsement strength (implied success rate) |
| `oracle_shared` | same endorsement to all consulters within a round? |
| `adoption_fraction` | fraction of agents who consult the oracle |
| `consult_prob` | per-round probability an adopter consults |
| `endogenous_trust`, `trust_learning_rate` | adaptive-trust mechanism (Study 3) |
| `max_rounds`, `stable_rounds`, `consensus_threshold` | stopping & classification |

## Extending the model

Natural next steps the code is structured to support:

- **Many oracles vs one** — add a small set of partially-correlated sources and
  vary their correlation (monopoly vs oligopoly of sources).
- **Tighter loops** — let the live oracle learn only from the agents who report
  to it, or make it one of the networked agents.
- **Testimony vs evidence** — implement the decision-weight oracle alongside the
  current testimony oracle and compare.
- **Richer problem structure** — replace the two arms with a many-armed bandit or
  an epistemic landscape.
- **LLM instantiation** — swap the synthetic oracle for an actual model queried
  with the same protocol.
