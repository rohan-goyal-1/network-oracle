"""
model.py
========
Core simulation engine for the "shared-oracle problem in network epistemology"
project: the Zollman / Bala-Goyal networked two-armed-bandit model of collective
inquiry, extended with a SHARED ORACLE that some or all agents consult.

Base model (no oracle)
----------------------
  * `n_agents` agents on a graph (nodes = agents, edges = who shares results).
  * Two actions:
      - SAFE arm A : known success rate `p_safe` (default 0.5); pulling it is
        uninformative about B.
      - RISKY arm B: true rate `p_safe + epsilon` (B is genuinely better),
        UNKNOWN to agents.
  * Each agent holds a Beta(alpha, beta) posterior over B's rate. Each round it
    pulls the arm it currently judges better (B iff E[p_B] > p_safe), runs
    `n_pulls` Bernoulli trials, and everyone updates on the pooled B-results of
    themselves + their network neighbours (agents share DATA, not opinions).
  * Pulling A yields no information about B, so a community that prematurely
    sours on B can stop testing it. Sparser networks can therefore converge on
    the truth MORE often -- the Zollman "transient diversity" effect.

The shared oracle (the contribution)
-------------------------------------
A single source O that a fraction `adoption_fraction` of agents consult each
round. We model it as TESTIMONY folded into the consulter's posterior: each
consultation adds `oracle_tau` pseudo-observations of B at an "implied rate"
v in (0,1) that encodes what the oracle is currently saying. Belief and action
are thus the same quantity (no decoupling).

  * "frozen": O points at the better arm with probability `oracle_r` (its
    reliability). r < 0.5 = a systematically BIASED source. When it endorses B
    the implied rate is `oracle_h`; when it endorses A it is `1 - oracle_h`.
    With `oracle_shared=True` every consulter gets the SAME endorsement that
    round (this correlation is the monoculture mechanism); with False each
    consulter draws independently (the contrast condition that isolates the
    effect of correlation from the effect of mere noise).
  * "live": O keeps its own Beta posterior, updated each round from the
    community's pooled B-data, and broadcasts its current mean as the implied
    rate. This closes a FEEDBACK LOOP -- O's broadcast shapes who pulls B, which
    determines the data O sees, which updates O. If exploration of B stops, O
    freezes at its current (possibly wrong) view.

Convergence is judged on BELIEF: the run ends when all agents sit on the same
side of `p_safe` for `stable_rounds` consecutive rounds (or `max_rounds`), and
the outcome is classified by the final beliefs.

Plain NumPy + NetworkX; runs on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_oracle.metrics import binary_entropy
from network_oracle.networks import build_network, neighborhood_matrix


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # --- community & problem ---
    n_agents: int = 10
    p_safe: float = 0.5            # known success rate of the safe arm A
    epsilon: float = 0.05          # B's true rate = p_safe + epsilon (B is better)
    n_pulls: int = 1               # Bernoulli trials per agent per round
    prior_strength: float = 4.0    # concentration of agents' Beta priors over p_B
    max_rounds: int = 800
    consensus_threshold: float = 0.99   # fraction-believing-B needed to call consensus
    stable_rounds: int = 8         # consecutive same-side rounds to declare convergence

    # --- network ---
    topology: str = "complete"     # complete|cycle|wheel|star|er|ws|ba
    topo_param: float = None        # er: edge prob; ws: ring degree k; ba: attachment m

    # --- oracle ---
    oracle_kind: str = "none"      # none|frozen|live
    oracle_r: float = 0.7          # frozen reliability  (P endorse the better arm)
    oracle_tau: float = 1.0        # trust: pseudo-observations added per consultation
    oracle_h: float = 0.75         # endorsement strength (implied success rate when endorsing B)
    oracle_shared: bool = True     # same endorsement to all consulters within a round
    consult_prob: float = 1.0      # P an adopter consults O on a given round
    adoption_fraction: float = 1.0 # fraction of agents who ever consult O
    oracle_init_obs: int = 0       # live oracle: prior pseudo-obs per arm (0 -> Beta(1,1))
    oracle_sees_all: bool = True   # live oracle learns from all B-pulls (else adopters only)

    # --- endogenous trust: consulters scale tau by observed oracle accuracy ---
    endogenous_trust: bool = False
    trust_learning_rate: float = 0.1

    # --- bookkeeping ---
    no_early_stop: bool = False    # set True when recording full-length histories


# --------------------------------------------------------------------------- #
#  Single simulation run
# --------------------------------------------------------------------------- #
def run_simulation(p: Params, seed: int = 0, record_history: bool = False) -> dict:
    """Run one community to convergence (or `max_rounds`). Returns a metrics dict.

    correct        : bool  -- reached correct consensus (B better)?
    consensus      : str   -- 'correct' | 'incorrect' | 'none'
    rounds         : int   -- rounds elapsed
    frac_believe_B : float -- fraction with E[p_B] > p_safe at the end
    history        : dict | None -- per-round time series if record_history=True
    """
    rng = np.random.default_rng(seed)
    n = p.n_agents
    p_b = p.p_safe + p.epsilon

    graph = build_network(p, rng)
    A_self = neighborhood_matrix(graph, n)

    # agents' priors over p_B: diverse means, weak concentration
    means = rng.uniform(0.0, 1.0, n)
    alpha = np.clip(means * p.prior_strength, 1e-3, None)
    beta = np.clip((1.0 - means) * p.prior_strength, 1e-3, None)

    # who consults the oracle
    n_adopt = int(round(p.adoption_fraction * n))
    adopters = np.zeros(n, dtype=bool)
    if n_adopt > 0:
        adopters[rng.choice(n, size=n_adopt, replace=False)] = True

    tau_i = np.full(n, float(p.oracle_tau))        # per-agent trust (pseudo-count weight)
    agree_ema = np.full(n, 0.5)                     # endogenous-trust state

    seed_obs = max(0, int(p.oracle_init_obs))       # live oracle posterior
    oa = 1.0 + 0.5 * seed_obs
    ob = 1.0 + 0.5 * seed_obs

    hist = ({"frac_B": [], "entropy": [], "mean_belief": [], "mean_trust": [],
             "oracle_value": [], "oracle_endorses_B": []} if record_history else None)

    same_side = 0
    rounds = p.max_rounds

    for t in range(p.max_rounds):
        e_pb = alpha / (alpha + beta)
        pulls_B = e_pb > p.p_safe

        # ---- experiments for B-pullers ----
        successes = np.zeros(n)
        trials = np.zeros(n)
        idxB = np.flatnonzero(pulls_B)
        if idxB.size:
            k = rng.binomial(p.n_pulls, p_b, size=idxB.size).astype(float)
            successes[idxB] = k
            trials[idxB] = p.n_pulls

        # ---- update posterior from pooled experimental evidence (own + neighbours) ----
        nb_succ = A_self @ successes
        nb_tri = A_self @ trials
        alpha = alpha + nb_succ
        beta = beta + (nb_tri - nb_succ)

        # ---- oracle endorsement this round -> implied rate v in (0,1) per consulter ----
        oracle_value = np.nan
        oracle_endorses_B = np.nan
        if p.oracle_kind != "none":
            consult = adopters & (rng.random(n) < p.consult_prob)
            if p.oracle_kind == "frozen":
                if p.oracle_shared:
                    bit = rng.random() < p.oracle_r
                    v_arr = np.full(n, p.oracle_h if bit else 1.0 - p.oracle_h)
                    oracle_endorses_B = float(bit)
                else:
                    bits = rng.random(n) < p.oracle_r
                    v_arr = np.where(bits, p.oracle_h, 1.0 - p.oracle_h)
                    oracle_endorses_B = float(bits.mean())
                oracle_value = float(np.mean(v_arr))
            else:  # live
                e_o = oa / (oa + ob)
                v_arr = np.full(n, e_o)
                oracle_value = e_o
                oracle_endorses_B = float(e_o > p.p_safe)

            kO = np.maximum(tau_i, 0.0)
            add_a = np.where(consult, kO * v_arr, 0.0)
            add_b = np.where(consult, kO * (1.0 - v_arr), 0.0)
            alpha = alpha + add_a
            beta = beta + add_b

        # ---- live oracle learns from the community's B-data (feedback loop) ----
        if p.oracle_kind == "live":
            if p.oracle_sees_all:
                s_tot, t_tot = successes.sum(), trials.sum()
            else:
                s_tot, t_tot = successes[adopters].sum(), trials[adopters].sum()
            oa += s_tot
            ob += (t_tot - s_tot)

        # ---- endogenous trust: can agents "discipline" a bad oracle? ----
        if p.endogenous_trust and p.oracle_kind != "none" and trials.sum() > 0:
            emp = successes.sum() / trials.sum()
            oracle_said_B = (oracle_endorses_B >= 0.5)
            evidence_says_B = emp > p.p_safe
            agree = 1.0 if (oracle_said_B == evidence_says_B) else 0.0
            lr = p.trust_learning_rate
            agree_ema = np.where(adopters, (1 - lr) * agree_ema + lr * agree, agree_ema)
            # start at full base trust; withdraw trust as the oracle's agreement with
            # the evidence falls below a coin flip (reaches ~0 trust by agreement 0.3)
            factor = np.clip((agree_ema - 0.3) / 0.2, 0.0, 1.0)
            tau_i = np.where(adopters, factor * p.oracle_tau, tau_i)

        # ---- record history ----
        if record_history:
            fB = float(pulls_B.mean())
            hist["frac_B"].append(fB)
            hist["entropy"].append(binary_entropy(fB))
            hist["mean_belief"].append(float(e_pb.mean()))
            hist["mean_trust"].append(float(tau_i[adopters].mean()) if adopters.any() else float("nan"))
            hist["oracle_value"].append(float(oracle_value))
            hist["oracle_endorses_B"].append(float(oracle_endorses_B))

        # ---- convergence: all agents on the same side of p_safe, stably ----
        if not p.no_early_stop:
            new_e = alpha / (alpha + beta)
            if (new_e > p.p_safe).all() or (new_e <= p.p_safe).all():
                same_side += 1
                if same_side >= p.stable_rounds:
                    rounds = t + 1
                    break
            else:
                same_side = 0

    # ---- classify final consensus ----
    e_pb = alpha / (alpha + beta)
    frac_believe_B = float((e_pb > p.p_safe).mean())
    if frac_believe_B >= p.consensus_threshold:
        consensus, correct = "correct", True
    elif frac_believe_B <= (1.0 - p.consensus_threshold):
        consensus, correct = "incorrect", False
    else:
        consensus, correct = "none", False

    return {"correct": correct, "consensus": consensus, "rounds": rounds,
            "frac_believe_B": frac_believe_B, "history": hist}
