"""Matched comparisons of planners against exact Tiger references.

Each case fixes one decision problem (belief, physics, discount, horizon and
opponent model) and asks a planner for its root policy and action values. The
same problem is solved exactly, and the row records:

- ``first_action_loss`` = V*(b) - sum_a pi(a | b) Q*(b, a): the expected value
  lost by the planner's root decision when every later decision is optimal. It
  handles ties exactly and is not an episode regret.
- ``max_abs_q_error`` = max_a |Q_hat(b, a) - Q*(b, a)|.
- ``optimal_q_error`` = the same maximum over the exactly optimal actions only
  (``optimal_q_error``): the error of the value the decision rests on.

For I-POMCP, Q_hat is the post-search chance-scored value that decides
(``planning.chance_scoring``). The same search's empirical-backup values are
recorded beside it (``search_q``, ``search_first_action_loss``,
``search_max_abs_q_error``, ``search_optimal_q_error``), so the two are compared on identical trees, and
``scores_exact`` says whether the scored root is exact by construction.

Problems
--------
- Level 1: agent i against uniform-random L0 j; reference = alpha-vector value
  iteration (``reference.pomdp_exact_vi``).
- Level 2 with an exact opponent: j is an exact L1 agent replanning at a fixed
  depth, with a point prior on its private belief. The reference is the exact
  I-POMDP.
- Level 2 with a finite opponent: j is solved by the same kind of planner as i
  with the declared ``opponent_budget`` (I-POMCP simulations, or RTS samples
  per node, as in IPF + RTS where every level is solved by RTS). The reference
  is either the exact best response to that declared opponent (``declared``:
  search error only) or the exact I-POMDP (``exact``: search plus opponent-model
  error).

Planners: ``mcts`` (I-POMCP; budget = root simulations) and ``rts``
(reachability-tree sampling; budget = sampled observations per node). The units
differ, so cost is compared as the planner's process CPU time for the root
decision.

Mid-episode problems: ``history_steps=k`` conditions i on the first k steps of
a fixed history before the decision (i listens except for one opening at step
4, j acts uniformly at random, the environment samples states and i's
observations; ``sample_history``). The planner conditions with its own bank's
filter (exact or particle), the exact reference with the exact filter, so the
problem is the same while exact supports grow with k. The CPU of those k
updates is recorded separately from the decision's.

Beliefs: exact by default; ``particles=N`` bounds the planner's beliefs to N
atoms, by the compressed filter (``particle_filter="compressed"``) or the
published sampled IPF (``"sampled"``; ``rts`` with it is IPF + RTS). The exact I-POMDP reference is always solved in a separate
exact bank, so it never inherits particle error. A ``declared`` reference would
not be exact with particles (the declared opponent's own updates would be
sampled), so that combination is rejected.
"""

import argparse
import itertools
import json
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import LISTEN, OPEN_LEFT, TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.policy import greedy_policy
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner
from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver


def evaluate_case(
    planner_kind,
    horizon,
    budget,
    seed,
    belief_p,
    *,
    gamma=0.95,
    exploration=1.0,
    level=1,
    opponent_depth=None,
    opponent_belief=0.5,
    opponent_budget=None,
    reference="declared",
    particles=None,
    particle_filter="compressed",
    history_steps=0,
    nested_belief=0.5,
    opponent_observer=None,
    opponent_stopping=None,
):
    """Solve one matched problem with the planner and the exact reference.

    ``reference`` matters only for a level-2 or level-3 problem with a finite
    opponent: "declared" scores the planner against the best response to its
    own finite opponent models (search error only); "exact" scores it against
    the exact I-POMDP, where every lower level is exact (search plus model
    error).

    At level 3, j is a level-2 agent whose private point belief is
    ``opponent_belief`` and who models i as a level-1 agent with private point
    belief ``nested_belief`` (against uniform L0 j).

    ``opponent_observer`` (levels 2-3) is called as ``observer(model, policy)``
    for every model of the opponent (j at level - 1) whose modeled policy the
    planner requests, with the policy it got. These are the beliefs at which the
    opponent's budget must be qualified (``experiments.modeled_qualification``);
    the bank's policy cache means each distinct model is reported once.

    ``opponent_stopping`` (I-POMCP only) gives every finite modeled level a
    ``replicates.ModeledStopping`` rule, with ``opponent_budget`` as its cap.
    """
    _validate_problem(
        planner_kind, level, opponent_depth, opponent_belief, opponent_budget, reference, particles
    )
    if history_steps and level == 1:
        raise ValueError("The L1 alpha-vector reference is solved at the prior; use level 2")
    if opponent_observer is not None and level == 1:
        raise ValueError("Level 1 has a uniform L0 opponent; there is no modeled policy to observe")
    if not 0 <= nested_belief <= 1:
        raise ValueError("Nested belief must lie in [0,1]")
    model = TigerModel()
    bank = SolverBank(seed, particles=particles, particle_filter=particle_filter)
    key = SolverKey("i", level)
    config = planner_config(planner_kind, budget, horizon, gamma, exploration)
    planner = _planner(planner_kind, key, model, bank, config)
    bank.register_solver(key, planner)
    # L0 action streams are never drawn in a single root decision; seed them anyway.
    for agent in ("i", "j"):
        rng = random.Random(bank.stream_seed("actions", f"{agent}_L0"))
        bank.register_solver(SolverKey(agent, 0), RandomPlanner(model.get_all_actions(agent), rng))
    best_response = BestResponseReference(model, bank, gamma)
    # Every intentional level below i: exact, or the same kind of finite planner.
    for key_below in _modeled_keys(level):
        if opponent_budget is None:
            provider = ExactIPOMDPPlanner(key_below, model, bank, best_response, opponent_depth)
        else:
            provider = _planner(
                planner_kind,
                key_below,
                model,
                bank,
                planner_config(
                    planner_kind,
                    opponent_budget,
                    opponent_depth,
                    gamma,
                    exploration,
                    modeled_stopping=opponent_stopping,
                ),
            )
        if opponent_observer is not None and key_below.level == level - 1:
            provider = _ObservedSolver(provider, opponent_observer)
        bank.register_solver(key_below, provider)
    belief = _two_state_belief(model, level, belief_p, opponent_belief, nested_belief)
    planner.set_initial_belief(belief)
    prior_model = planner.model()
    history = sample_history(
        model, history_steps, random.Random(f"history/{seed}/{belief_p}/{opponent_belief}")
    )
    started = time.process_time()
    for action, observation in history:
        planner.update_root(action, observation)
    update_cpu_seconds = time.process_time() - started
    started = time.process_time()
    policy = planner.policy_for(planner.model())
    planner_cpu_seconds = time.process_time() - started
    estimates = dict(planner.recommendation_values)
    if level == 1:
        oracle_q = ExactPOMDPSolver(model, horizon=horizon, gamma=gamma).q_values(belief_p, horizon)
    elif reference == "exact" or opponent_budget is None:
        oracle_q = _exact_ipomdp_q(
            model, prior_model, history, seed, gamma, opponent_depth, horizon, level
        )
    else:
        oracle_q = best_response.q_values(planner.model(), horizon)
    if set(estimates) != set(oracle_q):
        raise ValueError("Planner must evaluate every legal action")
    value = max(oracle_q.values())

    def loss(policy):
        return value - sum(prob * oracle_q[action] for action, prob in policy.items())

    def q_error(estimates):
        return max(abs(estimates[a] - oracle_q[a]) for a in oracle_q)

    search = {}
    if planner_kind == "mcts":
        search = {
            "search_q": dict(planner.search_values),
            "search_first_action_loss": loss(greedy_policy(planner.search_values)),
            "search_max_abs_q_error": q_error(planner.search_values),
            "search_optimal_q_error": optimal_q_error(planner.search_values, oracle_q),
            "scores_exact": planner.scores.exact,
            "fallback_actions": planner.scores.fallback_actions,
        }
    return [
        {
            "planner": planner_kind,
            "planner_config": asdict(config),
            "level": level,
            "opponent": _opponent_description(
                level, opponent_depth, opponent_belief, opponent_budget, nested_belief
            )
            | ({"stopping": asdict(opponent_stopping)} if opponent_stopping else {}),
            "reference": reference if level >= 2 and opponent_budget is not None else "exact",
            "particles": particles,
            "particle_filter": particle_filter if particles else None,
            "history_steps": history_steps,
            "belief_support": len(planner.belief.mass),
            "history_update_cpu_seconds": update_cpu_seconds,
            "horizon": horizon,
            "budget": budget,
            "seed": seed,
            "belief_p": belief_p,
            "gamma": gamma,
            "oracle_q": oracle_q,
            "estimated_q": estimates,
            "policy": policy,
            "first_action_loss": loss(policy),
            "max_abs_q_error": q_error(estimates),
            "optimal_q_error": optimal_q_error(estimates, oracle_q),
            "planner_cpu_seconds": planner_cpu_seconds,
            **search,
        }
    ]


OPTIMALITY_TOLERANCE = 1e-9


def optimal_q_error(estimates, exact):
    """max |Q_hat(b, a) - Q*(b, a)| over the actions that are exactly optimal.

    The all-action maximum (``max_abs_q_error``) is dominated by clearly wrong
    actions: search rightly spends few simulations below them, so their
    continuation values stay noisy while the decision is unaffected (RESULTS.md
    Section 12.2: L1 H5 at 5,000 simulations, 1.35 over all actions vs 0.12 on
    the optimal one). Error bounds on the value an agent acts on are reported
    with this metric; both are recorded.
    """
    best = max(exact.values())
    return max(abs(estimates[a] - q) for a, q in exact.items() if q >= best - OPTIMALITY_TOLERANCE)


class _ObservedSolver:
    """A registered solver that reports each modeled policy it computes, with its model.

    The bank asks a registered solver only for modeled policies, so delegating
    ``policy_for`` and the physics it checks frames against is the whole contract.
    """

    def __init__(self, solver, observer):
        self.solver, self.observer = solver, observer
        self.pomdp_model = solver.pomdp_model

    def policy_for(self, model, modeled=False):
        policy = self.solver.policy_for(model, modeled)
        if modeled:
            self.observer(model, policy)
        return policy


def planner_config(planner_kind, budget, depth, gamma, exploration, modeled_stopping=None):
    """The configuration every oracle problem gives a planner of this kind and budget."""
    if planner_kind == "mcts":
        return PlannerConfig(
            simulations=budget,
            depth=depth,
            modeled_simulations=budget,
            gamma=gamma,
            exploration=exploration,
            modeled_stopping=modeled_stopping,
        )
    if modeled_stopping is not None:
        raise ValueError("Sequential modeled stopping is defined for I-POMCP only")
    return RTSConfig(depth=depth, observation_samples=budget, gamma=gamma)


def _planner(planner_kind, key, model, bank, config):
    planner_class = IPOMCPPlanner if planner_kind == "mcts" else RTSPlanner
    return planner_class(key, model, bank, config)


HISTORY_ACTIONS = (LISTEN, LISTEN, LISTEN, OPEN_LEFT, LISTEN, LISTEN, LISTEN, LISTEN)


def sample_history(physics, steps, rng):
    """i's first ``steps`` (action, observation) pairs of a Tiger history.

    i follows ``HISTORY_ACTIONS``, j acts uniformly at random, and the
    environment samples the initial state, transitions and i's observations
    from ``rng``. Every observation has positive probability under every
    belief (Tiger's sensors are noisy), so any agent can condition on it.
    """
    if not 0 <= steps <= len(HISTORY_ACTIONS):
        raise ValueError(f"history_steps must lie in [0, {len(HISTORY_ACTIONS)}]")
    state, history = physics.sample_initial_state(rng), []
    for action in HISTORY_ACTIONS[:steps]:
        joint = {"i": action, "j": rng.choice(physics.get_all_actions("j"))}
        state = physics.sample_transition(state, joint, rng)
        history.append((action, physics.sample_observation(state, joint, "i", rng)))
    return history


# The exact reference's enumeration guard. The default guard (200,000 branches)
# stopped the L2 H3 reference at history 6; with this budget it completed there in
# 146 s (one problem), so mid-episode problems stay scorable.
REFERENCE_BRANCHES = 50_000_000


def _modeled_keys(level):
    """Solver keys of the intentional levels below an i-agent of ``level``.

    i at level L models j at L - 1, who models i at L - 2, and so on down to
    level 1; both level-0 agents are uniform random.
    """
    return [SolverKey("j" if (level - lvl) % 2 else "i", lvl) for lvl in range(level - 1, 0, -1)]


def _exact_ipomdp_q(model, prior_model, history, seed, gamma, opponent_depth, horizon, level):
    """Q of the exact I-POMDP at the exact posterior after ``history``, every lower level exact."""
    bank = SolverBank(seed)
    best_response = BestResponseReference(model, bank, gamma, max_branches=REFERENCE_BRANCHES)
    for agent in ("i", "j"):
        rng = random.Random(bank.stream_seed("actions", f"{agent}_L0"))
        bank.register_solver(SolverKey(agent, 0), RandomPlanner(model.get_all_actions(agent), rng))
    for key in _modeled_keys(level):
        bank.register_solver(
            key, ExactIPOMDPPlanner(key, model, bank, best_response, opponent_depth)
        )
    agent_model = prior_model
    for action, observation in history:
        posterior = bank.filter.update(agent_model, action, observation, terminal=False).belief
        agent_model = MentalModel(agent_model.frame, posterior)
    return best_response.q_values(agent_model, horizon)


def _two_state_belief(model, level, belief_p, opponent_belief, nested_belief=0.5):
    """i's exact two-state prior over a chain of point models, down to uniform L0.

    The chain's tiger beliefs are i's ``belief_p``, then j's ``opponent_belief``
    (level >= 2), then the modeled i's ``nested_belief`` (level 3).
    """
    return _point_model(model, "i", level, (belief_p, opponent_belief, nested_belief)).belief


def _point_model(model, agent, level, tiger_beliefs):
    """A mental model of ``agent`` at ``level`` whose belief puts one model on the opponent."""
    if level == 0:
        return MentalModel(AgentFrame(agent, 0, model))
    other = _point_model(model, "j" if agent == "i" else "i", level - 1, tiger_beliefs[1:])
    p = tiger_beliefs[0]
    belief = FiniteBelief(
        ((InteractiveState(TIGER_LEFT, other), p), (InteractiveState(TIGER_RIGHT, other), 1 - p))
    )
    return MentalModel(AgentFrame(agent, level, model), belief)


def _opponent_description(level, depth, belief, budget, nested_belief):
    if level == 1:
        return {"policy": "uniform_l0"}
    description = {"level": level - 1, "depth": depth, "initial_belief_p": belief}
    if level == 3:
        description["nested_initial_belief_p"] = nested_belief
    if budget is None:
        return {"policy": "exact_fixed_depth", **description}
    return {"policy": "same_planner", "budget": budget, **description}


def _validate_problem(
    planner_kind, level, opponent_depth, opponent_belief, opponent_budget, reference, particles
):
    if planner_kind not in ("mcts", "rts"):
        raise ValueError("Unknown planner")
    if reference not in ("declared", "exact"):
        raise ValueError("reference must be 'declared' or 'exact'")
    if particles is not None and reference == "declared" and opponent_budget is not None:
        raise ValueError("A declared reference is not exact with particle beliefs; use 'exact'")
    if level not in (1, 2, 3):
        raise ValueError("Reference level must be 1, 2 or 3")
    if not 0 <= opponent_belief <= 1:
        raise ValueError("Opponent belief must lie in [0,1]")
    if level == 1 and (opponent_depth is not None or opponent_budget is not None):
        raise ValueError("Level 1 has a uniform L0 opponent; it takes no opponent depth or budget")
    if level >= 2:
        if type(opponent_depth) is not int or opponent_depth < 1:
            raise ValueError("Levels 2-3 require an explicit positive opponent depth")


def run_oracle_comparison(
    out,
    *,
    budgets,
    horizons,
    seeds,
    beliefs,
    planners=("mcts",),
    seed_start=0,
    gamma=0.95,
    exploration=1.0,
    level=1,
    opponent_depth=None,
    opponent_belief=0.5,
    opponent_budget=None,
    reference="declared",
    particles=None,
    particle_filter="compressed",
    history_steps=0,
    workers=2,
    timeout=2400,
    max_rss_mb=4096,
):
    """Run every (planner, horizon, budget, seed, belief) case under supervision.

    Every outcome, failed or not, is written before interpretation. There is no
    pass/fail threshold here: a study states its tolerance before running.
    """
    for kind in planners:
        _validate_problem(
            kind, level, opponent_depth, opponent_belief, opponent_budget, reference, particles
        )
    if type(seeds) is not int or seeds < 1 or type(seed_start) is not int or seed_start < 0:
        raise ValueError("Need a positive seed count and a nonnegative first seed")
    # Each planner enforces its own minimum budget (MCTS: one simulation per action).
    if not budgets or any(b < 1 for b in budgets) or not horizons or any(h < 1 for h in horizons):
        raise ValueError("Need positive budgets and horizons")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty output directory")
    settings = dict(
        budgets=list(budgets),
        horizons=list(horizons),
        seeds=seeds,
        seed_start=seed_start,
        beliefs=list(beliefs),
        planners=list(planners),
        gamma=gamma,
        exploration=exploration,
        level=level,
        opponent_depth=opponent_depth,
        opponent_belief=opponent_belief,
        opponent_budget=opponent_budget,
        reference=reference,
        particles=particles,
        particle_filter=particle_filter,
        history_steps=history_steps,
        workers=workers,
        timeout=timeout,
        max_rss_mb=max_rss_mb,
    )
    lock = source_hashes(REPOSITORY_ROOT)
    (directory / "manifest.json").write_text(
        json.dumps({"settings": settings, "source_sha256": lock}, indent=2)
    )
    cases = list(
        itertools.product(
            planners, horizons, budgets, range(seed_start, seed_start + seeds), beliefs
        )
    )
    options = dict(
        gamma=gamma,
        exploration=exploration,
        level=level,
        opponent_depth=opponent_depth,
        opponent_belief=opponent_belief,
        opponent_budget=opponent_budget,
        reference=reference,
        particles=particles,
        particle_filter=particle_filter,
        history_steps=history_steps,
    )
    jobs = {i: partial(evaluate_case, *case, **options) for i, case in enumerate(cases)}
    summaries = []
    started = time.monotonic()
    try:
        for identifier, outcome in supervise_jobs(
            jobs,
            workers=workers,
            timeout_seconds=timeout,
            max_rss_mb=max_rss_mb,
            log_directory=directory / "logs",
            source_lock=(REPOSITORY_ROOT, lock),
        ):
            outcome["case"] = cases[identifier]
            (directory / f"case-{identifier}.json").write_text(
                json.dumps(outcome, indent=2, allow_nan=False)
            )
            summaries.append({k: v for k, v in outcome.items() if k not in {"rows", "traceback"}})
    finally:
        (directory / "summary.json").write_text(
            json.dumps(
                dict(elapsed_monotonic_seconds=time.monotonic() - started, cases=summaries),
                indent=2,
                allow_nan=False,
            )
        )
    if len(summaries) != len(cases) or any(row["status"] != "complete" for row in summaries):
        raise RuntimeError("Oracle panel has failed or missing cases; inspect saved outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--beliefs", type=float, nargs="+", required=True)
    parser.add_argument("--planners", choices=["mcts", "rts"], nargs="+", default=["mcts"])
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--opponent-depth", type=int)
    parser.add_argument("--opponent-belief", type=float, default=0.5)
    parser.add_argument("--opponent-budget", type=int)
    parser.add_argument("--reference", choices=["declared", "exact"], default="declared")
    parser.add_argument("--particles", type=int)
    parser.add_argument(
        "--particle-filter", choices=["compressed", "sampled"], default="compressed"
    )
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-rss-mb", type=float, default=4096)
    args = vars(parser.parse_args())
    run_oracle_comparison(args.pop("out"), **args)


if __name__ == "__main__":
    main()
