"""Matched comparisons of planners against exact Tiger references.

Each case fixes one decision problem (belief, physics, discount, horizon and
opponent model) and asks a planner for its root policy and action values. The
same problem is solved exactly, and the row records:

- ``first_action_loss`` = V*(b) - sum_a pi(a | b) Q*(b, a): the expected value
  lost by the planner's root decision when every later decision is optimal. It
  handles ties exactly and is not an episode regret.
- ``max_abs_q_error`` = max_a |Q_hat(b, a) - Q*(b, a)|.

For I-POMCP, Q_hat is the post-search chance-scored value that decides
(``planning.chance_scoring``). The same search's empirical-backup values are
recorded beside it (``search_q``, ``search_first_action_loss``,
``search_max_abs_q_error``), so the two are compared on identical trees, and
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
):
    """Solve one matched problem with the planner and the exact reference.

    ``reference`` matters only for a level-2 problem with a finite opponent:
    "declared" scores the planner against the best response to its own finite
    opponent model (search error only); "exact" scores it against the exact
    I-POMDP, where the opponent is the exact L1 agent (search plus model error).
    """
    _validate_problem(
        planner_kind, level, opponent_depth, opponent_belief, opponent_budget, reference, particles
    )
    if history_steps and level == 1:
        raise ValueError("The L1 alpha-vector reference is solved at the prior; use level 2")
    model = TigerModel()
    bank = SolverBank(seed, particles=particles, particle_filter=particle_filter)
    key = SolverKey("i", level)
    config = _planner_config(planner_kind, budget, horizon, gamma, exploration)
    planner = _planner(planner_kind, key, model, bank, config)
    bank.register_solver(key, planner)
    # L0 action streams are never drawn in a single root decision; seed them anyway.
    for agent in ("i", "j"):
        rng = random.Random(bank.stream_seed("actions", f"{agent}_L0"))
        bank.register_solver(SolverKey(agent, 0), RandomPlanner(model.get_all_actions(agent), rng))
    best_response = BestResponseReference(model, bank, gamma)
    if level == 2:
        if opponent_budget is None:
            provider = ExactIPOMDPPlanner(
                SolverKey("j", 1), model, bank, best_response, opponent_depth
            )
        else:
            provider = _planner(
                planner_kind,
                SolverKey("j", 1),
                model,
                bank,
                _planner_config(planner_kind, opponent_budget, opponent_depth, gamma, exploration),
            )
        bank.register_solver(SolverKey("j", 1), provider)
    belief = _two_state_belief(model, level, belief_p, opponent_belief)
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
            model, prior_model, history, seed, gamma, opponent_depth, horizon
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
            "scores_exact": planner.scores.exact,
            "fallback_actions": planner.scores.fallback_actions,
        }
    return [
        {
            "planner": planner_kind,
            "planner_config": asdict(config),
            "level": level,
            "opponent": _opponent_description(
                level, opponent_depth, opponent_belief, opponent_budget
            ),
            "reference": reference if level == 2 and opponent_budget is not None else "exact",
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
            "planner_cpu_seconds": planner_cpu_seconds,
            **search,
        }
    ]


def _planner_config(planner_kind, budget, depth, gamma, exploration):
    if planner_kind == "mcts":
        return PlannerConfig(
            simulations=budget,
            depth=depth,
            modeled_simulations=budget,
            gamma=gamma,
            exploration=exploration,
        )
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


def _exact_ipomdp_q(model, prior_model, history, seed, gamma, opponent_depth, horizon):
    """Q of the exact I-POMDP at the exact posterior after ``history``, j exact L1."""
    bank = SolverBank(seed)
    best_response = BestResponseReference(model, bank, gamma, max_branches=REFERENCE_BRANCHES)
    rng = random.Random(bank.stream_seed("actions", "i_L0"))
    bank.register_solver(SolverKey("i", 0), RandomPlanner(model.get_all_actions("i"), rng))
    bank.register_solver(
        SolverKey("j", 1),
        ExactIPOMDPPlanner(SolverKey("j", 1), model, bank, best_response, opponent_depth),
    )
    agent_model = prior_model
    for action, observation in history:
        posterior = bank.filter.update(agent_model, action, observation, terminal=False).belief
        agent_model = MentalModel(agent_model.frame, posterior)
    return best_response.q_values(agent_model, horizon)


def _two_state_belief(model, level, belief_p, opponent_belief):
    """i's exact two-state prior, with j uniform L0 (level 1) or an L1 point model."""
    if level == 1:
        opponent = MentalModel(AgentFrame("j", 0, model))
    else:
        l0 = MentalModel(AgentFrame("i", 0, model))
        private = FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, l0), opponent_belief),
                (InteractiveState(TIGER_RIGHT, l0), 1 - opponent_belief),
            )
        )
        opponent = MentalModel(AgentFrame("j", 1, model), private)
    return FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, opponent), belief_p),
            (InteractiveState(TIGER_RIGHT, opponent), 1 - belief_p),
        )
    )


def _opponent_description(level, depth, belief, budget):
    if level == 1:
        return {"policy": "uniform_l0"}
    if budget is None:
        return {"policy": "exact_l1_fixed_depth", "depth": depth, "initial_belief_p": belief}
    return {
        "policy": "same_planner_l1",
        "depth": depth,
        "budget": budget,
        "initial_belief_p": belief,
    }


def _validate_problem(
    planner_kind, level, opponent_depth, opponent_belief, opponent_budget, reference, particles
):
    if planner_kind not in ("mcts", "rts"):
        raise ValueError("Unknown planner")
    if reference not in ("declared", "exact"):
        raise ValueError("reference must be 'declared' or 'exact'")
    if particles is not None and reference == "declared" and opponent_budget is not None:
        raise ValueError("A declared reference is not exact with particle beliefs; use 'exact'")
    if level not in (1, 2):
        raise ValueError("Reference level must be 1 or 2")
    if not 0 <= opponent_belief <= 1:
        raise ValueError("Opponent belief must lie in [0,1]")
    if level == 1 and (opponent_depth is not None or opponent_budget is not None):
        raise ValueError("Level 1 has a uniform L0 opponent; it takes no opponent depth or budget")
    if level == 2:
        if type(opponent_depth) is not int or opponent_depth < 1:
            raise ValueError("Level 2 requires an explicit positive opponent depth")


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
