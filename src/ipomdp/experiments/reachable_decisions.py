"""Decisions at histories reachable from the common prior, every planner against the exact I-POMDP (PLAN.md S4).

Why reachable histories
-----------------------
The decision grids of the gates (``panel``, ``problems``) place each agent at
named point beliefs. I-NTMCP cannot start from those: its model class is one
opponent level under a common prior, and it builds every belief from its own
search's particles. So the compute-to-benchmark comparison (message M1) uses
problems all three planners can take: agent i's own history after ``steps``
steps from the common prior. They are also the beliefs agents reach in play.

Problems and references
-----------------------
A problem is (domain, level, horizon, seed, steps). Its history is generated
once and stored with its exact Q (``reference``):

- the world starts from the common prior;
- i acts uniformly at random among the actions that do not end the episode
  (``wumpus_qualification``'s "random" script, made domain-generic); a history
  with no such action raises ``HistoryEnded`` and the problem has no decision;
- j acts as the exact I-POMDP's own model of it (``ModeledAgent``), so every
  history has positive evidence under the reference;
- the reference is the exact I-POMDP of level ``level`` with point-mass
  nesting (j at level - 1, and so on down to uniform level 0), I-NTMCP's model
  class, so every planner solves the same declared problem.

Decisions
---------
``decide`` replays the stored history through one planner arm and scores its
policy by the first-action loss against the stored Q (ties uniform, as the
paper defines it): I-POMCP (root and modeled budgets, compressed bound N),
IPF + RTS (K sampled observations, N sampled particles) or I-NTMCP (M
simulations; ``baselines.intmcp``). CPU is the final decision's process time;
I-NTMCP also records its history replay, because it rebuilds every belief from
its own searches.

One process per job (``reference`` or ``decide``), one JSON file per result,
so runs are driven by ``xargs`` and resume by skipping existing files.

Usage:
    python -m ipomdp.experiments.reachable_decisions reference --domain uav --level 2 --horizon 3 \\
        --seed 0 --steps 2 --out refs/uav_l2_h3_s0_k2.json
    python -m ipomdp.experiments.reachable_decisions decide --problem refs/uav_l2_h3_s0_k2.json \\
        --planner mcts --budget 1000 --modeled 400 --particles 64 --seed 0 --out rows/....json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from ipomdp.baselines.rts import RTSConfig
from ipomdp.belief import canonical
from ipomdp.belief.finite_filter import UnsupportedObservation
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import REFERENCE_BRANCHES
from ipomdp.experiments.wumpus_qualification import (
    HistoryEnded,
    ModeledAgent,
    _model_at,
    nested_levels,
)
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.policy import greedy_policy
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner, build_exact_agent

GAMMA = 0.95


def make_physics(domain):
    if domain == "uav":
        from ipomdp.domains.uav import UAVModel

        return UAVModel()
    if domain == "tiger-joint":
        from ipomdp.domains.tiger import JointTigerModel

        return JointTigerModel(together_safe=40.0)
    if domain == "wumpus3":
        from ipomdp.domains.wumpus.model import WumpusModel

        return WumpusModel(3, 3, 1)
    raise ValueError(f"Unknown domain {domain!r}")


def history(physics, steps, seed, opponent):
    """i's first ``steps`` (action, observation) pairs: i uniform among non-ending actions, j as ``opponent``."""
    world = random.Random(f"reachable/{seed}")
    script = random.Random(f"reachable/script/{seed}")
    state = world.choices(*zip(*physics.initial_distribution()))[0]
    out = []
    for _ in range(steps):
        other = opponent.act(world)
        safe = [
            a
            for a in physics.get_all_actions("i")
            if not physics.is_terminal(
                physics.sample_transition(state, {"i": a, "j": other}, random.Random(0))
            )
        ]
        if not safe:
            raise HistoryEnded("Every action of i ends the episode")
        joint = {"i": script.choice(safe), "j": other}
        state = physics.sample_transition(state, joint, world)
        if physics.is_terminal(state):
            raise HistoryEnded("The episode ended during the history")
        observation = physics.sample_observation(state, joint, "i", world)
        out.append((joint["i"], observation))
        opponent.observe(other, physics.sample_observation(state, joint, "j", world))
    return out


def reference(domain, level, horizon, seed, steps, solve=True):
    """The problem's history and the exact I-POMDP's Q after it (point-mass nesting).

    ``solve=False`` stops after the history: the row has the belief's digest and support but no Q, so a study
    can drop problems whose belief another problem already has before paying for their exact solves.
    """
    if level < 2:
        raise ValueError("Reachable decisions need an intentional opponent: level >= 2")
    physics = make_physics(domain)
    agent = build_exact_agent("i", "j", level, physics, nested_levels(level), horizon, GAMMA)
    opponent = ModeledAgent(agent.solver_bank, _model_at(agent.belief, level - 1))
    steps_taken = history(physics, steps, seed, opponent)
    for action, observation in steps_taken:
        agent.update_root(action, observation)
    if not solve:
        return dict(domain=domain, level=level, horizon=horizon, seed=seed, steps=steps,
                    history=steps_taken, support=len(agent.belief.mass),
                    digest=canonical.belief_digest(agent.belief))
    started = time.process_time()
    q = agent.q_values(agent.model())
    return dict(
        domain=domain,
        level=level,
        horizon=horizon,
        seed=seed,
        steps=steps,
        history=steps_taken,
        q={str(a): v for a, v in q.items()},
        reference_cpu_seconds=time.process_time() - started,
        support=len(agent.belief.mass),
        digest=canonical.belief_digest(agent.belief),
    )


def _decoded(value):
    """JSON lists back to the tuples observations are made of."""
    return tuple(_decoded(v) for v in value) if isinstance(value, list) else value


def _planner(physics, kind, seed, budget, horizon, level, modeled, particles, particle_filter,
             modeled_exact=False):
    """i at ``level`` with point-mass nesting; every modeled level is solved by the same kind of planner.

    With ``modeled_exact``, every modeled level is instead the exact best response at ``horizon`` to the
    levels below it, on the planner's own belief filter: search only at i's root (R2, RESULTS 12.97).
    """
    if kind == "mcts":
        config = PlannerConfig(
            simulations=budget, depth=horizon, modeled_simulations=modeled, gamma=GAMMA
        )
        lower = PlannerConfig(
            simulations=modeled, depth=horizon, modeled_simulations=modeled, gamma=GAMMA
        )
        bank = SolverBank(seed, particles=particles or None, max_branches=REFERENCE_BRANCHES)
    elif kind == "rts":
        config = lower = RTSConfig(depth=horizon, observation_samples=budget, gamma=GAMMA)
        bank = SolverBank(
            seed,
            particles=particles or None,
            particle_filter=particle_filter,
            max_branches=REFERENCE_BRANCHES,
        )
    else:
        raise ValueError(f"Unknown planner {kind!r}")
    nested = nested_levels(level)
    planner = HierarchyBuilder(bank).build(
        "i",
        level,
        physics,
        "j",
        config=config,
        modeled_configs=dict.fromkeys(range(1, level), lower),
        level_weights=nested[level],
        nested_level_weights=nested,
    )
    if modeled_exact:
        reference = BestResponseReference(
            physics, bank, GAMMA, max_branches=10_000_000, cache_size=200_000
        )
        for who in ("i", "j"):
            for lower_level in range(1, level):
                key = SolverKey(who, lower_level)
                exact = ExactIPOMDPPlanner(key, physics, bank, reference, horizon)
                exact.set_initial_belief(bank.get_solver(key).initial_belief)
                bank.replace_solver(key, exact)
    return planner


def _exact_opponent_policy(physics, level, horizon):
    """j's exact policy (the reference's own model of j) as a function of j's own history."""
    agent = build_exact_agent("i", "j", level, physics, nested_levels(level), horizon, GAMMA)
    bank = agent.solver_bank
    start = _model_at(agent.belief, level - 1)
    models = {(): start}

    def model(steps):
        if steps not in models:
            action, observation = steps[-1]
            models[steps] = bank.filter.advance(model(steps[:-1]), action, observation, terminal=False)
        return models[steps]

    return lambda steps: bank.policy(model(steps))


def _loss(policy, q):
    if q is None:
        return None
    best = max(q.values())
    return best - sum(p * q[str(a)] for a, p in policy.items())


def decide(problem, kind, budget, seed, modeled=0, particles=0, particle_filter="sampled",
           variant="published", modeled_exact=False, intmcp_opponent="search"):
    """One decision of an arm on a stored problem, scored against its exact Q.

    A digest-only problem (no exact Q, beyond the exact solver's reach) gives an unscored row: its CPU and
    values, with ``scored`` false and no loss.
    """
    physics = make_physics(problem["domain"])
    level, horizon, q = problem["level"], problem["horizon"], problem.get("q")
    steps_taken = [(a, _decoded(o)) for a, o in problem["history"]]
    row = dict(
        domain=problem["domain"], level=level, horizon=horizon, problem_seed=problem["seed"],
        steps=problem["steps"], digest=problem["digest"], planner=kind, budget=budget, seed=seed,
        modeled=modeled, particles=particles, scored=q is not None,
    )
    if kind == "rts":
        row["particle_filter"] = particle_filter
    if kind == "intmcp":
        from ipomdp.baselines.intmcp import INTMCPAgent

        row["intmcp_variant"] = variant
        opponent = intmcp_opponent
        if intmcp_opponent != "search":
            row["intmcp_opponent"] = intmcp_opponent
        if intmcp_opponent == "exact":
            opponent = _exact_opponent_policy(physics, level, horizon)
        started = time.process_time()
        agent = INTMCPAgent(physics, level, budget, horizon, GAMMA, seed, variant=variant,
                            opponent=opponent)
        agent.search()
        for action, observation in steps_taken:
            agent.update_root(action, observation)
            agent.search()
        total = time.process_time() - started
        policy = greedy_policy(agent.values)
        return row | dict(
            loss=_loss(policy, q), policy={str(a): p for a, p in policy.items()},
            values={str(a): v for a, v in agent.values.items()}, cpu_seconds=agent.search_cpu_seconds,
            history_cpu_seconds=total - agent.search_cpu_seconds,
            forced_share=max(agent.forced_shares, default=0.0),
        )
    if modeled_exact:
        if kind != "mcts":
            raise ValueError("Exact modeled levels are an I-POMCP arm")
        row["modeled"] = "exact"
    planner = _planner(physics, kind, seed, budget, horizon, level, modeled or budget, particles,
                       particle_filter, modeled_exact)
    try:
        for action, observation in steps_taken:
            planner.update_root(action, observation)
        started = time.process_time()
        policy = planner.policy_for(planner.model())
    except UnsupportedObservation as error:
        # A sampled belief can lose every world that explains an observation (counted as not optimal).
        return row | dict(failure=str(error), loss=None)
    row |= dict(
        loss=_loss(policy, q), policy={str(a): p for a, p in policy.items()},
        values={str(a): v for a, v in planner.recommendation_values.items()},
        cpu_seconds=time.process_time() - started,
    )
    if kind == "mcts":
        row["exact_root"] = planner.scores.exact
    return row


def _write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--domain", required=True)
    ref.add_argument("--level", type=int, required=True)
    ref.add_argument("--horizon", type=int, required=True)
    ref.add_argument("--seed", type=int, required=True)
    ref.add_argument("--steps", type=int, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--digest-only", action="store_true",
                     help="write the history and belief digest without the exact solve")
    dec = sub.add_parser("decide")
    dec.add_argument("--problem", type=Path, required=True)
    dec.add_argument("--planner", choices=("mcts", "rts", "intmcp"), required=True)
    dec.add_argument("--budget", type=int, required=True)
    dec.add_argument("--seed", type=int, default=0)
    dec.add_argument("--modeled", type=int, default=0)
    dec.add_argument("--particles", type=int, default=0)
    dec.add_argument("--particle-filter", choices=("sampled", "smoothed"), default="sampled")
    dec.add_argument("--intmcp-variant", choices=("published", "tuned"), default="published")
    dec.add_argument("--intmcp-opponent", choices=("search", "greedy", "exact"), default="search",
                     help="diagnostic: how I-NTMCP's nested tree predicts j (R3)")
    dec.add_argument("--modeled-exact", action="store_true",
                     help="solve every modeled level exactly (I-POMCP searches only at i's root)")
    dec.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        return
    if args.command == "reference":
        try:
            row = reference(args.domain, args.level, args.horizon, args.seed, args.steps,
                            solve=not args.digest_only)
        except HistoryEnded as error:
            row = dict(domain=args.domain, level=args.level, horizon=args.horizon, seed=args.seed,
                       steps=args.steps, ended=str(error))
    else:
        problem = json.loads(args.problem.read_text())
        if problem.get("ended"):
            return
        started = time.process_time()
        row = decide(problem, args.planner, args.budget, args.seed, args.modeled, args.particles,
                     args.particle_filter, args.intmcp_variant, args.modeled_exact,
                     args.intmcp_opponent)
        row["total_cpu_seconds"] = time.process_time() - started
    _write(args.out, row)


if __name__ == "__main__":
    main()
