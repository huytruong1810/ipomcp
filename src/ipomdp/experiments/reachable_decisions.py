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

Wide prior (T4)
---------------
With ``prior`` = {"kind": "wide", "M": M, "w0": w0} (``--prior-m``, ``--prior-w0``), i's
initial belief spreads over M level-1 models of j with different initial beliefs
(``wide_prior``), and j's true model is drawn from that prior. The problem file stores
the prior and its digest; ``decide`` rebuilds it and checks the digest. I-NTMCP cannot
take such a prior (its model class fixes one common prior per level), so it refuses.

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
from ipomdp.experiments import wide_prior
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import REFERENCE_BRANCHES
from ipomdp.experiments.wumpus_qualification import (
    HistoryEnded,
    ModeledAgent,
    _model_at,
    nested_levels,
)
from ipomdp.planning.config import (
    FRONTIER_BOUNDS,
    PREDICTIONS,
    LEAF_EVALS,
    SELECTIONS,
    SCORING,
    UNSAMPLED_VALUES,
    PlannerConfig,
)
from ipomdp.planning.chance_scoring import compression_bound, deviation_bound
from ipomdp.planning.policy import TIE_TOLERANCE, greedy_policy
from ipomdp.planning.stopping import ModeledResolution
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.shared_horizon import SharedHorizonReference
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
    if domain == "wumpus4":
        from ipomdp.domains.wumpus.model import WumpusModel

        return WumpusModel(4, 4, 2)
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


def _check_prior(prior, level):
    if prior is None:
        return None
    if prior.get("kind") != "wide":
        raise ValueError(f"Unknown prior {prior!r}")
    if level != 2:
        raise ValueError("The wide prior is defined for i at level 2 (j at level 1 or 0)")
    return int(prior["M"]), float(prior.get("w0", 0.0))


def _wide_rows(bank, physics, prior):
    """i's wide prior in the bank filter's representation (exact, compressed or particles)."""
    m_count, w0 = prior
    return bank.filter.initial_belief(wide_prior.prior_rows(
        physics, m_count, w0, bank.filter.initial_belief, bank.physical_prior(physics)))


def reference(domain, level, horizon, seed, steps, solve=True, cache_size=200_000, prior=None,
              given_history=None, opponent_horizon="fixed"):
    """The problem's history and the exact I-POMDP's Q after it (point-mass nesting, or ``prior``).

    ``given_history`` replays a stored history of i instead of generating one (the same history under
    several priors; it must have positive evidence under this prior, which the update checks).

    ``solve=False`` stops after the history: the row has the belief's digest and support but no Q, so a study
    can drop problems whose belief another problem already has before paying for their exact solves.

    ``opponent_horizon="shared"`` also stores ``q_shared``: the exact Q when the modeled agent plays, at r
    steps to go, its own r-step law (``reference.shared_horizon``; opponents of level <= 1).
    """
    if level < 2:
        raise ValueError("Reachable decisions need an intentional opponent: level >= 2")
    physics = make_physics(domain)
    wide = _check_prior(prior, level)
    # The branch guard only raises (values are unchanged); the 4 x 4 Wumpus prior needs the planner's guard.
    agent = build_exact_agent("i", "j", level, physics, nested_levels(level), horizon, GAMMA,
                              cache_size=cache_size, max_branches=REFERENCE_BRANCHES)
    extra = {}
    if wide is None:
        start = _model_at(agent.belief, level - 1)
    else:
        bank = agent.solver_bank
        agent.set_initial_belief(_wide_rows(bank, physics, wide))
        start = wide_prior.true_opponent(physics, *wide, seed, bank.filter.initial_belief)
        extra = dict(prior=dict(prior), prior_digest=wide_prior.prior_digest(physics, *wide),
                     true_opponent_level=start.frame.level,
                     true_opponent_p=None if start.belief is None else _p_first(physics, start.belief))
    if given_history is None:
        steps_taken = history(physics, steps, seed, ModeledAgent(agent.solver_bank, start))
    else:
        steps_taken = [(a, _decoded(o)) for a, o in given_history]
        steps = len(steps_taken)
        extra.pop("true_opponent_level", None), extra.pop("true_opponent_p", None)
        extra["history_given"] = True
    for action, observation in steps_taken:
        agent.update_root(action, observation)
    if not solve:
        return dict(domain=domain, level=level, horizon=horizon, seed=seed, steps=steps,
                    history=steps_taken, support=len(agent.belief.mass),
                    digest=canonical.belief_digest(agent.belief)) | extra
    started = time.process_time()
    q = agent.q_values(agent.model())
    if opponent_horizon == "shared":
        shared = SharedHorizonReference(physics, agent.solver_bank, GAMMA, max_branches=REFERENCE_BRANCHES)
        extra["q_shared"] = {str(a): v for a, v in shared.q_values(agent.model(), horizon).items()}
        extra["opponent_horizon"] = "shared"
    elif opponent_horizon != "fixed":
        raise ValueError("opponent_horizon must be 'fixed' or 'shared'")
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
    ) | extra


def _p_first(physics, belief):
    """A level-1 model's initial probability of the first state of the physical prior."""
    first = tuple(physics.initial_distribution())[0][0]
    return sum(w for atom, w in belief.mass if atom.state == first) / sum(w for _, w in belief.mass)


def _decoded(value):
    """JSON lists back to the tuples observations are made of."""
    return tuple(_decoded(v) for v in value) if isinstance(value, list) else value


def _planner(physics, kind, seed, budget, horizon, level, modeled, particles, particle_filter,
             modeled_exact=False, cache_size=200_000, prior=None, rts_published=False, certified=None,
             modeled_options=None, compression="stratified"):
    """i at ``level`` with point-mass nesting; every modeled level is solved by the same kind of planner.

    With ``modeled_exact``, every modeled level is instead the exact best response at ``horizon`` to the
    levels below it, on the planner's own belief filter: search only at i's root (R2, RESULTS 12.97).
    ``prior`` (``_check_prior``'s (M, w0)) replaces i's root belief by the wide prior in the bank's
    representation.

    ``rts_published`` (IPF + RTS only) is the algorithm as Doshi and Gmytrasiewicz (2009) publish it
    (RESULTS 12.116): their interactive particle filter (exact bottom level, enumerated observations of the other
    agent; "published-smoothed" when ``particle_filter`` is "smoothed"), their RTS backup, and every level-1 model
    (their level 0) solved exactly.

    ``certified`` (I-POMCP only): ``PlannerConfig`` options of the certified planner (audit spec 3.0) for
    i's own decision; ``modeled_options`` (I-POMCP only) are those of every modeled level (certified search,
    ``modeled_resolution``, ``modeled_prediction``). Both default to today's configuration.
    """
    if certified and kind != "mcts":
        raise ValueError("Certified options are I-POMCP arms")
    if kind == "mcts":
        config = PlannerConfig(
            simulations=budget, depth=horizon, modeled_simulations=modeled, gamma=GAMMA,
            **(certified or {}),
        )
        lower = PlannerConfig(
            simulations=modeled, depth=horizon, modeled_simulations=modeled, gamma=GAMMA,
            **(modeled_options or {}),
        )
        bank = SolverBank(seed, particles=particles or None, max_branches=REFERENCE_BRANCHES,
                          compression=compression)
    elif kind == "rts":
        backup = "published" if rts_published else "sparse"
        config = lower = RTSConfig(depth=horizon, observation_samples=budget, gamma=GAMMA, backup=backup)
        if rts_published:
            particle_filter = "published-smoothed" if particle_filter == "smoothed" else "published"
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
            physics, bank, GAMMA, max_branches=10_000_000, cache_size=cache_size
        )
        for who in ("i", "j"):
            for lower_level in range(1, level):
                key = SolverKey(who, lower_level)
                exact = ExactIPOMDPPlanner(key, physics, bank, reference, horizon)
                exact.set_initial_belief(bank.get_solver(key).initial_belief)
                bank.replace_solver(key, exact)
    if rts_published:
        if kind != "rts":
            raise ValueError("The published configuration is an IPF + RTS arm")
        reference = BestResponseReference(physics, bank, GAMMA, max_branches=10_000_000, cache_size=cache_size)
        for who in ("i", "j"):
            key = SolverKey(who, 1)
            if level > 1:
                exact = ExactIPOMDPPlanner(key, physics, bank, reference, horizon)
                exact.set_initial_belief(bank.get_solver(key).initial_belief)
                bank.replace_solver(key, exact)
    if prior is not None:
        planner.set_initial_belief(_wide_rows(bank, physics, prior))
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
           variant="published", modeled_exact=False, intmcp_opponent="search", ladder=None,
           cache_size=200_000, rts_published=False, certified=None, modeled_options=None,
           compression="stratified", intmcp_uct_c=None, intmcp_record=False):
    """One decision of an arm on a stored problem, scored against its exact Q.

    ``ladder`` (I-POMCP only): budgets below ``budget`` at which the same tree is also scored, each stage
    recorded under ``ladder`` with its certified intervals (V2 coverage); the row's own fields are the stage
    at ``budget``, identical to a decision without a ladder.

    ``certified``: non-default ``PlannerConfig`` options of i's own planner (``_planner``), recorded in the row.
    ``modeled_options``: the same for every modeled level. With them the row also carries Theorem 1b's
    deviation bound per root action, the widened intervals and their containment, and ``model_certified``
    (no modeled solve ended unresolved, so the searched model is the exact I-POMDP).
    ``compression="head-tail"`` (I-POMCP, with ``particles``): head-tail compression of every belief; the row
    carries the compression slack per root action and the intervals widened by it (audit spec 2.3).

    A digest-only problem (no exact Q, beyond the exact solver's reach) gives an unscored row: its CPU and
    values, with ``scored`` false and no loss.
    """
    physics = make_physics(problem["domain"])
    level, horizon, q = problem["level"], problem["horizon"], problem.get("q")
    wide = _check_prior(problem.get("prior"), level)
    if wide is not None and wide_prior.prior_digest(physics, *wide) != problem["prior_digest"]:
        raise ValueError("The rebuilt wide prior does not match the problem's prior digest")
    steps_taken = [(a, _decoded(o)) for a, o in problem["history"]]
    row = dict(
        domain=problem["domain"], level=level, horizon=horizon, problem_seed=problem["seed"],
        steps=problem["steps"], digest=problem["digest"], planner=kind, budget=budget, seed=seed,
        modeled=modeled, particles=particles, scored=q is not None,
    )
    if kind == "rts":
        row["particle_filter"] = particle_filter
        if rts_published:
            row["rts_variant"] = "published"
    if wide is not None:
        row["prior"] = problem["prior"]
        if kind == "intmcp":
            raise ValueError("I-NTMCP cannot start from a wide prior: its model class fixes one "
                             "common prior per nesting level")
    if kind == "intmcp":
        from ipomdp.baselines.intmcp import INTMCPAgent

        row["intmcp_variant"] = variant
        opponent = intmcp_opponent
        if intmcp_opponent != "search":
            row["intmcp_opponent"] = intmcp_opponent
        if intmcp_opponent == "exact":
            opponent = _exact_opponent_policy(physics, level, horizon)
        started = time.process_time()
        if intmcp_uct_c is not None:
            row["intmcp_uct_c"] = intmcp_uct_c
        record = None
        if intmcp_record and intmcp_opponent == "search":
            record = _exact_opponent_policy(physics, level, horizon)
        stride = max(1, budget // 1000)  # about 20,000 recorded queries per decision at any budget
        if record is not None:
            row["intmcp_record_stride"] = stride
        agent = INTMCPAgent(physics, level, budget, horizon, GAMMA, seed, variant=variant,
                            opponent=opponent, uct_c=intmcp_uct_c, record_predictions=record,
                            record_stride=stride)
        agent.search()
        for action, observation in steps_taken:
            agent.update_root(action, observation)
            agent.search()
        total = time.process_time() - started
        policy = greedy_policy(agent.values)
        if record is not None:
            row["intmcp_predictions"] = agent.prediction_summary()
        if "q_shared" in problem:
            row["loss_shared"] = _loss(policy, problem["q_shared"])
        return row | dict(
            loss=_loss(policy, q), policy={str(a): p for a, p in policy.items()},
            values={str(a): v for a, v in agent.values.items()}, cpu_seconds=agent.search_cpu_seconds,
            history_cpu_seconds=total - agent.search_cpu_seconds,
            forced_share=max(agent.forced_shares, default=0.0),
        )
    defaults = PlannerConfig(1, 1, 1)
    certified = {k: v for k, v in (certified or {}).items() if getattr(defaults, k) != v}
    if certified:
        row["certified"] = certified
    if compression != "stratified":
        if kind != "mcts" or not particles:
            raise ValueError("Head-tail compression is an I-POMCP arm with particles")
        row["compression"] = compression
    modeled_options = {k: v for k, v in (modeled_options or {}).items() if getattr(defaults, k) != v}
    if modeled_options:
        if modeled_exact:
            raise ValueError("Modeled options need searched modeled levels")
        row["modeled_options"] = {k: (str(v) if k == "modeled_resolution" else v)
                                  for k, v in modeled_options.items()}
    if modeled_exact:
        if kind != "mcts":
            raise ValueError("Exact modeled levels are an I-POMCP arm")
        row["modeled"] = "exact"
    planner = _planner(physics, kind, seed, budget, horizon, level, modeled or budget, particles,
                       particle_filter, modeled_exact, cache_size, wide, rts_published, certified,
                       modeled_options, compression)
    try:
        for action, observation in steps_taken:
            planner.update_root(action, observation)
        started = time.process_time()
        policy = planner.policy_for(planner.model())
        cpu_seconds = time.process_time() - started
        if ladder:
            if kind != "mcts":
                raise ValueError("A budget ladder is an I-POMCP arm")
            stages = planner.anytime_scores(planner.model(), sorted(set(ladder) | {budget}))
    except UnsupportedObservation as error:
        # A sampled belief can lose every world that explains an observation (counted as not optimal).
        return row | dict(failure=str(error), loss=None)
    if "q_shared" in problem:
        row["loss_shared"] = _loss(policy, problem["q_shared"])
    row |= dict(
        loss=_loss(policy, q), policy={str(a): p for a, p in policy.items()},
        values={str(a): v for a, v in planner.recommendation_values.items()},
        cpu_seconds=cpu_seconds, root_support=len(planner.belief.mass),
    )
    if ladder:
        row["ladder"] = [_stage(simulations, scores, cpu, q) | work for simulations, scores, cpu, work in stages]
    if kind == "mcts":
        bounds = planner.scores.bounds
        row |= dict(exact_root=planner.scores.exact, bounds={str(a): b for a, b in bounds.items()},
                    bound_width=max(hi - lo for lo, hi in bounds.values()))
        if certified:
            row |= _heuristic(planner.scores)
        if compression != "stratified":
            slack = compression_bound(planner.root, planner.model(), horizon, planner.solver_bank,
                                      planner._history_successor, GAMMA)
            losses = planner.solver_bank.search_filter._losses
            row |= dict(compression_slack={str(a): v for a, v in slack.items()},
                        max_loss=max(losses.values(), default=0.0),
                        bounds_compression_widened={str(a): (lo - slack[a], hi + slack[a])
                                                    for a, (lo, hi) in bounds.items()})
            # The theorem's target: the searched model (this bank's opponent laws and nested filter, this root
            # belief) without compressing i's own belief below the root, which the reference enumerates.
            searched = BestResponseReference(physics, planner.solver_bank, GAMMA).q_values(planner.model(), horizon)
            row |= dict(q_searched_uncompressed={str(a): v for a, v in searched.items()},
                        bounds_compression_widened_contain_searched=all(
                            lo - slack[a] - 1e-9 <= searched[a] <= hi + slack[a] + 1e-9
                            for a, (lo, hi) in bounds.items()))
            if q is not None:
                row["bounds_compression_widened_contain_reference"] = all(
                    lo - slack[a] - 1e-9 <= q[str(a)] <= hi + slack[a] + 1e-9 for a, (lo, hi) in bounds.items())
        if modeled_options:
            bank = planner.solver_bank
            deviation = deviation_bound(planner.root, planner.model(), horizon, bank,
                                        planner._history_successor, GAMMA)
            widened = {a: (lo - deviation[a], hi + deviation[a]) for a, (lo, hi) in bounds.items()}
            row |= dict(deviation={str(a): v for a, v in deviation.items()},
                        bounds_widened={str(a): b for a, b in widened.items()},
                        model_certified=not bank.unresolved_levels,
                        unresolved_levels=sorted(bank.unresolved_levels))
            if q is not None:
                row["bounds_widened_contain_reference"] = all(
                    lo - 1e-9 <= q[str(a)] <= hi + 1e-9 for a, (lo, hi) in widened.items())
        if q is not None:
            # Certified for the searched model; they must contain q when that model is the exact one
            # (exact modeled levels and exact beliefs).
            row["bounds_contain_reference"] = all(
                lo - 1e-9 <= q[str(a)] <= hi + 1e-9 for a, (lo, hi) in bounds.items())
    return row


def _stage(simulations, scores, cpu_seconds, q):
    """One budget of a ladder: values, certified intervals and, when scored, loss and containment."""
    bounds = scores.bounds
    stage = dict(budget=simulations, cpu_seconds=cpu_seconds, exact_root=scores.exact,
                 values={str(a): v for a, v in scores.values.items()},
                 bounds={str(a): b for a, b in bounds.items()},
                 bound_width=max(hi - lo for lo, hi in bounds.values()))
    if scores.heuristic_branches:
        stage |= _heuristic(scores)
    if q is not None:
        stage["loss"] = _loss(greedy_policy(scores.values, TIE_TOLERANCE if scores.exact else 0.0), q)
        stage["bounds_contain_reference"] = all(
            lo - 1e-9 <= q[str(a)] <= hi + 1e-9 for a, (lo, hi) in bounds.items())
    return stage


def certified_options(args):
    """``PlannerConfig`` options of i's own planner from the ``decide`` flags."""
    options = dict(frontier_bounds=args.frontier_bounds, scoring=args.scoring,
                   unsampled_value=args.unsampled, exact_tail=args.exact_tail, prune=args.prune,
                   selection=args.selection, leaf_eval=args.leaf_eval, lazy_expansion=args.lazy)
    if args.profile == "certified":
        options |= dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                        leaf_eval="bound", lazy_expansion=True)
    return options


def modeled_options(args):
    """``PlannerConfig`` options of every modeled level from the ``decide`` flags.

    ``--profile certified`` gives modeled levels the certified search of i's own decision (selection
    included), resolution to the ``--modeled`` cap, and plausible prediction.
    """
    options = {}
    if args.modeled_exact:
        return options  # exact modeled levels: nothing to search or resolve
    cap = args.modeled or args.budget
    if args.modeled_resolve or args.profile == "certified":
        options["modeled_resolution"] = ModeledResolution(pilot=min(args.modeled_pilot, cap))
        options["modeled_prediction"] = args.modeled_prediction
    if args.profile == "certified":
        options |= {k: v for k, v in certified_options(args).items()} | {"modeled_prediction": "plausible"}
    return options


def _heuristic(scores):
    return dict(heuristic_branches=scores.heuristic_branches, heuristic_mass=scores.heuristic_mass)


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
    ref.add_argument("--reference-cache", type=int, default=200_000,
                     help="entries of the exact reference's caches (memory against CPU; values unchanged)")
    ref.add_argument("--prior-m", type=int,
                     help="wide prior: M level-1 models of j on a midpoint grid of initial beliefs")
    ref.add_argument("--prior-w0", type=float, default=0.0,
                     help="wide prior: weight of a level-0 (uniformly random) j")
    ref.add_argument("--history-from", type=Path,
                     help="replay the history stored in this problem file instead of generating one")
    ref.add_argument("--opponent-horizon", choices=("fixed", "shared"), default="fixed",
                     help="shared: also store q_shared, the modeled agent planning its remaining steps only")
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
    dec.add_argument("--reference-cache", type=int, default=200_000,
                     help="with --modeled-exact: entries of the exact solver's caches (memory against CPU)")
    dec.add_argument("--rts-published", action="store_true",
                     help="IPF + RTS exactly as published (RESULTS 12.116): their filter, backup and exact level 0")
    dec.add_argument("--ladder", type=int, nargs="+",
                     help="I-POMCP: also score the same tree at these smaller budgets (anytime intervals)")
    dec.add_argument("--frontier-bounds", choices=FRONTIER_BOUNDS, default="scalar",
                     help="I-POMCP certified intervals: scalar (default) or belief-dependent I-FIB")
    dec.add_argument("--scoring", choices=SCORING, default="legacy",
                     help="I-POMCP scoring of unsampled branches: legacy (default) or probability-weighted")
    dec.add_argument("--unsampled", choices=UNSAMPLED_VALUES, default="lower",
                     help="stand-in of an unsampled branch under --scoring pw")
    dec.add_argument("--exact-tail", action="store_true",
                     help="certified search: solve two-steps-to-go histories exactly when expanded")
    dec.add_argument("--prune", action="store_true",
                     help="certified search: bounds at every history, action elimination, early stop")
    dec.add_argument("--selection", choices=SELECTIONS, default="ucb")
    dec.add_argument("--leaf-eval", choices=LEAF_EVALS, default="rollout")
    dec.add_argument("--lazy", action="store_true", help="certified search: lazy frontier expansion")
    dec.add_argument("--profile", choices=("certified",),
                     help="certified: --frontier-bounds ifib --scoring pw --exact-tail --prune "
                          "--leaf-eval bound --lazy (selection stays as given)")
    dec.add_argument("--modeled-resolve", action="store_true",
                     help="modeled levels: deterministic resolution to the --modeled cap (Theorem 1b)")
    dec.add_argument("--modeled-pilot", type=int, default=50, help="first resolution checkpoint")
    dec.add_argument("--modeled-prediction", choices=PREDICTIONS, default="argmax")
    dec.add_argument("--compression", choices=("stratified", "head-tail"), default="stratified",
                     help="I-POMCP with --particles: how beliefs are compressed (head-tail records its loss)")
    dec.add_argument("--intmcp-uct-c", type=float, help="I-NTMCP exploration constant (default: theirs)")
    dec.add_argument("--intmcp-record", action="store_true",
                     help="I-NTMCP: log the nested prediction against the exact law of j (search opponent only)")
    dec.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        return
    if args.command == "reference":
        try:
            prior = (None if args.prior_m is None
                     else dict(kind="wide", M=args.prior_m, w0=args.prior_w0))
            given = (None if args.history_from is None
                     else json.loads(args.history_from.read_text())["history"])
            row = reference(args.domain, args.level, args.horizon, args.seed, args.steps,
                            solve=not args.digest_only, cache_size=args.reference_cache, prior=prior,
                            given_history=given, opponent_horizon=args.opponent_horizon)
        except HistoryEnded as error:
            row = dict(domain=args.domain, level=args.level, horizon=args.horizon, seed=args.seed,
                       steps=args.steps, ended=str(error))
            if prior is not None:
                row["prior"] = prior
    else:
        problem = json.loads(args.problem.read_text())
        if problem.get("ended"):
            return
        started = time.process_time()
        row = decide(problem, args.planner, args.budget, args.seed, args.modeled, args.particles,
                     args.particle_filter, args.intmcp_variant, args.modeled_exact,
                     args.intmcp_opponent, args.ladder, args.reference_cache, args.rts_published,
                     certified_options(args), modeled_options(args), args.compression,
                     args.intmcp_uct_c, args.intmcp_record)
        row["total_cpu_seconds"] = time.process_time() - started
    _write(args.out, row)


if __name__ == "__main__":
    main()
