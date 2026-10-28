"""Matched comparisons of planners against exact references.

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
- Levels 2-4 with exact opponents: every intentional level below i is an
  exact agent replanning at a fixed depth, with point priors on the private
  beliefs. The reference is the exact I-POMDP.
- Levels 2-4 with finite opponents: ``modeled`` names how each level below i
  is solved (``modeled_levels``: per level, the same kind of planner as i with
  its own budget and optional stopping rule, or exact; RTS samples per node as
  in IPF + RTS, where every level is solved by RTS). The reference is either
  the exact best response to that declared hierarchy (``declared``: search
  error only) or the exact I-POMDP (``exact``: search plus opponent-model
  error).

Physics: the published Tiger (``domains.tiger.TigerModel``) by default, or
any domain with a problem ``chain`` (``experiments.problems``): Tiger's payoff
variants take the default point beliefs; other domains (UAV reconnaissance)
give a chain and, for mid-episode problems, their own ``history_actions``.

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

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import LISTEN, OPEN_LEFT, TigerModel
from ipomdp.experiments.modeled_levels import (
    EXACT,
    make_planner,
    parse_stack,
    planner_config,
    register_levels,
    stack_label,
)
from ipomdp.experiments.problems import chain_model, tiger_chain
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes
from ipomdp.planning.policy import greedy_policy
from ipomdp.planning.solver_bank import FILTERS, SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
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
    modeled=EXACT,
    reference="declared",
    particles=None,
    particle_filter="compressed",
    history_steps=0,
    nested_beliefs=None,
    opponent_observer=None,
    physics=None,
    chain=None,
    problem=None,
    history_actions=None,
):
    """Solve one matched problem with the planner and the exact reference.

    ``modeled`` (levels 2-4) labels the modeled stack, one solve per level
    below i, top-down (``modeled_levels``): ``exact``, ``400``, ``6400s/400``.

    ``reference="none"`` returns the planner's decision, values and CPU without
    solving a reference: for drivers that need i's search but not its score,
    where the exact reference is intractable (``modeled_qualification`` at L3
    beyond H3; RESULTS.md Section 12.7).

    Otherwise ``reference`` matters only for a level-2 to level-4 problem with a finite
    modeled level: "declared" scores the planner against the best response to its
    own finite opponent models (search error only); "exact" scores it against
    the exact I-POMDP, where every lower level is exact (search plus model
    error).

    At level L >= 3, j is a level-(L-1) agent whose private point belief is
    ``opponent_belief``; ``nested_beliefs`` gives the private point beliefs of
    the L - 2 models below j, top-down (at level 3, the level-1 i that j models
    against uniform L0 j). None puts 0.5 at every nested level.

    ``physics`` is the Tiger model every agent shares (None: the published
    game); a payoff variant changes only rewards, so the problems are the same.

    ``opponent_observer`` (levels 2-4) is called as ``observer(model, policy)``
    for every model of the opponent (j at level - 1) whose modeled policy the
    planner requests, with the policy it got. These are the beliefs at which the
    opponent's budget must be qualified (``experiments.modeled_qualification``);
    the bank's policy cache means each distinct model is reported once.

    ``chain`` (``experiments.problems``) replaces the Tiger point beliefs
    (``belief_p``, ``opponent_belief``, ``nested_beliefs``) for any domain: one
    physical belief per level, top-down, each a function of the enclosing state.
    ``problem`` labels such a problem in the row. Level-1 problems on Tiger are
    scored by alpha-vector value iteration; on other domains by the exact best
    response to uniform L0 (``BestResponseReference``), which is the same
    exact value by enumeration. ``history_actions`` are i's actions for
    ``history_steps`` (default ``HISTORY_ACTIONS``, Tiger's).
    """
    stack = _validate_problem(
        planner_kind, level, opponent_depth, opponent_belief, modeled, reference, particles
    )
    if history_steps and level == 1:
        raise ValueError("The L1 alpha-vector reference is solved at the prior; use level 2")
    if opponent_observer is not None and level == 1:
        raise ValueError("Level 1 has a uniform L0 opponent; there is no modeled policy to observe")
    nested_beliefs = _nested_chain(level, nested_beliefs)
    model = TigerModel() if physics is None else physics
    bank = SolverBank(seed, particles=particles, particle_filter=particle_filter)
    key = SolverKey("i", level)
    config = planner_config(planner_kind, budget, horizon, gamma, exploration)
    planner = make_planner(planner_kind, key, model, bank, config)
    bank.register_solver(key, planner)
    best_response = BestResponseReference(model, bank, gamma)

    def observed(key_below, solver):
        if opponent_observer is not None and key_below.level == level - 1:
            return _ObservedSolver(solver, opponent_observer)
        return solver

    register_levels(
        bank,
        model,
        "i",
        level,
        stack,
        opponent_depth,
        best_response,
        planner_kind=planner_kind,
        gamma=gamma,
        exploration=exploration,
        wrap=observed,
    )
    if chain is None:
        chain = tiger_chain(belief_p, opponent_belief, nested_beliefs)
        tiger_problem = True
    else:
        tiger_problem = False
    belief = chain_model(model, "i", level, chain).belief
    planner.set_initial_belief(belief)
    prior_model = planner.model()
    history = sample_history(
        model,
        history_steps,
        random.Random(f"history/{seed}/{belief_p}/{opponent_belief}"),
        HISTORY_ACTIONS if history_actions is None else history_actions,
    )
    started = time.process_time()
    for action, observation in history:
        planner.update_root(action, observation)
    update_cpu_seconds = time.process_time() - started
    started = time.process_time()
    policy = planner.policy_for(planner.model())
    planner_cpu_seconds = time.process_time() - started
    estimates = dict(planner.recommendation_values)
    row = {
        "planner": planner_kind,
        "planner_config": asdict(config),
        "level": level,
        "opponent": _opponent_description(
            level, opponent_depth, opponent_belief, stack, nested_beliefs
        ),
        "reference": reference if any(stack) else "exact",
        "particles": particles,
        "particle_filter": particle_filter if particles else None,
        "history_steps": history_steps,
        "belief_support": len(planner.belief.mass),
        "history_update_cpu_seconds": update_cpu_seconds,
        "horizon": horizon,
        "budget": budget,
        "seed": seed,
        "belief_p": belief_p if tiger_problem else None,
        "problem": problem,
        "gamma": gamma,
        "estimated_q": estimates,
        "policy": policy,
        "planner_cpu_seconds": planner_cpu_seconds,
    }
    if reference == "none":
        return [row]
    started = time.process_time()
    if level == 1 and tiger_problem and isinstance(model, TigerModel):
        oracle_q = ExactPOMDPSolver(model, horizon=horizon, gamma=gamma).q_values(belief_p, horizon)
    elif level == 1:
        oracle_q = best_response.q_values(prior_model, horizon)
    elif reference == "exact" or not any(stack):
        oracle_q = _exact_ipomdp_q(
            model, prior_model, history, seed, gamma, opponent_depth, horizon, level
        )
    else:
        oracle_q = best_response.q_values(planner.model(), horizon)
    oracle_cpu_seconds = time.process_time() - started
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
            # Certified for the searched model (chance_scoring); they contain
            # oracle_q whenever the reference is that model: level 1, a
            # declared reference, or exact opponents, with exact beliefs.
            "bound_width": max(hi - lo for lo, hi in planner.scores.bounds.values()),
            "bounds_contain_reference": all(
                lo - 1e-9 <= oracle_q[a] <= hi + 1e-9
                for a, (lo, hi) in planner.scores.bounds.items()
            ),
        }
    return [
        row
        | {
            "oracle_q": oracle_q,
            "oracle_cpu_seconds": oracle_cpu_seconds,
            "first_action_loss": loss(policy),
            "max_abs_q_error": q_error(estimates),
            "optimal_q_error": optimal_q_error(estimates, oracle_q),
            **search,
        }
    ]


OPTIMALITY_TOLERANCE = 1e-9
# Certification standard (THEORY.md): a decision is epsilon-correct when its
# first-action loss is at most this, in reward units.
CERTIFICATION_EPSILON = 0.05


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


HISTORY_ACTIONS = (LISTEN, LISTEN, LISTEN, OPEN_LEFT, LISTEN, LISTEN, LISTEN, LISTEN)


def sample_history(physics, steps, rng, actions=HISTORY_ACTIONS):
    """i's first ``steps`` (action, observation) pairs of a history.

    i follows ``actions`` (Tiger's ``HISTORY_ACTIONS`` by default), j acts
    uniformly at random, and the environment samples the initial state,
    transitions and i's observations from ``rng``. A history that reaches a
    terminal state is rejected: the decision after it would not exist.
    """
    if not 0 <= steps <= len(actions):
        raise ValueError(f"history_steps must lie in [0, {len(actions)}]")
    state, history = physics.sample_initial_state(rng), []
    for action in actions[:steps]:
        joint = {"i": action, "j": rng.choice(physics.get_all_actions("j"))}
        state = physics.sample_transition(state, joint, rng)
        if physics.is_terminal(state):
            raise ValueError("The sampled history ends the episode; choose another seed")
        history.append((action, physics.sample_observation(state, joint, "i", rng)))
    return history


# The exact reference's enumeration guard. The default guard (200,000 branches)
# stopped the L2 H3 reference at history 6; with this budget it completed there in
# 146 s (one problem), so mid-episode problems stay scorable.
REFERENCE_BRANCHES = 50_000_000


def _exact_ipomdp_q(model, prior_model, history, seed, gamma, opponent_depth, horizon, level):
    """Q of the exact I-POMDP at the exact posterior after ``history``, every lower level exact."""
    bank = SolverBank(seed)
    best_response = BestResponseReference(model, bank, gamma, max_branches=REFERENCE_BRANCHES)
    register_levels(bank, model, "i", level, (None,) * (level - 1), opponent_depth, best_response)
    agent_model = prior_model
    for action, observation in history:
        posterior = bank.filter.update(agent_model, action, observation, terminal=False).belief
        agent_model = MentalModel(agent_model.frame, posterior)
    return best_response.q_values(agent_model, horizon)


def episode_reference_q(physics, level, history, horizon, *, level_weights=None, gamma=0.95):
    """Q* of the exact I-POMDP for a study's agent i at ``level`` after its realised ``history``.

    The prior is the one an episode's i holds (``experiments.bootstrapper``): the
    domain's common physical prior, ``level_weights`` over j's level (default
    uniform) and uniform level priors in the nested models. Every intentional
    level below i is exact at depth ``horizon``. ``history`` is i's own
    (action, observation) pairs, so a realised decision of any planner is scored
    against the I-POMDP it was declared to solve.
    """
    from ipomdp.experiments.bootstrapper import HierarchyBuilder
    from ipomdp.planning.config import PlannerConfig

    if level < 2:
        raise ValueError("The exact I-POMDP reference needs an intentional opponent: level >= 2")
    # The builder only lays out the prior; its planners are never queried.
    solver = HierarchyBuilder(SolverBank(0)).build(
        "i",
        level,
        physics,
        "j",
        config=PlannerConfig(simulations=1, depth=horizon, modeled_simulations=1, gamma=gamma),
        level_weights=level_weights,
        nested_level_weights={n: dict.fromkeys(range(n), 1 / n) for n in range(1, level)},
    )
    prior_model = MentalModel(AgentFrame("i", level, physics), solver.initial_belief)
    return _exact_ipomdp_q(physics, prior_model, history, 0, gamma, horizon, horizon, level)


def _nested_chain(level, nested_beliefs):
    """The private beliefs of the models below j, top-down: one per level 1 .. level - 2."""
    depth = max(level - 2, 0)
    chain = (0.5,) * depth if nested_beliefs is None else tuple(nested_beliefs)
    if len(chain) != depth:
        raise ValueError(f"Level {level} takes {depth} nested beliefs, got {len(chain)}")
    if not all(0 <= p <= 1 for p in chain):
        raise ValueError("Nested beliefs must lie in [0,1]")
    return chain


def _opponent_description(level, depth, belief, stack, nested_beliefs):
    if level == 1:
        return {"policy": "uniform_l0"}
    description = {"level": level - 1, "depth": depth, "initial_belief_p": belief}
    if nested_beliefs:
        description["nested_initial_beliefs_p"] = list(nested_beliefs)
    if not any(stack):
        return {"policy": "exact_fixed_depth", **description}
    return {"policy": "same_planner", "modeled": stack_label(stack), **description}


def _validate_problem(
    planner_kind, level, opponent_depth, opponent_belief, modeled, reference, particles
):
    """Check one problem's options; returns its modeled stack (empty at level 1)."""
    if planner_kind not in ("mcts", "rts"):
        raise ValueError("Unknown planner")
    if reference not in ("declared", "exact", "none"):
        raise ValueError("reference must be 'declared', 'exact' or 'none'")
    if level not in (1, 2, 3, 4):
        raise ValueError("Reference level must be 1, 2, 3 or 4")
    if not 0 <= opponent_belief <= 1:
        raise ValueError("Opponent belief must lie in [0,1]")
    if level == 1:
        if opponent_depth is not None or modeled != EXACT:
            raise ValueError(
                "Level 1 has a uniform L0 opponent; it takes no opponent depth or budget"
            )
        return ()
    if type(opponent_depth) is not int or opponent_depth < 1:
        raise ValueError("Levels 2-4 require an explicit positive opponent depth")
    stack = parse_stack(modeled, level - 1)
    if particles is not None and reference == "declared" and any(stack):
        raise ValueError("A declared reference is not exact with particle beliefs; use 'exact'")
    if planner_kind != "mcts" and any(solve and solve.stopping for solve in stack):
        raise ValueError("Sequential modeled stopping is defined for I-POMCP only")
    return stack


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
    modeled=EXACT,
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
            kind, level, opponent_depth, opponent_belief, modeled, reference, particles
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
        modeled=modeled,
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
        modeled=modeled,
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
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--opponent-depth", type=int)
    parser.add_argument("--opponent-belief", type=float, default=0.5)
    parser.add_argument("--modeled", default=EXACT, help="modeled stack, e.g. 400 or 6400s")
    parser.add_argument("--reference", choices=["declared", "exact", "none"], default="declared")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--particle-filter", choices=sorted(FILTERS), default="compressed")
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-rss-mb", type=float, default=4096)
    args = vars(parser.parse_args())
    run_oracle_comparison(args.pop("out"), **args)


if __name__ == "__main__":
    main()
