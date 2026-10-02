"""Matched comparisons of planners against exact Tiger references.

Each case fixes one decision problem (belief, physics, discount, horizon and
opponent model) and asks a planner for its root policy and action values. The
same problem is solved exactly, and the row records:

- ``first_action_loss`` = V*(b) - sum_a pi(a | b) Q*(b, a): the expected value
  lost by the planner's root decision when every later decision is optimal. It
  handles ties exactly and is not an episode regret.
- ``max_abs_q_error`` = max_a |Q_hat(b, a) - Q*(b, a)|.

Problems
--------
- Level 1: agent i against uniform-random L0 j; reference = alpha-vector value
  iteration (``reference.pomdp_exact_vi``).
- Level 2 with an exact opponent: j is an exact L1 agent replanning at a fixed
  depth, with a point prior on its private belief; reference =
  ``reference.ipomdp_exact_vi``.
- Level 2 with a finite opponent: j is an I-POMCP L1 with a declared budget;
  reference = exhaustive best response to that same finite policy
  (``reference.finite_policy_l2``). This isolates search error.

MCTS simulations and RTS particles per branch are different units of work, so
budgets are reported as given and cost is reported as measured wall time and
memory, never as "equal compute".
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.finite_policy_l2 import FinitePolicyL2Reference
from ipomdp.reference.fixed_tiger_opponent import FixedTigerL1Policy
from ipomdp.reference.ipomdp_exact_vi import ExactIPOMDPSolver
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
):
    """Solve one matched problem with the planner and the exact reference."""
    _validate_problem(planner_kind, level, opponent_depth, opponent_belief, opponent_budget)
    model = TigerModel()
    bank = SolverBank(seed=seed)
    key = SolverKey("i", level)
    if planner_kind == "mcts":
        config = PlannerConfig(
            simulations=budget,
            depth=horizon,
            modeled_simulations=budget,
            gamma=gamma,
            exploration=exploration,
        )
        planner = IPOMCPPlanner(key, model, bank, config)
    else:
        config = RTSConfig(max_depth=horizon, obs_branching=9, num_particles=budget, gamma=gamma)
        planner = RTSPlanner(key, model, bank, config)
    bank.register_solver(key, planner)
    bank.register_solver(SolverKey("j", 0), RandomPlanner(model.get_all_actions("j")))
    if level == 2:
        bank.register_solver(SolverKey("i", 0), RandomPlanner(model.get_all_actions("i")))
        if opponent_budget is None:
            provider = FixedTigerL1Policy(model, opponent_depth, gamma)
        else:
            provider = IPOMCPPlanner(
                SolverKey("j", 1),
                model,
                bank,
                PlannerConfig(
                    simulations=opponent_budget,
                    depth=opponent_depth,
                    modeled_simulations=opponent_budget,
                    gamma=gamma,
                    exploration=exploration,
                ),
            )
        bank.register_solver(SolverKey("j", 1), provider)
    belief = _two_state_belief(model, level, belief_p, opponent_belief)
    planner.set_initial_belief(belief)
    started = time.process_time()
    policy = planner.policy_for(planner.model())
    planner_cpu_seconds = time.process_time() - started
    estimates = (
        dict(planner.recommendation_values)
        if planner_kind == "mcts"
        else planner.get_action_values()
    )
    oracle_q = _reference_q(
        model,
        bank,
        planner,
        level,
        horizon,
        gamma,
        belief_p,
        opponent_belief,
        opponent_depth,
        opponent_budget,
    )
    if set(estimates) != set(oracle_q):
        raise ValueError("Planner must evaluate every legal action")
    value = max(oracle_q.values())
    return [
        {
            "planner": planner_kind,
            "planner_config": asdict(config),
            "level": level,
            "opponent": _opponent_description(
                level, opponent_depth, opponent_belief, opponent_budget
            ),
            "horizon": horizon,
            "budget": budget,
            "seed": seed,
            "belief_p": belief_p,
            "gamma": gamma,
            "oracle_q": oracle_q,
            "estimated_q": estimates,
            "policy": policy,
            "first_action_loss": value
            - sum(prob * oracle_q[action] for action, prob in policy.items()),
            "max_abs_q_error": max(abs(estimates[a] - oracle_q[a]) for a in oracle_q),
            "planner_cpu_seconds": planner_cpu_seconds,
        }
    ]


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


def _reference_q(
    model,
    bank,
    planner,
    level,
    horizon,
    gamma,
    belief_p,
    opponent_belief,
    opponent_depth,
    opponent_budget,
):
    if level == 1:
        return ExactPOMDPSolver(model, horizon=horizon, gamma=gamma).q_values(belief_p, horizon)
    if opponent_budget is None:
        solver = ExactIPOMDPSolver(
            model, horizon=horizon, gamma=gamma, opponent_horizon=opponent_depth
        )
        return solver.q_values(belief_p, opponent_belief, horizon)
    # Evaluated after the planner's solve: the shared, cached finite opponent
    # policies define the environment, never i's estimates.
    return FinitePolicyL2Reference(model, bank, gamma).q_values(planner.model(), horizon)


def _opponent_description(level, depth, belief, budget):
    if level == 1:
        return {"policy": "uniform_l0"}
    if budget is None:
        return {"policy": "exact_l1_fixed_depth", "depth": depth, "initial_belief_p": belief}
    return {"policy": "ipomcp_l1", "depth": depth, "budget": budget, "initial_belief_p": belief}


def _validate_problem(planner_kind, level, opponent_depth, opponent_belief, opponent_budget):
    if planner_kind not in ("mcts", "rts"):
        raise ValueError("Unknown planner")
    if level not in (1, 2):
        raise ValueError("Reference level must be 1 or 2")
    if not 0 <= opponent_belief <= 1:
        raise ValueError("Opponent belief must lie in [0,1]")
    if level == 1 and (opponent_depth is not None or opponent_budget is not None):
        raise ValueError("Level 1 has a uniform L0 opponent; it takes no opponent depth or budget")
    if level == 2:
        if type(opponent_depth) is not int or opponent_depth < 1:
            raise ValueError("Level 2 requires an explicit positive opponent depth")
        if planner_kind != "mcts":
            raise ValueError("The level-2 comparison evaluates MCTS only")


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
    workers=2,
    timeout=2400,
    max_rss_mb=4096,
):
    """Run every (planner, horizon, budget, seed, belief) case under supervision.

    Every outcome, failed or not, is written before interpretation. There is no
    pass/fail threshold here: a study states its tolerance before running.
    """
    for kind in planners:
        _validate_problem(kind, level, opponent_depth, opponent_belief, opponent_budget)
    if type(seeds) is not int or seeds < 1 or type(seed_start) is not int or seed_start < 0:
        raise ValueError("Need a positive seed count and a nonnegative first seed")
    if not budgets or any(b < 3 for b in budgets) or not horizons or any(h < 1 for h in horizons):
        raise ValueError("Need budgets >= 3 and positive horizons")
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-rss-mb", type=float, default=4096)
    args = vars(parser.parse_args())
    run_oracle_comparison(args.pop("out"), **args)


if __name__ == "__main__":
    main()
