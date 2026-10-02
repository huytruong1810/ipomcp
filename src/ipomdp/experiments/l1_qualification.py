"""Level-1 qualification: I-POMCP root decisions versus exact value iteration.

The decision problem is agent i in two-agent Tiger against a uniform-random L0 j.
That is a two-state POMDP, so alpha-vector value iteration gives exact
finite-horizon Q values (``reference.pomdp_exact_vi``). For every
(budget, horizon H, belief P(tiger left), seed) the planner, configured with
depth H, solves the same problem from the exact belief, and each row records:

- ``optimal``: every action the planner's policy can choose is an exact maximizer;
- ``loss``: V*(b) - sum_a pi(a | b) Q*(b, a);
- ``max_abs_q_error``: max_a |Q_hat(b, a) - Q*(b, a)|;
- ``planner_cpu_seconds``: process CPU time of the root decision.

This is the first rung of the small-scale-first ladder. A planner change must
pass it (all decisions optimal at the qualifying budget, Q error shrinking with
budget) before any larger study is run.
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver

BELIEFS = (0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.85, 0.9, 0.95, 0.98)
OPTIMALITY_TOLERANCE = 1e-9


def evaluate_cell(budget, horizon, seeds, beliefs=BELIEFS, gamma=0.95):
    """All (seed, belief) rows of one (budget, horizon) cell."""
    physics = TigerModel()
    exact = ExactPOMDPSolver(physics, horizon=horizon, gamma=gamma)
    exact.solve()
    config = PlannerConfig(
        simulations=budget, depth=horizon, modeled_simulations=budget, gamma=gamma
    )
    rows = []
    for seed in seeds:
        bank = SolverBank(seed)
        planner = HierarchyBuilder(bank).build(
            "i", 1, physics, "j", config=config, level_weights={0: 1.0}
        )
        l0 = MentalModel(AgentFrame("j", 0, physics))
        for p in beliefs:
            belief = FiniteBelief(
                ((InteractiveState(TIGER_LEFT, l0), p), (InteractiveState(TIGER_RIGHT, l0), 1 - p))
            )
            started = time.process_time()
            policy = planner.policy_for(planner.model(belief))
            cpu = time.process_time() - started
            estimate = dict(planner.recommendation_values)
            q = exact.q_values(p, horizon)
            best = max(q.values())
            rows.append(
                dict(
                    budget=budget,
                    horizon=horizon,
                    seed=seed,
                    belief=p,
                    policy=policy,
                    estimated_q=estimate,
                    exact_q=q,
                    loss=math.fsum(w * (best - q[a]) for a, w in policy.items()),
                    optimal=all(
                        q[a] >= best - OPTIMALITY_TOLERANCE for a, w in policy.items() if w > 0
                    ),
                    max_abs_q_error=max(abs(estimate[a] - q[a]) for a in q),
                    planner_cpu_seconds=cpu,
                )
            )
    return rows


def summarize(rows):
    """Per (budget, horizon): n, optimal rate, loss and Q-error statistics, CPU time."""
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["budget", "horizon"])
        .agg(
            n=("optimal", "size"),
            optimal_rate=("optimal", "mean"),
            mean_loss=("loss", "mean"),
            max_loss=("loss", "max"),
            mean_max_q_error=("max_abs_q_error", "mean"),
            max_q_error=("max_abs_q_error", "max"),
            mean_cpu_seconds=("planner_cpu_seconds", "mean"),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    keys = [(b, h) for b in args.budgets for h in args.horizons]
    jobs = {i: partial(evaluate_cell, b, h, tuple(args.seeds)) for i, (b, h) in enumerate(keys)}
    rows, failures = [], []
    for index, outcome in supervise_jobs(
        jobs, workers=args.workers, timeout_seconds=3600, max_rss_mb=2048
    ):
        if outcome["status"] == "complete":
            rows.extend(outcome["rows"])
        else:
            failures.append(dict(cell=keys[index], reason=outcome["reason"]))
    (args.output / "rows.json").write_text(json.dumps(rows))
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = summarize(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    if failures:
        raise SystemExit(f"{len(failures)} cells failed; see failures.json")


if __name__ == "__main__":
    main()
