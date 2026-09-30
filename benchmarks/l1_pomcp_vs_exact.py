"""Level-1 I-POMCP (POMCP against uniform L0) versus exact alpha-vector value iteration.

Same Tiger physics as the L3-vs-L2 study (growl .85, creak .90). For each planner
profile, root budget, horizon H (= lookahead), belief P(tiger left) and search
seed, compare the planner's root Q values and greedy policy with the exact
finite-horizon solution. Runtime is per root decision; exact solve time is
reported separately per horizon.
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

from examples.experiments.tiger_l3_vs_l2_experiment import _config, _exploration, tiger
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.process_supervisor import supervise_jobs

BELIEFS = (0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.85, 0.9, 0.95, 0.98)


def cell(profile, budget, horizon, seeds):
    physics = tiger()
    start = time.perf_counter()
    exact = ExactPOMDPSolver(physics, horizon=horizon, gamma=0.95)
    exact.solve()
    exact_seconds = time.perf_counter() - start
    rows = []
    for seed in seeds:
        bank = SolverBank(seed=seed)
        bank.set_physical_prior(physics, (TIGER_LEFT, TIGER_RIGHT))
        planner = I_POMDP_Bootstrapper(bank).create_solver(
            agent_id="i",
            level=1,
            model=physics,
            other_agent_ids=["j"],
            level_weights={0: 1.0},
            n_particles=2,
            config=_config(profile, budget, horizon, 0.95, 1),
            exploration_strategy=_exploration(profile),
        )
        l0 = MentalModel(AgentFrame("j", 0, physics))
        for p in BELIEFS:
            model = MentalModel(
                AgentFrame("i", 1, physics),
                FiniteBelief(
                    (
                        (InteractiveState(TIGER_LEFT, l0), p),
                        (InteractiveState(TIGER_RIGHT, l0), 1 - p),
                    )
                ),
            )
            start = time.perf_counter()
            policy = planner.policy_for(model)
            seconds = time.perf_counter() - start
            estimate = dict(planner.recommendation_values)
            q = exact.q_values(p, horizon)
            best = max(q.values())
            rows.append(
                dict(
                    profile=profile,
                    budget=budget,
                    horizon=horizon,
                    seed=seed,
                    belief=p,
                    policy=policy,
                    estimated_q=estimate,
                    exact_q=q,
                    loss=math.fsum(w * (best - q[a]) for a, w in policy.items()),
                    optimal=all(q[a] >= best - 1e-9 for a, w in policy.items() if w > 0),
                    exact_gap=sorted(q.values())[-1] - sorted(q.values())[-2],
                    max_abs_q_error=max(abs(estimate[a] - q[a]) for a in q),
                    q_error_chosen=math.fsum(w * (estimate[a] - q[a]) for a, w in policy.items()),
                    plan_seconds=seconds,
                    exact_solve_seconds=exact_seconds,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", nargs="+", default=["default", "validated"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 1000, 10000, 50000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    keys = [(p, b, h) for p in args.profiles for b in args.budgets for h in args.horizons]
    jobs = {i: partial(cell, *key, args.seeds) for i, key in enumerate(keys)}
    rows = []
    for index, outcome in supervise_jobs(
        jobs, workers=args.workers, timeout_seconds=3600, max_rss_mb=2048
    ):
        if outcome["status"] != "complete":
            print("FAILED", keys[index], outcome.get("reason"), flush=True)
            continue
        rows.extend(outcome["rows"])
        print(keys[index], "done", round(outcome["wall_seconds"], 1), flush=True)
        (args.output / "rows.json").write_text(json.dumps(rows))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
