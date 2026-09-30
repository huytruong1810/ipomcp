"""Stage 3: real-agent root decisions versus the exhaustive finite-policy reference.

For each agent, horizon H and pilot trial, build that agent exactly as in the
experiment but with lookahead H, replay its own recorded actions/observations
through the exact filter, and at selected steps compare the planner's
recommendation with FinitePolicyReference at horizon H on the same bank. The
reference is exact conditional on the bank's finite modeled policies (which also
use depth H), so this checks search accuracy, not opponent optimality. Replay
under depth H need not reproduce the depth-5 pilot's nested beliefs; every
replayed belief is nevertheless a valid exact posterior of this model.
"""

import argparse
import ast
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import pandas as pd

from examples.experiments.tiger_l3_vs_l2_experiment import build_agent
from solvers.exact.finite_policy_reference import FinitePolicyReference
from utils.process_supervisor import supervise_jobs
from utils.source_lock import source_hashes

ROOT = Path(__file__).resolve().parents[1]
LEVEL = {"i": 3, "j": 2}


def _observation(text):
    value = ast.literal_eval(text)  # recorded as repr of a (growl, creak) tuple
    if not (isinstance(value, tuple) and len(value) == 2):
        raise ValueError(f"Unexpected observation {text!r}")
    return value


def evaluate(agent, horizon, trial, rows, steps, profile, sims, modeled_sims, gamma, seed):
    random.seed(seed)
    planner = build_agent(agent, LEVEL[agent], profile, sims, horizon, gamma, modeled_sims)
    reference = FinitePolicyReference(planner.pomdp_model, planner.solver_bank, gamma)
    results, previous = [], 0
    for step in sorted(steps):
        for row in rows[previous:step]:
            planner.update_root(row[f"action_{agent}"], _observation(row[f"obs_{agent}"]))
        previous = step
        model = planner.model()
        start = time.perf_counter()
        policy = planner.policy_for(model)
        plan_seconds = time.perf_counter() - start
        estimates = dict(planner.recommendation_values)
        start = time.perf_counter()
        exact = reference.q_values(model, horizon)
        reference_seconds = time.perf_counter() - start
        best = max(exact.values())
        ordered = sorted(exact.values(), reverse=True)
        results.append(
            dict(
                agent=agent,
                level=LEVEL[agent],
                horizon=horizon,
                trial=trial,
                step=step,
                belief_support=len(model.belief.mass),
                policy=policy,
                estimated_q=estimates,
                exact_q=exact,
                expected_loss=math.fsum(p * (best - exact[a]) for a, p in policy.items()),
                optimal=all(exact[a] >= best - 1e-9 for a, p in policy.items() if p > 0),
                exact_gap=ordered[0] - ordered[1],
                max_abs_q_error=max(abs(estimates[a] - exact[a]) for a in exact),
                plan_seconds=plan_seconds,
                reference_seconds=reference_seconds,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", type=Path, required=True, help="Pilot run directory")
    parser.add_argument("--trials", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--agents", nargs="+", default=["i", "j"])
    parser.add_argument("--profile", default="default")
    parser.add_argument("--modeled-sims", type=int, default=25)
    parser.add_argument("--sims-i", type=int, default=20000)
    parser.add_argument("--sims-j", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = {
        t: pd.read_csv(args.pilot / "trials" / f"{t:06}.csv", keep_default_na=False)
        .sort_values("step")
        .iloc[1:]
        .to_dict("records")
        for t in args.trials
    }
    jobs, keys = {}, {}
    for index, (agent, horizon, trial) in enumerate(
        (a, h, t) for a in args.agents for h in args.horizons for t in args.trials
    ):
        keys[index] = (agent, horizon, trial)
        jobs[index] = partial(
            evaluate,
            agent,
            horizon,
            trial,
            frames[trial],
            args.steps,
            args.profile,
            args.sims_i if agent == "i" else args.sims_j,
            args.modeled_sims,
            0.95,
            trial * 10000 + 1337,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    rows, failures = [], []
    for index, outcome in supervise_jobs(
        jobs,
        workers=args.workers,
        timeout_seconds=args.timeout,
        max_rss_mb=4096,
        log_directory=args.output / "logs",
        source_lock=(ROOT, source_hashes(ROOT)),
    ):
        agent, horizon, trial = keys[index]
        if outcome["status"] == "complete":
            rows.extend(outcome["rows"])
        else:
            failures.append(dict(agent=agent, horizon=horizon, trial=trial, **outcome))
        print(
            agent, horizon, trial, outcome["status"], round(outcome["wall_seconds"], 1), flush=True
        )
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2, default=str))
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(["agent", "horizon"])
            .agg(
                cases=("optimal", "size"),
                optimal=("optimal", "sum"),
                mean_loss=("expected_loss", "mean"),
                max_loss=("expected_loss", "max"),
                violations=("expected_loss", lambda s: int((s > 0.020 + 1e-8).sum())),
                mean_max_q_error=("max_abs_q_error", "mean"),
                max_q_error=("max_abs_q_error", "max"),
                mean_plan_s=("plan_seconds", "mean"),
                mean_reference_s=("reference_seconds", "mean"),
            )
            .reset_index()
        )
        (args.output / "summary.csv").write_text(summary.to_csv(index=False))
        print(summary.to_string(index=False))
    print(f"{len(failures)} failed jobs")


if __name__ == "__main__":
    main()
