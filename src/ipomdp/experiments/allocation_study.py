"""Allocation study: root search versus opponent modeling, at matched cost.

Contribution 1 claims a balance between simulating possible futures (the
planner's root simulations) and simulating other minds (the simulations spent
solving each modeled opponent). This study measures that trade-off on the L2
Tiger problem, where the exact I-POMDP is tractable:

- the planner models j as an I-POMCP L1 with ``modeled`` simulations;
- the planner searches with ``root`` simulations;
- the reference is the exact I-POMDP (j an exact L1 agent of the same depth),
  so the loss includes both search error and opponent-model error.

Each row records first-action loss, maximum Q error and the planner's CPU time.
Comparing rows of similar CPU across different (root, modeled) splits shows
which split buys the most accuracy, which is the evidence an allocation rule
must rest on.
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.experiments.oracle import evaluate_case
from ipomdp.infra.process_supervisor import supervise_jobs

BELIEFS = (0.05, 0.2, 0.5, 0.8, 0.95)
OPPONENT_BELIEFS = (0.085, 0.5, 0.915)


def evaluate_cell(root, modeled, horizon, seeds):
    return [
        dict(
            evaluate_case(
                "mcts",
                horizon,
                root,
                seed,
                p,
                level=2,
                opponent_depth=horizon,
                opponent_budget=modeled,
                opponent_belief=p_j,
                reference="exact",
            )[0],
            modeled=modeled,
        )
        for seed in seeds
        for p in BELIEFS
        for p_j in OPPONENT_BELIEFS
    ]


def summarize(rows):
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["horizon", "budget", "modeled"])
        .agg(
            n=("first_action_loss", "size"),
            optimal_rate=("first_action_loss", lambda x: float((x <= 1e-9).mean())),
            mean_loss=("first_action_loss", "mean"),
            max_loss=("first_action_loss", "max"),
            mean_q_error=("max_abs_q_error", "mean"),
            mean_cpu_seconds=("planner_cpu_seconds", "mean"),
        )
        .reset_index()
        .rename(columns={"budget": "root"})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots", type=int, nargs="+", default=[250, 1000, 4000])
    parser.add_argument("--modeled", type=int, nargs="+", default=[5, 25, 100, 400])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cells = list(itertools.product(args.roots, args.modeled, args.horizons))
    jobs = {
        i: partial(evaluate_cell, r, m, h, tuple(args.seeds)) for i, (r, m, h) in enumerate(cells)
    }
    rows, failures = [], []
    for index, outcome in supervise_jobs(
        jobs, workers=args.workers, timeout_seconds=7200, max_rss_mb=4096
    ):
        if outcome["status"] == "complete":
            rows.extend(outcome["rows"])
        else:
            failures.append(dict(cell=cells[index], reason=outcome["reason"]))
    (args.output / "rows.json").write_text(json.dumps(rows))
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = summarize(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    if failures:
        raise SystemExit(f"{len(failures)} cells failed; see failures.json")


if __name__ == "__main__":
    main()
