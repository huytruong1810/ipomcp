"""L2 panel: planners scored against the exact I-POMDP on a grid of budgets.

Contributions 1 and 2 both rest on one measurement: how good is a level-2
Tiger decision, and what did it cost, compared with the exact I-POMDP. This
driver runs that measurement over a grid of cells

    (planner, horizon, budget, modeled budget, particles)

where

- ``planner`` is ``mcts`` (I-POMCP; budget = root simulations) or ``rts``
  (reachability-tree sampling; budget = sampled observations per node);
- ``modeled`` is the budget of the modeled opponent j-L1, solved by the same
  kind of planner (I-POMCP simulations, or RTS samples as in IPF + RTS where
  every level is solved by RTS), at depth equal to the horizon;
- ``particles`` is the interactive particle filter's N, or 0 for exact beliefs.

Every cell is evaluated on the same problems: 5 beliefs of i x 3 private beliefs
of j x the seeds. The reference is the exact I-POMDP (j an exact L1 agent of the
same depth, all beliefs exact), solved in its own exact bank, so each loss
includes search error, opponent-model error and, with particles, belief error.

Two questions read off the same rows:

- **Allocation** (I-POMCP): at similar CPU, which split between root and
  modeled simulations buys the most accuracy;
- **Baseline** (I-POMCP vs IPF + RTS): accuracy against CPU for each planner.

Each row records first-action loss, maximum Q error and the planner's CPU time
for the decision (``experiments.oracle``). One supervised job per (cell, seed).
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.experiments.oracle import evaluate_case
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes

BELIEFS = (0.05, 0.2, 0.5, 0.8, 0.95)
OPPONENT_BELIEFS = (0.085, 0.5, 0.915)
CELL = ("planner", "horizon", "budget", "modeled", "particles")


def evaluate_cell(planner, horizon, budget, modeled, particles, seed):
    """Every (belief of i, belief of j) problem of one cell for one seed."""
    return [
        dict(
            evaluate_case(
                planner,
                horizon,
                budget,
                seed,
                p,
                level=2,
                opponent_depth=horizon,
                opponent_budget=modeled,
                opponent_belief=p_j,
                reference="exact",
                particles=particles or None,
            )[0],
            modeled=modeled,
            particles=particles,
            opponent_belief=p_j,
        )
        for p in BELIEFS
        for p_j in OPPONENT_BELIEFS
    ]


def summarize(rows):
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(list(CELL))
        .agg(
            n=("first_action_loss", "size"),
            optimal_rate=("first_action_loss", lambda x: float((x <= 1e-9).mean())),
            mean_loss=("first_action_loss", "mean"),
            max_loss=("first_action_loss", "max"),
            mean_q_error=("max_abs_q_error", "mean"),
            mean_cpu_seconds=("planner_cpu_seconds", "mean"),
        )
        .reset_index()
    )


def run_panel(output, cells, seeds, workers, timeout_seconds=7200, max_rss_mb=4096):
    """Run every (cell, seed) job under supervision; write rows, failures and summary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = source_hashes(REPOSITORY_ROOT)
    manifest = dict(cells=[dict(zip(CELL, c)) for c in cells], seeds=list(seeds))
    (output / "manifest.json").write_text(
        json.dumps(dict(panel=manifest, source_sha256=lock), indent=2)
    )
    jobs_index = list(itertools.product(cells, seeds))
    jobs = {i: partial(evaluate_cell, *cell, seed) for i, (cell, seed) in enumerate(jobs_index)}
    rows, failures = [], []
    for index, outcome in supervise_jobs(
        jobs,
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_rss_mb=max_rss_mb,
        log_directory=output / "logs",
        source_lock=(REPOSITORY_ROOT, lock),
    ):
        if outcome["status"] == "complete":
            rows.extend(outcome["rows"])
        else:
            cell, seed = jobs_index[index]
            failures.append(dict(cell=dict(zip(CELL, cell)), seed=seed, reason=outcome["reason"]))
    (output / "rows.json").write_text(json.dumps(rows))
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = summarize(rows) if rows else pd.DataFrame()
    summary.to_csv(output / "summary.csv", index=False)
    return summary, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planners", choices=["mcts", "rts"], nargs="+", default=["mcts"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 4000])
    parser.add_argument("--modeled", type=int, nargs="+", default=[5, 25, 100, 400])
    parser.add_argument(
        "--particles", type=int, nargs="+", default=[0], help="0 means exact beliefs"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = list(
        itertools.product(args.planners, args.horizons, args.budgets, args.modeled, args.particles)
    )
    summary, failures = run_panel(args.output, cells, args.seeds, args.workers)
    print(summary.round(4).to_string(index=False))
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
