"""L2 panel: planners scored against the exact I-POMDP on a grid of budgets.

Contributions 1 and 2 both rest on one measurement: how good is a level-2
Tiger decision, and what did it cost, compared with the exact I-POMDP. This
driver runs that measurement over a grid of cells

    (planner, horizon, budget, modeled budget, particles, particle filter, history)

where

- ``planner`` is ``mcts`` (I-POMCP; budget = root simulations) or ``rts``
  (reachability-tree sampling; budget = sampled observations per node);
- ``modeled`` is the budget of the modeled opponent j-L1, solved by the same
  kind of planner (I-POMCP simulations, or RTS samples as in IPF + RTS where
  every level is solved by RTS), at depth equal to the horizon; 0 means j is the
  exact L1 agent, which isolates i's own search and belief error;
- ``particles`` bounds belief supports to N atoms, or 0 for exact beliefs;
- ``particle filter`` is ``compressed`` (exact updates compressed to N; the
  planner's default) or ``sampled`` (the published IPF; with ``rts`` this is
  IPF + RTS); ``exact`` for exact beliefs;
- ``history`` is the number of real steps i conditions on before deciding
  (``oracle.sample_history``): exact supports grow with it, IPF supports do not.

Every cell is evaluated on the same problems: beliefs of i x private beliefs of
j x the seeds (by default 5 x 3; ``beliefs`` and ``opponent_beliefs`` widen the
grid). With ``history`` > 0 the seed also draws the history's states, j's
actions and i's observations, so seeds add problems, not only planner
randomness; at ``history`` 0 they add planner randomness only. The reference is the exact I-POMDP (j an exact L1 agent of the
same depth, all beliefs exact), solved in its own exact bank, so each loss
includes search error, opponent-model error and, with particles, belief error.

Three questions read off the same rows:

- **Allocation** (I-POMCP): at similar CPU, which split between root and
  modeled simulations buys the most accuracy;
- **Baseline** (I-POMCP vs IPF + RTS): accuracy against CPU for each planner;
- **Belief representation**: at which history length the IPF's bounded supports
  repay its fixed cost over exact beliefs, and at what accuracy.

Each row records first-action loss, maximum Q error and the planner's CPU time
for the decision (``experiments.oracle``). One supervised job per (cell, seed).
The summary gives each optimal rate with its exact 95% Clopper-Pearson interval,
because rates near 1 from tens of problems bound little (30/30 rules out only
rates below 0.884).
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.analysis.statistics import clopper_pearson
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes

BELIEFS = (0.05, 0.2, 0.5, 0.8, 0.95)
OPPONENT_BELIEFS = (0.085, 0.5, 0.915)
CELL = ("planner", "horizon", "budget", "modeled", "particles", "particle_filter", "history")


def evaluate_cell(
    planner,
    horizon,
    budget,
    modeled,
    particles,
    particle_filter,
    history,
    seed,
    beliefs=BELIEFS,
    opponent_beliefs=OPPONENT_BELIEFS,
):
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
                opponent_budget=modeled or None,
                opponent_belief=p_j,
                reference="exact",
                particles=particles or None,
                particle_filter=particle_filter if particles else "compressed",
                history_steps=history,
            )[0],
            modeled=modeled,
            particles=particles,
            particle_filter=particle_filter if particles else "exact",
            history=history,
            opponent_belief=p_j,
        )
        for p in beliefs
        for p_j in opponent_beliefs
    ]


def _optimal(losses):
    """Zero-loss flags; empty for planners without the column (RTS search columns)."""
    return losses.dropna() <= 1e-9


def _optimal_rate(losses):
    flags = _optimal(losses)
    return float(flags.mean()) if len(flags) else float("nan")


def _optimal_rate_low(losses):
    flags = _optimal(losses)
    return clopper_pearson(int(flags.sum()), len(flags))[0]


def _optimal_rate_high(losses):
    flags = _optimal(losses)
    return clopper_pearson(int(flags.sum()), len(flags))[1]


def _mean(flags):
    flags = flags.dropna()
    return float(flags.astype(float).mean()) if len(flags) else float("nan")


def summarize(rows):
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(list(CELL))
        .agg(
            n=("first_action_loss", "size"),
            optimal_rate=("first_action_loss", _optimal_rate),
            optimal_rate_low=("first_action_loss", _optimal_rate_low),
            optimal_rate_high=("first_action_loss", _optimal_rate_high),
            mean_loss=("first_action_loss", "mean"),
            max_loss=("first_action_loss", "max"),
            mean_q_error=("max_abs_q_error", "mean"),
            max_q_error=("max_abs_q_error", "max"),
            search_optimal_rate=("search_first_action_loss", _optimal_rate),
            search_max_loss=("search_first_action_loss", "max"),
            search_mean_q_error=("search_max_abs_q_error", "mean"),
            exact_root_rate=("scores_exact", _mean),
            mean_cpu_seconds=("planner_cpu_seconds", "mean"),
            mean_update_cpu_seconds=("history_update_cpu_seconds", "mean"),
            mean_support=("belief_support", "mean"),
        )
        .reset_index()
    )


def run_panel(
    output,
    cells,
    seeds,
    workers,
    beliefs=BELIEFS,
    opponent_beliefs=OPPONENT_BELIEFS,
    timeout_seconds=7200,
    max_rss_mb=4096,
):
    """Run every (cell, seed) job under supervision; write rows, failures and summary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = source_hashes(REPOSITORY_ROOT)
    manifest = dict(
        cells=[dict(zip(CELL, c)) for c in cells],
        seeds=list(seeds),
        beliefs=list(beliefs),
        opponent_beliefs=list(opponent_beliefs),
    )
    (output / "manifest.json").write_text(
        json.dumps(dict(panel=manifest, source_sha256=lock), indent=2)
    )
    jobs_index = list(itertools.product(cells, seeds))
    problems = dict(beliefs=tuple(beliefs), opponent_beliefs=tuple(opponent_beliefs))
    jobs = {
        i: partial(evaluate_cell, *cell, seed, **problems)
        for i, (cell, seed) in enumerate(jobs_index)
    }
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
    parser.add_argument(
        "--particle-filters", choices=["compressed", "sampled"], nargs="+", default=["compressed"]
    )
    parser.add_argument("--history", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--beliefs", type=float, nargs="+", default=list(BELIEFS))
    parser.add_argument("--opponent-beliefs", type=float, nargs="+", default=list(OPPONENT_BELIEFS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exact-belief cells ignore the filter; keep one of each.
    cells = sorted(
        {
            (planner, horizon, budget, modeled, n, flt if n else "exact", history)
            for planner, horizon, budget, modeled, n, flt, history in itertools.product(
                args.planners,
                args.horizons,
                args.budgets,
                args.modeled,
                args.particles,
                args.particle_filters,
                args.history,
            )
        }
    )
    summary, failures = run_panel(
        args.output, cells, args.seeds, args.workers, args.beliefs, args.opponent_beliefs
    )
    print(summary.round(4).to_string(index=False))
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
