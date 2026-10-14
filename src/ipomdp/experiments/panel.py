"""Panel: planners scored against the exact I-POMDP on a grid of budgets.

Contributions 1 and 2 both rest on one measurement: how good is a level-1 to
level-4 decision, and what did it cost, compared with the exact solution.
This driver runs that measurement over a grid of cells

    (planner, level, horizon, budget, modeled stack, particles, particle filter, history)

where

- ``planner`` is ``mcts`` (I-POMCP; budget = root simulations) or ``rts``
  (reachability-tree sampling; budget = sampled observations per node);
- ``level`` is i's level, 1 to 4: i models j at level - 1, who models i at
  level - 2, and so on down to uniform L0 (level 1 takes ``exact`` as its
  modeled label, since its uniform L0 opponent needs no solve);
- ``modeled`` labels how each modeled intentional level below i is solved, at
  depth equal to the horizon (``modeled_levels``): the same kind of planner
  with its own budget (I-POMCP simulations, or RTS samples as in IPF + RTS
  where every level is solved by RTS) and, for I-POMCP, optionally sequential
  stopping (``6400s``); per level top-down (``6400s/400`` at level 3), or one
  label for all. ``exact`` makes every lower level exact, which isolates i's
  own search and belief error;
- ``particles`` bounds belief supports to N atoms, or 0 for exact beliefs;
- ``particle filter`` is ``compressed`` (exact updates compressed to N; the
  planner's default) or ``sampled`` (the published IPF; with ``rts`` this is
  IPF + RTS); ``exact`` for exact beliefs;
- ``history`` is the number of real steps i conditions on before deciding
  (``oracle.sample_history``): exact supports grow with it, IPF supports do not.

Every cell of a level is evaluated on the same problems x the seeds.
``problems(level)`` lists them (``experiments.problems``): by default
``tiger_problems``, beliefs of i x private beliefs of j (5 x 3), and at level
L >= 3 also x the private beliefs of the L - 2 models below j, crossed at
every nested level; ``uav_problems`` gives the UAV grid. With ``history`` > 0
the seed also draws the history's states, j's actions and i's observations,
so seeds add problems, not only planner randomness; at ``history`` 0 they add
planner randomness only. The reference is the exact I-POMDP (every lower
level exact at the same depth, all beliefs exact), solved in its own exact
bank on the panel's physics (the published Tiger, a payoff variant such as
``JointTigerModel``, or another domain such as ``UAVModel``), so each loss
includes search error, opponent-model error and, with particles, belief
error; level 1 is scored against the exact best response to uniform L0.

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
from ipomdp.belief import canonical
from ipomdp.domains.tiger import JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.modeled_levels import EXACT, parse_stack, stack_label
from ipomdp.experiments.oracle import OPTIMALITY_TOLERANCE, evaluate_case
from ipomdp.experiments.problems import (
    TIGER_BELIEFS,
    TIGER_NESTED_BELIEFS,
    TIGER_OPPONENT_BELIEFS,
    tiger_problems,
    uav_problems,
)
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes

CELL = (
    "planner",
    "level",
    "horizon",
    "budget",
    "modeled",
    "particles",
    "particle_filter",
    "history",
)


def evaluate_cell(
    planner,
    level,
    horizon,
    budget,
    modeled,
    particles,
    particle_filter,
    history,
    seed,
    problems=tiger_problems,
    physics=None,
):
    """Every problem of ``problems(level)`` for one cell and seed."""
    if level not in (1, 2, 3, 4):
        raise ValueError("The panel scores levels 1 to 4")
    opponent = {}
    if level >= 2:
        opponent = dict(opponent_depth=horizon, modeled=modeled, reference="exact")
    elif modeled != EXACT:
        raise ValueError("Level 1 has a uniform L0 opponent; its modeled label is 'exact'")
    return [
        dict(
            evaluate_case(
                planner,
                horizon,
                budget,
                seed,
                problem.case.get("belief_p"),
                level=level,
                particles=particles or None,
                particle_filter=particle_filter if particles else "compressed",
                history_steps=history,
                physics=physics,
                **opponent,
                **{k: v for k, v in problem.case.items() if k != "belief_p"},
            )[0],
            modeled=modeled,
            particles=particles,
            particle_filter=particle_filter if particles else "exact",
            history=history,
            **problem.labels,
        )
        for problem in problems(level)
    ]


def _optimal(losses):
    """Zero-loss flags; empty for planners without the column (RTS search columns)."""
    return losses.dropna() <= OPTIMALITY_TOLERANCE


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
            mean_optimal_q_error=("optimal_q_error", "mean"),
            search_optimal_rate=("search_first_action_loss", _optimal_rate),
            search_max_loss=("search_first_action_loss", "max"),
            search_mean_q_error=("search_max_abs_q_error", "mean"),
            search_mean_optimal_q_error=("search_optimal_q_error", "mean"),
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
    problems=tiger_problems,
    timeout_seconds=7200,
    max_rss_mb=4096,
    physics=None,
):
    """Run every (cell, seed) job under supervision; write rows, failures and summary.

    ``problems(level)`` lists each level's problems; it must pickle (a module
    function or a ``functools.partial`` of one). ``physics`` is the model of
    every problem (None: the published Tiger).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # One label per configuration, so "400" and "400/400" at level 3 share a cell.
    cells = [
        (
            *cell[:4],
            stack_label(parse_stack(cell[4], cell[1] - 1)) if cell[1] > 1 else cell[4],
            *cell[5:],
        )
        for cell in cells
    ]
    lock = source_hashes(REPOSITORY_ROOT)
    manifest = dict(
        cells=[dict(zip(CELL, c)) for c in cells],
        seeds=list(seeds),
        problems={
            level: [problem.name for problem in problems(level)]
            for level in sorted({cell[1] for cell in cells})
        },
        physics=canonical.physics_identity(TigerModel() if physics is None else physics),
    )
    (output / "manifest.json").write_text(
        json.dumps(dict(panel=manifest, source_sha256=lock), indent=2)
    )
    jobs_index = list(itertools.product(cells, seeds))
    jobs = {
        i: partial(evaluate_cell, *cell, seed, problems=problems, physics=physics)
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
    parser.add_argument("--levels", type=int, choices=[1, 2, 3, 4], nargs="+", default=[2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 4000])
    parser.add_argument(
        "--modeled",
        nargs="+",
        default=[EXACT, "100", "400"],
        help="modeled stacks (modeled_levels): exact, 400, 6400s, 6400s/400",
    )
    parser.add_argument(
        "--particles", type=int, nargs="+", default=[0], help="0 means exact beliefs"
    )
    parser.add_argument(
        "--particle-filters", choices=["compressed", "sampled"], nargs="+", default=["compressed"]
    )
    parser.add_argument("--history", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--domain",
        choices=["tiger", "uav"],
        default="tiger",
        help="tiger: point-belief problems; uav: experiments.problems.uav_problems",
    )
    parser.add_argument("--beliefs", type=float, nargs="+", default=list(TIGER_BELIEFS))
    parser.add_argument(
        "--opponent-beliefs", type=float, nargs="+", default=list(TIGER_OPPONENT_BELIEFS)
    )
    parser.add_argument(
        "--nested-beliefs",
        type=float,
        nargs="+",
        default=list(TIGER_NESTED_BELIEFS),
        help="private beliefs of the models below j, crossed at each nested level (levels 3-4)",
    )
    parser.add_argument(
        "--together-safe",
        type=float,
        help="JointTigerModel's reward for opening the safe door together (unset: published Tiger)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exact-belief cells ignore the filter; keep one of each.
    cells = sorted(
        {
            (planner, level, horizon, budget, modeled, n, flt if n else "exact", history)
            for planner, level, horizon, budget, modeled, n, flt, history in itertools.product(
                args.planners,
                args.levels,
                args.horizons,
                args.budgets,
                args.modeled,
                args.particles,
                args.particle_filters,
                args.history,
            )
        }
    )
    if args.domain == "uav":
        physics = UAVModel()
        problems = partial(uav_problems, physics)
    else:
        physics = None if args.together_safe is None else JointTigerModel(args.together_safe)
        problems = partial(
            tiger_problems,
            beliefs=tuple(args.beliefs),
            opponent_beliefs=tuple(args.opponent_beliefs),
            nested_beliefs=tuple(args.nested_beliefs),
        )
    summary, failures = run_panel(
        args.output, cells, args.seeds, args.workers, problems, physics=physics
    )
    print(summary.round(4).to_string(index=False))
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
