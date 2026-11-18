"""A miniature campaign: the whole pipeline end to end on small problems, every decision checked against the exact solver.

Why
---
Each layer has its own tests, but a campaign is the layers wired together:
references written by one process and read by another, planners started by the
job queue, rows read back by the analysis, endpoints and a figure built from
them. A defect in that wiring (a flag that is not passed, a row field read
under the wrong name, a file read before it is complete) passes every unit
test. This module runs the same path in a few minutes on problems small enough
for the exact I-POMDP solver, so it can run before any large campaign.

What it runs
------------
1. **References** (``reachable_decisions reference``): coordination Tiger at
   level 2, horizons 2 and 3, and level 3, horizon 2, three seeds, histories
   of 0 and 1 steps; each stores the exact I-POMDP's action values.
2. **Decisions** on every reference (``reachable_decisions decide``), through
   ``infra.job_queue`` with each decision waiting for its reference:
   - I-POMCP in the certified profile with exact modeled levels, under the
     observation rule of the adopted search (``sample``) and the gap rule
     (``gap``), with a budget ladder;
   - IPF + RTS as published (sampled interactive particle filter, N = 64);
   - I-NTMCP (the authors' planner, tuned variant).
3. **Episodes**: a two-trial level matrix of coordination Tiger (levels 0 to 2,
   horizon 2) run as a supervised suite, with the pre-registered endpoints and
   one figure built from the rows.

What it checks (``failures`` in the summary; the command exits 1 if any)
------------------------------------------------------------------------
- every job finished and wrote its output;
- for every certified I-POMCP row: the intervals of every ladder stage contain
  the exact values, and a stage whose intervals settle the decision plays only
  exact maximizers (first-action loss 0);
- every baseline row carries a first-action loss against the same exact values
  (baselines are approximations; their losses are reported, not required to
  be 0);
- the episode suite completes every (cell, trial) panel and its endpoints and
  figure are produced.

Usage: ``python -m ipomdp.experiments.miniature --out DIR [--workers 4]``. The
output directory holds the queue, the job record, every row, the episode suite,
``summary.json`` and ``decisions.csv``.
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import pandas as pd

from ipomdp.analysis import endpoints, figures
from ipomdp.experiments.suite import COORDINATION, level_matrix, run_suite
from ipomdp.infra.job_queue import Job, run_queue
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.planning.policy import plausible_sets

PROBLEMS = [(2, 2), (2, 3), (3, 2)]  # (level, horizon)
SEEDS = (0, 1, 2)
STEPS = (0, 1)
TOLERANCE = 1e-9


def _command(*arguments):
    """A decision-driver command line run by the current interpreter, with the I-NTMCP submodule importable."""
    submodule = REPOSITORY_ROOT / "third_party" / "i-ntmcp"
    prefix = f'env PYTHONPATH={shlex.quote(str(submodule))}:"$PYTHONPATH" {shlex.quote(sys.executable)}'
    return prefix + " -m ipomdp.experiments.reachable_decisions " + " ".join(map(str, arguments))


def decision_jobs(out):
    """The queue: one reference per problem, then four planner arms that wait for it."""
    jobs = []
    for level, horizon in PROBLEMS:
        for seed in SEEDS:
            for steps in STEPS:
                name = f"l{level}_h{horizon}_s{seed}_k{steps}"
                reference = out / "refs" / f"{name}.json"
                jobs.append(Job("reference", str(reference), _command(
                    "reference", "--domain tiger-joint", f"--level {level}", f"--horizon {horizon}",
                    f"--seed {seed}", f"--steps {steps}", f"--out {reference}"), 500.0))
                arms = {
                    "ipomcp-sample": ("--planner mcts --budget 300 --modeled-exact --particles 0 "
                                      "--ladder 10 30 100 --profile certified"),
                    "ipomcp-gap": ("--planner mcts --budget 300 --modeled-exact --particles 0 "
                                   "--ladder 10 30 100 --profile certified --outcome gap --root-rule open"),
                    "ipf-rts": "--planner rts --budget 4 --particles 64 --rts-published",
                    "intmcp": "--planner intmcp --budget 1000 --intmcp-variant tuned",
                }
                for arm, flags in arms.items():
                    row = out / "rows" / arm / f"{name}.json"
                    jobs.append(Job(f"decide:{arm}", str(row), _command(
                        "decide", f"--problem {reference}", flags, "--seed 0", f"--out {row}"),
                        800.0, (str(reference),)))
    return jobs


def check_decisions(out, jobs):
    """(table of decision rows, failures) for every decision job of the queue."""
    rows, failures = [], []
    for job in jobs:
        if not job.tag.startswith("decide:"):
            continue
        path = Path(job.out)
        if not path.exists():
            failures.append(f"missing row {path}")
            continue
        row = json.loads(path.read_text())
        arm = job.tag.split(":", 1)[1]
        problem = json.loads(Path(job.needs[0]).read_text())
        if "q" not in problem:
            continue  # the history ended the episode: no decision to score
        q = problem["q"]
        record = dict(arm=arm, level=row["level"], horizon=row["horizon"], seed=row["problem_seed"],
                      steps=row["steps"], loss=row.get("loss"), failure=row.get("failure"))
        if row.get("failure"):
            failures.append(f"{path}: {row['failure']}")
        if arm.startswith("ipomcp"):
            settled_any = False
            for stage in row["ladder"]:
                bounds = {a: tuple(b) for a, b in stage["bounds"].items()}
                if not all(lo - TOLERANCE <= q[a] <= hi + TOLERANCE for a, (lo, hi) in bounds.items()):
                    failures.append(f"{path}: stage {stage['budget']} intervals miss the exact values")
                certain, plausible = plausible_sets(bounds)
                if certain == plausible:
                    settled_any = True
                    if stage["loss"] is None or stage["loss"] > TOLERANCE:
                        failures.append(f"{path}: stage {stage['budget']} is settled but loses {stage['loss']}")
            record["settled"] = settled_any
        elif record["loss"] is None or not math.isfinite(record["loss"]):
            failures.append(f"{path}: no first-action loss")
        rows.append(record)
    return pd.DataFrame(rows), failures


def run_episodes(out, workers):
    """A two-trial level matrix (levels 0 to 2, horizon 2) as a supervised suite; (cell frames, failures)."""
    spec = level_matrix("tiger-joint", levels=(0, 1, 2), depth=2, simulations=50, particles=64, steps=6,
                        trials=2, domain_options=COORDINATION, timeout_seconds=600.0, max_rss_mb=2048.0)
    cells = run_suite(spec, out / "episodes", workers)
    failures = [f"episode cell {label} has {frame['trial'].nunique()} of 2 trials"
                for label, frame in cells.items() if frame["trial"].nunique() != 2]
    failures += [f"episode cell {label} missing" for label in spec.cells if label not in cells]
    return cells, failures


def run(out, workers=4):
    """Run the miniature campaign into ``out``; returns the summary written to ``summary.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = decision_jobs(out)
    (out / "queue.tsv").write_text("".join(
        "\t".join([job.tag, job.out, job.command, str(job.peak_mb), " ".join(job.needs)]) + "\n" for job in jobs))
    queue = run_queue(jobs, workers=workers, log_path=out / "jobs.jsonl", stop_file=out / "STOP",
                      rss_cap_mb=6000.0, min_available_mb=4000.0)
    failures = [f"job {path} exited {status}" for path, status in queue["failed"]]
    failures += [f"job {path} never got its inputs" for path in queue["blocked"]]
    table, decision_failures = check_decisions(out, jobs)
    failures += decision_failures
    table.to_csv(out / "decisions.csv", index=False)
    cells, episode_failures = run_episodes(out, workers)
    failures += episode_failures
    matrix = {tuple(int(part[1:]) for part in label.split("_")): frame for label, frame in cells.items()}
    endpoint = endpoints.one_level_endpoint(matrix)
    endpoint.to_csv(out / "one_level_endpoint.csv", index=False)
    endpoints.cell_table(cells).to_csv(out / "cell_table.csv", index=False)
    figures.cumulative_rewards(cells["i2_j1"], out / "rewards_i2_j1.png")
    summary = dict(
        jobs=len(jobs), ran=queue["ran"], already_done=queue["done"],
        decisions={arm: dict(n=int(len(part)), mean_loss=float(part["loss"].mean()),
                             optimal=int((part["loss"] <= TOLERANCE).sum()))
                   for arm, part in table.groupby("arm")} if len(table) else {},
        certified_settled=int(table.get("settled", pd.Series(dtype=bool)).fillna(False).sum()),
        episode_cells=len(cells), one_level_comparisons=len(endpoint), failures=failures,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    summary = run(args.out, args.workers)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    for failure in summary["failures"]:
        print("FAILURE", failure)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
