"""Exercise every actual suite condition at full planning budgets and horizons.

Trial panels are qualification evidence, not an adequately powered payoff study.
The shared supervisor enforces limits and preserves failures. No source config is
monkeypatched and no missing observation is repaired to make a panel complete.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

from core.config import DEFAULT_PARTICLE_SCHEDULE, DEFAULT_SIM_SCHEDULE, ExperimentConfig
from examples.experiments.deep_hierarchy_prior_experiment import (
    DeepHierarchyTigerRunner,
)
from examples.experiments.deep_hierarchy_prior_experiment import (
    experiment_conditions as prior_conditions,
)
from examples.experiments.level_convergence_matrix_experiment import MatrixCellTigerRunner
from examples.tiger.runners.planner_comparison import (
    ControlledConditionRunner,
)
from examples.tiger.runners.planner_comparison import (
    experiment_conditions as comparison_conditions,
)
from utils.process_supervisor import supervise_jobs


def conditions(config):
    for index, condition in enumerate(prior_conditions(5)):
        runner = DeepHierarchyTigerRunner(
            config,
            None,
            condition["level_i"],
            condition["level_j"],
            condition["prior_i"],
            condition["prior_j"],
            planning_depth=5,
        )
        yield f"prior-{index}", runner
    for index, condition in enumerate(comparison_conditions(3)):
        runner = ControlledConditionRunner(
            config,
            None,
            condition["type"],
            2,
            1,
            condition["sims"],
            3,
            n_particles=condition["particles"],
            obs_branching=6,
        )
        yield f"comparison-{index}", runner
    for i in range(5):
        for j in range(5):
            yield f"matrix-{i}-{j}", MatrixCellTigerRunner(config, None, i, j, planning_depth=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-rss-mb", type=float, default=3072)
    args = parser.parse_args()
    config = ExperimentConfig(
        n_trials=args.trials,
        max_steps=args.steps,
        max_workers=args.workers,
        trial_timeout_seconds=args.timeout,
        max_trial_rss_mb=args.max_rss_mb,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Qualification output must be empty")
    cases = list(conditions(config))
    source = Path(__file__).resolve().parents[1] / "src"
    manifest = {
        "config": asdict(config),
        "source": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))
        },
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulation_schedule": DEFAULT_SIM_SCHEDULE,
        "initial_sample_schedule": DEFAULT_PARTICLE_SCHEDULE,
        "cases": {
            name: {k: v for k, v in vars(runner).items() if k not in {"config", "log_dir"}}
            for name, runner in cases
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    labels, jobs = {}, {}
    for name, runner in cases:
        for seed in range(args.trials):
            identifier = len(jobs)
            labels[identifier] = (name, seed)
            jobs[identifier] = partial(runner._run_single_trial_parallel, seed)
    results = []
    for identifier, outcome in supervise_jobs(
        jobs,
        workers=args.workers,
        timeout_seconds=args.timeout,
        max_rss_mb=args.max_rss_mb,
        log_directory=out / "logs",
    ):
        name, seed = labels[identifier]
        rows = outcome.get("rows")
        if outcome["status"] == "complete":
            if len(rows) != args.steps + 1 or [r["step"] for r in rows] != list(
                range(args.steps + 1)
            ):
                outcome.update(status="failed", reason="Incomplete time panel")
        outcome.update(condition=name, seed=seed)
        (out / f"{name}-seed-{seed}.json").write_text(
            json.dumps(outcome, indent=2, allow_nan=False)
        )
        summary = {k: v for k, v in outcome.items() if k not in {"rows", "traceback"}}
        results.append(summary)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(summary), flush=True)
    if any(r["status"] != "complete" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
