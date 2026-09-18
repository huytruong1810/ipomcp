"""Audit driver for the exact WSL working tree snapshot."""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def episode(repo, seed, steps):
    sys.path.insert(0, str(Path(repo) / "src"))
    from core.config import ExperimentConfig
    from examples.experiments.deep_hierarchy_prior_experiment import DeepHierarchyTigerRunner

    runner = DeepHierarchyTigerRunner(
        ExperimentConfig(n_trials=1, max_steps=steps),
        None,
        3,
        2,
        {2: 0.8, 1: 0.1, 0: 0.1},
        {1: 0.8, 0: 0.2},
        planning_depth=5,
    )
    start = time.perf_counter()
    rows = runner._run_single_trial_parallel(seed)
    return dict(seed=seed, seconds=time.perf_counter() - start, rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        str(f.relative_to(a.repo)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in Path(a.repo).rglob("*.py")
        if ".venv" not in f.parts and "results" not in f.parts
    }
    (out / "manifest.json").write_text(json.dumps(dict(arguments=vars(a), files=files), indent=2))
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed(
            [pool.submit(episode, a.repo, seed, a.steps) for seed in range(a.trials)]
        ):
            result = future.result()
            (out / f"trial_{result['seed']:03}.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            print(
                json.dumps(
                    dict(
                        seed=result["seed"],
                        seconds=result["seconds"],
                        reward_i=result["rows"][-1]["cum_reward_i"],
                        reward_j=result["rows"][-1]["cum_reward_j"],
                    )
                ),
                flush=True,
            )
    (out / "timing.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter() - start)))
