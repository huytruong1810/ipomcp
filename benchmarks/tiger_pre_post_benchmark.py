"""Reproducible Tiger audit with isolated, resource-bounded trial processes.

The source checkout is explicit. Every completed or failed trial gets a record;
timeouts are failures, never low-reward samples or silently discarded observations.
The supervisor enforces wall-time and resident-memory limits for each child.
Scientific interpretation still requires an adequate number of completed trials.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def episode(args, seed):
    sys.path.insert(0, str(Path(args.repo) / "src"))
    import resource

    from core.config import ExperimentConfig
    from examples.experiments import deep_hierarchy_prior_experiment as experiment

    # Overrides are explicit audit parameters, recorded in the manifest. There is
    # no automatic budget reduction to make a slow configuration appear feasible.
    experiment.SIM_SCHEDULE[3] = args.sims_i
    experiment.SIM_SCHEDULE[2] = args.sims_j
    experiment.PARTICLE_SCHEDULE[3] = args.particles_i
    experiment.PARTICLE_SCHEDULE[2] = args.particles_j
    runner = experiment.DeepHierarchyTigerRunner(
        ExperimentConfig(n_trials=1, max_steps=args.steps),
        None,
        3,
        2,
        {2: 0.8, 1: 0.1, 0: 0.1},
        {1: 0.8, 0: 0.2},
        planning_depth=args.depth,
    )
    wall, cpu = time.perf_counter(), time.process_time()
    rows = runner._run_single_trial_parallel(seed)
    result = dict(
        seed=seed,
        status="complete",
        seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        rows=rows,
    )
    path = Path(args.out) / f"trial_{seed:03}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(path)


def supervise(args, seed):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo",
        args.repo,
        "--out",
        args.out,
        "--steps",
        str(args.steps),
        "--depth",
        str(args.depth),
        "--sims-i",
        str(args.sims_i),
        "--sims-j",
        str(args.sims_j),
        "--particles-i",
        str(args.particles_i),
        "--particles-j",
        str(args.particles_j),
        "--worker-seed",
        str(seed),
    ]
    log_path = Path(args.out) / f"trial_{seed:03}.log"
    started = time.monotonic()
    failure = None
    peak_rss = 0.0
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                status_file = Path(f"/proc/{process.pid}/status")
                try:
                    for line in status_file.read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            peak_rss = max(peak_rss, int(line.split()[1]) / 1024)
                except FileNotFoundError:
                    pass  # The process can exit between poll() and the /proc read.
                if elapsed > args.trial_timeout:
                    failure = "wall-time limit exceeded"
                elif peak_rss > args.max_rss_mb:
                    failure = "resident-memory limit exceeded"
                if failure:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    path = Path(args.out) / f"trial_{seed:03}.json"
    if code != 0 or failure or not path.is_file():
        result = dict(
            seed=seed,
            status="failed",
            reason=failure or f"worker exited {code}",
            wall_seconds=time.monotonic() - started,
            monitored_peak_rss_mb=peak_rss,
            log=str(log_path),
        )
        path.write_text(json.dumps(result, indent=2))
    else:
        result = json.loads(path.read_text())
    return {k: v for k, v in result.items() if k != "rows"} | (
        {
            "reward_i": result["rows"][-1]["cum_reward_i"],
            "reward_j": result["rows"][-1]["cum_reward_j"],
        }
        if "rows" in result
        else {}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--sims-i", type=int, default=20000)
    parser.add_argument("--sims-j", type=int, default=15000)
    parser.add_argument("--particles-i", type=int, default=2000)
    parser.add_argument("--particles-j", type=int, default=1500)
    parser.add_argument("--trial-timeout", type=float, default=900)
    parser.add_argument("--max-rss-mb", type=float, default=3072)
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    if (
        min(
            args.trials,
            args.steps,
            args.workers,
            args.depth,
            args.sims_i,
            args.sims_j,
            args.particles_i,
            args.particles_j,
            args.trial_timeout,
            args.max_rss_mb,
        )
        <= 0
    ):
        parser.error("All budgets and limits must be positive")
    if args.worker_seed is not None:
        episode(args, args.worker_seed)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Audit output directory must be empty; existing evidence is never overwritten")
    sources = {
        str(p.relative_to(args.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((Path(args.repo) / "src").rglob("*.py"))
    }
    manifest = dict(
        arguments=vars(args),
        sources=sources,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(supervise, args, seed) for seed in range(args.trials)]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                wall_seconds=time.monotonic() - start,
                trials=sorted(results, key=lambda r: r["seed"]),
            ),
            indent=2,
        )
    )
    if any(result["status"] != "complete" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
