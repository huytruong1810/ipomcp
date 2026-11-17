"""Run a queue of one-process jobs on a shared machine: bounded workers, bounded memory, resumable.

Why
---
Every campaign job is one command that writes one output file (a decision
row, a reference, an episode), so a campaign is a list of commands. Running
such a list was done by a different shell loop in every result directory,
each with its own memory test and stop convention, and none of them measured
the quantity the machine's sharing agreement is stated in: the total memory
of this project's jobs. This module is the one runner.

Queue
-----
A tab-separated file, one job per line:

    TAG <tab> OUT <tab> COMMAND [<tab> PEAK_MB [<tab> NEEDS]]

- ``OUT``: the file the command writes. A job whose ``OUT`` exists is done
  and is never run again, which is all that resuming needs. Commands must
  therefore write ``OUT`` atomically (write a temporary file, then rename),
  as the package's drivers do.
- ``PEAK_MB``: the job's expected peak resident memory, used for the memory
  budget until the job itself is measured larger; empty means the runner's
  default.
- ``NEEDS``: space-separated files that must exist before the job starts
  (references before the arms that are scored against them). A job whose
  needs never appear is reported as blocked, not run.

Rules for starting a job
------------------------
A pending job starts only when all of these hold:

1. fewer than ``workers`` jobs are running;
2. the stop file does not exist (creating it stops new starts; running jobs
   finish, and the next invocation resumes);
3. the machine has at least ``min_available_mb`` of available memory
   (``MemAvailable``), so other projects' jobs are not pushed into swap;
4. the budget holds: the sum over running jobs of max(expected peak,
   measured process-tree memory), plus the new job's expected peak, is at
   most ``rss_cap_mb``. A job whose own expected peak exceeds the cap may
   start only when nothing else is running: it runs alone.

Rules 3 and 4 are applied in queue order: while the first job whose inputs
exist does not fit, no later job starts, so a large job is never starved by
a stream of small ones. A job that only waits for its inputs does not hold
the others back.

Rule 4 uses the larger of the declared and the measured figure, so an
under-declared job reserves what it really takes as soon as it is seen to.

Record
------
One JSON line per finished job is appended to the log: tag, output, exit
status, wall seconds, CPU seconds (user plus system, children included) and
peak resident memory, read from the kernel's accounting for that process
(``os.wait4``), and ``ok``: exit status 0 *and* the output exists. A job that
exits 0 without its output is a failure, not a success with a missing file.
The record is keyed by the output path, never by a counter, so two runs of
one queue can be compared line by line.

Jobs run at ``niceness`` 10 with one thread per numerical library, each in
its own session; its console output goes to ``<log stem>.out/<output file
name>.out``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ipomdp.infra.resources import mem_available_mb, process_tree_rss_mb

DEFAULT_PEAK_MB = 1500.0
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@dataclass(frozen=True)
class Job:
    """One line of a queue (see the module docstring)."""

    tag: str
    out: str
    command: str
    peak_mb: float | None = None
    needs: tuple = ()

    def __post_init__(self):
        if not self.tag or not self.out or not self.command:
            raise ValueError("A job needs a tag, an output path and a command")
        if self.peak_mb is not None and not self.peak_mb > 0:
            raise ValueError("A job's expected peak must be positive")


def read_queue(path):
    """The jobs of a queue file, in file order; duplicate outputs are refused."""
    jobs, seen = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if not 3 <= len(fields) <= 5:
            raise ValueError(f"{path}:{number}: expected 3 to 5 tab-separated fields, got {len(fields)}")
        tag, out, command, *rest = fields
        peak = float(rest[0]) if rest and rest[0].strip() else None
        needs = tuple(rest[1].split()) if len(rest) > 1 else ()
        if out in seen:
            raise ValueError(f"{path}:{number}: output {out} appears twice")
        seen.add(out)
        jobs.append(Job(tag, out, command, peak, needs))
    return jobs


def run_queue(
    jobs,
    *,
    workers,
    log_path,
    stop_file,
    rss_cap_mb,
    min_available_mb,
    default_peak_mb=DEFAULT_PEAK_MB,
    niceness=10,
    poll_seconds=1.0,
    available_mb=mem_available_mb,
    tree_rss_mb=process_tree_rss_mb,
):
    """Run every pending job of ``jobs`` under the rules of the module docstring.

    Returns {"done": already finished before the call, "ran": started here and ok,
    "failed": [(out, exit status)], "blocked": [out of jobs whose needs never appeared],
    "stopped": pending jobs left because of the stop file}. ``available_mb`` and
    ``tree_rss_mb`` are the two measurements, replaceable in tests.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not rss_cap_mb > 0 or not min_available_mb >= 0 or not default_peak_mb > 0 or not poll_seconds > 0:
        raise ValueError("Memory limits and the poll interval must be positive")
    log_path, stop_file = Path(log_path), Path(stop_file)
    console = log_path.with_suffix(".out")
    console.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not Path(job.out).exists()]
    summary = {"done": len(jobs) - len(pending), "ran": 0, "failed": [], "blocked": [], "stopped": 0}
    running = {}  # pid -> (job, process, started wall, started monotonic)

    def expected(job):
        return default_peak_mb if job.peak_mb is None else job.peak_mb

    def committed():
        return sum(max(expected(job), tree_rss_mb(pid)) for pid, (job, *_rest) in running.items())

    def has_inputs(job):
        return all(Path(need).exists() for need in job.needs)

    def fits(job):
        if available_mb() < min_available_mb:
            return False
        if expected(job) > rss_cap_mb:
            return not running
        return committed() + expected(job) <= rss_cap_mb

    with log_path.open("a") as log:
        while pending or running:
            for pid in list(running):
                finished, status, usage = os.wait4(pid, os.WNOHANG)
                if finished == 0:
                    continue
                job, process, started, clock = running.pop(pid)
                process.returncode = os.waitstatus_to_exitcode(status)
                ok = process.returncode == 0 and Path(job.out).exists()
                record = dict(
                    tag=job.tag, out=job.out, rc=process.returncode, ok=ok,
                    wall_seconds=round(time.monotonic() - clock, 3),
                    cpu_seconds=round(usage.ru_utime + usage.ru_stime, 3),
                    peak_rss_mb=round(usage.ru_maxrss / 1024, 1),
                    started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                if ok:
                    summary["ran"] += 1
                else:
                    summary["failed"].append((job.out, process.returncode))
            stopping = stop_file.exists()
            if stopping and not running:
                summary["stopped"] = len(pending)
                break
            started_one = False
            if pending and not stopping and len(running) < workers:
                for index, job in enumerate(pending):
                    if not has_inputs(job):
                        continue  # a later job may be ready; this one waits for its inputs
                    if not fits(job):
                        # Memory decides in queue order: letting smaller jobs overtake would starve a
                        # large one for as long as small ones keep arriving.
                        break
                    pending.pop(index)
                    Path(job.out).parent.mkdir(parents=True, exist_ok=True)
                    with (console / (Path(job.out).name + ".out")).open("w") as stream:
                        process = subprocess.Popen(
                            job.command, shell=True, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, **THREAD_ENVIRONMENT},
                            preexec_fn=lambda: os.nice(niceness),
                        )
                    running[process.pid] = (job, process, time.time(), time.monotonic())
                    started_one = True
                    break
            if not running and pending and not started_one and not stopping:
                # Nothing runs and nothing may start. Memory can free up; missing inputs cannot appear,
                # because nothing is left running that could write them.
                if not any(has_inputs(job) for job in pending):
                    summary["blocked"] = [job.out for job in pending]
                    break
            if not started_one:
                time.sleep(poll_seconds)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queue", type=Path, help="tab-separated job list: TAG, OUT, COMMAND[, PEAK_MB[, NEEDS]]")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True, help="JSON-lines record, appended")
    parser.add_argument("--stop-file", type=Path, required=True, help="create this file to stop new starts")
    parser.add_argument("--rss-cap-mb", type=float, required=True, help="budget for the total memory of running jobs")
    parser.add_argument("--min-available-mb", type=float, required=True,
                        help="start no job below this much available memory on the machine")
    parser.add_argument("--default-peak-mb", type=float, default=DEFAULT_PEAK_MB)
    parser.add_argument("--tags", default="", help="comma-separated tags (or tag prefixes before ':') to run")
    parser.add_argument("--first", type=int, default=0, help="only the first N jobs of each tag (a pilot)")
    args = parser.parse_args(argv)
    jobs = read_queue(args.queue)
    tags = {tag for tag in args.tags.split(",") if tag}
    if tags:
        jobs = [job for job in jobs if job.tag in tags or job.tag.split(":")[0] in tags]
    if args.first:
        seen, kept = {}, []
        for job in jobs:
            seen[job.tag] = seen.get(job.tag, 0) + 1
            if seen[job.tag] <= args.first:
                kept.append(job)
        jobs = kept
    summary = run_queue(
        jobs, workers=args.workers, log_path=args.log, stop_file=args.stop_file, rss_cap_mb=args.rss_cap_mb,
        min_available_mb=args.min_available_mb, default_peak_mb=args.default_peak_mb,
    )
    print(json.dumps(summary))
    return 0 if not summary["failed"] and not summary["blocked"] else 1


if __name__ == "__main__":
    sys.exit(main())
