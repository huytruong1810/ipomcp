"""Linux trial isolation with enforced wall-time and process-tree RSS budgets.

Each job owns a new process session. Its result crosses an atomic file boundary,
so a large episode cannot block on an undrained pipe. The parent polls the worker
and descendants, kills the session on budget violation, and always reaps workers.
RSS is sampled, not a hard allocation ceiling: brief spikes between polls can be
missed. Wall limits include worker startup. No failed job becomes scientific data.

Workers use spawn so numerical-library threads and locks are not inherited.
Jobs must be importable, picklable functions or bound methods; callers use a
standard __main__ guard. Startup/import costs count toward the wall/RSS limits.
Call from the main thread. Each trial starts with independent interpreter and
planner state, which also bounds cross-trial allocator growth.
"""

import contextlib
import multiprocessing
import os
import pickle
import signal
import tempfile
import threading
import time
import traceback
from pathlib import Path


def _worker(job, result_path, log_path):
    os.setsid()
    with open(log_path, "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            result = {"status": "complete", "rows": job()}
        except BaseException as error:
            result = {
                "status": "failed",
                "error_type": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
            }
        result["finished_monotonic"] = time.monotonic()
        path = Path(result_path)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


def _rss_tree_mb(pid):
    """Sum resident memory for the worker and its current descendants.

    Shared pages count in each process's RSS, making this conservative for forks.
    A process may exit between reads; only that race is ignored. Other telemetry
    errors propagate rather than silently disabling the resource limit.
    """
    pending, seen, kib = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    kib += int(line.split()[1])
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(map(int, children.split()))
        except FileNotFoundError:
            pass
    return kib / 1024


def _stop(process):
    # A very short deadline can precede setsid(). In that startup race only the
    # worker itself exists; killing its PID is safe and never targets our session.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.kill()
    process.join()


def supervise_jobs(jobs, *, workers, timeout_seconds, max_rss_mb, log_directory=None):
    """Yield (job_id, outcome) for every callable in a mapping of independent jobs.

    The caller validates successful scientific panels and owns checkpointing.
    Errors include process exit, Python traceback, resource reason and measured
    wall/RSS telemetry. Closing this iterator cancels and reaps all active jobs.
    Persistent logs are retained when log_directory is supplied.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Linux trial supervision must start on the main thread")
    if not Path("/proc/self/status").exists():
        raise RuntimeError("Trial supervision requires Linux /proc")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    import math

    if any(not math.isfinite(x) or x <= 0 for x in (timeout_seconds, max_rss_mb)):
        raise ValueError("Resource limits must be finite and positive")
    if any(type(identifier) is not int or identifier < 0 for identifier in jobs):
        raise ValueError("Job identifiers must be nonnegative integers")
    # Numerical libraries can own native threads even when Python calls us
    # from its main thread. Fork would inherit their locks with no owner in the
    # child. Spawn starts a clean interpreter; job callables must be picklable
    # module-level functions or bound methods. Startup counts toward the limit.
    context = multiprocessing.get_context("spawn")
    waiting = iter(jobs.items())
    active = {}
    exhausted = False
    next_index = 0
    with tempfile.TemporaryDirectory(prefix="ipomcp-trials-") as scratch:
        logs = Path(log_directory) if log_directory else Path(scratch)
        logs.mkdir(parents=True, exist_ok=True)
        try:
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    try:
                        identifier, job = next(waiting)
                    except StopIteration:
                        exhausted = True
                        break
                    index = next_index
                    next_index += 1
                    result_path = Path(scratch, f"{index}.pickle")
                    log_path = logs / f"job-{identifier}.log"
                    process = context.Process(target=_worker, args=(job, result_path, log_path))
                    started = time.monotonic()
                    process.start()
                    active[identifier] = [process, started, 0.0, result_path, log_path]
                for identifier, entry in list(active.items()):
                    process, started, peak, result_path, log_path = entry
                    peak = max(peak, _rss_tree_mb(process.pid))
                    entry[2] = peak
                    elapsed = time.monotonic() - started
                    reason = None
                    if process.is_alive() and elapsed > timeout_seconds:
                        reason = "wall-time limit exceeded"
                    elif peak > max_rss_mb:
                        reason = "resident-memory limit exceeded"
                    if process.is_alive() and reason is None:
                        continue
                    _stop(process)
                    if reason or process.exitcode != 0 or not result_path.exists():
                        result = {
                            "status": "failed",
                            "reason": reason
                            or f"worker exited {process.exitcode} without a valid result",
                        }
                    else:
                        with result_path.open("rb") as stream:
                            result = pickle.load(stream)
                        elapsed = result.pop("finished_monotonic") - started
                        if elapsed > timeout_seconds:
                            result = {"status": "failed", "reason": "wall-time limit exceeded"}
                    result.update(
                        wall_seconds=elapsed,
                        monitored_peak_rss_mb=peak,
                        log=str(log_path) if log_directory else None,
                    )
                    del active[identifier]
                    yield identifier, result
                if active:
                    time.sleep(0.05)
        finally:
            for process, *_ in active.values():
                _stop(process)
