"""Fault injection verifies isolation, enforced limits and resumable failures."""

import json
import os
import time
from functools import partial
from pathlib import Path

import pytest

from ipomdp.experiments.runner import is_complete, run_study
from ipomdp.infra.process_supervisor import supervise_jobs
from tests.study_fixtures import fails_until_recovered, random_tiger


def _sleep():
    time.sleep(30)
    return []


def _allocate():
    allocation = bytearray(512 * 1024 * 1024)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    time.sleep(30)


def _crash():
    os._exit(17)


def _error():
    raise ValueError("injected inference failure")


def _success():
    return [{"worker_pid": os.getpid()}]


def test_supervisor_retains_errors_and_reaps_timed_out_workers(tmp_path):
    started = time.monotonic()
    outcomes = dict(
        supervise_jobs(
            {0: _sleep, 1: _crash, 2: _error, 3: _success},
            workers=2,
            timeout_seconds=5,
            max_rss_mb=4096,
            log_directory=tmp_path,
        )
    )
    assert time.monotonic() - started < 20
    assert outcomes[0]["reason"] == "wall-time limit exceeded"
    assert "17" in outcomes[1]["reason"]
    assert outcomes[2]["error_type"] == "ValueError"
    assert "injected inference failure" in outcomes[2]["traceback"]
    assert outcomes[3]["status"] == "complete"
    pid = outcomes[3]["rows"][0]["worker_pid"]
    assert pid != os.getpid() and not Path(f"/proc/{pid}").exists()
    assert len(list(tmp_path.glob("*.log"))) == 4


def test_supervisor_enforces_rss_budget():
    limit = 256  # Fresh child RSS is independent of parent import history.
    outcomes = dict(supervise_jobs({0: _allocate}, workers=1, timeout_seconds=15, max_rss_mb=limit))
    assert outcomes[0]["reason"] == "resident-memory limit exceeded"
    assert outcomes[0]["monitored_peak_rss_mb"] > limit


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_trial_is_recorded_and_resume_runs_only_missing_trials(tmp_path, workers):
    spec = random_tiger(trials=2, steps=1)
    marker = tmp_path / "recovered"
    out = tmp_path / "run"
    flaky = partial(fails_until_recovered, str(marker))
    with pytest.raises(RuntimeError, match="checkpointed"):
        run_study(spec, out, workers=workers, episode=flaky)
    checkpoint = out / "trials/000000.csv"
    original = checkpoint.read_bytes()
    assert not is_complete(out, spec)
    failures = [json.loads(p.read_text()) for p in (out / "attempts").glob("*/trial-1.json")]
    assert failures[0]["error_type"] == "ValueError"
    marker.touch()
    result = run_study(spec, out, workers=workers, episode=flaky)
    assert len(result) == 4
    assert checkpoint.read_bytes() == original
    assert is_complete(out, spec)
    # Historical failure records survive a successful retry.
    assert len(list((out / "attempts").glob("*/trial-1.json"))) == 2


def _spawn_descendant(pid_file):
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    Path(pid_file).write_text(str(child.pid))
    time.sleep(30)


def test_timeout_stops_descendant_processes(tmp_path):
    path = tmp_path / "child.pid"
    outcomes = dict(
        supervise_jobs(
            {0: partial(_spawn_descendant, str(path))},
            workers=1,
            timeout_seconds=5,
            max_rss_mb=4096,
        )
    )
    assert outcomes[0]["status"] == "failed"
    pid = int(path.read_text())
    status = Path(f"/proc/{pid}/status")
    # An orphan may briefly remain a zombie awaiting init's reap, but cannot run.
    if status.exists():
        assert "State:\tZ" in status.read_text()


def test_supervisor_monotonic_intervals_fit_parent_and_concurrency():
    """All worker lifetimes lie inside the parent's same-clock measurement."""
    started = time.monotonic()
    outcomes = dict(
        supervise_jobs(
            dict.fromkeys(range(6), _success),
            workers=2,
            timeout_seconds=10,
            max_rss_mb=4096,
        )
    )
    finished = time.monotonic()
    intervals = []
    for outcome in outcomes.values():
        assert outcome["status"] == "complete"
        start, end = outcome["started_monotonic"], outcome["finished_monotonic"]
        assert started <= start <= end <= finished
        assert outcome["wall_seconds"] == pytest.approx(end - start)
        intervals.extend(((start, 1), (end, -1)))
    active = 0
    for _, change in sorted(intervals):
        active += change
        assert 0 <= active <= 2
    assert active == 0
    assert sum(o["wall_seconds"] for o in outcomes.values()) <= 2 * (finished - started)
