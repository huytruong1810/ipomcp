"""Fault injection verifies isolation, enforced limits and resumable failures."""

import json
import os
import time
from functools import partial
from pathlib import Path

import pytest

from core.config import ExperimentConfig
from examples.tiger.runners.tiger_baseline_runner import TigerBaselineRunner
from utils.generic_batch_runner import is_batch_complete
from utils.process_supervisor import supervise_jobs


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


class RecoverableRunner(TigerBaselineRunner):
    def _run_single_trial_parallel(self, trial_id):
        if trial_id == 1 and not (Path(self.log_dir) / "recovered").exists():
            raise ValueError("injected recoverable worker failure")
        return super()._run_single_trial_parallel(trial_id)


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_failure_record_checkpoint_and_resume(tmp_path, workers):
    config = ExperimentConfig(n_trials=2, max_steps=1)
    runner = RecoverableRunner(config, str(tmp_path))
    with pytest.raises(RuntimeError, match="checkpointed"):
        runner.run_batch(max_workers=workers)
    checkpoint = tmp_path / "trials/000000.csv"
    original = checkpoint.read_bytes()
    assert not is_batch_complete(tmp_path / "batch_results.csv", 2, 1)
    failures = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*/trial-1.json")]
    assert failures[0]["error_type"] == "ValueError"
    (tmp_path / "recovered").touch()
    result = runner.run_batch(max_workers=workers)
    assert len(result) == 4
    assert checkpoint.read_bytes() == original
    assert is_batch_complete(tmp_path / "batch_results.csv", 2, 1)
    # Historical failure records survive a successful retry.
    assert len(list((tmp_path / "attempts").glob("*/trial-1.json"))) == 2


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
