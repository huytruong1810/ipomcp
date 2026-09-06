"""
test_telemetry.py — Unit tests for the OS-Level Resource Monitoring and Telemetry Suite.
"""

import os
import json
import tempfile
from core.telemetry import (
    SystemMonitor,
    MemoryWatchdog,
    TelemetryLogger,
    SystemResourceSnapshot
)


def test_system_monitor_process_memory():
    rss, vms = SystemMonitor.get_process_memory()
    assert isinstance(rss, int)
    assert isinstance(vms, int)
    assert rss > 0
    assert vms >= rss


def test_system_monitor_host_memory():
    mem = SystemMonitor.get_host_memory()
    assert "total_ram_mb" in mem
    assert "available_ram_mb" in mem
    assert "ram_used_pct" in mem
    assert mem["total_ram_mb"] > 0
    assert mem["available_ram_mb"] > 0
    assert 0.0 <= mem["ram_used_pct"] <= 100.0


def test_system_monitor_capture_snapshot():
    snap = SystemMonitor.capture_snapshot()
    assert isinstance(snap, SystemResourceSnapshot)
    assert snap.process_rss_mb > 0.0
    assert snap.process_vms_mb > 0.0
    assert snap.os_regime in ["DRAM_BOUND_NORMAL", "MEMORY_PRESSURE", "SWAP_THRASHING"]

    snap_dict = snap.to_dict()
    assert isinstance(snap_dict, dict)
    assert "process_rss_mb" in snap_dict
    assert "host_available_ram_mb" in snap_dict


def test_memory_watchdog_enforcement():
    # Normal thresholds
    watchdog = MemoryWatchdog(max_process_rss_mb=50000.0, min_host_available_mb=100.0)
    status = watchdog.check_and_enforce()
    assert status["safe"] is True

    # Soft GC threshold trigger
    aggressive_watchdog = MemoryWatchdog(auto_gc_threshold_mb=1.0)
    status2 = aggressive_watchdog.check_and_enforce()
    assert status2["action_taken"] == "gc_collect"
    assert aggressive_watchdog.gc_trigger_count == 1


def test_telemetry_logger_jsonl_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "telemetry.jsonl")
        logger = TelemetryLogger(output_file=log_path)

        logger.log_event("step_completed", {"step": 1, "reward": 10.0})
        logger.log_event("step_completed", {"step": 2, "reward": 20.0})

        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2
        assert lines[0]["event"] == "step_completed"
        assert lines[0]["payload"]["step"] == 1
        assert "system_vitals" in lines[0]
        assert "process_rss_mb" in lines[0]["system_vitals"]
