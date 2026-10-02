"""
telemetry.py — OS-Level Resource Monitoring, Performance Watchdogs, and High-Fidelity Telemetry.

Provides Linux process and host measurements for Monte Carlo Tree Search
and recursive multi-agent reasoning:
- Live process RSS, VMS, CPU time, and major/minor page faults.
- Host physical RAM, available memory, swap utilization, and load averages via `/proc`.
- Operating regime classification:
    * DRAM_BOUND_NORMAL: Memory thresholds were not exceeded; no CPU diagnosis.
    * MEMORY_PRESSURE: Available RAM dropping below safety margins (< 15%).
    Swap occupancy alone cannot establish active swapping or thrashing.
- Memory watchdog sentry enforcing soft/hard limits and automated GC triggers.
- Streaming JSONL telemetry logger for real-time benchmarking and post-hoc diagnostics.
"""

import gc
import json
import os
import resource
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


@dataclass(slots=True)
class SystemResourceSnapshot:
    """Snapshot of process-level and host-level system vitals."""

    timestamp_utc: str
    process_rss_bytes: int
    process_rss_mb: float
    process_vms_bytes: int
    process_vms_mb: float
    process_cpu_user_sec: float
    process_cpu_sys_sec: float
    process_page_faults_minor: int
    process_page_faults_major: int
    host_total_ram_mb: float
    host_available_ram_mb: float
    host_ram_used_pct: float
    host_swap_used_mb: float
    host_swap_total_mb: float
    host_load_avg_1m: float
    os_regime: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """
    High-performance, zero-subprocess Linux system monitor.
    Directly parses `/proc/self/statm`, `/proc/meminfo`, and kernel `getrusage`.
    """

    @staticmethod
    def get_process_memory() -> Tuple[int, int]:
        """
        Returns (rss_bytes, vms_bytes) for the current process.
        Uses /proc/self/statm for microsecond-resolution Linux memory tracking.
        """
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * PAGE_SIZE, int(fields[0]) * PAGE_SIZE

    @staticmethod
    def get_host_memory() -> Dict[str, float]:
        """
        Extracts MemTotal, MemAvailable, SwapTotal, SwapFree from /proc/meminfo.
        All returned values are in Megabytes (MB).
        """
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = float(value.split()[0]) / 1024.0
        total = meminfo["MemTotal"]
        available = meminfo["MemAvailable"]
        return {
            "total_ram_mb": total,
            "available_ram_mb": available,
            "swap_total_mb": meminfo["SwapTotal"],
            "swap_used_mb": meminfo["SwapTotal"] - meminfo["SwapFree"],
            "ram_used_pct": (total - available) / total * 100,
        }

    @classmethod
    def capture_snapshot(cls) -> SystemResourceSnapshot:
        """Captures a complete, unified system vitals record."""
        now_utc = datetime.now(timezone.utc).isoformat()
        rss_bytes, vms_bytes = cls.get_process_memory()
        host_mem = cls.get_host_memory()

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user = float(usage.ru_utime)
        cpu_sys = float(usage.ru_stime)
        minflt = int(usage.ru_minflt)
        majflt = int(usage.ru_majflt)

        load_1m = os.getloadavg()[0]

        # Classify OS regime
        used_pct = host_mem["ram_used_pct"]
        swap_used = host_mem["swap_used_mb"]

        if swap_used > 500.0 and used_pct > 92.0:
            regime = "MEMORY_PRESSURE"
        elif used_pct > 85.0 or host_mem["available_ram_mb"] < 2500.0:
            regime = "MEMORY_PRESSURE"
        else:
            regime = "DRAM_BOUND_NORMAL"

        return SystemResourceSnapshot(
            timestamp_utc=now_utc,
            process_rss_bytes=rss_bytes,
            process_rss_mb=rss_bytes / (1024.0 * 1024.0),
            process_vms_bytes=vms_bytes,
            process_vms_mb=vms_bytes / (1024.0 * 1024.0),
            process_cpu_user_sec=cpu_user,
            process_cpu_sys_sec=cpu_sys,
            process_page_faults_minor=minflt,
            process_page_faults_major=majflt,
            host_total_ram_mb=host_mem["total_ram_mb"],
            host_available_ram_mb=host_mem["available_ram_mb"],
            host_ram_used_pct=used_pct,
            host_swap_used_mb=swap_used,
            host_swap_total_mb=host_mem["swap_total_mb"],
            host_load_avg_1m=load_1m,
            os_regime=regime,
        )


class MemoryWatchdog:
    """
    Proactive Memory Sentry & Guardrail.
    Monitors process and host memory thresholds to prevent OOM events and swap thrashing.
    """

    def __init__(
        self,
        max_process_rss_mb: float = 3500.0,
        min_host_available_mb: float = 2000.0,
        auto_gc_threshold_mb: float = 1500.0,
    ):
        self.max_process_rss_mb = max_process_rss_mb
        self.min_host_available_mb = min_host_available_mb
        self.auto_gc_threshold_mb = auto_gc_threshold_mb
        self.gc_trigger_count = 0

    def check_and_enforce(self) -> Dict[str, Any]:
        """
        Evaluates memory state against safety thresholds.
        Executes aggressive garbage collection if memory threshold is reached.
        Returns diagnostic status dictionary.
        """
        snapshot = SystemMonitor.capture_snapshot()
        status = {
            "safe": True,
            "warning": False,
            "action_taken": "none",
            "snapshot": snapshot.to_dict(),
        }

        # Check soft GC threshold
        if snapshot.process_rss_mb >= self.auto_gc_threshold_mb:
            gc.collect()
            self.gc_trigger_count += 1
            status["action_taken"] = "gc_collect"
            status["warning"] = True

        # Check host available memory threshold
        if snapshot.host_available_ram_mb < self.min_host_available_mb:
            status["warning"] = True
            status["safe"] = False
            status["reason"] = (
                f"Host available RAM ({snapshot.host_available_ram_mb:.1f} MB) below floor ({self.min_host_available_mb:.1f} MB)"
            )

        # Check process RSS ceiling
        if snapshot.process_rss_mb > self.max_process_rss_mb:
            status["warning"] = True
            status["safe"] = False
            status["reason"] = (
                f"Process RSS ({snapshot.process_rss_mb:.1f} MB) exceeded ceiling ({self.max_process_rss_mb:.1f} MB)"
            )

        return status


class TelemetryLogger:
    """
    Synchronous append-only JSONL writer. Call outside critical planning loops.
    Write errors propagate; missing evidence is not a successful experiment.
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Appends a structured event record with embedded system vitals."""
        snapshot = SystemMonitor.capture_snapshot()
        record = {
            "timestamp": snapshot.timestamp_utc,
            "event": event_type,
            "system_vitals": snapshot.to_dict(),
            "payload": data,
        }
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
