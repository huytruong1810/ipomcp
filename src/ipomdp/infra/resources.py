"""Resource measurement on Linux ``/proc``: one process, a process tree, the machine.

Three measurements, each used where a decision depends on it:

- ``process_rss_mb``: the current resident memory of this process, recorded in
  experiment rows;
- ``process_tree_rss_mb``: the resident memory of a process and its
  descendants, which the trial supervisor limits and the job queue budgets;
- ``mem_available_mb``: the machine's available memory, below which the job
  queue starts nothing (the machine is shared with other projects).

CPU time is not measured here: decisions are timed with ``time.process_time``
where they are made, and finished jobs report the kernel's own accounting.
"""

import os
from pathlib import Path

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def process_rss_mb() -> float:
    """Current (not peak) resident memory of this process, in MiB."""
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * _PAGE_SIZE / 2**20


def process_tree_rss_mb(pid) -> float:
    """Resident memory of ``pid`` and its current descendants, in MiB; 0 if it has exited.

    Shared pages count in each process's figure, which makes the sum conservative
    for forked children. A process may exit between two reads; only that race is
    ignored. Any other error propagates, so a failed measurement can never read
    as "no memory used" and silently disable a limit.
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
        except (FileNotFoundError, ProcessLookupError):
            pass
    return kib / 1024


def mem_available_mb() -> float:
    """The kernel's estimate of memory available to new work without swapping (``MemAvailable``), in MiB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable line")
