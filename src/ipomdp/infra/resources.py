"""Resource measurement for experiment records (Linux ``/proc``).

Only what experiment records use is kept: the current resident memory of this
process. Process-tree memory limits are enforced by ``process_supervisor``;
CPU time is measured with ``time.process_time`` where decisions are timed.
"""

import os
from pathlib import Path

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def process_rss_mb() -> float:
    """Current (not peak) resident memory of this process, in MiB."""
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * _PAGE_SIZE / 2**20
