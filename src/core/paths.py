"""
paths.py — Authoritative filesystem path definitions and results directory routing.

Enforces clean separation between source code and runtime artifacts:
- All generated datasets, figures, JSON dumps, and execution logs route strictly
  into `<project_root>/results/<category>/<run_name>`.
- The source tree (`src/`) remains completely pristine and immutable at runtime.
"""

import os
from pathlib import Path

# Source checkout root. Installed applications should set IPOMCP_RESULTS_DIR.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = Path(os.environ.get("IPOMCP_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()


def get_results_dir(category: str, run_name: str) -> str:
    """
    Returns an absolute path to a centralized results directory and ensures it exists.

    Args:
        category: Top-level result domain (e.g., 'deep_prior', 'oracle', 'payoff_matrix', 'tiger', 'uav', 'wumpus', 'benchmarks').
        run_name: Subdirectory name for the specific experiment run or condition.

    Returns:
        Absolute path string to the target results directory.
    """
    target = (RESULTS_ROOT / category / run_name).resolve()
    if not target.is_relative_to(RESULTS_ROOT):
        raise ValueError("Result path must remain inside RESULTS_ROOT.")
    os.makedirs(target, exist_ok=True)
    return str(target)
