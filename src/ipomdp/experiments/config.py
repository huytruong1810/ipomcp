"""Episode-batch execution parameters.

Trial design (how many episodes, how many steps) and execution limits (workers,
per-trial wall time and resident memory) are recorded in every run manifest.
There are no defaults for the design fields: every study must state them.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Attributes:
    n_trials: Independent episodes; trial ids 0..n_trials-1 seed the environment.
    max_steps: Decisions per episode. Planners are not told this number.
    max_workers: Concurrent trial processes.
    trial_timeout_seconds: Wall-time limit per trial, including interpreter start.
    max_trial_rss_mb: Resident-memory limit per trial process tree (sampled).
    """

    n_trials: int
    max_steps: int
    max_workers: int = 2
    trial_timeout_seconds: float = 2400.0
    max_trial_rss_mb: float = 4096.0

    def __post_init__(self):
        if type(self.n_trials) is not int or self.n_trials < 1:
            raise ValueError("n_trials must be a positive integer")
        if type(self.max_steps) is not int or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if type(self.max_workers) is not int or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        for name in ("trial_timeout_seconds", "max_trial_rss_mb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
