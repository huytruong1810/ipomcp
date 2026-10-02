"""Small studies and episode functions for testing execution infrastructure.

Workers are spawned interpreters, so episode functions must be importable at
module level; that is why these live in a module rather than inside tests.
"""

from pathlib import Path

from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, StudySpec


def random_tiger(trials=2, steps=3):
    """Uniform-random L0 versus L0 Tiger: cheap and deterministic per trial."""
    return StudySpec(
        name="random-tiger",
        domain="tiger",
        trials=trials,
        steps=steps,
        agents={"i": AgentSpec(level=0), "j": AgentSpec(level=0)},
        trial_timeout_seconds=120.0,
        max_trial_rss_mb=2048.0,
    )


def fails_until_recovered(marker, spec, trial):
    """Trial 1 fails until the ``marker`` file exists; other trials run normally."""
    if trial == 1 and not Path(marker).exists():
        raise ValueError("injected recoverable worker failure")
    return run_episode(spec, trial)
