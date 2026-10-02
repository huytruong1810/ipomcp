"""Minimal batch runners for testing execution infrastructure.

Workers are spawned interpreters, so runner classes must be importable at module
level; that is why these live in a module rather than inside test functions.
"""

from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.batch_runner import GenericBatchRunner
from ipomdp.planning.random_planner import RandomPlanner


class RandomTigerRunner(GenericBatchRunner):
    """Uniform-random L0 versus L0 Tiger: cheap, deterministic per trial seed."""

    def _setup_domain(self):
        env = TigerModel(creak_accuracy=1.0)
        planner_i = RandomPlanner(env.get_all_actions("i"))
        planner_j = RandomPlanner(env.get_all_actions("j"))
        return env, planner_i, planner_j, env.get_initial_state()
