"""Minimal batch runners for testing execution infrastructure.

Workers are spawned interpreters, so runner classes must be importable at module
level; that is why these live in a module rather than inside test functions.
"""

import random

from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.batch_runner import GenericBatchRunner
from ipomdp.planning.random_planner import RandomPlanner


class RandomTigerRunner(GenericBatchRunner):
    """Uniform-random L0 versus L0 Tiger: cheap, deterministic per trial seed."""

    def _setup_domain(self, seed):
        env = TigerModel()
        planner_i = RandomPlanner(env.get_all_actions("i"), random.Random(seed))
        planner_j = RandomPlanner(env.get_all_actions("j"), random.Random(seed + 1))
        return env, planner_i, planner_j
