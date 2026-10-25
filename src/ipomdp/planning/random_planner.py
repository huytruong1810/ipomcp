"""Level 0: an agent that acts uniformly at random and holds no belief.

In this codebase L0 is uniform random over the agent's full action set, which is
the convention of the interactive particle filter and nested MCTS literature.
It differs from Gmytrasiewicz & Doshi (2005), where L0 is an optimizing POMDP
agent; studies must state which convention they use.
"""

from ipomdp.planning.planner import Planner


class RandomPlanner(Planner):
    def __init__(self, actions, rng):
        if not actions:
            raise ValueError("A random policy requires a nonempty action set")
        self.actions = list(actions)
        self.rng = rng

    def get_action(self, belief=None):
        return self.rng.choice(self.actions)
