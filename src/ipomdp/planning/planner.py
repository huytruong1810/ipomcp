"""The online decision interface shared by every agent model.

An L0 agent acts without a belief; intentional planners condition on their own
actions and observations through ``update_root``. Rendering and reporting live
in ``ipomdp.analysis``, not here.
"""

import abc


class Planner(abc.ABC):
    @abc.abstractmethod
    def get_action(self, belief=None):
        """Sample the next action from the planner's policy."""

    def update_root(self, action, observation):
        """Condition on one's own action and observation; L0 has nothing to update."""

    def get_detailed_stats(self):
        return {}
