"""Near-tie escalation: decide with a larger belief bound only where the decision is close.

Inside the exact window, every departure of the level grid's agents from the
Bayes-optimal agent is a belief-compression error at a near-tie (RESULTS
12.74-12.75). The planner's own gap between its two best action values flags
those decisions (AUC 0.996; a gap of at most 1 catches 93.5% of the misses and
flags 1.5% of decisions, 12.77). A larger bound on the real-step belief alone
does not fix them, because at level 3 the harmful compression is in the nested
beliefs the search builds (12.78).

So the agent keeps a shadow: the same planner built at a larger bound with the
same seeds, conditioned on the same realized actions and observations. The base
planner decides; when its top-two gap is at most ``gap``, the shadow decides
instead. Both update on every real step, so the shadow's belief is always its
own exact-as-bounded belief, not a copy of the base's.
"""

from ipomdp.planning.planner import Planner


class EscalatingPlanner(Planner):
    def __init__(self, base, shadow, gap):
        if not gap >= 0:
            raise ValueError("The escalation gap must be nonnegative")
        self.base, self.shadow, self.gap = base, shadow, gap
        self.escalated, self.base_gap = False, None
        self._decided_by = base

    # The episode records read the agent's belief, key and bank: the base's.
    @property
    def belief(self):
        return self.base.belief

    @property
    def key(self):
        return self.base.key

    @property
    def solver_bank(self):
        return self.base.solver_bank

    def get_action(self, belief=None):
        action = self.base.get_action(belief)
        values = sorted(
            self.base.get_detailed_stats().get("action_values", {}).values(), reverse=True
        )
        self.base_gap = values[0] - values[1] if len(values) > 1 else float("inf")
        self.escalated = self.base_gap <= self.gap
        self._decided_by = self.shadow if self.escalated else self.base
        return self.shadow.get_action(belief) if self.escalated else action

    def update_root(self, action, observation):
        self.base.update_root(action, observation)
        self.shadow.update_root(action, observation)

    def get_detailed_stats(self):
        stats = dict(self._decided_by.get_detailed_stats())
        stats["escalated"] = self.escalated
        stats["base_gap"] = self.base_gap
        return stats
