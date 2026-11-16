"""Exact best response under the shared-horizon convention for the modeled agent.

Why
---
The benchmark's modeled agent replans with the full lookahead H at every step
(``BestResponseReference`` with the bank's policies). I-NTMCP's nested trees
are searched only from current histories, so at depth d of i's lookahead its
modeled agent has H - d steps left: the two solve different games for d >= 1.
This reference solves the second one exactly, so the I-NTMCP comparison can
be read under both conventions.

Convention: at a history of i with r steps to go, an opponent model m plays
uniformly over the maximizers (within ``TIE_TOLERANCE``) of its own r-step
values (the finite-horizon definition of 2005). Implemented for opponents of
level at most 1, whose belief update does not depend on any horizon (their
own opponent is the uniformly random level 0); above that it raises.
"""

import math
from functools import lru_cache

from ipomdp.planning.policy import TIE_TOLERANCE, greedy_policy
from ipomdp.reference.best_response import BestResponseReference


class SharedHorizonReference(BestResponseReference):
    def __init__(self, physics, bank, gamma, max_branches=200_000, cache_size=4096):
        super().__init__(physics, bank, gamma, max_branches, cache_size)
        # The opponent's own r-step values: the ordinary reference (its opponent is level 0).
        self._opponent = BestResponseReference(physics, bank, gamma, max_branches, cache_size)
        self._shared = lru_cache(maxsize=cache_size)(self._shared_q)
        self._law = lru_cache(maxsize=65536)(self._law_at)

    def clear_caches(self):
        super().clear_caches()
        self._opponent.clear_caches()
        self._shared.cache_clear()
        self._law.cache_clear()

    def q_values(self, model, horizon):
        """{action: Q_horizon(belief, action)} with the opponent's law at r steps to go its own r-step greedy law."""
        self._check_model(model)
        if type(horizon) is not int or horizon < 0:
            raise ValueError("Horizon must be a nonnegative integer")
        return dict(zip(self.physics.get_all_actions(model.frame.agent_id), self._shared(model, horizon)))

    def _law_at(self, opponent, r):
        if opponent.frame.level == 0:
            return tuple(self.bank.filter.action_distribution(opponent))
        if opponent.frame.level > 1:
            raise NotImplementedError("Shared-horizon laws are implemented for opponents of level <= 1")
        q = self._opponent.q_values(opponent, r)
        return tuple(greedy_policy(q, TIE_TOLERANCE).items())

    def _shared_q(self, model, r):
        actions = self.physics.get_all_actions(model.frame.agent_id)
        if r == 0:
            return tuple(0.0 for _ in actions)
        values = []
        for action in actions:
            reward, branches = self._shared_branches(model, action, r)
            if r == 1:
                values.append(reward)
                continue
            continuation = math.fsum(p * max(self._shared(successor, r - 1)) for _, p, successor in branches)
            values.append(reward + self.gamma * continuation)
        return tuple(values)

    def _shared_branches(self, model, action, r):
        """``BestResponseReference.branches`` with the opponent's law at ``r`` steps to go."""
        return self._enumerate(model, action, lambda opponent: self._law(opponent, r), expand=r > 1)
