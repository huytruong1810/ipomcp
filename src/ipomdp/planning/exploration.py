"""Horizon-bounded UCB action selection at a history node.

Score
-----
For history h, tried action a and remaining horizon H:

    score(h, a) = Q(h, a) + c * W_H * sqrt(log N(h) / N(h, a)),
    W_H = (max(0, r_max) - min(0, r_min)) * sum_{t < H} gamma^t.

W_H is the width of any discounted H-step return, using the domain's declared
reward bounds (``POMDPModel.reward_bounds``). Zero is included in the reward
interval because an episode can terminate early and contribute nothing.

Why this rule
-------------
UCB needs a scale. Raw rewards make the constant domain-dependent. Scaling by
extrema of *sampled* returns made exploration depend on which rare penalties a
search happened to draw, and it did not converge against exact references. A
declared bound is fixed before search, identical across particles, and shrinks
correctly with the remaining horizon. It is not a confidence interval: tree
estimates are non-stationary, so no finite-budget guarantee is claimed.

Untried actions are always selected before any tried action is scored, so every
legal action at a visited history is evaluated at least once. Exact ties are
broken uniformly at random (from the solve's own seeded stream) rather than by
action order, which would bias small-budget searches toward early actions.
"""

import math


class HorizonBoundUCB:
    def __init__(self, reward_min, reward_max, exploration):
        if not all(math.isfinite(x) for x in (reward_min, reward_max, exploration)):
            raise ValueError("Reward bounds and exploration constant must be finite")
        if reward_min > reward_max or exploration <= 0:
            raise ValueError("Need ordered reward bounds and a positive exploration constant")
        self.reward_min = min(0.0, reward_min)
        self.reward_max = max(0.0, reward_max)
        self.exploration = exploration

    @classmethod
    def for_agent(cls, model, agent_id, exploration):
        return cls(*model.reward_bounds(agent_id), exploration)

    def return_width(self, remaining_horizon, gamma):
        """Width of any discounted return over ``remaining_horizon`` decisions."""
        if type(remaining_horizon) is not int or remaining_horizon < 1:
            raise ValueError("Remaining horizon must be a positive integer")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Discount must lie in [0,1]")
        return (self.reward_max - self.reward_min) * math.fsum(
            gamma**step for step in range(remaining_horizon)
        )

    def select_action(self, node, actions, remaining_horizon, gamma, rng):
        if not actions:
            raise ValueError("UCB requires at least one legal action")
        for action in actions:
            if action not in node.action_counts:
                return action
        coefficient = self.exploration * self.return_width(remaining_horizon, gamma)
        log_n = math.log(node.visit_count)
        scores = {
            action: node.action_values[action]
            + coefficient * math.sqrt(log_n / node.action_counts[action])
            for action in actions
        }
        best = max(scores.values())
        return rng.choice([action for action, score in scores.items() if score == best])
