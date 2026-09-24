"""UCB action-selection strategies over history-node estimates.

Untried actions are visited before scoring tried actions. NormalizedUCB uses
search-local empirical return bounds to scale exploitation values; the constant
still needs empirical calibration. An additive epsilon and degenerate-range
rule are numerical conventions, not a proof of exact scale invariance for all
reward transformations or a new UCT convergence result.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List

from core.pomdp_model import Action
from solvers.node import POMCPNode


class ExplorationStrategy(ABC):
    @abstractmethod
    def select_action(self, node: POMCPNode, available_actions: List[Action], **kwargs) -> Action:
        """
        Selects an action from the node to balance exploration and exploitation.
        """
        pass


class StandardUCB(ExplorationStrategy):
    """
    Classic UCB1. Requires manual tuning of 'c' to match the domain's reward scale.
    """

    def __init__(self, exploration_const: float = 1.0):
        self.c = exploration_const

    def select_action(self, node: POMCPNode, available_actions: List[Action], **kwargs) -> Action:
        return self._select(node, available_actions, self.c)

    @staticmethod
    def _select(node, available_actions, coefficient):
        if not available_actions:
            raise ValueError("UCB requires at least one legal action")
        best_action = None
        best_value = -float("inf")

        log_n = math.log(node.visit_count) if node.visit_count > 0 else 0

        for action in available_actions:
            # Untried actions have infinite priority
            if action not in node.action_counts:
                return action

            q = node.action_values.get(action, 0.0)
            n = node.action_counts[action]

            ucb_val = q + coefficient * math.sqrt(log_n / n)

            if ucb_val > best_value:
                best_value = ucb_val
                best_action = action

        return best_action if best_action is not None else random.choice(available_actions)


class NormalizedUCB(ExplorationStrategy):
    """
    Empirical-return-range UCB with a dimensionless tuning coefficient.

    The planner supplies extrema pooled across visited depths. This is a
    heuristic scale, not a certified bound or a domain-independent calibration.
    In particular, deep sampled penalties can inflate exploration at shallow
    histories. The strategy has no mutable state between solves.
    """

    def __init__(self, exploration_const: float = 1.0, epsilon: float = 1e-6):
        self.c = exploration_const
        self.epsilon = epsilon

    def select_action(self, node: POMCPNode, available_actions: List[Action], **kwargs) -> Action:
        best_action = None
        best_value = -float("inf")

        log_n = math.log(node.visit_count) if node.visit_count > 0 else 0

        # Extract dynamically discovered local bounds provided by the planner.
        # Rigorously guard against uninitialized (inf / -inf), invalid, or degenerate (q_max <= q_min) bounds.
        q_min = kwargs.get("q_min", 0.0)
        q_max = kwargs.get("q_max", 0.0)

        degenerate_bounds = (
            math.isinf(q_min)
            or math.isinf(q_max)
            or math.isnan(q_min)
            or math.isnan(q_max)
            or q_max <= q_min
        )
        q_range = (q_max - q_min) + self.epsilon if not degenerate_bounds else 1.0

        for action in available_actions:
            if action not in node.action_counts:
                return action

            q_raw = node.action_values.get(action, 0.0)
            n = node.action_counts[action]

            # 1. Normalize Q to [0, 1] relative to the current tree's discoveries.
            # If bounds are degenerate, assign neutral value 0.5 to avoid biasing exploration.
            q_norm = 0.5 if degenerate_bounds else ((q_raw - q_min) / q_range)

            # 2. Standard UCB calculation on normalized value
            ucb_val = q_norm + self.c * math.sqrt(log_n / n)

            if ucb_val > best_value:
                best_value = ucb_val
                best_action = action

        return best_action if best_action is not None else random.choice(available_actions)


class HorizonBoundUCB(StandardUCB):
    """Scale UCB by a declared reward interval and the remaining horizon.

    For rewards in [lo, hi], the discounted H-step return has width at most
    (hi-lo) * sum(gamma**t for t in range(H)). Include zero in the interval
    because an episode can terminate early. The caller must supply bounds
    valid for every reachable physical state and joint action, not estimates
    from the current particle or previously sampled extrema.

    The score is Q(h,a) + c * width(H) * sqrt(log N(h) / N(h,a)). This changes
    exploration only: it neither clips values nor removes actions nor changes
    mean-return backups. A valid return interval alone does not make these
    nonstationary tree estimates statistical confidence bounds, and does not
    certify finite-budget accuracy for any choice of c.
    """

    def __init__(self, reward_min, reward_max, exploration_const=1.0):
        if not all(math.isfinite(x) for x in (reward_min, reward_max, exploration_const)):
            raise ValueError("Reward bounds and exploration constant must be finite")
        if reward_min > reward_max or exploration_const <= 0:
            raise ValueError("Need ordered reward bounds and positive exploration constant")
        super().__init__(exploration_const)
        self.reward_min = min(0.0, reward_min)
        self.reward_max = max(0.0, reward_max)

    def return_width(self, remaining_horizon, gamma):
        """Compute the finite discounted width, including gamma=0 and gamma=1."""
        if type(remaining_horizon) is not int or remaining_horizon < 1:
            raise ValueError("Remaining horizon must be a positive integer")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Discount must lie in [0,1]")
        return (self.reward_max - self.reward_min) * sum(
            gamma**step for step in range(remaining_horizon)
        )

    def select_action(self, node, available_actions, *, remaining_horizon, gamma, **kwargs):
        width = self.return_width(remaining_horizon, gamma)
        return self._select(node, available_actions, self.c * width)
