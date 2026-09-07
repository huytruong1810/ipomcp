# Absolute Path: <project_root>/solvers/exploration.py

"""
exploration — UCB strategies for MCTS.

DESIGN DECISION RECORD (Phase 2 Overhaul):
------------------------------------------
1. STATELESS NORMALIZED UCB:
   Previously, NormalizedUCB was dependent on the planner resetting its Q-bounds
   on every step. Because we moved to a stateless, locally-scoped bounds tracking
   system (to prevent parallel universe JIT bleeding), `select_action` now strictly
   requires `q_min` and `q_max` kwargs to perform its normalizations locally.

   Epsilon was added to the denominator to prevent division-by-zero during the
   very first rollouts when all Q-values are identical.
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
        best_action = None
        best_value = -float('inf')

        log_n = math.log(node.visit_count) if node.visit_count > 0 else 0

        for action in available_actions:
            # Untried actions have infinite priority
            if action not in node.action_counts:
                return action

            q = node.action_values.get(action, 0.0)
            n = node.action_counts[action]

            ucb_val = q + self.c * math.sqrt(log_n / n)

            if ucb_val > best_value:
                best_value = ucb_val
                best_action = action

        return best_action if best_action is not None else random.choice(available_actions)


class NormalizedUCB(ExplorationStrategy):
    """
    Scale-Invariant UCB.

    [PHASE 2 FIX]: Uses the dynamically passed `q_min` and `q_max` from `kwargs`
    to normalize the Q-values to [0, 1]. This ensures the exploration constant
    remains domain-independent, while keeping the specific MCTS process stateless.
    """

    def __init__(self, exploration_const: float = 1.0, epsilon: float = 1e-6):
        self.c = exploration_const
        self.epsilon = epsilon

    def select_action(self, node: POMCPNode, available_actions: List[Action], **kwargs) -> Action:
        best_action = None
        best_value = -float('inf')

        log_n = math.log(node.visit_count) if node.visit_count > 0 else 0

        # Extract dynamically discovered local bounds provided by the planner.
        # Rigorously guard against uninitialized (inf / -inf), invalid, or degenerate (q_max <= q_min) bounds.
        q_min = kwargs.get('q_min', 0.0)
        q_max = kwargs.get('q_max', 0.0)

        degenerate_bounds = (
            math.isinf(q_min) or math.isinf(q_max) or
            math.isnan(q_min) or math.isnan(q_max) or
            q_max <= q_min
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