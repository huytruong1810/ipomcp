"""Search-tree history nodes and the empirical chance-weighted Bellman backup.

A node is one private action-observation history h of the planning agent.
Children are indexed by (action, observation), forming an AND-OR tree. Nodes
hold search statistics only. The authoritative belief at a history is computed
by the belief update (``belief.finite_filter``), never estimated from the
states that simulations happened to route through a node. Earlier versions kept
a particle reservoir per node; no estimator used it, so it was removed.

Parent links are weak so that rerooting after a real step frees discarded
subtrees.
"""

import math
import weakref
from typing import Any, Dict, Optional

from ipomdp.model.pomdp import Action, Observation


class POMCPNode:
    def __init__(self, parent: Optional["POMCPNode"] = None) -> None:
        self.visit_count: int = 0
        # N(h, a): every sampled outcome of a, terminal ones included.
        self.action_counts: Dict[Action, int] = {}
        # Q(h, a): current estimates. Exact boundary nodes store integrated values
        # here without inventing action counts.
        self.action_values: Dict[Action, float] = {}
        # Mean exact expected immediate reward per action (see bellman_backup).
        self.immediate_reward_means: Dict[Action, float] = {}
        # C(h, a, o): nonterminal outcomes with observation o.
        self.continuation_counts: Dict[Action, Dict[Observation, int]] = {}
        # A frontier created by one rollout carries that rollout's estimate until
        # its own actions are evaluated. None means "no estimate", never zero.
        self.rollout_value: Optional[float] = None
        self.children: Dict[Action, Dict[Observation, "POMCPNode"]] = {}
        self._parent_ref = weakref.ref(parent) if parent is not None else None

    @property
    def parent(self):
        return self._parent_ref() if self._parent_ref is not None else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parent_ref"] = None
        return state

    def get_child(self, action: Action, observation: Observation) -> Optional["POMCPNode"]:
        return self.children.get(action, {}).get(observation)

    def create_child(self, action: Action, observation: Observation) -> "POMCPNode":
        siblings = self.children.setdefault(action, {})
        if observation not in siblings:
            siblings[observation] = POMCPNode(parent=self)
        return siblings[observation]

    def value_estimate(self) -> float:
        """V(h) = max over evaluated actions; a fresh frontier uses its rollout.

        Unevaluated actions are not given fictitious zero values; UCB schedules
        them on later visits. Maximization is per history, never across
        observations.
        """
        if self.action_values:
            return max(self.action_values.values())
        if self.rollout_value is None:
            raise ValueError("History node has neither action nor rollout estimates")
        return self.rollout_value

    def bellman_backup(self, action: Action, immediate: float, gamma: float) -> float:
        """Recompute Q(h, a) from the empirical chance law and current child values.

            Q(h, a) = Rbar(h, a) + gamma * sum_o C(h, a, o) / N(h, a) * V(h a o)

        The caller has already incremented N(h, a) and, for a nonterminal
        outcome, C(h, a, o). Terminal outcomes stay in the denominator with zero
        continuation; dividing by survivors only would inflate future value.
        Child values are re-read on every backup, so the cost of early
        exploratory actions deep in the tree is not frozen into ancestors, which
        is the defect of averaging sampled returns.

        ``immediate`` is the exact expected immediate reward at this history,
        so Rbar is an average of equal values; keeping the running mean makes
        the operator valid for any unbiased immediate-reward estimate.

        Chance frequencies are empirical, and the max over noisy children is
        optimistic at small counts. This is an estimator, not a bound.
        """
        n = self.action_counts[action]
        if n <= 0:
            raise ValueError("Bellman backup requires a counted action")
        previous = self.immediate_reward_means.get(action, 0.0)
        mean_reward = previous + (immediate - previous) / n
        self.immediate_reward_means[action] = mean_reward
        outcomes = self.continuation_counts.get(action, {})
        if sum(outcomes.values()) > n:
            raise ValueError("Continuation counts exceed action count")
        continuation = (
            math.fsum(
                count * self.children[action][obs].value_estimate()
                for obs, count in outcomes.items()
            )
            / n
        )
        q = mean_reward + gamma * continuation
        self.action_values[action] = q
        return q

    def to_dict(self, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Serialize statistics of this subtree for diagnostics and rendering."""
        result: Dict[str, Any] = {
            "visit_count": self.visit_count,
            "action_values": {str(a): v for a, v in self.action_values.items()},
            "action_counts": {str(a): c for a, c in self.action_counts.items()},
            "rollout_value": self.rollout_value,
            "immediate_reward_means": {str(a): v for a, v in self.immediate_reward_means.items()},
            "continuation_counts": {
                str(a): {str(o): n for o, n in counts.items()}
                for a, counts in self.continuation_counts.items()
            },
        }
        if current_depth < max_depth:
            result["children"] = {
                str(action): {
                    str(obs): child.to_dict(max_depth, current_depth + 1)
                    for obs, child in obs_map.items()
                }
                for action, obs_map in self.children.items()
            }
        return result
