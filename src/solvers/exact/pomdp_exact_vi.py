"""Exact Value Iteration with Incremental Pruning for 2-State POMDPs / Level 1 I-POMDPs.

Computes the exact analytical, Bayes-optimal value function V_t*(p), action-value
vectors Q_t*(p, a), policy decision boundaries, and alpha-vector representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from solvers.exact.alpha_vector import (
    AlphaVector2D,
    prune_2d,
    prune_2d_with_intervals,
)


@dataclass(frozen=True)
class PolicySegment:
    """A contiguous interval of the belief space where an action is optimal."""

    left_p: float
    right_p: float
    action: Any
    alpha_vector: AlphaVector2D

    def contains(self, p: float, tol: float = 1e-9) -> bool:
        return (self.left_p - tol) <= p <= (self.right_p + tol)

    def __repr__(self) -> str:
        return f"[{self.left_p:.4f}, {self.right_p:.4f}] => {self.action}"


class ExactPOMDPSolver:
    """Exact Value Iteration solver for 2-State POMDPs and Level 1 I-POMDPs.

    Supports arbitrary finite planning horizons H, discount factor gamma, and
    arbitrary fixed opponent policies (e.g. uniform L0).
    """

    def __init__(
        self,
        model: Any,
        horizon: int = 5,
        gamma: float = 0.95,
        opponent_policy: Optional[Dict[Any, float]] = None,
        agent_id: str = "i",
        opponent_id: str = "j",
    ):
        self.model = model
        self.horizon = horizon
        self.gamma = gamma
        self.agent_id = agent_id
        self.opponent_id = opponent_id

        # Determine states and actions
        from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT

        self.states: Tuple[Any, Any] = (TIGER_LEFT, TIGER_RIGHT)
        self.actions: Tuple[Any, ...] = tuple(model.get_all_actions(agent_id))
        self.opponent_actions: Tuple[Any, ...] = tuple(model.get_all_actions(opponent_id))

        if opponent_policy is None:
            # Default to uniform L0
            n_opp = len(self.opponent_actions)
            self.opponent_policy = {a: 1.0 / n_opp for a in self.opponent_actions}
        else:
            self.opponent_policy = dict(opponent_policy)

        self.observations: Tuple[Any, ...] = tuple(model.get_all_observations(agent_id))

        # Caches for horizon t: 1..H
        # alpha_sets[t]: non-dominated alpha vectors for horizon t
        # action_alpha_sets[t][a]: non-dominated alpha vectors for action a at horizon t
        # segments[t]: piecewise linear segments over [0, 1]
        self.alpha_sets: Dict[int, List[AlphaVector2D]] = {}
        self.action_alpha_sets: Dict[int, Dict[Any, List[AlphaVector2D]]] = {}
        self.segments: Dict[int, List[PolicySegment]] = {}

        # Precompute immediate rewards and transitions
        self._immediate_rewards: Dict[Any, Tuple[float, float]] = {}
        self._precompute_immediate_rewards()

    def _transition_prob(self, s_prev: Any, s_next: Any, a_i: Any, a_j: Any) -> float:
        """Query physical transition probability."""
        if hasattr(self.model, "persistent") and self.model.persistent:
            return 1.0 if s_prev == s_next else 0.0
        # In Tiger, opening a door resets location to uniform 0.5
        from examples.tiger.model.tiger_model import LISTEN

        if a_i == LISTEN and a_j == LISTEN:
            return 1.0 if s_prev == s_next else 0.0
        return 0.5

    def _precompute_immediate_rewards(self) -> None:
        """Compute R(s, a) integrated over opponent policy and physical transitions."""
        s0, s1 = self.states
        for a in self.actions:
            r0 = 0.0
            r1 = 0.0
            for a_j, p_j in self.opponent_policy.items():
                joint = {self.agent_id: a, self.opponent_id: a_j}
                for sp in self.states:
                    t0 = self._transition_prob(s0, sp, a, a_j)
                    if t0 > 0:
                        r0 += p_j * t0 * self.model.get_reward(s0, joint, sp, self.agent_id)
                    t1 = self._transition_prob(s1, sp, a, a_j)
                    if t1 > 0:
                        r1 += p_j * t1 * self.model.get_reward(s1, joint, sp, self.agent_id)
            self._immediate_rewards[a] = (r0, r1)

    def solve(self, max_horizon: Optional[int] = None) -> None:
        """Execute Incremental Pruning value iteration up to max_horizon."""
        target_h = max_horizon or self.horizon
        s0, s1 = self.states

        # Base case Gamma_0 = { (0, 0) }
        current_gamma = [AlphaVector2D(0.0, 0.0, None)]
        self.alpha_sets[0] = current_gamma

        for h in range(1, target_h + 1):
            if h in self.alpha_sets:
                current_gamma = self.alpha_sets[h]
                continue

            all_actions_gamma: List[AlphaVector2D] = []
            action_sets: Dict[Any, List[AlphaVector2D]] = {}

            for a in self.actions:
                r0, r1 = self._immediate_rewards[a]
                # Incremental pruning across observations
                w_set = [AlphaVector2D(r0, r1, a)]

                for obs in self.observations:
                    # Projection: Gamma^{a, obs}
                    tau_set: List[AlphaVector2D] = []
                    for alpha_prev in current_gamma:
                        tau_0 = 0.0
                        tau_1 = 0.0
                        for a_j, p_j in self.opponent_policy.items():
                            joint = {self.agent_id: a, self.opponent_id: a_j}
                            for sp in self.states:
                                val_prev = alpha_prev.v0 if sp == s0 else alpha_prev.v1
                                t0 = self._transition_prob(s0, sp, a, a_j)
                                if t0 > 0:
                                    o_prob0 = self.model.get_observation_prob(
                                        obs, sp, joint, self.agent_id
                                    )
                                    tau_0 += p_j * t0 * o_prob0 * val_prev

                                t1 = self._transition_prob(s1, sp, a, a_j)
                                if t1 > 0:
                                    o_prob1 = self.model.get_observation_prob(
                                        obs, sp, joint, self.agent_id
                                    )
                                    tau_1 += p_j * t1 * o_prob1 * val_prev

                        tau_set.append(AlphaVector2D(self.gamma * tau_0, self.gamma * tau_1, a))

                    tau_set = prune_2d(tau_set)

                    # Cross-sum: W = Prune(W \oplus tau_set)
                    cross = [
                        AlphaVector2D(w.v0 + tau.v0, w.v1 + tau.v1, a)
                        for w in w_set
                        for tau in tau_set
                    ]
                    w_set = prune_2d(cross)

                action_sets[a] = w_set
                all_actions_gamma.extend(w_set)

            # Global upper envelope for horizon h
            intervals = prune_2d_with_intervals(all_actions_gamma)
            current_gamma = [v for v, _, _ in intervals]

            self.alpha_sets[h] = current_gamma
            self.action_alpha_sets[h] = action_sets
            self.segments[h] = [
                PolicySegment(left_p=lx, right_p=rx, action=v.action, alpha_vector=v)
                for v, lx, rx in intervals
            ]

    def value(self, belief_p: float, horizon: Optional[int] = None) -> float:
        """Evaluate optimal value V_h*(p) at belief p = P(s_0)."""
        h = horizon or self.horizon
        if h not in self.alpha_sets:
            self.solve(h)
        return max(v.value(belief_p) for v in self.alpha_sets[h])

    def q_values(self, belief_p: float, horizon: Optional[int] = None) -> Dict[Any, float]:
        """Compute exact action-value vector Q_h*(p, a) for all actions."""
        h = horizon or self.horizon
        if h not in self.action_alpha_sets:
            self.solve(h)
        return {
            a: max(v.value(belief_p) for v in self.action_alpha_sets[h][a]) for a in self.actions
        }

    def policy(self, belief_p: float, horizon: Optional[int] = None) -> Any:
        """Return the Bayes-optimal action argmax_a Q_h*(p, a)."""
        h = horizon or self.horizon
        if h not in self.segments:
            self.solve(h)
        for seg in self.segments[h]:
            if seg.contains(belief_p):
                return seg.action
        # Fallback to direct max of q_values
        q_vals = self.q_values(belief_p, h)
        return max(q_vals, key=q_vals.get)

    def decision_boundaries(self, horizon: Optional[int] = None) -> List[PolicySegment]:
        """Return the list of optimal policy segments over [0, 1]."""
        h = horizon or self.horizon
        if h not in self.segments:
            self.solve(h)
        return list(self.segments[h])
