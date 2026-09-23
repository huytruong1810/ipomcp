"""Exact Multi-Agent Interactive Value Iteration for Level 2 I-POMDPs.

Computes the exact Bayes-optimal policy, value function, and action-values for an
L2 agent facing an optimizing L1 opponent over finite horizons H <= 5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from examples.tiger.model.tiger_model import (
    LISTEN,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver


class ExactIPOMDPSolver:
    """Exact Bayes-optimal solver for Level 2 I-POMDP in the Tiger domain.

    Solves the Level 1 opponent POMDP exactly via Incremental Pruning, then
    performs backward induction over the reachable interactive belief tree
    IS_{i, 2} = S x M_{j, 1}.
    """

    def __init__(
        self,
        model: Optional[TigerModel] = None,
        horizon: int = 3,
        gamma: float = 0.95,
        agent_id: str = "i",
        opponent_id: str = "j",
    ):
        self.model = model or TigerModel()
        self.horizon = horizon
        self.gamma = gamma
        self.agent_id = agent_id
        self.opponent_id = opponent_id

        self.states: Tuple[Any, Any] = (TIGER_LEFT, TIGER_RIGHT)
        self.actions: Tuple[Any, ...] = tuple(self.model.get_all_actions(agent_id))
        self.opponent_actions: Tuple[Any, ...] = tuple(self.model.get_all_actions(opponent_id))
        self.observations_i: Tuple[Any, ...] = tuple(self.model.get_all_observations(agent_id))
        self.observations_j: Tuple[Any, ...] = tuple(self.model.get_all_observations(opponent_id))

        # 1. Solve Level 1 opponent solver
        self.l1_solver = ExactPOMDPSolver(
            model=self.model,
            horizon=self.horizon,
            gamma=self.gamma,
            agent_id=self.opponent_id,
            opponent_id=self.agent_id,
        )
        self.l1_solver.solve(self.horizon)

        # Caches for backward induction values:
        # _v_table[remaining_steps][(state, round(b_j, 6))] -> float
        # _q_table[remaining_steps][(state, round(b_j, 6))][a_i] -> float
        self._v_table: Dict[int, Dict[Tuple[Any, float], float]] = {}
        self._q_table: Dict[int, Dict[Tuple[Any, float], Dict[Any, float]]] = {}

        self.solve(self.horizon)

    def _se_j(self, b_j_tl: float, a_j: Any, o_j: Any) -> Optional[float]:
        """Bayesian update for opponent j modeling agent i as uniform L0."""
        unnorm = {}
        n_actions_i = len(self.actions)
        for s_prev, mass in [(TIGER_LEFT, b_j_tl), (TIGER_RIGHT, 1.0 - b_j_tl)]:
            for a_i in self.actions:
                joint = {self.agent_id: a_i, self.opponent_id: a_j}
                p_ai = 1.0 / n_actions_i
                for s_next in self.states:
                    t = self._transition_prob(s_prev, s_next, a_i, a_j)
                    o_prob = self.model.get_observation_prob(o_j, s_next, joint, self.opponent_id)
                    unnorm[s_next] = unnorm.get(s_next, 0.0) + mass * p_ai * t * o_prob

        total = sum(unnorm.values())
        if total < 1e-15:
            return None
        return unnorm[TIGER_LEFT] / total

    def _transition_prob(self, s_prev: Any, s_next: Any, a_i: Any, a_j: Any) -> float:
        """Query physical transition probability."""
        if hasattr(self.model, "persistent") and self.model.persistent:
            return 1.0 if s_prev == s_next else 0.0
        if a_i == LISTEN and a_j == LISTEN:
            return 1.0 if s_prev == s_next else 0.0
        return 0.5

    def _get_opponent_policy(self, b_j_tl: float, remaining: int) -> Dict[Any, float]:
        """Query exact Level 1 opponent policy distribution."""
        if remaining <= 0:
            return {a: 1.0 / len(self.opponent_actions) for a in self.opponent_actions}
        # Deterministic optimal choice (or uniform over ties)
        q_vals = self.l1_solver.q_values(b_j_tl, remaining)
        max_q = max(q_vals.values())
        best_actions = [a for a, q in q_vals.items() if np.isclose(q, max_q, atol=1e-9)]
        return {
            a: 1.0 / len(best_actions) if a in best_actions else 0.0 for a in self.opponent_actions
        }

    def solve(self, horizon: int) -> None:
        """Compute backward induction over reachable interactive states."""
        self._v_table.clear()
        self._q_table.clear()

        # Base case: remaining = 0 -> value = 0.0
        self._v_table[0] = {}
        self._q_table[0] = {}

        # 1. Forward pass: Collect all reachable opponent beliefs at each remaining step
        reachable_beliefs: Dict[int, set[float]] = {self.horizon: {0.5}}
        current_layer = {0.5}

        for rem in range(self.horizon, 0, -1):
            next_layer: set[float] = set()
            for b in current_layer:
                opp_policy = self._get_opponent_policy(b, rem)
                for a_j, p_aj in opp_policy.items():
                    if p_aj == 0:
                        continue
                    for o_j in self.observations_j:
                        b_next = self._se_j(b, a_j, o_j)
                        if b_next is not None:
                            next_layer.add(round(b_next, 6))
            reachable_beliefs[rem - 1] = next_layer
            current_layer = next_layer

        # 2. Backward pass: Dynamic programming from rem = 1 to H
        for rem in range(1, horizon + 1):
            self._v_table[rem] = {}
            self._q_table[rem] = {}

            for b in reachable_beliefs[rem]:
                opp_policy = self._get_opponent_policy(b, rem)
                for s in self.states:
                    key = (s, b)
                    self._q_table[rem][key] = {}

                    for a_i in self.actions:
                        q_val = 0.0
                        for a_j, p_aj in opp_policy.items():
                            if p_aj == 0:
                                continue
                            joint = {self.agent_id: a_i, self.opponent_id: a_j}
                            for sp in self.states:
                                t = self._transition_prob(s, sp, a_i, a_j)
                                if t == 0:
                                    continue
                                r = self.model.get_reward(s, joint, sp, self.agent_id)
                                continuation = 0.0
                                if rem > 1:
                                    # Integrate over opponent observation
                                    for o_j in self.observations_j:
                                        o_prob = self.model.get_observation_prob(
                                            o_j, sp, joint, self.opponent_id
                                        )
                                        if o_prob == 0:
                                            continue
                                        b_next = self._se_j(b, a_j, o_j)
                                        if b_next is not None:
                                            b_next_r = round(b_next, 6)
                                            continuation += o_prob * self._v_table[rem - 1].get(
                                                (sp, b_next_r), 0.0
                                            )

                                q_val += p_aj * t * (r + self.gamma * continuation)

                        self._q_table[rem][key][a_i] = q_val
                    self._v_table[rem][key] = max(self._q_table[rem][key].values())

    def q_values(
        self,
        p_tl: float,
        b_j_tl: float = 0.5,
        horizon: Optional[int] = None,
    ) -> Dict[Any, float]:
        """Compute exact Level 2 action-values Q_h*(b_i, a) under joint belief."""
        h = horizon or self.horizon
        b_j_r = round(b_j_tl, 6)
        s0, s1 = self.states

        q_s0 = self._q_table[h].get((s0, b_j_r))
        q_s1 = self._q_table[h].get((s1, b_j_r))

        if q_s0 is None or q_s1 is None:
            # If belief was not in reachable discrete set, fallback to nearest
            keys_s0 = [k for k in self._q_table[h] if k[0] == s0]
            nearest_k0 = min(keys_s0, key=lambda k: abs(k[1] - b_j_tl))
            nearest_k1 = (s1, nearest_k0[1])
            q_s0 = self._q_table[h][nearest_k0]
            q_s1 = self._q_table[h][nearest_k1]

        return {a: p_tl * q_s0[a] + (1.0 - p_tl) * q_s1[a] for a in self.actions}

    def value(
        self,
        p_tl: float,
        b_j_tl: float = 0.5,
        horizon: Optional[int] = None,
    ) -> float:
        """Compute exact Level 2 value V_h*(b_i)."""
        return max(self.q_values(p_tl, b_j_tl, horizon).values())

    def policy(
        self,
        p_tl: float,
        b_j_tl: float = 0.5,
        horizon: Optional[int] = None,
    ) -> Any:
        """Return Bayes-optimal Level 2 action."""
        q_vals = self.q_values(p_tl, b_j_tl, horizon)
        return max(q_vals, key=q_vals.get)
