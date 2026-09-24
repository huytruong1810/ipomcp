"""Finite-horizon L2 Tiger planning by exhaustive joint-belief recursion.

The hidden state is (physical tiger, L1 opponent private belief). Only the
protagonist's observation selects a continuation policy. Maximizing separately
for hidden states would instead give a clairvoyant relaxation, not an I-POMDP.
This reference uses floating-point arithmetic without belief rounding, state
aggregation, probability cutoffs, or nearest-belief substitution. It is intended
for bounded reference problems, not long-horizon production planning.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver


class ExactIPOMDPSolver:
    """L2 against an exact L1 opponent who models uniform-random L0.

    Both agents use the same decreasing remaining horizon and discount. The
    public convenience interface starts from a point belief about the opponent's
    private belief. Internally, posteriors retain correlations between physical
    state and all reachable opponent beliefs. Opponent actions are marginalized,
    never revealed to the protagonist. Private observations are conditionally
    independent given post-transition state and joint action, as in TigerModel.
    Equal L1 action values use a uniform tie policy within absolute tolerance
    1e-10 (no relative tolerance). This is a declared numerical convention.
    """

    def __init__(self, model=None, horizon=3, gamma=0.95, agent_id="i", opponent_id="j"):
        self.model = TigerModel() if model is None else model
        if not isinstance(self.model, TigerModel):
            raise TypeError("This reference solver supports TigerModel only")
        self._check_horizon(horizon)
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if {agent_id, opponent_id} != {"i", "j"}:
            raise ValueError("Tiger requires distinct agents i and j")
        self.horizon, self.gamma = horizon, gamma
        self.agent_id, self.opponent_id = agent_id, opponent_id
        self.states = (TIGER_LEFT, TIGER_RIGHT)
        self.actions = tuple(self.model.get_all_actions(agent_id))
        self.opponent_actions = tuple(self.model.get_all_actions(opponent_id))
        self.l1_solver = ExactPOMDPSolver(
            self.model,
            horizon=horizon,
            gamma=gamma,
            agent_id=opponent_id,
            opponent_id=agent_id,
        )
        # Per-instance bounded caches cannot retain other solver instances. The
        # physics object must remain fixed for this solver's lifetime.
        self._opponent_update = lru_cache(maxsize=8192)(self._opponent_update)
        self._opponent_policy = lru_cache(maxsize=8192)(self._opponent_policy)
        self._joint_q = lru_cache(maxsize=8192)(self._joint_q)
        self.solve(horizon)

    @staticmethod
    def _check_horizon(horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")

    @staticmethod
    def _check_probability(value):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("belief probability must lie in [0, 1]")

    def solve(self, horizon):
        """Set the planning horizon; joint beliefs are solved lazily on demand."""
        self._check_horizon(horizon)
        self.horizon = horizon
        self.l1_solver.solve(horizon)

    def _opponent_policy(self, belief, remaining):
        values = self.l1_solver.q_values(belief, remaining)
        best = max(values.values())
        actions = tuple(a for a, value in values.items() if abs(value - best) <= 1e-10)
        return tuple((a, 1.0 / len(actions)) for a in actions)

    def _opponent_update(self, belief, action, observation):
        """Update j using only j's action/observation under its subjective L0."""
        weights = defaultdict(float)
        for state, mass in ((TIGER_LEFT, belief), (TIGER_RIGHT, 1 - belief)):
            for action_i in self.actions:
                joint = {self.agent_id: action_i, self.opponent_id: action}
                for following, transition in self.model.transition_distribution(state, joint):
                    likelihood = self.model.get_observation_prob(
                        observation, following, joint, self.opponent_id
                    )
                    weights[following] += mass * transition * likelihood / len(self.actions)
        total = math.fsum(weights.values())
        if total == 0:
            raise ValueError("Opponent observation has zero subjective probability")
        return weights[TIGER_LEFT] / total

    def _joint_q(self, belief, remaining):
        """Bellman backup with one maximization per observable posterior.

        Each belief entry is (state, opponent P(TL), probability). A branch
        accumulates unnormalized joint posterior mass before choosing any future
        action; neither hidden state nor opponent private observation is exposed.
        """
        if remaining == 0:
            return tuple(0.0 for _ in self.actions)
        result = []
        for action in self.actions:
            reward = 0.0
            branches = defaultdict(lambda: defaultdict(float))
            for state, opponent_belief, mass in belief:
                for other_action, policy_mass in self._opponent_policy(opponent_belief, remaining):
                    joint = {self.agent_id: action, self.opponent_id: other_action}
                    for following, transition in self.model.transition_distribution(state, joint):
                        event = mass * policy_mass * transition
                        reward += event * self.model.get_reward(
                            state, joint, following, self.agent_id
                        )
                        if remaining == 1:
                            continue
                        for other_obs, other_likelihood in self.model.observation_distribution(
                            following, joint, self.opponent_id
                        ):
                            if other_likelihood == 0:
                                continue
                            next_opponent = self._opponent_update(
                                opponent_belief, other_action, other_obs
                            )
                            for obs, likelihood in self.model.observation_distribution(
                                following, joint, self.agent_id
                            ):
                                weight = event * other_likelihood * likelihood
                                if weight > 0:
                                    branches[obs][(following, next_opponent)] += weight
            continuation = 0.0
            for weights in branches.values():
                evidence = math.fsum(weights.values())
                posterior = tuple((s, b, w / evidence) for (s, b), w in sorted(weights.items()))
                continuation += evidence * max(self._joint_q(posterior, remaining - 1))
            result.append(reward + self.gamma * continuation)
        return tuple(result)

    def q_values(self, p_tl, b_j_tl=0.5, horizon=None):
        """Action values at a point prior about j's private belief (unrounded)."""
        self._check_probability(p_tl)
        self._check_probability(b_j_tl)
        h = self.horizon if horizon is None else horizon
        self._check_horizon(h)
        belief = tuple(
            (s, b_j_tl, w) for s, w in ((TIGER_LEFT, p_tl), (TIGER_RIGHT, 1 - p_tl)) if w > 0
        )
        return dict(zip(self.actions, self._joint_q(belief, h)))

    def value(self, p_tl, b_j_tl=0.5, horizon=None):
        return max(self.q_values(p_tl, b_j_tl, horizon).values())

    def policy(self, p_tl, b_j_tl=0.5, horizon=None):
        values = self.q_values(p_tl, b_j_tl, horizon)
        return max(values, key=values.get)
