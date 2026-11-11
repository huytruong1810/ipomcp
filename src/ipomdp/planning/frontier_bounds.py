"""Belief-dependent frontier bounds: the interactive fast-informed bound (I-FIB).

Why
---
Certified scoring (``chance_scoring``) must bound what search left unknown:
unsampled observation branches, rollout frontiers and untried actions. The
scalar bounds of ``chance_scoring._Span`` use only the declared reward range
and the steps to go, so they ignore the belief and the end of the episode. In
UAV no return exceeds the spotting reward (50), yet the scalar upper bound is
142.6 at three steps to go; that alone sets the certified widths of the
paper's H = 4 and 5 cells.

Construction (audit spec 2.1, "Frontier soundness")
---------------------------------------------------
Project onto physical states. The first step uses the modeled action law of
every atom of b, which the planner holds; afterwards the other agent's
actions are chosen by a state-informed helper (upper bound) or adversary
(lower bound). With R-bar(s, a_i, a_j) the expected immediate reward
(terminal successors included) and the recursion over r steps to go,

    F^r(s, a_i, a_j) = R-bar + gamma * sum_o max_a' sum_{s' alive} T O_i Qbar^{r-1}(s', a')
    Qbar^r(s, a_i)   = max_{a_j} F^r(s, a_i, a_j)
    alpha^r_a(s)     = min_{a_j} [R-bar(s, a, a_j) + gamma * sum_{s' alive} T alpha^{r-1}_a(s')]

and, for a belief b with w_b(s, a_j) = sum_m b(s, m) pi_j(a_j | m) and the
unnormalized alive posterior mu_{b,a,o}(s') = sum_{s,a_j} w_b T O_i,

    Q^r_I-FIB(b, a) = ER(b, a) + gamma * sum_o max_a' sum_s' mu(s') Qbar^{r-1}(s', a')
    L^r(b, a)       = ER(b, a) + gamma * sum_o max_a' sum_s' mu(s') alpha^{r-1}_{a'}(s')

satisfy L^r <= Q^r <= Q^r_I-FIB for the exact value Q^r of the searched
model (own belief not compressed inside the lookahead). The per-branch terms
bound p_o * V^{r-1}(b_ao) from the parent's belief, without computing b_ao.

The other agent's own observations and model updates never enter: the helper
and the adversary see the state, which dominates any model of the other agent.

This is the reference path: exact Python floats with ``math.fsum``, memoized
per physical state. Equal inputs give equal bounds; nothing here is random.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

from ipomdp.belief.finite_filter import checked_distribution

KINDS = ("ifib",)


@dataclass(frozen=True)
class NodeBounds:
    """Bounds at one belief with r steps to go.

    Attributes:
        q_lo: {a: L^r(b, a)}.
        q_hi: {a: Q^r_I-FIB(b, a)}.
        branch_lo: {a: {o: lower bound on p_o * V^{r-1}(b_ao)}}, probability-weighted.
        branch_hi: {a: {o: upper bound on p_o * V^{r-1}(b_ao)}}, probability-weighted.
    """

    q_lo: dict
    q_hi: dict
    branch_lo: dict
    branch_hi: dict


class FrontierBounds:
    """I-FIB tables for one agent of one physics, shared by every belief of that agent."""

    def __init__(self, physics, agent_id, other_id, gamma, depth, kind="ifib", cache_size=65536):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.physics, self.agent_id, self.other_id = physics, agent_id, other_id
        self.gamma, self.depth, self.kind = gamma, depth, kind
        self.actions = tuple(physics.get_all_actions(agent_id))
        self.other_actions = tuple(physics.get_all_actions(other_id))
        self._kernels, self._qbar, self._alpha = {}, {}, {}
        self.node_bounds = lru_cache(maxsize=cache_size)(self._node_bounds)

    def clear_caches(self):
        """Drop per-belief bounds (memory only; the state tables are kept)."""
        self.node_bounds.cache_clear()

    # ------------------------------------------------------------------ #
    # Physical tables
    # ------------------------------------------------------------------ #

    def _kernel(self, state, action, other_action):
        """(R-bar, ((s', T), ...) alive, {o: ((s', T * O_i), ...)}) for one state and joint action."""
        key = (state, action, other_action)
        cached = self._kernels.get(key)
        if cached is not None:
            return cached
        joint = {self.agent_id: action, self.other_id: other_action}
        rewards, alive, by_observation = [], [], {}
        for following, transition in checked_distribution(
            self.physics.transition_distribution(state, joint), "Transition"
        ):
            rewards.append(
                transition * self.physics.get_reward(state, joint, following, self.agent_id)
            )
            if self.physics.is_terminal(following):
                continue
            alive.append((following, transition))
            for observation, likelihood in checked_distribution(
                self.physics.observation_distribution(following, joint, self.agent_id),
                "Observation",
            ):
                by_observation.setdefault(observation, []).append(
                    (following, transition * likelihood)
                )
        kernel = (
            math.fsum(rewards),
            tuple(alive),
            {o: tuple(rows) for o, rows in by_observation.items()},
        )
        self._kernels[key] = kernel
        return kernel

    def qbar(self, r, state):
        """Qbar^r(s, .) over the agent's actions (partner-optimistic, state-informed)."""
        if r == 0:
            return tuple(0.0 for _ in self.actions)
        key = (r, state)
        cached = self._qbar.get(key)
        if cached is not None:
            return cached
        values = []
        for action in self.actions:
            best = -math.inf
            for other_action in self.other_actions:
                reward, _, by_observation = self._kernel(state, action, other_action)
                continuation = math.fsum(
                    max(
                        math.fsum(w * self.qbar(r - 1, s)[k] for s, w in rows)
                        for k in range(len(self.actions))
                    )
                    for rows in by_observation.values()
                )
                best = max(best, reward + self.gamma * continuation)
            values.append(best)
        values = tuple(values)
        self._qbar[key] = values
        return values

    def alpha_lo(self, r, state):
        """alpha^r_a(s) over the agent's actions: repeat a, adversarial partner."""
        if r == 0:
            return tuple(0.0 for _ in self.actions)
        key = (r, state)
        cached = self._alpha.get(key)
        if cached is not None:
            return cached
        values = []
        for k, action in enumerate(self.actions):
            worst = math.inf
            for other_action in self.other_actions:
                reward, alive, _ = self._kernel(state, action, other_action)
                continuation = math.fsum(t * self.alpha_lo(r - 1, s)[k] for s, t in alive)
                worst = min(worst, reward + self.gamma * continuation)
            values.append(worst)
        values = tuple(values)
        self._alpha[key] = values
        return values

    # ------------------------------------------------------------------ #
    # Per-belief bounds
    # ------------------------------------------------------------------ #

    def _node_bounds(self, model, r, bank):
        """``NodeBounds`` of ``model`` with ``r`` steps to go (``bank`` supplies pi_j and ER)."""
        if model.frame.agent_id != self.agent_id or model.frame.pomdp_model is not self.physics:
            raise ValueError("Bounds were built for another agent or physics")
        if type(r) is not int or not 1 <= r <= self.depth:
            raise ValueError("Steps to go must lie in [1, depth]")
        rewards = dict(bank.expected_rewards(model))
        if r == 1:
            return NodeBounds(dict(rewards), dict(rewards), {}, {})
        weights = {}
        for atom, mass in model.belief.mass:
            if self.physics.is_terminal(atom.state):
                continue
            if atom.opponent.frame.agent_id != self.other_id:
                raise ValueError("Bounds were built for another opponent")
            for other_action, probability in bank.filter.action_distribution(atom.opponent):
                weights.setdefault((atom.state, other_action), []).append(mass * probability)
        weights = {key: math.fsum(terms) for key, terms in weights.items()}
        q_lo, q_hi, branch_lo, branch_hi = {}, {}, {}, {}
        n = len(self.actions)
        for action in self.actions:
            mu = {}
            for (state, other_action), weight in weights.items():
                if weight == 0:
                    continue
                _, _, by_observation = self._kernel(state, action, other_action)
                for observation, rows in by_observation.items():
                    target = mu.setdefault(observation, {})
                    for following, mass in rows:
                        target.setdefault(following, []).append(weight * mass)
            lows, highs = {}, {}
            for observation, posterior in mu.items():
                posterior = {s: math.fsum(terms) for s, terms in posterior.items()}
                if math.fsum(posterior.values()) <= 0:
                    continue
                highs[observation] = max(
                    math.fsum(p * self.qbar(r - 1, s)[k] for s, p in posterior.items())
                    for k in range(n)
                )
                lows[observation] = max(
                    math.fsum(p * self.alpha_lo(r - 1, s)[k] for s, p in posterior.items())
                    for k in range(n)
                )
            branch_lo[action], branch_hi[action] = lows, highs
            q_lo[action] = rewards[action] + self.gamma * math.fsum(lows.values())
            q_hi[action] = rewards[action] + self.gamma * math.fsum(highs.values())
        return NodeBounds(q_lo, q_hi, branch_lo, branch_hi)
