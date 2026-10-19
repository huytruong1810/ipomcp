"""Exact finite-horizon best response to declared opponent models, any domain and level.

What "exact" means here
-----------------------
This codebase uses three distinct notions, and every error report must say which:

1. **Exact best response** (this module): the agent's finite-horizon decision
   problem is solved by exhaustive enumeration, *conditional on* the opponent
   models declared in the bank. With finite MCTS opponents this isolates the
   agent's own search error.
2. **Exact I-POMDP**: every intentional model, real or nested, is itself an
   exact best response (``reference.exact_agent``), recursively down to L0.
3. **Exact POMDP**: an L1 agent against a fixed L0 is a POMDP, solved
   independently by alpha-vector value iteration (``reference.pomdp_exact_vi``).

Computation
-----------
For own action a, the enumeration sums over the joint belief's atoms
(s, m_j), the opponent's action distribution at m_j, transitions s', the
opponent's private observation o_j (advancing m_j by its own subjective
update), and the agent's own observation o_i. Events are grouped by o_i into
successor beliefs, and the value is

    Q_H(b, a) = E[R | b, a] + gamma * sum_{o_i} P(o_i | b, a) max_a' Q_{H-1}(b_{a o_i}, a'),

with one maximization per own observation history, never per hidden state.
Terminal successors contribute their reward and no continuation.

The opponent's subjective update is the bank's filter, because that update
defines the private models at which the declared policies are queried; it is
validated independently in the filter tests. The Bellman enumeration itself
shares no code with the planner.
"""

import math
from collections import defaultdict
from functools import lru_cache

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import InferenceBudgetExceeded, checked_distribution


class BestResponseReference:
    def __init__(self, physics, bank, gamma, max_branches=200_000, cache_size=4096):
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Discount must lie in [0,1]")
        if type(max_branches) is not int or max_branches < 1:
            raise ValueError("Branch budget must be a positive integer")
        self.physics, self.bank, self.gamma = physics, bank, gamma
        self.max_branches = max_branches
        # Recursive solves revisit many models; eviction changes cost only.
        self._q = lru_cache(maxsize=cache_size)(self._q)
        self.branches = lru_cache(maxsize=cache_size)(self.branches)

    def clear_caches(self):
        """Drop cached values and branches (memory only, never results)."""
        self._q.cache_clear()
        self.branches.cache_clear()

    def q_values(self, model, horizon):
        """{action: Q_horizon(belief, action)} for every action of the agent."""
        self._check_model(model)
        if type(horizon) is not int or horizon < 0:
            raise ValueError("Horizon must be a nonnegative integer")
        return dict(
            zip(self.physics.get_all_actions(model.frame.agent_id), self._q(model, horizon))
        )

    def branches(self, model, action):
        """(expected reward, ((own observation, probability, successor model), ...)).

        Successor probabilities sum to the probability that the episode continues.
        """
        self._check_model(model)
        me = model.frame.agent_id
        rewards, rows, terminal_mass, count = [], defaultdict(list), [], 0
        for atom, prior in model.belief.mass:
            opponent = atom.opponent
            other = opponent.frame.agent_id
            for other_action, policy_mass in self.bank.filter.action_distribution(opponent):
                joint = {me: action, other: other_action}
                for following, transition in checked_distribution(
                    self.physics.transition_distribution(atom.state, joint), "Transition"
                ):
                    event = prior * policy_mass * transition
                    rewards.append(
                        event * self.physics.get_reward(atom.state, joint, following, me)
                    )
                    if self.physics.is_terminal(following):
                        terminal_mass.append(event)
                        continue
                    for advanced, private_mass in self._advanced(
                        opponent, other_action, following, joint
                    ):
                        for observation, likelihood in checked_distribution(
                            self.physics.observation_distribution(following, joint, me),
                            "Observation",
                        ):
                            count += 1
                            if count > self.max_branches:
                                raise InferenceBudgetExceeded("Reference branch budget exceeded")
                            rows[observation].append(
                                (
                                    InteractiveState(following, advanced),
                                    event * private_mass * likelihood,
                                )
                            )
        branches = tuple(
            (
                observation,
                math.fsum(w for _, w in entries),
                MentalModel(model.frame, FiniteBelief(tuple(entries))),
            )
            for observation, entries in rows.items()
        )
        total = math.fsum([*(p for _, p, _ in branches), *terminal_mass])
        if not math.isclose(total, 1, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Reference branches do not conserve probability")
        return math.fsum(rewards), branches

    def _advanced(self, opponent, other_action, following, joint):
        """The opponent's possible successor models with their probabilities."""
        if opponent.frame.level == 0:
            return ((opponent, 1.0),)
        other = opponent.frame.agent_id
        return tuple(
            (
                self.bank.filter.advance(opponent, other_action, private, terminal=False),
                private_mass,
            )
            for private, private_mass in checked_distribution(
                self.physics.observation_distribution(following, joint, other),
                "Private observation",
            )
        )

    def _immediate(self, model):
        """E[R | b, a] for every action, enumerated without the planner's integrator."""
        me = model.frame.agent_id
        values = []
        for action in self.physics.get_all_actions(me):
            terms = []
            for atom, prior in model.belief.mass:
                for other_action, mass in self.bank.filter.action_distribution(atom.opponent):
                    joint = {me: action, atom.opponent.frame.agent_id: other_action}
                    for following, transition in checked_distribution(
                        self.physics.transition_distribution(atom.state, joint), "Transition"
                    ):
                        reward = self.physics.get_reward(atom.state, joint, following, me)
                        terms.append(prior * mass * transition * reward)
            values.append(math.fsum(terms))
        return tuple(values)

    def _q(self, model, horizon):
        actions = self.physics.get_all_actions(model.frame.agent_id)
        if horizon == 0:
            return tuple(0.0 for _ in actions)
        if horizon == 1:
            return self._immediate(model)
        values = []
        for action in actions:
            reward, branches = self.branches(model, action)
            continuation = math.fsum(
                p * max(self._q(successor, horizon - 1)) for _, p, successor in branches
            )
            values.append(reward + self.gamma * continuation)
        return tuple(values)

    def _check_model(self, model):
        frame = model.frame
        if frame.level < 1 or frame.pomdp_model is not self.physics:
            raise ValueError("Reference requires an intentional agent with the declared physics")
        for atom, _ in model.belief.mass:
            if self.physics.is_terminal(atom.state):
                raise ValueError("Condition on continuation before evaluating a belief")
            other = atom.opponent.frame
            if other.agent_id == frame.agent_id or other.level >= frame.level:
                raise ValueError("Opponents must be lower-level models of the other agent")
            if other.pomdp_model is not self.physics:
                raise ValueError("Opponent models must use the declared physics")
