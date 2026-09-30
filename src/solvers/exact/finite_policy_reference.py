"""Exhaustive Tiger best response to a bank's finite opponent policies, any level.

This generalizes FinitePolicyL2Reference to either protagonist and to beliefs
mixing opponent levels (for example L0/L1/L2 opponents of an L3 agent). Exact
means enumeration of the protagonist's finite-horizon decision problem
conditional on the bank's declared policies: uniform L0 and its registered
finite modeled planners. It is not an optimal-opponent or equilibrium solution.

Opponent subjective updates use the bank's filter, as in the L2 reference,
because that rule defines the private models at which the finite policies are
queried. Uniform L0 opponents have no belief and remain unchanged. The outer
Bellman enumeration is independent of the planner's search code.
"""

import math
from collections import defaultdict
from functools import lru_cache

from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import InferenceBudgetExceeded, checked_distribution


class FinitePolicyReference:
    """Finite-horizon best response with one maximum per own observation history.

    Physics and the bank's policy providers must stay fixed for this instance's
    lifetime. Budget exhaustion raises rather than approximating.
    """

    def __init__(self, physics, bank, gamma=0.95, max_branches=200000):
        if not isinstance(physics, TigerModel):
            raise TypeError("Finite-policy reference supports Tiger only")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Discount must lie in [0,1]")
        if type(max_branches) is not int or max_branches < 1:
            raise ValueError("Branch budget must be a positive integer")
        self.physics, self.bank, self.gamma = physics, bank, gamma
        self.max_branches = max_branches
        self._q = lru_cache(maxsize=4096)(self._q)
        self.branches = lru_cache(maxsize=4096)(self.branches)

    def _check_model(self, model):
        frame = model.frame
        if frame.level < 1 or frame.pomdp_model is not self.physics:
            raise ValueError("Reference requires an intentional agent with the declared physics")
        for atom, _ in model.belief.mass:
            other = atom.opponent.frame
            if (
                atom.state not in (TIGER_LEFT, TIGER_RIGHT)
                or other.agent_id == frame.agent_id
                or other.level >= frame.level
                or other.pomdp_model is not self.physics
            ):
                raise ValueError("Opponents must be lower-level Tiger models of the other agent")

    def _joint(self, model, action, other_action, opponent):
        return {model.frame.agent_id: action, opponent.frame.agent_id: other_action}

    def branches(self, model, action):
        """Return expected reward and (own observation, evidence, successor model).

        Opponent actions and private observations are marginalized. Intentional
        opponents update on their own action/observation only; L0 is unchanged.
        """
        self._check_model(model)
        me = model.frame.agent_id
        if action not in self.physics.get_all_actions(me):
            raise ValueError("Unknown protagonist action")
        rewards, rows, count = [], defaultdict(list), 0
        for atom, prior in model.belief.mass:
            opponent = atom.opponent
            other = opponent.frame.agent_id
            for other_action, policy_mass in self.bank.filter.action_distribution(
                opponent, atom.state
            ):
                joint = self._joint(model, action, other_action, opponent)
                for following, transition in checked_distribution(
                    self.physics.transition_distribution(atom.state, joint), "Transition"
                ):
                    event = prior * policy_mass * transition
                    rewards.append(
                        event * self.physics.get_reward(atom.state, joint, following, me)
                    )
                    if opponent.frame.level == 0:
                        privates = ((None, 1.0),)
                    else:
                        privates = checked_distribution(
                            self.physics.observation_distribution(following, joint, other),
                            "Private observation",
                        )
                    for private, private_mass in privates:
                        if private is None:
                            advanced = opponent
                        else:
                            posterior = self.bank.filter.update(
                                opponent, other_action, private, terminal=False
                            )
                            advanced = MentalModel(opponent.frame, posterior.belief)
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
        branches = []
        for observation, entries in rows.items():
            evidence = math.fsum(weight for _, weight in entries)
            branches.append(
                (observation, evidence, MentalModel(model.frame, FiniteBelief(tuple(entries))))
            )
        if not math.isclose(
            math.fsum(item[1] for item in branches), 1, abs_tol=1e-12, rel_tol=1e-12
        ):
            raise ValueError("Reference observation branches do not conserve probability")
        return math.fsum(rewards), tuple(branches)

    def _immediate(self, model):
        me = model.frame.agent_id
        values = []
        for action in self.physics.get_all_actions(me):
            terms = []
            for atom, prior in model.belief.mass:
                for other_action, mass in self.bank.filter.action_distribution(
                    atom.opponent, atom.state
                ):
                    joint = self._joint(model, action, other_action, atom.opponent)
                    for following, transition in checked_distribution(
                        self.physics.transition_distribution(atom.state, joint), "Transition"
                    ):
                        terms.append(
                            prior
                            * mass
                            * transition
                            * self.physics.get_reward(atom.state, joint, following, me)
                        )
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
                mass * max(self._q(successor, horizon - 1)) for _, mass, successor in branches
            )
            values.append(reward + self.gamma * continuation)
        return tuple(values)

    def q_values(self, model, horizon):
        self._check_model(model)
        if type(horizon) is not int or horizon < 0:
            raise ValueError("Horizon must be a nonnegative integer")
        return dict(
            zip(self.physics.get_all_actions(model.frame.agent_id), self._q(model, horizon))
        )
