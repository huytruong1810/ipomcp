"""Exhaustive Tiger L2 response to a declared finite-computation L1 policy.

The opponent policy is defined by a fixed SolverBank, including its seed and
registered modeled planner. Exact means enumeration of the protagonist's finite
decision problem conditional on that policy, not an optimal opponent. Private
models retain their full immutable representation: reconstructing a scalar
belief can perturb deterministic opponent search seeds.

The outer Bellman enumeration below is separate from the production filter.
Opponent subjective updates deliberately use the bank's update rule because
that rule defines the private models at which its finite policy is queried.
Independent small-case tests check that shared rule against scalar enumeration.
"""

import math
from collections import defaultdict
from functools import lru_cache

from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import InferenceBudgetExceeded, checked_distribution


class FinitePolicyL2Reference:
    """Finite-horizon best response with one maximum per private observation.

    Physics and policy providers must remain fixed for this instance's lifetime.
    Cache eviction changes cost only; budget exhaustion raises rather than
    substituting an approximate posterior or a different opponent.
    """

    def __init__(self, physics, bank, gamma=0.95, max_branches=100000):
        if not isinstance(physics, TigerModel):
            raise TypeError("Finite-policy L2 reference supports Tiger only")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Discount must lie in [0,1]")
        if type(max_branches) is not int or max_branches < 1:
            raise ValueError("Branch budget must be a positive integer")
        self.physics, self.bank, self.gamma = physics, bank, gamma
        self.max_branches = max_branches
        self.actions = tuple(physics.get_all_actions("i"))
        self._q = lru_cache(maxsize=1024)(self._q)
        self.branches = lru_cache(maxsize=1024)(self.branches)

    def _check_model(self, model):
        if (
            model.frame.agent_id != "i"
            or model.frame.level != 2
            or model.frame.pomdp_model is not self.physics
        ):
            raise ValueError("Reference requires i/L2 with its declared Tiger physics")
        for atom, _ in model.belief.mass:
            other = atom.opponent
            if (
                atom.state not in (TIGER_LEFT, TIGER_RIGHT)
                or other.frame.agent_id != "j"
                or other.frame.level != 1
                or other.frame.pomdp_model is not self.physics
            ):
                raise ValueError("Reference requires Tiger states and a pure j/L1 opponent")
            for private, _ in other.belief.mass:
                frame = private.opponent.frame
                if (
                    private.state not in (TIGER_LEFT, TIGER_RIGHT)
                    or frame.agent_id != "i"
                    or frame.level != 0
                    or frame.pomdp_model is not self.physics
                ):
                    raise ValueError(
                        "The L1 opponent must model uniform i/L0 under the same physics"
                    )

    def branches(self, model, action):
        """Return reward and (own observation, evidence, full successor model).

        Hidden opponent action/observation are marginalized before constructing
        the next decision problem. The opponent update receives its own action
        and observation only, never the true outer state or i's action.
        """
        self._check_model(model)
        if action not in self.actions:
            raise ValueError("Unknown protagonist action")
        rewards = []
        rows = defaultdict(list)
        count = 0
        for atom, prior in model.belief.mass:
            opponent = atom.opponent
            for other_action, policy_mass in self.bank.filter.action_distribution(
                opponent, atom.state
            ):
                joint = {"i": action, "j": other_action}
                for following, transition in checked_distribution(
                    self.physics.transition_distribution(atom.state, joint), "Transition"
                ):
                    event = prior * policy_mass * transition
                    rewards.append(
                        event * self.physics.get_reward(atom.state, joint, following, "i")
                    )
                    for private, private_mass in checked_distribution(
                        self.physics.observation_distribution(following, joint, "j"),
                        "Private observation",
                    ):
                        posterior = self.bank.filter.update(
                            opponent, other_action, private, terminal=False
                        )
                        advanced = MentalModel(opponent.frame, posterior.belief)
                        for observation, likelihood in checked_distribution(
                            self.physics.observation_distribution(following, joint, "i"),
                            "Observation",
                        ):
                            count += 1
                            if count > self.max_branches:
                                raise InferenceBudgetExceeded("L2 reference branch budget exceeded")
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

    def _q(self, model, horizon):
        if horizon == 0:
            return tuple(0.0 for _ in self.actions)
        if horizon == 1:
            # No next private state is needed at the terminal decision horizon.
            # Rewards are independently integrated, not copied from planner Q.
            values = []
            for action in self.actions:
                terms = []
                for atom, prior in model.belief.mass:
                    for other_action, mass in self.bank.filter.action_distribution(
                        atom.opponent, atom.state
                    ):
                        joint = {"i": action, "j": other_action}
                        for following, transition in checked_distribution(
                            self.physics.transition_distribution(atom.state, joint), "Transition"
                        ):
                            terms.append(
                                prior
                                * mass
                                * transition
                                * self.physics.get_reward(atom.state, joint, following, "i")
                            )
                values.append(math.fsum(terms))
            return tuple(values)
        values = []
        for action in self.actions:
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
        return dict(zip(self.actions, self._q(model, horizon)))
