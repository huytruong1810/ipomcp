"""The physical-domain contract every environment implements.

A domain declares finite probability kernels; everything else is derived
------------------------------------------------------------------------
A domain states *what can happen and with what probability*:

- ``initial_distribution()``: the exact prior over physical states;
- ``transition_distribution(s, a)``: P(s' | s, a) for a joint action a;
- ``observation_distribution(s', a, i)``: P(o_i | s', a) for agent i;
- ``get_reward(s, a, s', i)`` with declared ``reward_bounds(i)``.

Sampling (``sample_initial_state``, ``sample_transition``,
``sample_observation``) and likelihoods (``observation_probability``) are
derived from those kernels here, so a sampler can never disagree with the law
that exact belief updates and exact references use. The rejected alternative,
a domain that implements its sampler and its likelihood separately, makes the
two agree only by hand, and a disagreement is invisible: sampled search and
exact filtering would then solve two different games.

A kernel is a tuple of ``(outcome, probability)`` pairs with distinct outcomes
and strictly positive probabilities that sum to one. Zero-probability outcomes
are omitted rather than listed, so that a kernel's listed outcomes are exactly
its support: exhaustive references, reachability enumeration and branch
budgets all count listed outcomes. (``belief.finite_filter.checked_distribution``
drops a zero entry if one is listed; the contract tests reject it.)

Observations depend on the post-transition state and the joint action. Any
transition event an observation must report (for example a bump or a scream)
has to be recorded in the post-transition state.

Information discipline
----------------------
- The action set depends only on the agent: ``get_all_actions(agent_id)``. An
  action availability that varied with the hidden state would let a planner
  branch on information it does not have. Actions that have no effect in some
  states (shooting without an arrow) are allowed and simply have no effect.
- A rollout policy sees only the agent's private rollout memory, never the
  sampled state (``rollout_action``).

Randomness
----------
Every sampling method takes an explicit ``random.Random``. No code in the
package uses the process-global ``random`` module, so independent streams
(environment, each planner) cannot interfere and results do not depend on call
order across components.
"""

import abc
import random
from typing import Any, Dict, Hashable, List, Optional, Tuple

State = Hashable
Action = Hashable
Observation = Hashable
AgentID = Hashable
Distribution = Tuple[Tuple[Any, float], ...]


class POMDPModel(abc.ABC):
    # ------------------------------------------------------------------ #
    # Declared by each domain
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def initial_distribution(self) -> Distribution:
        """Exact prior: ((state, probability), ...), probabilities summing to one."""

    @abc.abstractmethod
    def transition_distribution(
        self, state: State, joint_action: Dict[AgentID, Action]
    ) -> Distribution:
        """P(s' | s, joint action) as ((next_state, probability), ...)."""

    @abc.abstractmethod
    def observation_distribution(
        self, state: State, joint_action: Dict[AgentID, Action], agent_id: AgentID
    ) -> Distribution:
        """P(o | s', joint action) for ``agent_id``, given the post-transition state."""

    @abc.abstractmethod
    def get_reward(
        self,
        state: State,
        joint_action: Dict[AgentID, Action],
        next_state: State,
        agent_id: AgentID,
    ) -> float:
        """R_i(s, joint action, s')."""

    @abc.abstractmethod
    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        """Inclusive (min, max) of ``get_reward`` over every reachable transition.

        The planner's exploration scale and every certified value interval are
        derived from these bounds, so they must hold for all states and joint
        actions, not only the likely ones: a reward outside them makes a
        certificate unsound. They should also be attained, because slack widens
        every interval (``tests/test_domain_exhaustive.py`` checks both).
        """

    @abc.abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Whether the episode has ended; terminal states have no future reward."""

    @abc.abstractmethod
    def get_all_actions(self, agent_id: AgentID) -> List[Action]:
        """The agent's complete, state-independent action set."""

    @abc.abstractmethod
    def get_all_observations(self, agent_id: AgentID) -> List[Observation]:
        """Every observation the agent can receive."""

    def rollout_action(
        self, agent_id: AgentID, memory: Optional[Any], rng: random.Random
    ) -> Action:
        """Default rollout policy: uniform over the agent's actions.

        Domains may override with a heuristic that uses only ``memory``, the
        agent's private rollout memory (see ``update_rollout_memory``).
        """
        return rng.choice(self.get_all_actions(agent_id))

    def update_rollout_memory(
        self, memory: Optional[Any], action: Action, observation: Observation, agent_id: AgentID
    ) -> Optional[Any]:
        """Advance the private rollout memory on the agent's own action and observation.

        The planner initialises the memory with the physical marginal of the
        agent's belief. The default policy needs no memory and keeps it as is.
        """
        return memory

    # ------------------------------------------------------------------ #
    # Derived from the kernels
    # ------------------------------------------------------------------ #

    def sample_initial_state(self, rng: random.Random) -> State:
        return _sample(self.initial_distribution(), rng)

    def sample_transition(
        self, state: State, joint_action: Dict[AgentID, Action], rng: random.Random
    ) -> State:
        return _sample(self.transition_distribution(state, joint_action), rng)

    def sample_observation(
        self,
        state: State,
        joint_action: Dict[AgentID, Action],
        agent_id: AgentID,
        rng: random.Random,
    ) -> Observation:
        return _sample(self.observation_distribution(state, joint_action, agent_id), rng)

    def observation_probability(
        self,
        observation: Observation,
        state: State,
        joint_action: Dict[AgentID, Action],
        agent_id: AgentID,
    ) -> float:
        for value, probability in self.observation_distribution(state, joint_action, agent_id):
            if value == observation:
                return probability
        return 0.0


def _sample(distribution: Distribution, rng: random.Random):
    if len(distribution) == 1:
        return distribution[0][0]
    values, weights = zip(*distribution)
    return rng.choices(values, weights=weights, k=1)[0]
