"""I-NTMCP on this package's domains: the authors' planner, unchanged, as a baseline.

Reference: J. Schwartz, R. Zhou and H. Kurniawati, "Online Planning for
Interactive-POMDPs using Nested Monte Carlo Tree Search", IROS 2022. Code:
github.com/RDLLab/i-ntmcp at commit 63deec3, vendored as the submodule
``third_party/i-ntmcp``; put that directory on ``PYTHONPATH`` to import this
module (``docs/RELATED_WORK.md`` explains why it is the modern baseline).

What is adapted, and what is not
--------------------------------
Only the domain is adapted: ``POSGAdapter`` presents a kernel-only domain
(``model.pomdp``) through I-NTMCP's ``POSGModel`` interface. The planner,
``NestedSearchTree``, runs as published, with the authors' defaults:
``uct_c`` = r_max - r_min, 1/16 extra particles per step and uniformly random
rollouts.

- **Reinvigoration: two variants.** I-NTMCP refills a belief after a real
  step to about M (1 + 1/16) particles by rejection sampling from the parent
  node. After 3,000 rejections it *force-fills* the rest with particles whose
  own observation contradicts the real one. With deterministic senses (Wumpus)
  and large M this can dominate the belief (RESULTS.md 12.65). The authors'
  domains supply a domain-specific reinvigoration function to keep this rare.
  - ``variant="published"``: the authors' code and defaults exactly, with no
    reinvigoration function.
  - ``variant="tuned"``, the strong baseline. A reinvigoration function from
    the kernels draws s' with probability proportional to
    T(s' | s, a) O_i(o_i | s', a) from the parent particle's state s, and the
    other agent's observation from its kernel. The rejection budget scales with
    the target (``TUNED_REJECTIONS`` per particle), so force-filling is rare.
    This uses our explicit kernels, more than I-NTMCP's generative-model
    contract, so it favours the baseline.

- **Dynamics.** ``step`` draws s' from the transition kernel and each agent's
  observation from its observation kernel independently given (s', a), the
  factorisation the exact filters use. Rewards are ``get_reward``; an episode
  ends at a terminal state. Neither agent observes anything before its first
  action, so the initial joint observation is I-NTMCP's null observation, and
  every agent's initial belief is the domain's common prior (I-NTMCP's model
  class: a common initial belief).
- **Levels.** I-NTMCP's level 0 is a POMDP planner facing a uniformly random
  opponent, where this package's level 0 *is* the uniformly random agent. Our
  level L is therefore their nesting level L - 1.
- **Horizon.** The H-step lookahead of this package's planners is set through
  I-NTMCP's public discount horizon: a search stops below depth
  ceil(log ε / log γ), so ε = γ^(H - 1.5) gives depth H - 1 at every nesting
  level. (Their ``step_limit`` would not: nested trees never advance their step
  count.)
- **Determinism.** I-NTMCP draws from the process-global ``random`` module, so
  ``INTMCPAgent`` seeds it, and the adapter's own draws use a seeded
  ``random.Random``. Agent action spaces are fixed lists in domain order; the
  library's default would build them from a set, whose order varies with hash
  randomisation.
"""

import math
import random
import time

import intmcp.model as M
from intmcp.tree.nst import NestedSearchTree

from ipomdp.planning.planner import Planner


class _Wrapped:
    """A domain value presented as an I-NTMCP state, action or observation."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return type(other) is type(self) and other.value == self.value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


class _State(_Wrapped, M.State):
    __slots__ = ()


class _Action(_Wrapped, M.Action):
    __slots__ = ()


class _Observation(_Wrapped, M.Observation):
    __slots__ = ()


class POSGAdapter(M.POSGModel):
    """A two-agent kernel domain as an I-NTMCP POSG; agent index k is ``agents[k]``."""

    def __init__(self, physics, agents, seed, reinvigorate=False):
        super().__init__(num_agents=2)
        self._reinvigorates = reinvigorate
        if type(seed) is not int:
            raise TypeError("The adapter needs an explicit integer seed")
        self.physics, self.agents = physics, tuple(agents)
        self.rng = random.Random(seed)
        self.agent_action_spaces = [
            [_Action(a) for a in physics.get_all_actions(k)] for k in agents
        ]
        self.agent_obs_spaces = [
            [_Observation(o) for o in physics.get_all_observations(k)] for k in agents
        ]
        # A sampler, as in the authors' own environments (``M.Belief.sample``
        # indexes its dict by position and fails).
        self._prior = M.InitialParticleBelief(self.sample_initial_state)
        lows, highs = zip(*(physics.reward_bounds(k) for k in agents))
        self._bounds = (min(lows), max(highs))
        # The state the latest ``step`` started from: I-NTMCP calls the
        # reinvigoration function right after stepping a parent particle, with
        # only the resulting state, so the exact conditional needs this.
        self._last_source = None

    @property
    def state_space(self):
        return [_State(s) for s, _ in self.physics.initial_distribution()]

    @property
    def obs_space(self):
        return M.JointObservation.construct_joint_obs_space(self.agent_obs_spaces)

    @property
    def action_space(self):
        return M.JointAction.construct_joint_action_space(self.agent_action_spaces)

    @property
    def initial_belief(self):
        return self._prior

    @property
    def r_min(self):
        return self._bounds[0]

    @property
    def r_max(self):
        return self._bounds[1]

    def get_agent_action_space(self, agent_id):
        return self.agent_action_spaces[agent_id]

    def get_agent_obs_space(self, agent_id):
        return self.agent_obs_spaces[agent_id]

    def get_init_belief(self, agent_id, obs):
        return self._prior

    def sample_initial_state(self):
        return _State(self.physics.sample_initial_state(self.rng))

    def sample_initial_obs(self, state):
        return M.JointObservation.get_joint_null_obs(self.num_agents)

    @property
    def reinvigorate_fn(self):
        return self._reinvigorate if self._reinvigorates else None

    def _reinvigorate(self, agent_id, state, action, obs):
        """A successor of the parent particle consistent with ``obs``: s' ~ T(s' | s, a) O(obs | s', a)."""
        joint = {k: action.get_agent_action(n).value for n, k in enumerate(self.agents)}
        observer = self.agents[agent_id]
        weighted = [
            (
                following,
                p * self.physics.observation_probability(obs.value, following, joint, observer),
            )
            for following, p in self.physics.transition_distribution(self._last_source, joint)
        ]
        weighted = [(s, w) for s, w in weighted if w > 0]
        if not weighted:
            raise IndexError("No successor of this particle explains the observation")
        states, weights = zip(*weighted)
        following = self.rng.choices(states, weights=weights, k=1)[0]
        observations = tuple(
            obs
            if n == agent_id
            else _Observation(self.physics.sample_observation(following, joint, k, self.rng))
            for n, k in enumerate(self.agents)
        )
        return _State(following), M.JointObservation(observations)

    def step(self, state, action):
        self._last_source = state.value
        joint = {k: action.get_agent_action(n).value for n, k in enumerate(self.agents)}
        following = self.physics.sample_transition(state.value, joint, self.rng)
        observations = tuple(
            _Observation(self.physics.sample_observation(following, joint, k, self.rng))
            for k in self.agents
        )
        rewards = [self.physics.get_reward(state.value, joint, following, k) for k in self.agents]
        return (
            _State(following),
            M.JointObservation(observations),
            rewards,
            self.physics.is_terminal(following),
        )

    def is_terminal(self, state):
        return self.physics.is_terminal(state.value)

    def get_outcome(self, state):
        return [None] * self.num_agents


TUNED_REJECTIONS = 50
"""The tuned variant's rejection budget per particle of a belief's refill target."""

VARIANTS = ("published", "tuned")


class INTMCPAgent:
    """An I-NTMCP agent of this package's ``level`` (their nesting ``level - 1``) at the prior.

    ``search()`` runs one I-NTMCP search from the current history and returns
    the chosen action; ``update_root(action, observation)`` conditions on one
    real step. I-NTMCP builds the next step's beliefs from the particles its
    search leaves, so a history is replayed as search, update, search, ...
    ``values`` holds the root's action values after the last search, and
    ``search_cpu_seconds`` its process CPU. ``forced_shares`` records, per real
    update, the share of the new root belief whose own history contradicts the
    real one (force-filled particles). ``uct_c`` overrides the authors' default
    exploration constant, r_max - r_min (None keeps it), to match a published
    configuration.
    """

    def __init__(
        self,
        physics,
        level,
        simulations,
        horizon,
        gamma,
        seed,
        variant="published",
        agents=("i", "j"),
        uct_c=None,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if level < 1:
            raise ValueError("An I-NTMCP agent models its opponent: level must be at least 1")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        random.seed(seed)
        self.model = POSGAdapter(physics, agents, seed, reinvigorate=variant == "tuned")
        self.tree = NestedSearchTree.initialize(
            model=self.model,
            ego_agent=0,
            gamma=gamma,
            nesting_level=level - 1,
            num_sims=simulations,
            epsilon=gamma ** (horizon - 1.5),
            uct_c=uct_c,
        )
        if variant == "tuned":
            target = simulations * 17 / 16
            limit = math.ceil(TUNED_REJECTIONS * target / NestedSearchTree.SAMPLE_RETRY_LIMIT)
            for tree in _trees(self.tree):
                tree.SAMPLE_FAIL_LIMIT = limit
        self.values, self.search_cpu_seconds, self.forced_shares = {}, 0.0, []
        self._started = False

    def search(self):
        started = time.process_time()
        if self._started:
            action = self.tree.get_action()
        else:
            action = self.tree.step(M.Observation.get_null_obs())
            self._started = True
        self.search_cpu_seconds = time.process_time() - started
        root = self.tree.traverse(self.tree.history)
        self.values = {node.h.value: node.v for node in root.children if node.n > 0}
        return action.value

    def update_root(self, action, observation):
        if not self._started:
            raise RuntimeError("I-NTMCP updates from the particles of a search: search first")
        self.tree.update(_Action(action), _Observation(observation))
        particles = self.tree.traverse(self.tree.history).belief.particles
        own = self.tree.history
        forced = sum(p.history.get_agent_history(0) != own for p in particles)
        self.forced_shares.append(forced / len(particles) if particles else 1.0)


class INTMCPPlanner(Planner):
    """An ``INTMCPAgent`` behind the episode interface of ``experiments.episodes``.

    It holds no ``belief`` attribute, so episodes record no belief columns or
    prediction for it. ``get_detailed_stats`` reports the root action values of
    the last search and ``forced_share``, the share of force-filled particles in
    the belief that search started from (0 at the prior).
    """

    def __init__(self, agent):
        self.agent = agent

    def get_action(self, belief=None):
        return self.agent.search()

    def update_root(self, action, observation):
        self.agent.update_root(action, observation)

    def get_detailed_stats(self):
        shares = self.agent.forced_shares
        return {
            "action_values": {str(a): v for a, v in self.agent.values.items()},
            "forced_share": shares[-1] if shares else 0.0,
        }


def _trees(tree):
    """``tree`` and every tree nested below it."""
    yield tree
    for nested in tree.nested_trees.values():
        yield from _trees(nested)
