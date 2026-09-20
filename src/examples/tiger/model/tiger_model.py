"""Two-agent Tiger with simultaneous actions and post-transition observations.

Opening rewards depend on the pre-transition tiger. In reset mode, any opening
draws a new uniform tiger before observations. A listening agent hears a noisy
growl about that new state and a creak about the other action; an opening agent
receives silence. Persistent mode suppresses state resets. These choices define
the experiment and must not be silently swapped for another Tiger variant.

The always-listen rollout is an explicit finite-budget heuristic. It is not an
optimal policy and does not repair the planner's interactive filtering errors.
"""

import math
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from core.pomdp_model import AgentID, POMDPModel

# --- CONSTANTS ---
# States
TIGER_LEFT = "TL"
TIGER_RIGHT = "TR"

# Actions
LISTEN = "L"
OPEN_LEFT = "OL"
OPEN_RIGHT = "OR"

# Observation Signals
GROWL_LEFT = "GL"
GROWL_RIGHT = "GR"
CREAK_LEFT = "CL"
CREAK_RIGHT = "CR"
SILENCE = "S"

# Type Aliases for Clarity
TigerState = str
TigerAction = str
TigerObservation = Tuple[str, str]  # (Growl, Creak)


class TigerModel(POMDPModel):
    def __init__(
        self, growl_accuracy: dict = None, creak_accuracy: float = 1.0, persistent: bool = False
    ):
        """
        Initializes the Multi-Agent Tiger domain.

        Args:
            growl_accuracy: Dict mapping agent_id to their ability to hear the tiger.
                            Default is 0.85 (standard benchmark).
            creak_accuracy: The probability of correctly hearing the opponent open a door.
            persistent: If True, the tiger does not reset its position after a door is opened.
        """
        if growl_accuracy is None:
            self.growl_acc = {"i": 0.85, "j": 0.85}
        else:
            self.growl_acc = dict(growl_accuracy)

        if set(self.growl_acc) != {"i", "j"}:
            raise ValueError("Tiger growl accuracy must specify agents i and j")
        if any(
            not math.isfinite(p) or not 0 <= p <= 1
            for p in [*self.growl_acc.values(), creak_accuracy]
        ):
            raise ValueError("Sensor accuracies must lie in [0,1]")
        self.creak_acc = creak_accuracy
        self.persistent = persistent

    def get_initial_state(self, rng=None) -> TigerState:
        """The episode begins with the tiger uniformly distributed."""
        choice_fn = rng.choice if rng is not None else random.choice
        return choice_fn([TIGER_LEFT, TIGER_RIGHT])

    def sample_transition(
        self, state: TigerState, joint_action: Dict[AgentID, TigerAction], rng=None
    ) -> TigerState:
        """
        If *anyone* opens a door, the episode conceptually resets, and the tiger
        is placed randomly behind one of the two doors (unless persistent is True).
        """
        if not self.persistent:
            for act in joint_action.values():
                if act in (OPEN_LEFT, OPEN_RIGHT):
                    choice_fn = rng.choice if rng is not None else random.choice
                    return choice_fn([TIGER_LEFT, TIGER_RIGHT])
        return state

    def transition_distribution(self, state, joint_action):
        """Exact support: persistent/listening dynamics or an independent uniform reset."""
        if not self.persistent and any(a in (OPEN_LEFT, OPEN_RIGHT) for a in joint_action.values()):
            return ((TIGER_LEFT, 0.5), (TIGER_RIGHT, 0.5))
        return ((state, 1.0),)

    def sample_observation(
        self,
        state: TigerState,
        joint_action: Dict[AgentID, TigerAction],
        agent_id: AgentID,
        rng=None,
    ) -> TigerObservation:
        """
        Generative observation model. Used during MCTS rollout simulation and environment stepping.
        """
        my_action = joint_action.get(agent_id, LISTEN)

        # If I open a door, the noise deafens me. I hear absolutely nothing.
        if my_action != LISTEN:
            return (SILENCE, SILENCE)

        # Identify the opponent's action (assumes 2-player for this logic)
        other_action = LISTEN
        for aid, act in joint_action.items():
            if aid != agent_id:
                other_action = act
                break

        rand_fn = rng.random if rng is not None else random.random

        # 1. Sample Growl (Tiger Noise)
        acc = self.growl_acc.get(agent_id, 0.85)
        if state == TIGER_LEFT:
            obs_growl = GROWL_LEFT if rand_fn() < acc else GROWL_RIGHT
        else:
            obs_growl = GROWL_RIGHT if rand_fn() < acc else GROWL_LEFT

        # 2. Sample Creak (Opponent Door Noise)
        r = rand_fn()
        err = max(0.0, 1.0 - self.creak_acc)
        p_wrong = min(0.05, err)
        if other_action == LISTEN:
            obs_creak = SILENCE
        elif other_action == OPEN_LEFT:
            if r < self.creak_acc:
                obs_creak = CREAK_LEFT
            elif r < self.creak_acc + p_wrong:
                obs_creak = CREAK_RIGHT
            else:
                obs_creak = SILENCE
        elif other_action == OPEN_RIGHT:
            if r < self.creak_acc:
                obs_creak = CREAK_RIGHT
            elif r < self.creak_acc + p_wrong:
                obs_creak = CREAK_LEFT
            else:
                obs_creak = SILENCE
        else:
            obs_creak = SILENCE

        return (obs_growl, obs_creak)

    def observation_distribution(self, state, joint_action, agent_id):
        """Reuse the finite sensor law without changing probabilities or ordering.

        Recursive filtering queries the same two-state/action sensor law millions
        of times. Memoizing its immutable tuple avoids repeating the full token
        enumeration. Physics is fixed during a bank's lifetime, as required by
        all policy/filter caches. This bounded class-level cache adds no mutable
        fields to physics fingerprints, consumes no RNG, and retains at most 512
        keys across model instances. It does not approximate any belief mass.
        """
        return self._observation_distribution_cached(state, tuple(joint_action.items()), agent_id)

    @lru_cache(maxsize=512)
    def _observation_distribution_cached(self, state, joint_items, agent_id):
        return super().observation_distribution(state, dict(joint_items), agent_id)

    def get_observation_prob(
        self,
        observation: TigerObservation,
        state: TigerState,
        joint_action: Dict[AgentID, TigerAction],
        agent_id: AgentID,
    ) -> float:
        """
        Evaluative observation model. Used to re-weight beliefs in the Particle Filter.
        MUST perfectly match the probability distribution generated by `sample_observation`.
        """
        obs_growl, obs_creak = observation
        my_action = joint_action.get(agent_id, LISTEN)

        # Deafened agents have 100% probability of hearing Silence
        if my_action != LISTEN:
            return 1.0 if observation == (SILENCE, SILENCE) else 0.0

        other_action = LISTEN
        for aid, act in joint_action.items():
            if aid != agent_id:
                other_action = act
                break

        # 1. Evaluate Growl Probability
        acc = self.growl_acc.get(agent_id, 0.85)
        if state == TIGER_LEFT:
            if obs_growl == GROWL_LEFT:
                p_growl = acc
            elif obs_growl == GROWL_RIGHT:
                p_growl = 1.0 - acc
            else:
                p_growl = 0.0
        else:
            if obs_growl == GROWL_RIGHT:
                p_growl = acc
            elif obs_growl == GROWL_LEFT:
                p_growl = 1.0 - acc
            else:
                p_growl = 0.0

        # 2. Evaluate Creak Probability
        err = max(0.0, 1.0 - self.creak_acc)
        p_wrong = min(0.05, err)
        p_silence = max(0.0, err - p_wrong)

        if other_action == LISTEN:
            p_creak = 1.0 if obs_creak == SILENCE else 0.0
        elif other_action == OPEN_LEFT:
            if obs_creak == CREAK_LEFT:
                p_creak = self.creak_acc
            elif obs_creak == CREAK_RIGHT:
                p_creak = p_wrong
            elif obs_creak == SILENCE:
                p_creak = p_silence
            else:
                p_creak = 0.0
        elif other_action == OPEN_RIGHT:
            if obs_creak == CREAK_RIGHT:
                p_creak = self.creak_acc
            elif obs_creak == CREAK_LEFT:
                p_creak = p_wrong
            elif obs_creak == SILENCE:
                p_creak = p_silence
            else:
                p_creak = 0.0
        else:
            p_creak = 1.0 if obs_creak == SILENCE else 0.0

        # Assume conditional independence between Tiger noise and Opponent noise
        return p_growl * p_creak

    def get_reward(
        self,
        state: TigerState,
        joint_action: Dict[AgentID, TigerAction],
        next_state: TigerState,
        agent_id: AgentID,
    ) -> float:
        """
        Standard Tiger Benchmark Rewards:
        Listen = -1
        Open Door w/ Tiger = -100
        Open Door w/ Gold = +10
        """
        my_action = joint_action.get(agent_id)

        if my_action == LISTEN:
            return -1.0

        if my_action == OPEN_LEFT:
            return -100.0 if state == TIGER_LEFT else 10.0

        if my_action == OPEN_RIGHT:
            return -100.0 if state == TIGER_RIGHT else 10.0

        return 0.0

    def is_terminal(self, state: TigerState) -> bool:
        """Tiger is an infinite-horizon task. The state resets, it never terminates."""
        return False

    def get_all_actions(self, agent_id: AgentID) -> List[TigerAction]:
        """Satisfies POMDPModel abstract method."""
        return [LISTEN, OPEN_LEFT, OPEN_RIGHT]

    def get_all_observations(self, agent_id: AgentID) -> List[TigerObservation]:
        """
        Satisfies POMDPModel abstract method.
        Returns the Cartesian product of all possible (Growl, Creak) tuples.
        """
        growls = [GROWL_LEFT, GROWL_RIGHT, SILENCE]
        creaks = [CREAK_LEFT, CREAK_RIGHT, SILENCE]
        return [(g, c) for g in growls for c in creaks]

    def get_rollout_action(
        self, state: TigerState, agent_id: AgentID, belief: Optional[Dict[TigerState, float]] = None
    ) -> TigerAction:
        """Information-seeking rollout policy.

        Conditions strictly on private belief confidence without inspecting hidden state.
        When confidence is high (>= 92%, corresponding to >= 2 consistent growls),
        it opens the safe door opposite the tiger to collect treasure (+10).
        Otherwise, it listens to gather evidence (-1).
        """
        if belief is not None:
            p_tl = belief.get(TIGER_LEFT, 0.5)
            p_tr = belief.get(TIGER_RIGHT, 0.5)
            if p_tl >= 0.92:
                return OPEN_RIGHT
            elif p_tr >= 0.92:
                return OPEN_LEFT

        return LISTEN

    def update_rollout_belief(
        self,
        belief: Optional[Dict[TigerState, float]],
        action: TigerAction,
        observation: TigerObservation,
        agent_id: AgentID,
    ) -> Dict[TigerState, float]:
        """Bayesian update for private physical belief during MCTS rollout simulation.

        Only conditions on the agent's own action and private observation.
        Door opening uniformly resets the physical tiger.
        """
        if belief is None:
            belief = {TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}

        if action in (OPEN_LEFT, OPEN_RIGHT):
            return {TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}

        obs_growl, _ = observation
        acc = self.growl_acc.get(agent_id, 0.85)

        p_tl = belief.get(TIGER_LEFT, 0.5)
        p_tr = belief.get(TIGER_RIGHT, 0.5)

        if obs_growl == GROWL_LEFT:
            p_tl_unnorm = p_tl * acc
            p_tr_unnorm = p_tr * (1.0 - acc)
        elif obs_growl == GROWL_RIGHT:
            p_tl_unnorm = p_tl * (1.0 - acc)
            p_tr_unnorm = p_tr * acc
        else:
            p_tl_unnorm = p_tl
            p_tr_unnorm = p_tr

        total = p_tl_unnorm + p_tr_unnorm
        if total <= 0.0:
            return {TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}

        return {
            TIGER_LEFT: p_tl_unnorm / total,
            TIGER_RIGHT: p_tr_unnorm / total,
        }
