"""Two-agent Tiger, exactly as specified by Gmytrasiewicz & Doshi (2005).

Source: P. J. Gmytrasiewicz and P. Doshi, "A Framework for Sequential Planning
in Multi-Agent Settings", JAIR 24 (2005), Section 7.1 and Table 1.

States: tiger left (TL) or right (TR); initially 50/50.
Actions (each agent): listen (L), open left (OL), open right (OR).

Transition: if either agent opens a door, the tiger is placed uniformly at
random; otherwise it stays.

Reward (each agent, own action only, pre-transition state): listen -1; opening
the tiger's door -100; opening the other door +10. The other agent's action
affects an agent only through the state and the observations.

Observation (agent k, post-transition state, joint action): a pair
(growl, creak) from {GL, GR} x {CL, CR, S}:
- If k opened a door, all six observations are equally likely (1/6): an
  opening agent learns nothing.
- If k listened, the growl is correct with probability g_k (0.85) and the
  creak reports the other agent's action independently: correct with
  probability c (0.9), each of the two other creak tokens with (1 - c) / 2.
  "Correct" means S when the other agent listened, CL when it opened left and
  CR when it opened right. A listening opponent therefore produces a false
  creak with probability 1 - c.

Level 0 in this codebase is uniformly random (not the paper's L0 POMDP agent);
that convention belongs to the hierarchy builder, not to the physics.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ipomdp.model.pomdp import AgentID, POMDPModel

TIGER_LEFT = "TL"
TIGER_RIGHT = "TR"
STATES = (TIGER_LEFT, TIGER_RIGHT)

LISTEN = "L"
OPEN_LEFT = "OL"
OPEN_RIGHT = "OR"
ACTIONS = (LISTEN, OPEN_LEFT, OPEN_RIGHT)

GROWL_LEFT = "GL"
GROWL_RIGHT = "GR"
CREAK_LEFT = "CL"
CREAK_RIGHT = "CR"
SILENCE = "S"
GROWLS = (GROWL_LEFT, GROWL_RIGHT)
CREAKS = (CREAK_LEFT, CREAK_RIGHT, SILENCE)
OBSERVATIONS = tuple((g, c) for g in GROWLS for c in CREAKS)

# The creak that correctly reports each action of the other agent.
_TRUE_CREAK = {LISTEN: SILENCE, OPEN_LEFT: CREAK_LEFT, OPEN_RIGHT: CREAK_RIGHT}

# Rollout heuristic threshold: open the safe door once the tiger's side is this
# likely. 0.92 is reached after two consistent growls at accuracy 0.85.
ROLLOUT_CONFIDENCE = 0.92


class TigerModel(POMDPModel):
    def __init__(self, growl_accuracy: Optional[Dict[str, float]] = None, creak_accuracy=0.9):
        self.growl_acc = {"i": 0.85, "j": 0.85} if growl_accuracy is None else dict(growl_accuracy)
        if set(self.growl_acc) != {"i", "j"}:
            raise ValueError("Tiger growl accuracy must specify agents i and j")
        if any(
            not math.isfinite(p) or not 0 <= p <= 1
            for p in [*self.growl_acc.values(), creak_accuracy]
        ):
            raise ValueError("Sensor accuracies must lie in [0,1]")
        self.creak_acc = creak_accuracy

    def initial_distribution(self):
        return ((TIGER_LEFT, 0.5), (TIGER_RIGHT, 0.5))

    def transition_distribution(self, state, joint_action):
        if any(action != LISTEN for action in joint_action.values()):
            return ((TIGER_LEFT, 0.5), (TIGER_RIGHT, 0.5))
        return ((state, 1.0),)

    def observation_distribution(self, state, joint_action, agent_id):
        other_action = next(a for agent, a in joint_action.items() if agent != agent_id)
        return _observation_law(
            state, joint_action[agent_id], other_action, self.growl_acc[agent_id], self.creak_acc
        )

    def get_reward(self, state, joint_action, next_state, agent_id):
        action = joint_action[agent_id]
        if action == LISTEN:
            return -1.0
        tiger_door = OPEN_LEFT if state == TIGER_LEFT else OPEN_RIGHT
        return -100.0 if action == tiger_door else 10.0

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        return (-100.0, 10.0)

    def is_terminal(self, state) -> bool:
        return False

    def get_all_actions(self, agent_id: AgentID) -> List[str]:
        return list(ACTIONS)

    def get_all_observations(self, agent_id: AgentID) -> List[Tuple[str, str]]:
        return list(OBSERVATIONS)

    def rollout_action(self, agent_id, memory, rng):
        """Listen until one side is ROLLOUT_CONFIDENCE likely, then open the other door.

        Uses only the private physical memory maintained by
        ``update_rollout_memory``; it never inspects the sampled tiger.
        """
        if memory[TIGER_LEFT] >= ROLLOUT_CONFIDENCE:
            return OPEN_RIGHT
        if memory[TIGER_RIGHT] >= ROLLOUT_CONFIDENCE:
            return OPEN_LEFT
        return LISTEN

    def update_rollout_memory(self, memory, action, observation, agent_id):
        """Physical-only Bayes update, marginalising the other agent as uniform L0.

        This cheap memory drives the rollout heuristic only. It is not the
        interactive posterior, which the planner carries separately. Memoized
        (``_rollout_posterior``): it ran 2M times in four L3 episode steps, 14% of
        their CPU, over few distinct arguments.
        """
        return dict(_rollout_posterior(self, tuple(memory.items()), action, observation, agent_id))

    def _rollout_update(self, memory, action, observation, agent_id):
        other = "j" if agent_id == "i" else "i"
        posterior = {TIGER_LEFT: 0.0, TIGER_RIGHT: 0.0}
        for state, mass in memory.items():
            for other_action in ACTIONS:
                joint = {agent_id: action, other: other_action}
                for following, transition in self.transition_distribution(state, joint):
                    likelihood = self.observation_probability(
                        observation, following, joint, agent_id
                    )
                    posterior[following] += mass * transition * likelihood / len(ACTIONS)
        evidence = math.fsum(posterior.values())
        if evidence == 0:
            raise ValueError("Impossible Tiger observation for the rollout memory")
        return {state: mass / evidence for state, mass in posterior.items()}


@lru_cache(maxsize=65536)
def _rollout_posterior(physics, memory, action, observation, agent_id):
    """``physics._rollout_update`` keyed by value; ``memory`` is its items in order.

    Module-level, so physics gains no attribute (its attributes are its
    canonical identity, ``canonical.physics_identity``).
    """
    return tuple(physics._rollout_update(dict(memory), action, observation, agent_id).items())


@lru_cache(maxsize=None)
def _observation_law(state, own_action, other_action, growl_accuracy, creak_accuracy):
    """Memoised: called millions of times by exact filtering; at most 2*3*3 keys per sensor."""
    if own_action != LISTEN:
        return tuple((o, 1 / len(OBSERVATIONS)) for o in OBSERVATIONS)
    true_growl = GROWL_LEFT if state == TIGER_LEFT else GROWL_RIGHT
    true_creak = _TRUE_CREAK[other_action]
    law = []
    for growl, creak in OBSERVATIONS:
        p_growl = growl_accuracy if growl == true_growl else 1 - growl_accuracy
        p_creak = creak_accuracy if creak == true_creak else (1 - creak_accuracy) / 2
        if p_growl * p_creak > 0:
            law.append(((growl, creak), p_growl * p_creak))
    return tuple(law)


class JointTigerModel(TigerModel):
    """Two-agent Tiger whose door payoffs depend on what the other agent does.

    Why
    ---
    In the published game an agent's reward depends only on its own door, so
    the other agent matters only through resets and creaks, and a deeper level
    of reasoning cannot pay off in reward (RESULTS.md 12.18, 12.21). These
    variants keep the published states, transitions and observations and change
    only the payoff of opening a door, so every exact reference, gate and tool
    applies unchanged:

    - ``together_safe``: each agent's reward for opening the safe door when the
      other opens the same door in the same step (published: 10). Above 10 it
      is a coordination bonus; below 10 the agents split a prize (a race).
    - ``solo_penalty``: subtracted from an agent that opens a door while the
      other listens (published: 0); opening is rewarded only in step.

    Opening the tiger's door costs -100 (minus ``solo_penalty`` when alone) and
    listening -1, as published.
    """

    def __init__(self, together_safe=10.0, solo_penalty=0.0, **sensors):
        super().__init__(**sensors)
        if not math.isfinite(together_safe) or not math.isfinite(solo_penalty) or solo_penalty < 0:
            raise ValueError("Joint payoffs must be finite, with a nonnegative solo penalty")
        self.together_safe = float(together_safe)
        self.solo_penalty = float(solo_penalty)

    def get_reward(self, state, joint_action, next_state, agent_id):
        action = joint_action[agent_id]
        if action == LISTEN:
            return -1.0
        other_action = next(a for agent, a in joint_action.items() if agent != agent_id)
        tiger_door = OPEN_LEFT if state == TIGER_LEFT else OPEN_RIGHT
        reward = -100.0 if action == tiger_door else 10.0
        if action != tiger_door and other_action == action:
            reward = self.together_safe
        return reward - self.solo_penalty if other_action == LISTEN else reward

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        # Both are reached: the tiger's door opened alone; the safe door opened
        # beside an opener of the other door (10) or of the same door.
        return (-100.0 - self.solo_penalty, max(10.0, self.together_safe))
