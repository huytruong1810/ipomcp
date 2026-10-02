"""Two-agent Wumpus world: a human (agent i) seeking gold and a wumpus (agent j) hunting.

Specification
-------------
Grid ``width x height`` (default 4 x 4). The human starts at (0, 0) facing east,
holding one arrow.

Prior (exact, uniform and factored):
- wumpus cell uniform over the cells other than (0, 0), orientation uniform;
- gold cell uniform over the cells other than (0, 0), independent of the rest
  (it may coincide with the wumpus or a pit);
- ``n_pits`` pits on a uniformly random subset of the cells other than (0, 0).
For 4 x 4 with 2 pits that is 15 * 4 * 15 * C(15, 2) = 94,500 equally likely
states. The prior is enumerated exactly: a sampled prior missed the true world
in every probe and made deterministic sensors contradict the belief.

Actions: human Forward, TurnLeft, TurnRight, Grab, Shoot; wumpus Forward,
TurnLeft, TurnRight. The action sets are fixed: shooting after the arrow is
spent is allowed and has no effect beyond its cost, so the action set never
reveals hidden state.

Dynamics (simultaneous, deterministic):
- Shooting uses the pre-move poses and kills a wumpus in the line of fire.
- After moving, the human dies in a pit, on the wumpus's cell, or by swapping
  cells with a living wumpus. Death takes priority over grabbing gold.
- Grab succeeds on the gold cell. Gold or death ends the episode.
- Bumps (forward into a wall) and screams (a new kill) are recorded in the
  successor state, so observations can report events rather than positions.

Observations (deterministic, sets of tokens):
- human: Bump, Stench (living wumpus adjacent), Breeze (pit adjacent), Glitter
  (on uncollected gold), Scream (wumpus killed this step);
- wumpus: Bump, Breeze, Glitter, Footstep (living human within Manhattan
  distance 2).

Rewards: human +1000 gold, -1000 death, -10 per shot, -1 otherwise; wumpus
+1000 for killing the human by collision or swap, -1000 when shot, -1 otherwise.
Terminal states give zero reward.
"""

import itertools
from functools import lru_cache
from typing import Dict, List, Tuple

from ipomdp.domains.wumpus.constants import (
    ACTION_FORWARD,
    ACTION_GRAB,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    AGENT_HUMAN,
    AGENT_WUMPUS,
    EAST,
    HUMAN_ACTIONS,
    NORTH,
    OBS_BREEZE,
    OBS_BUMP,
    OBS_FOOTSTEP,
    OBS_GLITTER,
    OBS_SCREAM,
    OBS_STENCH,
    SOUTH,
    WEST,
    WUMPUS_ACTIONS,
)
from ipomdp.domains.wumpus.state import AgentPose, WumpusState
from ipomdp.model.pomdp import Action, AgentID, POMDPModel

START = (0, 0)


class WumpusModel(POMDPModel):
    def __init__(self, width=4, height=4, n_pits=2):
        if min(width, height) < 2 or n_pits < 0 or n_pits > width * height - 1:
            raise ValueError("Need at least a 2 x 2 grid and room for the pits")
        self.width, self.height, self.n_pits = width, height, n_pits

    # ------------------------------------------------------------------ #
    # Kernels
    # ------------------------------------------------------------------ #

    def initial_distribution(self):
        # Cached outside the instance: physics identity is derived from the
        # instance's attributes, so derived data must not be stored on it.
        return _uniform_prior(self.width, self.height, self.n_pits)

    def transition_distribution(self, state, joint_action):
        return ((self._next_state(state, joint_action), 1.0),)

    def observation_distribution(self, state, joint_action, agent_id):
        return ((self._percepts(state, agent_id), 1.0),)

    def get_reward(self, state, joint_action, next_state, agent_id):
        if self.is_terminal(state):
            return 0.0
        if agent_id == AGENT_HUMAN:
            if not next_state.human_alive:
                return -1000.0
            if next_state.has_gold:
                return 1000.0
            return -10.0 if joint_action[AGENT_HUMAN] == ACTION_SHOOT else -1.0
        if state.human_alive and not next_state.human_alive and next_state.wumpus_alive:
            human, wumpus = next_state.human_pose.pos(), next_state.wumpus_pose.pos()
            swapped = human == state.wumpus_pose.pos() and wumpus == state.human_pose.pos()
            if human == wumpus or swapped:
                return 1000.0
        if state.wumpus_alive and not next_state.wumpus_alive:
            return -1000.0
        return -1.0

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        return (-1000.0, 1000.0)

    def is_terminal(self, state) -> bool:
        return (not state.human_alive) or state.has_gold

    def get_all_actions(self, agent_id: AgentID) -> List[Action]:
        return list(WUMPUS_ACTIONS if agent_id == AGENT_WUMPUS else HUMAN_ACTIONS)

    def get_all_observations(self, agent_id: AgentID):
        if agent_id == AGENT_WUMPUS:
            tokens = [OBS_BUMP, OBS_BREEZE, OBS_GLITTER, OBS_FOOTSTEP]
        else:
            tokens = [OBS_BUMP, OBS_STENCH, OBS_BREEZE, OBS_GLITTER, OBS_SCREAM]
        return [
            tuple(sorted(combo))
            for size in range(len(tokens) + 1)
            for combo in itertools.combinations(tokens, size)
        ]

    # ------------------------------------------------------------------ #
    # Dynamics and sensors
    # ------------------------------------------------------------------ #

    def _next_state(self, state: WumpusState, joint_action: Dict[AgentID, Action]) -> WumpusState:
        if self.is_terminal(state):
            return state
        human_action, wumpus_action = joint_action[AGENT_HUMAN], joint_action[AGENT_WUMPUS]
        human = self._move(state.human_pose, human_action)
        wumpus = (
            self._move(state.wumpus_pose, wumpus_action)
            if state.wumpus_alive
            else state.wumpus_pose
        )
        has_gold = human_action == ACTION_GRAB and human.pos() == state.gold_location
        has_arrow, wumpus_alive = state.has_arrow, state.wumpus_alive
        if human_action == ACTION_SHOOT and has_arrow:
            has_arrow = False
            if wumpus_alive and _in_line_of_fire(state.human_pose, state.wumpus_pose):
                wumpus_alive = False
        human_alive = human.pos() not in state.pit_locations
        if wumpus_alive:
            swapped = (
                human.pos() == state.wumpus_pose.pos() and wumpus.pos() == state.human_pose.pos()
            )
            if human.pos() == wumpus.pos() or swapped:
                human_alive = False
        return WumpusState(
            human_pose=human,
            wumpus_pose=wumpus,
            human_alive=human_alive,
            wumpus_alive=wumpus_alive,
            has_gold=has_gold,
            gold_location=state.gold_location,
            has_arrow=has_arrow,
            pit_locations=state.pit_locations,
            grid_size=state.grid_size,
            human_bumped=human_action == ACTION_FORWARD and human.pos() == state.human_pose.pos(),
            wumpus_bumped=(
                state.wumpus_alive
                and wumpus_action == ACTION_FORWARD
                and wumpus.pos() == state.wumpus_pose.pos()
            ),
            screamed=state.wumpus_alive and not wumpus_alive,
        )

    def _move(self, pose: AgentPose, action: Action) -> AgentPose:
        if action == ACTION_FORWARD:
            return pose.forward(self.width, self.height)
        if action == ACTION_TURN_LEFT:
            return pose.turn_left()
        if action == ACTION_TURN_RIGHT:
            return pose.turn_right()
        return pose

    def _percepts(self, state: WumpusState, agent_id: AgentID):
        percepts = set()
        if agent_id == AGENT_HUMAN:
            here = state.human_pose.pos()
            if state.human_bumped:
                percepts.add(OBS_BUMP)
            if state.wumpus_alive and _adjacent(here, state.wumpus_pose.pos()):
                percepts.add(OBS_STENCH)
            if any(_adjacent(here, pit) for pit in state.pit_locations):
                percepts.add(OBS_BREEZE)
            if here == state.gold_location and not state.has_gold:
                percepts.add(OBS_GLITTER)
            if state.screamed:
                percepts.add(OBS_SCREAM)
        else:
            here = state.wumpus_pose.pos()
            if state.wumpus_bumped:
                percepts.add(OBS_BUMP)
            if any(_adjacent(here, pit) for pit in state.pit_locations):
                percepts.add(OBS_BREEZE)
            if here == state.gold_location:
                percepts.add(OBS_GLITTER)
            human = state.human_pose.pos()
            if state.human_alive and abs(human[0] - here[0]) + abs(human[1] - here[1]) <= 2:
                percepts.add(OBS_FOOTSTEP)
        return tuple(sorted(percepts))


def _adjacent(a, b) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _in_line_of_fire(shooter: AgentPose, target: AgentPose) -> bool:
    (sx, sy), (tx, ty) = shooter.pos(), target.pos()
    if shooter.orientation == NORTH:
        return tx == sx and ty > sy
    if shooter.orientation == SOUTH:
        return tx == sx and ty < sy
    if shooter.orientation == EAST:
        return ty == sy and tx > sx
    if shooter.orientation == WEST:
        return ty == sy and tx < sx
    return False


@lru_cache(maxsize=4)
def _uniform_prior(width, height, n_pits):
    hazards = [(x, y) for x in range(width) for y in range(height) if (x, y) != START]
    human = AgentPose(*START, EAST)
    states = [
        WumpusState(
            human_pose=human,
            wumpus_pose=AgentPose(*wumpus_cell, orientation),
            human_alive=True,
            wumpus_alive=True,
            has_gold=False,
            gold_location=gold,
            has_arrow=True,
            pit_locations=frozenset(pits),
            grid_size=(width, height),
        )
        for wumpus_cell in hazards
        for orientation in (NORTH, EAST, SOUTH, WEST)
        for gold in hazards
        for pits in itertools.combinations(hazards, n_pits)
    ]
    return tuple((state, 1 / len(states)) for state in states)
