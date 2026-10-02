"""Two-agent UAV pursuit: a pursuer (agent i) and an evader (agent j) on a grid.

Specification (declared here; this project's version of the UAV benchmark)
---------------------------------------------------------------------------
State: (pursuer cell, evader cell) on a ``rows x cols`` grid.
Prior: uniform over all pairs of distinct cells.

Actions (each agent): move N, S, E, W (clamped at the boundary) or SENSE (stay).
Moves are simultaneous and deterministic.

Capture: the episode ends when both agents occupy the same cell after moving.
Agents that swap cells pass each other without capture.

Rewards: pursuer -0.1 per step and +1 on capture; evader +0.1 per step and -1
on capture.

Observation (agent k): (own cell, direction token).
- Each agent always knows its own cell. An earlier version hid the pursuer's
  own position, which made capture mostly luck and said little about planning.
- The direction token reports where the other agent is relative to k after the
  move: the sign of the row offset and of the column offset, giving one of
  eight compass tokens (N, NE, E, SE, S, SW, W, NW).
- When k SENSEs, the token is correct with probability ``sensor_accuracy``,
  and otherwise uniform over the seven wrong tokens. When k moves, it receives
  the token NONE: moving and sensing are exclusive.

Capture states are terminal, so no observation is emitted from them in planning.
"""

import math
from functools import lru_cache
from typing import List, Tuple

from ipomdp.model.pomdp import AgentID, POMDPModel

PURSUER = "i"
EVADER = "j"
MOVE_N, MOVE_S, MOVE_E, MOVE_W, SENSE = "N", "S", "E", "W", "sense"
ACTIONS = (MOVE_N, MOVE_S, MOVE_E, MOVE_W, SENSE)
_DELTAS = {MOVE_N: (-1, 0), MOVE_S: (1, 0), MOVE_E: (0, 1), MOVE_W: (0, -1), SENSE: (0, 0)}
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
NONE = "none"
# Sign of (row offset, column offset) of the other agent -> compass token.
_DIRECTION_OF = {
    (-1, 0): "N",
    (-1, 1): "NE",
    (0, 1): "E",
    (1, 1): "SE",
    (1, 0): "S",
    (1, -1): "SW",
    (0, -1): "W",
    (-1, -1): "NW",
}


class UAVModel(POMDPModel):
    def __init__(self, rows=3, cols=3, sensor_accuracy=0.85):
        if type(rows) is not int or type(cols) is not int or rows * cols < 2 or min(rows, cols) < 1:
            raise ValueError("The grid needs at least two cells")
        if not math.isfinite(sensor_accuracy) or not 0 <= sensor_accuracy <= 1:
            raise ValueError("Sensor accuracy must lie in [0,1]")
        self.rows, self.cols, self.acc = rows, cols, sensor_accuracy

    def cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def initial_distribution(self):
        pairs = [(u, t) for u in self.cells() for t in self.cells() if u != t]
        return tuple((pair, 1 / len(pairs)) for pair in pairs)

    def _move(self, cell, action):
        dr, dc = _DELTAS[action]
        return (min(max(cell[0] + dr, 0), self.rows - 1), min(max(cell[1] + dc, 0), self.cols - 1))

    def transition_distribution(self, state, joint_action):
        pursuer, evader = state
        return (
            (
                (
                    self._move(pursuer, joint_action[PURSUER]),
                    self._move(evader, joint_action[EVADER]),
                ),
                1.0,
            ),
        )

    def observation_distribution(self, state, joint_action, agent_id):
        own, other = state if agent_id == PURSUER else (state[1], state[0])
        if joint_action[agent_id] != SENSE:
            return (((own, NONE), 1.0),)
        return tuple(
            ((own, token), p) for token, p in _sensor_law(_direction(own, other), self.acc)
        )

    def get_reward(self, state, joint_action, next_state, agent_id):
        caught = next_state[0] == next_state[1]
        if agent_id == PURSUER:
            return 1.0 if caught else -0.1
        return -1.0 if caught else 0.1

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        return (-0.1, 1.0) if agent_id == PURSUER else (-1.0, 0.1)

    def is_terminal(self, state) -> bool:
        return state[0] == state[1]

    def get_all_actions(self, agent_id: AgentID) -> List[str]:
        return list(ACTIONS)

    def get_all_observations(self, agent_id: AgentID):
        return [(cell, token) for cell in self.cells() for token in (*DIRECTIONS, NONE)]


def _direction(own, other):
    offset = (_sign(other[0] - own[0]), _sign(other[1] - own[1]))
    if offset == (0, 0):
        raise ValueError("Direction is undefined when both agents share a cell")
    return _DIRECTION_OF[offset]


def _sign(x):
    return (x > 0) - (x < 0)


@lru_cache(maxsize=None)
def _sensor_law(true_token, accuracy):
    wrong = (1 - accuracy) / (len(DIRECTIONS) - 1)
    law = ((t, accuracy if t == true_token else wrong) for t in DIRECTIONS)
    return tuple((t, p) for t, p in law if p > 0)
