"""UAV reconnaissance (Doshi & Gmytrasiewicz 2009, Section 8.3.2), with declared values.

The 2009 paper gives the structure; it leaves the sensor accuracy, rewards,
the target's observations and the priors unstated. Values below come from the
same group's UAV reconnaissance domain (Zeng & Doshi 2012, JAIR 43, Section 7
and Appendix B.3) where it states them, and are declared here otherwise; see
docs/DOMAINS.md, "UAV reconnaissance", for the table of sources.

Agents: the UAV ``i`` and the ground target ``j`` (T) on a ``rows x cols`` grid
of sectors (3 x 3 in the paper).

State: (UAV sector, target sector), each a (row, column) pair. The paper merges
the left and right columns by symmetry (36 states), but with two agents that
collapse cannot tell whether both side-column agents share a sector, so it
cannot express spotting; the full grid is used (the mirror symmetry is tested).

Prior: the UAV starts in ``uav_start`` (default bottom-left), known to both
agents, so with deterministic moves it always knows its own sector (the paper:
"aware of its own location"); the target is uniform over the other sectors.

Actions (each agent): move N, S, E, W to the adjacent sector (deterministic,
clamped at the grid's edge) or listen (stay). The paper's text also allows the
target diagonal moves, but it gives each agent 5 actions twice; 5 are kept.

Spotting: when both are in the same sector after the joint move; the game ends.

Observations:
- UAV: the row of the target (0 = top row, the paper's TR; 1 = CR; 2 = BR).
  Listening reports the correct row with probability ``uav_accuracy`` (0.8)
  and each other row equally; moving gives a uniformly random row.
- Target: (its own sector, the row of the UAV). Its own sector is exact (the
  package's prior is common and never conditions on the true state, so the
  target learns its sector here, from its first observation on). The UAV's row
  is reported as for the UAV, with ``target_accuracy`` (0.6: "high
  uncertainty").

Rewards: the UAV gets ``spot_reward`` (50) on spotting and pays ``step_cost``
(5) for every other step; the target gets -``spot_reward`` when spotted and 0
otherwise.
"""

import math
from functools import lru_cache
from typing import List, Tuple

from ipomdp.model.pomdp import AgentID, POMDPModel

UAV = "i"
TARGET = "j"
MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN = "N", "S", "E", "W", "listen"
ACTIONS = (MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN)
_DELTAS = {MOVE_N: (-1, 0), MOVE_S: (1, 0), MOVE_E: (0, 1), MOVE_W: (0, -1), LISTEN: (0, 0)}


class UAVModel(POMDPModel):
    def __init__(
        self,
        rows=3,
        cols=3,
        uav_start=None,
        uav_accuracy=0.8,
        target_accuracy=0.6,
        spot_reward=50.0,
        step_cost=5.0,
    ):
        if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1:
            raise ValueError("The grid needs positive integer dimensions")
        if rows < 2 or rows * cols < 2:
            raise ValueError("The grid needs at least two rows")
        start = (rows - 1, 0) if uav_start is None else tuple(uav_start)
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            raise ValueError("The UAV must start inside the grid")
        for accuracy in (uav_accuracy, target_accuracy):
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("Sensor accuracies must lie in [0, 1]")
        if not math.isfinite(spot_reward) or not math.isfinite(step_cost) or spot_reward <= 0:
            raise ValueError("Rewards must be finite, with a positive spotting reward")
        self.rows, self.cols, self.uav_start = rows, cols, start
        self.uav_accuracy, self.target_accuracy = uav_accuracy, target_accuracy
        self.spot_reward, self.step_cost = float(spot_reward), float(step_cost)

    def cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def initial_distribution(self):
        targets = [cell for cell in self.cells() if cell != self.uav_start]
        return tuple(((self.uav_start, cell), 1 / len(targets)) for cell in targets)

    def _move(self, cell, action):
        dr, dc = _DELTAS[action]
        return (min(max(cell[0] + dr, 0), self.rows - 1), min(max(cell[1] + dc, 0), self.cols - 1))

    def transition_distribution(self, state, joint_action):
        uav, target = state
        following = (self._move(uav, joint_action[UAV]), self._move(target, joint_action[TARGET]))
        return ((following, 1.0),)

    def observation_distribution(self, state, joint_action, agent_id):
        uav, target = state
        if agent_id == UAV:
            accuracy = self.uav_accuracy if joint_action[UAV] == LISTEN else None
            return _row_report(target[0], self.rows, accuracy)
        accuracy = self.target_accuracy if joint_action[TARGET] == LISTEN else None
        return tuple(((target, row), p) for row, p in _row_report(uav[0], self.rows, accuracy))

    def get_reward(self, state, joint_action, next_state, agent_id):
        spotted = self.is_terminal(next_state)
        if agent_id == UAV:
            return self.spot_reward if spotted else -self.step_cost
        return -self.spot_reward if spotted else 0.0

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        if agent_id == UAV:
            return (min(-self.step_cost, self.spot_reward), max(-self.step_cost, self.spot_reward))
        return (-self.spot_reward, 0.0)

    def is_terminal(self, state) -> bool:
        return state[0] == state[1]

    def get_all_actions(self, agent_id: AgentID) -> List[str]:
        return list(ACTIONS)

    def get_all_observations(self, agent_id: AgentID):
        rows = list(range(self.rows))
        if agent_id == UAV:
            return rows
        return [(cell, row) for cell in self.cells() for row in rows]


@lru_cache(maxsize=None)
def _row_report(true_row, rows, accuracy):
    """((row, probability), ...): correct with ``accuracy``, else uniform; None = uniform."""
    if accuracy is None:
        return tuple((row, 1 / rows) for row in range(rows))
    wrong = (1 - accuracy) / (rows - 1)
    law = ((row, accuracy if row == true_row else wrong) for row in range(rows))
    return tuple((row, p) for row, p in law if p > 0)
