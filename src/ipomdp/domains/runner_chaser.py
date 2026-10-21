"""Runner-Chaser (Schwartz, Zhou & Kurniawati, IROS 2022), as kernels, for the I-NTMCP fidelity check.

This is the authors' own test domain (``intmcp.envs.rc.RCModel`` in
``third_party/i-ntmcp``), restated as finite kernels so that their planner can
run on it through ``baselines.intmcp.POSGAdapter`` and be compared with their
own runner (PLAN.md S2). Every rule below follows ``RCModel``;
``tests/test_runner_chaser.py`` checks next states, observations, rewards and
termination against it for every state and joint action.

Agents: the runner ``r`` and the chaser ``c`` on a grid with blocked cells and
safe cells. Locations are cell indices ``row * width + col``.

State: (runner location, chaser location). Both start cells are fixed by the
layout, so the prior is a point mass.

Actions (each agent): N, S, E, W. A move off the grid or into a blocked cell
leaves the agent in place. The chaser moves first; if it lands on the runner's
current cell, the runner does not move.

Observations (each agent): what lies in the four adjacent cells (N, S, E, W):
the opponent, a wall (off the grid or blocked) or nothing. ``RCModel`` also
reports the agent's start cell in every observation; with fixed start cells
that report is constant, so it is left out.

Rewards, from the state after the joint move: the runner gets +100 when it
stands on a safe cell, -100 when it shares or borders the chaser's cell
(bordering counts through blocked cells, as in ``RCModel``) and -1 otherwise.
The chaser gets the negatives of the first two and -1 otherwise. Either event
ends the episode.

The authors' initial observation (start cell and surroundings, before the first
action) is not represented: the adapter starts from I-NTMCP's null
observation. With fixed start cells that are not adjacent, it carries no
information.
"""

from typing import List, Tuple

from ipomdp.model.pomdp import AgentID, POMDPModel

RUNNER = "r"
CHASER = "c"
NORTH, SOUTH, EAST, WEST = "N", "S", "E", "W"
ACTIONS = (NORTH, SOUTH, EAST, WEST)
OPPONENT, WALL, EMPTY = "X", "#", "0"

R_SAFE, R_CAPTURE, R_STEP = 100.0, -100.0, -1.0

# The authors' "choose a path" layouts (``intmcp.envs.rc.grid``):
# (height, width, blocked cells, safe cells, runner start, chaser start).
LAYOUTS = {
    "cap3": (3, 3, (4,), (0, 5), 7, 1),
    "cap4": (4, 4, (5, 6, 9, 15), (0, 7), 14, 2),
}


class RunnerChaserModel(POMDPModel):
    def __init__(self, layout="cap4"):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(LAYOUTS)}")
        height, width, blocked, safe, runner, chaser = LAYOUTS[layout]
        self.layout, self.height, self.width = layout, height, width
        self.blocked, self.safe = frozenset(blocked), frozenset(safe)
        self.start = (runner, chaser)

    def initial_distribution(self):
        return ((self.start, 1.0),)

    def _move(self, loc, action):
        row, col = divmod(loc, self.width)
        if action == NORTH:
            row = max(0, row - 1)
        elif action == SOUTH:
            row = min(self.height - 1, row + 1)
        elif action == EAST:
            col = min(self.width - 1, col + 1)
        else:
            col = max(0, col - 1)
        following = row * self.width + col
        return loc if following in self.blocked else following

    def _neighbours(self, loc):
        """In-grid cells bordering ``loc``, blocked or not (``get_neighbouring_locs(loc, True)``)."""
        w, h = self.width, self.height
        out = []
        if loc >= w:
            out.append(loc - w)
        if loc < (h - 1) * w:
            out.append(loc + w)
        if loc % w < w - 1:
            out.append(loc + 1)
        if loc % w > 0:
            out.append(loc - 1)
        return out

    def transition_distribution(self, state, joint_action):
        runner, chaser = state
        chaser_next = self._move(chaser, joint_action[CHASER])
        runner_next = runner if chaser_next == runner else self._move(runner, joint_action[RUNNER])
        return (((runner_next, chaser_next), 1.0),)

    def _surroundings(self, loc, opponent):
        w, h = self.width, self.height
        cells = (
            loc - w if loc >= w else -1,
            loc + w if loc < (h - 1) * w else -1,
            loc + 1 if loc % w < w - 1 else -1,
            loc - 1 if loc % w > 0 else -1,
        )
        return tuple(
            OPPONENT if cell == opponent else WALL if cell == -1 or cell in self.blocked else EMPTY
            for cell in cells
        )

    def observation_distribution(self, state, joint_action, agent_id):
        runner, chaser = state
        if agent_id == RUNNER:
            return ((self._surroundings(runner, chaser), 1.0),)
        return ((self._surroundings(chaser, runner), 1.0),)

    def _outcome(self, state):
        """+1 if the runner is safe, -1 if caught, 0 otherwise (RCModel's order of checks)."""
        runner, chaser = state
        if runner in self.safe:
            return 1
        if runner == chaser or runner in self._neighbours(chaser):
            return -1
        return 0

    def get_reward(self, state, joint_action, next_state, agent_id):
        outcome = self._outcome(next_state)
        if outcome == 0:
            return R_STEP
        runner_reward = R_SAFE if outcome == 1 else R_CAPTURE
        return runner_reward if agent_id == RUNNER else -runner_reward

    def reward_bounds(self, agent_id: AgentID) -> Tuple[float, float]:
        return (R_CAPTURE, R_SAFE)

    def is_terminal(self, state) -> bool:
        return self._outcome(state) != 0

    def get_all_actions(self, agent_id: AgentID) -> List[str]:
        return list(ACTIONS)

    def get_all_observations(self, agent_id: AgentID):
        tokens = (OPPONENT, WALL, EMPTY)
        return [(n, s, e, w) for n in tokens for s in tokens for e in tokens for w in tokens]
