from dataclasses import dataclass
from typing import FrozenSet, Tuple

from examples.wumpus.model.constants import EAST, NORTH, SOUTH, WEST


@dataclass(frozen=True)
class AgentPose:
    x: int
    y: int
    orientation: int  # 0-3 (N, E, S, W)

    def forward(self, width: int, height: int) -> "AgentPose":
        dx, dy = 0, 0
        if self.orientation == NORTH:
            dy = 1
        elif self.orientation == SOUTH:
            dy = -1
        elif self.orientation == EAST:
            dx = 1
        elif self.orientation == WEST:
            dx = -1

        # Clamp to grid boundaries provided by the model
        nx = max(0, min(width - 1, self.x + dx))
        ny = max(0, min(height - 1, self.y + dy))

        return AgentPose(nx, ny, self.orientation)

    def turn_left(self) -> "AgentPose":
        return AgentPose(self.x, self.y, (self.orientation - 1) % 4)

    def turn_right(self) -> "AgentPose":
        return AgentPose(self.x, self.y, (self.orientation + 1) % 4)

    def pos(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class WumpusState:
    # Entities
    human_pose: AgentPose
    wumpus_pose: AgentPose

    # Status
    human_alive: bool
    wumpus_alive: bool

    # Resources
    has_gold: bool
    gold_location: Tuple[int, int]

    has_arrow: bool

    # Map Configuration (Static per episode, but needed for transition logic)
    pit_locations: FrozenSet[Tuple[int, int]]
    grid_size: Tuple[int, int]

    # Transition events make the post-state observation kernel Markov. Merely
    # facing a wall after moving does not imply a bump; a dead wumpus does not
    # imply a new scream on each subsequent attempted shot.
    human_bumped: bool = False
    wumpus_bumped: bool = False
    screamed: bool = False

    def __repr__(self):
        h_s = "Alive" if self.human_alive else "Dead"
        if self.has_gold and self.human_alive:
            h_s = "WON"
        return (
            f"S(H:{self.human_pose.pos()}/{h_s}, "
            f"W:{self.wumpus_pose.pos()}/{self.wumpus_alive}, "
            f"Gold:{self.has_gold})"
        )
