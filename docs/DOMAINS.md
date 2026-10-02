# Domain specifications

Each domain implements `model.POMDPModel` with exact finite kernels. The
docstring of each module is the authoritative specification; this page
summarises them and records their sources.

## Two-agent Tiger (`domains/tiger.py`)

**Source:** Gmytrasiewicz & Doshi (2005), *A Framework for Sequential Planning in Multi-Agent Settings*, JAIR 24, Section 7.1 and Table 1 ([arXiv 1109.2135](https://arxiv.org/abs/1109.2135)). It is implemented exactly as published.

| Element | Specification |
|---|---|
| States, prior | tiger left (TL) or right (TR), 50/50 |
| Actions | listen (L), open left (OL), open right (OR) |
| Transition | if either agent opens a door, the tiger resets uniformly; otherwise it stays |
| Reward (own action only) | listen −1; opening the tiger's door −100; opening the other door +10 |
| Observation of a listener | (growl, creak). The growl is correct with probability 0.85. The creak reports the other agent's action (S if it listened, CL/CR if it opened left/right): correct 0.9, each other token 0.05 |
| Observation of an opener | uniform over the six (growl, creak) pairs |

A listening opponent therefore causes a false creak with probability 0.1.

## UAV pursuit (`domains/uav.py`)

This is a declared specification for this project.

| Element | Specification |
|---|---|
| State, prior | (pursuer cell, evader cell) on a rows × cols grid (default 3 × 3); uniform over distinct cells |
| Actions | N, S, E, W (clamped at the boundary), or SENSE (stay) |
| Capture | both agents on the same cell after moving; the episode ends. Swapping cells is not a capture |
| Rewards | pursuer −0.1 per step and +1 on capture; evader +0.1 per step and −1 when caught |
| Observation | (own cell, token). Each agent knows its own cell. SENSE reports the other agent's compass direction (N, NE, …, NW): correct with probability 0.85, otherwise uniform over the other seven. Moving yields NONE |

An earlier version hid the pursuer's own position, which made capture largely luck.

## Wumpus world (`domains/wumpus/`)

This is a declared specification for this project.

| Element | Specification |
|---|---|
| Grid | default 4 × 4; the human starts at (0, 0) facing east, with one arrow |
| Prior | exact and uniform: wumpus cell and orientation, gold cell and a set of pit cells, all over cells other than (0, 0). That is 94,500 states for 4 × 4 with 2 pits |
| Actions | human: Forward, TurnLeft, TurnRight, Grab, Shoot; wumpus: Forward, TurnLeft, TurnRight. The sets are fixed: shooting without an arrow is allowed, has no effect and costs 10 |
| Dynamics | simultaneous and deterministic. A shot kills a wumpus in the line of fire. The human dies in a pit, on the wumpus's cell, or by swapping cells with a living wumpus. Gold or death ends the episode |
| Observations | deterministic. Human: Bump, Stench, Breeze, Glitter, Scream. Wumpus: Bump, Breeze, Glitter, Footstep (human within Manhattan distance 2) |
| Rewards | human +1000 gold, −1000 death, −10 per shot, −1 otherwise; wumpus +1000 for killing the human, −1000 when shot, −1 otherwise |

**Known limit.** Deterministic sensors make particle beliefs fragile. If no particle matches the true world, an observation has zero evidence and the update raises. Exact beliefs at level 1 are tractable. The IPF needs reinvigoration before Wumpus studies at higher levels.
