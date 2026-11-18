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

### Joint-payoff Tiger (`JointTigerModel`, study domain `tiger-joint`)

A declared variant for this project. In the published game an agent's reward depends only on its own door, so reasoning one level deeper about the other agent is worth almost nothing in reward (by the exact solver: no decision changes at horizon 2, and about 0.01 on average at horizon 3). The variant keeps the states, transitions and observations above and changes only the payoff of opening a door:

| Parameter | Meaning | Published value |
|---|---|---|
| `together_safe` | each agent's reward for opening the safe door when the other opens the same door in the same step | 10 |
| `solo_penalty` | subtracted from an agent that opens a door while the other listens | 0 |

The coordination game is `together_safe = 40`, chosen by an exact screen of the value of one more level of reasoning over a grid of payoffs. Every payoff keeps the declared reward bounds attained (`JointTigerModel.reward_bounds`, checked exhaustively in `tests/test_domain_exhaustive.py`).

## UAV reconnaissance (`domains/uav.py`)

The domain of Doshi & Gmytrasiewicz (2009), "Monte Carlo Sampling Methods for Approximating Interactive POMDPs", JAIR 34, Section 8.3.2 (arXiv:1401.3455): **the paper's structure, with declared values where it is silent.** The 2009 paper states no sensor accuracy, rewards, target observations or priors. Values come from the same group's UAV reconnaissance domain in Zeng & Doshi (2012), "Exploiting Model Equivalences for Solving Interactive Dynamic Influence Diagrams", JAIR 43, Section 7 and Appendix B.3 (arXiv:1401.4600), where it states them (its CPTs are "available upon request"); the rest are declared here. The original values were not available.

| Element | 2009 paper | Here | Source of the value |
|---|---|---|---|
| Grid | 3 × 3 sectors; UAV i, ground target T (agent j) | same (`rows`, `cols`) | 2009 |
| State | 36: rows and side-or-center columns of both, sides merged by symmetry | 81: (i's sector, T's sector); same sector = spotted (terminal) | the merged state cannot tell whether two side-column agents share a sector, so it cannot express spotting. The mirror symmetry holds for every kernel and is tested |
| Actions | N, S, E, W, listen for each ("5 actions", stated twice) | same; moves deterministic, clamped at the edge | 2009 (its remark that T may move diagonally contradicts the 5 actions; 5 kept) |
| i's observations | TR, CR, BR: the row of T | rows 0 (TR), 1 (CR), 2 (BR). Listen: correct with 0.8, each other row 0.1. Moving: uniform | Zeng & Doshi 2012 (0.8 on listening; "random observations during other actions") |
| i's position | known to i | i starts in the bottom-left sector, known to both; moves are deterministic, so i always knows its sector | 2009 ("aware of its own location"). Zeng & Doshi (2012, Appendix B.3) move the UAV in the intended direction with probability 0.67; that noise is not adopted, because with it the UAV would not know its own sector, which the 2009 paper states it does |
| T's observations | "informed … that it could be in danger of being spotted … with high uncertainty" | (T's own sector, i's row). Listen: i's row correct with 0.6, each other 0.2. Moving: uniform | declared: "high uncertainty" is below i's 0.8 |
| Prior | i uncertain about T's sector | T uniform over the 8 sectors other than i's start | 2009 |
| Spotting | "If the UAV moves to a sector containing a target … the game ends" | both in one sector after the joint move; terminal. Swapping sectors is not spotting | 2009 |
| i's reward | in [0, 1] (though Fig. 16 plots rewards up to about 12) | +50 on spotting, −5 for every other step | Zeng & Doshi 2012 |
| T's reward | objectives that conflict with i's | −50 when spotted, 0 otherwise | declared |
| Discount | — | 0.95 (package default) | declared |

**Deviation.** The package's prior is common to every agent and nesting level and never conditions on the true state (`experiments/bootstrapper.py`). T's own sector is therefore part of T's observation, so T learns its sector from its first observation; its first decision is made without it.


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

## Runner-Chaser (`domains/runner_chaser.py`)

The test domain of Schwartz, Zhou & Kurniawati (2022), restated as finite kernels so that their planner can be run on it through this package and compared with their own runner. Every rule follows their `RCModel` (`third_party/i-ntmcp`), and `tests/test_runner_chaser.py` checks next states, observations, rewards and termination against it for every state and joint action of each layout.

| Element | Specification |
|---|---|
| Agents | the runner `r` and the chaser `c`, on a grid with blocked and safe cells; layouts `cap3`, `cap4`, `cap7` (the authors' "choose a path" grids) |
| State | (runner cell, chaser cell); both start cells are fixed by the layout, so the prior is a point mass |
| Actions | N, S, E, W for each; a move off the grid or into a blocked cell leaves the agent in place; the chaser moves first, and the runner does not move if the chaser lands on its cell |
| Observations | for each agent, what lies in the four adjacent cells: the other agent, a wall, or nothing |
| Rewards | runner +100 on a safe cell, -100 when it shares or borders the chaser's cell, -1 otherwise; the chaser the negatives of the first two and -1 otherwise; either event ends the episode |

