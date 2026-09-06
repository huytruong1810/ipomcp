# Absolute Path: <project_root>/examples/wumpus/model/constants.py

"""
constants — Definitions for the Wumpus World Domain.

DESIGN DECISION RECORD (Phase 3 Overhaul):
------------------------------------------
1. "Blind Hunter" Fix: Added `OBS_FOOTSTEP`. Previously, the Wumpus received
   zero observations about the Human. Without any sensory link to the opponent,
   a Level-2 Wumpus behaves exactly like a Level-0 Wumpus (random walk).
   This observation constant enables the Wumpus to "hear" the Human, activating
   the I-POMCP mental tracking for pursuit-evasion dynamics.
"""

# Orientations
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Actions
ACTION_FORWARD = "Forward"
ACTION_TURN_LEFT = "TurnLeft"
ACTION_TURN_RIGHT = "TurnRight"
ACTION_GRAB = "Grab"
ACTION_SHOOT = "Shoot"

# Agent IDs
AGENT_HUMAN = 'i'
AGENT_WUMPUS = 'j'

# Action Sets
HUMAN_ACTIONS = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_GRAB, ACTION_SHOOT]
WUMPUS_ACTIONS = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]

# Observations
OBS_STENCH = "Stench"
OBS_BREEZE = "Breeze"
OBS_GLITTER = "Glitter"
OBS_BUMP = "Bump"
OBS_SCREAM = "Scream"
OBS_NONE = "None"
OBS_FOOTSTEP = "Footstep"  # [PHASE 3 FIX]: Auditory cue for the Wumpus