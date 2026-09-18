"""Action and sensor symbols for the two-agent Wumpus domain.

Footsteps indicate a living human within Manhattan distance two. This supplies
information to the wumpus but does not guarantee any particular pursuit policy.
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
AGENT_HUMAN = "i"
AGENT_WUMPUS = "j"

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
OBS_FOOTSTEP = "Footstep"  # Auditory cue for the Wumpus
