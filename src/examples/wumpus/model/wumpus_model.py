# Absolute Path: <project_root>/examples/wumpus/model/wumpus_model.py

"""
wumpus_model — Multi-agent Wumpus World dynamics.

DESIGN DECISION RECORD (Phase 3 Overhaul):
------------------------------------------
1. ERADICATING GENERATIVE-EVALUATIVE ASYMMETRY:
   Previously, the model injected a fake `epsilon_noise` (0.02) into `get_observation_prob`
   to keep particles alive when an impossible observation was received. However, the
   generative `sample_observation` NEVER produced this noise! This completely broke
   the mathematical symmetry of the Particle Filter, inflating "dead" particles.
   FIX: We removed the epsilon hack. `get_observation_prob` now strictly returns 1.0 or 0.0
   because Wumpus sensors are deterministic given the physical state. (Particle deprivation
   is now correctly handled by the planner's reinvigoration logic from Phase 2).

2. ACTIVATING LEVEL-K PURSUIT DYNAMICS:
   Added `OBS_FOOTSTEP` to the Wumpus's observation generation. The Wumpus can now
   hear the Human if they are within a Manhattan distance of 2. This forces the
   Wumpus to update its mental model of the Human, enabling Level-2 ambush tactics.

3. DYNAMIC ACTION PRUNING:
   Implemented `get_legal_actions`. If the Human has already fired their arrow,
   `ACTION_SHOOT` is removed from the legal action list. This massively reduces
   the MCTS branching factor, speeding up simulations.
"""

import random
from typing import Dict, List
from core.pomdp_model import POMDPModel, State, Action, Observation, AgentID
from examples.wumpus.model.constants import *
from examples.wumpus.model.wumpus_state import WumpusState, AgentPose


class WumpusModel(POMDPModel):
    def __init__(self, width=4, height=4, n_pits=2):
        self.width = width
        self.height = height
        self.n_pits = n_pits

    def get_initial_state(self) -> State:
        # 1. Human (0, 0) East
        h_pose = AgentPose(0, 0, EAST)

        # 2. Exclude Safe Zone (Start) from hazards
        safe_cells = {(0, 0)}
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        available_cells = [c for c in all_cells if c not in safe_cells]

        # 3. Wumpus Random
        w_pos = random.choice(available_cells)
        w_pose = AgentPose(w_pos[0], w_pos[1], random.choice([NORTH, SOUTH, EAST, WEST]))

        # 4. Gold Random
        g_pos = random.choice(available_cells)

        # 5. Pits Random
        pits = set()
        curr_available = list(available_cells)
        for _ in range(min(self.n_pits, len(curr_available))):
            p = random.choice(curr_available)
            pits.add(p)
            curr_available.remove(p)

        return WumpusState(
            human_pose=h_pose, wumpus_pose=w_pose,
            human_alive=True, wumpus_alive=True,
            has_gold=False, gold_location=g_pos, has_arrow=True,
            pit_locations=frozenset(pits), grid_size=(self.width, self.height)
        )

    def get_all_actions(self, agent_id: AgentID) -> List[Action]:
        return WUMPUS_ACTIONS if str(agent_id) == AGENT_WUMPUS else HUMAN_ACTIONS

    def get_all_observations(self, agent_id: AgentID) -> List[Observation]:
        """Returns all possible discrete observation tuples for the specified agent in Wumpus World."""
        import itertools
        if str(agent_id) == AGENT_WUMPUS:
            tokens = [OBS_BUMP, OBS_BREEZE, OBS_GLITTER, OBS_FOOTSTEP]
        else:
            tokens = [OBS_BUMP, OBS_STENCH, OBS_BREEZE, OBS_GLITTER, OBS_SCREAM]

        all_obs = []
        for r in range(len(tokens) + 1):
            for combo in itertools.combinations(tokens, r):
                all_obs.append(tuple(sorted(list(combo))))
        return all_obs

    # [PHASE 3 FIX]: Prune MCTS branches mathematically impossible for the current state.
    def get_legal_actions(self, state: WumpusState, agent_id: AgentID) -> List[Action]:
        if str(agent_id) == AGENT_WUMPUS:
            return WUMPUS_ACTIONS

        # Human agent
        legal = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_GRAB]
        if state.has_arrow:
            legal.append(ACTION_SHOOT)
        return legal

    def sample_transition(self, state: WumpusState, joint_action: Dict[AgentID, Action]) -> State:
        if not state.human_alive or state.has_gold:
            return state

        h_act = joint_action.get(AGENT_HUMAN)
        w_act = joint_action.get(AGENT_WUMPUS)
        w, h = state.grid_size

        h_next_pose = self._process_move(state.human_pose, h_act, w, h)
        w_next_pose = self._process_move(state.wumpus_pose, w_act, w, h) if state.wumpus_alive else state.wumpus_pose

        has_gold = state.has_gold
        has_arrow = state.has_arrow
        wumpus_alive = state.wumpus_alive

        if h_act == ACTION_GRAB and h_next_pose.pos() == state.gold_location:
            has_gold = True

        if h_act == ACTION_SHOOT and has_arrow:
            has_arrow = False
            if wumpus_alive and self._check_hit(state.human_pose, state.wumpus_pose):
                wumpus_alive = False

        human_alive = True
        if h_next_pose.pos() in state.pit_locations:
            human_alive = False

        if wumpus_alive:
            if h_next_pose.pos() == w_next_pose.pos(): human_alive = False
            if (h_next_pose.pos() == state.wumpus_pose.pos()) and (w_next_pose.pos() == state.human_pose.pos()):
                human_alive = False

        return WumpusState(
            human_pose=h_next_pose, wumpus_pose=w_next_pose,
            human_alive=human_alive, wumpus_alive=wumpus_alive,
            has_gold=has_gold, gold_location=state.gold_location,
            has_arrow=has_arrow,
            pit_locations=state.pit_locations, grid_size=state.grid_size
        )

    def _process_move(self, pose: AgentPose, action: Action, w: int, h: int) -> AgentPose:
        if action == ACTION_FORWARD: return pose.forward(w, h)
        if action == ACTION_TURN_LEFT: return pose.turn_left()
        if action == ACTION_TURN_RIGHT: return pose.turn_right()
        return pose

    def _check_hit(self, shooter: AgentPose, target: AgentPose) -> bool:
        sx, sy = shooter.pos()
        tx, ty = target.pos()
        if shooter.orientation == NORTH: return (tx == sx) and (ty > sy)
        if shooter.orientation == SOUTH: return (tx == sx) and (ty < sy)
        if shooter.orientation == EAST: return (ty == sy) and (tx > sx)
        if shooter.orientation == WEST: return (ty == sy) and (tx < sx)
        return False

    def sample_observation(self, state: WumpusState, joint_action: Dict[AgentID, Action],
                           agent_id: AgentID) -> Observation:
        obs = set()
        my_act = joint_action.get(agent_id)
        w, h = state.grid_size

        if agent_id == AGENT_HUMAN:
            if my_act == ACTION_FORWARD and self._is_facing_wall(state.human_pose, w, h):
                obs.add(OBS_BUMP)
            if state.wumpus_alive and self._is_adjacent(state.human_pose.pos(), state.wumpus_pose.pos()):
                obs.add(OBS_STENCH)
            for pit in state.pit_locations:
                if self._is_adjacent(state.human_pose.pos(), pit):
                    obs.add(OBS_BREEZE)
            if state.human_pose.pos() == state.gold_location and not state.has_gold:
                obs.add(OBS_GLITTER)
            if my_act == ACTION_SHOOT and not state.wumpus_alive:
                obs.add(OBS_SCREAM)

        elif agent_id == AGENT_WUMPUS:
            if my_act == ACTION_FORWARD and self._is_facing_wall(state.wumpus_pose, w, h):
                obs.add(OBS_BUMP)
            for pit in state.pit_locations:
                if self._is_adjacent(state.wumpus_pose.pos(), pit):
                    obs.add(OBS_BREEZE)
            if state.wumpus_pose.pos() == state.gold_location:
                obs.add(OBS_GLITTER)

            # [PHASE 3 FIX]: Wumpus can hear the human moving if they are within Manhattan distance of 2.
            # This is strictly required for the Wumpus to build a mental model of the human's position.
            if state.human_alive:
                dist = abs(state.human_pose.x - state.wumpus_pose.x) + abs(state.human_pose.y - state.wumpus_pose.y)
                if dist <= 2:
                    obs.add(OBS_FOOTSTEP)

        return tuple(sorted(list(obs)))

    def _is_adjacent(self, p1, p2) -> bool:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def _is_facing_wall(self, pose: AgentPose, w: int, h: int) -> bool:
        x, y = pose.x, pose.y
        if pose.orientation == NORTH and y == h - 1: return True
        if pose.orientation == SOUTH and y == 0: return True
        if pose.orientation == EAST and x == w - 1: return True
        if pose.orientation == WEST and x == 0: return True
        return False

    def get_reward(self, state: WumpusState, joint_action: Dict[AgentID, Action], next_state: WumpusState,
                   agent_id: AgentID) -> float:
        if str(agent_id) == AGENT_HUMAN:
            if next_state.has_gold: return 1000.0
            if not next_state.human_alive: return -1000.0
            if joint_action.get(AGENT_HUMAN) == ACTION_SHOOT: return -10.0
            return -1.0

        if str(agent_id) == AGENT_WUMPUS:
            if state.human_alive and not next_state.human_alive:
                h_pos = next_state.human_pose.pos()
                w_pos = next_state.wumpus_pose.pos()
                collision = (h_pos == w_pos)
                swap = (h_pos == state.wumpus_pose.pos()) and (w_pos == state.human_pose.pos())

                if collision or swap:
                    return 1000.0

            if state.wumpus_alive and not next_state.wumpus_alive: return -1000.0
            return -1.0

    def get_observation_prob(self, observation: Observation, state: WumpusState, joint_action: Dict[AgentID, Action],
                             agent_id: AgentID) -> float:
        # [PHASE 3 FIX]: Strict Mathematical Symmetry.
        # Wumpus sensors are deterministic. Fake epsilon noise has been removed.
        true_obs = set(self.sample_observation(state, joint_action, agent_id))

        if set(observation) == true_obs:
            return 1.0
        return 0.0

    def is_terminal(self, state: WumpusState) -> bool:
        return (not state.human_alive) or state.has_gold