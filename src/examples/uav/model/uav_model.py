import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.pomdp_model import Action, AgentID, Observation, POMDPModel, State

# Grid Dimensions
ROWS, COLS = 3, 3

# Actions
MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN = "N", "S", "E", "W", "listen"

# Action Sets
UAV_ACTIONS = [MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN]
TARGET_ACTIONS = [MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN]

# Observations
OBS_POS_MAP = {ROW_IDX: f"R{ROW_IDX}" for ROW_IDX in range(ROWS)}


@dataclass(frozen=True)
class UAVState:
    uav_pos: Tuple[int, int]  # (row, col)
    target_pos: Tuple[int, int]  # (row, col)

    def __repr__(self):
        return f"S(U:{self.uav_pos}, T:{self.target_pos})"

    def __eq__(self, other):
        if not isinstance(other, UAVState):
            return False
        return self.uav_pos == other.uav_pos and self.target_pos == other.target_pos

    # Make hashable for dictionary keys
    def __hash__(self):
        return hash((self.uav_pos, self.target_pos))


class UAVModel(POMDPModel):
    def __init__(self, sensor_accuracy=0.85):
        self.acc = sensor_accuracy

    def get_initial_state(self, rng=None) -> State:
        rand_int = rng.randint if rng is not None else random.randint
        u_r, u_c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
        t_r, t_c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
        while (t_r, t_c) == (u_r, u_c):
            t_r, t_c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
        return UAVState((u_r, u_c), (t_r, t_c))

    def get_all_actions(self, agent_id: AgentID) -> List[Action]:
        return TARGET_ACTIONS if str(agent_id).lower() == "j" else UAV_ACTIONS

    def get_legal_actions(self, state: UAVState, agent_id: AgentID) -> List[Action]:
        """Keep all clamped moves available: own coordinates are not observed.

        Pruning by a particle's hidden position gives the search information the
        policy does not have. A boundary move remains a valid stay-in-place action.
        """
        return self.get_all_actions(agent_id)

    def get_all_observations(self, agent_id: AgentID) -> List[Observation]:
        """Returns all discrete observation tokens for the UAV domain."""
        return list(OBS_POS_MAP.values())

    def sample_transition(
        self, state: UAVState, joint_action: Dict[AgentID, Action], rng=None
    ) -> State:
        # Parse Actions
        u_act = joint_action.get("i")
        t_act = joint_action.get("j")

        # Move Agents
        new_u = self._move(state.uav_pos, u_act)
        new_t = self._move(state.target_pos, t_act)

        return UAVState(new_u, new_t)

    def _move(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        r, c = pos
        if LISTEN == action:
            return pos
        elif MOVE_N == action:
            r -= 1
        elif MOVE_S == action:
            r += 1
        elif MOVE_E == action:
            c += 1
        elif MOVE_W == action:
            c -= 1

        # Clamp
        r = max(0, min(ROWS - 1, r))
        c = max(0, min(COLS - 1, c))
        return r, c

    def sample_observation(
        self, state: UAVState, joint_action: Dict[AgentID, Action], agent_id: AgentID, rng=None
    ) -> Observation:
        rand_float = rng.random if rng is not None else random.random
        choice_fn = rng.choice if rng is not None else random.choice
        my_action = joint_action.get(str(agent_id).lower())
        choices = list(OBS_POS_MAP.values())
        if my_action == LISTEN:
            true_row = state.target_pos[0] if str(agent_id).lower() == "i" else state.uav_pos[0]
            true_obs = OBS_POS_MAP[true_row]
            if rand_float() < self.acc:
                return true_obs  # accurate obs
            choices.remove(true_obs)  # faulty: equal likelihood of the remaining obs
        return choice_fn(choices)

    def get_observation_prob(
        self,
        observation: Observation,
        state: UAVState,
        joint_action: Dict[AgentID, Action],
        agent_id: AgentID,
    ) -> float:
        if observation not in OBS_POS_MAP.values():
            return 0.0
        my_action = joint_action.get(str(agent_id).lower())
        # Not listening leads to uninformed obs
        if my_action != LISTEN:
            return 1.0 / len(OBS_POS_MAP)

        true_row = state.target_pos[0] if str(agent_id).lower() == "i" else state.uav_pos[0]
        true_obs = OBS_POS_MAP[true_row]

        return self.acc if observation == true_obs else ((1.0 - self.acc) / (len(OBS_POS_MAP) - 1))

    def get_reward(
        self,
        state: UAVState,
        joint_action: Dict[AgentID, Action],
        next_state: UAVState,
        agent_id: AgentID,
    ) -> float:
        """
        Rewards depend on the result of the movement (next_state).
        """
        # Check if the move resulted in a capture
        is_caught = next_state.uav_pos == next_state.target_pos

        # UAV Perspective
        if str(agent_id).lower() == "i":
            return 1.0 if is_caught else -0.1

        # Target Perspective
        return -1.0 if is_caught else 0.1

    def is_terminal(self, state: UAVState) -> bool:
        return state.uav_pos == state.target_pos
