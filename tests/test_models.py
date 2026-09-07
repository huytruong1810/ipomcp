import pytest
from examples.tiger.model.tiger_model import TigerModel, TIGER_LEFT, TIGER_RIGHT, LISTEN, OPEN_LEFT, OPEN_RIGHT, GROWL_LEFT, GROWL_RIGHT, SILENCE, CREAK_LEFT, CREAK_RIGHT
from examples.uav.model.uav_model import UAVModel, UAVState, MOVE_N, MOVE_S, MOVE_E, MOVE_W, LISTEN as UAV_LISTEN
from examples.wumpus.model.wumpus_model import WumpusModel
from examples.wumpus.model.constants import *


def test_tiger_model_symmetry():
    model = TigerModel(growl_accuracy={"i": 0.85, "j": 0.85}, creak_accuracy=0.90)
    
    # 1. State reset on open door
    s0 = TIGER_LEFT
    s_next = model.sample_transition(s0, {"i": OPEN_LEFT, "j": LISTEN})
    assert s_next in [TIGER_LEFT, TIGER_RIGHT]
    
    # 2. Observation probability symmetry
    # Listener hearing growl
    prob = model.get_observation_prob((GROWL_LEFT, SILENCE), TIGER_LEFT, {"i": LISTEN, "j": LISTEN}, "i")
    assert prob == pytest.approx(0.85 * 1.0)
    
    # Opener deafened
    prob_deaf = model.get_observation_prob((SILENCE, SILENCE), TIGER_LEFT, {"i": OPEN_LEFT, "j": LISTEN}, "i")
    assert prob_deaf == 1.0
    
    # 3. Action and observation spaces
    assert len(model.get_all_actions("i")) == 3
    assert len(model.get_all_observations("i")) == 9


def test_uav_model_dynamics():
    model = UAVModel(sensor_accuracy=0.85)
    s = UAVState((1, 1), (2, 2))
    
    # Move UAV North and Target South
    s_next = model.sample_transition(s, {"i": MOVE_N, "j": MOVE_S})
    assert s_next.uav_pos == (0, 1)
    assert s_next.target_pos == (2, 2)  # clamped at grid boundary
    
    # Observation prob
    prob = model.get_observation_prob("R2", s, {"i": UAV_LISTEN, "j": UAV_LISTEN}, "i")
    assert prob == pytest.approx(0.85)
    
    assert len(model.get_all_actions("i")) == 5
    assert len(model.get_all_observations("i")) == 3


def test_wumpus_model_mechanics():
    model = WumpusModel(width=4, height=4, n_pits=1)
    s = model.get_initial_state()
    
    # Legal actions before shooting
    legal = model.get_legal_actions(s, AGENT_HUMAN)
    assert ACTION_SHOOT in legal
    
    # Action and observation enumerations
    assert len(model.get_all_actions(AGENT_HUMAN)) == 5
    assert len(model.get_all_actions(AGENT_WUMPUS)) == 3
    assert len(model.get_all_observations(AGENT_HUMAN)) == 32


def test_uav_model_legal_actions_and_rng():
    import random
    model = UAVModel(sensor_accuracy=0.85)

    # At top-left corner (0, 0), MOVE_N and MOVE_W are illegal
    s_corner = UAVState((0, 0), (2, 2))
    legal_uav = set(model.get_legal_actions(s_corner, "i"))
    assert MOVE_N not in legal_uav
    assert MOVE_W not in legal_uav
    assert MOVE_S in legal_uav
    assert MOVE_E in legal_uav
    assert UAV_LISTEN in legal_uav

    # At center (1, 1), all moves are legal
    s_center = UAVState((1, 1), (2, 2))
    legal_center = set(model.get_legal_actions(s_center, "i"))
    assert len(legal_center) == 5

    # RNG reproducibility
    rng1 = random.Random(42)
    s1 = model.get_initial_state(rng=rng1)
    rng2 = random.Random(42)
    s2 = model.get_initial_state(rng=rng2)
    assert s1 == s2


def test_wumpus_model_rng_reproducibility():
    import random
    model = WumpusModel(width=4, height=4, n_pits=2)
    rng1 = random.Random(99)
    s1 = model.get_initial_state(rng=rng1)
    rng2 = random.Random(99)
    s2 = model.get_initial_state(rng=rng2)
    assert s1 == s2
