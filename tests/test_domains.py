"""Specification checks for each domain (behaviour the contract suite cannot know)."""

import pytest

from ipomdp.domains.tiger import (
    CREAK_LEFT,
    CREAK_RIGHT,
    GROWL_LEFT,
    GROWL_RIGHT,
    LISTEN,
    OBSERVATIONS,
    OPEN_LEFT,
    OPEN_RIGHT,
    SILENCE,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from ipomdp.domains.uav import MOVE_E, MOVE_N, MOVE_S, NONE, SENSE, UAVModel
from ipomdp.domains.wumpus.constants import ACTION_SHOOT, AGENT_HUMAN, AGENT_WUMPUS
from ipomdp.domains.wumpus.model import WumpusModel

# --------------------------------------------------------------------------- #
# Tiger: Gmytrasiewicz & Doshi (2005), Table 1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,other,observation,expected",
    [
        (TIGER_LEFT, LISTEN, (GROWL_LEFT, SILENCE), 0.85 * 0.9),
        (TIGER_LEFT, LISTEN, (GROWL_LEFT, CREAK_LEFT), 0.85 * 0.05),  # false creak
        (TIGER_LEFT, LISTEN, (GROWL_RIGHT, CREAK_RIGHT), 0.15 * 0.05),
        (TIGER_LEFT, OPEN_LEFT, (GROWL_LEFT, CREAK_LEFT), 0.85 * 0.9),
        (TIGER_LEFT, OPEN_LEFT, (GROWL_LEFT, SILENCE), 0.85 * 0.05),
        (TIGER_RIGHT, OPEN_RIGHT, (GROWL_RIGHT, CREAK_RIGHT), 0.85 * 0.9),
        (TIGER_RIGHT, OPEN_RIGHT, (GROWL_LEFT, CREAK_LEFT), 0.15 * 0.05),
    ],
)
def test_tiger_listener_observations_match_table_1(state, other, observation, expected):
    model = TigerModel()
    for agent, opponent in (("i", "j"), ("j", "i")):
        joint = {agent: LISTEN, opponent: other}
        assert model.observation_probability(observation, state, joint, agent) == pytest.approx(
            expected
        )


@pytest.mark.parametrize("own", [OPEN_LEFT, OPEN_RIGHT])
def test_tiger_opening_agent_observes_uniformly(own):
    law = TigerModel().observation_distribution(TIGER_LEFT, {"i": own, "j": LISTEN}, "i")
    assert dict(law) == pytest.approx({o: 1 / 6 for o in OBSERVATIONS})


def test_tiger_transition_and_rewards_match_table_1():
    model = TigerModel()
    assert model.transition_distribution(TIGER_LEFT, {"i": LISTEN, "j": LISTEN}) == (
        (TIGER_LEFT, 1.0),
    )
    for joint in ({"i": OPEN_LEFT, "j": LISTEN}, {"i": LISTEN, "j": OPEN_RIGHT}):
        assert dict(model.transition_distribution(TIGER_RIGHT, joint)) == {
            TIGER_LEFT: 0.5,
            TIGER_RIGHT: 0.5,
        }
    # Rewards depend on the agent's own action only.
    for other in (LISTEN, OPEN_LEFT, OPEN_RIGHT):
        joint = {"i": OPEN_LEFT, "j": other}
        assert model.get_reward(TIGER_LEFT, joint, TIGER_RIGHT, "i") == -100
        assert model.get_reward(TIGER_RIGHT, joint, TIGER_LEFT, "i") == 10
        assert model.get_reward(TIGER_RIGHT, {"i": LISTEN, "j": other}, TIGER_RIGHT, "i") == -1


def test_tiger_rollout_memory_marginalises_a_uniform_opponent():
    model = TigerModel()
    prior = {TIGER_LEFT: 0.9, TIGER_RIGHT: 0.1}
    memory = model.update_rollout_memory(prior, LISTEN, (GROWL_LEFT, SILENCE), "i")
    # Independent enumeration: the opponent listens (tiger stays, silence .9) or
    # opens either door (tiger resets uniformly, silence .05), each with 1/3.
    growl = {TIGER_LEFT: 0.85, TIGER_RIGHT: 0.15}
    joint = {s: prior[s] / 3 * growl[s] * 0.9 + 2 / 3 * 0.5 * growl[s] * 0.05 for s in prior}
    assert memory[TIGER_LEFT] == pytest.approx(joint[TIGER_LEFT] / sum(joint.values()))


# --------------------------------------------------------------------------- #
# UAV
# --------------------------------------------------------------------------- #


def test_uav_agents_know_their_own_cell_and_sense_direction():
    model = UAVModel(sensor_accuracy=0.85)
    state = ((2, 0), (0, 2))  # evader north-east of the pursuer
    law = dict(model.observation_distribution(state, {"i": SENSE, "j": MOVE_N}, "i"))
    assert law[((2, 0), "NE")] == pytest.approx(0.85)
    assert all(cell == (2, 0) for cell, _ in law)
    moving = model.observation_distribution(state, {"i": MOVE_E, "j": SENSE}, "i")
    assert moving == ((((2, 0), NONE), 1.0),)
    evader = dict(model.observation_distribution(state, {"i": MOVE_E, "j": SENSE}, "j"))
    assert evader[((0, 2), "SW")] == pytest.approx(0.85)


def test_uav_capture_on_shared_cell_only():
    model = UAVModel()
    meet = model.transition_distribution(((1, 0), (1, 2)), {"i": MOVE_E, "j": "W"})[0][0]
    assert meet == ((1, 1), (1, 1)) and model.is_terminal(meet)
    assert model.get_reward(((1, 0), (1, 2)), {"i": MOVE_E, "j": "W"}, meet, "i") == 1.0
    swap = model.transition_distribution(((1, 0), (1, 1)), {"i": MOVE_E, "j": "W"})[0][0]
    assert swap == ((1, 1), (1, 0)) and not model.is_terminal(swap)
    assert model.transition_distribution(((0, 0), (2, 2)), {"i": MOVE_N, "j": MOVE_S})[0][0] == (
        (0, 0),
        (2, 2),
    )


# --------------------------------------------------------------------------- #
# Wumpus
# --------------------------------------------------------------------------- #


def test_wumpus_prior_is_exact_and_uniform():
    prior = WumpusModel().initial_distribution()
    assert len(prior) == 15 * 4 * 15 * 105
    assert {p for _, p in prior} == {1 / len(prior)}


def test_wumpus_shooting_without_arrow_is_allowed_and_costly():
    model = WumpusModel()
    state, _ = model.initial_distribution()[0]
    spent = model.transition_distribution(
        state, {AGENT_HUMAN: ACTION_SHOOT, AGENT_WUMPUS: "TurnLeft"}
    )[0][0]
    assert not spent.has_arrow
    joint = {AGENT_HUMAN: ACTION_SHOOT, AGENT_WUMPUS: "TurnLeft"}
    again = model.transition_distribution(spent, joint)[0][0]
    assert again.has_arrow is False and again.wumpus_alive == spent.wumpus_alive
    assert model.get_reward(spent, joint, again, AGENT_HUMAN) == -10
