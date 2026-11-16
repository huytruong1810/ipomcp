"""Exhaustive small-scale checks of the domains and of canonical identity.

``test_domain_contracts.py`` samples a few states per domain. Here every state
reachable from the prior of a small configuration is visited with every joint
action, so a kernel that fails to normalize, a reward outside its declared
bounds or an undeclared observation cannot hide in a state the sample missed.
The published Tiger is compared with Table 1 of Gmytrasiewicz and Doshi (2005),
typed in from the paper entry by entry, and the Wumpus rules are checked on
hand-built situations. None of these checks reads a value from the code it
tests: the expectations are the paper's table, the declared specification, or
arithmetic done here.
"""

import itertools
import math
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from ipomdp.domains.runner_chaser import RunnerChaserModel
from ipomdp.domains.tiger import (
    CREAK_LEFT,
    CREAK_RIGHT,
    GROWL_LEFT,
    GROWL_RIGHT,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    SILENCE,
    TIGER_LEFT,
    TIGER_RIGHT,
    JointTigerModel,
    TigerModel,
)
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.constants import (
    ACTION_FORWARD,
    ACTION_GRAB,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    AGENT_HUMAN,
    AGENT_WUMPUS,
    EAST,
    NORTH,
    OBS_FOOTSTEP,
    OBS_SCREAM,
    OBS_STENCH,
    WEST,
)
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.domains.wumpus.state import AgentPose

# Every configuration here is small enough to enumerate in about a second.
SMALL = {
    "tiger": (TigerModel(), ("i", "j")),
    "tiger-coordination": (JointTigerModel(together_safe=40.0), ("i", "j")),
    "tiger-race": (JointTigerModel(together_safe=0.0, solo_penalty=10.0), ("i", "j")),
    "uav-3x3": (UAVModel(), ("i", "j")),
    "uav-2x2": (UAVModel(rows=2, cols=2), ("i", "j")),
    "wumpus-2x2": (WumpusModel(width=2, height=2, n_pits=1), ("i", "j")),
    "runner-chaser-cap3": (RunnerChaserModel("cap3"), ("r", "c")),
    "runner-chaser-cap4": (RunnerChaserModel("cap4"), ("r", "c")),
}


def joint_actions(model, agents):
    return [
        dict(zip(agents, pair))
        for pair in itertools.product(*(model.get_all_actions(agent) for agent in agents))
    ]


def reachable(model, agents):
    """Every state reachable from the prior; terminal states are kept but not expanded."""
    seen = {state for state, _ in model.initial_distribution()}
    frontier = list(seen)
    joints = joint_actions(model, agents)
    while frontier:
        following = []
        for state in frontier:
            if model.is_terminal(state):
                continue
            for joint in joints:
                for successor, _ in model.transition_distribution(state, joint):
                    if successor not in seen:
                        seen.add(successor)
                        following.append(successor)
        frontier = following
    return seen


def assert_distribution(distribution, support=None):
    values = [value for value, _ in distribution]
    assert len(values) == len(set(values)), "repeated outcomes"
    assert all(math.isfinite(p) and p > 0 for _, p in distribution)
    assert math.fsum(p for _, p in distribution) == pytest.approx(1, abs=1e-12)
    if support is not None:
        assert set(values) <= support


@pytest.mark.parametrize("name", SMALL)
def test_every_reachable_kernel_is_a_distribution_and_every_reward_is_within_bounds(name):
    model, agents = SMALL[name]
    joints = joint_actions(model, agents)
    observations = {agent: set(model.get_all_observations(agent)) for agent in agents}
    bounds = {agent: model.reward_bounds(agent) for agent in agents}
    assert_distribution(model.initial_distribution())
    states = reachable(model, agents)
    assert any(not model.is_terminal(state) for state in states)
    for state in states:
        if model.is_terminal(state):
            continue
        for joint in joints:
            transitions = model.transition_distribution(state, joint)
            assert_distribution(transitions, states)
            for successor, _ in transitions:
                for agent in agents:
                    low, high = bounds[agent]
                    assert low <= model.get_reward(state, joint, successor, agent) <= high
                    # The exact filter reads the observation law of a terminal successor
                    # too (it conditions on termination afterwards), so it must be a
                    # distribution there as well.
                    assert_distribution(
                        model.observation_distribution(successor, joint, agent), observations[agent]
                    )


@pytest.mark.parametrize("name", SMALL)
def test_declared_reward_bounds_are_attained(name):
    """Loose bounds are sound but waste the planner's exploration scale and widen every interval."""
    model, agents = SMALL[name]
    joints = joint_actions(model, agents)
    for agent in agents:
        rewards = [
            model.get_reward(state, joint, successor, agent)
            for state in reachable(model, agents)
            if not model.is_terminal(state)
            for joint in joints
            for successor, _ in model.transition_distribution(state, joint)
        ]
        assert (min(rewards), max(rewards)) == model.reward_bounds(agent)


# --------------------------------------------------------------------------- #
# Tiger: Table 1 of Gmytrasiewicz & Doshi (2005), JAIR 24, Section 7.1, as printed
# --------------------------------------------------------------------------- #

TABLE_1_COLUMNS = [
    (GROWL_LEFT, CREAK_LEFT),
    (GROWL_LEFT, CREAK_RIGHT),
    (GROWL_LEFT, SILENCE),
    (GROWL_RIGHT, CREAK_LEFT),
    (GROWL_RIGHT, CREAK_RIGHT),
    (GROWL_RIGHT, SILENCE),
]
# Observation function of a listening agent: (the other agent's action, state) -> the row.
TABLE_1_LISTENER = {
    (LISTEN, TIGER_LEFT): [0.85 * 0.05, 0.85 * 0.05, 0.85 * 0.9, 0.15 * 0.05, 0.15 * 0.05, 0.15 * 0.9],
    (LISTEN, TIGER_RIGHT): [0.15 * 0.05, 0.15 * 0.05, 0.15 * 0.9, 0.85 * 0.05, 0.85 * 0.05, 0.85 * 0.9],
    (OPEN_LEFT, TIGER_LEFT): [0.85 * 0.9, 0.85 * 0.05, 0.85 * 0.05, 0.15 * 0.9, 0.15 * 0.05, 0.15 * 0.05],
    (OPEN_LEFT, TIGER_RIGHT): [0.15 * 0.9, 0.15 * 0.05, 0.15 * 0.05, 0.85 * 0.9, 0.85 * 0.05, 0.85 * 0.05],
    (OPEN_RIGHT, TIGER_LEFT): [0.85 * 0.05, 0.85 * 0.9, 0.85 * 0.05, 0.15 * 0.05, 0.15 * 0.9, 0.15 * 0.05],
    (OPEN_RIGHT, TIGER_RIGHT): [0.15 * 0.05, 0.15 * 0.9, 0.15 * 0.05, 0.85 * 0.05, 0.85 * 0.9, 0.85 * 0.05],
}
# Reward function of agent i: (a_i, a_j) -> (reward in TL, reward in TR).
TABLE_1_REWARD_I = {
    (OPEN_RIGHT, OPEN_RIGHT): (10, -100),
    (OPEN_LEFT, OPEN_LEFT): (-100, 10),
    (OPEN_RIGHT, OPEN_LEFT): (10, -100),
    (OPEN_LEFT, OPEN_RIGHT): (-100, 10),
    (LISTEN, LISTEN): (-1, -1),
    (LISTEN, OPEN_RIGHT): (-1, -1),
    (OPEN_RIGHT, LISTEN): (10, -100),
    (LISTEN, OPEN_LEFT): (-1, -1),
    (OPEN_LEFT, LISTEN): (-100, 10),
}
# Reward function of agent j, as printed (it is i's with the roles exchanged).
TABLE_1_REWARD_J = {
    (OPEN_RIGHT, OPEN_RIGHT): (10, -100),
    (OPEN_LEFT, OPEN_LEFT): (-100, 10),
    (OPEN_RIGHT, OPEN_LEFT): (-100, 10),
    (OPEN_LEFT, OPEN_RIGHT): (10, -100),
    (LISTEN, LISTEN): (-1, -1),
    (LISTEN, OPEN_RIGHT): (10, -100),
    (OPEN_RIGHT, LISTEN): (-1, -1),
    (LISTEN, OPEN_LEFT): (-100, 10),
    (OPEN_LEFT, LISTEN): (-1, -1),
}
TIGER_ACTIONS = (LISTEN, OPEN_LEFT, OPEN_RIGHT)
TIGER_STATES = (TIGER_LEFT, TIGER_RIGHT)


def test_tiger_transition_function_equals_table_1():
    model = TigerModel()
    for a_i, a_j, state in itertools.product(TIGER_ACTIONS, TIGER_ACTIONS, TIGER_STATES):
        law = dict(model.transition_distribution(state, {"i": a_i, "j": a_j}))
        if a_i == a_j == LISTEN:
            assert law == {state: 1.0}
        else:
            assert law == {TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}


def test_tiger_reward_functions_equal_table_1():
    model = TigerModel()
    for (a_i, a_j), by_state in TABLE_1_REWARD_I.items():
        for state, expected in zip(TIGER_STATES, by_state):
            for successor in TIGER_STATES:
                joint = {"i": a_i, "j": a_j}
                assert model.get_reward(state, joint, successor, "i") == expected
                assert model.get_reward(state, joint, successor, "j") == TABLE_1_REWARD_J[a_i, a_j][
                    TIGER_STATES.index(state)
                ]


@pytest.mark.parametrize("agent, other", [("i", "j"), ("j", "i")])
def test_tiger_observation_function_equals_table_1(agent, other):
    model = TigerModel()
    for (other_action, state), row in TABLE_1_LISTENER.items():
        law = dict(model.observation_distribution(state, {agent: LISTEN, other: other_action}, agent))
        assert law == pytest.approx(dict(zip(TABLE_1_COLUMNS, row)), abs=1e-15)
    for own, other_action, state in itertools.product(
        (OPEN_LEFT, OPEN_RIGHT), TIGER_ACTIONS, TIGER_STATES
    ):
        law = dict(model.observation_distribution(state, {agent: own, other: other_action}, agent))
        assert law == pytest.approx({column: 1 / 6 for column in TABLE_1_COLUMNS}, abs=1e-15)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"together_safe": 40.0},
        {"together_safe": 0.0, "solo_penalty": 30.0},
        {"together_safe": -200.0},
        {"together_safe": -150.0, "solo_penalty": 20.0},
        {"together_safe": 5.0, "solo_penalty": 3.0},
    ],
)
def test_joint_tiger_bounds_contain_every_reward_for_any_payoff(options):
    """The bounds scale exploration and every certified interval, so they must hold for all payoffs."""
    model = JointTigerModel(**options)
    rewards = [
        model.get_reward(state, {"i": a, "j": b}, state, "i")
        for state, a, b in itertools.product(TIGER_STATES, TIGER_ACTIONS, TIGER_ACTIONS)
    ]
    assert (min(rewards), max(rewards)) == model.reward_bounds("i") == model.reward_bounds("j")


# --------------------------------------------------------------------------- #
# Wumpus: the declared rules on hand-built situations
# --------------------------------------------------------------------------- #


def wumpus_state(model, **changes):
    state, _ = model.initial_distribution()[0]
    return replace(state, **changes)


def step(model, state, human, wumpus):
    joint = {AGENT_HUMAN: human, AGENT_WUMPUS: wumpus}
    ((following, probability),) = model.transition_distribution(state, joint)
    assert probability == 1.0
    return joint, following


def rewards(model, state, joint, following):
    return tuple(model.get_reward(state, joint, following, agent) for agent in (AGENT_HUMAN, AGENT_WUMPUS))


def percepts(model, state, joint, agent):
    ((observation, probability),) = model.observation_distribution(state, joint, agent)
    assert probability == 1.0
    return observation


def test_wumpus_kills_by_moving_onto_the_human_or_by_swapping_cells():
    model = WumpusModel(n_pits=0)
    meet = wumpus_state(model, human_pose=AgentPose(0, 0, EAST), wumpus_pose=AgentPose(2, 0, WEST))
    joint, after = step(model, meet, ACTION_FORWARD, ACTION_FORWARD)
    assert after.human_pose.pos() == after.wumpus_pose.pos() == (1, 0)
    assert not after.human_alive and model.is_terminal(after)
    assert rewards(model, meet, joint, after) == (-1000.0, 1000.0)
    swap = wumpus_state(model, human_pose=AgentPose(1, 0, EAST), wumpus_pose=AgentPose(2, 0, WEST))
    joint, after = step(model, swap, ACTION_FORWARD, ACTION_FORWARD)
    assert (after.human_pose.pos(), after.wumpus_pose.pos()) == ((2, 0), (1, 0))
    assert not after.human_alive
    assert rewards(model, swap, joint, after) == (-1000.0, 1000.0)


def test_wumpus_death_takes_priority_over_grabbing_the_gold():
    model = WumpusModel(n_pits=0)
    state = wumpus_state(
        model,
        human_pose=AgentPose(1, 1, EAST),
        wumpus_pose=AgentPose(2, 1, WEST),
        gold_location=(1, 1),
    )
    joint, after = step(model, state, ACTION_GRAB, ACTION_FORWARD)
    assert not after.human_alive and model.is_terminal(after)
    assert rewards(model, state, joint, after) == (-1000.0, 1000.0)
    # Without the wumpus arriving, the same grab wins.
    joint, after = step(model, state, ACTION_GRAB, ACTION_TURN_LEFT)
    assert after.human_alive and after.has_gold and model.is_terminal(after)
    assert rewards(model, state, joint, after) == (1000.0, -1.0)


def test_wumpus_shot_uses_the_poses_before_the_move_and_needs_the_line_of_fire():
    model = WumpusModel(n_pits=0)
    # The wumpus steps out of the row in the same step; the shot was fired at where it stood.
    state = wumpus_state(model, human_pose=AgentPose(0, 0, EAST), wumpus_pose=AgentPose(3, 0, NORTH))
    joint, after = step(model, state, ACTION_SHOOT, ACTION_FORWARD)
    assert not after.wumpus_alive and not after.has_arrow and after.screamed
    assert rewards(model, state, joint, after) == (-10.0, -1000.0)
    assert OBS_SCREAM in percepts(model, after, joint, AGENT_HUMAN)
    assert OBS_STENCH not in percepts(model, after, joint, AGENT_HUMAN)
    # A dead wumpus neither moves nor kills.
    joint, later = step(model, replace(after, human_pose=AgentPose(3, 0, NORTH)), ACTION_FORWARD, ACTION_FORWARD)
    assert later.wumpus_pose == after.wumpus_pose and later.human_alive
    # Not in the line of fire: the arrow is spent, nothing dies.
    state = wumpus_state(model, human_pose=AgentPose(0, 0, EAST), wumpus_pose=AgentPose(0, 3, NORTH))
    joint, after = step(model, state, ACTION_SHOOT, ACTION_TURN_LEFT)
    assert after.wumpus_alive and not after.has_arrow and not after.screamed
    assert rewards(model, state, joint, after) == (-10.0, -1.0)


def test_wumpus_pit_death_is_not_a_kill_by_the_wumpus():
    model = WumpusModel(n_pits=0)
    state = wumpus_state(
        model,
        human_pose=AgentPose(0, 0, EAST),
        wumpus_pose=AgentPose(3, 3, NORTH),
        pit_locations=frozenset({(1, 0)}),
    )
    joint, after = step(model, state, ACTION_FORWARD, ACTION_TURN_LEFT)
    assert not after.human_alive
    assert rewards(model, state, joint, after) == (-1000.0, -1.0)


@pytest.mark.parametrize(
    "wumpus, heard", [((2, 0), True), ((1, 1), True), ((3, 0), False), ((2, 1), False)]
)
def test_wumpus_hears_footsteps_within_manhattan_distance_two(wumpus, heard):
    model = WumpusModel(n_pits=0)
    state = wumpus_state(model, human_pose=AgentPose(0, 0, EAST), wumpus_pose=AgentPose(*wumpus, NORTH))
    joint = {AGENT_HUMAN: ACTION_TURN_LEFT, AGENT_WUMPUS: ACTION_TURN_LEFT}
    assert (OBS_FOOTSTEP in percepts(model, state, joint, AGENT_WUMPUS)) is heard


# --------------------------------------------------------------------------- #
# Canonical identity across processes
# --------------------------------------------------------------------------- #

_DIGEST_SCRIPT = """
import random
from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.runner_chaser import RunnerChaserModel
from ipomdp.domains.tiger import JointTigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel

# Strings are what the interpreter's hash seed reorders (integers hash to themselves), so the encoding of
# a set and of a dict of strings is printed first: without canonical sorting these lines differ by seed.
words = ["growl", "creak", "silence", "stench", "breeze", "glitter", "bump", "scream", "footstep"]
print(canonical.digest(canonical.stable_value(set(words))),
      canonical.digest(canonical.stable_value(frozenset(zip(words, range(9))))),
      canonical.digest(canonical.stable_value({word: {word[::-1]} for word in words})))

for physics, other in [(JointTigerModel(together_safe=40.0), "j"), (UAVModel(), "j"),
                       (WumpusModel(width=3, height=3, n_pits=2), "j"), (RunnerChaserModel("cap4"), "c")]:
    opponent = MentalModel(AgentFrame(other, 0, physics))
    rows = [(InteractiveState(state, opponent), p) for state, p in physics.initial_distribution()][:400]
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    first, second = FiniteBelief(tuple(rows)), FiniteBelief(tuple(shuffled))
    assert first == second and canonical.canonical_mass(first) == canonical.canonical_mass(second)
    print(canonical.digest(canonical.physics_identity(physics)), canonical.belief_digest(first),
          canonical.belief_digest(second))
"""


def test_digests_do_not_depend_on_the_hash_seed_or_on_construction_order():
    """Search seeds are digests, so a digest that moved with ``PYTHONHASHSEED`` would make runs irreproducible.

    Sets and frozensets (Runner-Chaser's cells, Wumpus's pits) iterate in an order that
    depends on the interpreter's hash seed; the canonical encoding must sort them away.
    """
    outputs = []
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run(
            [sys.executable, "-c", _DIGEST_SCRIPT], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    lines = outputs[0].splitlines()
    assert len(lines) == 5 and all(len(digest) == 64 for digest in lines[0].split())
    for line in lines[1:]:
        physics, first, second = line.split()
        assert first == second and len(physics) == 64
