"""Analytic boundary cases for search routing and deterministic event sensors."""

import random
from dataclasses import replace

import pytest

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig
from examples.tiger.model.tiger_model import TigerModel
from examples.wumpus.model.constants import (
    ACTION_FORWARD,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    AGENT_HUMAN,
    AGENT_WUMPUS,
    EAST,
    NORTH,
    OBS_BUMP,
    OBS_SCREAM,
)
from examples.wumpus.model.wumpus_model import WumpusModel
from examples.wumpus.model.wumpus_state import AgentPose
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper


def test_search_does_not_resample_its_own_root_belief():
    random.seed(3401)
    planner = I_POMDP_Bootstrapper(SolverBank()).create_solver(
        "i",
        1,
        TigerModel(),
        ["j"],
        n_particles=50,
        config=IPOMCPConfig(mcts=MCTSConfig(n_sims=200, max_depth=3, node_capacity=50)),
    )
    before = planner.belief
    planner.get_action()
    assert planner.belief is before
    assert not planner.root.belief_particles
    assert planner.root._total_particles_routed == 0
    # Every root traversal reaches exactly one first-level observation child.
    for action, count in planner.root.action_counts.items():
        assert (
            sum(child._total_particles_routed for child in planner.root.children[action].values())
            == count
        )


def test_bump_is_blocked_motion_not_arrival_at_boundary():
    model = WumpusModel(width=4, height=4, n_pits=0)
    state = replace(
        model.get_initial_state(random.Random(3)),
        human_pose=AgentPose(2, 0, EAST),
        wumpus_pose=AgentPose(0, 3, NORTH),
    )
    actions = {AGENT_HUMAN: ACTION_FORWARD, AGENT_WUMPUS: ACTION_TURN_LEFT}
    arrived = model.sample_transition(state, actions)
    assert arrived.human_pose.x == 3
    assert OBS_BUMP not in model.sample_observation(arrived, actions, AGENT_HUMAN)
    blocked = model.sample_transition(arrived, actions)
    observation = model.sample_observation(blocked, actions, AGENT_HUMAN)
    assert OBS_BUMP in observation
    assert model.get_observation_prob(observation, blocked, actions, AGENT_HUMAN) == 1


def test_scream_is_a_new_kill_and_terminal_reward_is_not_repeated():
    model = WumpusModel(n_pits=0)
    state = replace(
        model.get_initial_state(random.Random(3)),
        human_pose=AgentPose(0, 0, EAST),
        wumpus_pose=AgentPose(2, 0, NORTH),
    )
    actions = {AGENT_HUMAN: ACTION_SHOOT, AGENT_WUMPUS: ACTION_TURN_LEFT}
    killed = model.sample_transition(state, actions)
    assert OBS_SCREAM in model.sample_observation(killed, actions, AGENT_HUMAN)
    repeated = model.sample_transition(killed, actions)
    assert OBS_SCREAM not in model.sample_observation(repeated, actions, AGENT_HUMAN)
    dead = replace(state, human_alive=False)
    assert model.get_reward(dead, actions, dead, AGENT_HUMAN) == 0
    simultaneous = replace(state, human_alive=False, has_gold=True)
    assert model.get_reward(state, actions, simultaneous, AGENT_HUMAN) == -1000


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MCTSConfig(n_sims=0),
        lambda: MCTSConfig(node_capacity=-1),
        lambda: MCTSConfig(gamma=float("nan")),
        lambda: ExperimentConfig(n_trials=0),
    ],
)
def test_invalid_configuration_fails_before_work(factory):
    with pytest.raises(ValueError):
        factory()


def test_tiger_creak_resets_rollout_belief():
    from examples.tiger.model.tiger_model import (
        CREAK_RIGHT,
        GROWL_LEFT,
        LISTEN,
        OPEN_RIGHT,
        SILENCE,
        TIGER_LEFT,
        TIGER_RIGHT,
    )

    model = TigerModel()
    prior_high_confidence = {TIGER_LEFT: 0.01, TIGER_RIGHT: 0.99}

    # If opponent opened a door (creak right observed), old belief resets to 0.5 before applying growl left
    updated = model.update_rollout_belief(
        prior_high_confidence, LISTEN, (GROWL_LEFT, CREAK_RIGHT), "i"
    )
    assert updated[TIGER_LEFT] == pytest.approx(0.85)
    assert updated[TIGER_RIGHT] == pytest.approx(0.15)

    # If agent opens door itself, deafened and reset to 0.5/0.5
    self_opened = model.update_rollout_belief(
        prior_high_confidence, OPEN_RIGHT, (SILENCE, SILENCE), "i"
    )
    assert self_opened == {TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}


def test_tiger_persistent_rollout_open_does_not_reset():
    model = TigerModel(persistent=True)
    prior = {"TL": 0.9, "TR": 0.1}
    assert model.update_rollout_belief(prior, "OR", ("S", "S"), "i") == pytest.approx(prior)


def test_tiger_impossible_rollout_observation_is_not_repaired():
    from ipomdp.finite_filter import UnsupportedObservation

    with pytest.raises(UnsupportedObservation):
        TigerModel().update_rollout_belief({"TL": 0.5, "TR": 0.5}, "OR", ("GL", "S"), "i")


def test_rollout_propagates_opponent_before_next_decision(monkeypatch):
    """A later rollout event must receive the updated private mental model."""
    from ipomdp.finite_belief import InteractiveState, MentalModel
    from ipomdp.frame import AgentFrame

    planner = I_POMDP_Bootstrapper(SolverBank()).create_solver(
        "i",
        2,
        TigerModel(),
        ["j"],
        n_particles=20,
        config=IPOMCPConfig(mcts=MCTSConfig(n_sims=20, max_depth=3)),
    )
    atom = planner.belief.mass[0][0]
    sentinel = MentalModel(AgentFrame("j", 0, planner.pomdp_model))
    updated = InteractiveState(atom.state, sentinel)
    calls = []

    def tree_step(particle, *args):
        calls.append(particle.opponent)
        return updated, {"i": "L", "j": "L"}, -1, False

    def final_event(particle, *args):
        assert particle.opponent is sentinel
        return atom.state, {"i": "L", "j": "L"}, -1, False

    monkeypatch.setattr(planner.gen_model, "tree_step", tree_step)
    monkeypatch.setattr(planner.gen_model, "sample_event", final_event)
    planner._rollout(atom, 1, belief={"TL": 0.5, "TR": 0.5})
    assert calls == [atom.opponent]


def test_search_keeps_all_legal_actions_below_root():
    planner = I_POMDP_Bootstrapper(SolverBank(seed=17)).create_solver(
        "i",
        1,
        TigerModel(),
        ["j"],
        n_particles=100,
        config=IPOMCPConfig(mcts=MCTSConfig(n_sims=2000, max_depth=2)),
    )
    planner.get_action()
    child = planner.root.children["OL"][("S", "S")]
    assert set(child.action_counts) == {"L", "OL", "OR"}
