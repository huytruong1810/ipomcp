"""Analytic boundary cases for search routing and deterministic event sensors."""

import random
from dataclasses import replace

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.domains.wumpus.constants import (
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
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.domains.wumpus.state import AgentPose
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from tests.study_fixtures import random_tiger


def test_search_does_not_modify_its_own_root_belief():
    random.seed(3401)
    planner = HierarchyBuilder(SolverBank(0)).build(
        "i",
        1,
        TigerModel(),
        "j",
        config=PlannerConfig(simulations=200, depth=3, modeled_simulations=10),
    )
    before = planner.belief
    planner.get_action()
    assert planner.belief is before
    # Every root traversal reaches exactly one first-level child or terminates.
    for action, count in planner.root.action_counts.items():
        assert sum(planner.root.continuation_counts[action].values()) == count


def test_bump_is_blocked_motion_not_arrival_at_boundary():
    model = WumpusModel(width=4, height=4, n_pits=0)
    state = replace(
        model.sample_initial_state(random.Random(3)),
        human_pose=AgentPose(2, 0, EAST),
        wumpus_pose=AgentPose(0, 3, NORTH),
    )
    actions = {AGENT_HUMAN: ACTION_FORWARD, AGENT_WUMPUS: ACTION_TURN_LEFT}
    arrived = model.transition_distribution(state, actions)[0][0]
    assert arrived.human_pose.x == 3
    assert OBS_BUMP not in model.observation_distribution(arrived, actions, AGENT_HUMAN)[0][0]
    blocked = model.transition_distribution(arrived, actions)[0][0]
    observation = model.observation_distribution(blocked, actions, AGENT_HUMAN)[0][0]
    assert OBS_BUMP in observation
    assert model.observation_probability(observation, blocked, actions, AGENT_HUMAN) == 1


def test_scream_is_a_new_kill_and_terminal_reward_is_not_repeated():
    model = WumpusModel(n_pits=0)
    state = replace(
        model.sample_initial_state(random.Random(3)),
        human_pose=AgentPose(0, 0, EAST),
        wumpus_pose=AgentPose(2, 0, NORTH),
    )
    actions = {AGENT_HUMAN: ACTION_SHOOT, AGENT_WUMPUS: ACTION_TURN_LEFT}
    killed = model.transition_distribution(state, actions)[0][0]
    assert OBS_SCREAM in model.observation_distribution(killed, actions, AGENT_HUMAN)[0][0]
    repeated = model.transition_distribution(killed, actions)[0][0]
    assert OBS_SCREAM not in model.observation_distribution(repeated, actions, AGENT_HUMAN)[0][0]
    dead = replace(state, human_alive=False)
    assert model.get_reward(dead, actions, dead, AGENT_HUMAN) == 0
    simultaneous = replace(state, human_alive=False, has_gold=True)
    assert model.get_reward(state, actions, simultaneous, AGENT_HUMAN) == -1000


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PlannerConfig(simulations=0, depth=1, modeled_simulations=1),
        lambda: PlannerConfig(simulations=1, depth=0, modeled_simulations=1),
        lambda: PlannerConfig(simulations=1, depth=1, modeled_simulations=1, gamma=float("nan")),
        lambda: random_tiger(trials=0),
    ],
)
def test_invalid_configuration_fails_before_work(factory):
    with pytest.raises(ValueError):
        factory()


def test_tiger_rollout_memory_resets_after_own_opening():
    """An opening agent observes uniformly, so its memory returns to 50/50."""
    model = TigerModel()
    for observation in [("GL", "S"), ("GR", "CL")]:
        memory = model.update_rollout_memory({"TL": 0.01, "TR": 0.99}, "OR", observation, "i")
        assert memory == pytest.approx({"TL": 0.5, "TR": 0.5})


def test_tiger_impossible_rollout_observation_is_not_repaired():
    with pytest.raises(ValueError, match="Impossible"):
        TigerModel().update_rollout_memory({"TL": 0.5, "TR": 0.5}, "L", ("S", "S"), "i")


def test_rollout_propagates_opponent_before_next_decision(monkeypatch):
    """A later rollout event must receive the updated private mental model."""
    from ipomdp.belief.finite_belief import InteractiveState, MentalModel
    from ipomdp.belief.frame import AgentFrame

    planner = HierarchyBuilder(SolverBank(0)).build(
        "i",
        2,
        TigerModel(),
        "j",
        config=PlannerConfig(simulations=20, depth=4, modeled_simulations=5),
    )
    atom = planner.belief.mass[0][0]
    sentinel = MentalModel(AgentFrame("j", 0, planner.pomdp_model))
    updated = InteractiveState(atom.state, sentinel)
    received = []

    def tree_step(particle, *args):
        received.append(particle.opponent)
        return updated, {"i": "L", "j": "L"}, -1, False

    monkeypatch.setattr(planner.gen_model, "tree_step", tree_step)
    monkeypatch.setattr(planner, "_history_successor", lambda model, *args: model)
    planner._rollout(atom, 1, {"TL": 0.5, "TR": 0.5}, planner.model(), random.Random(0))
    # Depth 1 and 2 are sampled steps; depth 3 is the exact last decision.
    assert received == [atom.opponent, sentinel]


def test_search_keeps_all_legal_actions_below_root():
    planner = HierarchyBuilder(SolverBank(seed=17)).build(
        "i",
        1,
        TigerModel(),
        "j",
        config=PlannerConfig(simulations=2000, depth=3, modeled_simulations=10),
    )
    planner.get_action()
    # The opener observes uniformly; take its most visited observation branch.
    child = max(planner.root.children["OL"].values(), key=lambda node: node.visit_count)
    assert set(child.action_counts) == {"L", "OL", "OR"}
