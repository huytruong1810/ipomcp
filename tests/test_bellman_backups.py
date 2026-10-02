"""Independent counterexamples for the empirical chance-weighted Bellman backup."""

import random
from types import SimpleNamespace

import pytest

from ipomdp.experiments.oracle import evaluate_case
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.node import POMCPNode
from ipomdp.planning.solver_types import SolverKey


def test_chance_weighting_is_not_maximum_or_uniform_average_over_outcomes():
    root = POMCPNode()
    root.action_counts["a"] = 10
    root.immediate_reward_means["a"] = 2
    root.continuation_counts["a"] = {"frequent": 9, "rare": 1}
    root.create_child("a", "frequent").action_values = {"good": 10, "bad": -20}
    root.create_child("a", "rare").action_values = {"good": -100, "bad": -200}
    # Chance expectation = .9 * 10 + .1 * -100 = -1, then discount.
    assert root.bellman_backup("a", 2, 0.5) == pytest.approx(1.5)


def test_terminal_probability_is_not_renormalized_away():
    root = POMCPNode()
    root.action_counts["a"] = 10  # Eight terminal events and two continuing.
    root.continuation_counts["a"] = {"alive": 2}
    root.create_child("a", "alive").action_values = {"best": 10}
    assert root.bellman_backup("a", 0, 1) == 2


def test_backup_rereads_improved_child_values_instead_of_averaging_old_returns():
    root = POMCPNode()
    child = root.create_child("a", "obs")
    child.rollout_value = -100
    root.action_counts["a"] = 1
    root.continuation_counts["a"] = {"obs": 1}
    assert root.bellman_backup("a", 0, 1) == -100
    child.action_values = {"best": 5, "other": -10}
    root.action_counts["a"] = 2
    root.continuation_counts["a"]["obs"] = 2
    assert root.bellman_backup("a", 0, 1) == 5  # Not -47.5.


def test_frontier_initialization_is_explicit_and_does_not_invent_zero_actions():
    node = POMCPNode()
    with pytest.raises(ValueError, match="neither"):
        node.value_estimate()
    node.rollout_value = -7
    assert node.value_estimate() == -7
    node.action_values = {"evaluated": -5}
    assert node.value_estimate() == -5


def test_simulator_counts_same_observation_with_terminal_and_continuing_outcomes(monkeypatch):
    model = SimpleNamespace(
        is_terminal=lambda state: state == "dead",
        get_all_actions=lambda agent: ["a"],
        reward_bounds=lambda agent: (0.0, 1.0),
        sample_observation=lambda *args: "same",
        update_rollout_memory=lambda *args: None,
    )
    atom = SimpleNamespace(state="alive")
    bank = SimpleNamespace(
        expected_rewards=lambda history: (("a", 0.0),),
        stream_seed=lambda *a: 0,
        draw=lambda belief, rng: atom,
    )
    planner = IPOMCPPlanner(
        SolverKey("i", 1),
        model,
        bank,
        PlannerConfig(simulations=2, depth=3, modeled_simulations=2, gamma=1),
    )
    events = iter(
        [
            (SimpleNamespace(state="alive"), {}, 0, False),
            (SimpleNamespace(state="dead"), {}, 0, True),
        ]
    )
    monkeypatch.setattr(planner.gen_model, "tree_step", lambda *args: next(events))
    monkeypatch.setattr(planner, "_rollout", lambda *args: 10)
    history = SimpleNamespace(belief=None)
    monkeypatch.setattr(planner, "_history_successor", lambda *args: history)
    root = POMCPNode()
    assert planner._simulate(root, 0, None, history, random.Random(0)) == 10
    # Second sample terminates: Q = (10 + 0) / 2, terminal mass kept in the denominator.
    assert planner._simulate(root, 0, None, history, random.Random(0)) == 5
    assert root.continuation_counts == {"a": {"same": 1}}
    assert root.action_counts == {"a": 2}


def test_one_step_values_equal_analytic_expected_rewards():
    row = evaluate_case("mcts", 1, 100, 12, 0.07)[0]
    assert row["estimated_q"] == pytest.approx({"L": -1, "OL": 2.3, "OR": -92.3})


def test_two_step_uniform_belief_hand_calculation():
    row = evaluate_case("mcts", 2, 100, 12, 0.5)[0]
    assert row["estimated_q"] == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95})
    assert row["policy"] == {"L": 1.0}
