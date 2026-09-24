"""Independent counterexamples for history/chance Bellman backup semantics."""

from types import SimpleNamespace

import pytest

from core.config import IPOMCPConfig, MCTSConfig
from examples.experiments.planner_oracle_experiment import evaluate_case
from solvers.i_pomcp import IPOMCPPlanner
from solvers.node import POMCPNode
from solvers.solver_types import SolverKey


def test_chance_weighting_is_not_maximum_or_uniform_average_over_outcomes():
    root = POMCPNode()
    root.action_counts["a"] = 10
    root.immediate_reward_means["a"] = 2
    root.continuation_counts["a"] = {"frequent": 9, "rare": 1}
    root.create_child("a", "frequent").action_values = {"good": 10, "bad": -20}
    root.create_child("a", "rare").action_values = {"good": -100, "bad": -200}
    # Chance expectation = .9 * 10 + .1 * -100 = -1, then discount.
    assert root.empirical_bellman_backup("a", 2, 0.5) == pytest.approx(1.5)


def test_terminal_probability_is_not_renormalized_away():
    root = POMCPNode()
    root.action_counts["a"] = 10  # Eight terminal events and two continuing.
    root.continuation_counts["a"] = {"alive": 2}
    root.create_child("a", "alive").action_values = {"best": 10}
    assert root.empirical_bellman_backup("a", 0, 1) == 2


def test_backup_rereads_improved_child_values_instead_of_averaging_old_returns():
    root = POMCPNode()
    child = root.create_child("a", "obs")
    child.rollout_value = -100
    root.action_counts["a"] = 1
    root.continuation_counts["a"] = {"obs": 1}
    assert root.empirical_bellman_backup("a", 0, 1) == -100
    child.action_values = {"best": 5, "other": -10}
    root.action_counts["a"] = 2
    root.continuation_counts["a"]["obs"] = 2
    assert root.empirical_bellman_backup("a", 0, 1) == 5  # Not -47.5.


def test_frontier_initialization_is_explicit_and_does_not_invent_zero_actions():
    node = POMCPNode()
    with pytest.raises(ValueError, match="neither"):
        node.value_estimate()
    node.rollout_value = -7
    assert node.value_estimate() == -7
    node.action_values = {"evaluated": -5}
    assert node.value_estimate() == -5
    node.action_values = {"evaluated": 2}
    assert node.value_estimate() == 2


def test_simulator_counts_same_observation_with_terminal_and_continuing_outcomes(monkeypatch):
    model = SimpleNamespace(
        is_terminal=lambda state: state == "dead",
        get_legal_actions=lambda *args: ["a"],
        sample_observation=lambda *args: "same",
        update_rollout_belief=lambda *args: None,
    )
    planner = IPOMCPPlanner(
        SolverKey("i", 1),
        model,
        ["a"],
        None,
        config=IPOMCPConfig(mcts=MCTSConfig(max_depth=2, gamma=1, backup="empirical_bellman")),
    )
    events = iter(
        [
            (SimpleNamespace(state="alive"), {}, 0, False),
            (SimpleNamespace(state="dead"), {}, 0, True),
        ]
    )
    monkeypatch.setattr(planner.gen_model, "tree_step", lambda *args: next(events))
    monkeypatch.setattr(planner, "_rollout", lambda *args, **kwargs: 10)
    root = POMCPNode()
    bounds = {"q_min": float("inf"), "q_max": -float("inf"), "root_rewards": {"a": 0}}
    atom = SimpleNamespace(state="alive")
    assert planner._simulate(atom, root, 0, bounds) == 10
    assert planner._simulate(atom, root, 0, bounds) == 5
    assert root.continuation_counts == {"a": {"same": 1}}
    assert root.action_counts == {"a": 2}


@pytest.mark.parametrize("exact", [False, True])
def test_one_step_bellman_matches_analytic_rewards(exact):
    row = evaluate_case(
        "mcts", 1, 100, 12, 0.07, 0.95, exact_final_step=exact, backup="empirical_bellman"
    )[0]
    assert row["estimated_q"] == pytest.approx({"L": -1, "OL": 2.3, "OR": -92.3})
    assert row["backup"] == "empirical_bellman"


def test_bellman_exact_tail_uniform_two_step_hand_calculation():
    row = evaluate_case(
        "mcts", 2, 100, 12, 0.5, 0.95, exact_final_step=True, backup="empirical_bellman"
    )[0]
    assert row["estimated_q"] == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95})
    assert row["policy"] == {"L": 1.0}


def test_invalid_backup_is_rejected():
    with pytest.raises(ValueError, match="backup"):
        MCTSConfig(backup="max_hidden_state")
    with pytest.raises(ValueError, match="MCTS-only"):
        evaluate_case("rts", 1, 3, 0, 0.5, 0.95, backup="empirical_bellman")
