"""Final root scoring must preserve search and expose incomplete support."""

import random
from dataclasses import replace

import pytest

from core.config import IPOMCPConfig, MCTSConfig
from ipomdp.canonical import atom_row
from solvers.node import POMCPNode
from solvers.policy import greedy_policy
from tests.test_exact_final_step import make_planner


def configured(depth=3, enabled=True):
    planner = make_planner(p=0.8, depth=depth)
    planner.config = replace(
        planner.config,
        mcts=replace(planner.config.mcts, backup="empirical_bellman", exact_root_chance=enabled),
    )
    return planner


def test_nonuniform_terminal_mass_missing_support_and_ties(monkeypatch):
    planner = configured()
    monkeypatch.setattr(planner.solver_bank, "expected_rewards", lambda m: (("a", 2), ("b", 2)))
    monkeypatch.setattr(
        planner.solver_bank, "continuation_law", lambda m, a: (("x", 0.2), ("y", 0.3))
    )
    node = POMCPNode()
    node.action_values = {"a": -17, "b": -19}
    for action in node.action_values:
        node.create_child(action, "x").action_values = {"u": 5, "v": 1}
        node.create_child(action, "y").rollout_value = 10
    before = node.to_dict(max_depth=10)
    values, details = planner._root_recommendation(planner.model(), node)
    assert values == {"a": 5.8, "b": 5.8}  # surviving probability .5, not 1
    assert greedy_policy(values) == {"a": 0.5, "b": 0.5}
    assert details["a"]["continuation_probability"] == 0.5
    assert before == node.to_dict(max_depth=10)
    node.children["b"]["y"].rollout_value = None
    values, details = planner._root_recommendation(planner.model(), node)
    assert values["a"] == 5.8 and values["b"] == -19
    assert details["b"]["status"] == "empirical_fallback"
    assert details["b"]["missing_probability"] == 0.3
    del node.children["b"]["y"]
    assert planner._root_recommendation(planner.model(), node)[1] == details


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_real_search_identical_including_particles_and_rng(depth):
    control, candidate = configured(depth, False), configured(depth, True)
    before = random.getstate()
    control.policy_for(control.model())
    candidate.policy_for(candidate.model())
    assert random.getstate() == before

    def compare(a, b):
        assert a.to_dict(max_depth=depth + 1) == b.to_dict(max_depth=depth + 1)
        assert [atom_row(p, 1.0) for p in a.belief_particles] == [
            atom_row(p, 1.0) for p in b.belief_particles
        ]
        for action, children in a.children.items():
            for obs, child in children.items():
                compare(child, b.children[action][obs])

    compare(control.root, candidate.root)
    assert control.config.to_search_dict() == candidate.config.to_search_dict()
    assert "exact_root_chance" not in control.config.to_dict()["mcts"]
    assert candidate.config.to_dict()["mcts"]["exact_root_chance"] is True


@pytest.mark.parametrize(
    "depth,gamma,tail,status",
    [
        (1, 0.95, False, "no_continuation"),
        (3, 0, False, "no_continuation"),
        (2, 0.95, True, "exact_tail"),
    ],
)
def test_no_continuation_and_exact_boundary(monkeypatch, depth, gamma, tail, status):
    planner = configured(depth)
    planner.config = replace(
        planner.config,
        mcts=replace(
            planner.config.mcts,
            gamma=gamma,
            exact_final_step=tail,
            exact_tail_depth=2 if tail else 1,
        ),
    )

    def forbidden(*args):
        raise AssertionError("No continuation enumeration required")

    monkeypatch.setattr(planner.solver_bank, "continuation_law", forbidden)
    node = POMCPNode()
    node.action_values = {"L": 42}
    values, details = planner._root_recommendation(planner.model(), node)
    assert details["L"]["status"] == status
    assert values["L"] == (42 if tail else -1)


def test_errors_propagate_and_configuration_is_explicit(monkeypatch):
    with pytest.raises(ValueError):
        MCTSConfig(exact_root_chance=True)
    with pytest.raises(ValueError):
        MCTSConfig(exact_root_chance=1, backup="empirical_bellman")
    assert "exact_root_chance" not in IPOMCPConfig().to_dict()["mcts"]
    planner = configured()
    node = POMCPNode()
    node.action_values = {"L": 0}

    def fail(*args):
        raise RuntimeError("enumeration limit")

    monkeypatch.setattr(planner.solver_bank, "continuation_law", fail)
    with pytest.raises(RuntimeError, match="enumeration limit"):
        planner._root_recommendation(planner.model(), node)


def test_small_l2_paired_evaluator_and_protocol():
    from examples.experiments.planner_oracle_experiment import evaluate_case
    from examples.experiments.root_chance_experiment import paired

    row = evaluate_case(
        "mcts",
        3,
        30,
        4,
        0.2,
        0.95,
        level=2,
        opponent_depth=3,
        opponent_budget=25,
        opponent_belief=0.085,
        exact_final_step=True,
        exact_tail_depth=2,
        exact_history_rewards=True,
        backup="empirical_bellman",
    )[0]
    result = paired(row)[0]
    assert result["identical_search_tree"] and result["exact_control_replay"]
    candidate = result["candidate"]
    assert candidate["estimated_q"] != {}
    assert candidate["recommendation_rule"] == "exact_root_chance_complete_support"
    assert candidate["search_estimated_q"] == row["estimated_q"]
    assert candidate["policy"] == greedy_policy(candidate["estimated_q"])
