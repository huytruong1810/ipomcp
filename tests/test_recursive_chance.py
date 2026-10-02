"""Recursive integration, partial coverage, and unchanged search contracts."""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.config import IPOMCPConfig, MCTSConfig
from solvers.node import POMCPNode
from solvers.policy import greedy_policy
from solvers.recursive_chance import recursive_recommendation
from tests.test_root_chance import configured


def toy():
    root = POMCPNode()
    root.action_values = {"A": 0.0, "B": 2.0}
    child = root.create_child("A", "o")
    child.action_values = {"X": 0.0}
    child.create_child("X", "high").action_values = {"Z": 10.0}
    child.create_child("X", "low").action_values = {"Z": 0.0}
    rewards = {"r": {"A": 0.0, "B": 2.0}, "c": {"X": 0.0}, "high": {"Z": 10.0}, "low": {"Z": 0.0}}
    laws = {("r", "A"): (("o", 1.0),), ("r", "B"): (), ("c", "X"): (("high", 0.9), ("low", 0.1))}
    planner = SimpleNamespace(
        config=IPOMCPConfig(
            mcts=MCTSConfig(
                gamma=1.0,
                max_depth=3,
                exact_final_step=True,
                backup="empirical_bellman",
                recursive_chance=True,
            )
        ),
        _final_action_values=lambda m: rewards[m],
        _history_successor=lambda m, a, o: "c" if m == "r" else o,
        solver_bank=SimpleNamespace(continuation_law=lambda m, a: laws[m, a]),
    )
    return planner, root, rewards, laws


def test_recursive_corrects_descendant_error_without_changing_tree():
    planner, root, _, _ = toy()
    before = root.to_dict(max_depth=10)
    scores, detail = recursive_recommendation(planner, "r", root)
    assert scores == {"A": 9.0, "B": 2.0}
    assert greedy_policy(scores) == {"A": 1.0}
    assert greedy_policy(root.action_values) == {"B": 1.0}
    assert detail["root_subtree_exact"] and detail["visited_histories"] == 4
    assert root.to_dict(max_depth=10) == before


def test_missing_support_is_whole_action_fallback_and_propagates():
    planner, root, _, _ = toy()
    del root.children["A"]["o"].children["X"]["low"]
    scores, d = recursive_recommendation(planner, "r", root)
    assert scores["A"] == 0 and not d["root_subtree_exact"]
    child = d["nodes"][1]["actions"]["X"]
    assert child["status"] == "empirical_fallback" and child["missing_probability"] == 0.1
    assert d["nodes"][0]["actions"]["A"]["approximate_child_probability"] == 1.0
    assert d["action_status_counts"]["empirical_fallback"] == 1


def test_frontier_and_unevaluated_actions_are_not_exact():
    planner, root, rewards, _ = toy()
    child = root.children["A"]["o"]
    rewards["c"]["Y"] = 50.0
    scores, d = recursive_recommendation(planner, "r", root)
    assert scores["A"] == 9 and not d["root_subtree_exact"]
    assert d["nodes"][1]["unevaluated_actions"] == ["Y"]
    child.action_values.clear()
    child.rollout_value = 4.0
    scores, d = recursive_recommendation(planner, "r", root)
    assert scores["A"] == 4.0 and d["rollout_frontiers"] == 1 and not d["root_subtree_exact"]


def test_terminal_mass_and_uniform_ties():
    planner, root, _, laws = toy()
    laws["c", "X"] = (("high", 0.2),)
    scores, d = recursive_recommendation(planner, "r", root)
    assert scores == {"A": 2.0, "B": 2.0} and greedy_policy(scores) == {"A": 0.5, "B": 0.5}
    assert d["root_subtree_exact"]


def test_errors_propagate_and_no_cross_solve_cache():
    planner, root, _, laws = toy()
    recursive_recommendation(planner, "r", root)
    laws["c", "X"] = (("low", 1.0),)
    assert recursive_recommendation(planner, "r", root)[0]["A"] == 0

    def fail(*args):
        raise RuntimeError("inference limit")

    planner._history_successor = fail
    with pytest.raises(RuntimeError, match="inference limit"):
        recursive_recommendation(planner, "r", root)


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_full_search_digest_rng_and_default_metadata_unchanged(depth):
    from examples.experiments.root_chance_experiment import tree_digest

    control = configured(depth, False)
    candidate = configured(depth, False)
    candidate.config = replace(
        candidate.config, mcts=replace(candidate.config.mcts, recursive_chance=True)
    )
    before = random.getstate()
    control.policy_for(control.model())
    candidate.policy_for(candidate.model())
    assert random.getstate() == before and tree_digest(control) == tree_digest(candidate)
    assert candidate.config.to_search_dict() == control.config.to_search_dict()
    assert "recursive_chance" not in control.config.to_dict()["mcts"]
    assert candidate.config.to_dict()["mcts"]["recursive_chance"]
    if depth == 1:
        assert candidate.recommendation_values == control.recommendation_values
        assert candidate.root_chance_details["root_subtree_exact"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recursive_chance": 1},
        {"recursive_chance": True},
        {"recursive_chance": True, "exact_root_chance": True, "backup": "empirical_bellman"},
    ],
)
def test_invalid_modes(kwargs):
    with pytest.raises(ValueError):
        MCTSConfig(**kwargs)


def test_small_l2_evaluator_keeps_opponent_and_reference():
    from examples.experiments.planner_oracle_experiment import evaluate_case

    args = dict(
        level=2,
        opponent_depth=3,
        opponent_budget=25,
        opponent_belief=0.085,
        exact_final_step=True,
        exact_tail_depth=2,
        exact_history_rewards=True,
        backup="empirical_bellman",
    )
    control = evaluate_case("mcts", 3, 30, 4, 0.2, 0.95, **args)[0]
    candidate = evaluate_case("mcts", 3, 30, 4, 0.2, 0.95, recursive_chance=True, **args)[0]
    assert candidate["search_estimated_q"] == control["estimated_q"]
    assert candidate["oracle_q"] == control["oracle_q"]
    assert candidate["opponent_model"] == control["opponent_model"]
    assert candidate["policy"] == greedy_policy(candidate["estimated_q"])
    assert "recursive_chance" in candidate and "root_chance" not in candidate


def test_three_arm_small_l2_runner():
    from examples.experiments.planner_oracle_experiment import evaluate_case
    from examples.experiments.recursive_chance_experiment import triple
    from examples.experiments.root_chance_experiment import replay

    control = evaluate_case(
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
    root_only = replay(control, True)[0]
    result = triple(control, root_only)[0]
    assert result["exact_control_replay"] and result["exact_root_only_replay"]
    assert result["candidate_tree_sha256"] == result["control_tree_sha256"]


@pytest.mark.parametrize("gamma,depth", [(0.0, 3), (1.0, 1)])
def test_no_continuation_does_not_enumerate(gamma, depth):
    planner, root, _, _ = toy()
    planner.config = replace(
        planner.config,
        mcts=replace(planner.config.mcts, gamma=gamma, max_depth=depth, exact_final_step=False),
    )

    def forbidden(*args):
        raise AssertionError("No continuation required")

    planner.solver_bank.continuation_law = forbidden
    scores, details = recursive_recommendation(planner, "r", root)
    assert scores == {"A": 0.0, "B": 2.0} and details["root_subtree_exact"]
