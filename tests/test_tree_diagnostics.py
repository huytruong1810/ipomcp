"""Algebraic tests for post-search diagnostics, independent of Tiger search."""

from examples.experiments.tree_diagnostics import describe_node
from solvers.node import POMCPNode


class Reference:
    gamma = 0.5

    def q_values(self, model, horizon):
        if model == "root":
            return {"a": 1.0}
        return {"good": 2.0, "bad": 0.0}

    def branches(self, model, action):
        assert model == "root"
        return 0.0, [("seen", 0.5, "left"), ("unseen", 0.5, "right")]


def test_decomposition_keeps_unseen_probability_and_child_regret():
    root = POMCPNode()
    root.action_counts = {"a": 2}
    root.continuation_counts = {"a": {"seen": 2}}
    root.immediate_reward_means = {"a": 0.0}
    root.action_values = {"a": 1.5}
    child = root.create_child("a", "seen")
    child.action_values = {"good": 1.0, "bad": 3.0}
    result = describe_node(root, "root", 2, Reference(), str)
    action = result["actions"]["a"]
    assert action["reconstruction_residual"] == 0
    assert action["continuation_error"] == 0.5
    seen, unseen = action["branches"]
    assert seen["child"]["local_action_loss"] == 2.0
    assert seen["child"]["greedy_estimation_error"] == 3.0
    assert seen["child"]["value_error"] == 1.0
    assert unseen["child"]["kind"] == "unexpanded"
    assert unseen["value"] is None


def test_rollout_frontier_is_not_a_selected_action():
    node = POMCPNode()
    node.rollout_value = -4.0
    result = describe_node(node, "left", 1, Reference(), str)
    assert result["kind"] == "rollout_frontier"
    assert result["value_error"] == -6.0
    assert "local_action_loss" not in result


def test_exact_ties_use_uniform_action_loss():
    node = POMCPNode()
    node.action_values = {"good": 3.0, "bad": 3.0}
    result = describe_node(node, "left", 1, Reference(), str)
    assert result["local_action_loss"] == 1.0
    assert result["greedy_estimation_error"] == 2.0
    assert result["value_error"] == 1.0


class DeepReference:
    gamma = 0.5

    def q_values(self, model, horizon):
        return {"a": 4.0, "b": 2.0} if model == "root" else {"good": 2.0, "bad": 0.0}

    def branches(self, model, action):
        if model == "root":
            return (3.0 if action == "a" else 1.0), [("x", 1.0, "child")]
        return (-1.0 if action == "good" else -3.0), [("end", 1.0, "leaf")]


def deep_tree():
    root = POMCPNode()
    root.action_counts = {"a": 1, "b": 1}
    root.immediate_reward_means = {"a": 3.0, "b": 1.0}
    root.continuation_counts = {"a": {"x": 1}, "b": {"x": 1}}
    left = root.create_child("a", "x")
    left.rollout_value = -4.0
    right = root.create_child("b", "x")
    right.action_values = {"good": 1.0, "bad": 3.0}
    # Exact-boundary-style synthetic node makes local loss inseparable from its
    # residual boundary error unless expanded; test it separately below.
    root.action_values = {"a": 1.0, "b": 2.5}
    return root


def test_trace_reconstructs_frontier_boundary_and_unresolved_mass():
    from examples.experiments.tree_diagnostics import trace_margin

    root = deep_tree()
    partial = trace_margin(root, "root", 3, DeepReference(), str, {"b": 1.0}, max_nodes=1)
    assert partial["target_margin_error"] == 3.5
    assert partial["contributions"]["rollout_frontier"] == 3.0
    assert partial["unresolved_signed"] == 0.5
    assert partial["unresolved_absolute"] == 0.5
    full = trace_margin(root, "root", 3, DeepReference(), str, {"b": 1.0})
    assert full["contributions"]["exact_tail"] == 0.5
    assert full["unresolved_absolute"] == 0
    assert full["reconstruction_residual"] == 0


def test_trace_subtracts_local_loss_and_handles_uniform_greedy_ties():
    from examples.experiments.tree_diagnostics import trace_margin

    class ReferenceWithLeaf(DeepReference):
        def q_values(self, model, horizon):
            return {"v": 6.0} if model == "leaf" else super().q_values(model, horizon)

    root = deep_tree()
    child = root.get_child("b", "x")
    child.action_values = {"good": 3.0, "bad": 3.0}
    child.action_counts = {"good": 1, "bad": 1}
    child.immediate_reward_means = {"good": -1.0, "bad": -3.0}
    child.continuation_counts = {"good": {"end": 1}, "bad": {"end": 1}}
    child.create_child("good", "end").rollout_value = 8.0
    child.create_child("bad", "end").rollout_value = 12.0
    result = trace_margin(root, "root", 3, ReferenceWithLeaf(), str, {"b": 1.0})
    assert result["contributions"]["local_action_loss"] == -0.5
    assert result["contributions"]["rollout_frontier"] == 4.0
    assert result["reconstruction_residual"] == 0


def test_trace_keeps_zero_count_branches_in_chance_error():
    from examples.experiments.tree_diagnostics import trace_margin

    class ChanceReference:
        gamma = 1.0

        def q_values(self, model, horizon):
            if model == "root":
                return {"a": 1.0, "b": 0.0}
            return {"v": 4.0 if model == "hi" else 0.0}

        def branches(self, model, action):
            return 0.0, (
                [("hi", 0.25, "hi"), ("lo", 0.75, "lo")] if action == "a" else [("lo", 1.0, "lo")]
            )

    root = POMCPNode()
    root.action_values = {"a": 4.0, "b": 0.0}
    root.action_counts = {"a": 1, "b": 1}
    root.immediate_reward_means = {"a": 0.0, "b": 0.0}
    root.continuation_counts = {"a": {"hi": 1}, "b": {"lo": 1}}
    root.create_child("a", "hi").action_values = {"v": 4.0}
    root.create_child("b", "lo").action_values = {"v": 0.0}
    result = trace_margin(root, "root", 2, ChanceReference(), str, {"b": 1.0})
    assert result["target_margin_error"] == -3.0
    assert result["contributions"]["chance"] == -3.0
    assert result["reconstruction_residual"] == 0


def test_diagnostic_control_equality_and_explicit_candidate_contract():
    from examples.experiments.planner_oracle_experiment import evaluate_case
    from examples.experiments.tree_diagnostics import diagnose

    original = evaluate_case(
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
        exact_history_rewards=True,
        backup="empirical_bellman",
    )[0]
    control = diagnose(original, trace_nodes=16)[0]
    assert control["exact_replay"] is True
    assert control["row"] == original
    candidate = diagnose(original, trace_nodes=16, tail_depth=2)[0]
    assert candidate["exact_replay"] is False
    assert candidate["row"]["opponent_model"] == original["opponent_model"]
    assert candidate["row"]["oracle_q"] == original["oracle_q"]
    assert candidate["trace_policy"] == original["policy"]
    assert abs(candidate["margin_trace"]["reconstruction_residual"]) < 1e-8
