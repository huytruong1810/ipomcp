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
