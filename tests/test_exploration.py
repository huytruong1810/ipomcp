from solvers.exploration import NormalizedUCB, StandardUCB
from solvers.node import POMCPNode


def test_standard_ucb_untried_actions():
    strategy = StandardUCB(exploration_const=1.0)
    node = POMCPNode()
    actions = ["A1", "A2", "A3"]

    # Untried actions should be prioritized first
    chosen = strategy.select_action(node, actions)
    assert chosen in actions


def test_normalized_ucb_scale_invariance():
    strategy = NormalizedUCB(exploration_const=1.414)
    node = POMCPNode()
    node.visit_count = 100

    # Action 1 visited 90 times with value 1000
    node.action_counts["A1"] = 90
    node.action_values["A1"] = 1000.0

    # Action 2 visited 10 times with value 900
    node.action_counts["A2"] = 10
    node.action_values["A2"] = 900.0

    # Normalized UCB uses local bounds
    chosen = strategy.select_action(node, ["A1", "A2"], q_min=900.0, q_max=1000.0)
    assert chosen in ["A1", "A2"]


def test_normalized_ucb_degenerate_and_uninitialized_bounds():
    strategy = NormalizedUCB(exploration_const=1.414)
    node = POMCPNode()
    node.visit_count = 10
    node.action_counts["A1"] = 5
    node.action_values["A1"] = 10.0
    node.action_counts["A2"] = 5
    node.action_values["A2"] = 10.0

    # 1. Equal bounds (q_min == q_max)
    chosen_equal = strategy.select_action(node, ["A1", "A2"], q_min=10.0, q_max=10.0)
    assert chosen_equal in ["A1", "A2"]

    # 2. Inverted / uninitialized bounds (q_min > q_max, e.g. inf / -inf)
    chosen_inv = strategy.select_action(node, ["A1", "A2"], q_min=float("inf"), q_max=float("-inf"))
    assert chosen_inv in ["A1", "A2"]
