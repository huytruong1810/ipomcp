import pytest

from solvers.exploration import HorizonBoundUCB, NormalizedUCB, StandardUCB
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


@pytest.mark.parametrize("gamma,expected", [(0, 110), (0.5, 192.5), (1, 330)])
def test_horizon_reward_width(gamma, expected):
    assert HorizonBoundUCB(-100, 10).return_width(3, gamma) == pytest.approx(expected)


def test_early_termination_is_included_in_reward_interval():
    assert HorizonBoundUCB(2, 5).return_width(2, 1) == 10
    assert HorizonBoundUCB(-5, -2).return_width(2, 1) == 10


def test_horizon_bound_changes_exploration_without_sample_extrema():
    node = POMCPNode()
    node.visit_count = 100
    node.action_counts = {"good": 90, "uncertain": 10}
    node.action_values = {"good": 10, "uncertain": 0}
    strategy = HorizonBoundUCB(-10, 10)
    assert (
        strategy.select_action(node, ["good", "uncertain"], remaining_horizon=1, gamma=1) == "good"
    )
    assert (
        strategy.select_action(node, ["good", "uncertain"], remaining_horizon=3, gamma=1)
        == "uncertain"
    )
    assert (
        strategy.select_action(
            node,
            ["good", "uncertain"],
            remaining_horizon=1,
            gamma=1,
            q_min=-1e20,
            q_max=1e20,
        )
        == "good"
    )
    assert strategy.select_action(node, ["good", "new"], remaining_horizon=1, gamma=1) == "new"


@pytest.mark.parametrize("lo,hi,c", [(1, -1, 1), (0, float("inf"), 1), (-1, 1, 0)])
def test_invalid_reward_bound_contract(lo, hi, c):
    with pytest.raises(ValueError):
        HorizonBoundUCB(lo, hi, c)


@pytest.mark.parametrize("horizon,gamma", [(0, 0.95), (2.5, 1), (1, float("nan")), (1, 1.1)])
def test_invalid_bound_horizon_or_discount(horizon, gamma):
    with pytest.raises(ValueError):
        HorizonBoundUCB(-1, 1).return_width(horizon, gamma)
