"""Horizon-bounded UCB: scale from declared reward bounds, untried first, unbiased ties."""

import random

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.planning.exploration import HorizonBoundUCB
from ipomdp.planning.node import POMCPNode


def node_with(counts, values):
    node = POMCPNode()
    node.visit_count = sum(counts.values())
    node.action_counts = dict(counts)
    node.action_values = dict(values)
    return node


@pytest.mark.parametrize("gamma,expected", [(0, 110), (0.5, 192.5), (1, 330)])
def test_horizon_reward_width(gamma, expected):
    assert HorizonBoundUCB(-100, 10, 1).return_width(3, gamma) == pytest.approx(expected)


def test_early_termination_is_included_in_reward_interval():
    assert HorizonBoundUCB(2, 5, 1).return_width(2, 1) == 10
    assert HorizonBoundUCB(-5, -2, 1).return_width(2, 1) == 10


def test_bounds_come_from_the_domain_contract():
    strategy = HorizonBoundUCB.for_agent(TigerModel(), "i", 2.0)
    assert (strategy.reward_min, strategy.reward_max, strategy.exploration) == (-100, 10, 2.0)


def test_exploration_grows_with_remaining_horizon():
    node = node_with({"good": 90, "uncertain": 10}, {"good": 10, "uncertain": 0})
    strategy = HorizonBoundUCB(-10, 10, 1)
    assert strategy.select_action(node, ["good", "uncertain"], 1, 1) == "good"
    assert strategy.select_action(node, ["good", "uncertain"], 3, 1) == "uncertain"


def test_untried_actions_are_selected_first():
    node = node_with({"good": 90}, {"good": 10})
    assert HorizonBoundUCB(-10, 10, 1).select_action(node, ["good", "new"], 1, 1) == "new"


def test_exact_ties_are_broken_uniformly_not_by_action_order():
    node = node_with({"a": 10, "b": 10}, {"a": 1.0, "b": 1.0})
    strategy = HorizonBoundUCB(-1, 1, 1)
    random.seed(0)
    chosen = {strategy.select_action(node, ["a", "b"], 2, 0.9) for _ in range(50)}
    assert chosen == {"a", "b"}


@pytest.mark.parametrize("lo,hi,c", [(1, -1, 1), (0, float("inf"), 1), (-1, 1, 0)])
def test_invalid_reward_bound_contract(lo, hi, c):
    with pytest.raises(ValueError):
        HorizonBoundUCB(lo, hi, c)


@pytest.mark.parametrize("horizon,gamma", [(0, 0.95), (2.5, 1), (1, float("nan")), (1, 1.1)])
def test_invalid_bound_horizon_or_discount(horizon, gamma):
    with pytest.raises(ValueError):
        HorizonBoundUCB(-1, 1, 1).return_width(horizon, gamma)
