"""The unified exact reference on a terminating domain (UAV pursuit)."""

import math

import pytest

from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference


def pursuer(horizon, simulations=200):
    physics = UAVModel(rows=2, cols=3)
    bank = SolverBank(5)
    config = PlannerConfig(simulations=simulations, depth=horizon, modeled_simulations=simulations)
    planner = HierarchyBuilder(bank).build("i", 1, physics, "j", config=config)
    return physics, bank, planner


def test_branches_conserve_probability_with_terminal_mass():
    physics, bank, planner = pursuer(2)
    reference = BestResponseReference(physics, bank, 0.95)
    for action in physics.get_all_actions("i"):
        _, branches = reference.branches(planner.model(), action)
        continuing = math.fsum(p for _, p, _ in branches)
        assert 0 < continuing <= 1
        for _, _, successor in branches:
            assert not any(physics.is_terminal(atom.state) for atom, _ in successor.belief.mass)


def test_exact_values_respect_reward_bounds_and_match_planner_leaf():
    physics, bank, planner = pursuer(1)
    reference = BestResponseReference(physics, bank, 0.95)
    exact = reference.q_values(planner.model(), 1)
    low, high = physics.reward_bounds("i")
    assert all(low <= q <= high for q in exact.values())
    planner.policy_for(planner.model())
    assert planner.recommendation_values == pytest.approx(exact)
