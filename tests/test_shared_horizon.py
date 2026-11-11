"""Shared-horizon reference (audit spec 3.8): the modeled agent plays its own r-step law at r steps to go."""

import math

import pytest

from ipomdp.domains.tiger import JointTigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.wumpus_qualification import nested_levels
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import build_exact_agent
from ipomdp.reference.shared_horizon import SharedHorizonReference

GAMMA = 0.95


def agent(physics, level, horizon, nested=None):
    return build_exact_agent("i", "j", level, physics, nested or nested_levels(level), horizon, GAMMA)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_a_level_zero_opponent_gives_the_fixed_reference(horizon):
    physics = JointTigerModel(together_safe=40.0)
    exact = agent(physics, 1, horizon)
    model = exact.model()
    fixed = BestResponseReference(physics, exact.solver_bank, GAMMA).q_values(model, horizon)
    shared = SharedHorizonReference(physics, exact.solver_bank, GAMMA).q_values(model, horizon)
    for action, value in fixed.items():
        assert math.isclose(shared[action], value, rel_tol=0, abs_tol=1e-12)


@pytest.mark.parametrize("physics", [JointTigerModel(together_safe=40.0), UAVModel(rows=2, cols=2)])
def test_at_horizon_one_both_conventions_agree(physics):
    exact = agent(physics, 2, 1)
    model = exact.model()
    fixed = BestResponseReference(physics, exact.solver_bank, GAMMA).q_values(model, 1)
    shared = SharedHorizonReference(physics, exact.solver_bank, GAMMA).q_values(model, 1)
    for action, value in fixed.items():
        assert math.isclose(shared[action], value, rel_tol=0, abs_tol=1e-12)


def test_the_root_step_uses_the_full_horizon_law_and_the_conventions_can_differ():
    # At the root both conventions play j's H-step law; below it the shared one plays shorter laws.
    physics = JointTigerModel(together_safe=40.0)
    differ = 0
    for horizon in (2, 3):
        exact = agent(physics, 2, horizon)
        model = exact.model()
        fixed = BestResponseReference(physics, exact.solver_bank, GAMMA)
        shared = SharedHorizonReference(physics, exact.solver_bank, GAMMA)
        for atom, _ in model.belief.mass:
            if atom.opponent.frame.level == 1:
                law = dict(shared._law(atom.opponent, horizon))
                assert law == pytest.approx(exact.solver_bank.policy(atom.opponent))
        a, b = fixed.q_values(model, horizon), shared.q_values(model, horizon)
        differ += any(abs(a[k] - b[k]) > 1e-9 for k in a)
    assert differ >= 1


def test_deeper_opponents_are_refused():
    physics = JointTigerModel(together_safe=40.0)
    exact = agent(physics, 3, 2)
    with pytest.raises(NotImplementedError):
        SharedHorizonReference(physics, exact.solver_bank, GAMMA).q_values(exact.model(), 2)


def test_the_reference_stores_q_shared_and_decide_reports_loss_shared():
    from ipomdp.experiments import reachable_decisions

    problem = reachable_decisions.reference("tiger-joint", 2, 2, 0, 1, opponent_horizon="shared")
    assert set(problem["q_shared"]) == set(problem["q"])
    row = reachable_decisions.decide(problem, "mcts", 200, 0, modeled=50)
    assert "loss_shared" in row and row["loss_shared"] >= 0
