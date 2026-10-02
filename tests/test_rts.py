"""The IPF + RTS baseline: exact one-step values, convergence in K, hierarchies, purity."""

import pytest

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank


def test_one_decision_values_are_the_exact_expected_rewards():
    row = evaluate_case("rts", 1, 5, 0, 0.07)[0]
    assert row["estimated_q"] == pytest.approx(row["oracle_q"])


def test_two_step_values_converge_to_exact_value_iteration_as_samples_grow():
    errors = [
        sum(evaluate_case("rts", 2, k, seed, 0.3)[0]["max_abs_q_error"] for seed in range(3)) / 3
        for k in (5, 400)
    ]
    assert errors[1] < errors[0]
    assert errors[1] < 0.5


@pytest.mark.parametrize("modeled", ["rts", "mcts"])
def test_rts_hierarchies_build_and_act(modeled):
    rts = RTSConfig(depth=2, observation_samples=3)
    modeled_config = (
        rts if modeled == "rts" else PlannerConfig(simulations=20, depth=2, modeled_simulations=20)
    )
    planner = HierarchyBuilder(SolverBank(1, particles=32)).build(
        "i", 2, TigerModel(), "j", config=rts, modeled_config=modeled_config
    )
    assert isinstance(planner, RTSPlanner)
    assert planner.get_action() in planner.actions
    planner.update_root("L", ("GL", "S"))
    assert planner.get_action() in planner.actions


def test_modeled_rts_policy_is_a_pure_function_of_belief():
    bank = SolverBank(2)
    planner = HierarchyBuilder(bank).build(
        "i", 1, TigerModel(), "j", config=RTSConfig(depth=2, observation_samples=4)
    )
    first = planner.policy_for(planner.model(), modeled=True)
    bank.clear_caches()
    assert planner.policy_for(planner.model(), modeled=True) == first
