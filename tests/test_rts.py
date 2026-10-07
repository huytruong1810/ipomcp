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
        "i", 2, TigerModel(), "j", config=rts, modeled_configs={1: modeled_config}
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


L2 = dict(level=2, opponent_depth=2, opponent_belief=0.2)


@pytest.mark.parametrize("planner", ["rts", "mcts"])
def test_particle_planners_are_scored_against_the_exact_ipomdp(planner):
    """With the IPF the reference is still solved on exact beliefs, so it is unchanged."""
    budget = 3 if planner == "rts" else 60
    exact = evaluate_case(planner, 2, budget, 4, 0.3, **L2)[0]
    sampled = evaluate_case(planner, 2, budget, 4, 0.3, particles=16, **L2)[0]
    assert sampled["oracle_q"] == exact["oracle_q"]
    assert sampled["particles"] == 16 and exact["particles"] is None


def test_finite_opponents_are_solved_by_the_planners_own_kind():
    row = evaluate_case("rts", 2, 2, 4, 0.3, modeled="2", reference="exact", **L2)[0]
    assert row["opponent"]["policy"] == "same_planner"
    assert row["planner_config"] == {"depth": 2, "observation_samples": 2, "gamma": 0.95}


def test_declared_reference_is_rejected_with_particle_beliefs():
    with pytest.raises(ValueError, match="not exact with particle beliefs"):
        evaluate_case("rts", 2, 2, 4, 0.3, modeled="2", particles=16, **L2)


def test_mid_episode_reference_is_the_exact_posterior_whatever_the_planners_beliefs():
    exact = evaluate_case("mcts", 2, 60, 4, 0.3, history_steps=3, **L2)[0]
    sampled = evaluate_case("rts", 2, 3, 4, 0.3, history_steps=3, particles=16, **L2)[0]
    at_prior = evaluate_case("mcts", 2, 60, 4, 0.3, **L2)[0]
    assert sampled["oracle_q"] == exact["oracle_q"] != at_prior["oracle_q"]
    assert exact["history_steps"] == 3 and exact["belief_support"] > at_prior["belief_support"]
    with pytest.raises(ValueError, match="solved at the prior"):
        evaluate_case("mcts", 1, 60, 4, 0.3, history_steps=1)
