"""Post-search chance scoring: exact where the tree is complete, search Q elsewhere."""

import copy
import math

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.planning.chance_scoring import score_tree
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference


def make_planner(depth, simulations, p=0.5, gamma=0.95):
    physics = TigerModel()
    bank = SolverBank(seed=17)
    planner = HierarchyBuilder(bank).build(
        "i",
        1,
        physics,
        "j",
        config=PlannerConfig(
            simulations=simulations, depth=depth, modeled_simulations=10, gamma=gamma
        ),
        level_weights={0: 1.0},
    )
    opponent = MentalModel(AgentFrame("j", 0, physics))
    planner.set_initial_belief(
        FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, opponent), p),
                (InteractiveState(TIGER_RIGHT, opponent), 1 - p),
            )
        )
    )
    return planner


def rescore(planner):
    return score_tree(
        planner.root,
        planner.model(),
        planner.config.depth,
        planner.solver_bank,
        planner._history_successor,
        planner.config.gamma,
    )


def test_continuation_law_matches_the_reference_branches():
    planner = make_planner(depth=2, simulations=10, p=0.7)
    bank, model = planner.solver_bank, planner.model()
    reference = BestResponseReference(planner.pomdp_model, bank, planner.config.gamma)
    for action in planner.actions:
        _, branches = reference.branches(model, action)
        expected = {o: p for o, p, _ in branches if p > 0}
        assert dict(bank.continuation_law(model, action)) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("p", [0.08, 0.5, 0.85])
def test_a_complete_tree_scores_to_the_exact_bellman_values(p):
    planner = make_planner(depth=2, simulations=3000, p=p)
    planner.policy_for(planner.model())
    scores = planner.scores
    assert scores.exact and scores.fallback_actions == 0 and scores.frontiers == 0
    reference = BestResponseReference(planner.pomdp_model, planner.solver_bank, 0.95)
    exact = reference.q_values(planner.model(), 2)
    for action, value in exact.items():
        assert math.isclose(scores.values[action], value, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize("p", [0.08, 0.5, 0.85])
def test_two_steps_left_are_exact_with_branches_never_sampled(p):
    # One simulation per action samples one branch of six; the rest are still
    # exact, because each child is the last decision.
    planner = make_planner(depth=2, simulations=3, p=p)
    planner.policy_for(planner.model())
    scores = planner.scores
    assert scores.exact and scores.fallback_actions == 0
    reference = BestResponseReference(planner.pomdp_model, planner.solver_bank, 0.95)
    exact = reference.q_values(planner.model(), 2)
    for action, value in exact.items():
        assert math.isclose(scores.values[action], value, rel_tol=0, abs_tol=1e-9)


def test_missing_branches_keep_the_whole_search_value():
    planner = make_planner(depth=3, simulations=12)
    planner.policy_for(planner.model())
    scores, root, model = planner.scores, planner.root, planner.model()
    assert not scores.exact and scores.fallback_actions > 0
    for action, search_q in root.action_values.items():
        law = planner.solver_bank.continuation_law(model, action)
        if any(root.get_child(action, o) is None for o, _ in law):
            assert scores.values[action] == search_q


def test_scoring_reads_the_tree_without_changing_it():
    planner = make_planner(depth=3, simulations=200)
    planner.policy_for(planner.model())
    before = copy.deepcopy(planner.root.to_dict(max_depth=10))
    again = rescore(planner)
    assert planner.root.to_dict(max_depth=10) == before
    assert again == planner.scores
    assert planner.search_values == planner.root.action_values


def test_a_frontier_enters_with_its_rollout_value_and_is_approximate():
    planner = make_planner(depth=3, simulations=3000)
    planner.policy_for(planner.model())
    bank, model = planner.solver_bank, planner.model()
    law = bank.continuation_law(model, "L")
    assert all(planner.root.get_child("L", o) is not None for o, _ in law)
    observation, probability = law[0]
    child = planner.root.get_child("L", observation)
    child_model = planner._history_successor(model, "L", observation)
    child_value = max(
        score_tree(child, child_model, 2, bank, planner._history_successor, 0.95).values.values()
    )
    child.action_values.clear()
    child.rollout_value = child_value + 10.0
    scores = rescore(planner)
    assert scores.frontiers == 1 and not scores.exact
    shift = planner.config.gamma * probability * 10.0
    assert scores.values["L"] == pytest.approx(planner.scores.values["L"] + shift)


def test_zero_discount_scores_immediate_rewards():
    planner = make_planner(depth=3, simulations=30, gamma=0.0)
    planner.policy_for(planner.model())
    expected = dict(planner.solver_bank.expected_rewards(planner.model()))
    assert planner.recommendation_values == pytest.approx(expected)


def test_a_child_off_the_chance_law_is_an_error():
    planner = make_planner(depth=2, simulations=30)
    planner.policy_for(planner.model())
    planner.root.create_child("L", ("not", "an observation"))
    with pytest.raises(ValueError, match="zero probability"):
        rescore(planner)


@pytest.mark.parametrize("p", [0.08, 0.85])
def test_l2_complete_tree_matches_the_best_response_to_the_declared_opponent(p):
    """Empirical frequencies leave Q error that exact chance weights remove."""
    row = evaluate_case(
        "mcts",
        2,
        2000,
        3,
        p,
        level=2,
        opponent_depth=2,
        modeled="200",
        opponent_belief=0.3,
        reference="declared",
    )[0]
    assert row["scores_exact"]
    assert row["max_abs_q_error"] < 1e-9 < row["search_max_abs_q_error"]
