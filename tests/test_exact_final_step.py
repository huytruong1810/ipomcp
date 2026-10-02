"""Analytic and information-boundary checks for the exact last-decision values.

These distinguish a belief-conditioned Bellman boundary (integrate, then
maximize) from clairvoyant per-particle maximization. They do not assert
finite-budget multi-step optimality; see the L1 qualification for that.
"""

import random

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import UnsupportedObservation
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.node import POMCPNode
from ipomdp.planning.solver_bank import SolverBank


def make_planner(p=0.5, depth=1, simulations=100):
    physics = TigerModel()
    bank = SolverBank(seed=17)
    bank.set_physical_prior(physics, (TIGER_LEFT, TIGER_RIGHT))
    planner = HierarchyBuilder(bank).build(
        "i",
        1,
        physics,
        "j",
        config=PlannerConfig(simulations=simulations, depth=depth, modeled_simulations=10),
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


def test_exact_leaf_integrates_before_maximizing_and_ignores_rollout_memory(monkeypatch):
    planner = make_planner()

    def forbidden(*args, **kwargs):
        raise AssertionError("A one-step exact value must not sample an action or hidden event")

    monkeypatch.setattr(planner.gen_model, "tree_step", forbidden)
    for atom, _ in planner.belief.mass:
        # Give the heuristic memory a misleading point mass. The leaf must use
        # the authoritative full posterior, not this heuristic or sampled state.
        before = random.getstate()
        assert planner._rollout(atom, 0, {atom.state: 1}, planner.model()) == -1
        assert random.getstate() == before
        node = POMCPNode()
        assert planner._simulate(atom, node, 0, {}, planner.model()) == -1
        assert node.action_values == pytest.approx({"L": -1, "OL": -45, "OR": -45})
        assert node.action_counts == {}  # No fictitious per-action samples.
        assert node.visit_count == 1


def test_history_conditioning_uses_private_observation_and_resets_correctly():
    planner = make_planner(depth=2)
    heard = planner._history_successor(planner.model(), "L", ("GL", "S"))
    p_left = sum(w for atom, w in heard.belief.mass if atom.state == TIGER_LEFT)
    assert p_left == pytest.approx(0.85)
    assert planner._leaf_values(heard) == pytest.approx({"L": -1, "OL": -83.5, "OR": -6.5})
    opened = planner._history_successor(heard, "OR", ("S", "S"))
    assert planner._leaf_values(opened) == pytest.approx({"L": -1, "OL": -45, "OR": -45})
    with pytest.raises(UnsupportedObservation):
        planner._history_successor(heard, "OR", ("GL", "S"))


@pytest.mark.parametrize("belief", [0.08, 0.5, 0.92])
def test_one_decision_values_equal_the_exact_reference(belief):
    row = evaluate_case("mcts", 1, 3, 17, belief)[0]
    assert row["estimated_q"] == pytest.approx(row["oracle_q"])
    assert row["first_action_loss"] == pytest.approx(0)


def test_uniform_two_step_values_are_exact_for_every_sampled_observation():
    # From a uniform Tiger prior, every next private belief is either uniform
    # (after opening) or .15/.85 (after listening); listening is optimal with
    # one decision left in all cases. Hence the continuation is exactly -1.
    planner = make_planner(depth=2)
    assert planner.policy_for(planner.model()) == {"L": 1.0}
    assert planner.root.action_values == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95})
    assert sum(planner.root.action_counts.values()) == 100
