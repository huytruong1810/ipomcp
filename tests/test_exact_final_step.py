"""Analytic and information-boundary checks for the optional one-step tail.

These tests distinguish a belief-conditioned Bellman boundary from clairvoyant
per-particle maximization. They do not assert finite-budget multi-step optimality.
"""

import random

import pytest

from core.config import IPOMCPConfig, MCTSConfig
from examples.experiments.planner_oracle_experiment import evaluate_case
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import UnsupportedObservation
from ipomdp.frame import AgentFrame
from solvers.node import POMCPNode
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper


def make_planner(p=0.5, depth=1):
    physics = TigerModel()
    planner = I_POMDP_Bootstrapper(SolverBank(seed=17)).create_level1_solver(
        "i",
        physics,
        ["j"],
        n_particles=2,
        config=IPOMCPConfig(
            mcts=MCTSConfig(
                max_depth=depth,
                n_sims=100,
                exact_final_step=True,
            )
        ),
    )
    opponent = MentalModel(AgentFrame("j", 0, physics))
    planner.set_initial_belief(
        FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, opponent), p),
                (InteractiveState(TIGER_RIGHT, opponent), 1 - p),
            )
        ),
        0,
    )
    return planner


def test_exact_leaf_integrates_before_maximizing_and_ignores_rollout_memory(monkeypatch):
    planner = make_planner()

    def forbidden(*args, **kwargs):
        raise AssertionError("A one-step exact value must not sample an action or hidden event")

    monkeypatch.setattr(planner.gen_model, "sample_event", forbidden)
    for atom, _ in planner.belief.mass:
        # Give the heuristic memory a misleading point mass. The tail must use
        # the authoritative full posterior, not this heuristic or sampled state.
        before = random.getstate()
        assert (
            planner._rollout(atom, 0, belief={atom.state: 1}, history_model=planner.model()) == -1
        )
        assert random.getstate() == before
        node = POMCPNode()
        assert planner._simulate(atom, node, 0, {}, history_model=planner.model()) == -1
        assert node.action_values == pytest.approx({"L": -1, "OL": -45, "OR": -45})
        assert node.action_counts == {}  # No fictitious per-action samples.
        assert node.visit_count == 1


def test_history_conditioning_uses_private_observation_and_resets_correctly():
    planner = make_planner(depth=2)
    heard = planner._history_successor(planner.model(), "L", ("GL", "S"))
    p_left = sum(w for atom, w in heard.belief.mass if atom.state == TIGER_LEFT)
    assert p_left == pytest.approx(0.85)
    assert planner._final_action_values(heard) == pytest.approx({"L": -1, "OL": -83.5, "OR": -6.5})
    opened = planner._history_successor(heard, "OR", ("S", "S"))
    assert planner._final_action_values(opened) == pytest.approx({"L": -1, "OL": -45, "OR": -45})
    with pytest.raises(UnsupportedObservation):
        planner._history_successor(heard, "OR", ("GL", "S"))


def test_exact_tail_requires_authoritative_history_model():
    planner = make_planner()
    with pytest.raises(ValueError, match="private-history belief"):
        planner._rollout(planner.belief.mass[0][0], 0)


@pytest.mark.parametrize("belief", [0.08, 0.5, 0.92])
def test_exact_final_step_matches_analytic_one_step_rewards(belief):
    row = evaluate_case("mcts", 1, 3, 17, belief, 0.95, exact_final_step=True)[0]
    assert row["exact_final_step"] is True
    assert row["estimated_q"] == pytest.approx(row["oracle_q"])
    assert row["first_action_loss"] == pytest.approx(0)


def test_exact_leaf_is_not_silently_applied_to_rts():
    with pytest.raises(ValueError, match="MCTS-only"):
        evaluate_case("rts", 1, 3, 17, 0.5, 0.95, exact_final_step=True)


@pytest.mark.parametrize("value", [1, "true", None])
def test_exact_final_step_flag_is_strictly_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        MCTSConfig(exact_final_step=value)


def test_uniform_two_step_values_are_exact_for_every_sampled_observation():
    # From a uniform Tiger prior, every next private belief is either uniform
    # (after opening) or .15/.85 (after listening); listening is optimal with
    # one decision left in all cases. Hence continuation is exactly -1.
    planner = make_planner(depth=2)
    assert planner.policy_for(planner.model()) == {"L": 1.0}
    assert planner.root.action_values == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95})
    assert sum(planner.root.action_counts.values()) == 100
