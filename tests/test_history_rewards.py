"""Analytic internal-history reward checks, independent of oracle action values."""

from types import SimpleNamespace

import pytest

from core.config import IPOMCPConfig, MCTSConfig
from examples.experiments.planner_oracle_experiment import evaluate_case, run_oracle_comparison
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.i_pomcp import IPOMCPPlanner
from solvers.node import POMCPNode
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey


def setup(backup):
    physics = TigerModel()
    bank = SolverBank(seed=1)
    opponent = MentalModel(AgentFrame("j", 0, physics))
    belief = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, opponent), 0.07),
            (InteractiveState(TIGER_RIGHT, opponent), 0.93),
        )
    )
    model = MentalModel(AgentFrame("i", 1, physics), belief)
    planner = IPOMCPPlanner(
        SolverKey("i", 1),
        physics,
        physics.get_all_actions("i"),
        bank,
        config=IPOMCPConfig(
            mcts=MCTSConfig(max_depth=2, backup=backup, exact_history_rewards=True)
        ),
        exploration_strategy=SimpleNamespace(select_action=lambda *args, **kwargs: "OL"),
    )
    return planner, model


@pytest.mark.parametrize("backup", ["sampled", "empirical_bellman"])
@pytest.mark.parametrize("state", [TIGER_LEFT, TIGER_RIGHT])
@pytest.mark.parametrize("conditioned", [False, True])
def test_internal_reward_uses_posterior_not_sampled_hidden_state(backup, state, conditioned):
    planner, model = setup(backup)
    p = 0.07
    if conditioned:
        # Silence in creak means j listened, so growl Bayes updating is analytic.
        model = planner._history_successor(model, "L", ("GL", "S"))
        p = 0.07 * 0.85 / (0.07 * 0.85 + 0.93 * 0.15)
    opponent = model.belief.mass[0][0].opponent
    node = POMCPNode()
    bounds = {"q_min": float("inf"), "q_max": -float("inf"), "root_rewards": {"OL": 999}}
    value = planner._simulate(
        InteractiveState(state, opponent), node, 1, bounds, history_model=model
    )
    assert value == pytest.approx(10 - 110 * p)
    assert node.action_values["OL"] == pytest.approx(10 - 110 * p)
    # Exact rewards do not turn a sampled final tree node into a solved boundary.
    assert node.action_counts == {"OL": 1}


@pytest.mark.parametrize("backup", ["sampled", "empirical_bellman"])
def test_terminal_reward_is_integrated_without_invented_continuation(backup, monkeypatch):
    planner, model = setup(backup)
    # Exercise a non-final tree event which publicly terminates. The reward
    # integrator is independently stubbed as the specified expectation; a large
    # sampled reward must not leak into either backup or trigger posterior work.
    planner.config = IPOMCPConfig(
        mcts=MCTSConfig(max_depth=3, backup=backup, exact_history_rewards=True)
    )
    atom = model.belief.mass[0][0]
    monkeypatch.setattr(
        planner.gen_model, "tree_step", lambda *args: (atom, {"i": "OL", "j": "L"}, -100, True)
    )
    monkeypatch.setattr(planner.solver_bank, "expected_rewards", lambda model: (("OL", 3.0),))

    def unexpected(*args):
        raise AssertionError("A terminal outcome has no continuing posterior")

    monkeypatch.setattr(planner, "_history_successor", unexpected)
    node = POMCPNode()
    bounds = {"q_min": float("inf"), "q_max": -float("inf")}
    assert (
        planner._simulate(
            atom,
            node,
            1,
            bounds,
            belief={TIGER_LEFT: 0.07, TIGER_RIGHT: 0.93},
            history_model=model,
        )
        == 3.0
    )
    assert node.continuation_counts == {}


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("rewards", [False, True])
def test_options_independently_control_history_and_tail(tail, rewards, monkeypatch):
    counts = {"history": 0, "tail": 0}
    history = IPOMCPPlanner._history_successor
    final = IPOMCPPlanner._final_action_values

    def track_history(self, *args):
        counts["history"] += 1
        return history(self, *args)

    def track_tail(self, *args):
        counts["tail"] += 1
        return final(self, *args)

    monkeypatch.setattr(IPOMCPPlanner, "_history_successor", track_history)
    monkeypatch.setattr(IPOMCPPlanner, "_final_action_values", track_tail)
    row = evaluate_case(
        "mcts",
        3,
        30,
        8,
        0.5,
        0.95,
        exact_final_step=tail,
        exact_history_rewards=rewards,
        backup="empirical_bellman",
    )[0]
    assert bool(counts["history"]) == (tail or rewards)
    assert bool(counts["tail"]) == tail
    assert row["exact_history_rewards"] == rewards
    assert row["planner_config"]["mcts"]["exact_history_rewards"] == rewards


def test_option_does_not_change_modeled_policy_or_reference():
    common = dict(
        level=2,
        opponent_depth=3,
        opponent_budget=25,
        opponent_belief=0.085,
        exact_final_step=True,
        backup="empirical_bellman",
    )
    control = evaluate_case("mcts", 2, 30, 4, 0.2, 0.95, **common)[0]
    candidate = evaluate_case("mcts", 2, 30, 4, 0.2, 0.95, exact_history_rewards=True, **common)[0]
    assert control["opponent_model"] == candidate["opponent_model"]
    assert not candidate["opponent_model"]["config"]["mcts"]["exact_history_rewards"]
    assert control["oracle_q"] == candidate["oracle_q"]


def test_invalid_option_rejected_before_artifacts(tmp_path):
    with pytest.raises(ValueError, match="exact_history_rewards"):
        MCTSConfig(exact_history_rewards=1)
    with pytest.raises(ValueError, match="MCTS-only"):
        run_oracle_comparison(tmp_path / "invalid", planners=("rts",), exact_history_rewards=True)
    assert not (tmp_path / "invalid").exists()
