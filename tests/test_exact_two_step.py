"""Independent boundary, survival and policy-identity tests; not qualification."""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.domains.tiger import TIGER_LEFT
from ipomdp.experiments.oracle import evaluate_case, run_oracle_comparison
from ipomdp.planning.config import IPOMCPConfig, MCTSConfig
from ipomdp.planning.node import POMCPNode
from ipomdp.planning.solver_bank import SolverBank
from tests.test_exact_final_step import make_planner


@pytest.mark.parametrize("depth", [0, 3, True, 1.5])
def test_reject_unbounded_or_ambiguous_depth(depth):
    with pytest.raises(ValueError, match="exact_tail_depth"):
        MCTSConfig(exact_final_step=True, exact_tail_depth=depth)


def test_depth_requires_enabled_boundary_before_artifact_creation(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        run_oracle_comparison(tmp_path / "invalid", exact_tail_depth=2)
    assert not (tmp_path / "invalid").exists()


def test_two_step_tail_integrates_observations_before_hidden_state_max(monkeypatch):
    planner = make_planner(p=0.08, depth=2)
    planner.config = replace(planner.config, mcts=replace(planner.config.mcts, exact_tail_depth=2))
    # j/L0 listens with probability 1/3. Then growl evidence is .206/.794;
    # GL -> listen (-1), GR -> OL with weighted reward 8.5*.92-15*.08=6.62.
    # With probability 2/3 j opens, resetting to uniform: V1=-1.
    continuation = (-0.206 + 6.62) / 3 - 2 / 3
    expected = {"L": -1 + 0.95 * continuation, "OL": 1.2 - 0.95, "OR": -91.2 - 0.95}

    def forbidden(*args, **kwargs):
        raise AssertionError("The solved tail must not draw a hidden trajectory")

    monkeypatch.setattr(planner.gen_model, "tree_step", forbidden)
    for atom, _ in planner.belief.mass:
        node = POMCPNode()
        before = random.getstate()
        value = planner._simulate(atom, node, 0, {}, history_model=planner.model())
        assert node.action_values == pytest.approx(expected)
        assert value == pytest.approx(max(expected.values()))
        assert node.action_counts == {}
        assert planner._rollout(atom, 0, history_model=planner.model()) == pytest.approx(value)
        assert random.getstate() == before


def test_continuation_law_keeps_survival_mass_and_branch_limit():
    physics = SimpleNamespace(
        get_legal_actions=lambda *args: ["a"],
        get_all_actions=lambda *args: ["a"],
        transition_distribution=lambda *args: (("dead", 0.8), ("live", 0.2)),
        is_terminal=lambda state: state == "dead",
        observation_distribution=lambda *args: (("x", 0.75), ("y", 0.25)),
    )
    opponent = SimpleNamespace(frame=SimpleNamespace(agent_id="j", level=0, pomdp_model=physics))
    model = SimpleNamespace(
        frame=SimpleNamespace(agent_id="i", pomdp_model=physics),
        belief=SimpleNamespace(mass=((SimpleNamespace(state="live", opponent=opponent), 1.0),)),
    )
    bank = SolverBank(seed=1)
    assert dict(bank._continuation_law(model, "a")) == pytest.approx({"x": 0.15, "y": 0.05})
    bank.filter.max_branches = 1
    with pytest.raises(InferenceBudgetExceeded):
        bank._continuation_law(model, "a")


def test_protagonist_tail_preserves_finite_opponent_and_reference():
    common = dict(
        level=2,
        opponent_depth=3,
        opponent_budget=25,
        opponent_belief=0.085,
        exact_final_step=True,
        backup="empirical_bellman",
        exact_history_rewards=True,
    )
    control = evaluate_case("mcts", 2, 30, 4, 0.2, 0.95, **common)[0]
    candidate = evaluate_case("mcts", 2, 30, 4, 0.2, 0.95, exact_tail_depth=2, **common)[0]
    assert candidate["opponent_model"] == control["opponent_model"]
    assert candidate["oracle_q"] == control["oracle_q"]
    assert candidate["estimated_q"] == pytest.approx(candidate["oracle_q"])
    assert candidate["planner_config"]["mcts"]["exact_tail_depth"] == 2
    assert "exact_tail_depth" not in control["planner_config"]["mcts"]
    assert "exact_tail_depth" not in IPOMCPConfig().to_dict()["mcts"]


def test_two_step_backup_does_not_condition_on_survival(monkeypatch):
    planner = make_planner(depth=2)
    monkeypatch.setattr(
        planner,
        "_final_action_values",
        lambda model: {"a": 3.0} if model == "root" else {"best": 10.0},
    )
    monkeypatch.setattr(
        planner.solver_bank, "continuation_law", lambda model, action: (("live", 0.2),)
    )
    monkeypatch.setattr(planner, "_history_successor", lambda *args: "child")
    assert dict(planner._tail_action_values("root", 2)) == pytest.approx({"a": 4.9})


def test_first_expansion_stores_exact_boundary_without_sampled_action_counts():
    planner = make_planner(p=0.08, depth=3)
    planner.config = replace(
        planner.config,
        mcts=replace(planner.config.mcts, exact_tail_depth=2, backup="empirical_bellman"),
    )
    node = POMCPNode()
    bounds = {
        "q_min": float("inf"),
        "q_max": -float("inf"),
        "root_rewards": dict(planner.solver_bank.expected_rewards(planner.model())),
    }
    atom = planner.belief.mass[0][0]
    planner._simulate(
        atom, node, 0, bounds, belief={atom.state: 1.0}, history_model=planner.model()
    )
    assert sum(node.action_counts.values()) == 1
    action = next(iter(node.children))
    assert len(node.children[action]) == 1
    observation, child = next(iter(node.children[action].items()))
    assert child.action_counts == {}
    assert set(child.action_values) == {"L", "OL", "OR"}
    assert child.rollout_value == pytest.approx(max(child.action_values.values()))
    # Independent alpha-vector oracle, with the same observable posterior.
    from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver

    posterior = planner._history_successor(planner.model(), action, observation)
    p_left = sum(mass for atom, mass in posterior.belief.mass if atom.state == TIGER_LEFT)
    reference = ExactPOMDPSolver(planner.pomdp_model, horizon=2, gamma=0.95)
    assert child.action_values == pytest.approx(reference.q_values(p_left, 2))
