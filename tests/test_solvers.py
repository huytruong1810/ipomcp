"""Hierarchy construction and reproducible policy integration at bounded budgets."""

import random

import pytest

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import LISTEN, OPEN_LEFT, OPEN_RIGHT, TigerModel
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.policy import greedy_policy
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from tests.reference_tiger import finite_horizon_values
from tests.test_finite_filter import l1


def config(depth=1, simulations=30, modeled=10):
    return PlannerConfig(simulations=simulations, depth=depth, modeled_simulations=modeled)


def build(level, model=None, seed=12, **options):
    bank = SolverBank(seed)
    options.setdefault("config", config())
    solver = HierarchyBuilder(bank).build("i", level, model or TigerModel(), "j", **options)
    return bank, solver


def test_random_planner_and_shared_greedy_tie_rule():
    assert RandomPlanner([LISTEN, OPEN_LEFT, OPEN_RIGHT], random.Random(0)).get_action() in [
        LISTEN,
        OPEN_LEFT,
        OPEN_RIGHT,
    ]
    assert greedy_policy({LISTEN: 1, OPEN_LEFT: 1, OPEN_RIGHT: -1}) == {LISTEN: 0.5, OPEN_LEFT: 0.5}
    with pytest.raises(ValueError):
        greedy_policy({LISTEN: float("nan")})


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_build_all_levels_with_uniform_level_prior(level):
    bank, solver = build(level)
    for lower in range(level):
        assert bank.has_solver(SolverKey("i", lower))
        assert bank.has_solver(SolverKey("j", lower))
    masses = dict.fromkeys(range(level), 0.0)
    for atom, mass in solver.belief.mass:
        masses[atom.opponent.frame.level] += mass
        assert isinstance(atom.opponent, MentalModel)
        assert (atom.opponent.belief is None) == (atom.opponent.frame.level == 0)
    assert list(masses.values()) == pytest.approx([1 / level] * level)
    assert solver.get_action() in solver.actions


def test_nested_priors_are_exact():
    _, solver = build(3, nested_level_weights={3: {2: 1}, 2: {1: 0.8, 0: 0.2}, 1: {0: 1}})
    for atom, _ in solver.belief.mass:
        assert atom.opponent.frame.level == 2
        assert sum(
            w for a, w in atom.opponent.belief.mass if a.opponent.frame.level == 1
        ) == pytest.approx(0.8)


def test_private_policy_cache_is_query_order_invariant_and_does_not_mutate_roots():
    bank, solver = build(2, config=config(depth=2))
    mental = next(atom.opponent for atom, _ in solver.belief.mass if atom.opponent.frame.level == 1)
    original_rng = random.getstate()
    first = bank.policy(mental)
    assert random.getstate() == original_rng
    assert bank.get_solver_for_frame(mental.frame).root.visit_count == 0
    bank.clear_caches()
    assert bank.policy(mental) == first
    first.clear()
    assert bank.policy(mental)


def test_equal_budgets_use_identical_real_and_modeled_policy():
    bank, solver = build(1, seed=20, config=config(depth=2, simulations=20, modeled=20))
    assert solver.policy_for(solver.model()) == bank.policy(solver.model())


def test_nested_initial_models_share_the_unconditional_physical_prior():
    bank, solver = build(2, model=WumpusModel(width=3, height=3, n_pits=1), seed=4)
    outer_states = {atom.state for atom, _ in solver.belief.mass}
    for atom, _ in solver.belief.mass:
        if atom.opponent.belief:
            assert {inner.state for inner, _ in atom.opponent.belief.mass} == outer_states
    assert bank.physical_prior(solver.pomdp_model) == solver.pomdp_model.initial_distribution()


@pytest.mark.parametrize("budget", [3, 10, 100])
def test_one_step_values_equal_exact_expected_rewards(budget):
    model = TigerModel()
    _, solver = build(1, model=model, seed=7, config=config(simulations=budget, modeled=budget))
    for probability in [0.5, 0.85, 0.99]:
        actual = solver.policy_for(l1(model, "i", probability))
        reference = finite_horizon_values(probability, 1)
        assert solver.root.action_values == pytest.approx(reference)
        assert actual == greedy_policy(reference)


def test_frame_lookup_rejects_different_physics_even_at_same_level():
    bank, solver = build(1, seed=2)
    assert bank.get_solver_for_frame(solver.model().frame) is solver
    with pytest.raises(ValueError, match="Frame physics"):
        bank.get_solver_for_frame(AgentFrame("i", 1, TigerModel()))


def test_builder_preserves_reused_solver_physics_identity():
    bank = SolverBank(seed=11)
    builder = HierarchyBuilder(bank)
    private_physics = TigerModel()
    nested = builder.build("j", 1, private_physics, "i", config=config())
    outer = builder.build("i", 2, TigerModel(), "j", config=config(), level_weights={1: 1})
    for atom, _ in outer.belief.mass:
        assert atom.opponent.frame.pomdp_model is private_physics
        assert bank.get_solver_for_frame(atom.opponent.frame) is nested
    assert outer.get_action() in outer.actions
