"""Hierarchy construction and reproducible policy integration at bounded budgets."""

import random

import pytest

from core.config import IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from examples.tiger.model.tiger_model import LISTEN, OPEN_LEFT, OPEN_RIGHT, TigerModel
from ipomdp.finite_belief import MentalModel
from solvers.policy import greedy_policy
from solvers.random_planner import RandomPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
from utils.bootstrapper import I_POMDP_Bootstrapper


def config(depth=1):
    return IPOMCPConfig(
        mcts=MCTSConfig(n_sims=30, max_depth=depth, node_capacity=20),
        opponent=OpponentPolicyConfig(n_sims=10),
    )


def test_random_planner_and_shared_greedy_tie_rule():
    assert RandomPlanner([LISTEN, OPEN_LEFT, OPEN_RIGHT]).get_action() in [
        LISTEN,
        OPEN_LEFT,
        OPEN_RIGHT,
    ]
    assert greedy_policy({LISTEN: 1, OPEN_LEFT: 1, OPEN_RIGHT: -1}) == {LISTEN: 0.5, OPEN_LEFT: 0.5}
    with pytest.raises(ValueError):
        greedy_policy({LISTEN: float("nan")})


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_bootstrap_all_levels_and_bounded_policy(level):
    bank = SolverBank(seed=12)
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", level, TigerModel(), ["j"], n_particles=40, config=config()
    )
    assert solver.initial_sample_count == 40
    for lower in range(level):
        assert bank.has_solver(SolverKey("i", lower))
        assert bank.has_solver(SolverKey("j", lower))
    masses = {lower: 0 for lower in range(level)}
    for atom, mass in solver.belief.mass:
        masses[atom.opponent.frame.level] += mass
        assert isinstance(atom.opponent, MentalModel)
        assert (atom.opponent.belief is None) == (atom.opponent.frame.level == 0)
    assert list(masses.values()) == pytest.approx([1 / level] * level)
    assert solver.get_action() in solver.actions


def test_nested_priors_are_exact_and_not_clipped_by_node_capacity():
    bank = SolverBank(seed=2)
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i",
        3,
        TigerModel(),
        ["j"],
        n_particles=600,
        nested_level_weights={3: {2: 1}, 2: {1: 0.8, 0: 0.2}, 1: {0: 1}},
        config=config(),
    )
    assert solver.initial_sample_count == 600
    for atom, _ in solver.belief.mass:
        assert atom.opponent.frame.level == 2
        assert sum(
            w for a, w in atom.opponent.belief.mass if a.opponent.frame.level == 1
        ) == pytest.approx(0.8)
    assert solver.root.capacity == 20  # Search reservoir has no authority over belief mass.


def test_private_policy_cache_is_query_order_invariant_and_does_not_mutate_roots():
    bank = SolverBank(seed=12)
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", 2, TigerModel(), ["j"], n_particles=40, config=config(2)
    )
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
    bank = SolverBank(seed=20)
    cfg = IPOMCPConfig(
        mcts=MCTSConfig(n_sims=20, max_depth=2), opponent=OpponentPolicyConfig(n_sims=20)
    )
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", 1, TigerModel(), ["j"], n_particles=40, config=cfg
    )
    assert solver.policy_for(solver.model()) == bank.policy(solver.model())


def test_nested_initial_models_share_the_unconditional_empirical_physical_prior():
    from examples.wumpus.model.wumpus_model import WumpusModel

    bank = SolverBank(seed=4)
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", 2, WumpusModel(), ["j"], n_particles=20, config=config()
    )
    outer_states = {atom.state for atom, _ in solver.belief.mass}
    for atom, _ in solver.belief.mass:
        if atom.opponent.belief:
            assert {inner.state for inner, _ in atom.opponent.belief.mass} == outer_states
    with pytest.raises(ValueError, match="same initial physical sample count"):
        bank.initial_states(solver.pomdp_model, 30)


@pytest.mark.parametrize("budget", [3, 10, 100])
def test_one_step_mcts_values_equal_exact_expected_rewards(budget):
    from tests.reference_tiger import finite_horizon_values
    from tests.test_finite_filter import l1

    bank = SolverBank(seed=7)
    model = TigerModel()
    cfg = IPOMCPConfig(
        mcts=MCTSConfig(n_sims=budget, max_depth=1), opponent=OpponentPolicyConfig(n_sims=budget)
    )
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", 1, model, ["j"], n_particles=20, config=cfg
    )
    for probability in [0.5, 0.85, 0.99]:
        subjective = l1(model, "i", probability)
        actual = solver.policy_for(subjective)
        reference = finite_horizon_values(probability, 1)
        assert solver.root.action_values == pytest.approx(reference)
        assert actual == greedy_policy(reference)


@pytest.mark.parametrize("root_family", ["rts", "mcts"])
def test_explicit_modeled_family_is_independent_of_root(root_family):
    from core.config import RTSConfig
    from solvers.i_pomcp import IPOMCPPlanner
    from solvers.rts_planner import RTSPlanner

    bank = SolverBank(seed=19)
    boot = I_POMDP_Bootstrapper(bank)
    physics = TigerModel()
    modeled = config(1)
    if root_family == "rts":
        solver = boot.create_rts_solver(
            "i",
            3,
            physics,
            ["j"],
            n_particles=20,
            config=RTSConfig(max_depth=1, num_particles=20),
            modeled_config=modeled,
        )
        assert isinstance(solver, RTSPlanner)
    else:
        solver = boot.create_solver(
            "i", 3, physics, ["j"], n_particles=20, config=config(2), modeled_config=modeled
        )
    for who in ["i", "j"]:
        for level in [1, 2]:
            nested = bank.get_solver(SolverKey(who, level))
            assert isinstance(nested, IPOMCPPlanner)
            assert nested.config == modeled
    original = solver.belief
    assert solver.get_action() in solver.actions
    assert solver.belief is original


def test_frame_lookup_rejects_different_physics_even_at_same_level():
    from ipomdp.frame import AgentFrame

    bank = SolverBank(seed=2)
    physics = TigerModel()
    solver = I_POMDP_Bootstrapper(bank).create_solver(
        "i", 1, physics, ["j"], n_particles=20, config=config()
    )
    assert bank.get_solver_for_frame(solver.model().frame) is solver
    with pytest.raises(ValueError, match="Frame physics"):
        bank.get_solver_for_frame(AgentFrame("i", 1, TigerModel()))


def test_bootstrap_preserves_reused_solver_physics_identity():
    bank = SolverBank(seed=11)
    boot = I_POMDP_Bootstrapper(bank)
    private_physics = TigerModel()
    nested = boot.create_solver("j", 1, private_physics, ["i"], n_particles=20, config=config())
    outer = boot.create_solver(
        "i", 2, TigerModel(), ["j"], level_weights={1: 1}, n_particles=20, config=config()
    )
    for atom, _ in outer.belief.mass:
        assert atom.opponent.frame.pomdp_model is private_physics
        assert bank.get_solver_for_frame(atom.opponent.frame) is nested
    assert outer.get_action() in outer.actions


@pytest.mark.parametrize("family", ["rts", "ipomcp"])
def test_controlled_comparison_predicts_declared_opponent_private_policy(family):
    from core.config import ExperimentConfig
    from examples.tiger.runners.planner_comparison import ControlledConditionRunner

    random.seed(31)
    runner = ControlledConditionRunner(
        ExperimentConfig(n_trials=1, max_steps=3), None, family, 2, 1, 30, 2, n_particles=20
    )
    _, protagonist, opponent, _ = runner._setup_domain()
    predicted = protagonist.belief.mass[0][0].opponent
    assert predicted.belief is not opponent.belief
    root = opponent.root
    for observation in [("GL", "S"), ("GR", "S"), ("GL", "CR")]:
        assert protagonist.solver_bank.policy(predicted) == opponent.policy_for(opponent.model())
        predicted = MentalModel(
            predicted.frame,
            protagonist.solver_bank.filter.update(
                predicted, LISTEN, observation, terminal=False
            ).belief,
        )
        opponent.update_root(LISTEN, observation)
    assert protagonist.solver_bank.policy(predicted) == opponent.policy_for(opponent.model())
    assert root.visit_count == 0  # Modeling did not search the original live root.
