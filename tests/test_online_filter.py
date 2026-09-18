"""Application-level Bayesian regressions replacing heuristic preservation tests.

Both former strict xfails now require actual posterior values. Tests that demanded
type floors or prior restoration were removed because those requirements were the
demonstrated bug; their replacement requires history-conditioned joint measures.
"""

import random

import pytest

from core.config import IPOMCPConfig, MCTSConfig, RTSConfig
from examples.tiger.model.tiger_model import (
    CREAK_RIGHT,
    GROWL_LEFT,
    LISTEN,
    SILENCE,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import (
    FiniteInteractiveFilter,
    InferenceBudgetExceeded,
    UnsupportedObservation,
)
from ipomdp.frame import AgentFrame
from solvers.i_pomcp import IPOMCPPlanner
from solvers.rts_planner import RTSPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
from tests.test_finite_filter import fixture, greedy_one_step, l1, left_probability


@pytest.fixture(params=[IPOMCPPlanner, RTSPlanner])
def planner(request):
    physics = TigerModel()
    bank = SolverBank(seed=10)
    bank.filter = FiniteInteractiveFilter(greedy_one_step)
    cls = request.param
    config = (
        IPOMCPConfig(mcts=MCTSConfig(n_sims=12, max_depth=2))
        if cls is IPOMCPPlanner
        else RTSConfig(max_depth=2, num_particles=12)
    )
    solver = cls(SolverKey("i", 2), physics, physics.get_all_actions("i"), bank, config=config)
    solver.set_initial_belief(fixture(physics).belief, 200)
    return solver


def test_creak_updates_type_under_fixed_known_policies(planner):
    planner.update_root(LISTEN, (GROWL_LEFT, CREAK_RIGHT))
    assert all(atom.opponent.frame.level == 0 for atom, _ in planner.belief.mass)
    assert left_probability(planner.belief) == pytest.approx(0.85)


def test_online_update_matches_full_joint_reference_and_ignores_search_counts(planner):
    expected = planner.solver_bank.filter.update(
        planner.model(), LISTEN, (GROWL_LEFT, SILENCE)
    ).belief
    if isinstance(planner, IPOMCPPlanner):
        child = planner.root.create_child(LISTEN, (GROWL_LEFT, SILENCE))
        child.visit_count = 10000
        child.add_particle(planner.belief.mass[0][0])
    planner.update_root(LISTEN, (GROWL_LEFT, SILENCE))
    assert planner.belief == expected
    for physical, expected_private in [(TIGER_LEFT, 0.85), (TIGER_RIGHT, 0.15)]:
        rows = [
            (a, w)
            for a, w in planner.belief.mass
            if a.state == physical and a.opponent.frame.level == 1
        ]
        mass = sum(w for _, w in rows)
        assert sum(
            w for a, w in rows if left_probability(a.opponent.belief) > 0.5
        ) / mass == pytest.approx(expected_private)


def test_unsupported_evidence_fails_without_changing_live_belief(planner):
    before = planner.belief
    with pytest.raises(UnsupportedObservation):
        planner.update_root(LISTEN, ("invalid", SILENCE))
    assert planner.belief is before


def test_nested_private_model_advances_with_a_simulated_step():
    physics = TigerModel()
    bank = SolverBank(seed=3)
    bank.filter = FiniteInteractiveFilter(lambda model: {LISTEN: 1.0})
    inner = l1(physics, "i")
    middle = MentalModel(
        AgentFrame("j", 2, physics),
        FiniteBelief(tuple((InteractiveState(s, inner), 0.5) for s in (TIGER_LEFT, TIGER_RIGHT))),
    )
    solver = IPOMCPPlanner(SolverKey("i", 3), physics, physics.get_all_actions("i"), bank)
    random.seed(182)
    following, _, _, _ = solver.gen_model.tree_step(
        InteractiveState(TIGER_LEFT, middle), LISTEN, "i", physics
    )
    for atom, _ in following.opponent.belief.mass:
        p = left_probability(atom.opponent.belief)
        assert min(abs(p - 0.85), abs(p - 0.15)) < 1e-12
    assert left_probability(inner.belief) == 0.5


def test_enumeration_budget_fails_explicitly():
    model = fixture(TigerModel())
    with pytest.raises(InferenceBudgetExceeded):
        FiniteInteractiveFilter(greedy_one_step, max_branches=1).update(
            model, LISTEN, (GROWL_LEFT, SILENCE)
        )
