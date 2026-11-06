"""The interactive particle filter and RTS backup as Doshi & Gmytrasiewicz (2009) publish them (RESULTS 12.116)."""

import math

import pytest

from ipomdp.baselines.rts import RTSConfig
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import FiniteInteractiveFilter
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, JointTigerModel, TigerModel
from ipomdp.experiments import reachable_decisions as rd
from ipomdp.experiments.modeled_levels import make_planner, parse_stack, register_levels
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference


def _j(physics, p):
    belief = FiniteBelief(tuple((InteractiveState(s, MentalModel(AgentFrame("i", 0, physics))), q)
                                for s, q in ((TIGER_LEFT, p), (TIGER_RIGHT, 1 - p))))
    return MentalModel(AgentFrame("j", 1, physics), belief)


def _bank(physics, particles, depth=2, gamma=0.9, backup="sparse", k=2, particle_filter="published"):
    bank = SolverBank(0, particles=particles, particle_filter=particle_filter)
    key = SolverKey("i", 2)
    planner = make_planner("rts", key, physics, bank, RTSConfig(depth, k, gamma, backup))
    bank.register_solver(key, planner)
    register_levels(bank, physics, "i", 2, parse_stack("exact", 1), depth,
                    BestResponseReference(physics, bank, gamma), planner_kind="rts", gamma=gamma)
    return bank, planner


def _prior(physics, grid=(0.1, 0.5, 0.9)):
    return [(InteractiveState(s, _j(physics, p)), 1.0) for s in (TIGER_LEFT, TIGER_RIGHT) for p in grid]


def test_level_one_updates_are_exact():
    physics = TigerModel()
    bank, _ = _bank(physics, 50)
    model = _j(physics, 0.3)
    exact = FiniteInteractiveFilter(bank.policy).update(model, "L", ("GL", "S")).belief
    assert bank.filter.update(model, "L", ("GL", "S")).belief == exact
    assert bank.filter.initial_belief(model.belief.mass) == model.belief


def test_level_two_posterior_is_n_particles_with_exact_nested_beliefs():
    physics = TigerModel()
    bank, planner = _bank(physics, 50)
    prior = bank.filter.initial_belief(_prior(physics))
    model = MentalModel(AgentFrame("i", 2, physics), prior)
    post = bank.filter.update(model, "L", ("GL", "S"))
    total = math.fsum(w for _, w in post.belief.mass)
    assert all(abs(w / total * 50 - round(w / total * 50)) < 1e-9 for _, w in post.belief.mass)
    # nested j beliefs are exact distributions (not multiples of 1/50 in general)
    nested = {atom.opponent.belief for atom, _ in post.belief.mass}
    assert all(len(b.mass) == 2 for b in nested)
    assert post == bank.filter.update(model, "L", ("GL", "S"))  # pure function of its arguments
    assert 0 < post.evidence < 1


@pytest.mark.parametrize("game", [TigerModel, lambda: JointTigerModel(together_safe=40.0)])
def test_full_backup_on_exact_beliefs_is_the_exact_q(game):
    physics = game()
    bank, planner = _bank(physics, None, depth=2, gamma=0.95, backup="full")
    rows = _prior(physics)
    planner.set_initial_belief(bank.filter.initial_belief(rows))
    planner.policy_for(planner.model())
    exact_bank, exact_planner = _bank(physics, None, depth=2, gamma=0.95, backup="full")
    reference = BestResponseReference(physics, exact_bank, 0.95)
    q = reference.q_values(exact_planner.model(FiniteBelief(tuple(rows))), 2)
    assert planner.recommendation_values == pytest.approx(q, abs=1e-9)


def test_published_backup_counts_only_sampled_observations():
    physics = TigerModel()
    _, full = _bank(physics, None, depth=2, backup="full")
    _, published = _bank(physics, None, depth=2, backup="published", k=1)
    for p in (full, published):
        p.set_initial_belief(FiniteBelief(tuple(_prior(physics))))
        p.policy_for(p.model())
    # one sampled observation keeps only its own likelihood: the listening continuation shrinks toward 0
    assert abs(published.recommendation_values["L"] + 1) < abs(full.recommendation_values["L"] + 1)


def test_sparse_backup_is_the_default_and_keeps_old_seeds():
    assert RTSConfig(2, 2).backup == "sparse"
    with pytest.raises(ValueError):
        RTSConfig(2, 2, backup="other")
