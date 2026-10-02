"""The interactive particle filter: purity, bounded support, agreement with exact updates."""

import math

import pytest

from ipomdp.belief.canonical import belief_digest
from ipomdp.belief.finite_filter import UnsupportedObservation
from ipomdp.belief.particle_filter import ParticleInteractiveFilter
from ipomdp.domains.tiger import GROWL_LEFT, LISTEN, SILENCE, TIGER_LEFT, TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from tests.test_finite_filter import fixture, greedy_one_step, left_probability

NESTED = {2: {1: 0.5, 0: 0.5}, 1: {0: 1.0}}


def agent(**bank_options):
    config = PlannerConfig(simulations=20, depth=2, modeled_simulations=20)
    return HierarchyBuilder(SolverBank(3, **bank_options)).build(
        "i", 2, TigerModel(), "j", config=config, nested_level_weights=NESTED
    )


def supports(model):
    if model.belief is None:
        return []
    return [len(model.belief.mass)] + [
        n for atom, _ in model.belief.mass for n in supports(atom.opponent)
    ]


def test_updates_are_pure_functions_of_their_arguments():
    first, second = agent(particles=64), agent(particles=64)
    for planner in (first, second):
        planner.update_root(LISTEN, (GROWL_LEFT, SILENCE))
    # Separate banks hold separate physics objects, so compare canonical identity.
    assert belief_digest(first.belief) == belief_digest(second.belief)
    kernel = first.solver_bank.filter
    model = first.model()
    assert kernel.update(model, LISTEN, (GROWL_LEFT, SILENCE)) == kernel.update(
        model, LISTEN, (GROWL_LEFT, SILENCE)
    )


def test_support_is_bounded_at_every_nesting_level():
    planner = agent(particles=16)
    for observation in [("GL", "S"), ("GR", "CL"), ("GL", "S")]:
        planner.update_root(LISTEN, observation)
    assert max(supports(planner.model())) <= 16


def test_posterior_agrees_with_the_exact_update_within_monte_carlo_error():
    exact, approximate = agent(), agent(particles=4000)
    for observation in [("GL", "S"), ("GL", "S")]:
        exact.update_root(LISTEN, observation)
        approximate.update_root(LISTEN, observation)
    p = left_probability(exact.belief)
    assert abs(left_probability(approximate.belief) - p) < 4 * math.sqrt(p * (1 - p) / 4000) + 1e-3


def test_no_explaining_particle_raises_instead_of_repairing():
    kernel = ParticleInteractiveFilter(greedy_one_step, particles=32, seed=0)
    with pytest.raises(UnsupportedObservation):
        kernel.update(fixture(TigerModel()), LISTEN, ("not", "an observation"))


def test_initial_belief_draws_from_the_exact_prior():
    kernel = ParticleInteractiveFilter(greedy_one_step, particles=1000, seed=0)
    rows = tuple(fixture(TigerModel()).belief.mass)
    belief = kernel.initial_belief(rows)
    assert len(belief.mass) <= len(rows)
    assert abs(left_probability(belief) - 0.5) < 0.06
    assert sum(w for atom, w in belief.mass if atom.state == TIGER_LEFT) == left_probability(belief)
