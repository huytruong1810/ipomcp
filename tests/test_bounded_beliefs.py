"""Bounded beliefs (compressed and sampled filters): purity, bounded support, accuracy."""

import math

import pytest

from ipomdp.belief.canonical import belief_digest
from ipomdp.belief.compressed_filter import CompressedInteractiveFilter
from ipomdp.belief.finite_belief import InteractiveState, MentalModel
from ipomdp.belief.finite_filter import FiniteInteractiveFilter, UnsupportedObservation
from ipomdp.belief.frame import AgentFrame
from ipomdp.belief.particle_filter import ParticleInteractiveFilter
from ipomdp.domains.tiger import GROWL_LEFT, LISTEN, SILENCE, TIGER_LEFT, TigerModel
from ipomdp.domains.wumpus.model import WumpusModel
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


FILTERS = pytest.mark.parametrize("particle_filter", ["compressed", "sampled"])


@FILTERS
def test_updates_are_pure_functions_of_their_arguments(particle_filter):
    first, second = (agent(particles=64, particle_filter=particle_filter) for _ in range(2))
    for planner in (first, second):
        planner.update_root(LISTEN, (GROWL_LEFT, SILENCE))
    # Separate banks hold separate physics objects, so compare canonical identity.
    assert belief_digest(first.belief) == belief_digest(second.belief)
    kernel = first.solver_bank.filter
    model = first.model()
    assert kernel.update(model, LISTEN, (GROWL_LEFT, SILENCE)) == kernel.update(
        model, LISTEN, (GROWL_LEFT, SILENCE)
    )


@FILTERS
def test_support_is_bounded_at_every_nesting_level(particle_filter):
    planner = agent(particles=16, particle_filter=particle_filter)
    for observation in [("GL", "S"), ("GR", "CL"), ("GL", "S")]:
        planner.update_root(LISTEN, observation)
    assert max(supports(planner.model())) <= 16


@FILTERS
def test_posterior_agrees_with_the_exact_update_within_monte_carlo_error(particle_filter):
    exact, approximate = agent(), agent(particles=4000, particle_filter=particle_filter)
    for observation in [("GL", "S"), ("GL", "S")]:
        exact.update_root(LISTEN, observation)
        approximate.update_root(LISTEN, observation)
    p = left_probability(exact.belief)
    assert abs(left_probability(approximate.belief) - p) < 4 * math.sqrt(p * (1 - p) / 4000) + 1e-3


@pytest.mark.parametrize("kind", [CompressedInteractiveFilter, ParticleInteractiveFilter])
def test_no_explaining_particle_raises_instead_of_repairing(kind):
    kernel = kind(greedy_one_step, particles=32, seed=0)
    with pytest.raises(UnsupportedObservation):
        kernel.update(fixture(TigerModel()), LISTEN, ("not", "an observation"))


def test_sampled_initial_belief_draws_from_the_exact_prior():
    kernel = ParticleInteractiveFilter(greedy_one_step, particles=1000, seed=0)
    rows = tuple(fixture(TigerModel()).belief.mass)
    belief = kernel.initial_belief(rows)
    assert len(belief.mass) <= len(rows)
    assert abs(left_probability(belief) - 0.5) < 0.06
    assert sum(w for atom, w in belief.mass if atom.state == TIGER_LEFT) == left_probability(belief)


def test_compressed_beliefs_are_exact_while_supports_fit():
    exact, compressed = agent(), agent(particles=10_000)
    for observation in [("GL", "S"), ("GR", "CL"), ("GL", "S")]:
        exact.update_root(LISTEN, observation)
        compressed.update_root(LISTEN, observation)
    assert belief_digest(compressed.belief) == belief_digest(exact.belief)
    model = exact.model()
    assert compressed.solver_bank.filter.update(
        compressed.model(), LISTEN, ("GL", "S")
    ).evidence == pytest.approx(
        exact.solver_bank.filter.update(model, LISTEN, ("GL", "S")).evidence
    )


def wumpus_prior(physics):
    l0 = MentalModel(AgentFrame("j", 0, physics))
    return tuple((InteractiveState(s, l0), p) for s, p in physics.initial_distribution())


def test_compression_errs_by_less_than_one_over_n_per_atom():
    physics = WumpusModel(3, 3, 1)  # 2,048 states, more than N
    rows = wumpus_prior(physics)
    compressed = CompressedInteractiveFilter(greedy_one_step, particles=100, seed=0)
    belief = compressed.initial_belief(rows)
    assert len(belief.mass) <= 100
    mass = dict(belief.mass)
    assert max(abs(mass.get(atom, 0.0) - p) for atom, p in rows) < 1 / 100


def test_compressed_update_is_the_exact_update_of_its_compressed_prior():
    physics = WumpusModel(3, 3, 1)
    compressed = CompressedInteractiveFilter(greedy_one_step, particles=100, seed=0)
    exact = FiniteInteractiveFilter(greedy_one_step)
    model = MentalModel(
        AgentFrame("i", 1, physics), compressed.initial_belief(wumpus_prior(physics))
    )
    action, observation = "Forward", ()
    posterior = compressed.update(model, action, observation)
    reference = exact.update(model, action, observation)
    # Evidence is exact under the compressed prior; the posterior (three wumpus
    # moves per atom) exceeds N and is compressed with per-atom error below 1/N.
    assert posterior.evidence == pytest.approx(reference.evidence)
    assert len(reference.belief.mass) > 100 >= len(posterior.belief.mass)
    mass = dict(posterior.belief.mass)
    assert max(abs(mass.get(atom, 0.0) - p) for atom, p in reference.belief.mass) < 1 / 100


def test_filters_are_chosen_by_name_and_validated():
    assert isinstance(SolverBank(0, particles=8).filter, CompressedInteractiveFilter)
    assert isinstance(
        SolverBank(0, particles=8, particle_filter="sampled").filter, ParticleInteractiveFilter
    )
    assert isinstance(SolverBank(0).filter, FiniteInteractiveFilter)
    with pytest.raises(ValueError, match="particle_filter"):
        SolverBank(0, particles=8, particle_filter="other")


def test_search_beliefs_are_bounded_while_real_beliefs_stay_exact():
    exact = agent()
    split = agent(search_particles=8)
    for observation in [("GL", "S"), ("GR", "CL"), ("GL", "S")]:
        exact.update_root(LISTEN, observation)
        split.update_root(LISTEN, observation)
    assert belief_digest(split.belief) == belief_digest(exact.belief)
    assert len(split.belief.mass) > 8
    bank = split.solver_bank
    assert len(bank.search_model(split.model()).belief.mass) <= 8
    assert isinstance(bank.search_filter, CompressedInteractiveFilter)
    assert split.get_action() in split.actions
    plain = SolverBank(0)
    assert (
        plain.search_filter is plain.filter and plain.search_model(exact.model()) == exact.model()
    )
