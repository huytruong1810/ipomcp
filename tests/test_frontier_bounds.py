"""Belief-dependent frontier bounds (I-FIB): the chain of the frontier theorem on small domains."""

import random

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, JointTigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.chance_scoring import _Span
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference

GAMMA = 0.95
SLACK = 1e-9


def hierarchy(physics, level, depth):
    """A bank holding i at ``level`` and j's models below it (cheap searched opponents)."""
    bank = SolverBank(seed=5)
    config = PlannerConfig(simulations=20, depth=depth, modeled_simulations=20, gamma=GAMMA)
    weights = {k: 1 / level for k in range(level)}
    HierarchyBuilder(bank).build("i", level, physics, "j", config=config, level_weights=weights)
    return bank


def j_models(physics, level, rng, states):
    """A few models of j: level 0, and at level 1 random beliefs over the given states."""
    models = [MentalModel(AgentFrame("j", 0, physics))]
    if level >= 2:
        i0 = MentalModel(AgentFrame("i", 0, physics))
        for _ in range(2):
            raw = [rng.random() for _ in states]
            total = sum(raw)
            models.append(
                MentalModel(
                    AgentFrame("j", 1, physics),
                    FiniteBelief(tuple((InteractiveState(s, i0), w / total) for s, w in zip(states, raw))),
                )
            )
    return models


def random_model(physics, level, rng, states):
    atoms = [(s, m) for s in states for m in j_models(physics, level, rng, states)]
    raw = [rng.random() ** 3 for _ in atoms]
    total = sum(raw)
    belief = FiniteBelief(
        tuple((InteractiveState(s, m), w / total) for (s, m), w in zip(atoms, raw))
    )
    return MentalModel(AgentFrame("i", level, physics), belief)


CASES = [
    ("tiger", 1, 3),
    ("tiger", 2, 3),
    ("uav", 1, 3),
]


def setup(domain, level, depth):
    if domain == "tiger":
        physics = JointTigerModel(together_safe=40.0)
        states = [TIGER_LEFT, TIGER_RIGHT]
    else:
        physics = UAVModel(rows=2, cols=2)
        states = [s for s in ((u, t) for u in physics.cells() for t in physics.cells())
                  if not physics.is_terminal(s)]
    return physics, states, hierarchy(physics, level, depth)


@pytest.mark.parametrize("domain, level, depth", CASES)
def test_the_frontier_chain_holds_at_random_beliefs(domain, level, depth):
    physics, states, bank = setup(domain, level, depth)
    reference = BestResponseReference(physics, bank, GAMMA)
    span = _Span(*physics.reward_bounds("i"), GAMMA, depth)
    rng = random.Random(0)
    for _ in range(50 if level == 1 else 12):
        model = random_model(physics, level, rng, rng.sample(states, rng.randint(1, len(states))))
        frontier = bank.frontier_bounds(model, GAMMA, depth)
        for r in range(1, depth + 1):
            exact = reference.q_values(model, r)
            nb = frontier.node_bounds(model, r, bank)
            for k, action in enumerate(frontier.actions):
                unit = sum(m * frontier.qbar(r, atom.state)[k] for atom, m in model.belief.mass)
                assert span.lo[r] - SLACK <= nb.q_lo[action] <= exact[action] + SLACK
                assert exact[action] <= nb.q_hi[action] + SLACK
                assert nb.q_hi[action] <= unit + SLACK
                assert unit <= span.hi[r] + SLACK


@pytest.mark.parametrize("domain, level, depth", CASES)
def test_branch_bounds_bracket_the_probability_weighted_successor_value(domain, level, depth):
    physics, states, bank = setup(domain, level, depth)
    reference = BestResponseReference(physics, bank, GAMMA)
    rng = random.Random(1)
    for _ in range(20 if level == 1 else 6):
        model = random_model(physics, level, rng, rng.sample(states, rng.randint(1, len(states))))
        frontier = bank.frontier_bounds(model, GAMMA, depth)
        for r in range(2, depth + 1):
            nb = frontier.node_bounds(model, r, bank)
            for action in frontier.actions:
                _, branches = reference.branches(model, action)
                positive = {o: (p, s) for o, p, s in branches if p > 0}
                assert set(nb.branch_hi[action]) == set(positive)
                assert set(nb.branch_hi[action]) == {o for o, _ in bank.continuation_law(model, action)}
                for o, (p, successor) in positive.items():
                    value = p * max(reference.q_values(successor, r - 1).values())
                    assert nb.branch_lo[action][o] - SLACK <= value <= nb.branch_hi[action][o] + SLACK


def test_no_uav_bound_exceeds_the_spotting_reward():
    # Spotting ends the episode, so no return exceeds 50; the scalar bound is 226 at r = 5.
    physics = UAVModel()
    bank = SolverBank(seed=0)
    model = random_model(physics, 1, random.Random(2), [((2, 0), (0, 2))])
    frontier = bank.frontier_bounds(model, GAMMA, 5)
    states = [(u, t) for u in physics.cells() for t in physics.cells() if u != t]
    for r in range(1, 6):
        assert max(max(frontier.qbar(r, s)) for s in states) <= 50.0 + SLACK
    assert _Span(*physics.reward_bounds("i"), GAMMA, 5).hi[5] > 200


def test_the_bank_shares_one_table_per_agent_and_clears_only_beliefs():
    physics, states, bank = setup("tiger", 1, 3)
    rng = random.Random(3)
    first, second = (random_model(physics, 1, rng, states) for _ in range(2))
    frontier = bank.frontier_bounds(first, GAMMA, 3)
    assert bank.frontier_bounds(second, GAMMA, 3) is frontier
    assert bank.frontier_bounds(first, GAMMA, 4) is not frontier
    before = frontier.node_bounds(first, 3, bank)
    bank.clear_caches()
    assert frontier.node_bounds.cache_info().currsize == 0
    assert frontier.node_bounds(first, 3, bank) == before
