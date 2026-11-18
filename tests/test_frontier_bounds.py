"""Belief-dependent frontier bounds (I-FIB): the chain of the frontier theorem on small domains."""

import math
import random

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, JointTigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.model.pomdp import POMDPModel
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


class RevealModel(POMDPModel):
    """Each agent observes the other's action exactly, and every reward is negative.

    An observation then excludes all but one action of the other agent. A bound
    that maximizes over that agent's actions must keep the excluded ones at
    zero: dropping them charges every observation branch in full, which falls
    below the exact value when continuations are negative.
    """

    def __init__(self, miss=-3.0):
        self.miss = float(miss)

    def initial_distribution(self):
        return (("A", 0.5), ("B", 0.5))

    def transition_distribution(self, state, joint_action):
        if joint_action["i"] == joint_action["j"]:
            return ((state, 1.0),)
        return (("A", 0.5), ("B", 0.5))

    def observation_distribution(self, state, joint_action, agent_id):
        other_action = next(a for agent, a in joint_action.items() if agent != agent_id)
        return ((other_action, 1.0),)

    def get_reward(self, state, joint_action, next_state, agent_id):
        return -1.0 if (state == "A") == (joint_action[agent_id] == "u") else self.miss

    def reward_bounds(self, agent_id):
        return (min(-1.0, self.miss), max(-1.0, self.miss))

    def is_terminal(self, state):
        return False

    def get_all_actions(self, agent_id):
        return ["u", "v"]

    def get_all_observations(self, agent_id):
        return ["u", "v"]


CASES = [
    ("tiger", 1, 3),
    ("tiger", 2, 3),
    ("uav", 1, 3),
    ("reveal", 1, 3),
    ("reveal", 2, 3),
]


def setup(domain, level, depth):
    if domain == "tiger":
        physics = JointTigerModel(together_safe=40.0)
        states = [TIGER_LEFT, TIGER_RIGHT]
    elif domain == "reveal":
        physics = RevealModel()
        states = ["A", "B"]
    else:
        physics = UAVModel(rows=2, cols=2)
        states = [s for s in ((u, t) for u in physics.cells() for t in physics.cells())
                  if not physics.is_terminal(s)]
    return physics, states, hierarchy(physics, level, depth)


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("domain, level, depth", CASES)
def test_itib_lies_between_the_exact_value_and_ifib(domain, level, depth):
    physics, states, bank = setup(domain, level, depth)
    reference = BestResponseReference(physics, bank, GAMMA)
    rng = random.Random(4)
    tighter = 0
    for _ in range(30 if level == 1 else 8):
        model = random_model(physics, level, rng, rng.sample(states, rng.randint(1, len(states))))
        ifib = bank.frontier_bounds(model, GAMMA, depth, "ifib")
        itib = bank.frontier_bounds(model, GAMMA, depth, "itib")
        for r in range(1, depth + 1):
            exact = reference.q_values(model, r)
            a, b = ifib.node_bounds(model, r, bank), itib.node_bounds(model, r, bank)
            for action in ifib.actions:
                assert exact[action] <= b.q_hi[action] + SLACK
                assert b.q_hi[action] <= a.q_hi[action] + SLACK
                tighter += b.q_hi[action] < a.q_hi[action] - 1e-9
                if domain == "uav":  # deterministic transitions: the two coincide
                    assert math.isclose(b.q_hi[action], a.q_hi[action], rel_tol=0, abs_tol=1e-9)
            if r >= 2:
                for action in ifib.actions:
                    _, branches = reference.branches(model, action)
                    for o, p, successor in branches:
                        if p > 0:
                            value = p * max(reference.q_values(successor, r - 1).values())
                            assert value <= b.branch_hi[action][o] + SLACK
    # Measured (Nov 12): I-TIB equals I-FIB on these tiger cells too (a reset's observation is
    # uninformative, and I-FIB already conditions i's next action on the observation only).
    assert tighter >= 0


def test_itib_keeps_opponent_actions_that_an_observation_excludes():
    # Every reward is -1, so Q^3 = -(1 + 0.95 + 0.95^2) whatever anyone does, and both upper
    # bounds must equal it. Maximizing only over the opponent actions that can produce an
    # observation gave -3.755 here: each of i's two observations was charged a full branch.
    physics = RevealModel(miss=-1.0)
    bank = hierarchy(physics, 1, 3)
    model = MentalModel(
        AgentFrame("i", 1, physics),
        FiniteBelief(((InteractiveState("A", MentalModel(AgentFrame("j", 0, physics))), 1.0),)),
    )
    exact = BestResponseReference(physics, bank, GAMMA).q_values(model, 3)
    for kind in ("ifib", "itib"):
        upper = bank.frontier_bounds(model, GAMMA, 3, kind).node_bounds(model, 3, bank).q_hi
        for action, value in exact.items():
            assert math.isclose(value, -(1 + GAMMA + GAMMA**2), rel_tol=0, abs_tol=1e-12)
            assert math.isclose(upper[action], value, rel_tol=0, abs_tol=1e-9)
