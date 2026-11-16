"""The exact references against brute-force solvers that share no code with them.

Every planner result in this project is scored against ``ipomdp.reference``.
These tests score the references themselves, at horizons short enough to
enumerate, against ``tests/independent_reference.py`` (plain dictionaries over
the domains' kernels) and, for Tiger, against a closed form and hand-computed
numbers. A defect in a reference would otherwise be invisible: planner and
reference could agree on a wrong value.
"""

import random

import pytest

from ipomdp.domains.runner_chaser import RunnerChaserModel
from ipomdp.domains.tiger import JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.reference.alpha_vector import AlphaVector2D, prune_2d_with_intervals
from ipomdp.reference.exact_agent import build_exact_agent
from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver
from ipomdp.reference.shared_horizon import SharedHorizonReference
from tests.independent_reference import (
    level_one_q,
    level_two_q,
    prior_rows,
    tiger_closed_form_q,
)

GAMMA = 0.95
BELIEFS = [0.0, 0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.98, 1.0]

# --------------------------------------------------------------------------- #
# Tiger, level 1: hand values, a closed form, alpha vectors, enumeration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", BELIEFS)
def test_one_step_tiger_values_are_the_hand_formula(p):
    """Q_1 = (-1, 10 - 110 p, -100 + 110 p): listening, opening left, opening right at P(tiger left) = p."""
    solver = ExactPOMDPSolver(TigerModel(), horizon=1, gamma=GAMMA)
    assert solver.q_values(p, 1) == pytest.approx({"L": -1.0, "OL": 10 - 110 * p, "OR": -100 + 110 * p}, abs=1e-12)


def test_two_step_tiger_values_at_the_uniform_belief_are_the_hand_numbers():
    """At p = 0.5 every growl leaves the posterior at 0.85 or 0.15, inside the listening region (0.1, 0.9).

    So V_1 = -1 after any observation of a listener, and after opening (posterior 0.5):
    Q_2(L) = -1 + 0.95 (-1) = -1.95 and Q_2(OL) = Q_2(OR) = -45 + 0.95 (-1) = -45.95.
    """
    solver = ExactPOMDPSolver(TigerModel(), horizon=2, gamma=GAMMA)
    assert solver.q_values(0.5, 2) == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95}, abs=1e-12)
    exact = build_exact_agent("i", "j", 1, TigerModel(), {1: {0: 1.0}}, 2, GAMMA)
    assert exact.q_values(exact.model()) == pytest.approx({"L": -1.95, "OL": -45.95, "OR": -45.95}, abs=1e-12)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_alpha_vector_values_equal_the_closed_form_belief_tree(horizon):
    """Value iteration over alpha vectors against plain recursion over the analytical Bayes filter."""
    solver = ExactPOMDPSolver(TigerModel(), horizon=horizon, gamma=GAMMA)
    for p in BELIEFS:
        assert solver.q_values(p, horizon) == pytest.approx(tiger_closed_form_q(p, horizon, GAMMA), abs=1e-9)


# --------------------------------------------------------------------------- #
# Alpha-vector pruning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(60))
def test_pruned_envelope_equals_the_pointwise_maximum(seed):
    """The surviving vectors, on their intervals, reproduce max over all vectors at every belief."""
    rng = random.Random(seed)
    count = rng.randint(1, 25)
    vectors = [AlphaVector2D(rng.uniform(-100, 100), rng.uniform(-100, 100), index) for index in range(count)]
    if seed % 3 == 0:  # parallel lines and exact duplicates
        vectors += [AlphaVector2D(v.v0 + 1.0, v.v1 + 1.0, "parallel") for v in vectors[:3]] + vectors[:2]
    if seed % 4 == 0:  # a line that is best only outside [0, 1]
        vectors.append(AlphaVector2D(-900.0, 200.0, "outside"))
    intervals = prune_2d_with_intervals(vectors)
    assert intervals[0][1] == 0.0 and intervals[-1][2] == 1.0
    for (_, _, right), (_, left, _) in zip(intervals, intervals[1:]):
        assert right == pytest.approx(left, abs=1e-9)
    for step in range(201):
        p = step / 200
        best = max(v.value(p) for v in vectors)
        assert max(v.value(p) for v, _, _ in intervals) == pytest.approx(best, abs=1e-7)
        owner = next(v for v, left, right in intervals if left - 1e-9 <= p <= right + 1e-9)
        assert owner.value(p) == pytest.approx(best, abs=1e-6)


# --------------------------------------------------------------------------- #
# Level 1 in every domain: the enumerating best response against brute force
# --------------------------------------------------------------------------- #

LEVEL_ONE = {
    "tiger": (TigerModel(), "i", "j", 4),
    "tiger-coordination": (JointTigerModel(together_safe=40.0), "j", "i", 3),
    "uav": (UAVModel(rows=2, cols=2), "i", "j", 4),
    "uav-target": (UAVModel(rows=2, cols=2), "j", "i", 3),
    "uav-3x3": (UAVModel(), "i", "j", 2),
    "wumpus-human": (WumpusModel(width=2, height=2, n_pits=1), "i", "j", 2),
    "wumpus-wumpus": (WumpusModel(width=2, height=2, n_pits=1), "j", "i", 2),
    "runner": (RunnerChaserModel("cap3"), "r", "c", 4),
    "chaser": (RunnerChaserModel("cap3"), "c", "r", 4),
}


@pytest.mark.parametrize("name", LEVEL_ONE)
def test_level_one_values_equal_brute_force_in_every_domain(name):
    physics, agent, other, deepest = LEVEL_ONE[name]
    prior = dict(physics.initial_distribution())
    for horizon in range(1, deepest + 1):
        exact = build_exact_agent(agent, other, 1, physics, {1: {0: 1.0}}, horizon, GAMMA)
        expected = level_one_q(physics, agent, other, prior, horizon, GAMMA, {})
        assert exact.q_values(exact.model()) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Level 2 in every domain: the exact I-POMDP against brute force, both horizon conventions
# --------------------------------------------------------------------------- #

LEVEL_TWO = {
    "tiger": (TigerModel(), "i", "j", 3),
    "tiger-coordination": (JointTigerModel(together_safe=40.0), "i", "j", 3),
    "tiger-race": (JointTigerModel(together_safe=0.0, solo_penalty=10.0), "j", "i", 2),
    "uav": (UAVModel(rows=2, cols=2), "i", "j", 3),
    "uav-target": (UAVModel(rows=2, cols=2), "j", "i", 3),
    "runner": (RunnerChaserModel("cap3"), "r", "c", 3),
    "chaser": (RunnerChaserModel("cap3"), "c", "r", 3),
}
NESTED = {2: {1: 1.0}, 1: {0: 1.0}}


@pytest.mark.parametrize("name", LEVEL_TWO)
def test_exact_level_two_values_equal_brute_force(name):
    """The modeled level-1 agent replans with the full lookahead at every step."""
    physics, agent, other, deepest = LEVEL_TWO[name]
    for horizon in range(1, deepest + 1):
        exact = build_exact_agent(agent, other, 2, physics, NESTED, horizon, GAMMA)
        expected = level_two_q(physics, agent, other, prior_rows(physics), horizon, GAMMA, horizon, False, {})
        assert exact.q_values(exact.model()) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", LEVEL_TWO)
def test_shared_horizon_level_two_values_equal_brute_force(name):
    """The modeled level-1 agent plays its r-step law when the modeling agent has r steps left."""
    physics, agent, other, deepest = LEVEL_TWO[name]
    differ = False
    for horizon in range(1, deepest + 1):
        exact = build_exact_agent(agent, other, 2, physics, NESTED, horizon, GAMMA)
        shared = SharedHorizonReference(physics, exact.solver_bank, GAMMA).q_values(exact.model(), horizon)
        expected = level_two_q(physics, agent, other, prior_rows(physics), horizon, GAMMA, horizon, True, {})
        assert shared == pytest.approx(expected, abs=1e-9)
        fixed = exact.q_values(exact.model())
        differ = differ or any(abs(shared[a] - fixed[a]) > 1e-6 for a in fixed)
    if name == "tiger-coordination":
        assert differ, "the two conventions should disagree somewhere, or this test compares nothing"


@pytest.mark.parametrize("name", ["tiger-coordination", "uav", "uav-target", "runner"])
@pytest.mark.parametrize("level_one", [0.7, 0.25])
def test_exact_level_two_values_with_a_mixed_prior_over_opponent_levels(name, level_one):
    """The opponent is level 1 with probability ``level_one`` and the random level 0 otherwise."""
    physics, agent, other, deepest = LEVEL_TWO[name]
    nested = {2: {1: level_one, 0: 1.0 - level_one}, 1: {0: 1.0}}
    for horizon in range(1, deepest + 1):
        exact = build_exact_agent(agent, other, 2, physics, nested, horizon, GAMMA)
        expected = level_two_q(physics, agent, other, prior_rows(physics, level_one), horizon, GAMMA, horizon, False, {})
        assert exact.q_values(exact.model()) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "nested, message",
    [
        ({2: {1: -0.5, 0: 1.5}, 1: {0: 1.0}}, "nonnegative"),
        ({2: {1: 0.0, 0: 0.0}, 1: {0: 1.0}}, "positive total"),
        ({2: {2: 1.0}, 1: {0: 1.0}}, "can only model"),
        ({2: {1: float("nan")}, 1: {0: 1.0}}, "nonnegative"),
    ],
)
def test_the_exact_agent_refuses_invalid_level_priors(nested, message):
    with pytest.raises(ValueError, match=message):
        build_exact_agent("i", "j", 2, TigerModel(), nested, 2, GAMMA)


def test_a_level_missing_from_the_declared_priors_is_uniform_over_lower_levels():
    stated = build_exact_agent("i", "j", 2, TigerModel(), {2: {1: 0.5, 0: 0.5}, 1: {0: 1.0}}, 2, GAMMA)
    default = build_exact_agent("i", "j", 2, TigerModel(), {}, 2, GAMMA)
    assert default.q_values(default.model()) == stated.q_values(stated.model())


@pytest.mark.parametrize(
    "physics, agent, other, message",
    [
        (UAVModel(rows=2, cols=2), "i", "j", "exactly two states"),
        (RunnerChaserModel("cap3"), "r", "c", "exactly two states"),
        (TigerModel(), "i", "i", "distinct"),
    ],
)
def test_the_two_state_solver_refuses_what_it_cannot_represent(physics, agent, other, message):
    with pytest.raises(ValueError, match=message):
        ExactPOMDPSolver(physics, horizon=2, gamma=GAMMA, agent_id=agent, opponent_id=other)


def test_the_two_state_solver_names_no_domain():
    """It solves the coordination Tiger's level-1 problem as well (the same kernels, other rewards)."""
    physics = JointTigerModel(together_safe=40.0, solo_penalty=5.0)
    solver = ExactPOMDPSolver(physics, horizon=3, gamma=GAMMA, agent_id="j", opponent_id="i")
    for p in (0.5, 0.07, 0.95):
        prior = {"TL": p, "TR": 1 - p}
        assert solver.q_values(p, 3) == pytest.approx(level_one_q(physics, "j", "i", prior, 3, GAMMA, {}), abs=1e-9)
