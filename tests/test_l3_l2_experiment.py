"""Contracts for the L3-vs-L2 Tiger study: priors, sensors, filter and exact reference."""

import itertools
import math
import random

import pytest

from examples.experiments.tiger_l3_vs_l2_experiment import (
    PRIOR_I,
    PRIOR_J,
    build_agent,
    hierarchy_values,
    physical_belief,
    predicted_opponent_actions,
    tiger,
)
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.finite_policy_l2 import FinitePolicyL2Reference
from solvers.exact.finite_policy_reference import FinitePolicyReference
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from solvers.solver_bank import SolverBank
from tests.test_finite_policy_l2 import setup as l2_setup
from utils.generic_batch_runner import _get_opponent_level_distribution

ACTIONS = ("L", "OL", "OR")
STATES = (TIGER_LEFT, TIGER_RIGHT)


def test_explicit_physical_prior_is_exact_and_not_replaceable():
    physics, bank = tiger(), SolverBank(seed=1)
    bank.set_physical_prior(physics, STATES)
    assert bank.initial_states(physics, 2) == STATES
    with pytest.raises(ValueError, match="already fixed"):
        bank.set_physical_prior(physics, STATES)
    with pytest.raises(ValueError, match="sample count"):
        bank.initial_states(physics, 2000)


@pytest.mark.parametrize("agent_id,level", [("i", 3), ("j", 2)])
def test_declared_priors_and_hierarchies(agent_id, level):
    planner = build_agent(agent_id, level, "default", 100, 2, 0.95, 5)
    other = "j" if agent_id == "i" else "i"
    expected = PRIOR_I if agent_id == "i" else PRIOR_J
    levels = _get_opponent_level_distribution(planner, other)
    for k, mass in expected.items():
        assert levels[f"prob_l{k}_{other}"] == pytest.approx(mass, abs=1e-15)
    assert physical_belief(planner) == 0.5
    tree = hierarchy_values(planner, agent_id)
    if agent_id == "i":
        assert tree["i_L3/j_L2/i_L1"] == pytest.approx(0.05)
        assert tree["i_L3/j_L1/i_L0"] == pytest.approx(0.8)
    else:
        assert tree["j_L2/i_L1/j_L0"] == pytest.approx(0.5)
    predictions = predicted_opponent_actions(planner)
    for row in predictions.values():
        assert math.fsum(row.values()) == pytest.approx(1)
    assert predictions["0"] == pytest.approx(dict.fromkeys(ACTIONS, 1 / 3))


@pytest.mark.parametrize("agent", ["i", "j"])
def test_creak_point_nine_sensor_is_normalized_and_sampler_consistent(agent):
    physics = tiger()
    other = "j" if agent == "i" else "i"
    for state, own, theirs in itertools.product(STATES, ACTIONS, ACTIONS):
        joint = {agent: own, other: theirs}
        total = math.fsum(
            physics.get_observation_prob(o, state, joint, agent)
            for o in physics.get_all_observations(agent)
        )
        assert total == pytest.approx(1, abs=1e-15)
    rng, joint, n = random.Random(7), {agent: "L", other: "OL"}, 40000
    counts = {}
    for _ in range(n):
        o = physics.sample_observation(TIGER_LEFT, joint, agent, rng=rng)
        counts[o] = counts.get(o, 0) + 1
    for o in physics.get_all_observations(agent):
        p = physics.get_observation_prob(o, TIGER_LEFT, joint, agent)
        assert abs(counts.get(o, 0) / n - p) < 5 * math.sqrt(p * (1 - p) / n) + 1e-12


@pytest.mark.parametrize("p,p_j", [(0.5, 0.5), (0.05, 0.085), (0.8, 0.915)])
def test_general_reference_matches_l2_reference(p, p_j):
    physics, bank, _, model = l2_setup(p=p, p_j=p_j)
    old, new = FinitePolicyL2Reference(physics, bank), FinitePolicyReference(physics, bank)
    for horizon in (1, 2, 3):
        assert new.q_values(model, horizon) == pytest.approx(
            old.q_values(model, horizon), abs=1e-12
        )


@pytest.mark.parametrize("agent", ["i", "j"])
@pytest.mark.parametrize("p", [0.5, 0.2, 0.93])
def test_general_reference_matches_alpha_vectors_against_uniform_l0(agent, p):
    physics = TigerModel(growl_accuracy={"i": 0.85, "j": 0.85}, creak_accuracy=0.9)
    other = "j" if agent == "i" else "i"
    l0 = MentalModel(AgentFrame(other, 0, physics))
    model = MentalModel(
        AgentFrame(agent, 1, physics),
        FiniteBelief(
            ((InteractiveState(TIGER_LEFT, l0), p), (InteractiveState(TIGER_RIGHT, l0), 1 - p))
        ),
    )
    reference = FinitePolicyReference(physics, SolverBank(seed=3))
    exact = ExactPOMDPSolver(physics, horizon=4, gamma=0.95, agent_id=agent, opponent_id=other)
    exact.solve()
    for horizon in (1, 2, 3, 4):
        assert reference.q_values(model, horizon) == pytest.approx(
            exact.q_values(p, horizon), abs=1e-9
        )


def independent_level_posterior(planner, action, observation):
    """One-step Bayes over opponent level from the bank's policy law, no filter."""
    me = planner.key.agent_id
    physics = planner.pomdp_model
    weights = {}
    for atom, prior in planner.belief.mass:
        opponent = atom.opponent
        other = opponent.frame.agent_id
        law = (
            dict.fromkeys(ACTIONS, 1 / 3)
            if opponent.frame.level == 0
            else planner.solver_bank.policy(opponent)
        )
        for other_action, mass in law.items():
            joint = {me: action, other: other_action}
            for following, transition in physics.transition_distribution(atom.state, joint):
                weights[opponent.frame.level] = weights.get(opponent.frame.level, 0.0) + (
                    prior
                    * mass
                    * transition
                    * physics.get_observation_prob(observation, following, joint, me)
                )
    total = math.fsum(weights.values())
    return {level: w / total for level, w in weights.items()} if total else None


@pytest.mark.parametrize("agent_id,level", [("i", 3), ("j", 2)])
def test_filter_level_posterior_matches_independent_bayes(agent_id, level):
    planner = build_agent(agent_id, level, "default", 200, 3, 0.95, 10)
    other = "j" if agent_id == "i" else "i"
    physics = planner.pomdp_model
    checked = 0
    for action, observation in itertools.product(ACTIONS, physics.get_all_observations(agent_id)):
        expected = independent_level_posterior(planner, action, observation)
        if expected is None:
            continue
        posterior = planner.solver_bank.filter.update(
            planner.model(), action, observation, terminal=False
        )
        actual = {}
        for atom, weight in posterior.belief.mass:
            actual[atom.opponent.frame.level] = actual.get(atom.opponent.frame.level, 0.0) + weight
        for k, mass in expected.items():
            assert actual.get(k, 0.0) == pytest.approx(mass, abs=1e-12), (action, observation, k)
        assert other in {a.opponent.frame.agent_id for a, _ in posterior.belief.mass}
        checked += 1
    # Listening supports 2 growls x 3 creak tokens; each opening yields (S, S).
    assert checked == 8


def test_general_reference_handles_mixed_levels_and_conserves_probability():
    planner = build_agent("j", 2, "default", 200, 2, 0.95, 10)
    reference = FinitePolicyReference(planner.pomdp_model, planner.solver_bank)
    model = planner.model()
    assert reference.q_values(model, 1) == pytest.approx(
        dict(planner.solver_bank.expected_rewards(model)), abs=1e-12
    )
    for action in ACTIONS:
        _, branches = reference.branches(model, action)
        assert math.fsum(b[1] for b in branches) == pytest.approx(1, abs=1e-12)
        assert {a.opponent.frame.level for _, _, m in branches for a, _ in m.belief.mass} == {0, 1}
    values = reference.q_values(model, 2)
    assert max(values, key=values.get) == "L"


def _mixed_level_rows(rng, n_atoms, with_l0):
    """Unnormalized i/L2 rows over j/L1 models (distinct private beliefs) and optional j/L0."""
    physics = tiger()
    l0_i = MentalModel(AgentFrame("i", 0, physics))
    atoms = []
    for _ in range(n_atoms):
        p = rng.uniform(0.01, 0.99)
        private = FiniteBelief(
            ((InteractiveState(TIGER_LEFT, l0_i), p), (InteractiveState(TIGER_RIGHT, l0_i), 1 - p))
        )
        opponent = MentalModel(AgentFrame("j", 1, physics), private)
        atoms.append((InteractiveState(rng.choice(STATES), opponent), rng.random() ** 3))
    if with_l0:
        l0_j = MentalModel(AgentFrame("j", 0, physics))
        atoms += [(InteractiveState(state, l0_j), rng.random()) for state in STATES]
    return atoms


def _mixed_level_belief(rng, n_atoms, with_l0):
    return FiniteBelief(tuple(_mixed_level_rows(rng, n_atoms, with_l0)))


def _strata(belief):
    return {(atom.opponent.frame.level, atom.state) for atom, _ in belief.mass}


def test_systematic_resampling_keeps_marginals_and_per_atom_error_below_one_over_k_minus_s():
    """Per-atom error < 1/(K - S) for K > S strata; level/state marginals stay exact.

    Small K relative to S is the regime where the former "2/K" claim failed.
    """
    from ipomdp.finite_filter import resample

    rng = random.Random(5)
    checked = 0
    for _ in range(400):
        belief = _mixed_level_belief(rng, rng.randint(5, 60), rng.random() < 0.5)
        strata = len(_strata(belief))
        for k in (strata + 1, strata + 2, strata + 5, 32):
            if len(belief.mass) <= k:
                continue
            compressed = resample(belief, k)
            assert len(compressed.mass) <= k
            kept = dict(compressed.mass)
            error = max(abs(kept.get(atom, 0.0) - weight) for atom, weight in belief.mass)
            assert error < 1 / (k - strata) + 1e-12
            exact, approx = {}, {}
            for source, target in ((belief, exact), (compressed, approx)):
                for atom, weight in source.mass:
                    key = (atom.opponent.frame.level, atom.state)
                    target[key] = target.get(key, 0.0) + weight
            assert approx.keys() == exact.keys()
            for key, mass in exact.items():
                assert approx[key] == pytest.approx(mass, abs=1e-12)
            checked += 1
    assert checked > 500


def test_resampling_is_independent_of_atom_insertion_order():
    """Equal beliefs must compress identically, or cached posteriors depend on query order."""
    from ipomdp.finite_filter import resample

    rng = random.Random(11)
    for _ in range(50):
        # Build both from the same raw rows: normalization is exact (fsum), so the
        # two beliefs are equal and differ only in insertion order.
        rows = _mixed_level_rows(rng, 40, rng.random() < 0.5)
        belief = FiniteBelief(tuple(rows))
        rng.shuffle(rows)
        reordered = FiniteBelief(tuple(rows))
        assert reordered == belief
        assert [a for a, _ in reordered.mass] != [a for a, _ in belief.mass]
        for k in (8, 16):
            assert resample(reordered, k) == resample(belief, k)


def test_resampling_never_drops_a_level_stratum():
    from ipomdp.finite_filter import resample

    physics = tiger()
    l0_i = MentalModel(AgentFrame("i", 0, physics))
    l0_j = MentalModel(AgentFrame("j", 0, physics))
    atoms = [(InteractiveState(TIGER_LEFT, l0_j), 1e-6)]
    for index in range(200):
        private = FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, l0_i), 1 + index),
                (InteractiveState(TIGER_RIGHT, l0_i), 1 + 200 - index),
            )
        )
        opponent = MentalModel(AgentFrame("j", 1, physics), private)
        atoms.append((InteractiveState(STATES[index % 2], opponent), 1.0))
    compressed = resample(FiniteBelief(tuple(atoms)), 8)
    assert any(atom.opponent.frame.level == 0 for atom, _ in compressed.mass)


def test_bounded_filter_caps_support_at_every_nesting_level():
    planner = build_agent("i", 3, "default", 200, 2, 0.95, 5, max_support=8)
    exact = build_agent("i", 3, "default", 200, 2, 0.95, 5)
    for action, observation in [("L", ("GL", "S")), ("L", ("GR", "S")), ("L", ("GL", "S"))]:
        planner.update_root(action, observation)
        exact.update_root(action, observation)

    def supports(model):
        if model.belief is None:
            return []
        return [len(model.belief.mass)] + [
            n for atom, _ in model.belief.mass for n in supports(atom.opponent)
        ]

    assert max(supports(planner.model())) <= 8
    assert max(supports(exact.model())) > 8
