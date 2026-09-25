"""Bounded exact references before integration into sampled search.

The independent Tiger oracle below integrates scalar probabilities directly. It
does not use finite-filter transition enumeration, belief constructors, or updates.
Integration checks exercise the production immutable representation and updates;
these scalar enumerations remain independent checks of the probability law.
"""

import itertools
import math
import pickle
from dataclasses import FrozenInstanceError

import pytest

from examples.tiger.model.tiger_model import (
    CREAK_RIGHT,
    GROWL_LEFT,
    GROWL_RIGHT,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    SILENCE,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import FiniteInteractiveFilter, UnsupportedObservation
from ipomdp.frame import AgentFrame
from tests.reference_tiger import posterior as fixed_posterior
from tests.reference_tiger import transition_probability

STATES = (TIGER_LEFT, TIGER_RIGHT)
ACTIONS = (LISTEN, OPEN_LEFT, OPEN_RIGHT)


def l1(physics, agent="j", p_left=0.5):
    other = "i" if agent == "j" else "j"
    opponent = MentalModel(AgentFrame(other, 0, physics))
    belief = FiniteBelief(
        tuple(
            (InteractiveState(state, opponent), weight)
            for state, weight in [(TIGER_LEFT, p_left), (TIGER_RIGHT, 1 - p_left)]
        )
    )
    return MentalModel(AgentFrame(agent, 1, physics), belief)


def left_probability(belief):
    return math.fsum(w for atom, w in belief.mass if atom.state == TIGER_LEFT)


def greedy_one_step(model):
    p = left_probability(model.belief)
    values = {LISTEN: -1, OPEN_LEFT: 10 - 110 * p, OPEN_RIGHT: 110 * p - 100}
    maximum = max(values.values())
    best = [a for a, q in values.items() if q == maximum]
    return {a: 1 / len(best) for a in best}


def fixture(physics, p_private=0.5):
    random = MentalModel(AgentFrame("j", 0, physics))
    intentional = l1(physics, p_left=p_private)
    return MentalModel(
        AgentFrame("i", 2, physics),
        FiniteBelief(
            tuple(
                (InteractiveState(state, opponent), 0.5 * probability)
                for state in STATES
                for opponent, probability in [(random, 0.2), (intentional, 0.8)]
            )
        ),
    )


def oracle_l1(physics, p, action, obs):
    weights = {following: 0.0 for following in STATES}
    for previous, following, unknown in itertools.product(STATES, STATES, ACTIONS):
        prior = p if previous == TIGER_LEFT else 1 - p
        weights[following] += (
            prior
            / 3
            * transition_probability(previous, following, action, unknown, physics.persistent)
            * physics.get_observation_prob(obs, following, {"j": action, "i": unknown}, "j")
        )
    evidence = sum(weights.values())
    assert evidence > 0
    return weights[TIGER_LEFT] / evidence


def oracle_l2(physics, p_private, action, observation):
    """Enumerate one L2 step with a one-step-optimal L1 and random L0 mixture."""
    q = [-1, 10 - 110 * p_private, 110 * p_private - 100]
    best = [a for a, value in zip(ACTIONS, q) if value == max(q)]
    outcomes = {}
    for previous, level, other_action, following in itertools.product(
        STATES, (0, 1), ACTIONS, STATES
    ):
        p_action = 1 / 3 if level == 0 else (1 / len(best) if other_action in best else 0)
        joint = {"i": action, "j": other_action}
        predictive = (
            0.5
            * (0.2 if level == 0 else 0.8)
            * p_action
            * transition_probability(previous, following, action, other_action, physics.persistent)
            * physics.get_observation_prob(observation, following, joint, "i")
        )
        if not predictive:
            continue
        if level == 0:
            outcomes[(following, level, None)] = (
                outcomes.get((following, level, None), 0) + predictive
            )
        else:
            for private in physics.get_all_observations("j"):
                weight = predictive * physics.get_observation_prob(private, following, joint, "j")
                if weight:
                    p_next = oracle_l1(physics, p_private, other_action, private)
                    key = (following, level, p_next)
                    outcomes[key] = outcomes.get(key, 0) + weight
    evidence = sum(outcomes.values())
    return evidence, [
        (key, value / evidence) for key, value in outcomes.items()
    ] if evidence else []


def same_key(a, b):
    return a[:2] == b[:2] and (
        a[2] is b[2] if a[2] is None or b[2] is None else abs(a[2] - b[2]) < 1e-13
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("creak", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("p_private", [0.03, 0.5, 0.97])
def test_every_reachable_l2_observation_matches_independent_reference(persistent, creak, p_private):
    physics = TigerModel(creak_accuracy=creak, persistent=persistent)
    subject = fixture(physics, p_private)
    update = FiniteInteractiveFilter(greedy_one_step).update
    for action in ACTIONS:
        total = 0
        for observation in physics.get_all_observations("i"):
            evidence, expected = oracle_l2(physics, p_private, action, observation)
            if not evidence:
                with pytest.raises(UnsupportedObservation):
                    update(subject, action, observation)
                continue
            actual = update(subject, action, observation)
            total += actual.evidence
            assert actual.evidence == pytest.approx(evidence, abs=1e-13)
            actual_measure = [
                (
                    (
                        atom.state,
                        atom.opponent.frame.level,
                        left_probability(atom.opponent.belief) if atom.opponent.belief else None,
                    ),
                    mass,
                )
                for atom, mass in actual.belief.mass
            ]
            # Different summation orders can split an identical mathematical belief
            # into adjacent floating-point values. Compare full joint measures with
            # a numerical tolerance; production keys never round or merge this way.
            representatives = []
            for key, _ in [*expected, *actual_measure]:
                if not any(same_key(key, other) for other in representatives):
                    representatives.append(key)
            for key in representatives:
                expected_mass = math.fsum(w for k, w in expected if same_key(key, k))
                actual_mass = math.fsum(w for k, w in actual_measure if same_key(key, k))
                assert actual_mass == pytest.approx(expected_mass, abs=1e-13)
        assert total == pytest.approx(1.0)


def test_perfect_creak_eliminates_listening_type_without_floor():
    subject = fixture(TigerModel())
    result = FiniteInteractiveFilter(greedy_one_step).update(
        subject, LISTEN, (GROWL_LEFT, CREAK_RIGHT)
    )
    assert all(atom.opponent.frame.level == 0 for atom, _ in result.belief.mass)
    assert left_probability(result.belief) == pytest.approx(0.85)
    assert result.evidence == pytest.approx(0.2 / 3 / 2)


def test_nested_private_update_does_not_condition_on_outer_hidden_action():
    physics = TigerModel(creak_accuracy=0.4)
    subject = fixture(physics)
    result = FiniteInteractiveFilter(greedy_one_step).update(subject, LISTEN, (GROWL_LEFT, SILENCE))
    # For private (GL,S), j must consider unseen openings by its random L0 model
    # of i. We cannot feed the known outer LISTEN into j's private update.
    # Starting at p=.5 alone masks that mistake; a second step makes it visible.
    subject = MentalModel(subject.frame, result.belief)
    result = FiniteInteractiveFilter(greedy_one_step).update(subject, LISTEN, (GROWL_LEFT, SILENCE))
    private_probabilities = {
        left_probability(atom.opponent.belief)
        for atom, _ in result.belief.mass
        if atom.opponent.frame.level == 1
    }
    expected = oracle_l1(physics, 0.85, LISTEN, (GROWL_LEFT, SILENCE))
    leaked = 0.85**2 / (0.85**2 + 0.15**2)
    assert expected != pytest.approx(leaked)
    assert any(p == pytest.approx(expected) for p in private_probabilities)


def test_joint_state_private_belief_correlation_survives():
    result = FiniteInteractiveFilter(greedy_one_step).update(
        fixture(TigerModel()), LISTEN, (GROWL_LEFT, SILENCE)
    )
    # Condition on the L1 type. Given physical TL, j hears GL with probability .85;
    # given TR it hears GL with probability .15. Factorizing would erase this.
    for state, probability in [(TIGER_LEFT, 0.85), (TIGER_RIGHT, 0.15)]:
        rows = [
            (atom, w)
            for atom, w in result.belief.mass
            if atom.state == state and atom.opponent.frame.level == 1
        ]
        high = sum(w for atom, w in rows if left_probability(atom.opponent.belief) > 0.5)
        assert high / sum(w for _, w in rows) == pytest.approx(probability)


def test_l3_recursively_advances_deepest_intentional_beliefs():
    physics = TigerModel()
    inner = l1(physics, "i")
    middle = MentalModel(
        AgentFrame("j", 2, physics),
        FiniteBelief(tuple((InteractiveState(state, inner), 0.5) for state in STATES)),
    )
    outer = MentalModel(
        AgentFrame("i", 3, physics),
        FiniteBelief(tuple((InteractiveState(state, middle), 0.5) for state in STATES)),
    )
    result = FiniteInteractiveFilter(lambda model: {LISTEN: 1.0}).update(
        outer, LISTEN, (GROWL_LEFT, SILENCE)
    )
    for atom, _ in result.belief.mass:
        for nested, _ in atom.opponent.belief.mass:
            p = left_probability(nested.opponent.belief)
            assert min(abs(p - 0.85), abs(p - 0.15)) < 1e-12
    assert left_probability(inner.belief) == 0.5  # Original immutable value survives.


def test_repeated_quiet_listens_match_fixed_policy_reference():
    physics = TigerModel()
    subject = l1(physics, "i")
    reference = {(state, 0): 0.5 for state in STATES}
    kernel = FiniteInteractiveFilter(greedy_one_step)
    for obs in [(GROWL_LEFT, SILENCE), (GROWL_LEFT, SILENCE), (GROWL_RIGHT, SILENCE)]:
        result = kernel.update(subject, LISTEN, obs)
        reference = fixed_posterior(
            reference, LISTEN, obs, {0: dict.fromkeys(ACTIONS, 1 / 3)}, physics
        )
        subject = MentalModel(subject.frame, result.belief)
        assert left_probability(result.belief) == pytest.approx(reference[TIGER_LEFT, 0])


def test_cache_eviction_order_and_serialization_preserve_results():
    subject = fixture(TigerModel())
    kernel = FiniteInteractiveFilter(greedy_one_step, cache_size=1)
    observation = (GROWL_LEFT, SILENCE)
    first = kernel.update(subject, LISTEN, observation)
    kernel.update(subject, OPEN_LEFT, (SILENCE, SILENCE))
    assert kernel.update(subject, LISTEN, observation) == first
    kernel.update.cache_clear()
    assert kernel.update(subject, LISTEN, observation) == first
    restored = pickle.loads(pickle.dumps(subject))
    rebuilt = FiniteBelief(restored.belief.mass)
    assert restored.belief == rebuilt
    assert hash(restored.belief) == hash(rebuilt)
    assert left_probability(kernel.update(restored, LISTEN, observation).belief) == pytest.approx(
        left_probability(first.belief)
    )


def test_immutable_representation_rejects_mutation_and_invalid_nesting():
    physics = TigerModel()
    subject = fixture(physics)
    rows = list(subject.belief.mass)
    copied = FiniteBelief(rows)
    rows.clear()
    assert copied == subject.belief
    assert FiniteBelief(tuple(reversed(copied.mass))) == copied
    with pytest.raises(FrozenInstanceError):
        copied.mass = ()
    with pytest.raises(ValueError, match="strictly decrease"):
        MentalModel(AgentFrame("i", 1, physics), subject.belief)
    with pytest.raises(ValueError, match="no subjective belief"):
        MentalModel(AgentFrame("i", 0, physics), subject.belief)
    with pytest.raises(TypeError):
        MentalModel(AgentFrame("i", 2, physics), object())


@pytest.mark.parametrize("weights", [(0, 0), (-1, 2), (math.inf, 1), (math.nan, 1)])
def test_invalid_belief_mass_is_rejected(weights):
    source = fixture(TigerModel()).belief.mass
    with pytest.raises(ValueError):
        FiniteBelief(tuple((atom, weight) for (atom, _), weight in zip(source, weights)))


def test_policy_errors_and_nested_model_misspecification_fail_explicitly():
    subject = fixture(TigerModel())
    for policy in [{LISTEN: 0.8}, {LISTEN: -1, OPEN_LEFT: 2}, {"unknown": 1}]:
        with pytest.raises(ValueError):
            FiniteInteractiveFilter(lambda model: policy).update(
                subject, LISTEN, (GROWL_LEFT, SILENCE)
            )
    # A dogmatic private TL belief under persistent, perfect sensors cannot explain
    # an actual GR. Do not silently discard that positive outer branch or reset it.
    physics = TigerModel(growl_accuracy={"i": 1, "j": 1}, persistent=True)
    dogmatic = l1(physics, p_left=1)
    outer = MentalModel(
        AgentFrame("i", 2, physics), FiniteBelief(((InteractiveState(TIGER_RIGHT, dogmatic), 1),))
    )
    with pytest.raises(UnsupportedObservation, match="Nested"):
        FiniteInteractiveFilter(lambda model: {LISTEN: 1}).update(
            outer, LISTEN, (GROWL_RIGHT, SILENCE)
        )


def test_finite_transition_laws_match_tiger_reference_and_deterministic_domains():
    import random

    from examples.uav.model.uav_model import UAVModel
    from examples.wumpus.model.wumpus_model import WumpusModel

    for persistent in [False, True]:
        physics = TigerModel(persistent=persistent)
        for previous, action_i, action_j in itertools.product(STATES, ACTIONS, ACTIONS):
            distribution = dict(
                physics.transition_distribution(previous, {"i": action_i, "j": action_j})
            )
            for following in STATES:
                assert distribution.get(following, 0) == transition_probability(
                    previous, following, action_i, action_j, persistent
                )
    for physics, agents in [(UAVModel(), ("i", "j")), (WumpusModel(), ("i", "j"))]:
        state = physics.get_initial_state(rng=random.Random(91))
        for a, b in itertools.product(*(physics.get_all_actions(agent) for agent in agents)):
            joint = dict(zip(agents, (a, b)))
            distribution = physics.transition_distribution(state, joint)
            assert distribution == ((physics.sample_transition(state, joint), 1.0),)


def test_tiny_and_large_weights_normalize_without_overflow():
    subject = fixture(TigerModel())
    a, b = [atom for atom, _ in subject.belief.mass][:2]
    assert FiniteBelief(((a, 1e308), (b, 1e308))).mass == ((a, 0.5), (b, 0.5))
    with pytest.raises(FloatingPointError):
        FiniteBelief(((a, 1e-300), (b, 1e300)))


def test_observable_action_availability_must_agree_across_intentional_support():
    import random
    from dataclasses import replace

    from examples.wumpus.model.constants import ACTION_FORWARD, AGENT_HUMAN, AGENT_WUMPUS
    from examples.wumpus.model.wumpus_model import WumpusModel

    physics = WumpusModel()
    state = physics.get_initial_state(rng=random.Random(3))
    random_opponent = MentalModel(AgentFrame(AGENT_WUMPUS, 0, physics))
    inconsistent = MentalModel(
        AgentFrame(AGENT_HUMAN, 1, physics),
        FiniteBelief(
            (
                (InteractiveState(state, random_opponent), 0.5),
                (InteractiveState(replace(state, has_arrow=False), random_opponent), 0.5),
            )
        ),
    )
    kernel = FiniteInteractiveFilter(lambda model: {ACTION_FORWARD: 1.0})
    with pytest.raises(ValueError, match="availability differs"):
        kernel.action_distribution(inconsistent, state)


def test_wumpus_sparse_observation_law_matches_full_probability_table():
    import random

    from examples.wumpus.model.wumpus_model import WumpusModel

    model = WumpusModel()
    state = model.get_initial_state(random.Random(3))
    for a, b in itertools.product(model.get_all_actions("i"), model.get_all_actions("j")):
        joint = {"i": a, "j": b}
        following = model.sample_transition(state, joint)
        for agent in ["i", "j"]:
            sparse = dict(model.observation_distribution(following, joint, agent))
            assert sum(sparse.values()) == 1
            for obs in model.get_all_observations(agent):
                assert sparse.get(obs, 0) == model.get_observation_prob(
                    obs, following, joint, agent
                )
