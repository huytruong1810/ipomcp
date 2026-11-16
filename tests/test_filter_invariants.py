"""Small-scale checks of the belief filters against arithmetic done here, not against the filters' own code.

Three kinds of check, each independent of the implementation under test:

- **A closed form.** For a level-1 Tiger agent facing the uniformly random
  level 0, the posterior after any action and observation is a two-term sum
  written out below from Table 1 of Gmytrasiewicz and Doshi (2005). The exact
  filter must reproduce it along every observation sequence of three steps.
- **Laws that every Bayes update obeys, in any domain.** The evidences of all
  observations plus the probability that the episode ends sum to one, and the
  evidence-weighted posteriors average back to the one-step prediction (the
  posterior is a martingale). The prediction is enumerated here directly from
  the domain's kernels. These hold at every level, so they also check the
  recursion at level 2, in domains where no closed form is at hand (UAV,
  Wumpus with its deterministic sensors, Runner-Chaser).
- **Resampling arithmetic.** Systematic draws give each atom floor(N w) or
  ceil(N w) copies; stratified draws are within two copies of N w; and the
  sampled filter's error falls like 1/sqrt(N).
"""

import itertools
import math
import random

import pytest

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.finite_filter import FiniteInteractiveFilter, UnsupportedObservation
from ipomdp.belief.particle_filter import ParticleInteractiveFilter
from ipomdp.belief.resampling import stratified_draws, systematic_draws
from ipomdp.domains.runner_chaser import RunnerChaserModel
from ipomdp.domains.tiger import JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel
from tests.independent_reference import (
    GROWL,
    TIGER_OBSERVATIONS,
    level_one_update,
    nested_model,
    physical_marginal,
    signature,
    tiger_closed_form,
)

# --------------------------------------------------------------------------- #
# Models built by hand (no hierarchy builder, no solver bank)
# --------------------------------------------------------------------------- #


def uniform_policy(model):
    """A fixed policy for intentional opponents: any pure function of the model serves the invariants."""
    actions = model.frame.pomdp_model.get_all_actions(model.frame.agent_id)
    return {action: 1 / len(actions) for action in actions}


def leaning_split(actions, marginal):
    """The first action gets the mass of the most likely physical state, the others share the rest."""
    lead = max(marginal.values())
    if len(actions) == 1 or lead >= 1.0:
        return {actions[0]: 1.0}
    rest = (1.0 - lead) / (len(actions) - 1)
    return {action: lead if index == 0 else rest for index, action in enumerate(actions)}


def leaning_policy(model):
    """A policy that depends on the opponent's belief, so a wrong nested update changes predicted behavior."""
    actions = model.frame.pomdp_model.get_all_actions(model.frame.agent_id)
    return leaning_split(actions, physical_marginal(model.belief))


# --------------------------------------------------------------------------- #
# 1. Closed form: level-1 Tiger against the uniformly random level 0
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p_left", [0.5, 0.08, 0.93])
def test_level_one_tiger_posterior_equals_the_closed_form_along_every_three_step_history(p_left):
    physics = TigerModel()
    exact = FiniteInteractiveFilter(uniform_policy)
    start = nested_model(physics, "i", "j", 1, prior=(("TL", p_left), ("TR", 1 - p_left)))
    checked = 0
    for actions in itertools.product(("L", "OL", "OR"), repeat=3):
        if actions.count("L") < 2:
            continue  # an opening resets to 50/50 whatever came before; two listens exercise accumulation
        for observations in itertools.product(TIGER_OBSERVATIONS, repeat=3):
            model, expected = start, p_left
            for action, observation in zip(actions, observations):
                expected, evidence = tiger_closed_form(expected, action, observation)
                posterior = exact.update(model, action, observation, terminal=False)
                assert posterior.evidence == pytest.approx(evidence, abs=1e-14)
                assert physical_marginal(posterior.belief).get("TL", 0.0) == pytest.approx(expected, abs=1e-12)
                model = MentalModel(model.frame, posterior.belief)
                checked += 1
    assert checked == 7 * 216 * 3


def test_two_quiet_growls_against_a_listener_give_the_textbook_posterior():
    """Against an opponent known to listen, n consistent growls give 0.85^n / (0.85^n + 0.15^n)."""
    physics = TigerModel()
    exact = FiniteInteractiveFilter(lambda model: {"L": 1.0})
    # A level-2 agent whose level-1 opponent always listens: the tiger never resets.
    model = nested_model(physics, "i", "j", 2)
    for n in (1, 2, 3, 4):
        model = MentalModel(model.frame, exact.update(model, "L", ("GL", "S"), terminal=False).belief)
        assert physical_marginal(model.belief)["TL"] == pytest.approx(
            GROWL**n / (GROWL**n + (1 - GROWL) ** n), abs=1e-12
        )


# --------------------------------------------------------------------------- #
# 2. Laws of every Bayes update, enumerated from the kernels
# --------------------------------------------------------------------------- #

CASES = {
    "tiger-l1": (TigerModel(), "i", "j", 1, uniform_policy),
    "tiger-l2": (JointTigerModel(together_safe=40.0), "i", "j", 2, leaning_policy),
    "tiger-l3": (TigerModel(), "j", "i", 3, leaning_policy),
    "uav-l1": (UAVModel(rows=2, cols=2), "i", "j", 1, uniform_policy),
    "uav-l2-target": (UAVModel(rows=2, cols=2), "j", "i", 2, leaning_policy),
    "uav-3x3-l2": (UAVModel(), "i", "j", 2, leaning_policy),
    "wumpus-l1": (WumpusModel(width=2, height=2, n_pits=1), "i", "j", 1, uniform_policy),
    "wumpus-l2-wumpus": (WumpusModel(width=2, height=2, n_pits=1), "j", "i", 2, leaning_policy),
    "runner-chaser-l2": (RunnerChaserModel("cap3"), "r", "c", 2, leaning_policy),
}


def one_step_prediction(model, action, policy):
    """P(s', continues) and P(episode ends) after ``action``, straight from the kernels.

    Marginalizes the opponent's action by ``policy`` (uniform for level 0) and the transition; no
    observation enters, so this is the quantity the evidence-weighted posteriors must average to.
    """
    physics, agent = model.frame.pomdp_model, model.frame.agent_id
    continuing, ending = {}, 0.0
    for atom, mass in model.belief.mass:
        opponent = atom.opponent
        other = opponent.frame.agent_id
        choices = uniform_policy(opponent) if opponent.frame.level == 0 else policy(opponent)
        for other_action, p_action in choices.items():
            for successor, p_next in physics.transition_distribution(atom.state, {agent: action, other: other_action}):
                weight = mass * p_action * p_next
                if physics.is_terminal(successor):
                    ending += weight
                else:
                    continuing[successor] = continuing.get(successor, 0.0) + weight
    return continuing, ending


def assert_bayes_laws(exact, model, action, policy):
    """Total probability and the martingale property at one (belief, action); returns the posteriors."""
    physics, agent = model.frame.pomdp_model, model.frame.agent_id
    continuing, ending = one_step_prediction(model, action, policy)
    total, averaged, posteriors = 0.0, {}, {}
    for observation in physics.get_all_observations(agent):
        try:
            posterior = exact.update(model, action, observation, terminal=False)
        except UnsupportedObservation:
            continue
        assert posterior.evidence > 0
        total += posterior.evidence
        for state, mass in physical_marginal(posterior.belief).items():
            assert not physics.is_terminal(state)
            averaged[state] = averaged.get(state, 0.0) + posterior.evidence * mass
        posteriors[observation] = posterior.belief
    assert total + ending == pytest.approx(1.0, abs=1e-12)
    assert set(averaged) == set(continuing)
    for state, mass in continuing.items():
        assert averaged[state] == pytest.approx(mass, abs=1e-12)
    return posteriors


@pytest.mark.parametrize("name", CASES)
def test_evidence_sums_to_one_and_posteriors_average_to_the_prediction(name):
    """At the prior and along a branching sample of two-step histories, for every action."""
    physics, agent, other, level, policy = CASES[name]
    exact = FiniteInteractiveFilter(policy, max_branches=5_000_000)
    start = nested_model(physics, agent, other, level)
    actions = physics.get_all_actions(agent)
    frontier, visited = [start], 0
    for depth in range(2):
        following = []
        for model in frontier:
            for action in actions:
                posteriors = assert_bayes_laws(exact, model, action, policy)
                visited += 1
                # Continue from the first, the last and the middle observation in canonical order.
                keys = sorted(posteriors, key=repr)
                for key in {keys[0], keys[len(keys) // 2], keys[-1]} if keys else ():
                    following.append(MentalModel(model.frame, posteriors[key]))
        frontier = following[:6] if depth == 0 else []
    assert visited >= len(actions) * 2


def reference_level_two_updates(physics, agent, other, rows, action):
    """{own observation: (posterior over (s', the opponent's updated level-1 belief), evidence)}, by brute force.

    ``rows`` is ((state, the opponent's level-1 belief as {state: p}, mass), ...). The opponent acts by
    ``leaning_split`` at its own belief and updates that belief on its own action and private observation
    with ``independent_reference.level_one_update``: nothing here calls the filter under test.
    """
    other_actions = physics.get_all_actions(other)
    weights, advanced_cache = {}, {}
    for state, nested, mass in rows:
        nested_key = signature(nested)
        for other_action, p_action in leaning_split(other_actions, nested).items():
            joint = {agent: action, other: other_action}
            for successor, p_next in physics.transition_distribution(state, joint):
                if physics.is_terminal(successor):
                    continue
                for private, p_private in physics.observation_distribution(successor, joint, other):
                    cache_key = (nested_key, other_action, private)
                    if cache_key not in advanced_cache:
                        advanced_cache[cache_key] = signature(
                            level_one_update(physics, other, agent, nested, other_action, private)
                        )
                    key = (successor, advanced_cache[cache_key])
                    for observation, likelihood in physics.observation_distribution(successor, joint, agent):
                        table = weights.setdefault(observation, {})
                        table[key] = table.get(key, 0.0) + mass * p_action * p_next * likelihood * p_private
    out = {}
    for observation, table in weights.items():
        total = math.fsum(table.values())
        out[observation] = ({key: weight / total for key, weight in table.items()}, total)
    return out


@pytest.mark.parametrize(
    "physics, agent, other",
    [
        (TigerModel(), "i", "j"),
        (JointTigerModel(together_safe=40.0), "j", "i"),
        (UAVModel(rows=2, cols=2), "i", "j"),
        (UAVModel(rows=2, cols=2), "j", "i"),
        (WumpusModel(width=2, height=2, n_pits=1), "i", "j"),
        (WumpusModel(width=2, height=2, n_pits=1), "j", "i"),
        (RunnerChaserModel("cap3"), "c", "r"),
    ],
    ids=["tiger-i", "joint-tiger-j", "uav-uav", "uav-target", "wumpus-human", "wumpus-wumpus", "chaser"],
)
def test_level_two_joint_posterior_equals_a_brute_force_reference_over_two_steps(physics, agent, other):
    """The joint law of (physical state, opponent's private belief), not only its physical marginal.

    The Bayes laws above cannot see a wrong nested update (the filter would be consistent with itself);
    this comparison can, because the reference advances the opponent's belief with its own arithmetic.
    """
    exact = FiniteInteractiveFilter(leaning_policy, max_branches=5_000_000)
    model = nested_model(physics, agent, other, 2)
    prior = dict(physics.initial_distribution())
    rows = [(state, prior, p) for state, p in physics.initial_distribution()]
    compared = 0
    for step in range(2):
        chosen = None
        for action in physics.get_all_actions(agent):
            reference = reference_level_two_updates(physics, agent, other, rows, action)
            for observation in physics.get_all_observations(agent):
                if observation not in reference:
                    with pytest.raises(UnsupportedObservation):
                        exact.update(model, action, observation, terminal=False)
                    continue
                expected, evidence = reference[observation]
                posterior = exact.update(model, action, observation, terminal=False)
                assert posterior.evidence == pytest.approx(evidence, abs=1e-13)
                found = {}
                for atom, mass in posterior.belief.mass:
                    key = (atom.state, signature(physical_marginal(atom.opponent.belief)))
                    found[key] = found.get(key, 0.0) + mass
                assert set(found) == set(expected)
                for key, mass in expected.items():
                    assert found[key] == pytest.approx(mass, abs=1e-12)
                compared += 1
                # Continue from the most informative supported pair: the one with the largest joint support.
                if chosen is None or len(expected) > len(chosen[2]):
                    chosen = (action, observation, expected, posterior.belief)
        action, observation, _, belief = chosen
        model = MentalModel(model.frame, belief)
        rows = [(atom.state, physical_marginal(atom.opponent.belief), mass) for atom, mass in belief.mass]
    assert compared >= 4


def test_conditioning_on_termination_partitions_the_evidence():
    """evidence(o) = evidence(o, continues) + evidence(o, ends), in a domain whose episodes end."""
    physics = UAVModel(rows=2, cols=2)
    exact = FiniteInteractiveFilter(uniform_policy)
    model = nested_model(physics, "i", "j", 1)
    seen_both = False
    for action in physics.get_all_actions("i"):
        for observation in physics.get_all_observations("i"):
            parts = []
            for terminal in (False, True, None):
                try:
                    parts.append(exact.update(model, action, observation, terminal=terminal).evidence)
                except UnsupportedObservation:
                    parts.append(0.0)
            assert parts[0] + parts[1] == pytest.approx(parts[2], abs=1e-14)
            seen_both = seen_both or (parts[0] > 0 and parts[1] > 0)
    assert seen_both


# --------------------------------------------------------------------------- #
# 3. Resampling arithmetic and the sampled filter's rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(20))
def test_systematic_counts_are_floor_or_ceiling_and_stratified_counts_are_within_two(seed):
    rng = random.Random(seed)
    atoms = rng.randint(2, 40)
    weights = [rng.random() ** 3 + 1e-9 for _ in range(atoms)]
    total = math.fsum(weights)
    rows = tuple(zip(range(atoms), weights))
    for count in (1, 7, 64, 500):
        systematic = systematic_draws(rows, count, random.Random(seed + 1))
        stratified = stratified_draws(rows, count, random.Random(seed + 2))
        assert len(systematic) == len(stratified) == count
        for atom, weight in rows:
            expected = count * weight / total
            assert systematic.count(atom) in (math.floor(expected), math.ceil(expected))
            assert abs(stratified.count(atom) - expected) < 2


def test_resampling_is_unbiased():
    """Mean copies of each atom over many offsets equal N w, for both schemes."""
    rows = tuple(zip("abcde", (0.03, 0.4, 0.17, 0.25, 0.15)))
    count, repeats = 16, 4000
    for draw in (systematic_draws, stratified_draws):
        rng = random.Random(11)
        tally = dict.fromkeys("abcde", 0)
        for _ in range(repeats):
            for atom in draw(rows, count, rng):
                tally[atom] += 1
        for atom, weight in rows:
            # A count per repeat lies within one (systematic) or two (stratified) of N w, so the
            # standard error of the mean is below 2 / sqrt(repeats); five of them is the tolerance.
            assert tally[atom] / repeats == pytest.approx(count * weight, abs=5 * 2 / math.sqrt(repeats))


def test_sampled_filter_error_falls_like_one_over_root_n():
    """Mean total variation of the physical posterior, N = 64 against N = 1,024: a factor near 4."""
    physics = TigerModel()
    model = nested_model(physics, "i", "j", 2, prior=(("TL", 0.3), ("TR", 0.7)))
    exact = physical_marginal(
        FiniteInteractiveFilter(leaning_policy).update(model, "L", ("GL", "S"), terminal=False).belief
    )

    def mean_error(particles):
        errors = []
        for seed in range(24):
            sampled = ParticleInteractiveFilter(leaning_policy, particles, seed)
            belief = sampled.update(model, "L", ("GL", "S"), terminal=False).belief
            marginal = physical_marginal(belief)
            errors.append(0.5 * sum(abs(marginal.get(s, 0.0) - p) for s, p in exact.items()))
        return sum(errors) / len(errors)

    coarse, fine = mean_error(64), mean_error(1024)
    assert fine < coarse
    assert 2.0 < coarse / fine < 8.0
