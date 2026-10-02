"""The model contract, checked identically for every domain.

For each domain: the prior and every kernel reachable within a few steps are
normalized distributions over declared outcomes; rewards respect the declared
bounds; and the derived samplers reproduce the kernels' frequencies.
"""

import itertools
import math
import random

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel

DOMAINS = {
    "tiger": TigerModel(),
    "uav": UAVModel(),
    "wumpus": WumpusModel(width=3, height=3, n_pits=1),
}
AGENTS = ("i", "j")


def joint_actions(model):
    return [
        dict(zip(AGENTS, pair))
        for pair in itertools.product(*(model.get_all_actions(agent) for agent in AGENTS))
    ]


def reachable_states(model, steps=2, limit=60, seed=0):
    """A deterministic sample of states reachable from the prior."""
    rng = random.Random(seed)
    frontier = [model.sample_initial_state(rng) for _ in range(10)]
    seen = list(dict.fromkeys(frontier))
    for _ in range(steps):
        following = []
        for state in frontier:
            if model.is_terminal(state):
                continue
            for joint in rng.sample(joint_actions(model), min(4, len(joint_actions(model)))):
                following += [s for s, _ in model.transition_distribution(state, joint)]
        frontier = list(dict.fromkeys(following))[:limit]
        seen = list(dict.fromkeys(seen + frontier))[: 3 * limit]
    return seen


def assert_distribution(distribution, support=None):
    values = [value for value, _ in distribution]
    assert len(values) == len(set(values)), "repeated outcomes"
    assert all(math.isfinite(p) and p > 0 for _, p in distribution)
    assert math.fsum(p for _, p in distribution) == pytest.approx(1, abs=1e-12)
    if support is not None:
        assert set(values) <= set(support)


@pytest.mark.parametrize("name", DOMAINS)
def test_prior_is_a_distribution_over_nonterminal_states(name):
    model = DOMAINS[name]
    prior = model.initial_distribution()
    assert_distribution(prior)
    assert not any(model.is_terminal(state) for state, _ in prior)


@pytest.mark.parametrize("name", DOMAINS)
def test_kernels_are_distributions_and_rewards_respect_bounds(name):
    model = DOMAINS[name]
    bounds = {agent: model.reward_bounds(agent) for agent in AGENTS}
    for state in reachable_states(model):
        if model.is_terminal(state):
            continue
        for joint in joint_actions(model):
            transitions = model.transition_distribution(state, joint)
            assert_distribution(transitions)
            for following, _ in transitions:
                for agent in AGENTS:
                    low, high = bounds[agent]
                    assert low <= model.get_reward(state, joint, following, agent) <= high
                    if not model.is_terminal(following):
                        assert_distribution(
                            model.observation_distribution(following, joint, agent),
                            model.get_all_observations(agent),
                        )


@pytest.mark.parametrize("name", DOMAINS)
def test_samplers_reproduce_kernel_frequencies(name):
    """Total variation between 4,000 samples and the kernel stays below 0.05."""
    model = DOMAINS[name]
    rng = random.Random(1)
    state = next(s for s in reachable_states(model) if not model.is_terminal(s))
    for joint in rng.sample(joint_actions(model), 3):
        transitions = model.transition_distribution(state, joint)
        _assert_frequencies(lambda: model.sample_transition(state, joint, rng), transitions)
        following = transitions[0][0]
        if model.is_terminal(following):
            continue
        for agent in AGENTS:
            law = model.observation_distribution(following, joint, agent)
            _assert_frequencies(lambda: model.sample_observation(following, joint, agent, rng), law)
            for observation, probability in law:
                assert (
                    model.observation_probability(observation, following, joint, agent)
                    == probability
                )


def _assert_frequencies(draw, law, n=4000):
    counts = {}
    for _ in range(n):
        value = draw()
        counts[value] = counts.get(value, 0) + 1
    probabilities = dict(law)
    assert set(counts) <= set(probabilities)
    tv = 0.5 * sum(abs(counts.get(v, 0) / n - p) for v, p in probabilities.items())
    assert tv < 0.05


@pytest.mark.parametrize("name", DOMAINS)
def test_actions_depend_only_on_the_agent(name):
    """The contract has no state argument for actions: legality cannot leak state."""
    model = DOMAINS[name]
    for agent in AGENTS:
        actions = model.get_all_actions(agent)
        assert actions and len(actions) == len(set(actions))
        rng = random.Random(0)
        memory = {}
        for state, p in model.initial_distribution():
            memory[state] = memory.get(state, 0.0) + p
        assert model.rollout_action(agent, memory, rng) in actions
