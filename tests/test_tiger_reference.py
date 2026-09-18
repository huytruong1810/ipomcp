"""Exact probability and Bellman checks at the fixed-policy boundary."""

import pytest

from examples.tiger.model.tiger_model import (
    CREAK_RIGHT,
    GROWL_LEFT,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    SILENCE,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from tests.reference_tiger import ACTIONS, STATES, finite_horizon_values, posterior


def test_all_tiger_observation_kernels_normalize():
    for accuracy in (0.0, 0.4, 1.0):
        model = TigerModel(creak_accuracy=accuracy)
        for state in STATES:
            for first in ACTIONS:
                for second in ACTIONS:
                    for agent in ("i", "j"):
                        masses = [
                            model.get_observation_prob(obs, state, {"i": first, "j": second}, agent)
                            for obs in model.get_all_observations(agent)
                        ]
                        assert min(masses) >= 0
                        assert sum(masses) == pytest.approx(1)


def test_quiet_listens_accumulate_evidence_until_reset():
    policies = {"listener": {LISTEN: 1.0}}
    prior = {(TIGER_LEFT, "listener"): 0.5, (TIGER_RIGHT, "listener"): 0.5}
    after_one = posterior(prior, LISTEN, (GROWL_LEFT, SILENCE), policies)
    after_two = posterior(after_one, LISTEN, (GROWL_LEFT, SILENCE), policies)
    assert after_two[TIGER_LEFT, "listener"] == pytest.approx(0.85**2 / (0.85**2 + 0.15**2))


def test_reset_decouples_physics_but_does_not_erase_type_evidence():
    policies = {"listener": {LISTEN: 1.0}, "random": {action: 1 / 3 for action in ACTIONS}}
    prior = {
        (state, kind): mass / 2
        for state in STATES
        for kind, mass in [("listener", 0.8), ("random", 0.2)]
    }
    after = posterior(prior, LISTEN, (GROWL_LEFT, CREAK_RIGHT), policies)
    assert sum(weight for (state, kind), weight in after.items() if kind == "listener") == 0
    assert after[TIGER_LEFT, "random"] == pytest.approx(0.85)


def test_one_step_values_and_symmetry():
    one = finite_horizon_values(0.5, 1)
    assert one == pytest.approx({LISTEN: -1.0, OPEN_LEFT: -45.0, OPEN_RIGHT: -45.0})
    two = finite_horizon_values(0.5, 2)
    assert two[OPEN_LEFT] == pytest.approx(two[OPEN_RIGHT])
    assert max(two, key=two.get) == LISTEN
    assert finite_horizon_values(1.0, 1)[OPEN_RIGHT] == pytest.approx(10.0)
