"""The memoized Tiger sensor law must equal full, uncached enumeration."""

import itertools
import random

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.model.pomdp import POMDPModel


@pytest.mark.parametrize("accuracy", [0.0, 0.85, 1.0])
@pytest.mark.parametrize("creak", [0.0, 0.95, 1.0])
def test_cached_sensor_law_equals_full_uncached_enumeration(accuracy, creak):
    model = TigerModel(growl_accuracy={"i": accuracy, "j": 1 - accuracy}, creak_accuracy=creak)
    rng_state = random.getstate()
    for state, first, second, agent in itertools.product(
        ["TL", "TR"], ["L", "OL", "OR"], ["L", "OL", "OR"], ["i", "j"]
    ):
        joint = {"i": first, "j": second}
        reference = POMDPModel.observation_distribution(model, state, joint, agent)
        assert model.observation_distribution(state, joint, agent) == reference
        assert sum(p for _, p in reference) == pytest.approx(1)
    assert random.getstate() == rng_state
