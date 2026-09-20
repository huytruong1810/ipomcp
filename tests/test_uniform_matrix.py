"""Check the scientific prior and exact sensor-law optimization independently."""

import itertools
import random

import pytest

from core.config import ExperimentConfig
from core.pomdp_model import POMDPModel
from examples.experiments.level_convergence_matrix_experiment import MatrixCellTigerRunner
from examples.tiger.model.tiger_model import TigerModel


@pytest.mark.parametrize("levels", [(0, 0), (1, 4), (2, 1), (3, 2), (4, 4)])
def test_uniform_prior_does_not_disclose_actual_opponent_level(levels):
    runner = MatrixCellTigerRunner(ExperimentConfig(n_trials=1, max_steps=0), None, *levels)
    _, first, second, _ = runner._setup_domain()
    for solver, level in zip((first, second), levels):
        if level == 0:
            continue
        mass = dict.fromkeys(range(level), 0.0)
        for atom, weight in solver.initial_belief.mass:
            mass[atom.opponent.frame.level] += weight
        assert list(mass.values()) == pytest.approx([1 / level] * level)
        assert mass[0] > 0
    # Actual opponent level is not an input to either subjective prior rule.
    alternative = MatrixCellTigerRunner(runner.config, None, levels[0], 0)
    assert alternative.level_prior_i == runner.level_prior_i


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
