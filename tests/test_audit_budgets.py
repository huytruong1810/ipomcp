"""Actual modeled computation must agree with recorded experiment settings."""

import pytest

from core.config import ExperimentConfig, OpponentPolicyConfig
from examples.experiments.deep_hierarchy_prior_experiment import DeepHierarchyTigerRunner


@pytest.mark.parametrize("budget", [None, 17])
def test_prior_budget_reaches_both_real_agents_private_models(budget):
    runner = DeepHierarchyTigerRunner(
        ExperimentConfig(n_trials=1, max_steps=0),
        None,
        3,
        2,
        planning_depth=1,
        modeled_n_sims=budget,
    )
    expected = OpponentPolicyConfig().n_sims if budget is None else budget
    _, first, second, _ = runner._setup_domain()
    assert runner.modeled_n_sims == expected
    assert first.config.opponent.n_sims == second.config.opponent.n_sims == expected


def test_prior_rejects_unsupported_level_instead_of_extrapolating_budget():
    with pytest.raises(ValueError, match="configured levels"):
        DeepHierarchyTigerRunner(ExperimentConfig(), None, 5, 1)
