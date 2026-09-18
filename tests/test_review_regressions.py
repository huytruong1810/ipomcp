"""Small, deterministic regression cases for the independent code review.

Tests cover contracts and analytic counterexamples, not benchmark reward targets.
Remaining mathematical defects are tracked separately in test_theory_gaps.py.
"""

import gc
import json
import pickle
import random
import weakref
from types import SimpleNamespace

import pandas as pd
import pytest

from core.config import ExperimentConfig, RTSConfig
from core.distribution import DictDistribution, ParticleDistribution
from examples.tiger.model.tiger_model import LISTEN, TIGER_LEFT, TIGER_RIGHT, TigerModel
from examples.tiger.runners.tiger_baseline_runner import TigerBaselineRunner
from ipomdp.belief import AgentFrame, InteractiveParticle
from solvers.node import POMCPNode
from solvers.rts_planner import RTSPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
from utils.generic_batch_runner import extract_nested_belief_hierarchy, is_batch_complete


@pytest.mark.parametrize("weights", [[-1, 2], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_mass_is_rejected(weights):
    with pytest.raises(ValueError):
        ParticleDistribution(["a", "b"], weights)
    with pytest.raises(ValueError):
        DictDistribution(dict(zip(["a", "b"], weights)))


def test_cdf_boundary_never_selects_zero_mass(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    distribution = ParticleDistribution(["impossible", "possible"], [0, 1])
    assert distribution.sample() == "possible"
    assert distribution.resample(10) == ["possible"] * 10


def test_tiny_weights_and_defensive_copies():
    particles, weights = ["a", "b"], [1e-20, 3e-20]
    distribution = ParticleDistribution(particles, weights)
    particles[0] = "mutated"
    weights[0] = 100
    values, masses = distribution.values()
    values.clear()
    masses.clear()
    assert distribution["a"] == pytest.approx(0.25)
    distribution.normalize()
    assert distribution["a"] == pytest.approx(0.25)
    assert distribution.resample(0) == []
    with pytest.raises(ValueError):
        distribution.resample(-1)


def test_particles_are_shallow_immutable_and_pickleable():
    model = TigerModel()
    first, second = AgentFrame(1, 0, model), AgentFrame("1", 0, model)
    mapping = {1: (first, None), "1": (second, None)}
    particle = InteractiveParticle(TIGER_LEFT, mapping)
    equivalent = InteractiveParticle(TIGER_LEFT, dict(reversed(list(mapping.items()))))
    assert particle == equivalent
    assert hash(particle) == hash(equivalent)
    mapping.clear()
    assert len(particle.models) == 2
    with pytest.raises(TypeError):
        particle.models["new"] = (first, None)
    restored = pickle.loads(pickle.dumps(particle))
    assert restored.state == particle.state
    assert set(restored.models) == {1, "1"}
    assert AgentFrame("i", 1, model) != AgentFrame("i", 1, TigerModel())


def test_child_does_not_retain_discarded_parent_or_siblings():
    parent = POMCPNode()
    child = parent.create_child("a", "o")
    sibling = parent.create_child("b", "o")
    parent_ref, sibling_ref = weakref.ref(parent), weakref.ref(sibling)
    del parent, sibling
    gc.collect()
    assert parent_ref() is None
    assert sibling_ref() is None
    assert child.parent is None
    assert pickle.loads(pickle.dumps(child)).parent is None


def test_nested_hierarchy_averages_different_private_beliefs():
    model = TigerModel()

    def lower(level, count):
        node = POMCPNode()
        for _ in range(count):
            node.add_particle(
                InteractiveParticle(TIGER_LEFT, {"i": (AgentFrame("i", level, model), None)})
            )
        return node

    # Different reservoir sizes must not change the outer 1:3 mixture weights.
    left, right = lower(0, 2), lower(1, 10)
    root = POMCPNode()
    for child in [left, right, right, right]:
        root.add_particle(
            InteractiveParticle(TIGER_LEFT, {"j": (AgentFrame("j", 2, model), child)})
        )
    data = extract_nested_belief_hierarchy(SimpleNamespace(root=root, key=SolverKey("i", 3)))
    weights = dict(zip(data["ids"], data["values"]))
    assert weights["i_L3/j_L2/i_L0"] == pytest.approx(0.25)
    assert weights["i_L3/j_L2/i_L1"] == pytest.approx(0.75)
    assert weights["i_L3/j_L2"] == pytest.approx(1)


class TerminalToy(TigerModel):
    """A deterministic one-action model with analytic finite-horizon returns."""

    def get_all_actions(self, agent_id):
        return ["a"]

    def get_legal_actions(self, state, agent_id):
        return ["a"]

    def get_all_observations(self, agent_id):
        return ["o"]

    def get_observation_prob(self, observation, state, joint_action, agent_id):
        return 1.0

    def sample_transition(self, state, joint_action, rng=None):
        return "terminal" if state == "finish" else state

    def get_reward(self, state, joint_action, next_state, agent_id):
        return 2.0

    def is_terminal(self, state):
        return state == "terminal"


def test_rts_absorbing_terminal_and_mixed_survival_mass():
    planner = RTSPlanner(
        SolverKey("i", 1),
        TerminalToy(),
        ["a"],
        SolverBank(),
        config=RTSConfig(gamma=0.5, max_depth=2, num_particles=10),
    )
    # Half of the mass terminates after the first reward; only half gets step 2.
    belief = [InteractiveParticle("finish", {}), InteractiveParticle("alive", {})]
    assert planner._evaluate_action_branch(belief, "a", 0) == pytest.approx(2 + 0.5 * 0.5 * 2)
    assert planner._evaluate_action_branch([InteractiveParticle("terminal", {})], "a", 0) == 0


def test_batch_resume_requires_matching_manifest_and_complete_trials(tmp_path, monkeypatch):
    config = ExperimentConfig(n_trials=2, max_steps=3)
    runner = TigerBaselineRunner(config, str(tmp_path))
    result = runner.run_batch(max_workers=1)
    assert is_batch_complete(tmp_path / "batch_results.csv", 2, 3)
    monkeypatch.setattr(
        TigerBaselineRunner,
        "_run_single_trial_parallel",
        lambda *_: pytest.fail("Completed trials must not run again"),
    )
    resumed = TigerBaselineRunner(config, str(tmp_path)).run_batch(max_workers=1)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), resumed.reset_index(drop=True), check_dtype=False
    )
    with pytest.raises(ValueError, match="differs"):
        TigerBaselineRunner(ExperimentConfig(n_trials=2, max_steps=4), str(tmp_path)).run_batch(
            max_workers=1
        )
    checkpoint = tmp_path / "trials" / "000000.csv"
    pd.read_csv(checkpoint).iloc[:-1].to_csv(checkpoint, index=False)
    with pytest.raises(ValueError, match="Invalid trial"):
        TigerBaselineRunner(config, str(tmp_path)).run_batch(max_workers=1)


def test_snapshot_capture_preserves_episode_rng():
    runner = TigerBaselineRunner(ExperimentConfig(n_trials=1, max_steps=8))
    batch = pd.DataFrame(runner._run_single_trial_parallel(4))
    snapshot, _ = runner.run_single_trial_with_snapshots(4)
    columns = [
        "action_i",
        "action_j",
        "obs_i",
        "obs_j",
        "reward_i",
        "reward_j",
        "cum_reward_i",
        "cum_reward_j",
    ]
    pd.testing.assert_frame_equal(batch[columns], snapshot[columns])


def test_tiger_rollout_is_independent_of_hidden_state():
    model = TigerModel()
    assert model.get_rollout_action(TIGER_LEFT, "i") == LISTEN
    assert model.get_rollout_action(TIGER_RIGHT, "i") == LISTEN


def test_paired_statistics_reject_missing_seeds_and_adjust_pvalues(tmp_path):
    from examples.experiments.master_nested_ipomdp_benchmark import compute_statistical_significance

    data = pd.DataFrame(
        [
            dict(trial=t, step=1, condition=c, cum_reward_i=t)
            for c in ["A", "B", "C"]
            for t in range(4)
        ]
    )
    compute_statistical_significance(data, str(tmp_path))
    result = json.loads((tmp_path / "statistical_significance_tests.json").read_text())
    assert all(row["holm_pval"] == 1 for row in result.values())
    with pytest.raises(ValueError, match="complete common panel"):
        compute_statistical_significance(data.iloc[:-1], str(tmp_path))
