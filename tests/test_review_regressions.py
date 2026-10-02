"""Small, deterministic regression cases for the independent code review.

Tests cover contracts and analytic counterexamples, not benchmark reward targets.
Remaining mathematical and experimental limitations are tracked in BACKLOG.md.
"""

import gc
import json
import pickle
import weakref
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from ipomdp.analysis.hierarchy import nested_hierarchy
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.runner import is_complete, run_study
from ipomdp.planning.node import POMCPNode
from tests.study_fixtures import random_tiger


def test_interactive_states_are_immutable_and_pickleable():
    model = TigerModel()
    particle = InteractiveState(TIGER_LEFT, MentalModel(AgentFrame("j", 0, model)))
    with pytest.raises(FrozenInstanceError):
        particle.state = TIGER_RIGHT
    restored = pickle.loads(pickle.dumps(particle))
    assert restored.state == particle.state
    assert restored.opponent.frame.agent_id == "j"
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
    from tests.test_finite_filter import l1

    physics = TigerModel()
    lower_zero = MentalModel(AgentFrame("i", 0, physics))
    lower_one = l1(physics, "i")
    left = MentalModel(
        AgentFrame("j", 2, physics), FiniteBelief(((InteractiveState(TIGER_LEFT, lower_zero), 1),))
    )
    right = MentalModel(
        AgentFrame("j", 2, physics), FiniteBelief(((InteractiveState(TIGER_LEFT, lower_one), 1),))
    )
    belief = FiniteBelief(
        ((InteractiveState(TIGER_LEFT, left), 0.25), (InteractiveState(TIGER_LEFT, right), 0.75))
    )
    weights = {node.path: node.mass for node in nested_hierarchy(belief, "i", 3)}
    assert weights["i_L3/j_L2/i_L0"] == pytest.approx(0.25)
    assert weights["i_L3/j_L2/i_L1"] == pytest.approx(0.75)
    assert weights["i_L3/j_L2"] == pytest.approx(1)


def test_resume_requires_matching_manifest_and_complete_trials(tmp_path):
    spec = random_tiger(trials=2, steps=3)
    result = run_study(spec, tmp_path, workers=1)
    assert is_complete(tmp_path, spec)
    resumed = run_study(spec, tmp_path, workers=1, episode=_must_not_run)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), resumed.reset_index(drop=True), check_dtype=False
    )
    with pytest.raises(ValueError, match="different study"):
        run_study(random_tiger(trials=2, steps=4), tmp_path, workers=1)
    checkpoint = tmp_path / "trials" / "000000.csv"
    pd.read_csv(checkpoint).iloc[:-1].to_csv(checkpoint, index=False)
    with pytest.raises(ValueError, match="Invalid trial"):
        run_study(spec, tmp_path, workers=1)


def test_episodes_are_reproducible_per_trial():
    spec = random_tiger(trials=1, steps=8)
    first, second = pd.DataFrame(run_episode(spec, 4)), pd.DataFrame(run_episode(spec, 4))
    columns = ["action_i", "action_j", "observation_i", "observation_j", "cum_reward_i", "state"]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_paired_statistics_reject_missing_seeds_and_adjust_pvalues(tmp_path):
    from ipomdp.analysis.statistics import compute_statistical_significance

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


class TerminalToy(TigerModel):
    """A deterministic one-action model where "finish" ends the episode next step."""

    def get_all_actions(self, agent_id):
        return ["a"]

    def get_all_observations(self, agent_id):
        return ["o"]

    def observation_distribution(self, state, joint_action, agent_id):
        return (("o", 1.0),)

    def transition_distribution(self, state, joint_action):
        return (("terminal" if state == "finish" else state, 1.0),)

    def get_reward(self, state, joint_action, next_state, agent_id):
        return 2.0

    def is_terminal(self, state):
        return state == "terminal"


def test_public_continuation_is_conditioned_in_online_filter():
    from ipomdp.belief.finite_filter import FiniteInteractiveFilter

    physics = TerminalToy()
    opponent = MentalModel(AgentFrame("j", 0, physics))
    initial = MentalModel(
        AgentFrame("i", 1, physics),
        FiniteBelief(
            (
                (InteractiveState("finish", opponent), 0.5),
                (InteractiveState("alive", opponent), 0.5),
            )
        ),
    )
    kernel = FiniteInteractiveFilter(lambda model: {"a": 1})
    live = kernel.update(initial, "a", "o", terminal=False)
    stopped = kernel.update(initial, "a", "o", terminal=True)
    assert live.evidence == stopped.evidence == pytest.approx(0.5)
    assert {a.state for a, _ in live.belief.mass} == {"alive"}
    assert {a.state for a, _ in stopped.belief.mass} == {"terminal"}


def _must_not_run(spec, trial):
    raise AssertionError("Completed trials must not run again")
