"""Replaying realized episodes through the exact agent (analysis.exact_replay)."""

import pandas as pd
import pytest

from ipomdp.analysis.exact_replay import ExactReplay
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, StudySpec
from ipomdp.experiments.suite import hierarchy_agent, ipomcp

EXACT = {"kind": "exact", "horizon": 2}


def episodes(agents, steps, trials, domain="tiger"):
    spec = StudySpec(name="replay", domain=domain, trials=trials, steps=steps, agents=agents)
    return spec.physics(), [pd.DataFrame(run_episode(spec, t)) for t in range(trials)]


def test_exact_episodes_replay_with_full_agreement_and_no_error():
    nested = {1: {0: 1.0}}
    agents = {
        "i": AgentSpec(level=2, planner=EXACT, nested_level_weights=nested),
        "j": AgentSpec(level=1, planner=EXACT),
    }
    physics, frames = episodes(agents, 4, 3, domain="tiger-joint")
    replay = ExactReplay(physics, "i", 2, 2)
    for frame in frames:
        steps = replay.replay(frame)
        assert [s["step"] for s in steps] == [1, 2, 3, 4]
        assert all(s["agree"] for s in steps)
        assert max(s["level_tv"] for s in steps) == pytest.approx(0, abs=1e-12)
        assert max(s["prediction_tv"] for s in steps) == pytest.approx(0, abs=1e-12)


def test_a_level_one_planner_against_uniform_play_is_bayes_optimal():
    # Against a uniform L0, i-L1's belief is exact at any N, and at H2 chance scoring
    # makes its decision exact, so it must agree with the exact agent at every step.
    agents = {"i": hierarchy_agent(1, ipomcp(100, 2), "100", 64), "j": AgentSpec(level=0)}
    physics, frames = episodes(agents, 6, 3)
    replay = ExactReplay(physics, "i", 1, 2)
    for frame in frames:
        steps = replay.replay(frame)
        assert len(steps) == 6 and all(s["agree"] for s in steps)
        assert max(s["prediction_tv"] for s in steps) == pytest.approx(0, abs=1e-12)


def test_the_window_stops_the_replay():
    agents = {"i": AgentSpec(level=1, planner=EXACT), "j": AgentSpec(level=0)}
    physics, frames = episodes(agents, 5, 1)
    assert len(ExactReplay(physics, "i", 1, 2, max_steps=2).replay(frames[0])) == 2
    assert len(ExactReplay(physics, "i", 1, 2, max_branches=1).replay(frames[0])) == 1


def test_level_zero_has_no_exact_counterpart():
    with pytest.raises(ValueError):
        ExactReplay(
            episodes({"i": AgentSpec(level=0), "j": AgentSpec(level=0)}, 1, 1)[0], "i", 0, 2
        )
