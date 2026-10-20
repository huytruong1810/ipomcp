"""Near-tie escalation (planning.escalation, study escalation option)."""

import pandas as pd
import pytest

from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, StudySpec
from ipomdp.experiments.suite import COORDINATION, hierarchy_agent, ipomcp
from ipomdp.planning.escalation import EscalatingPlanner

PLANNER = ipomcp(100, 2)
MEASURED = ("_cpu_", "_wall_", "process_rss_mb")


def study(escalation=None, level=2, steps=4):
    agent = hierarchy_agent(level, PLANNER, "100", 4)
    if escalation is not None:
        agent = AgentSpec(**{**agent.__dict__, "escalation": escalation})
    return StudySpec(
        name="escalation",
        domain="tiger-joint",
        domain_options=COORDINATION,
        trials=1,
        steps=steps,
        agents={"i": agent, "j": hierarchy_agent(1, PLANNER, "100", 4)},
    )


def played(rows, drop=()):
    frame = pd.DataFrame(rows)
    keep = [c for c in frame.columns if not any(m in c for m in MEASURED) and c not in drop]
    return frame[keep]


def test_an_escalation_that_never_fires_leaves_the_episode_unchanged():
    for trial in range(3):
        plain = played(run_episode(study(), trial))
        never = run_episode(study({"gap": 0.0, "particles": 64}), trial)
        assert not any(row.get("escalated_i") for row in never)
        pd.testing.assert_frame_equal(plain, played(never, drop=("escalated_i", "base_gap_i")))


def test_near_ties_are_decided_by_the_shadow():
    rows = run_episode(study({"gap": 1e9, "particles": 64}), 0)
    decisions = [row for row in rows if row["step"] > 0]
    assert all(row["escalated_i"] for row in decisions)
    assert all(row["base_gap_i"] >= 0 for row in decisions)


@pytest.mark.parametrize(
    "escalation, message",
    [
        ({"gap": 1.0}, "exactly 'gap' and 'particles'"),
        ({"gap": 1.0, "particles": 4}, "above the agent's"),
        ({"gap": -1.0, "particles": 64}, "nonnegative"),
    ],
)
def test_invalid_escalations_are_rejected(escalation, message):
    with pytest.raises(ValueError, match=message):
        study(escalation)


def test_escalation_needs_a_belief_bound():
    with pytest.raises(ValueError, match="belief bound"):
        AgentSpec(level=1, planner=PLANNER, escalation={"gap": 1.0, "particles": 64})


def test_the_planner_rejects_a_negative_gap():
    with pytest.raises(ValueError):
        EscalatingPlanner(None, None, -1)
