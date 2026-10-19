"""Exact Bayes-optimal agents in episodes (study kind "exact", episodes' exact window)."""

import json
import math

import pytest

from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.domains.tiger import SILENCE, TigerModel
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, ExactConfig, StudySpec
from ipomdp.reference.exact_agent import build_exact_agent

EXACT = {"kind": "exact", "horizon": 2}


def study(steps, i=EXACT, j=EXACT, level_i=1, level_j=1):
    return StudySpec(
        name="exact",
        domain="tiger",
        trials=1,
        steps=steps,
        agents={"i": AgentSpec(level=level_i, planner=i), "j": AgentSpec(level=level_j, planner=j)},
    )


def test_exact_episodes_are_reproducible_and_record_beliefs():
    first, again = run_episode(study(3), 4), run_episode(study(3), 4)
    assert _played(first) == _played(again)
    assert [row["status"] for row in first] == ["Active"] * 4
    for row in first[1:]:
        assert row["prob_l0_j"] == pytest.approx(1.0)
        # i-L1 models j at L0, so it predicts uniform play.
        prediction = json.loads(row["prediction_i"])
        assert prediction == pytest.approx(dict.fromkeys(["L", "OL", "OR"], 1 / 3))
        assert set(json.loads(row["action_values_i"])) == {"L", "OL", "OR"}
    # Both L1 agents start uncertain and listen first.
    assert first[1]["action_i"] == first[1]["action_j"] == "L"


def _played(rows):
    """Rows without timings and memory, which vary between runs."""
    measured = ("_cpu_", "_wall_", "process_rss_mb")
    return [{k: v for k, v in row.items() if not any(m in k for m in measured)} for row in rows]


def test_rows_past_the_exact_window_carry_no_reward():
    rows = run_episode(study(4, i={**EXACT, "max_steps": 2}), 0)
    assert [row["status"] for row in rows] == ["Active"] * 3 + ["Window"] * 2
    assert all(math.isnan(row[f"cum_reward_{a}"]) for row in rows[3:] for a in "ij")
    assert not math.isnan(rows[2]["cum_reward_i"])


def test_an_exceeded_update_budget_ends_the_window():
    rows = run_episode(study(3, i={**EXACT, "max_branches": 1}), 0)
    assert [row["status"] for row in rows] == ["Active", "Window", "Window", "Window"]


def test_the_tie_stream_is_seeded_and_updates_are_budgeted():
    # The tie stream is a function of the seed alone.
    agent = build_exact_agent("i", "j", 1, TigerModel(), {1: {0: 1.0}}, 2, 0.95, seed=3)
    again = build_exact_agent("i", "j", 1, TigerModel(), {1: {0: 1.0}}, 2, 0.95, seed=3)
    assert agent.tie_rng.random() == again.tie_rng.random()
    with pytest.raises(InferenceBudgetExceeded):
        build_exact_agent(
            "i", "j", 1, TigerModel(), {1: {0: 1.0}}, 2, 0.95, max_branches=1
        ).update_root("L", ("GL", SILENCE))


@pytest.mark.parametrize(
    "options, message",
    [
        ({"particles": 64}, "Exact agents take no particles"),
        ({"modeled": "400"}, "Exact agents take no modeled"),
        ({"particle_filter": "sampled"}, "Exact agents take no particle_filter"),
    ],
)
def test_exact_agents_refuse_approximations(options, message):
    with pytest.raises(ValueError, match=message):
        AgentSpec(level=2, planner=EXACT, **options)


@pytest.mark.parametrize(
    "options", [{"horizon": 0}, {"horizon": 2, "max_steps": 0}, {"horizon": 2, "gamma": 0}]
)
def test_invalid_exact_configs_are_rejected(options):
    with pytest.raises(ValueError):
        ExactConfig(**options)


def test_exact_planner_loads_from_toml(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text(
        'name = "s"\ndomain = "tiger"\ntrials = 1\nsteps = 1\n'
        '[agents.i]\nlevel = 2\nplanner = { kind = "exact", horizon = 2, max_steps = 8 }\n'
        "[agents.j]\nlevel = 0\n"
    )
    assert StudySpec.from_toml(path).agents["i"].planner["max_steps"] == 8
