"""Small-scale checks of the analysis layer against enumeration and textbook definitions.

The analysis layer turns rows into the numbers a reader sees, so an error here
is an error in a reported result even when every planner and reference is
right. Each check compares one tool with something written out here:

- Rao-Blackwellized rewards against the posterior over whole state paths,
  enumerated path by path;
- the exact pair curve against a brute-force enumeration of the joint process
  of two level-1 agents built from ``tests/independent_reference.py``;
- Holm's correction and the Clopper-Pearson interval against their definitions,
  and the interval's coverage by exact enumeration of the binomial;
- the paired difference in its degenerate cases.
"""

import itertools
import math

import pandas as pd
import pytest
from scipy import stats

from ipomdp.analysis.endpoints import bootstrap_interval, paired_difference
from ipomdp.analysis.exact_curves import exact_pair_curve
from ipomdp.analysis.rao_blackwell import rao_blackwell
from ipomdp.analysis.statistics import clopper_pearson, holm
from ipomdp.domains.tiger import JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from tests.independent_reference import greedy, level_one_q, level_one_update

GAMMA = 0.95

# --------------------------------------------------------------------------- #
# Rao-Blackwellized rewards
# --------------------------------------------------------------------------- #


def episode_frame(steps):
    """An episode frame as ``experiments.episodes`` writes it: step 0, then one row per played step."""
    rows = [dict(trial=0, step=0, status="Active", action_i=None, action_j=None,
                 observation_i=None, observation_j=None)]
    for number, (joint, seen, status) in enumerate(steps, start=1):
        rows.append(dict(trial=0, step=number, status=status, action_i=str(joint["i"]), action_j=str(joint["j"]),
                         observation_i=None if seen is None else str(seen["i"]),
                         observation_j=None if seen is None else str(seen["j"])))
    return pd.DataFrame(rows)


def path_posterior_rewards(physics, steps):
    """E[r_t | all actions and observations] for each step and agent, by enumerating every state path."""
    paths = [((state,), p) for state, p in physics.initial_distribution()]
    for joint, seen, _ in steps:
        extended = []
        for path, weight in paths:
            for successor, p_next in physics.transition_distribution(path[-1], joint):
                if seen is None:  # the step that ended the episode: no observation, a terminal successor
                    likelihood = 1.0 if physics.is_terminal(successor) else 0.0
                elif physics.is_terminal(successor):
                    likelihood = 0.0
                else:
                    likelihood = math.prod(
                        dict(physics.observation_distribution(successor, joint, agent)).get(seen[agent], 0.0)
                        for agent in ("i", "j")
                    )
                if weight * p_next * likelihood > 0:
                    extended.append((path + (successor,), weight * p_next * likelihood))
        paths = extended
    total = math.fsum(weight for _, weight in paths)
    return {
        agent: [
            math.fsum(weight * physics.get_reward(path[t], joint, path[t + 1], agent) for path, weight in paths) / total
            for t, (joint, _, _) in enumerate(steps)
        ]
        for agent in ("i", "j")
    }


TIGER_EPISODE = [
    ({"i": "L", "j": "L"}, {"i": ("GL", "S"), "j": ("GR", "S")}, "Active"),
    ({"i": "L", "j": "OR"}, {"i": ("GL", "CR"), "j": ("GL", "CL")}, "Active"),
    ({"i": "OR", "j": "OR"}, {"i": ("GR", "S"), "j": ("GR", "CR")}, "Active"),
    ({"i": "L", "j": "L"}, {"i": ("GR", "S"), "j": ("GR", "CL")}, "Active"),
]
UAV_EPISODE = [
    ({"i": "listen", "j": "listen"}, {"i": 0, "j": ((0, 1), 1)}, "Active"),
    ({"i": "N", "j": "listen"}, {"i": 1, "j": ((0, 1), 0)}, "Active"),
    ({"i": "E", "j": "listen"}, None, "Terminal"),
]


@pytest.mark.parametrize(
    "physics, steps",
    [(JointTigerModel(together_safe=40.0), TIGER_EPISODE), (TigerModel(), TIGER_EPISODE[:2]),
     (UAVModel(rows=2, cols=2), UAV_EPISODE)],
    ids=["tiger-coordination", "tiger", "uav-ending"],
)
def test_rao_blackwell_rewards_equal_the_path_posterior_expectation(physics, steps):
    frame = rao_blackwell(episode_frame(steps), physics)
    expected = path_posterior_rewards(physics, steps)
    for agent in ("i", "j"):
        assert list(frame[f"rb_reward_{agent}"])[1:] == pytest.approx(expected[agent], abs=1e-12)
        assert frame[f"rb_cum_reward_{agent}"].iloc[-1] == pytest.approx(math.fsum(expected[agent]), abs=1e-12)
        assert frame[f"rb_reward_{agent}"].iloc[0] == 0.0


def test_a_record_of_probability_zero_is_refused():
    impossible = [({"i": "listen", "j": "listen"}, {"i": 0, "j": ((1, 0), 1)}, "Active")]  # the target on the UAV's cell
    with pytest.raises(ValueError, match="probability zero"):
        rao_blackwell(episode_frame(impossible), UAVModel(rows=2, cols=2))


# --------------------------------------------------------------------------- #
# Exact pair curve
# --------------------------------------------------------------------------- #


def brute_force_pair_curve(physics, levels, steps, horizon):
    """Expected cumulative rewards of two agents at level 0 or 1, by enumerating (state, belief i, belief j)."""
    prior = dict(physics.initial_distribution())
    cache = {}

    def policy(agent, other, belief):
        if belief is None:
            actions = physics.get_all_actions(agent)
            return {a: 1 / len(actions) for a in actions}
        return greedy(level_one_q(physics, agent, other, dict(belief), horizon, GAMMA, cache))

    start = {a: frozenset(prior.items()) if levels[a] else None for a in ("i", "j")}
    nodes = {(state, start["i"], start["j"]): p for state, p in prior.items()}
    cumulative, curve = {"i": 0.0, "j": 0.0}, []
    for _ in range(steps):
        following = {}
        for (state, b_i, b_j), mass in nodes.items():
            for a_i, p_i in policy("i", "j", b_i).items():
                for a_j, p_j in policy("j", "i", b_j).items():
                    joint = {"i": a_i, "j": a_j}
                    for successor, p_next in physics.transition_distribution(state, joint):
                        weight = mass * p_i * p_j * p_next
                        for agent in ("i", "j"):
                            cumulative[agent] += weight * physics.get_reward(state, joint, successor, agent)
                        if physics.is_terminal(successor):
                            continue
                        for o_i, q_i in physics.observation_distribution(successor, joint, "i"):
                            next_i = None if b_i is None else frozenset(
                                level_one_update(physics, "i", "j", dict(b_i), a_i, o_i).items())
                            for o_j, q_j in physics.observation_distribution(successor, joint, "j"):
                                next_j = None if b_j is None else frozenset(
                                    level_one_update(physics, "j", "i", dict(b_j), a_j, o_j).items())
                                key = (successor, next_i, next_j)
                                following[key] = following.get(key, 0.0) + weight * q_i * q_j
        nodes = following
        curve.append(dict(cumulative))
    return curve


@pytest.mark.parametrize(
    "physics, levels, steps, horizon",
    [
        (JointTigerModel(together_safe=40.0), {"i": 1, "j": 1}, 3, 2),
        (TigerModel(), {"i": 1, "j": 0}, 3, 3),
        (UAVModel(rows=2, cols=2), {"i": 1, "j": 1}, 3, 2),
        (UAVModel(rows=2, cols=2), {"i": 0, "j": 1}, 3, 2),
    ],
    ids=["tiger-coordination-1v1", "tiger-1v0", "uav-1v1", "uav-0v1"],
)
def test_the_exact_pair_curve_equals_brute_force_enumeration(physics, levels, steps, horizon):
    expected = brute_force_pair_curve(physics, levels, steps, horizon)
    rows = list(exact_pair_curve(physics, levels, steps, horizon, GAMMA))
    assert [row["step"] for row in rows] == list(range(1, steps + 1))
    for row, cumulative in zip(rows, expected):
        assert row["cum_reward_i"] == pytest.approx(cumulative["i"], abs=1e-9)
        assert row["cum_reward_j"] == pytest.approx(cumulative["j"], abs=1e-9)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_holm_on_a_hand_example_and_against_its_definition():
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert list(holm([0.5, 0.9])) == pytest.approx([1.0, 1.0])
    p = [0.2, 0.001, 0.04, 0.04, 0.6, 0.0004]
    order = sorted(range(len(p)), key=lambda k: p[k])
    expected, running = [0.0] * len(p), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        expected[index] = running
    assert list(holm(p)) == pytest.approx(expected)
    assert all(adjusted >= raw for adjusted, raw in zip(holm(p), p))


@pytest.mark.parametrize("successes, n", [(0, 30), (30, 30), (1, 30), (17, 30), (99, 100), (3, 7)])
def test_clopper_pearson_bounds_solve_the_binomial_tail_equations(successes, n):
    """The lower bound L has P(X >= k | L) = 0.025 and the upper bound U has P(X <= k | U) = 0.025."""
    low, high = clopper_pearson(successes, n)
    assert 0.0 <= low <= successes / n <= high <= 1.0
    if successes > 0:
        assert stats.binom.sf(successes - 1, n, low) == pytest.approx(0.025, abs=1e-9)
    else:
        assert low == 0.0
    if successes < n:
        assert stats.binom.cdf(successes, n, high) == pytest.approx(0.025, abs=1e-9)
    else:
        assert high == 1.0


def test_clopper_pearson_covers_at_least_its_level_for_every_true_rate():
    """Exact coverage: for each true rate, the binomial mass of the outcomes whose interval contains it."""
    n = 25
    intervals = [clopper_pearson(k, n) for k in range(n + 1)]
    for rate in [0.01, 0.07, 0.2, 0.5, 0.63, 0.9, 0.995]:
        coverage = sum(stats.binom.pmf(k, n, rate) for k, (low, high) in enumerate(intervals) if low <= rate <= high)
        assert coverage >= 0.95


def test_thirty_of_thirty_only_rules_out_rates_below_0_884():
    low, high = clopper_pearson(30, 30)
    assert high == 1.0 and low == pytest.approx(0.025 ** (1 / 30), abs=1e-12)  # L^30 = 0.025


def test_paired_difference_in_its_degenerate_cases():
    index = pd.RangeIndex(6)
    same = pd.Series([3.0, -1.0, 4.0, 1.0, 5.0, 9.0], index=index)
    assert paired_difference(same, same)["p"] == 1.0  # no difference at all
    shifted = paired_difference(same + 2.5, same)
    # Every trial differs by the same nonzero amount: the t statistic is infinite, so p is 0, not 1.
    assert shifted["mean"] == 2.5 and shifted["low"] == shifted["high"] == 2.5 and shifted["p"] == 0.0
    single = paired_difference(same.iloc[:1] + 1.0, same.iloc[:1])
    assert single["n"] == 1 and math.isnan(single["low"]) and single["p"] == 1.0  # one pair tests nothing
    partial = paired_difference(same.iloc[:4], same.iloc[2:] + 1.0)
    assert partial["n"] == 2  # only the trials both series hold


def test_paired_difference_matches_the_t_test_and_the_bootstrap_is_reproducible():
    a = pd.Series([1.0, 4.0, 2.5, 7.0, 3.0, 6.5, 5.0])
    b = pd.Series([0.5, 3.0, 3.5, 5.0, 1.0, 6.0, 2.0])
    result = paired_difference(a, b)
    assert result["mean"] == pytest.approx((a - b).mean())
    assert result["p"] == pytest.approx(stats.ttest_rel(a, b).pvalue)
    assert (result["low"], result["high"]) == bootstrap_interval((a - b).to_numpy())
    assert result["low"] < result["mean"] < result["high"]
    # A percentile interval of the mean never leaves the range of the data.
    assert (a - b).min() <= result["low"] and result["high"] <= (a - b).max()
    assert list(itertools.islice(iter(bootstrap_interval([1.0])), 2)) != [1.0, 1.0]  # fewer than two values: NaN
