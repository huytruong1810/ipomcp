"""Aggregation invariants for the L3-vs-L2 sunburst and reward figures."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import l3_l2_figures as figures  # noqa: E402


def frame(trees, rewards):
    return pd.DataFrame(
        [
            dict(trial=t, step=0, hierarchy_i=json.dumps(tree), cum_reward_i=r, cum_reward_j=-r)
            for t, (tree, r) in enumerate(zip(trees, rewards))
        ]
    )


def test_mean_hierarchy_averages_paths_and_keeps_partitions():
    a = {"i_L3": 1.0, "i_L3/j_L0": 0.2, "i_L3/j_L2": 0.8, "i_L3/j_L2/i_L1": 0.8}
    b = {"i_L3": 1.0, "i_L3/j_L0": 1.0}
    mean = figures.mean_hierarchy(frame([a, b], [0, 0]), "i")[0]
    assert mean == pytest.approx(
        {"i_L3": 1.0, "i_L3/j_L0": 0.6, "i_L3/j_L2": 0.4, "i_L3/j_L2/i_L1": 0.4}
    )


def test_broken_partition_is_rejected():
    bad = {"i_L3": 1.0, "i_L3/j_L0": 0.5, "i_L3/j_L1": 0.4}
    with pytest.raises(AssertionError):
        figures.check_hierarchy(bad)


def test_reward_interval_contains_mean():
    mean, low, high = figures.t_band([1.0, 2.0, 3.0, 4.0])
    assert low < mean == 2.5 < high
    assert figures.t_band([5.0, 5.0]) == (5.0, 5.0, 5.0)
