"""Pre-registered endpoints of the experiment suites (``experiments.suite``).

Every comparison pairs trials: suite cells share trial ids, so the environment
and both agents' streams match across cells until actions diverge. Comparing i
with j inside one cell is not done: each agent's sensor luck repeats across
every cell of a seed-paired suite (HANDOFF 6).

- ``paired_difference``: mean of a minus b over common trials, a 95% trial
  bootstrap interval and a two-sided paired t-test;
- ``one_level_endpoint``: the level matrix's primary endpoint, the agent at
  k + 1 minus the same agent at k against the same opponent level, for both
  agents, Holm-corrected over the family (RESULTS 12.27);
- ``arm_differences``: each arm minus a reference arm (planner comparison, prior
  sensitivity), Holm-corrected;
- ``cell_table``: per-cell rewards, welfare, prediction loss and misses (with
  Clopper–Pearson intervals) and CPU per step.

Bootstrap draws use a fixed seed, so a report is reproducible from its data.
"""

import numpy as np
import pandas as pd
from scipy import stats

from ipomdp.analysis.metrics import AGENTS, prediction_misses
from ipomdp.analysis.statistics import clopper_pearson, holm

DRAWS = 4000


def final_rewards(frame, agent, column="cum_reward"):
    """Cumulative reward at the last step, indexed by trial.

    ``column="rb_cum_reward"`` reads the Rao-Blackwellized curve
    (``analysis.rao_blackwell``): the same estimand at lower variance.
    """
    last = frame[frame["step"] == frame["step"].max()]
    return last.set_index("trial")[f"{column}_{agent}"]


def bootstrap_interval(values, seed=0, draws=DRAWS, level=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(values, (draws, len(values))).mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_difference(a, b, seed=0):
    """a − b over the trials both series hold: n, mean, bootstrap low/high, paired-t p."""
    common = a.index.intersection(b.index)
    difference = (a[common] - b[common]).to_numpy(dtype=float)
    low, high = bootstrap_interval(difference, seed)
    if len(difference) > 1 and difference.std() > 0:
        p = float(stats.ttest_1samp(difference, 0.0).pvalue)
    else:
        p = 1.0
    mean = float(difference.mean()) if len(difference) else float("nan")
    return {"n": len(difference), "mean": mean, "low": low, "high": high, "p": p}


def one_level_endpoint(cells, column="cum_reward"):
    """Paired one-level gains over a level matrix {(level_i, level_j): frame}.

    Rows: agent, step ("k->k+1"), opponent level, n, mean, low, high, p, p_holm.
    The family is every comparison present. ``column`` as in ``final_rewards``.
    """
    rows = []
    for (li, lj), frame in sorted(cells.items()):
        for agent, own, opponent, lower in (
            ("i", li, lj, (li - 1, lj)),
            ("j", lj, li, (li, lj - 1)),
        ):
            if own < 1 or lower not in cells:
                continue
            difference = paired_difference(
                final_rewards(frame, agent, column), final_rewards(cells[lower], agent, column)
            )
            rows.append(
                {"agent": agent, "step": f"{own - 1}->{own}", "opponent": opponent, **difference}
            )
    return _with_holm(rows)


def arm_differences(cells, reference, agent="i"):
    """Each arm's final reward minus the ``reference`` arm's, paired, Holm over arms."""
    base = final_rewards(cells[reference], agent)
    rows = [
        {"arm": arm, "reference": reference, **paired_difference(final_rewards(frame, agent), base)}
        for arm, frame in sorted(cells.items())
        if arm != reference
    ]
    return _with_holm(rows)


def cell_table(cells):
    """One row per cell: rewards and welfare with bootstrap intervals, loss, misses, CPU."""
    rows = []
    for label, frame in sorted(cells.items(), key=lambda item: str(item[0])):
        row = {"cell": label, "n": int(frame["trial"].nunique())}
        i, j = final_rewards(frame, "i"), final_rewards(frame, "j")
        for name, values in (("reward_i", i), ("reward_j", j), ("welfare", i + j)):
            row[name] = float(values.mean())
            row[f"{name}_low"], row[f"{name}_high"] = bootstrap_interval(values)
        try:
            misses = prediction_misses(frame)
        except ValueError:  # neither agent predicts (both at level 0)
            misses = pd.DataFrame(columns=["agent", "any_miss", "log_loss"])
        for agent in AGENTS:
            own = misses[misses["agent"] == agent]
            if own.empty:
                row[f"log_loss_{agent}"] = row[f"miss_trials_{agent}"] = None
                continue
            hits = int(own["any_miss"].sum())
            row[f"log_loss_{agent}"] = float(own["log_loss"].mean())
            row[f"miss_trials_{agent}"] = hits
            row[f"miss_low_{agent}"], row[f"miss_high_{agent}"] = clopper_pearson(hits, len(own))
        decisions = frame[frame["step"] > 0]
        row["cpu_per_step"] = float(
            decisions["planning_cpu_i"].mean() + decisions["planning_cpu_j"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def joint_safe(frame):
    """Per trial: steps in which both agents open the same safe door (the coordination bonus).

    Tiger-specific: the tiger's side before a step is the previous row's state.
    """
    steps = frame[frame["step"] > 0]
    before = frame.set_index(["trial", "step"])["state"]
    side = pd.Series(
        [before.get((t, s - 1)) for t, s in zip(steps["trial"], steps["step"])], index=steps.index
    )
    safe = side.map({"TL": "OR", "TR": "OL"})
    hits = (steps["action_i"] == steps["action_j"]) & (steps["action_i"] == safe)
    return hits.groupby(steps["trial"]).sum()


def _with_holm(rows):
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table["p"].to_numpy())
    return table
