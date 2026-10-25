"""Long-format metrics from episode records, computed once and reused by every figure.

Input is the per-step frame written by ``experiments.batch_runner`` (one row per
trial and step, step 0 being the prior). Every table is long format, so a
figure or test never parses column names itself.

Conventions:
- Cumulative rewards keep terminal padding rows, so trials that ended early
  stay in the average with their final return; averaging survivors only would
  bias returns upward.
- Costs are reported per decision for both agents, split into planning and
  belief update, in process CPU seconds (wall seconds are kept beside them).
- Level posteriors are recorded only while an episode is running; each step
  reports how many trials contribute.
- A prediction **miss** is a decision at which the agent gave the opponent's
  actual action probability below ``MISS_PROBABILITY`` (zero up to rounding).
  Bounded beliefs cause them by dropping the branch that actually happened
  (RESULTS 12.26, 12.29); the level-posterior distance cannot see them, so
  every bounded-belief run reports the miss rate.
- Confidence intervals resample whole trials (cluster bootstrap). Steps within
  a trial are correlated, so treating rows as independent would understate
  uncertainty.
"""

import json
import math
import re

import numpy as np
import pandas as pd

AGENTS = ("i", "j")
_LEVEL_COLUMN = re.compile(r"^prob_l(\d+)_([A-Za-z0-9]+)$")
MISS_PROBABILITY = 1e-9


def rewards(frame: pd.DataFrame) -> pd.DataFrame:
    """(trial, step, agent, reward, cumulative_reward), terminal padding included."""
    parts = [
        frame[["trial", "step"]].assign(
            agent=agent,
            reward=frame[f"reward_{agent}"],
            cumulative_reward=frame[f"cum_reward_{agent}"],
        )
        for agent in AGENTS
    ]
    return pd.concat(parts, ignore_index=True)


def costs(frame: pd.DataFrame) -> pd.DataFrame:
    """(trial, step, agent, component, cpu_seconds, wall_seconds) for decision steps."""
    decisions = frame[frame["step"] > 0]
    parts = [
        decisions[["trial", "step"]].assign(
            agent=agent,
            component=component,
            cpu_seconds=decisions[f"{component}_cpu_{agent}"],
            wall_seconds=decisions[f"{component}_wall_{agent}"],
        )
        for agent in AGENTS
        for component in ("planning", "update")
    ]
    return pd.concat(parts, ignore_index=True)


def level_posteriors(frame: pd.DataFrame) -> pd.DataFrame:
    """(trial, step, opponent, level, probability) from the prob_l{level}_{opponent} columns."""
    parts = []
    for column in frame.columns:
        match = _LEVEL_COLUMN.match(column)
        if match:
            recorded = frame[frame[column].notna()]
            parts.append(
                recorded[["trial", "step"]].assign(
                    opponent=match.group(2), level=int(match.group(1)), probability=recorded[column]
                )
            )
    if not parts:
        raise ValueError("The frame records no level posteriors")
    return pd.concat(parts, ignore_index=True)


def predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """(trial, step, agent, realized_probability, log_loss, miss) for each recorded prediction.

    ``realized_probability`` is what the agent's prediction (``prediction_{agent}``,
    recorded before the step's actions) gave the opponent's action at that step;
    ``log_loss`` is its negative log, floored at ``MISS_PROBABILITY``. Level-0
    agents and padding rows record no prediction and are left out.
    """
    parts = []
    decisions = frame[frame["step"] > 0]
    for agent in AGENTS:
        other = "j" if agent == "i" else "i"
        column = f"prediction_{agent}"
        if column not in decisions:
            continue
        for row in decisions[["trial", "step", column, f"action_{other}"]].itertuples(index=False):
            predicted = json.loads(row[2]) if isinstance(row[2], str) else None
            if not predicted:
                continue
            probability = predicted.get(row[3], 0.0)
            parts.append(
                {
                    "trial": row[0],
                    "step": row[1],
                    "agent": agent,
                    "realized_probability": probability,
                    "log_loss": -math.log(max(probability, MISS_PROBABILITY)),
                    "miss": probability < MISS_PROBABILITY,
                }
            )
    if not parts:
        raise ValueError("The frame records no predictions")
    return pd.DataFrame(parts)


def prediction_misses(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (trial, agent): misses, any_miss, the minimum realized probability, mean log-loss."""
    table = predictions(frame)
    return (
        table.groupby(["trial", "agent"])
        .agg(
            misses=("miss", "sum"),
            any_miss=("miss", "any"),
            min_realized_probability=("realized_probability", "min"),
            log_loss=("log_loss", "mean"),
        )
        .reset_index()
    )


def bootstrap_mean(table, value, by, draws=2000, seed=0, level=0.95):
    """Mean of ``value`` per group with a trial-clustered bootstrap interval.

    Returns columns: by..., n_trials, mean, low, high.
    """
    rng = np.random.default_rng(seed)
    trials = np.sort(table["trial"].unique())
    rows = []
    for key, group in table.groupby(list(by)):
        per_trial = group.groupby("trial")[value].mean().reindex(trials).dropna()
        values = per_trial.to_numpy()
        resampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(resampled, [(1 - level) / 2, (1 + level) / 2])
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, key)),
                "n_trials": len(values),
                "mean": values.mean(),
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)
