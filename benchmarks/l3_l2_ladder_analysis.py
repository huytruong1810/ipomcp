"""Error-versus-resources analysis for the L3-vs-L2 convergence ladder.

Two kinds of error, kept separate:
1. Exact on-history error. Measured only on steps where the exact shadow audit
   was active; coverage is reported beside every number and nothing is
   extrapolated to unaudited steps.
2. Outcome convergence. Cumulative rewards and mean level hierarchies of each
   rung are compared with the highest rung, paired by trial id (the environment
   streams match). This is convergence evidence, not exact error.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from l3_l2_convergence import RUNGS  # noqa: E402
from l3_l2_figures import load, mean_hierarchy, t_band  # noqa: E402

AGENTS = ("i", "j")


def _metric_columns(df, agent):
    names = [
        f"exact_regret_{agent}",
        f"search_regret_{agent}",
        f"policy_tv_{agent}",
    ]
    names += sorted(
        c
        for c in df.columns
        if c.startswith(("tv_depth", "phys_err_depth"))
        and c.endswith(f"_{agent}")
        and c != f"tv_depth0_{agent}"  # the root always has mass one
    )
    return [c for c in names if c in df.columns]


def on_history(df, rung):
    """Per step and pooled: mean with 95% t-interval and audited-trial coverage."""
    rows = []
    trials = df.trial.nunique()
    for agent in AGENTS:
        for column in _metric_columns(df, agent):
            # Regrets come from the decision audit; belief metrics from the belief audit.
            flag = f"{'decision' if 'regret' in column else 'belief'}_audited_{agent}"
            audited = df[df[flag].astype(str) == "True"]
            for step, group in [("pooled", audited[audited.step > 0])] + list(
                audited.groupby("step")
            ):
                values = group[column].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                mean, low, high = t_band(values)
                rows.append(
                    dict(
                        rung=rung,
                        agent=agent,
                        metric=column.rsplit("_", 1)[0],
                        step=step,
                        n=len(values),
                        coverage=len(values) / trials if step != "pooled" else math.nan,
                        mean=mean,
                        low=low,
                        high=high,
                        max=values.max(),
                        share_positive=float((values > 1e-9).mean())
                        if "regret" in column
                        else math.nan,
                    )
                )
    return pd.DataFrame(rows)


def hierarchy_tv(a, b):
    return 0.5 * math.fsum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def outcomes(frames, top):
    rows = []
    reference = frames[top]
    reference_h = {agent: mean_hierarchy(reference, agent) for agent in AGENTS}
    for rung, df in frames.items():
        hier = {agent: mean_hierarchy(df, agent) for agent in AGENTS}
        merged = df.merge(reference, on=["trial", "step"], suffixes=("", "_top"))
        for step, group in merged.groupby("step"):
            for agent in AGENTS:
                diff = (group[f"cum_reward_{agent}"] - group[f"cum_reward_{agent}_top"]).to_numpy()
                mean, low, high = t_band(diff)
                rows.append(
                    dict(
                        rung=rung,
                        agent=agent,
                        step=int(step),
                        mean_cum_reward=group[f"cum_reward_{agent}"].mean(),
                        diff_to_top=mean,
                        diff_low=low,
                        diff_high=high,
                        hierarchy_tv_to_top=hierarchy_tv(
                            hier[agent][step], reference_h[agent][step]
                        ),
                    )
                )
    return pd.DataFrame(rows)


def plot_error_vs_resources(history, out):
    pooled = history[history.step == "pooled"]
    metrics = [
        m
        for m in (
            "exact_regret",
            "search_regret",
            "policy_tv",
            "tv_depth1",
            "tv_depth2",
            "tv_depth3",
        )
        if m in set(pooled.metric)
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.4), squeeze=False)
    order = sorted(RUNGS)
    for ax, metric in zip(axes[0], metrics):
        for agent, color in (("i", "#e8702a"), ("j", "#3b7fc0")):
            part = (
                pooled[(pooled.metric == metric) & (pooled.agent == agent)]
                .set_index("rung")
                .reindex(order)
            )
            x = np.arange(len(order))
            ax.errorbar(
                x,
                part["mean"],
                yerr=[part["mean"] - part["low"], part["high"] - part["mean"]],
                marker="o",
                color=color,
                capsize=3,
                label=f"agent {agent}",
            )
        ax.set_xticks(range(len(order)), order)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("resource rung")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        "Exact on-history error vs I-POMCP resources (audited steps only; 95% t-intervals)",
        fontsize=10,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"error_vs_resources.{suffix}", dpi=200)
    plt.close(fig)


def plot_outcomes(frames, outcome, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(frames)))
    for color, (rung, df) in zip(colors, sorted(frames.items())):
        for ax, agent, style in ((axes[0], "i", "-"), (axes[0], "j", "--")):
            means = df.groupby("step")[f"cum_reward_{agent}"].mean()
            ax.plot(means.index, means.values, style, color=color, label=f"{rung} {agent}")
        part = outcome[outcome.rung == rung]
        for agent, style in (("i", "-"), ("j", "--")):
            p = part[part.agent == agent]
            axes[1].plot(p.step, p.diff_to_top, style, color=color, label=f"{rung} {agent}")
            axes[2].plot(p.step, p.hierarchy_tv_to_top, style, color=color, label=f"{rung} {agent}")
    axes[0].set_title("mean cumulative reward (solid i, dashed j)")
    axes[1].set_title("cumulative reward minus top rung (paired)")
    axes[2].set_title("mean-hierarchy TV to top rung")
    for ax in axes:
        ax.set_xlabel("time step")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"outcome_convergence.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="Rung directories named r1..r4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {run.name: load(run) for run in args.runs}
    history = pd.concat([on_history(df, rung) for rung, df in frames.items()], ignore_index=True)
    history.to_csv(args.output / "on_history_errors.csv", index=False)
    outcome = outcomes(frames, max(frames))
    outcome.to_csv(args.output / "outcome_convergence.csv", index=False)
    plot_error_vs_resources(history, args.output)
    plot_outcomes(frames, outcome, args.output)
    coverage = {
        rung: {
            agent: {
                "decision_steps": int((df.step > 0).sum()),
                **{
                    f"{kind}_audited": int(
                        (df[df.step > 0][f"{kind}_audited_{agent}"].astype(str) == "True").sum()
                    )
                    for kind in ("decision", "belief")
                },
                **{
                    f"{kind}_stops": df.groupby("trial")[f"{kind}_stop_{agent}"]
                    .last()
                    .value_counts()
                    .to_dict()
                    for kind in ("decision", "belief")
                },
            }
            for agent in AGENTS
        }
        for rung, df in frames.items()
    }
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2))
    pooled = history[history.step == "pooled"]
    print(
        pooled[
            ["rung", "agent", "metric", "n", "mean", "low", "high", "max", "share_positive"]
        ].to_string(index=False)
    )
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
