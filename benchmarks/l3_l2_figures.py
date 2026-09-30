"""Figures for the L3-vs-L2 Tiger study from a completed (or pilot) batch directory.

Sunbursts average each nested path's joint mass over trials at each step: the
expected belief hierarchy, whose spread across trials is shown separately by the
level-posterior bands. Physical tiger beliefs are omitted from the rings; their
trial average is uninformative (about .5). Cumulative rewards are undiscounted.
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
from matplotlib.patches import Wedge  # noqa: E402
from scipy import stats  # noqa: E402

GRID_STEPS = (0, 1, 2, 3, 5, 10, 15, 20)
# Agent hue families; darker shades for higher levels. Colors are fixed per path
# identity so slices keep their meaning across panels.
PALETTE = {
    "i": {0: "#fdd9b5", 1: "#f9a65a", 2: "#e8702a", 3: "#b04a0f"},
    "j": {0: "#c7dcef", 1: "#7fb2dc", 2: "#3b7fc0", 3: "#1d4f8a"},
}
AGENT_NAME = {"i": "agent i (L3)", "j": "agent j (L2)"}


def load(directory):
    path = directory / "batch_results.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    frames = [
        pd.read_csv(p, keep_default_na=False, na_values=[""])
        for p in sorted((directory / "trials").glob("*.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No trial CSVs in {directory}")
    return pd.concat(frames, ignore_index=True).sort_values(["trial", "step"])


def mean_hierarchy(df, agent):
    """Average joint mass per path id at each step; assert conservation invariants."""
    out = {}
    trials = df["trial"].nunique()
    for step, group in df.groupby("step"):
        totals = {}
        for text in group[f"hierarchy_{agent}"]:
            for node, value in json.loads(text).items():
                totals[node] = totals.get(node, 0.0) + value
        mean = {node: value / trials for node, value in totals.items()}
        check_hierarchy(mean)
        out[int(step)] = mean
    return out


def children(tree, node):
    return [n for n in tree if n.rsplit("/", 1)[0] == node and n != node]


def check_hierarchy(tree):
    roots = [n for n in tree if "/" not in n]
    assert len(roots) == 1 and math.isclose(tree[roots[0]], 1, abs_tol=1e-9), roots
    for node, value in tree.items():
        kids = children(tree, node)
        if kids:
            total = math.fsum(tree[k] for k in kids)
            # Intentional nodes partition their mass over the next opponent level.
            assert math.isclose(total, value, rel_tol=1e-9, abs_tol=1e-12), (node, total, value)


def _parse(node):
    agent, level = node.rsplit("/", 1)[-1].split("_L")
    return agent, int(level)


def draw_sunburst(ax, tree, title):
    root = next(n for n in tree if "/" not in n)
    width = 1.0
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-4.3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    agent, level = _parse(root)
    ax.add_patch(Wedge((0, 0), width, 0, 360, facecolor=PALETTE[agent][level], edgecolor="white"))
    ax.text(0, 0, f"{agent}\nL{level}", ha="center", va="center", fontsize=8, weight="bold")

    def ring(node, start, depth):
        angle = start
        for kid in sorted(children(tree, node)):
            value = tree[kid]
            sweep = 360.0 * value
            if sweep <= 0:
                continue
            agent, level = _parse(kid)
            ax.add_patch(
                Wedge(
                    (0, 0),
                    width * (depth + 1),
                    angle,
                    angle + sweep,
                    width=width,
                    facecolor=PALETTE[agent][level],
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            if sweep >= 14:
                mid = math.radians(angle + sweep / 2)
                radius = width * (depth + 0.5)
                parent_mass = tree[node]
                ax.text(
                    radius * math.cos(mid),
                    radius * math.sin(mid),
                    f"{agent} L{level}\n{value / parent_mass:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
            ring(kid, angle, depth + 1)
            angle += sweep

    ring(root, 90.0, 1)
    ax.set_title(title, fontsize=9)


def sunburst_grid(hierarchy, agent, n_trials, out):
    steps = [s for s in GRID_STEPS if s in hierarchy]
    cols = 4
    rows = math.ceil(len(steps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows))
    for ax in np.ravel(axes)[len(steps) :]:
        ax.axis("off")
    for ax, step in zip(np.ravel(axes), steps):
        draw_sunburst(ax, hierarchy[step], f"t = {step}")
    fig.suptitle(
        f"Opponent modelling of {AGENT_NAME[agent]}: mean nested level belief "
        f"over {n_trials} trials\n(inner to outer: whom it models; labels are "
        "probabilities conditional on the parent slice)",
        fontsize=10,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"sunburst_grid_{agent}.{suffix}", dpi=200)
    plt.close(fig)


def sunburst_html(hierarchy, agent, n_trials, out):
    from utils.plotting import plot_episode_sunburst_slider

    snapshots = {}
    for step, tree in hierarchy.items():
        ids = sorted(tree, key=lambda n: (n.count("/"), n))
        snapshots[step] = dict(
            ids=ids,
            labels=[" ".join(_parse(n)[0:1]) + f" L{_parse(n)[1]}" for n in ids],
            parents=[n.rsplit("/", 1)[0] if "/" in n else "" for n in ids],
            values=[tree[n] for n in ids],
            levels=[_parse(n)[1] for n in ids],
            hover_texts=[
                f"{n}<br>joint mass {tree[n]:.4f}"
                + (
                    f"<br>conditional on parent {tree[n] / tree[n.rsplit('/', 1)[0]]:.1%}"
                    if "/" in n
                    else ""
                )
                for n in ids
            ],
        )
    plot_episode_sunburst_slider(
        snapshots,
        title_prefix=f"{AGENT_NAME[agent]} mean opponent model over {n_trials} trials",
        save_dir=str(out),
        filename=f"sunburst_slider_{agent}",
    )


def t_band(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2 or values.std(ddof=1) == 0:
        return mean, mean, mean
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n)
    return mean, mean - half, mean + half


def bootstrap_band(values, rng, draws=5000):
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def cumulative_rewards(df, out):
    rng = np.random.default_rng(0)
    rows = []
    for step, group in df.groupby("step"):
        for agent in ("i", "j"):
            values = group[f"cum_reward_{agent}"].to_numpy()
            mean, low, high = t_band(values)
            b_low, b_high = bootstrap_band(values, rng)
            rows.append(
                dict(
                    step=int(step),
                    agent=agent,
                    n=len(values),
                    mean=mean,
                    t_low=low,
                    t_high=high,
                    boot_low=b_low,
                    boot_high=b_high,
                    sd=values.std(ddof=1) if len(values) > 1 else 0.0,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "cumulative_reward_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for agent, color in (("i", PALETTE["i"][2]), ("j", PALETTE["j"][2])):
        part = table[table.agent == agent]
        ax.plot(part.step, part["mean"], marker="o", ms=3, color=color, label=AGENT_NAME[agent])
        ax.fill_between(part.step, part.t_low, part.t_high, color=color, alpha=0.2)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("time step")
    ax.set_ylabel("cumulative reward (undiscounted)")
    n = int(table.n.max())
    ax.set_title(f"Cumulative reward, mean with 95% t-interval (N = {n} trials)")
    ax.set_xticks(range(0, int(table.step.max()) + 1, 2))
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"cumulative_rewards.{suffix}", dpi=200)
    plt.close(fig)
    return table


def level_posteriors(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (agent, other) in zip(axes, (("i", "j"), ("j", "i"))):
        columns = sorted(
            c for c in df.columns if c.startswith("prob_l") and c.endswith(f"_{other}")
        )
        for column in columns:
            level = int(column[len("prob_l") : -2])
            color = PALETTE[other][level]
            bands = [t_band(g[column].to_numpy()) for _, g in df.groupby("step")]
            steps = sorted(df.step.unique())
            ax.plot(steps, [b[0] for b in bands], color=color, lw=2, label=f"{other} is L{level}")
            ax.fill_between(
                steps, [b[1] for b in bands], [b[2] for b in bands], color=color, alpha=0.25
            )
        ax.set_title(f"{AGENT_NAME[agent]}: belief about {other}'s level")
        ax.set_xlabel("time step")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("posterior probability (mean, 95% t-interval)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"level_posteriors.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.run)
    n_trials = df["trial"].nunique()
    for agent in ("i", "j"):
        hierarchy = mean_hierarchy(df, agent)
        (out / f"mean_hierarchy_{agent}.json").write_text(json.dumps(hierarchy, indent=2))
        sunburst_grid(hierarchy, agent, n_trials, out)
        sunburst_html(hierarchy, agent, n_trials, out)
    table = cumulative_rewards(df, out)
    level_posteriors(df, out)
    final = table[table.step == table.step.max()]
    print(final.to_string(index=False))
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
