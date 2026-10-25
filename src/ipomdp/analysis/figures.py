"""Figures, rendered from the tables in ``metrics`` and the nodes in ``hierarchy``.

Static figures use matplotlib (PDF/PNG). Sunbursts are interactive plotly HTML,
because nested hierarchies are read by hovering and drilling down; they draw
every nesting level (no ring limit) and show each node's content and, when an
exact hierarchy is supplied, its error.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from ipomdp.analysis import metrics  # noqa: E402

LEVEL_COLORS = {0: "#94a3b8", 1: "#38bdf8", 2: "#818cf8", 3: "#c084fc", 4: "#f472b6"}
AGENT_COLORS = {"i": "#e8702a", "j": "#3b7fc0"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cumulative_rewards(frame, path, labels=None):
    """Mean cumulative reward of both agents per step, with trial-bootstrap 95% bands."""
    table = metrics.bootstrap_mean(metrics.rewards(frame), "cumulative_reward", ("agent", "step"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for agent, part in table.groupby("agent"):
        label = (labels or {}).get(agent, f"agent {agent}")
        ax.plot(part["step"], part["mean"], color=AGENT_COLORS[agent], label=label)
        ax.fill_between(
            part["step"], part["low"], part["high"], color=AGENT_COLORS[agent], alpha=0.2
        )
    n = int(table["n_trials"].min())
    ax.set(
        xlabel="step (0 = prior)",
        ylabel="cumulative reward (undiscounted)",
        title=f"n = {n} trials",
    )
    ax.legend()
    _save(fig, path)


def decision_costs(frame, path, labels=None):
    """Mean CPU seconds per decision for both agents, split into planning and belief update."""
    table = metrics.bootstrap_mean(
        metrics.costs(frame), "cpu_seconds", ("agent", "component", "step")
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (agent, component), part in table.groupby(["agent", "component"]):
        label = f"{(labels or {}).get(agent, f'agent {agent}')}: {component}"
        style = "-" if component == "planning" else "--"
        ax.plot(part["step"], part["mean"], style, color=AGENT_COLORS[agent], label=label)
    ax.set(xlabel="step", ylabel="CPU seconds per decision", yscale="log")
    ax.legend(fontsize=8)
    _save(fig, path)


def level_posteriors(frame, path, opponent):
    """Mean posterior over the opponent's level per step (mean over trials, with bands)."""
    table = metrics.level_posteriors(frame)
    table = table[table["opponent"] == opponent]
    summary = metrics.bootstrap_mean(table, "probability", ("level", "step"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for level, part in summary.groupby("level"):
        color = LEVEL_COLORS.get(level, "#64748b")
        ax.plot(part["step"], part["mean"], color=color, label=f"P({opponent} is L{level})")
        ax.fill_between(part["step"], part["low"], part["high"], color=color, alpha=0.2)
    ax.set(
        xlabel="step (0 = prior)", ylabel="posterior probability (mean over trials)", ylim=(0, 1)
    )
    ax.legend()
    _save(fig, path)


def _hover(node, error):
    lines = [f"<b>{node.agent} (L{node.level})</b>", f"joint mass: {node.mass:.3f}"]
    if node.physical is not None:
        top = sorted(node.physical.items(), key=lambda kv: -kv[1])[:4]
        lines.append("believes: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    if error is not None and node.depth in error:
        row = error[node.depth]
        lines.append(
            f"depth-{node.depth} error: mass TV {row['mass_tv']:.3f}, physical TV {row['physical_tv']:.3f}"
        )
    return "<br>".join(lines)


def _sunburst_trace(nodes, paths, error=None):
    by_path = {node.path: node for node in nodes}
    values, hovers = [], []
    for path in paths:
        node = by_path.get(path)
        values.append(node.mass if node else 0.0)
        hovers.append(_hover(node, error) if node else "absent at this step")
    reference = {node.path: node for node in nodes}
    return go.Sunburst(
        ids=list(paths),
        labels=[path.rsplit("/", 1)[-1].replace("_", " ") for path in paths],
        parents=[path.rsplit("/", 1)[0] if "/" in path else "" for path in paths],
        values=values,
        branchvalues="total",
        hovertext=hovers,
        hoverinfo="text",
        sort=False,
        marker=dict(
            colors=[
                LEVEL_COLORS.get(reference[p].level if p in reference else 0, "#cbd5e1")
                for p in paths
            ],
            line=dict(color="#ffffff", width=1),
        ),
    )


def sunburst(nodes, path, title, error=None):
    """One nested hierarchy as an interactive sunburst (all depths)."""
    paths = [node.path for node in nodes]
    fig = go.Figure(_sunburst_trace(nodes, paths, error))
    fig.update_layout(title=title, margin=dict(t=50, l=10, r=10, b=10))
    fig.write_html(path)
    return fig


def sunburst_over_time(snapshots, path, title, errors=None):
    """Hierarchies per step with a slider; every path keeps a fixed angular position."""
    steps = sorted(snapshots)
    paths = list(dict.fromkeys(node.path for step in steps for node in snapshots[step]))
    frames = [
        go.Frame(
            data=[_sunburst_trace(snapshots[step], paths, (errors or {}).get(step))],
            name=f"step {step}",
        )
        for step in steps
    ]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        margin=dict(t=60, l=10, r=10, b=80),
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        label=str(step),
                        args=[[f"step {step}"], dict(mode="immediate")],
                    )
                    for step in steps
                ],
                currentvalue=dict(prefix="step "),
            )
        ],
    )
    fig.write_html(path)
    return fig
