# Absolute Path: <project_root>/utils/plotting.py

"""
plotting.py — Interactive Plotly visualizations for debugging and analysis.

DESIGN DECISION RECORD (Phase 4 Overhaul):
------------------------------------------
1. PANDAS DICT SERIALIZATION FIX (`_plot_action_values`):
   When batch runners save to CSV, Pandas serializes internal dictionaries (like Q-values)
   into strings. If `plot_all_metrics` is called on a loaded CSV, `isinstance(val, dict)`
   silently fails. We introduced `ast.literal_eval` to safely cast strings back to
   Python dictionaries, preserving the action-value visualization pipeline.

2. PIPELINE HARDENING:
   Added strict `df.empty` and `status` column checks. If a batch run fails due to
   domain configuration errors, the plotter will safely abort instead of throwing
   fatal KeyErrors.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import ast
from typing import Dict, Any, Optional, List, Tuple


def _maybe_save_and_show(fig, filename: str, save_dir: str = None):
    """Helper to save interactive HTML plots if a directory is provided."""
    if save_dir:
        out_path = os.path.join(save_dir, f"{filename}.html")
        fig.write_html(out_path)
    else:
        fig.show()


def plot_all_metrics(df: pd.DataFrame,
                     agent_labels=None,
                     title_prefix: str = "",
                     save_dir: str = None):
    if df.empty or 'status' not in df.columns:
        print(f"[{title_prefix}] Dataframe is empty or missing data. Skipping interactive plots.")
        return

    if agent_labels is None:
        agent_labels = {'i': 'Agent I', 'j': 'Agent J'}

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Generating Interactive Debug Plots...")
    _plot_comparative_reward(df, agent_labels, title_prefix, save_dir)
    _plot_reward_vs_time(df, agent_labels, title_prefix, save_dir)
    _plot_belief_health(df, agent_labels, title_prefix, save_dir)
    _plot_action_values(df, agent_labels, title_prefix, save_dir)
    _plot_survival(df, title_prefix, save_dir)

    # Opponent level belief plots
    level_cols = [c for c in df.columns if c.startswith('prob_l')]
    if level_cols:
        _plot_opponent_level_belief(df, title_prefix, save_dir)

    # Domain specific plots
    if 'distance' in df.columns:
        _plot_distance(df, title_prefix, save_dir)
    if 'distance_chebyshev' in df.columns:
        _plot_distance(df, title_prefix, save_dir, col='distance_chebyshev')
    if 'distance_manhattan' in df.columns:
        _plot_distance(df, title_prefix, save_dir, col='distance_manhattan')


LEVEL_COLORS = {
    0: "#94a3b8",  # Slate gray (Level 0 sub-intentional)
    1: "#38bdf8",  # Sky blue (Level 1)
    2: "#818cf8",  # Indigo (Level 2)
    3: "#c084fc",  # Purple (Level 3)
    4: "#f472b6",  # Pink (Level 4)
}


def _plot_opponent_level_belief(df, prefix, save_dir):
    df_active = df[df['status'] == 'Active']
    if df_active.empty: return

    level_cols = [c for c in df.columns if c.startswith('prob_l')]
    if not level_cols: return

    grouped = df_active.groupby('step')[level_cols].mean().reset_index()

    # 1. Multi-line progression plot
    fig = go.Figure()
    for col in sorted(level_cols):
        parts = col.split('_')
        lvl_num = int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].isdigit() else 0
        lvl_label = f"P(Opponent Level-{lvl_num})"
        color = LEVEL_COLORS.get(lvl_num, "#64748b")
        fig.add_trace(go.Scatter(x=grouped['step'], y=grouped[col], mode='lines+markers',
                                 name=lvl_label, line=dict(color=color, width=2.5),
                                 marker=dict(size=7)))

    fig.update_layout(title=f"{prefix} - Opponent Level Belief Progression over Time",
                      xaxis_title="Step (t=0 is Initial Prior)", yaxis_title="Posterior Probability",
                      yaxis=dict(range=[0, 1.05]), hovermode="x unified")
    _maybe_save_and_show(fig, "opponent_level_belief", save_dir)

    # 2. Stacked area progression plot
    fig_area = go.Figure()
    for col in sorted(level_cols):
        parts = col.split('_')
        lvl_num = int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].isdigit() else 0
        lvl_label = f"Level-{lvl_num}"
        color = LEVEL_COLORS.get(lvl_num, "#64748b")
        fig_area.add_trace(go.Scatter(
            x=grouped['step'], y=grouped[col],
            mode='lines',
            stackgroup='one',
            groupnorm='fraction',
            name=lvl_label,
            line=dict(width=0.5, color=color),
            fillcolor=color
        ))

    fig_area.update_layout(title=f"{prefix} - Opponent Level Belief Composition (Stacked Area)",
                           xaxis_title="Step (t=0 is Initial Prior)", yaxis_title="Belief Probability Fraction",
                           yaxis=dict(range=[0, 1.0]), hovermode="x unified")
    _maybe_save_and_show(fig_area, "opponent_level_belief_stacked", save_dir)


def plot_nested_belief_sunburst(nested_data: Dict[str, Any], title: str = "Nested Mental Model Belief Hierarchy",
                                save_dir: Optional[str] = None, filename: str = "nested_belief_sunburst") -> go.Figure:
    """Generates an interactive Plotly Sunburst diagram of the multi-level nested belief tree."""
    if not nested_data or "ids" not in nested_data or len(nested_data["ids"]) == 0:
        return go.Figure()

    colors = [LEVEL_COLORS.get(lvl, "#cbd5e1") for lvl in nested_data.get("levels", [0] * len(nested_data["ids"]))]

    fig = go.Figure(go.Sunburst(
        ids=nested_data["ids"],
        labels=nested_data["labels"],
        parents=nested_data["parents"],
        values=nested_data["values"],
        branchvalues="total",
        hovertext=nested_data.get("hover_texts"),
        hoverinfo="text",
        marker=dict(colors=colors, line=dict(color="#ffffff", width=1.5)),
        insidetextorientation="radial",
        sort=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial, sans-serif")),
        margin=dict(t=50, l=10, r=10, b=10),
        paper_bgcolor="#ffffff"
    )

    if save_dir:
        _maybe_save_and_show(fig, filename, save_dir)
    return fig


def plot_episode_sunburst_slider(snapshots_by_step: Dict[int, Dict[str, Any]],
                                 title_prefix: str = "Episode Progression",
                                 save_dir: Optional[str] = None,
                                 filename: str = "nested_belief_sunburst_animated") -> go.Figure:
    """
    Generates an interactive Plotly Sunburst visualization equipped with a timestep scrubber slider,
    allowing frame-by-frame inspection of nested mental model belief redistribution across an episode.
    Maintains fixed slice angular positions across the entire animation sequence (sort=False with canonical hierarchy).
    """
    sorted_steps = sorted(snapshots_by_step.keys())
    if not sorted_steps:
        return go.Figure()

    # Collect the global canonical node hierarchy across all timesteps to keep slice positions constant
    canonical_order = []
    canonical_meta = {}
    for step in sorted_steps:
        snap = snapshots_by_step[step]
        if not snap or "ids" not in snap:
            continue
        for idx, nid in enumerate(snap["ids"]):
            if nid not in canonical_meta:
                canonical_order.append(nid)
                canonical_meta[nid] = {
                    "label": snap["labels"][idx],
                    "parent": snap["parents"][idx],
                    "level": snap.get("levels", [0])[idx] if "levels" in snap and idx < len(snap["levels"]) else 0,
                }

    if not canonical_order:
        return go.Figure()

    canonical_colors = [LEVEL_COLORS.get(canonical_meta[nid]["level"], "#cbd5e1") for nid in canonical_order]

    def _extract_aligned_frame_data(snap: Optional[Dict[str, Any]]) -> Tuple[List[float], List[str]]:
        step_id_map = {nid: i for i, nid in enumerate(snap.get("ids", []))} if snap else {}
        vals = []
        hovers = []
        for nid in canonical_order:
            if nid in step_id_map:
                orig_idx = step_id_map[nid]
                vals.append(float(snap["values"][orig_idx]))
                hovers.append(snap["hover_texts"][orig_idx] if "hover_texts" in snap and orig_idx < len(snap["hover_texts"]) else "")
            else:
                meta = canonical_meta[nid]
                vals.append(0.0)
                hovers.append(f"<b>{meta['label']}</b><br>Conditional Belief: 0.0%<br>Joint Mass: 0.000<br>Particles: 0")
        return vals, hovers

    step0 = sorted_steps[0]
    vals0, hovers0 = _extract_aligned_frame_data(snapshots_by_step[step0])

    fig = go.Figure(
        data=[go.Sunburst(
            ids=canonical_order,
            labels=[canonical_meta[nid]["label"] for nid in canonical_order],
            parents=[canonical_meta[nid]["parent"] for nid in canonical_order],
            values=vals0,
            branchvalues="total",
            hovertext=hovers0,
            hoverinfo="text",
            marker=dict(colors=canonical_colors, line=dict(color="#ffffff", width=1.5)),
            insidetextorientation="radial",
            sort=False
        )]
    )

    frames = []
    slider_steps = []

    for step in sorted_steps:
        snap = snapshots_by_step[step]
        vals, hovers = _extract_aligned_frame_data(snap)
        frame = go.Frame(
            data=[go.Sunburst(
                ids=canonical_order,
                labels=[canonical_meta[nid]["label"] for nid in canonical_order],
                parents=[canonical_meta[nid]["parent"] for nid in canonical_order],
                values=vals,
                branchvalues="total",
                hovertext=hovers,
                hoverinfo="text",
                marker=dict(colors=canonical_colors, line=dict(color="#ffffff", width=1.5)),
                insidetextorientation="radial",
                sort=False
            )],
            name=f"Step_{step}"
        )
        frames.append(frame)

        slider_steps.append({
            "args": [[f"Step_{step}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
            "label": f"t={step}" + (" (Prior)" if step == 0 else ""),
            "method": "animate"
        })

    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"{title_prefix}: Multi-Level Nested Belief Sunburst (Scrub Timestep)", font=dict(size=18)),
        margin=dict(t=60, l=10, r=10, b=80),
        paper_bgcolor="#ffffff",
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Timestep: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "y": 0,
            "steps": slider_steps
        }],
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}],
                    "label": "▶ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "⏸ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 60},
            "showactive": False,
            "type": "buttons",
            "x": 0.05,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    if save_dir:
        _maybe_save_and_show(fig, filename, save_dir)
    return fig


def _plot_comparative_reward(df, labels, prefix, save_dir):
    df_active = df[df['status'] == 'Active']
    if df_active.empty: return

    grouped = df_active.groupby('step').agg({
        'cum_reward_i': ['mean', 'std'],
        'cum_reward_j': ['mean', 'std']
    }).reset_index()

    fig = go.Figure()
    for agent_key, label in labels.items():
        col_mean = grouped[(f'cum_reward_{agent_key}', 'mean')]
        fig.add_trace(go.Scatter(x=grouped['step'], y=col_mean, mode='lines', name=label))

    fig.update_layout(title=f"{prefix} - Cumulative Reward over Time",
                      xaxis_title="Step", yaxis_title="Cumulative Reward")
    _maybe_save_and_show(fig, "cumulative_reward", save_dir)


def _plot_reward_vs_time(df, labels, prefix, save_dir):
    if 'planning_time_i' not in df.columns and 'planning_time' not in df.columns:
        return

    df_active = df[df['status'] == 'Active'].copy()
    time_col = 'planning_time_i' if 'planning_time_i' in df.columns else 'planning_time'

    df_active['cum_time'] = df_active.groupby('trial')[time_col].cumsum()
    df_active['time_bucket'] = df_active['cum_time'].round(1)

    grouped = df_active.groupby('time_bucket').agg({
        'cum_reward_i': 'mean',
        'cum_reward_j': 'mean',
        'trial': 'count'
    }).reset_index()

    # Filter out buckets with too few samples to prevent noisy tails
    grouped = grouped[grouped['trial'] > (df['trial'].nunique() * 0.1)]

    fig = go.Figure()
    for agent_key, label in labels.items():
        col = f'cum_reward_{agent_key}'
        if col in grouped.columns:
            fig.add_trace(go.Scatter(x=grouped['time_bucket'], y=grouped[col], mode='lines', name=label))

    fig.update_layout(title=f"{prefix} - Compute-Normalized Reward",
                      xaxis_title="Wall-Clock Time (s)", yaxis_title="Cumulative Reward")
    _maybe_save_and_show(fig, "compute_normalized_reward", save_dir)


def _plot_belief_health(df, labels, prefix, save_dir):
    df_active = df[df['status'] == 'Active']

    fig = go.Figure()
    for agent_key, label in labels.items():
        col = f"n_particles_{agent_key}"
        if col in df.columns:
            grouped = df_active.groupby('step')[col].mean().reset_index()
            fig.add_trace(go.Scatter(x=grouped['step'], y=grouped[col], mode='lines', name=f"{label} Active Particles"))

    fig.update_layout(title=f"{prefix} - Particle Filter Health",
                      xaxis_title="Step", yaxis_title="Number of Particles")
    _maybe_save_and_show(fig, "particle_health", save_dir)


def _plot_action_values(df, labels, prefix, save_dir):
    """
    Plots the internal MCTS Q-values over time for the root node.
    Includes Phase 4 fix to safely deserialize Pandas CSV string dictionaries.
    """
    df_active = df[df['status'] == 'Active']

    for agent_key, label in labels.items():
        col = f"action_values_{agent_key}"
        if col not in df.columns: continue

        val_dicts = df_active[col].dropna()
        if val_dicts.empty: continue

        expanded = []
        for _, row in df_active.iterrows():
            val = row[col]

            # [PHASE 4 FIX]: Deserialize from string if loaded via cold CSV
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    continue

            if val and isinstance(val, dict):
                for act, v in val.items():
                    expanded.append({
                        'step': row['step'],
                        'action': str(act),
                        'value': v,
                        'trial': row['trial']
                    })

        if not expanded: continue
        exp_df = pd.DataFrame(expanded)
        avg_exp = exp_df.groupby(['step', 'action'])['value'].mean().reset_index()

        fig = px.line(avg_exp, x='step', y='value', color='action',
                      title=f"{prefix} - {label} Root Action Q-Values")
        _maybe_save_and_show(fig, f"action_values_{agent_key}", save_dir)


def _plot_distance(df, prefix, save_dir, col='distance'):
    if col not in df.columns: return
    df_active = df[df['status'] == 'Active']
    grouped = df_active.groupby('step')[col].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped['step'], y=grouped[col], mode='lines', name='Mean Distance'))
    fig.update_layout(title=f"{prefix} - Inter-Agent Distance over Time",
                      xaxis_title="Step", yaxis_title="Distance")
    _maybe_save_and_show(fig, col, save_dir)


def _plot_survival(df, prefix, save_dir):
    steps = df['step'].unique()
    survival_rates = []

    for s in steps:
        active_trials = df[(df['step'] == s) & (df['status'] == 'Active')]['trial'].nunique()
        total_trials = df[df['step'] == s]['trial'].nunique()
        survival_rates.append(active_trials / total_trials if total_trials > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=survival_rates, mode='lines', name='Survival Rate'))
    fig.update_layout(title=f"{prefix} - Trial Survival Rate",
                      xaxis_title="Step", yaxis_title="Proportion Active",
                      yaxis=dict(range=[0, 1.05]))
    _maybe_save_and_show(fig, "survival_rate", save_dir)