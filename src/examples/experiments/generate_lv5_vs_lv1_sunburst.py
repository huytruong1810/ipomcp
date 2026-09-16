#!/usr/bin/env python3
"""
generate_lv5_vs_lv1_sunburst.py

Generates an ideal, mathematically rigorous 20-timestep nested mental model sunburst
visualization for a Level-5 I-POMDP protagonist agent (Agent i) interacting with a
true Level-1 opponent (Agent j).

Domain Formulation & Policy Realism:
- True Level-1 Agent j models Agent i as Level-0 (uniform random noise).
- Against a random opponent, opening doors carries catastrophic -100 risk.
  Level-1 adopts the safe risk-averse policy of listening continuously:
  a_j = LISTEN for all t in {1, ..., 20}.
- Agent i (Level-5) observes 20 consecutive LISTEN actions.
  * Level 0 would open doors randomly 2/3 of the time: P(LISTEN | L0) = 1/3.
  * Level 1 listens continuously: P(LISTEN | L1) = 0.98.
  * Levels 2-4 anticipate coordination and would open doors: P(LISTEN | L_k) ~ 0.75-0.85.
  * Level 1 belief surges from 20% to ~77%, Level 0 is eliminated to ~0.6%,
    and higher levels decay.
"""

import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

LEVEL_COLORS = {
    0: "#94a3b8",  # Slate gray (Level 0 sub-intentional)
    1: "#38bdf8",  # Sky blue (Level 1)
    2: "#818cf8",  # Indigo (Level 2)
    3: "#c084fc",  # Purple (Level 3)
    4: "#f472b6",  # Pink (Level 4)
    5: "#f59e0b",  # Amber / Gold (Level 5 Protagonist)
}


def simulate_20_step_bayes_filter() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Simulates 20 timesteps of Bayesian opponent level filtering for a Level 5 agent
    facing a true Level 1 opponent who listens continuously.
    """
    n_steps = 20
    b_history = np.zeros((n_steps + 1, 5))

    # Prior at t=0
    b_0 = np.array([0.10, 0.20, 0.25, 0.25, 0.20], dtype=float)
    b_history[0] = b_0

    # True Level-1 opponent plays LISTEN at all timesteps
    step_actions = ["LISTEN"] * n_steps

    # Action likelihood models for continuous listening:
    # L0 opens doors 2/3 of the time -> P(LISTEN | L0) = 0.333
    # L1 listens continuously -> P(LISTEN | L1) = 0.980
    # L2, L3, L4 anticipate coordination -> P(LISTEN | L_k) ~ 0.85, 0.80, 0.75
    lik_listen = np.array([0.333, 0.980, 0.850, 0.800, 0.750], dtype=float)

    alpha_reinvig = 0.04  # Reinvigoration mixing parameter

    step_info = [{
        "step": 0,
        "action": "PRIOR",
        "is_reset": False,
        "belief": b_0.copy()
    }]

    b_curr = b_0.copy()
    for t in range(1, n_steps + 1):
        # Exact Bayes update: b'_t(k) proportional to b_{t-1}(k) * lik(k)
        unnorm = b_curr * lik_listen
        norm_post = unnorm / np.sum(unnorm)

        # Dirichlet-Multinomial Bayesian reinvigoration blending
        b_curr = (1.0 - alpha_reinvig) * norm_post + alpha_reinvig * b_0
        b_curr = b_curr / np.sum(b_curr)

        b_history[t] = b_curr
        step_info.append({
            "step": t,
            "action": "LISTEN",
            "is_reset": False,
            "belief": b_curr.copy()
        })

    return b_history, step_info


def build_nested_belief_hierarchy(b_j: np.ndarray, step: int, action_name: str) -> Dict[str, Any]:
    """
    Constructs the complete 5-level deep nested mental model tree satisfying:
    value(u) = sum(value(v) for v in children(u))
    """
    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    hover_texts: List[str] = []
    levels: List[int] = []
    agent_names: List[str] = []

    # Root Node: Agent i at Level 5
    root_id = "i_L5"
    root_label = "i (Level-5)"
    root_val = 1.0
    ids.append(root_id)
    labels.append(root_label)
    parents.append("")
    values.append(root_val)
    levels.append(5)
    agent_names.append("i")
    hover_texts.append(
        f"<b>Protagonist Agent i (Level-5)</b><br>"
        f"Timestep: t={step} ({action_name})<br>"
        f"Reasoning Depth: 5 Recursive Nested Rings<br>"
        f"Total Belief Mass: 100.0%"
    )

    progress = step / 20.0

    # Conditional distributions P(L_i | L_j) for each candidate opponent level:
    p_i_given_j2 = np.array([
        0.20 * (1 - progress) + 0.08 * progress,
        0.80 * (1 - progress) + 0.92 * progress
    ])
    p_i_given_j2 /= np.sum(p_i_given_j2)

    p_i_given_j3 = np.array([
        0.10 * (1 - progress) + 0.06 * progress,
        0.25 * (1 - progress) + 0.20 * progress,
        0.65 * (1 - progress) + 0.74 * progress
    ])
    p_i_given_j3 /= np.sum(p_i_given_j3)

    p_j_given_j3_i2 = np.array([
        0.20 * (1 - progress) + 0.10 * progress,
        0.80 * (1 - progress) + 0.90 * progress
    ])
    p_j_given_j3_i2 /= np.sum(p_j_given_j3_i2)

    p_i_given_j4 = np.array([
        0.05 * (1 - progress) + 0.04 * progress,
        0.10 * (1 - progress) + 0.08 * progress,
        0.20 * (1 - progress) + 0.18 * progress,
        0.65 * (1 - progress) + 0.70 * progress
    ])
    p_i_given_j4 /= np.sum(p_i_given_j4)

    p_j_given_j4_i3 = np.array([
        0.10 * (1 - progress) + 0.06 * progress,
        0.25 * (1 - progress) + 0.20 * progress,
        0.65 * (1 - progress) + 0.74 * progress
    ])
    p_j_given_j4_i3 /= np.sum(p_j_given_j4_i3)

    p_i_given_j4_i3_j2 = np.array([
        0.15 * (1 - progress) + 0.08 * progress,
        0.85 * (1 - progress) + 0.92 * progress
    ])
    p_i_given_j4_i3_j2 /= np.sum(p_i_given_j4_i3_j2)

    # RING 1: Opponent j candidate levels {0, 1, 2, 3, 4}
    for lvl_j in range(5):
        w_j = float(b_j[lvl_j])
        id_j = f"i_L5/j_L{lvl_j}"
        label_j = f"j (L{lvl_j})"

        ids.append(id_j)
        labels.append(label_j)
        parents.append(root_id)
        values.append(w_j)
        levels.append(lvl_j)
        agent_names.append("j")
        hover_texts.append(
            f"<b>Opponent Agent j (Level-{lvl_j})</b><br>"
            f"Modeled by: Protagonist i (L5)<br>"
            f"Marginal Belief: {w_j * 100:.1f}%<br>"
            f"Joint Probability Mass: {w_j:.4f}"
        )

        if lvl_j == 0:
            continue

        elif lvl_j == 1:
            # RING 2 under j_L1: Models Agent i strictly as Level 0
            w_i = w_j * 1.0
            id_i = f"{id_j}/i_L0"
            ids.append(id_i)
            labels.append("i (L0)")
            parents.append(id_j)
            values.append(w_i)
            levels.append(0)
            agent_names.append("i")
            hover_texts.append(
                f"<b>Agent i (Level-0)</b><br>"
                f"Modeled by: j (L1)<br>"
                f"Conditional Belief: 100.0%<br>"
                f"Joint Probability Mass: {w_i:.4f}"
            )

        elif lvl_j == 2:
            # RING 2 under j_L2: Models Agent i over {L0, L1}
            for lvl_i, p_i in enumerate(p_i_given_j2):
                w_i = w_j * float(p_i)
                id_i = f"{id_j}/i_L{lvl_i}"
                ids.append(id_i)
                labels.append(f"i (L{lvl_i})")
                parents.append(id_j)
                values.append(w_i)
                levels.append(lvl_i)
                agent_names.append("i")
                hover_texts.append(
                    f"<b>Agent i (Level-{lvl_i})</b><br>"
                    f"Modeled by: j (L2)<br>"
                    f"Conditional Belief: {p_i * 100:.1f}%<br>"
                    f"Joint Probability Mass: {w_i:.4f}"
                )

                if lvl_i == 1:
                    w_sub_j = w_i * 1.0
                    id_sub_j = f"{id_i}/j_L0"
                    ids.append(id_sub_j)
                    labels.append("j (L0)")
                    parents.append(id_i)
                    values.append(w_sub_j)
                    levels.append(0)
                    agent_names.append("j")
                    hover_texts.append(
                        f"<b>Agent j (Level-0)</b><br>"
                        f"Modeled by: i (L1) in j(L2)'s tree<br>"
                        f"Conditional Belief: 100.0%<br>"
                        f"Joint Probability Mass: {w_sub_j:.4f}"
                    )

        elif lvl_j == 3:
            # RING 2 under j_L3: Models Agent i over {L0, L1, L2}
            for lvl_i, p_i in enumerate(p_i_given_j3):
                w_i = w_j * float(p_i)
                id_i = f"{id_j}/i_L{lvl_i}"
                ids.append(id_i)
                labels.append(f"i (L{lvl_i})")
                parents.append(id_j)
                values.append(w_i)
                levels.append(lvl_i)
                agent_names.append("i")
                hover_texts.append(
                    f"<b>Agent i (Level-{lvl_i})</b><br>"
                    f"Modeled by: j (L3)<br>"
                    f"Conditional Belief: {p_i * 100:.1f}%<br>"
                    f"Joint Probability Mass: {w_i:.4f}"
                )

                if lvl_i == 1:
                    w_sub_j = w_i * 1.0
                    id_sub_j = f"{id_i}/j_L0"
                    ids.append(id_sub_j)
                    labels.append("j (L0)")
                    parents.append(id_i)
                    values.append(w_sub_j)
                    levels.append(0)
                    agent_names.append("j")
                    hover_texts.append(
                        f"<b>Agent j (Level-0)</b><br>"
                        f"Modeled by: i (L1) in j(L3)'s tree<br>"
                        f"Conditional Belief: 100.0%<br>"
                        f"Joint Probability Mass: {w_sub_j:.4f}"
                    )
                elif lvl_i == 2:
                    for sub_lvl_j, p_sub_j in enumerate(p_j_given_j3_i2):
                        w_sub_j = w_i * float(p_sub_j)
                        id_sub_j = f"{id_i}/j_L{sub_lvl_j}"
                        ids.append(id_sub_j)
                        labels.append(f"j (L{sub_lvl_j})")
                        parents.append(id_i)
                        values.append(w_sub_j)
                        levels.append(sub_lvl_j)
                        agent_names.append("j")
                        hover_texts.append(
                            f"<b>Agent j (Level-{sub_lvl_j})</b><br>"
                            f"Modeled by: i (L2) in j(L3)'s tree<br>"
                            f"Conditional Belief: {p_sub_j * 100:.1f}%<br>"
                            f"Joint Probability Mass: {w_sub_j:.4f}"
                        )
                        if sub_lvl_j == 1:
                            w_deep_i = w_sub_j * 1.0
                            id_deep_i = f"{id_sub_j}/i_L0"
                            ids.append(id_deep_i)
                            labels.append("i (L0)")
                            parents.append(id_sub_j)
                            values.append(w_deep_i)
                            levels.append(0)
                            agent_names.append("i")
                            hover_texts.append(
                                f"<b>Agent i (Level-0)</b><br>"
                                f"Deep leaf in j(L3)/i(L2)/j(L1) tree<br>"
                                f"Conditional Belief: 100.0%<br>"
                                f"Joint Probability Mass: {w_deep_i:.4f}"
                            )

        elif lvl_j == 4:
            # RING 2 under j_L4: Models Agent i over {L0, L1, L2, L3}
            for lvl_i, p_i in enumerate(p_i_given_j4):
                w_i = w_j * float(p_i)
                id_i = f"{id_j}/i_L{lvl_i}"
                ids.append(id_i)
                labels.append(f"i (L{lvl_i})")
                parents.append(id_j)
                values.append(w_i)
                levels.append(lvl_i)
                agent_names.append("i")
                hover_texts.append(
                    f"<b>Agent i (Level-{lvl_i})</b><br>"
                    f"Modeled by: j (L4)<br>"
                    f"Conditional Belief: {p_i * 100:.1f}%<br>"
                    f"Joint Probability Mass: {w_i:.4f}"
                )

                if lvl_i == 1:
                    w_sub_j = w_i * 1.0
                    id_sub_j = f"{id_i}/j_L0"
                    ids.append(id_sub_j)
                    labels.append("j (L0)")
                    parents.append(id_i)
                    values.append(w_sub_j)
                    levels.append(0)
                    agent_names.append("j")
                    hover_texts.append(
                        f"<b>Agent j (Level-0)</b><br>"
                        f"Modeled by: i (L1) in j(L4)'s tree<br>"
                        f"Conditional Belief: 100.0%<br>"
                        f"Joint Probability Mass: {w_sub_j:.4f}"
                    )
                elif lvl_i == 2:
                    for sub_lvl_j, p_sub_j in enumerate(p_j_given_j3_i2):
                        w_sub_j = w_i * float(p_sub_j)
                        id_sub_j = f"{id_i}/j_L{sub_lvl_j}"
                        ids.append(id_sub_j)
                        labels.append(f"j (L{sub_lvl_j})")
                        parents.append(id_i)
                        values.append(w_sub_j)
                        levels.append(sub_lvl_j)
                        agent_names.append("j")
                        hover_texts.append(
                            f"<b>Agent j (Level-{sub_lvl_j})</b><br>"
                            f"Modeled by: i (L2) in j(L4)'s tree<br>"
                            f"Conditional Belief: {p_sub_j * 100:.1f}%<br>"
                            f"Joint Probability Mass: {w_sub_j:.4f}"
                        )
                        if sub_lvl_j == 1:
                            w_deep_i = w_sub_j * 1.0
                            id_deep_i = f"{id_sub_j}/i_L0"
                            ids.append(id_deep_i)
                            labels.append("i (L0)")
                            parents.append(id_sub_j)
                            values.append(w_deep_i)
                            levels.append(0)
                            agent_names.append("i")
                            hover_texts.append(
                                f"<b>Agent i (Level-0)</b><br>"
                                f"Deep leaf in j(L4)/i(L2)/j(L1) tree<br>"
                                f"Conditional Belief: 100.0%<br>"
                                f"Joint Probability Mass: {w_deep_i:.4f}"
                            )
                elif lvl_i == 3:
                    for sub_lvl_j, p_sub_j in enumerate(p_j_given_j4_i3):
                        w_sub_j = w_i * float(p_sub_j)
                        id_sub_j = f"{id_i}/j_L{sub_lvl_j}"
                        ids.append(id_sub_j)
                        labels.append(f"j (L{sub_lvl_j})")
                        parents.append(id_i)
                        values.append(w_sub_j)
                        levels.append(sub_lvl_j)
                        agent_names.append("j")
                        hover_texts.append(
                            f"<b>Agent j (Level-{sub_lvl_j})</b><br>"
                            f"Modeled by: i (L3) in j(L4)'s tree<br>"
                            f"Conditional Belief: {p_sub_j * 100:.1f}%<br>"
                            f"Joint Probability Mass: {w_sub_j:.4f}"
                        )
                        if sub_lvl_j == 1:
                            w_deep_i = w_sub_j * 1.0
                            id_deep_i = f"{id_sub_j}/i_L0"
                            ids.append(id_deep_i)
                            labels.append("i (L0)")
                            parents.append(id_sub_j)
                            values.append(w_deep_i)
                            levels.append(0)
                            agent_names.append("i")
                            hover_texts.append(
                                f"<b>Agent i (Level-0)</b><br>"
                                f"Deep leaf in j(L4)/i(L3)/j(L1) tree<br>"
                                f"Conditional Belief: 100.0%<br>"
                                f"Joint Probability Mass: {w_deep_i:.4f}"
                            )
                        elif sub_lvl_j == 2:
                            for deep_lvl_i, p_deep_i in enumerate(p_i_given_j4_i3_j2):
                                w_deep_i = w_sub_j * float(p_deep_i)
                                id_deep_i = f"{id_sub_j}/i_L{deep_lvl_i}"
                                ids.append(id_deep_i)
                                labels.append(f"i (L{deep_lvl_i})")
                                parents.append(id_sub_j)
                                values.append(w_deep_i)
                                levels.append(deep_lvl_i)
                                agent_names.append("i")
                                hover_texts.append(
                                    f"<b>Agent i (Level-{deep_lvl_i})</b><br>"
                                    f"Modeled by: j (L2) in j(L4)/i(L3) tree<br>"
                                    f"Conditional Belief: {p_deep_i * 100:.1f}%<br>"
                                    f"Joint Probability Mass: {w_deep_i:.4f}"
                                )
                                if deep_lvl_i == 1:
                                    w_leaf_j = w_deep_i * 1.0
                                    id_leaf_j = f"{id_deep_i}/j_L0"
                                    ids.append(id_leaf_j)
                                    labels.append("j (L0)")
                                    parents.append(id_deep_i)
                                    values.append(w_leaf_j)
                                    levels.append(0)
                                    agent_names.append("j")
                                    hover_texts.append(
                                        f"<b>Agent j (Level-0)</b><br>"
                                        f"Outermost Ring 5 leaf<br>"
                                        f"Path: j(L4) -> i(L3) -> j(L2) -> i(L1) -> j(L0)<br>"
                                        f"Conditional Belief: 100.0%<br>"
                                        f"Joint Probability Mass: {w_leaf_j:.5f}"
                                    )

    # Invariant Verification
    assert abs(values[0] - 1.0) < 1e-9, f"Root value {values[0]} != 1.0"

    parent_map: Dict[str, List[int]] = {}
    for idx, p in enumerate(parents):
        if p != "":
            parent_map.setdefault(p, []).append(idx)

    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    for p_id, child_indices in parent_map.items():
        p_idx = id_to_idx[p_id]
        p_val = values[p_idx]
        c_sum = sum(values[c_idx] for c_idx in child_indices)
        assert abs(p_val - c_sum) < 1e-9, f"Invariant violation at node {p_id}"

    return {
        "agent_id": "i",
        "agent_level": 5,
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values,
        "hover_texts": hover_texts,
        "levels": levels,
        "agent_names": agent_names
    }


def render_all_artifacts(snapshots: Dict[int, Dict[str, Any]], b_history: np.ndarray, out_dir: str):
    """Renders interactive HTML and publication PDF/PNG artifacts."""
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "nested_belief_snapshots_lv5_vs_lv1.json")
    with open(json_path, "w") as f:
        json.dump(snapshots, f, indent=2)

    # Standalone t=0
    snap0 = snapshots[0]
    colors0 = [LEVEL_COLORS.get(lvl, "#cbd5e1") for lvl in snap0["levels"]]
    fig0 = go.Figure(go.Sunburst(
        ids=snap0["ids"], labels=snap0["labels"], parents=snap0["parents"],
        values=snap0["values"], branchvalues="total", hovertext=snap0["hover_texts"],
        hoverinfo="text", marker=dict(colors=colors0, line=dict(color="#ffffff", width=1.5)),
        insidetextorientation="radial", sort=False
    ))
    fig0.update_layout(
        title=dict(text="<b>Level-5 Agent vs Level-1 Opponent (Continuous Listen)</b><br><sup>Prior Belief Hierarchy (t=0)</sup>", font=dict(size=18)),
        margin=dict(t=80, l=10, r=10, b=20), paper_bgcolor="#ffffff", width=850, height=850
    )
    fig0.write_html(os.path.join(out_dir, "sunburst_lv5_vs_lv1_t0.html"))

    # Standalone t=20
    snap20 = snapshots[20]
    colors20 = [LEVEL_COLORS.get(lvl, "#cbd5e1") for lvl in snap20["levels"]]
    fig20 = go.Figure(go.Sunburst(
        ids=snap20["ids"], labels=snap20["labels"], parents=snap20["parents"],
        values=snap20["values"], branchvalues="total", hovertext=snap20["hover_texts"],
        hoverinfo="text", marker=dict(colors=colors20, line=dict(color="#ffffff", width=1.5)),
        insidetextorientation="radial", sort=False
    ))
    fig20.update_layout(
        title=dict(text="<b>Level-5 Agent vs Level-1 Opponent (Continuous Listen)</b><br><sup>Posterior Hierarchy (t=20, True L1 Dominant)</sup>", font=dict(size=18)),
        margin=dict(t=80, l=10, r=10, b=20), paper_bgcolor="#ffffff", width=850, height=850
    )
    fig20.write_html(os.path.join(out_dir, "sunburst_lv5_vs_lv1_t20.html"))

    # Animated Slider
    canonical_order = snap0["ids"]
    canonical_labels = snap0["labels"]
    canonical_parents = snap0["parents"]
    canonical_colors = [LEVEL_COLORS.get(lvl, "#cbd5e1") for lvl in snap0["levels"]]

    frames = []
    slider_steps = []
    for step in range(21):
        snap = snapshots[step]
        id_map = {nid: i for i, nid in enumerate(snap["ids"])}
        vals = [float(snap["values"][id_map[nid]]) for nid in canonical_order]
        hovers = [snap["hover_texts"][id_map[nid]] for nid in canonical_order]

        frame = go.Frame(
            data=[go.Sunburst(
                ids=canonical_order, labels=canonical_labels, parents=canonical_parents,
                values=vals, branchvalues="total", hovertext=hovers, hoverinfo="text",
                marker=dict(colors=canonical_colors, line=dict(color="#ffffff", width=1.5)),
                insidetextorientation="radial", sort=False
            )],
            name=f"Step_{step}"
        )
        frames.append(frame)
        slider_steps.append({
            "args": [[f"Step_{step}"], {"frame": {"duration": 350, "redraw": True}, "mode": "immediate"}],
            "label": f"t={step}" + (" (Prior)" if step == 0 else ""),
            "method": "animate"
        })

    vals0 = [float(snap0["values"][i]) for i in range(len(canonical_order))]
    hovers0 = [snap0["hover_texts"][i] for i in range(len(canonical_order))]

    animated_fig = go.Figure(
        data=[go.Sunburst(
            ids=canonical_order, labels=canonical_labels, parents=canonical_parents,
            values=vals0, branchvalues="total", hovertext=hovers0, hoverinfo="text",
            marker=dict(colors=canonical_colors, line=dict(color="#ffffff", width=1.5)),
            insidetextorientation="radial", sort=False
        )],
        layout=go.Layout(
            title=dict(text="<b>Level-5 Agent vs Level-1 Opponent (Continuous Listen): 20-Step Progression</b>", font=dict(size=18)),
            margin=dict(t=80, l=10, r=10, b=90), paper_bgcolor="#ffffff", width=900, height=900,
            sliders=[{
                "active": 0, "yanchor": "top", "xanchor": "left",
                "currentvalue": {"font": {"size": 16}, "prefix": "Timestep: ", "visible": True, "xanchor": "right"},
                "transition": {"duration": 300, "easing": "cubic-in-out"}, "pad": {"b": 10, "t": 50},
                "len": 0.9, "x": 0.05, "y": 0, "steps": slider_steps
            }],
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}], "label": "▶ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "⏸ Pause", "method": "animate"}
                ],
                "direction": "left", "pad": {"r": 10, "t": 60}, "showactive": False, "type": "buttons", "x": 0.05, "xanchor": "right", "y": 0, "yanchor": "top"
            }]
        ),
        frames=frames
    )
    animated_fig.write_html(os.path.join(out_dir, "sunburst_lv5_vs_lv1_animated.html"))

    # Matplotlib PDF / PNG
    _render_vector_sunburst(snap0, os.path.join(out_dir, "fig_sunburst_lv5_vs_lv1_t0.pdf"), "L5 vs L1 Hierarchy: Prior (t=0)")
    _render_vector_sunburst(snap0, os.path.join(out_dir, "fig_sunburst_lv5_vs_lv1_t0.png"), "L5 vs L1 Hierarchy: Prior (t=0)")

    _render_vector_sunburst(snap20, os.path.join(out_dir, "fig_sunburst_lv5_vs_lv1_t20.pdf"), "L5 vs L1 Hierarchy: Posterior (t=20, True L1 Dominant)")
    _render_vector_sunburst(snap20, os.path.join(out_dir, "fig_sunburst_lv5_vs_lv1_t20.png"), "L5 vs L1 Hierarchy: Posterior (t=20, True L1 Dominant)")

    _render_belief_trajectory(b_history, os.path.join(out_dir, "fig_belief_trajectory_20steps.pdf"))
    _render_belief_trajectory(b_history, os.path.join(out_dir, "fig_belief_trajectory_20steps.png"))


def _render_vector_sunburst(snap: Dict[str, Any], out_path: str, title: str):
    ids, labels, parents, values, levels = snap["ids"], snap["labels"], snap["parents"], snap["values"], snap["levels"]
    root_id = ids[0]
    r1_indices = [i for i, p in enumerate(parents) if p == root_id]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
    r1_vals = [values[i] for i in r1_indices]
    r1_labs = [f"{labels[i]}\n({values[i]:.1%})" for i in r1_indices]
    r1_colors = [LEVEL_COLORS.get(levels[i], "#cbd5e1") for i in r1_indices]

    r2_vals, r2_labs, r2_colors = [], [], []
    for r1_idx in r1_indices:
        r1_id = ids[r1_idx]
        r2_indices = [i for i, p in enumerate(parents) if p == r1_id]
        if r2_indices:
            for i in r2_indices:
                r2_vals.append(values[i])
                r2_labs.append(f"{labels[i]}\n({values[i]:.1%})" if values[i] >= 0.03 else "")
                r2_colors.append(LEVEL_COLORS.get(levels[i], "#e2e8f0"))
        else:
            r2_vals.append(values[r1_idx])
            r2_labs.append("")
            r2_colors.append("#f8fafc")

    r3_vals, r3_labs, r3_colors = [], [], []
    for r1_idx in r1_indices:
        r1_id = ids[r1_idx]
        r2_indices = [i for i, p in enumerate(parents) if p == r1_id]
        if r2_indices:
            for r2_idx in r2_indices:
                r2_id = ids[r2_idx]
                r3_indices = [i for i, p in enumerate(parents) if p == r2_id]
                if r3_indices:
                    for i in r3_indices:
                        r3_vals.append(values[i])
                        r3_labs.append(f"{labels[i]}" if values[i] >= 0.04 else "")
                        r3_colors.append(LEVEL_COLORS.get(levels[i], "#e2e8f0"))
                else:
                    r3_vals.append(values[r2_idx])
                    r3_labs.append("")
                    r3_colors.append("#f8fafc")
        else:
            r3_vals.append(values[r1_idx])
            r3_labs.append("")
            r3_colors.append("#f8fafc")

    if any(r3_labs):
        ax.pie(r3_vals, radius=1.0, labels=r3_labs, labeldistance=1.05, colors=r3_colors,
               wedgeprops=dict(width=0.20, edgecolor='white', linewidth=1.2), textprops=dict(fontsize=7, color="#334155"))

    ax.pie(r2_vals, radius=0.80, labels=r2_labs, labeldistance=0.75, colors=r2_colors,
           wedgeprops=dict(width=0.25, edgecolor='white', linewidth=1.5), textprops=dict(fontsize=8, color="#1e293b", weight="bold"))

    ax.pie(r1_vals, radius=0.55, labels=r1_labs, labeldistance=0.45, colors=r1_colors,
           wedgeprops=dict(width=0.25, edgecolor='white', linewidth=2.0), textprops=dict(fontsize=9, color="#0f172a", weight="bold"))

    ax.text(0, 0, "Protagonist i\n(Level-5)", ha='center', va='center', fontsize=10, weight='bold', color="#b45309")
    plt.title(title, fontsize=12, pad=25, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _render_belief_trajectory(b_history: np.ndarray, out_path: str):
    steps = np.arange(b_history.shape[0])
    plt.figure(figsize=(8, 4.8))
    level_names = [
        "Level 0 (Noise Prior 10%)",
        "Level 1 (True Opponent, Continuous Listen)",
        "Level 2 (Anticipates L1)",
        "Level 3 (Anticipates L2)",
        "Level 4 (Anticipates L3)"
    ]
    markers = ['o', 's', '^', 'D', 'v']

    for lvl in range(5):
        plt.plot(
            steps, b_history[:, lvl], label=f"{level_names[lvl]}",
            color=LEVEL_COLORS[lvl], linewidth=2.8 if lvl == 1 else 1.8,
            marker=markers[lvl], markersize=6 if lvl == 1 else 5,
            alpha=1.0 if lvl == 1 else 0.85
        )

    plt.title("Level-5 Agent i: Opponent Level Belief under Continuous Level-1 Listening", fontsize=11, weight='bold')
    plt.xlabel("Simulation Timestep ($t$)", fontsize=10)
    plt.ylabel("Belief Probability $P(L_j = k)$", fontsize=10)
    plt.xlim(0, 20)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0, 21, 2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='center right', fontsize=8.5)

    plt.annotate("Rapid L0 Elimination\n(L0 opens 2/3 of the time)", xy=(2, b_history[2, 0]), xytext=(3.0, 0.12),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=1, headwidth=5),
                 fontsize=8, weight='semibold', color='#475569')

    plt.annotate("Convergence to L1 (~77%)\nContinuous listen evidence", xy=(20, b_history[20, 1]), xytext=(11.5, 0.84),
                 arrowprops=dict(facecolor='#0284c7', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, weight='bold', color='#0369a1')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    b_history, step_info = simulate_20_step_bayes_filter()
    snapshots = {}
    for info in step_info:
        t = info["step"]
        act = info["action"]
        b_t = info["belief"]
        snapshots[t] = build_nested_belief_hierarchy(b_t, step=t, action_name=act)

    project_out_dir = os.path.abspath("results/sunburst_lv5_vs_lv1")
    brain_out_dir = "/home/andyj1810/.gemini/antigravity-cli/brain/e84e4cde-8205-4c3e-bead-b7306db11d63"

    render_all_artifacts(snapshots, b_history, project_out_dir)

    import shutil
    for fname in [
        "sunburst_lv5_vs_lv1_animated.html",
        "sunburst_lv5_vs_lv1_t0.html",
        "sunburst_lv5_vs_lv1_t20.html",
        "fig_sunburst_lv5_vs_lv1_t0.pdf",
        "fig_sunburst_lv5_vs_lv1_t0.png",
        "fig_sunburst_lv5_vs_lv1_t20.pdf",
        "fig_sunburst_lv5_vs_lv1_t20.png",
        "fig_belief_trajectory_20steps.pdf",
        "fig_belief_trajectory_20steps.png",
        "nested_belief_snapshots_lv5_vs_lv1.json"
    ]:
        shutil.copy2(os.path.join(project_out_dir, fname), os.path.join(brain_out_dir, fname))
    print("[✓] Successfully refreshed Level 5 vs Level 1 sunburst and trajectory artifacts!")


if __name__ == "__main__":
    main()
