"""
paper_plots — High-DPI Static Graphic Generator for LaTeX integration.

Reads the output of `GenericBatchRunner` (batch_results.csv) and generates
publication-ready vector graphics (PDF) using Seaborn, with configurable agent labels. Journal layout requirements must be checked
for the intended venue; font styling alone is not a compliance guarantee.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import argparse
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_metadata import agent_labels as resolve_agent_labels

# Set publication styling (publication)
plt.style.use("seaborn-v0_8-paper")

sns.set_context("paper", font_scale=1.3)
sns.set_palette("colorblind")


def _resolve_title_prefix(output_dir: str, user_prefix: str = "") -> str:
    """Formats a clean title prefix from directory name if not explicitly provided."""
    if user_prefix:
        return user_prefix
    folder_name = os.path.basename(os.path.abspath(output_dir))
    clean_name = (
        folder_name.replace("Cell_", "").replace("cond_", "").replace("_", " ").replace("pct", "%")
    )
    return clean_name


def generate_paper_plots(
    csv_path: str,
    output_dir: str,
    agent_labels: Optional[Dict[str, str]] = None,
    title_prefix: str = "",
):
    """
    Main entry point for generating static publication figures from a batch CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("Dataframe is empty. No paper plots to generate.")
        return

    labels = resolve_agent_labels(df, agent_labels)
    prefix = _resolve_title_prefix(output_dir, title_prefix)

    df = df.sort_values(["trial", "step"]).copy()
    df["cum_time"] = df.groupby("trial")["planning_time_i"].cumsum()
    df["time_bucket"] = df["cum_time"].round(1)
    _plot_cumulative_reward(df, labels, prefix, output_dir)
    _plot_compute_normalized_reward(df, labels, prefix, output_dir)
    _plot_particle_health(df, labels, prefix, output_dir)
    level_cols = [c for c in df.columns if c.startswith("prob_l")]
    if level_cols:
        _plot_opponent_level_belief(df, level_cols, prefix, output_dir)

    print(f"Finished generating plots in: {output_dir}")


def _plot_opponent_level_belief(df: pd.DataFrame, level_cols: list, prefix: str, out_dir: str):
    """Plots online posterior distribution P(l_j = k | h_i^t) against Simulation Step."""
    plt.figure(figsize=(6.5, 4.5))

    for col in sorted(level_cols):
        parts = col.split("_")
        lvl_num = int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].isdigit() else 0
        lvl_label = f"P(Opponent Level-{lvl_num})"
        sns.lineplot(data=df, x="step", y=col, label=lvl_label, errorbar=("ci", 95), marker="o")

    plt.title(f"{prefix}\nOpponent Level Posterior Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Simulation Step (t=0 is Initial Prior)")
    plt.ylabel("Posterior Probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_opponent_level_belief.pdf"), format="pdf", dpi=300)
    plt.close()


def generate_nested_sunburst_pdf(
    nested_data: Dict[str, Any], out_path: str, title: str = "Nested Mental Model Hierarchy"
):
    """
    Renders a publication-ready concentric multi-ring donut chart of the nested mental model hierarchy for IEEE/ACM papers.
    """
    if not nested_data or "ids" not in nested_data or len(nested_data["ids"]) <= 1:
        return

    ids = nested_data["ids"]
    labels = nested_data["labels"]
    parents = nested_data["parents"]
    values = nested_data["values"]
    levels = nested_data.get("levels", [0] * len(ids))

    # Ring 1: Direct children of root
    root_id = ids[0]
    r1_indices = [idx for idx, p in enumerate(parents) if p == root_id]
    if not r1_indices:
        return

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    r1_vals = [values[i] for i in r1_indices]
    r1_labs = [f"{labels[i]}\n({values[i]:.1%})" for i in r1_indices]
    colors_palette = ["#93c5fd", "#818cf8", "#c084fc", "#f472b6", "#cbd5e1"]
    r1_colors = [colors_palette[levels[i] % len(colors_palette)] for i in r1_indices]

    # Ring 2: Children of Ring 1
    r2_vals = []
    r2_labs = []
    r2_colors = []
    for r1_idx in r1_indices:
        r1_id = ids[r1_idx]
        r2_indices = [idx for idx, p in enumerate(parents) if p == r1_id]
        if r2_indices:
            for idx in r2_indices:
                r2_vals.append(values[idx])
                r2_labs.append(f"{labels[idx]}\n({values[idx]:.1%})")
                r2_colors.append(
                    "#e2e8f0"
                    if levels[idx] == 0
                    else colors_palette[levels[idx] % len(colors_palette)]
                )
        else:
            r2_vals.append(values[r1_idx])
            r2_labs.append("")
            r2_colors.append("#f8fafc")

    # Draw Outer Ring (Ring 2)
    if any(r2_labs):
        ax.pie(
            r2_vals,
            radius=1.0,
            labels=r2_labs,
            labeldistance=1.05,
            colors=r2_colors,
            wedgeprops=dict(width=0.3, edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=8),
        )

    # Draw Inner Ring (Ring 1)
    ax.pie(
        r1_vals,
        radius=0.7,
        labels=r1_labs,
        labeldistance=0.5,
        colors=r1_colors,
        wedgeprops=dict(width=0.3, edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=9, weight="bold"),
    )

    ax.text(0, 0, f"{labels[0]}\n(Root)", ha="center", va="center", fontsize=10, weight="bold")

    plt.title(title, fontsize=11, pad=20, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()


def _plot_cumulative_reward(df: pd.DataFrame, labels: Dict[str, str], prefix: str, out_dir: str):
    """Plots Cumulative Reward against Simulation Step with dynamic agent labels."""
    plt.figure(figsize=(6.5, 4.5))

    if "cum_reward_i" in df.columns:
        sns.lineplot(
            data=df,
            x="step",
            y="cum_reward_i",
            label=labels["i"],
            errorbar=("ci", 95),
            color="#1f77b4",
            marker="o",
        )
    if "cum_reward_j" in df.columns:
        sns.lineplot(
            data=df,
            x="step",
            y="cum_reward_j",
            label=labels["j"],
            errorbar=("ci", 95),
            color="#ff7f0e",
            marker="s",
        )

    plt.title(f"{prefix}\nCumulative Reward over Time", fontsize=11, fontweight="bold")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward ($R$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_reward_vs_step.pdf"), format="pdf", dpi=300)
    plt.close()


def _plot_compute_normalized_reward(
    df: pd.DataFrame, labels: Dict[str, str], prefix: str, out_dir: str
):
    """Plots Cumulative Reward against Wall-Clock Planning Time with dynamic agent labels."""
    if "time_bucket" not in df.columns:
        return

    plt.figure(figsize=(6.5, 4.5))

    df = df.groupby(["trial", "time_bucket"], as_index=False).tail(1)
    min_required_samples = df["trial"].nunique() * 0.1
    plot_df = df[df.groupby("time_bucket")["trial"].transform("count") > min_required_samples]

    if plot_df.empty:
        plt.close()
        return

    if "cum_reward_i" in plot_df.columns:
        sns.lineplot(
            data=plot_df,
            x="time_bucket",
            y="cum_reward_i",
            label=labels["i"],
            errorbar=("ci", 95),
            color="#1f77b4",
        )
    if "cum_reward_j" in plot_df.columns:
        sns.lineplot(
            data=plot_df,
            x="time_bucket",
            y="cum_reward_j",
            label=labels["j"],
            errorbar=("ci", 95),
            color="#ff7f0e",
        )

    plt.title(f"{prefix}\nCompute-Normalized Cumulative Reward", fontsize=11, fontweight="bold")
    plt.xlabel("Wall-Clock Planning Time (s)")
    plt.ylabel("Cumulative Reward ($R$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_reward_vs_time.pdf"), format="pdf", dpi=300)
    plt.close()


def _plot_particle_health(df: pd.DataFrame, labels: Dict[str, str], prefix: str, out_dir: str):
    """Plots the active number of particles over time as a population-size diagnostic, not a proof of filter accuracy."""
    if "status" not in df.columns:
        return

    plt.figure(figsize=(6.5, 4.5))
    df_active = df[df["status"] == "Active"]

    if "n_particles_i" in df_active.columns:
        sns.lineplot(
            data=df_active,
            x="step",
            y="n_particles_i",
            label=f"{labels['i']} Particles",
            errorbar=("ci", 95),
            color="#1f77b4",
            marker="o",
        )
    if "n_particles_j" in df_active.columns:
        sns.lineplot(
            data=df_active,
            x="step",
            y="n_particles_j",
            label=f"{labels['j']} Particles",
            errorbar=("ci", 95),
            color="#ff7f0e",
            marker="s",
        )

    plt.title(f"{prefix}\nParticle Filter Health", fontsize=11, fontweight="bold")
    plt.xlabel("Simulation Step")
    plt.ylabel("Active Particles")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_particle_health.pdf"), format="pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Publication-Ready Plots from I-POMCP Batch CSV"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to batch_results.csv")
    parser.add_argument("--out", type=str, required=True, help="Output directory for PDF figures")
    args = parser.parse_args()

    generate_paper_plots(args.csv, args.out)
