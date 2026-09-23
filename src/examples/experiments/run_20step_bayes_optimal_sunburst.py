"""20-Step Bayes-Optimal Opponent Modeling Sunburst & Trajectory Benchmark.

Executes a full 20-step interactive episode in the Tiger domain for a Level 2 agent
facing an optimizing Level 1 opponent under an uncertain prior mixture (80% L1, 20% L0).
Records the complete nested belief hierarchy at every timestep t in [0, 20], generating:
1. An interactive Plotly Sunburst visualization with an animated timestep slider (t=0..20).
2. Publication-quality vector PDF multi-ring sunburst figures at key decision points.
3. A multi-panel trajectory plot of opponent model mass, physical state beliefs, and rewards.
4. An empirical CSV log of all episode metrics.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from core.config import IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from examples.tiger.model.tiger_model import (
    TIGER_LEFT,
    TigerModel,
)
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import extract_nested_belief_hierarchy
from utils.paper_plots import generate_nested_sunburst_pdf
from utils.plotting import plot_episode_sunburst_slider


def run_20step_sunburst_experiment(
    output_dir: str = "results/bayes_optimal",
    seed: int = 42,
    n_sims: int = 2000,
    l1_prior: float = 0.8,
) -> pd.DataFrame:
    """Run full 20-step episode tracking the complete nested hierarchy."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(" Running 20-Step Opponent Modeling Sunburst & Belief Trajectory")
    print(
        f" Prior: {l1_prior:.0%} L1, {1.0 - l1_prior:.0%} L0 | MCTS Sims: {n_sims} | Seed: {seed}"
    )
    print("============================================================")

    model = TigerModel()
    rng = random.Random(seed)

    # 1. Initialize Protagonist (Level 2)
    bank_i = SolverBank(seed=seed + 100)
    boot_i = I_POMDP_Bootstrapper(bank_i)
    cfg_i = IPOMCPConfig(
        mcts=MCTSConfig(n_sims=n_sims, max_depth=2, node_capacity=200),
        opponent=OpponentPolicyConfig(n_sims=50),
    )
    agent_i = boot_i.create_level2_solver(
        "i", model, ["j"], config=cfg_i, n_particles=1000, l1_probability=l1_prior
    )

    # 2. Initialize Opponent (Level 1)
    bank_j = SolverBank(seed=seed + 200)
    boot_j = I_POMDP_Bootstrapper(bank_j)
    cfg_j = IPOMCPConfig(
        mcts=MCTSConfig(n_sims=n_sims, max_depth=2, node_capacity=200),
        opponent=OpponentPolicyConfig(n_sims=50),
    )
    agent_j = boot_j.create_level1_solver("j", model, ["i"], config=cfg_j, n_particles=1000)

    true_state = model.get_initial_state(rng=rng)
    print(f"Initial physical state: {true_state}")

    snapshots: Dict[int, Dict[str, Any]] = {0: extract_nested_belief_hierarchy(agent_i, "i", 2)}

    # Track metrics
    records: List[Dict[str, Any]] = []
    cum_r_i = 0.0
    cum_r_j = 0.0

    # Initial t=0 record
    snap0 = snapshots[0]
    p_l1_init = snap0["values"][2] if len(snap0["values"]) > 2 else l1_prior
    records.append(
        {
            "step": 0,
            "true_state": str(true_state),
            "action_i": "None",
            "action_j": "None",
            "obs_i": "None",
            "obs_j": "None",
            "reward_i": 0.0,
            "cum_reward_i": 0.0,
            "reward_j": 0.0,
            "cum_reward_j": 0.0,
            "prob_l1_j": p_l1_init,
            "prob_l0_j": 1.0 - p_l1_init,
            "belief_p_tl": 0.5,
        }
    )

    for t in range(1, 21):
        a_i = agent_i.get_action()
        a_j = agent_j.get_action()
        joint = {"i": a_i, "j": a_j}

        # Physical transition and rewards
        next_state = model.sample_transition(true_state, joint, rng=rng)
        r_i = model.get_reward(true_state, joint, next_state, "i")
        r_j = model.get_reward(true_state, joint, next_state, "j")
        cum_r_i += r_i
        cum_r_j += r_j

        # Private observations
        o_i = model.sample_observation(next_state, joint, "i", rng=rng)
        o_j = model.sample_observation(next_state, joint, "j", rng=rng)

        # Belief updates
        agent_i.update_root(a_i, o_i)
        agent_j.update_root(a_j, o_j)

        snap = extract_nested_belief_hierarchy(agent_i, "i", 2)
        snapshots[t] = snap

        # Extract subjective probabilities from agent_i's belief
        p_l1 = snap["values"][2] if len(snap["values"]) > 2 else 0.0
        p_l0 = snap["values"][1] if len(snap["values"]) > 1 else (1.0 - p_l1)

        # Physical belief P(TL) from planner belief
        p_tl = 0.5
        if agent_i.belief and agent_i.belief.mass:
            tl_mass = sum(m for atom, m in agent_i.belief.mass if atom.state == TIGER_LEFT)
            total_m = sum(m for _, m in agent_i.belief.mass)
            p_tl = tl_mass / total_m if total_m > 0 else 0.5

        records.append(
            {
                "step": t,
                "true_state": str(true_state),
                "action_i": str(a_i),
                "action_j": str(a_j),
                "obs_i": str(o_i),
                "obs_j": str(o_j),
                "reward_i": r_i,
                "cum_reward_i": cum_r_i,
                "reward_j": r_j,
                "cum_reward_j": cum_r_j,
                "prob_l1_j": p_l1,
                "prob_l0_j": p_l0,
                "belief_p_tl": p_tl,
            }
        )

        print(
            f"Step {t:2d}: a_i={str(a_i):2s}, a_j={str(a_j):2s} | "
            f"r_i={r_i:5.1f} (cum={cum_r_i:5.1f}) | "
            f"o_i={str(o_i):15s} | b_i(L1)={p_l1:6.1%} | b_i(TL)={p_tl:5.2f}"
        )

        true_state = next_state

    df = pd.DataFrame(records)
    csv_path = out_path / "episode_20step_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved episode records to {csv_path}")

    # Generate 20-step Animated Sunburst
    print("\nGenerating interactive 20-step Plotly Sunburst with scrubber slider...")
    plot_episode_sunburst_slider(
        snapshots_by_step=snapshots,
        title_prefix="Bayes-Optimal Opponent Modeling Hierarchy (T=20)",
        save_dir=str(out_path),
        filename="sunburst_optimal_animated_T20",
    )
    print(f"Exported: {out_path / 'sunburst_optimal_animated_T20.html'}")

    # Export Inflection Point Sunburst PDFs
    inflection_steps = [0, 2, 3, 8, 20]
    for s in inflection_steps:
        if s in snapshots:
            pdf_path = out_path / f"fig_sunburst_step_{s}.pdf"
            generate_nested_sunburst_pdf(
                snapshots[s],
                str(pdf_path),
                title=f"Opponent Hierarchy (Step t={s})",
            )
            print(f"Exported: {pdf_path}")

    # Generate 3-Panel Trajectory Plot
    generate_trajectory_figure(df, out_path)

    return df


def generate_trajectory_figure(df: pd.DataFrame, out_path: Path) -> None:
    """Generate publication-ready 3-panel episode belief trajectory figure."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    steps = df["step"]

    # Panel 1: Opponent Type Mass Distribution
    axes[0].plot(
        steps, df["prob_l1_j"] * 100.0, "b-o", linewidth=2.5, label="Modeled $j$ as Level-1 ($L_1$)"
    )
    axes[0].plot(
        steps,
        df["prob_l0_j"] * 100.0,
        "r--s",
        linewidth=2.0,
        label="Modeled $j$ as Level-0 ($L_0$)",
    )
    axes[0].axhline(80.0, color="gray", linestyle=":", alpha=0.7, label="Initial Prior (80% $L_1$)")
    axes[0].set_ylabel("Belief Mass (%)", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Opponent Mental Model Distribution $b_i(m_j)$ Over 20 Steps",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="center right", framealpha=0.9, fontsize=10)
    axes[0].set_ylim(-2, 105)

    # Panel 2: Physical State Belief P(Tiger Left)
    axes[1].plot(
        steps,
        df["belief_p_tl"],
        "g-^",
        linewidth=2.5,
        label="Agent $i$ Belief $P(s = \\text{Tiger Left})$",
    )
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    axes[1].axhline(
        0.92, color="orange", linestyle="--", alpha=0.7, label="Open Right Threshold (>= 0.92)"
    )
    axes[1].axhline(
        0.08, color="purple", linestyle="--", alpha=0.7, label="Open Left Threshold (<= 0.08)"
    )

    # Mark door openings
    door_opens = df[df["action_i"].isin(["OL", "OR", "OPEN_LEFT", "OPEN_RIGHT"])]
    for _, row in door_opens.iterrows():
        axes[1].annotate(
            f"Action: {row['action_i']}\n(R={row['reward_i']:+.0f})",
            xy=(row["step"], row["belief_p_tl"]),
            xytext=(
                row["step"],
                row["belief_p_tl"] + (0.15 if row["belief_p_tl"] < 0.5 else -0.22),
            ),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#fef08a", edgecolor="#ca8a04", alpha=0.9
            ),
        )

    axes[1].set_ylabel("Belief $P(\\text{TL})$", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Physical State Estimation & Door Opening Decisions", fontsize=13, fontweight="bold"
    )
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)

    # Panel 3: Cumulative Rewards
    axes[2].plot(
        steps, df["cum_reward_i"], "b-D", linewidth=2.5, label="Agent $i$ ($L_2$) Cumulative Return"
    )
    axes[2].plot(
        steps,
        df["cum_reward_j"],
        "m--o",
        linewidth=2.0,
        label="Agent $j$ ($L_1$) Cumulative Return",
    )
    axes[2].axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    axes[2].set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Episode Timestep $t$", fontsize=12, fontweight="bold")
    axes[2].set_title(
        f"Cumulative Returns across 20-Step Trajectory (Final: $R_i={df['cum_reward_i'].iloc[-1]:+.1f}$)",
        fontsize=13,
        fontweight="bold",
    )
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    traj_pdf = out_path / "fig_episode_belief_trajectory_T20.pdf"
    plt.savefig(traj_pdf)
    plt.close()
    print(f"Exported: {traj_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20-Step Sunburst & Trajectory Benchmark")
    parser.add_argument(
        "--output-dir", type=str, default="results/bayes_optimal", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--sims", type=int, default=2000, help="MCTS simulation budget")
    args = parser.parse_args()

    run_20step_sunburst_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        n_sims=args.sims,
    )
