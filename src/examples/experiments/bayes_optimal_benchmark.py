"""Bayes-Optimal I-POMDP Mirroring Benchmark.

Compares I-POMCP against the exact analytical Bayes-optimal POMDP/I-POMDP solver
across value functions, action-value vectors, decision boundaries, and policy concordance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from examples.tiger.model.tiger_model import (
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper


def evaluate_ipomcp_at_belief(
    model: TigerModel,
    p_tl: float,
    depth: int,
    n_sims: int,
    seed: int = 42,
) -> Tuple[Any, Dict[Any, float]]:
    """Evaluate I-POMCP at an exact belief point p_tl = P(TIGER_LEFT)."""
    cfg = IPOMCPConfig(
        mcts=MCTSConfig(
            n_sims=n_sims,
            max_depth=depth,
            node_capacity=max(20, n_sims // 10),
        ),
        opponent=OpponentPolicyConfig(n_sims=10),
    )
    bank = SolverBank(seed=seed)
    boot = I_POMDP_Bootstrapper(bank)
    planner = boot.create_level1_solver("i", model, ["j"], config=cfg, n_particles=500)

    opp_frame = AgentFrame("j", 0, model)
    opp_model = MentalModel(opp_frame, None)
    mass = (
        (InteractiveState(TIGER_LEFT, opp_model), p_tl),
        (InteractiveState(TIGER_RIGHT, opp_model), 1.0 - p_tl),
    )
    planner.belief = FiniteBelief(mass)

    action = planner.get_action()
    q_vals = dict(planner.root.action_values)
    return action, q_vals


def run_mirroring_benchmark(
    horizon: int = 2,
    sim_budgets: Sequence[int] = (500, 2000, 10000, 25000),
    n_belief_points: int = 21,
    output_dir: str = "results/bayes_optimal",
    gamma: float = 0.95,
) -> pd.DataFrame:
    """Run full benchmarking suite comparing Exact VI vs I-POMCP."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(f" Running Bayes-Optimal Mirroring Benchmark (Horizon H={horizon})")
    print("============================================================")

    model = TigerModel()
    exact_solver = ExactPOMDPSolver(model, horizon=horizon, gamma=gamma)
    exact_solver.solve(horizon)

    belief_grid = np.linspace(0.02, 0.98, n_belief_points)
    exact_segments = exact_solver.decision_boundaries(horizon)
    print(f"Exact Policy Segments for Horizon {horizon}:")
    for seg in exact_segments:
        print(f"  {seg}")

    records: List[Dict[str, Any]] = []

    for n_sims in sim_budgets:
        print(
            f"\nEvaluating I-POMCP with N={n_sims} simulations across {n_belief_points} beliefs..."
        )
        for p in belief_grid:
            exact_q = exact_solver.q_values(p, horizon)
            exact_v = exact_solver.value(p, horizon)
            exact_a = exact_solver.policy(p, horizon)

            ipomcp_a, ipomcp_q = evaluate_ipomcp_at_belief(
                model=model,
                p_tl=float(p),
                depth=horizon,
                n_sims=n_sims,
                seed=int(p * 10000) + n_sims,
            )

            # Metrics
            action_match = int(ipomcp_a == exact_a)
            max_q_err = max(
                abs(ipomcp_q.get(a, 0.0) - exact_q[a]) for a in [LISTEN, OPEN_LEFT, OPEN_RIGHT]
            )
            v_err = abs(max(ipomcp_q.values()) - exact_v)

            records.append(
                {
                    "horizon": horizon,
                    "n_sims": n_sims,
                    "belief_p_tl": round(float(p), 4),
                    "exact_action": exact_a,
                    "ipomcp_action": ipomcp_a,
                    "action_match": action_match,
                    "exact_v": exact_v,
                    "ipomcp_v": max(ipomcp_q.values()),
                    "v_error": v_err,
                    "max_q_error": max_q_err,
                    "exact_q_listen": exact_q[LISTEN],
                    "ipomcp_q_listen": ipomcp_q.get(LISTEN, 0.0),
                    "exact_q_open_left": exact_q[OPEN_LEFT],
                    "ipomcp_q_open_left": ipomcp_q.get(OPEN_LEFT, 0.0),
                    "exact_q_open_right": exact_q[OPEN_RIGHT],
                    "ipomcp_q_open_right": ipomcp_q.get(OPEN_RIGHT, 0.0),
                }
            )

    df = pd.DataFrame(records)
    csv_file = out_path / "mirroring_metrics.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved raw benchmarking metrics to {csv_file}")

    # Generate Publication Figures
    generate_figures(df, exact_solver, horizon, out_path)

    # Print Summary Table
    print("\n--- Summary Performance by Simulation Budget ---")
    summary = (
        df.groupby("n_sims")
        .agg(
            concordance_pct=("action_match", lambda x: round(x.mean() * 100.0, 1)),
            mean_v_error=("v_error", lambda x: round(x.mean(), 3)),
            max_q_error=("max_q_error", lambda x: round(x.mean(), 3)),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    return df


def generate_figures(
    df: pd.DataFrame,
    exact_solver: ExactPOMDPSolver,
    horizon: int,
    out_path: Path,
) -> None:
    """Generate publication-ready comparison figures."""
    dense_p = np.linspace(0.0, 1.0, 300)
    exact_q_listen = [exact_solver.q_values(p, horizon)[LISTEN] for p in dense_p]
    exact_q_ol = [exact_solver.q_values(p, horizon)[OPEN_LEFT] for p in dense_p]
    exact_q_or = [exact_solver.q_values(p, horizon)[OPEN_RIGHT] for p in dense_p]

    # Figure 1: Q-Value Curves vs I-POMCP (N = max)
    max_sims = df["n_sims"].max()
    sub_df = df[df["n_sims"] == max_sims]

    plt.figure(figsize=(9, 5.5))
    plt.plot(dense_p, exact_q_listen, "b-", label="Exact Q*(b, Listen)", linewidth=2.0)
    plt.plot(dense_p, exact_q_ol, "r-", label="Exact Q*(b, Open Left)", linewidth=2.0)
    plt.plot(dense_p, exact_q_or, "g-", label="Exact Q*(b, Open Right)", linewidth=2.0)

    plt.scatter(
        sub_df["belief_p_tl"],
        sub_df["ipomcp_q_listen"],
        color="blue",
        marker="o",
        alpha=0.7,
        label=f"I-POMCP Q(Listen) [N={max_sims}]",
    )
    plt.scatter(
        sub_df["belief_p_tl"],
        sub_df["ipomcp_q_open_left"],
        color="red",
        marker="s",
        alpha=0.7,
        label=f"I-POMCP Q(Open Left) [N={max_sims}]",
    )
    plt.scatter(
        sub_df["belief_p_tl"],
        sub_df["ipomcp_q_open_right"],
        color="green",
        marker="^",
        alpha=0.7,
        label=f"I-POMCP Q(Open Right) [N={max_sims}]",
    )

    plt.xlabel("Belief $b(s = \\text{Tiger Left})$", fontsize=12)
    plt.ylabel("Action Value $Q(b, a)$", fontsize=12)
    plt.title(
        f"Bayes-Optimal vs I-POMCP Action Values (Horizon H={horizon})",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower center", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig1_pdf = out_path / "fig_exact_vs_ipomcp_q_curves.pdf"
    plt.savefig(fig1_pdf)
    plt.close()
    print(f"Exported figure: {fig1_pdf}")

    # Figure 2: Convergence of Mean Q-Error vs Simulation Budget
    conv = df.groupby("n_sims")["max_q_error"].mean().reset_index()
    plt.figure(figsize=(7, 4.5))
    plt.plot(conv["n_sims"], conv["max_q_error"], "ro-", linewidth=2.0, markersize=8)
    plt.xlabel("MCTS Simulations $N$", fontsize=12)
    plt.ylabel("Mean Q-Value Error $\\mathbb{E}[|Q_{\\text{MCTS}} - Q^*|]$", fontsize=12)
    plt.title(
        "Convergence to Bayes-Optimality vs Simulation Budget", fontsize=13, fontweight="bold"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig2_pdf = out_path / "fig_q_error_vs_simulations.pdf"
    plt.savefig(fig2_pdf)
    plt.close()
    print(f"Exported figure: {fig2_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayes-Optimal Mirroring Benchmark")
    parser.add_argument("--horizon", type=int, default=2, help="Planning horizon H")
    parser.add_argument("--quick", action="store_true", help="Run quick 2-budget check")
    parser.add_argument(
        "--output-dir", type=str, default="results/bayes_optimal", help="Output directory"
    )
    args = parser.parse_args()

    budgets = (500, 2000) if args.quick else (500, 2000, 10000, 25000)
    run_mirroring_benchmark(
        horizon=args.horizon,
        sim_budgets=budgets,
        n_belief_points=11 if args.quick else 21,
        output_dir=args.output_dir,
    )
