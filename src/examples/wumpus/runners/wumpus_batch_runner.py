import os
from datetime import datetime

import pandas as pd

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from core.logger import get_logger
from examples.wumpus.model.constants import AGENT_HUMAN, AGENT_WUMPUS
from examples.wumpus.model.wumpus_model import WumpusModel
from solvers.exploration import NormalizedUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("WumpusRunner")


class WumpusBatchRunner(GenericBatchRunner):
    """
    Experimental Group: Executes Level-2 Human vs Level-1 Wumpus agents in Wumpus World.
    """

    def _setup_domain(self):
        env = WumpusModel(width=4, height=4, n_pits=2)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=5, node_capacity=500)
        opponent_cfg = OpponentPolicyConfig(n_sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, opponent=opponent_cfg)

        # 1. L1 Wumpus with dedicated SolverBank
        bank_w = SolverBank()
        boot_w = I_POMDP_Bootstrapper(bank_w)
        boot_w.create_level0_solver(AGENT_HUMAN, env)
        boot_w.create_level0_solver(AGENT_WUMPUS, env)
        planner_w = boot_w.create_level1_solver(
            AGENT_WUMPUS,
            env,
            [AGENT_HUMAN],
            n_particles=1000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )

        # 2. L2 Human with isolated SolverBank
        bank_h = SolverBank()
        boot_h = I_POMDP_Bootstrapper(bank_h)
        boot_h.create_level0_solver(AGENT_HUMAN, env)
        boot_h.create_level0_solver(AGENT_WUMPUS, env)
        boot_h.create_level1_solver(
            AGENT_WUMPUS,
            env,
            [AGENT_HUMAN],
            n_particles=1000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )
        planner_h = boot_h.create_level2_solver(
            AGENT_HUMAN,
            env,
            [AGENT_WUMPUS],
            l1_probability=0.9,
            n_particles=1000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )

        return env, planner_h, planner_w, env.get_initial_state()

    def _get_custom_metrics(self, state, next_state, env, planner_i=None, planner_j=None):
        hx, hy = next_state.human_pose.pos()
        wx, wy = next_state.wumpus_pose.pos()
        dist = abs(hx - wx) + abs(hy - wy)

        # Result Classification
        outcome = "Active"
        if next_state.has_gold:
            outcome = "Win_Gold"
        elif not next_state.human_alive:
            if next_state.wumpus_alive and (
                (hx, hy) == (wx, wy)
                or ((hx, hy) == state.wumpus_pose.pos() and (wx, wy) == state.human_pose.pos())
            ):
                outcome = "Death_Wumpus"
            else:
                outcome = "Death_Pit"

        wumpus_died = state.wumpus_alive and not next_state.wumpus_alive

        return {
            "distance": dist,
            "outcome": outcome,
            "wumpus_killed": wumpus_died,
            "has_arrow": next_state.has_arrow,
        }


# We append a custom plotter for the outcome distribution
def plot_wumpus_specifics(df: pd.DataFrame, save_dir: str = None):
    import os

    import plotly.express as px

    if df.empty or "outcome" not in df.columns:
        return

    # Outcome Pie Chart
    # We only care about the FINAL step of each trial
    final_steps = df.sort_values("step").groupby("trial").tail(1)

    counts = final_steps["outcome"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Outcome",
        title="Wumpus World Outcomes (Win vs Death Types)",
        color="Outcome",
        color_discrete_map={
            "Win_Gold": "green",
            "Death_Wumpus": "red",
            "Death_Pit": "black",
            "Active": "blue",
        },
    )

    if save_dir:
        fig.write_html(os.path.join(save_dir, "wumpus_outcomes.html"))


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("wumpus", f"wumpus_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=1000, max_steps=50, export_trees=False, verbose=False)

    logger.info("Initializing Wumpus Batch Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = WumpusBatchRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        # Standard Plots
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "Human (L2)", "j": "Wumpus (L1)"},
            title_prefix="Wumpus Domain:",
            save_dir=log_dir,
        )

        # Domain Specific Plots
        plot_wumpus_specifics(df, save_dir=log_dir)

        # Paper Plots
        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info("Wumpus Batch execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
