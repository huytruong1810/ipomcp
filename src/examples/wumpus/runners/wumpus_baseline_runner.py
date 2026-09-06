from utils.generic_batch_runner import GenericBatchRunner
from examples.wumpus.model.wumpus_model import WumpusModel
from examples.wumpus.model.constants import AGENT_HUMAN, AGENT_WUMPUS
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.plotting import plot_all_metrics
from examples.wumpus.runners.wumpus_batch_runner import plot_wumpus_specifics
from utils.paper_plots import generate_paper_plots
from core.config import ExperimentConfig
from core.logger import get_logger

import os
from datetime import datetime

logger = get_logger("WumpusBaseline")


class WumpusBaselineRunner(GenericBatchRunner):
    """
    Control Group: Executes Level-0 (Random) vs Level-0 (Random) agents.
    Used to establish the performance floor of the Wumpus domain.
    """

    def _setup_domain(self):
        env = WumpusModel(width=4, height=4, n_pits=2)
        # 1. Register L0 Solver for Agent Wumpus
        bank_w = SolverBank()
        boot_w = I_POMDP_Bootstrapper(bank_w)
        key_w = boot_w.create_level0_solver(AGENT_WUMPUS, env)
        planner_w = bank_w.get_solver(key_w)

        # 2. Register L0 Solver for Agent Human
        bank_h = SolverBank()
        boot_h = I_POMDP_Bootstrapper(bank_h)
        key_h = boot_h.create_level0_solver(AGENT_HUMAN, env)
        planner_h = bank_h.get_solver(key_h)

        return env, planner_h, planner_w, env.get_initial_state()

    def _get_custom_metrics(self, state, next_state, env, planner_i=None, planner_j=None):
        hx, hy = state.human_pose.pos()
        wx, wy = state.wumpus_pose.pos()
        dist = abs(hx - wx) + abs(hy - wy)

        outcome = "Active"
        if next_state.has_gold:
            outcome = "Win_Gold"
        elif not next_state.human_alive:
            if state.wumpus_alive and (hx, hy) == (wx, wy):
                outcome = "Death_Wumpus"
            else:
                outcome = "Death_Pit"

        wumpus_died = (state.wumpus_alive and not next_state.wumpus_alive)

        return {
            "distance": dist,
            "outcome": outcome,
            "wumpus_killed": wumpus_died
        }


if __name__ == "__main__":
    from core.paths import get_results_dir
    log_dir = get_results_dir("wumpus", f"wumpus_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(
        n_trials=1000,
        max_steps=50,
        export_trees=False,
        verbose=False
    )

    logger.info("Initializing Wumpus Baseline Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = WumpusBaselineRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(df,
                         agent_labels={'i': 'Human (Random - L0)', 'j': 'Wumpus (Random - L0)'},
                         title_prefix="Wumpus Baseline:",
                         save_dir=log_dir)

        plot_wumpus_specifics(df, save_dir=log_dir)
        logger.info("Generating IEEE/ACM compliant vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info("Wumpus Baseline execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")