import os
from datetime import datetime

from core.config import ExperimentConfig
from core.logger import get_logger
from examples.tiger.model.tiger_model import TigerModel
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("TigerBaseline")


class TigerBaselineRunner(GenericBatchRunner):
    """
    Control Group: Executes Level-0 (Random) vs Level-0 (Random) agents.
    Used to establish the performance floor of the Tiger domain.

    Logging supports random policies without allocating unused MCTS roots.
    """

    def _setup_domain(self):
        env = TigerModel(creak_accuracy=1.0)
        bank = SolverBank()
        boot = I_POMDP_Bootstrapper(bank)

        # Level-0 agents do not require tree searches, they just act randomly
        key_i = boot.create_level0_solver("i", env)
        key_j = boot.create_level0_solver("j", env)

        planner_i = bank.get_solver(key_i)
        planner_j = bank.get_solver(key_j)

        return env, planner_i, planner_j, env.get_initial_state()


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("tiger", f"tiger_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Instantiate strict execution configuration
    exp_config = ExperimentConfig(n_trials=1000, max_steps=10, export_trees=False, verbose=False)

    logger.info("Initializing Tiger Baseline Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = TigerBaselineRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "Agent I (L0)", "j": "Agent J (L0)"},
            title_prefix="Tiger Baseline",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)

        logger.info("Tiger Baseline execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
