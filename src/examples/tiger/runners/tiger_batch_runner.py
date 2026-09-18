import os
from datetime import datetime

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from core.logger import get_logger
from examples.tiger.model.tiger_model import TigerModel
from solvers.exploration import NormalizedUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("TigerRunner")


class TigerBatchRunner(GenericBatchRunner):
    """
    Experimental Group: Executes Level-2 (I-POMCP) vs Level-1 (I-POMCP) agents.
    """

    def _setup_domain(self):
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=6, node_capacity=500)
        opponent_cfg = OpponentPolicyConfig(n_sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, opponent=opponent_cfg)

        # 1. Opponent Agent J (Level-1 I-POMCP) with dedicated SolverBank
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver("i", TigerModel(growl_accuracy=growl_dict))
        planner_j = boot_j.create_level1_solver(
            "j",
            TigerModel(growl_accuracy=growl_dict),
            ["i"],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2 ** (0.5)),
        )

        # 2. Protagonist Agent I (Level-2 I-POMCP) with dedicated SolverBank (cache-isolated)
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver("j", TigerModel(growl_accuracy=growl_dict))
        boot_i.create_level1_solver(
            "j",
            TigerModel(growl_accuracy=growl_dict),
            ["i"],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2 ** (0.5)),
        )
        planner_i = boot_i.create_level2_solver(
            "i",
            TigerModel(growl_accuracy=growl_dict),
            ["j"],
            l1_probability=1.0,
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2 ** (0.5)),
        )

        return env, planner_i, planner_j, env.get_initial_state()


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("tiger", f"tiger_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=1000, max_steps=6, export_trees=False, verbose=False)

    logger.info("Initializing Tiger Batch Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = TigerBatchRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "Agent I (L2)", "j": "Agent J (L1)"},
            title_prefix="Tiger L2 vs L1",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)

        logger.info("Tiger Batch execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
