from utils.generic_batch_runner import GenericBatchRunner
from examples.tiger.model.tiger_model import TigerModel
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.plotting import plot_all_metrics
from utils.paper_plots import generate_paper_plots
from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, JITConfig
from core.logger import get_logger

import os
from datetime import datetime
from typing import Dict, Optional

logger = get_logger("TigerMixtureExperiment")


class TigerMixtureRunner(GenericBatchRunner):
    """
    Evaluates Level-2 Agent I against Level-1 Agent J under configurable opponent level priors.
    IS_{i,2} = S x (Theta_j^0 union Theta_j^1)
    """

    def __init__(self,
                 config: Optional[ExperimentConfig] = None,
                 log_dir: Optional[str] = None,
                 agent_i_level_weights: Optional[Dict[int, float]] = None):
        super().__init__(config=config, log_dir=log_dir)
        self.agent_i_level_weights = agent_i_level_weights or {1: 0.5, 0: 0.5}

    def _setup_domain(self):
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        bank = SolverBank()
        boot = I_POMDP_Bootstrapper(bank)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=6, node_capacity=500)
        jit_cfg = JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, jit=jit_cfg)

        # 1. Target Opponent (Agent J) - True Level-1 agent
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver("i", TigerModel(growl_accuracy=growl_dict))
        planner_j = boot_j.create_level1_solver(
            "j",
            TigerModel(growl_accuracy=growl_dict),
            ["i"],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
        )

        # 2. Protagonist (Agent I) - Level-2 agent with custom mixture prior over opponent levels
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver("j", TigerModel(growl_accuracy=growl_dict))
        boot_i.create_level1_solver(
            "j",
            TigerModel(growl_accuracy=growl_dict),
            ["i"],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
        )
        planner_i = boot_i.create_level2_solver(
            "i",
            TigerModel(growl_accuracy=growl_dict),
            ["j"],
            level_weights=self.agent_i_level_weights,
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
        )

        return env, planner_i, planner_j, env.get_initial_state()


def run_comparative_mixture_experiment(n_trials: int = 50, max_steps: int = 6):
    from core.paths import get_results_dir
    base_results = get_results_dir("tiger", f"tiger_mixture_comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False)

    conditions = [
        ("50_50_mixture", {0: 0.5, 1: 0.5}, "50% L0 / 50% L1 Prior"),
        ("90_10_mixture", {0: 0.1, 1: 0.9}, "10% L0 / 90% L1 Prior"),
        ("100_0_l1", {0: 0.0, 1: 1.0}, "100% L1 Deterministic Prior"),
    ]

    all_dfs = []
    for name, weights, desc in conditions:
        log_dir = os.path.join(base_results, name)
        logger.info(f"Running Condition: {desc} (N={n_trials}, T={max_steps})...")
        runner = TigerMixtureRunner(config=exp_config, log_dir=log_dir, agent_i_level_weights=weights)
        df = runner.run_batch()
        df["condition"] = desc
        all_dfs.append(df)

        plot_all_metrics(df, agent_labels={"i": f"Agent I ({desc})", "j": "Agent J (L1)"},
                         title_prefix=f"Tiger ({desc}):", save_dir=log_dir)
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)

    import pandas as pd
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(base_results, "combined_mixture_results.csv"), index=False)
    logger.info(f"Mixture experiment completed. Saved to {base_results}")


if __name__ == "__main__":
    run_comparative_mixture_experiment(n_trials=50, max_steps=6)
