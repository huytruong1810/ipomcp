"""The sampled reachability-tree baseline: construction, values and batch execution."""

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.batch_runner import GenericBatchRunner
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.config import ExperimentConfig
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey

RTS = RTSConfig(max_depth=2, obs_branching=2, num_particles=30)


def test_level1_rts_planner_values_cover_every_action():
    env = TigerModel()
    planner = HierarchyBuilder(SolverBank()).build("i", 1, env, "j", config=RTS, prior_samples=30)
    assert isinstance(planner, RTSPlanner)
    assert planner.key == SolverKey("i", 1)
    assert planner.get_action() in env.get_all_actions("i")
    values = planner.get_action_values()
    assert set(values) == set(env.get_all_actions("i"))
    assert all(isinstance(q, float) for q in values.values())
    stats = planner.get_detailed_stats()
    assert stats["belief_size"] == len(planner.belief.mass)


def test_level2_rts_planner_models_declared_mcts_opponents():
    env = TigerModel()
    bank = SolverBank()
    planner = HierarchyBuilder(bank).build(
        "i",
        2,
        env,
        "j",
        config=RTS,
        modeled_config=PlannerConfig(simulations=20, depth=1, modeled_simulations=20),
        level_weights={0: 0.3, 1: 0.7},
        prior_samples=30,
    )
    assert isinstance(planner, RTSPlanner)
    assert planner.get_action() in env.get_all_actions("i")


class RTSVersusRandomRunner(GenericBatchRunner):
    def _setup_domain(self):
        env = TigerModel()
        builder = HierarchyBuilder(SolverBank())
        planner_i = builder.build("i", 1, env, "j", config=RTS, prior_samples=20)
        planner_j = builder.build("j", 0, env, "i")
        return env, planner_i, planner_j


def test_rts_batch_runner_execution(tmp_path):
    runner = RTSVersusRandomRunner(ExperimentConfig(n_trials=2, max_steps=3), str(tmp_path))
    df = runner.run_batch(max_workers=2)
    assert len(df) == 8  # 2 trials * (max_steps + 1)
    assert df["belief_support_i"].iloc[0] == 2
    assert {"prob_l0_j", "planning_cpu_i", "planning_cpu_j", "update_cpu_i"} <= set(df.columns)
