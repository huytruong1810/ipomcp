import pytest
from core.config import ExperimentConfig
from examples.tiger.runners.tiger_batch_runner import TigerBatchRunner
from examples.tiger.runners.tiger_baseline_runner import TigerBaselineRunner
from examples.uav.runners.uav_batch_runner import UAVBatchRunner
from examples.wumpus.runners.wumpus_batch_runner import WumpusBatchRunner


def test_tiger_batch_runner_standardized_time_axis():
    exp_config = ExperimentConfig(n_trials=2, max_steps=3, export_trees=False, verbose=False)
    runner = TigerBatchRunner(config=exp_config)
    df = runner.run_batch()
    assert not df.empty
    # 2 trials * (3 + 1) steps = 8 rows
    assert len(df) == 8
    assert set(df["step"].unique()) == {0, 1, 2, 3}
    
    # Assert t=0 initial condition
    step0 = df[df["step"] == 0]
    assert (step0["cum_reward_i"] == 0.0).all()
    assert (step0["cum_reward_j"] == 0.0).all()
    assert (step0["reward_i"] == 0.0).all()
    assert (step0["reward_j"] == 0.0).all()


def test_tiger_baseline_runner_standardized_time_axis():
    exp_config = ExperimentConfig(n_trials=2, max_steps=3, export_trees=False, verbose=False)
    runner = TigerBaselineRunner(config=exp_config)
    df = runner.run_batch()
    assert not df.empty
    assert len(df) == 8
    
    step0 = df[df["step"] == 0]
    assert (step0["cum_reward_i"] == 0.0).all()
    assert (step0["cum_reward_j"] == 0.0).all()


def test_uav_batch_runner_short():
    exp_config = ExperimentConfig(n_trials=2, max_steps=2, export_trees=False, verbose=False)
    runner = UAVBatchRunner(config=exp_config)
    df = runner.run_batch()
    assert not df.empty
    assert len(df) == 6  # 2 trials * (2 + 1) steps
    step0 = df[df["step"] == 0]
    assert (step0["cum_reward_i"] == 0.0).all()


def test_wumpus_batch_runner_short():
    exp_config = ExperimentConfig(n_trials=2, max_steps=2, export_trees=False, verbose=False)
    runner = WumpusBatchRunner(config=exp_config)
    df = runner.run_batch()
    assert not df.empty
    assert len(df) == 6  # 2 trials * (2 + 1) steps
    step0 = df[df["step"] == 0]
    assert (step0["cum_reward_i"] == 0.0).all()
