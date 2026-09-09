# Absolute Path: <project_root>/core/config.py

"""
config.py — Immutable configuration schemas for the I-POMDP infrastructure.

Centralizes all algorithmic hyperparameters (MCTS constraints, JIT expansion thresholds,
particle reinvigoration criteria, and batch experiment configurations) into strictly
typed, frozen dataclasses. Supports serialization for experimental reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict
import json
import os

# Centralized, high-fidelity simulation and particle schedules across reasoning levels (L0-L4)
DEFAULT_SIM_SCHEDULE: Dict[int, int] = {
    0: 0,
    1: 50000,
    2: 100000,
    3: 150000,
    4: 200000
}

# Escalating particle budget to cover combinatorial union of nested lower-level opponent models
DEFAULT_PARTICLE_SCHEDULE: Dict[int, int] = {
    0: 0,
    1: 2500,
    2: 5000,
    3: 7500,
    4: 10000
}


@dataclass(frozen=True)
class JITConfig:
    """Hyperparameters gating the Variance-Driven JIT Expansion."""
    temperature: float = 0.5
    entropy_threshold: float = 0.5
    visit_threshold: int = 10
    sims: int = 50


@dataclass(frozen=True)
class MCTSConfig:
    """Core bounds and capacity constraints for Monte Carlo Tree Search."""
    gamma: float = 0.95
    max_depth: int = 20
    n_sims: int = 1000
    node_capacity: int = 500
    exploration_fallback_const: float = 1.0


@dataclass(frozen=True)
class ReinvigorationConfig:
    """Parameters governing the computational resuscitation of starved mental models."""
    enabled: bool = True
    visit_threshold: int = 20
    sims: int = 20


@dataclass(frozen=True)
class RTSConfig:
    """Hyperparameters governing Reachability Tree Sampling (RTS) exact branching baseline.
    
    Attributes:
        gamma: Discount factor applied to future time-horizon rewards.
        max_depth: Maximum tree depth for reachability tree expansion.
        obs_branching: Maximum top-k highest likelihood observations to branch across.
        num_particles: Particle population size maintained for belief resampling.
    """
    gamma: float = 0.95
    max_depth: int = 3
    obs_branching: int = 3
    num_particles: int = 100


@dataclass(frozen=True)
class IPOMCPConfig:
    """Master configuration for an Interactive POMCP Level-k solver.
    
    Attributes:
        mcts: Monte Carlo Tree Search constraints and hyperparameters.
        jit: Variance-Driven Just-In-Time mental sub-tree expansion criteria.
        reinvigoration: Resuscitation thresholds for mental models facing particle starvation.
    """
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    jit: JITConfig = field(default_factory=JITConfig)
    reinvigoration: ReinvigorationConfig = field(default_factory=ReinvigorationConfig)


@dataclass(frozen=True)
class ExperimentConfig:
    """Execution parameters for the GenericBatchRunner pipeline.
    
    Attributes:
        n_trials: Number of independent experimental Monte Carlo trials to execute.
        max_steps: Maximum step horizon per trial before forced termination.
        export_trees: Whether to serialize and render Graphviz search trees.
        verbose: Whether to output per-step JSON metric files to disk.
    """
    n_trials: int = 1000
    max_steps: int = 10
    export_trees: bool = False
    verbose: bool = False

    def save(self, filepath: str) -> None:
        """Persists the experiment configuration to disk for scientific reproducibility."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)