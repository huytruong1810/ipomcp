"""
config.py — Immutable configuration schemas for the I-POMDP infrastructure.

Centralizes all algorithmic hyperparameters (MCTS constraints, JIT expansion thresholds,
particle reinvigoration criteria, and batch experiment configurations) into strictly
typed, frozen dataclasses. Supports serialization for experimental reproducibility.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

# Centralized, memory-safe simulation and particle schedules across reasoning levels (L0-L4)
DEFAULT_SIM_SCHEDULE: Dict[int, int] = {0: 0, 1: 10000, 2: 15000, 3: 20000, 4: 25000}

# Balanced particle budget across reasoning levels (L0-L4)
DEFAULT_PARTICLE_SCHEDULE: Dict[int, int] = {0: 0, 1: 1000, 2: 1500, 3: 2000, 4: 2000}


@dataclass(frozen=True)
class JITConfig:
    """Hyperparameters gating the Entropy-Gated JIT Expansion."""

    temperature: float = 0.5
    entropy_threshold: float = 0.6
    visit_threshold: int = 5
    sims: int = 10

    def __post_init__(self):
        _positive(self.temperature, "temperature")
        _unit(self.entropy_threshold, "entropy_threshold")
        _integer(self.visit_threshold, "visit_threshold", minimum=0)
        _integer(self.sims, "sims")


@dataclass(frozen=True)
class MCTSConfig:
    """Core bounds and capacity constraints for Monte Carlo Tree Search."""

    gamma: float = 0.95
    max_depth: int = 20
    n_sims: int = 1000
    node_capacity: int = 500
    exploration_const: float = 1.0

    def __post_init__(self):
        _unit(self.gamma, "gamma")
        _integer(self.max_depth, "max_depth")
        _integer(self.n_sims, "n_sims")
        _integer(self.node_capacity, "node_capacity")
        _positive(self.exploration_const, "exploration_const")


@dataclass(frozen=True)
class ReinvigorationConfig:
    """Parameters governing the computational resuscitation of starved mental models."""

    enabled: bool = True
    visit_threshold: int = 20
    sims: int = 20
    min_particles: int = 100
    preserve_levels: bool = True

    def __post_init__(self):
        _integer(self.visit_threshold, "visit_threshold", minimum=0)
        _integer(self.sims, "sims")
        _integer(self.min_particles, "min_particles")


@dataclass(frozen=True)
class RTSConfig:
    """Hyperparameters governing Reachability Tree Sampling (RTS) sampled lookahead baseline.

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

    def __post_init__(self):
        _unit(self.gamma, "gamma")
        _integer(self.max_depth, "max_depth")
        _integer(self.obs_branching, "obs_branching")
        _integer(self.num_particles, "num_particles")


@dataclass(frozen=True)
class IPOMCPConfig:
    """Master configuration for an Interactive POMCP Level-k solver.

    Attributes:
        mcts: Monte Carlo Tree Search constraints and hyperparameters.
        jit: Entropy-Gated Just-In-Time mental sub-tree expansion criteria.
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
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def __post_init__(self):
        _integer(self.n_trials, "n_trials")
        _integer(self.max_steps, "max_steps", minimum=0)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _unit(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
