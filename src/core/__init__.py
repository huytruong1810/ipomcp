"""
core — Abstract base classes for POMDP/I-POMDP environments.

This package defines the domain-agnostic interfaces that every concrete
environment (Tiger, UAV, Wumpus, …) must implement:

* :class:`POMDPModel`  – Transition, observation, and reward functions.
* :class:`Distribution` / :class:`ParticleDistribution` / :class:`DictDistribution`
  – Generic probability-distribution abstractions used by the belief module.

Type aliases (:data:`State`, :data:`Action`, :data:`Observation`, :data:`AgentID`)
are also exported from here so that downstream modules can import them from a
single canonical location.
"""

from core.pomdp_model import POMDPModel, State, Action, Observation, AgentID
from core.distribution import Distribution, ParticleDistribution, DictDistribution
from core.config import (
    JITConfig,
    MCTSConfig,
    ReinvigorationConfig,
    IPOMCPConfig,
    RTSConfig,
    ExperimentConfig,
    DEFAULT_SIM_SCHEDULE,
    DEFAULT_PARTICLE_SCHEDULE,
)
from core.logger import get_logger

__all__ = [
    "POMDPModel",
    "State",
    "Action",
    "Observation",
    "AgentID",
    "Distribution",
    "ParticleDistribution",
    "DictDistribution",
    "JITConfig",
    "MCTSConfig",
    "ReinvigorationConfig",
    "IPOMCPConfig",
    "RTSConfig",
    "ExperimentConfig",
    "DEFAULT_SIM_SCHEDULE",
    "DEFAULT_PARTICLE_SCHEDULE",
    "get_logger",
]
