"""Backward-compatibility proxy for planner_comparison_experiment.

The canonical implementation now resides in `examples.experiments.planner_comparison_experiment`.
This module re-exports all public symbols to ensure external callers and tests continue
to function without modification.
"""

from examples.experiments.planner_comparison_experiment import (
    ControlledConditionRunner,
    experiment_conditions,
    run_planner_comparison,
)

__all__ = [
    "ControlledConditionRunner",
    "experiment_conditions",
    "run_planner_comparison",
]
