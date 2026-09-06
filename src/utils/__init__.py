"""
utils — Utility modules for I-POMCP experiment execution and analysis.

Exports:
* :class:`I_POMDP_Bootstrapper` — Factory for solver instantiation and registration.
* :class:`GenericBatchRunner` — Multiprocessing experiment runner and metric aggregator.
* :func:`plot_all_metrics` — Interactive Plotly chart generator.
* :func:`generate_paper_plots` — Static IEEE/ACM publication figure generator.
* :class:`ForestVisualizer` — Graphviz I-POMCP belief forest rendering.
"""

from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner, extract_nested_belief_hierarchy
from utils.plotting import plot_all_metrics, plot_nested_belief_sunburst, plot_episode_sunburst_slider
from utils.paper_plots import generate_paper_plots, generate_nested_sunburst_pdf
from utils.visualizer import ForestVisualizer

__all__ = [
    "I_POMDP_Bootstrapper",
    "GenericBatchRunner",
    "extract_nested_belief_hierarchy",
    "plot_all_metrics",
    "plot_nested_belief_sunburst",
    "plot_episode_sunburst_slider",
    "generate_paper_plots",
    "generate_nested_sunburst_pdf",
    "ForestVisualizer",
]
