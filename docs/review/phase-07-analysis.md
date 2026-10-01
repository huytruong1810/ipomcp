# Phase 7 — Analysis and plotting (`src/utils/`)

Files reviewed in full (about 1,050 lines):
- `plotting.py`, `paper_plots.py`, `plot_metadata.py`, `visualizer.py`
- `regenerate_all_experiment_plots.py`, `statistics.py` (statistics was covered in Phase 6)
- The sunburst extractor `generic_batch_runner.extract_nested_belief_hierarchy` (also covered in Phase 6)

## Contribution 3 (interpretability) in the code today

- The **sunburst** shows, at each ring, the mass on each (agent, level) pair conditional on its parent path (`generic_batch_runner.py:96-138`). Each nested model of a given level is merged into one wedge.
- A plotly slider animates those wedges over steps, and a matplotlib PDF draws two rings.
- Level posteriors over time are plotted as means across trials.

## Findings

### Correctness

- **B1 — The "particle health" plot is always empty.** `_plot_belief_health` reads `n_particles_{i,j}` (`plotting.py:436`), but the runner writes `belief_support_{i,j}`. `paper_plots._plot_belief_support` reads the right column; the interactive version never draws anything.
- **B2 — "Compute-normalized reward" puts j's reward on i's clock.**
  - Both `plotting._plot_reward_vs_time` and `paper_plots._plot_compute_normalized_reward` bucket trials by **i's** cumulative planning time, then draw both i's and j's cumulative reward against it (`plotting.py:395-421`; `paper_plots.py:210-211,358-392`).
  - j's curve on i's clock is meaningless, and j's time is never recorded (Phase 6 B3).
- **B3 — The PDF sunburst draws only two rings** (`paper_plots.py:258-320`).
  - For L3 and L4 agents (rings: j's level → i's level → j's level → …), everything below depth 2 is dropped.
  - Contribution 3 is about *high* strategy levels, and the paper figure can't show them.
  - The plotly version shows all depths.
- **B4 — Level-posterior plots average over surviving trials only.**
  - They filter `status == "Active"` before averaging (`plotting.py:71`), which conditions on survival in terminating domains (UAV, Wumpus).
  - The cumulative-reward plot explicitly avoids that bias.
- **B5 — A stale hover string.** It says `"Particles: 0"` (`plotting.py:238`), a term from an old representation.
- **B6 — `_plot_comparative_reward` computes standard deviations but draws no interval** (`:372-381`). The HTML shows bare means.

### Interpretability (contribution 3)

- **I1 — The sunburst shows level structure, not mental content.**
  - Merging all nested models of a level into one wedge hides *what* each modeled agent believes: its physical belief, and its predicted action.
  - Two very different j/L1 beliefs (one confident TL, one confident TR) look identical.
  - The L3 study added per-level predicted actions (`predicted_opponent_actions`) and physical beliefs as separate columns, but not in the sunburst.
  - A useful interpretability view should colour or annotate wedges by the modeled agent's belief and by its predicted action distribution.
- **I2 — No error is attached to any interpretability output.**
  - The sunburst comes from the approximate belief. Its error against the exact hierarchy (nested-belief TV per depth, which the L3 ladder computed in `benchmarks/`) is not part of the plotting library.
  - "Interpretability with quantifiable error bounds" requires drawing exact vs approximate side by side, or showing per-wedge error.
- **I3 — Level posteriors are averaged across trials.** A mean of posteriors is not any agent's belief. Per-trial trajectories, or quantile bands, are more faithful. Labels should say "mean over trials".

### Structure

- **S1 — Two parallel plotting stacks** (plotly `plotting.py`, matplotlib/seaborn `paper_plots.py`) render the same metrics with separately coded bucketing and filtering logic. B2 is duplicated in both. Compute the metrics once in an analysis layer, then render.
- **S2 — A third sunburst implementation.** `benchmarks/l3_l2_figures.py` has its own sunburst grids. A fourth tree renderer lives in `visualizer.py`.
- **S3 — Global state at import.** `paper_plots` calls `plt.style.use(...)` and `sns.set_context(...)` when imported (`:170-173`). Several experiment modules call `matplotlib.use("Agg")` between imports.
- **S4 — `print` instead of logging** (`paper_plots.py:199,219`; `plotting.py:31,40`). `_maybe_save_and_show` opens a browser (`fig.show()`) when `save_dir` is None, which is surprising in batch code.
- **S5 — Parsing level numbers from column names** (`prob_l{k}_{agent}`, `plotting.py:84-85`, `paper_plots.py:227-228`) has a silent fallback to level 0. A long-format table (step, trial, agent, level, prob) removes both the parsing and the fallback.
- **S6 — `regenerate_all_experiment_plots`** catches all exceptions per directory, continues, then raises at the end. That's acceptable, but it only re-renders the two generic stacks.
- **S7 — `ForestVisualizer`** renders node visit counts only, with the depth hard-coded to 3. It is used by `Planner.visualize` (behind `export_trees`, which is off everywhere) and by `run_wumpus`.
- **S8 — `plot_metadata.agent_labels`** is a reasonable single source of label truth; it can stay.
