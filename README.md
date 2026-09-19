# Interactive POMCP research framework

This project studies finite hierarchies of agent models with Monte Carlo planning
in Tiger, UAV pursuit, and Wumpus domains. It contains an I-POMCP-style planner and
a sampled reachability-tree comparator. Both use immutable joint beliefs and a
recursive finite Bayesian filter. Finite priors and finite search remain explicit
approximations; passing tests does not prove optimality or convergence.

**The long full suite is not yet qualified.** All conditions have been exercised at full planning budgets; the matrix still
has model-support failures. Prior and controlled comparison conditions complete
the twenty-step qualification. Worker time/RSS supervision is now implemented. Read [the theory contract](docs/THEORY.md),
[open gates](BACKLOG.md), [benchmark evidence](docs/BENCHMARK.md), and
[handoff](HANDOFF.md). [The implementation plan](docs/IMPLEMENTATION_PLAN.md) tracks
completed work separately from outstanding qualification.

## Setup and validation

The supported execution environment is Linux, including WSL, with Python 3.12+.
Resource measurements use Linux `/proc`; unavailable measurements raise errors.
Install `uv`, then run from the repository root:

```bash
uv sync --frozen --group dev
uv run pytest -q -rxX
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv build --wheel

# Full planning budgets, one twenty-step trial for every actual condition.
uv run python benchmarks/qualify_suite.py --out results/qualification --steps 20
```

Graphviz tree rendering also requires the system `dot` executable. Python's
`graphviz` package is required; a failed rendering is reported rather than silently
omitted. Rendering is optional when tree export is disabled.

The current suite has 127 passing tests and no expected-failure theory exceptions.
Independent bounded Tiger references check filtering and one-step values; they
do not constitute a general convergence proof.

## Experiments

Level zero means **uniform-random behavior** in this codebase. It is not the
optimizing level-zero POMDP used in some I-POMDP literature. Higher levels model
mixtures over lower levels. Separate solver banks isolate the two real agents.
Tiger uses 85% growl accuracy and 100% creak accuracy in the principal experiments;
an opening resets the physical tiger before observations unless persistent mode
is explicitly selected. Its leaf rollout always listens, a finite-budget heuristic.

```bash
# One condition: L3 versus L2, with 80% prior mass on L2.
uv run python -m examples.experiments.deep_hierarchy_prior_experiment --trials 30 --steps 20 --condition 1

# Full-suite entry point; qualification blockers remain (see BACKLOG.md).
uv run python -m examples.experiments.run_benchmarks --trials 100 --steps 20 --suite all

# Resume exactly one suite, using its actual output directory.
uv run python -m examples.experiments.run_benchmarks --trials 100 --steps 20 --suite prior --resume-dir /absolute/run/directory

# Audit driver, explicitly binding the source checkout and output directory.
uv run python benchmarks/tiger_pre_post_benchmark.py --repo "$PWD" --out results/tiger-audit --trials 30 --steps 20 --workers 4
```

`run_benchmarks` supports `prior`, `comparison`, `matrix`, or `all`. Resuming `all`
with a single suite directory is rejected. Batch manifests bind source and
configuration; per-trial CSV checkpoints must contain every step, including step
zero and absorbing terminal padding. Completion markers bind final CSV bytes.
All batch and snapshot trials use process isolation and time/RSS limits from
ExperimentConfig. Failed attempts retain logs and diagnostics alongside successful
checkpoints. Incompatible old results require a fresh output directory; there is no schema
migration or compatibility reader.

Interaction horizon and planning depth are different quantities. The audit uses
20 environment decisions and depth-five search, with 20,000 L3 and 15,000 L2 root
simulations per decision. Initial physical sample counts are 2,000 and 1,500. Nested models share their own
bank's empirical physical prior; level weights are exact and search-node capacity
does not clip the live belief. Modeled MCTS policies use ten simulations by default,
which differs from the real-agent budgets and must be reported.

## Layout and result interpretation

`core/` defines probability, model, configuration, logging, and telemetry contracts.
`ipomdp/` holds particle/frame representations. `solvers/` implements search and
model propagation. `utils/` assembles hierarchies, executes batches, and renders
recorded data. `examples/` contains domains and experiment designs. `tests/`
contains regressions and bounded exact references. `benchmarks/` contains the
matched-run audit driver.

Results live under `results/`, or the `IPOMCP_RESULTS_DIR` override. Source control
ignores runtime results, caches, environments, and builds. Retain raw trial data,
configuration, source hashes, and execution logs when archiving a study; regenerate
figures from those records. The removed synthetic reward/sunburst generators are
not valid solver evidence. Their original source remains in Git history.

Reported cumulative rewards are undiscounted observed returns; planners optimize
a discounted, truncated objective. Compare independent trial totals with paired
seed analyses where designs share seeds. A confidence interval crossing zero is
not evidence of equivalence. Payoff matrices describe a finite policy set and do
not prove convergence as reasoning depth increases or an equilibrium of the
underlying partially observed game.
