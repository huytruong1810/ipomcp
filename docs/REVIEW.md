# Phased review and file coverage

The review covers all repository source scripts, tests, experiment entry points,
and the three local untracked generators captured in the starting snapshot. It
excludes installed dependencies, caches, and generated build copies. Runtime
artifact provenance is reviewed separately from source correctness. A reviewed
file can still have an explicitly open defect; inspection is not verification of
the entire algorithm.

| Phase | Coherent module | Main conclusion |
|---|---|---|
| 1 | Repository, packaging, notes, imports | WSL is authoritative; wheel package discovery fixed; stale maturity claims removed. |
| 2 | Probability, configuration, model and telemetry contracts | Reject invalid mass, preserve tiny probabilities, prevent caller mutation, fail visibly on missing measurements. |
| 3 | Beliefs, identities, registry and bootstrap | Stable shallow identity and read-only maps; actual subjective beliefs remain mutable shared search nodes. |
| 4 | Search, nested simulation and sampled lookahead | Root drift/double routing and terminal mass fixed; Bayesian and nested-history counterexamples remain. |
| 5 | Tiger, UAV and Wumpus domains | Information restrictions and transition-event observations corrected; Tiger rollout heuristic retained explicitly. |
| 6 | Episode execution and experimental design | One snapshot/batch path, strict resume, atomic trial files; seed-paired statistics and explicit suite routing. |
| 7 | Figures, synthetic scripts and result artifacts | Synthetic generators removed; real hierarchy mixtures aggregated; misleading oracle/optimality labels removed. |
| 8 | Unit references and integrated experiments | Exact bounded references plus new regressions; source-bound long Tiger comparison; no unqualified theorem or performance claim. |

## Cross-component assessment

The most consequential failure crosses representation, filtering, and planning:
an intentional model is represented by a mutable search-node pointer, JIT mutates
that node while it is being treated as latent state, the transition copies deeper
models without their private updates, and root reconstruction loses correlations
before the next search. Improving local containers or increasing simulations cannot
make that composite operator equal the stated interactive Bayes operator.

Likewise, a plotted level probability is only the frequency in the represented
heuristic particle population. Prior floors and epoch preservation can make it
appear stable without showing calibrated inference. A high reward or preserved
80% L2 mass is not proof that the inferred type is correct. The independent
counterexamples in `tests/test_theory_gaps.py`
make these limits executable.

An exact-oracle claim would have contaminated downstream speedup/optimality plots.
The sampled comparator is relabeled, terminal continuation is corrected, and
reward ratios and action agreement across different closed-loop histories are
removed. Matched budgets, complete nested propagation, and approximation-error
analysis remain necessary before stronger conclusions.

DRY/KISS improvements include the unified episode engine, shared figure-label
resolver, configuration-only RTS constructor, and removal of unreachable model
interface fallbacks. SOLID boundaries improve with explicit model contracts, but
the solver/node/belief coupling still violates a clean separation of concerns.
The patch is a validated hardening step and complete inspection record, not a
claim that every requested architectural/theoretical remediation is finished.

## Runtime and artifact decisions

Original ignored results are preserved as evidence in the authoritative checkout.
The three deleted `generate_*` scripts synthesized policies, likelihoods, and
trajectories without invoking the planner. They cannot substantiate solver reward
or belief-convergence claims. Removing them from the active tree avoids accidental
reuse; the snapshot commit preserves their provenance without maintaining a second
schema or compatibility code path.

Parent links are weak to prevent retained descendants from owning discarded
siblings. This addresses one reference path; it is not proof of leak-free total
memory. Per-node reservoirs bound local storage, not all nested search trees.
Worker lifetimes are bounded by explicit trial waves, and memory telemetry reports
current Linux RSS. Swap occupancy is not described as measured swap thrashing.

## Detailed module findings and acceptance boundaries

### Foundations

`distribution.py`: zero total mass now raises instead of silently manufacturing a
uniform distribution. CDF sampling uses the upper boundary so a zero-valued RNG
draw cannot select a leading zero-weight item. Tiny positive weights retain their
relative odds; `__getitem__` returns normalized mass even before `normalize()`.
Caller lists and returned lists cannot invalidate cached lookup state. Empty
support is representable but cannot be sampled. Systematic resampling has an
ordering-dependent variance; it is not universally minimum-variance resampling.

`config.py`: finite probabilities and positive integer budgets are validated at
construction. The unused reinvigoration `alpha` parameter is removed. Configuration
defaults still encode experimental assumptions; frozen dataclasses alone do not
prove compatible settings across a hierarchy. `paths.py` confines relative result
subpaths to the configured result root. `logger.py` closes old file handlers when
rerouted; standard file handlers are not a multiprocess atomic logging protocol.

`telemetry.py`: current RSS comes from `/proc/self/statm`; a failed read no longer
substitutes peak RSS. Missing memory/load measurements no longer become healthy
zeroes. On WSL, these are measurements of the Linux guest, not all Windows host
memory. `MemoryWatchdog` returns advisory flags and may collect garbage; callers
must act on the flags. Its existence does not guarantee prevention of OOM. The
full search still needs explicit resource budgets for unrestricted worker counts.

### Beliefs, construction and solver identity

`belief.py`: model mappings are defensively copied and read-only; hashes preserve
actual agent keys rather than stringifying them. Frame identity includes the
physical-model instance. Pickling reconstructs mappings. This is *shallow*
immutability: nodes and domain objects remain mutable. Equality is representation
identity, not behavioral or belief equivalence; it must not justify model merging.

`solver_bank.py` and `solver_types.py`: a key selects a planner implementation at a
level, not a complete intentional frame with its preferences, horizon, policy
parameters, or private belief. Reusing a bank under conflicting configurations
still needs a strict consistency contract. `bootstrapper.py` validates prior
support and weights before allocating invalid hierarchies. Convenience constructors
delegate to general construction. Initial root pooling is an allocation strategy;
it does not establish O(NL) complexity of complete online search or safe merging of
different future histories.

### Search and inference

`node.py`: Algorithm R is correct for its routed stream and capacity is validated.
Weak parent links remove one unwanted retention path. Neither fact establishes
the correctness of the stream itself. `i_pomcp.py`: root drift and duplicate child
routing are fixed, repeated frame/weight allocations are hoisted out of rebuilding
loops, and unreachable interface fallback code is removed. The epoch reset,
pseudocount/floor heuristics, canonical history selection, and factorized rebuilding
remain the main P0 findings. Tests that merely demand preservation of an 80% prior
are explicitly insufficient as inference tests.

`generative_model.py`: action sampling, private observation generation, shared-node
JIT updates, and deeper model copying were inspected together. Entropy is not
variance, so comments use entropy-gated terminology. The policy mismatch and frozen
nested-history defects are documented, not hidden behind an unsupported repair.
`exploration.py`: positive finite configuration and initial-action behavior are
covered; adaptive Q normalization is a heuristic and its limiting theorem is not
established here. `random_planner.py`: copies its action list and rejects empty
support; random play is an empirical comparator, not a guaranteed reward floor.

`rts_planner.py`: terminal returns and surviving observation mass are corrected.
Configuration no longer has duplicate per-argument overrides, and raw-state
compatibility branches are removed. Top-k branches still omit continuation mass;
private models still do not receive a complete subjective recursive filter, and
zero-weight retention remains a theoretical failure path. The duplicated predictive
propagation/filtering blocks should be replaced by the specified common filter,
not mechanically merged while their semantics are unresolved.

### Domains and experimental execution

Tiger's sampling and likelihood kernels normalize over every listed observation
for all action pairs in exact tests, including creak accuracy 0, 0.4, and 1. The
always-listen rollout never reads hidden state. Opening rewards use the old tiger;
growls after a reset use the new tiger. Persistent and reset variants answer
different questions and must not be pooled. Unsupported sensor inputs now fail.

UAV boundary moves already clamp to the grid; pruning them using a hidden particle
position exposed information through available actions. All moves remain available.
Unknown observation tokens have zero likelihood. Custom distances use post-state
positions. Wumpus bump/scream flags distinguish transition events; death beats a
simultaneous grab, absorbing rewards are zero, and swap collisions are recognized.
Thin terminal visualization scripts still contain independent setup loops and
different example budgets; they were read and imported, not benchmarked as though
identical to the unified batch engine.

`generic_batch_runner.py`: planner and environment RNG streams are separate;
snapshot capture uses the same episode path without extra RNG draws. Environment
streams are not counter-based, so changed action-dependent draw consumption can
weaken paired CRN alignment. Terminal metrics are recorded immediately and carried
through padding. Resume binds source/configuration, validates rectangular trial
panels, and writes trial files atomically. Explicit worker waves limit process
lifetime without automatic recycling while a queue is pending. A one-worker run
uses the same trial method directly for deterministic debugging.

`master_nested_ipomdp_benchmark.py`: paired statistics now join trial IDs, reject
missing final seeds, and use Holm adjustment across the condition pair family.
Reported Wilcoxon p-values remain unadjusted descriptive secondary statistics.
`level_convergence_matrix_experiment.py`: an agent cannot represent an equal or
higher opponent level under the finite hierarchy; its so-called exact prior clips
to the largest supported lower level. This is model misspecification, not exact
knowledge. Pure empirical best responses are not statistical equilibrium proofs.
`planner_comparison.py`: removes reward-ratio optimality percentages and action
agreement between unequal closed-loop histories. The comparison still needs
matched opponent semantics and approximation error estimates before scientific
claims of efficiency relative to a reference algorithm.

### Figures, packaging and verification

Hierarchy extraction integrates each private model conditional on its parent
mass, independent of that model's reservoir size. It rejects multi-opponent
sunbursts that would misrepresent separate marginals as exclusive branches.
Reward curves retain terminal padding. Time buckets select one row per trial
before aggregation. Agent labels derive from explicit labels or recorded level
columns rather than guessing scientific metadata from directory names. Figure
batch errors result in a non-successful run, rather than a misleading all-clear.

The build includes all nested domain packages; every Python module in the wheel
was imported from the extracted wheel. Formatting/linting checks cover source,
tests, and benchmark scripts. Regression tests cover actual failure modes and
small analytic values; the exact reference is restricted to fixed-policy Tiger.
The long experiment compares complete paired trial totals, not per-step rows
treated as independent samples. Runtime probes alternate source order in fresh
processes and record CPU time separately from shared-machine wall time.

## Per-file ledger

| File | Phase | Review disposition |
|---|---:|---|
| `.gitignore` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `.python-version` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `BACKLOG.md` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `HANDOFF.md` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `README.md` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `benchmarks/tiger_pre_post_benchmark.py` | 8 | Matched seed/horizon experiment driver; source hashes bind each run. No equivalence claim without margin. |
| `pyproject.toml` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
| `recreate_venv.sh` | 1 | Removed: destructive environment recreation and unrelated dependencies; use uv sync. |
| `src/core/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/core/config.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/core/distribution.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/core/logger.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/core/paths.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/core/pomdp_model.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/core/telemetry.py` | 2 | Reviewed numerical/model/configuration contracts; validation and failure reporting hardened. |
| `src/examples/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/experiments/deep_hierarchy_prior_experiment.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/experiments/generate_cumulative_reward_plots.py` | 7 | Removed: synthetic policies/probabilities; not solver evidence. |
| `src/examples/experiments/generate_lv3_vs_lv2_sunburst.py` | 7 | Removed: synthetic policies/probabilities; not solver evidence. |
| `src/examples/experiments/generate_lv5_vs_lv1_sunburst.py` | 7 | Removed: synthetic policies/probabilities; not solver evidence. |
| `src/examples/experiments/level_convergence_matrix_experiment.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/experiments/master_nested_ipomdp_benchmark.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/experiments/run_all_large_scale_benchmarks_N200.py` | 7 | Renamed generic benchmark entry point; trial count belongs in CLI configuration. |
| `src/examples/experiments/run_benchmarks.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/tiger/model/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/tiger/model/tiger_model.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/tiger/runners/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/tiger/runners/apples_to_apples_oracle_benchmark.py` | 4 | Renamed to remove unsupported exact-oracle claim; sampled RTS limitations remain. |
| `src/examples/tiger/runners/persistent_tiger_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/runners/planner_comparison.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/runners/tiger_baseline_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/runners/tiger_batch_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/runners/tiger_level3_experiment.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/tiger/runners/tiger_mixture_experiment.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/uav/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/uav/model/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/uav/model/uav_model.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/uav/model/uav_viz.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/uav/runners/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/uav/runners/run_rts.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/uav/runners/run_uav.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/uav/runners/uav_baseline_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/uav/runners/uav_batch_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/wumpus/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/wumpus/model/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/wumpus/model/constants.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/wumpus/model/wumpus_model.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/wumpus/model/wumpus_state.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/wumpus/model/wumpus_viz.py` | 5 | Reviewed dynamics, likelihood support, observation timing, and information available to policies; domain regressions added. |
| `src/examples/wumpus/runners/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/examples/wumpus/runners/run_wumpus.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/wumpus/runners/wumpus_baseline_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/examples/wumpus/runners/wumpus_batch_runner.py` | 6 | Reviewed budgets, priors, statistics, resume and labels. Thin interactive demos still duplicate setup (P1). |
| `src/ipomdp/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/ipomdp/belief.py` | 3 | Reviewed identity, construction, and sharing. Mapping safety improved; mutable nested model identity remains P0. |
| `src/solvers/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/solvers/exploration.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/generative_model.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/i_pomcp.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/node.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/planner.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/random_planner.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/rts_planner.py` | 4 | Reviewed search and update mechanics. Root routing/terminal returns fixed; interactive filter/policy limitations remain P0. |
| `src/solvers/solver_bank.py` | 3 | Reviewed identity, construction, and sharing. Mapping safety improved; mutable nested model identity remains P0. |
| `src/solvers/solver_types.py` | 3 | Reviewed identity, construction, and sharing. Mapping safety improved; mutable nested model identity remains P0. |
| `src/utils/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `src/utils/bootstrapper.py` | 3 | Reviewed identity, construction, and sharing. Mapping safety improved; mutable nested model identity remains P0. |
| `src/utils/generic_batch_runner.py` | 6 | Unified episode path, source/config manifest, rectangular checkpoints, atomic output, bounded worker waves. |
| `src/utils/paper_plots.py` | 7 | Reviewed plot estimands and artifacts. Weighted hierarchy, JSON fields, terminal rows, and explicit failures corrected. |
| `src/utils/plot_metadata.py` | 7 | Reviewed plot estimands and artifacts. Weighted hierarchy, JSON fields, terminal rows, and explicit failures corrected. |
| `src/utils/plotting.py` | 7 | Reviewed plot estimands and artifacts. Weighted hierarchy, JSON fields, terminal rows, and explicit failures corrected. |
| `src/utils/regenerate_all_experiment_plots.py` | 7 | Reviewed plot estimands and artifacts. Weighted hierarchy, JSON fields, terminal rows, and explicit failures corrected. |
| `src/utils/visualizer.py` | 7 | Reviewed plot estimands and artifacts. Weighted hierarchy, JSON fields, terminal rows, and explicit failures corrected. |
| `tests/__init__.py` | 1 | Reviewed exports/import effects; utils no longer eagerly imports plotting. |
| `tests/reference_tiger.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_distribution.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_exploration.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_integration.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_models.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_node.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_oracle_rts.py` | 4 | Renamed to remove unsupported exact-oracle claim; sampled RTS limitations remain. |
| `tests/test_reinvigoration.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_review_regressions.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_rts.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_search_and_domain_contracts.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_solvers.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_telemetry.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_theory_gaps.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_tiger_reference.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `tests/test_visualization.py` | 8 | Reviewed assertion strength. Includes deterministic regressions, exact fixed-policy Tiger reference, and explicit theory xfails. |
| `uv.lock` | 1 | Reviewed configuration/documentation; replaced stale assurances with current contracts and limitations. |
