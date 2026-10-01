# Phase 6 — Experiment infrastructure (`src/utils/` execution, `src/examples/**/runners`, `src/examples/experiments`)

Files reviewed in full (about 5,900 lines):
- **Execution:** `utils/generic_batch_runner.py`, `process_supervisor.py`, `source_lock.py`, `statistics.py`, `bootstrapper.py`
- **Tiger runners (6):** `tiger_batch_runner`, `tiger_baseline_runner`, `planner_comparison` (proxy), `persistent_tiger_runner`, `tiger_level3_experiment`, `tiger_mixture_experiment`
- **UAV runners (4):** `run_rts`, `run_uav`, `uav_baseline_runner`, `uav_batch_runner`
- **Wumpus runners (3):** `run_wumpus`, `wumpus_baseline_runner`, `wumpus_batch_runner`
- **Experiments (10):** `run_benchmarks`, `deep_hierarchy_prior_experiment`, `level_convergence_matrix_experiment`, `master_nested_ipomdp_benchmark`, `planner_comparison_experiment`, `planner_oracle_experiment`, `root_chance_experiment`, `recursive_chance_experiment`, `tree_diagnostics`, `tiger_l3_vs_l2_experiment`

## Inventory: 23 entry points, 4 kinds

| Kind | Scripts | Profile used | Status |
|---|---|---|---|
| **A. Episode batches** (`GenericBatchRunner` subclasses) | 11 runners and 4 suites | **default** (sampled backups, `NormalizedUCB` √2 or 1.0), except `tiger_l3_vs_l2` | Results are not interpretable as optimal play (Phase 3 T1) |
| **B. Matched oracle** (`planner_oracle_experiment`) | 1 | any (CLI) | The scientifically sound core. 22 positional parameters. |
| **C. Frozen-contract replays** (`root_chance`, `recursive_chance`, `tree_diagnostics`) | 3 | the validated profile and its candidates | They pin historical rows byte-for-byte, monkeypatch `IPOMCPPlanner.policy_for` with `unittest.mock` in production code, and use `assert` for control flow |
| **D. Interactive demos** (`run_uav`, `run_wumpus`) | 2 | default | Print-driven and unsupervised, using the global RNG and commented-out code |

There are 11 nearly identical `_setup_domain` bodies that build two banks, bootstrap levels and pick an exploration strategy. That is roughly 900 lines of DRY violation. Configuration values (sims, particles, depth, creak accuracy, c) are hard-coded literals scattered across them, with no single declared experiment specification.

## Findings

### Correctness

- **B1 — Snapshots never fire in the payoff matrix** (`level_convergence_matrix_experiment.py:317`). The condition `m >= 3 and cell_count == 1` can never hold: cell 1 is L0 vs L0. The sunbursts this suite promises are never produced.
- **B2 — The planner-comparison palette never matches.** It tests `"Sampled RTS" in c` (`planner_comparison_experiment.py:403`), but the condition name is "Approximate RTS (…)". The RTS bar is mis-coloured.
- **B3 — Only agent i's planning time is recorded** (`generic_batch_runner.py:294-296`).
  - Agent j's decision time and both agents' belief-update time are never measured.
  - Measurements use wall clock (`perf_counter`), not process CPU time, so contention contaminates them.
  - The ladder showed that up to 3× wall-time inflation under load is possible.
  - Contribution 2's runtime claims need per-decision CPU time for both agents, separated into search, nested solves and filtering.
- **B4 — The returned initial state is discarded and the call wastes randomness.**
  - `_setup_domain` returns an initial state that `_run_episode` discards and replaces (`:246-248`).
  - Generating that state also consumes the seeded global RNG.
- **B5 — Common random numbers are claimed but don't hold** (`:236`).
  - The transition stream is shared, but Tiger only draws from it when a door opens. Once two conditions take different actions, the streams desynchronise.
  - Paired tests (`statistics.py`) are still valid, because they pair by trial seed. But the variance reduction the comment implies is not guaranteed.
- **B6 — Dead objects** in `tiger_mixture_experiment.py:189-190` (`bank = SolverBank(); I_POMDP_Bootstrapper(bank)` is never used). `planner_comparison_experiment.py:274-295` contains dead `is None` checks.
- **B7 — The UAV belief metric reads an always-empty reservoir** (Phase 3 B1). `uav_belief_target_pos_mode` is never recorded.
- **B8 — The snapshot episode skips the source lock.** `run_single_trial_with_snapshots` doesn't pass `source_lock` (`:523-532`), unlike `run_batch`.

### Experimental design

- **E1 — Every episode suite uses the non-convergent default profile.** That covers suites 1–3 of `run_benchmarks` and all domain runners. Their recorded results (and any figures derived from them) can't support optimality or comparison claims.
- **E2 — Modeled-opponent budgets are tiny or unexplained.** Values are 10 (most runners), 25 (default) and 50,000 (planner comparison). The L3 ladder showed modeled-policy budget dominates opponent-model error: at r1 (10 sims) the level-belief error was TV 0.47–0.88. No suite chooses this budget from an error curve.
- **E3 — Empirical physical priors everywhere except the L3 study.**
  - Runners use 1,000–5,000 sampled states, so the 50/50 Tiger prior is sampled, the 72-state UAV prior is sampled, and the 94,500-state Wumpus prior is mostly missing (Phase 5 B1).
  - `set_physical_prior` exists but only the L3 study uses it.
- **E4 — The payoff matrix's "Nash equilibria" use exact float equality of noisy means** (`level_convergence_matrix_experiment.py:233-234`).
  - Levels are treated as pure strategies of a bimatrix game with no uncertainty.
  - With n = 200 and SEMs of several reward units, the "equilibria" are noise.
  - Contribution 3's equilibrium-finding claim needs confidence-aware best responses at the very least.
- **E5 — `master_nested_ipomdp_benchmark` uses planning depth 20 by default** with 10k simulations and 10 modeled simulations. It duplicates `deep_hierarchy_prior_experiment` and is not part of `run_benchmarks`.
- **E6 — Level-matrix priors.** Every agent gets a uniform prior over strictly lower levels, so an L4 agent puts only 25% on L3. That is declared, but it makes "higher level is better" partly a prior-dilution artefact.
- **E7 — Common-knowledge seeds in the planner comparison** (`planner_comparison_experiment.py:66-74`).
  - Both banks share `policy_seed`, so i's modeled j reproduces the real j's policy exactly whenever beliefs coincide.
  - This is a perfect-opponent-model setting, not a general benchmark. It is documented in the comments but not in the outputs.

### Structure

- **S1 — `GenericBatchRunner` does five jobs:**
  - the episode loop;
  - metric extraction;
  - sunburst data extraction (`extract_nested_belief_hierarchy`, which belongs in analysis);
  - manifest/resume logic;
  - checkpointing.

  It reassigns a module-level `logger` through `global` (`:215-216`).
- **S2 — Two source-hash mechanisms in one function:** `source_sha256` (src only) and `source_lock` (also src only) (`:404-410`). A third copy lives in `planner_oracle_experiment.py:433-436` and a fourth in `tree_diagnostics.py:325-328`. None covers `benchmarks/`, so scripts there run unlocked. That is exactly why the old handoff put new code in `benchmarks/` to dodge the lock.
- **S3 — Compatibility proxy.** `examples/tiger/runners/planner_comparison.py` is an explicit "Backward-compatibility proxy". Delete it.
- **S4 — Production code imports test tooling.** `unittest.mock.patch` is used in `root_chance_experiment.py:10` and `tree_diagnostics.py:20`. Capturing the solved tree should be an explicit return value of `policy_for` (or an observer hook), not a monkeypatch.
- **S5 — Frozen-contract replays assert byte equality with historical rows** (`root_chance_experiment.py:92`, `recursive_chance_experiment.py:193-195`, `tree_diagnostics.py:264`).
  - After any restructure these can never pass, by design.
  - They are versioned-history machinery, which the target codebase removes.
  - The **error-attribution algebra** in `tree_diagnostics.describe_node`/`trace_margin` (immediate/chance/continuation decomposition with a zero residual) is valuable and should become an analysis library function, decoupled from replay.
- **S6 — `planner_oracle_experiment.evaluate_case` takes 22 parameters.** About ten of them are planner flags repeated for the opponent. A typed `PlannerSpec` object, used for both the protagonist and the modeled opponent, removes the explosion (Phase 1 S1).
- **S7 — Wumpus outcome classification is duplicated** (`wumpus_baseline_runner.py` vs `wumpus_batch_runner.py`). Same for the UAV distance metric (three copies).
- **S8 — Hidden monkeypatching in my own `tiger_l3_vs_l2_experiment.py:194-204`.** It wraps `planner_j.get_action` to time j. B3 should be fixed in the runner instead. It also imports the private `_get_opponent_level_distribution`.
- **S9 — `statistics.py` hard-codes a p-value of 0.0** when the differences are constant and nonzero (`:278-281`), and applies Holm only to the t-test. It is otherwise sound: pairing by trial, Holm step-down.
- **S10 — `run_benchmarks` "optimal" suite** calls `run_oracle_comparison` with `seeds=n_trials`, conflating trials with seed count. Its banner text ("MASTER LARGE-SCALE BENCHMARK") is marketing.

### What is good and should be kept

- `process_supervisor`:
  - per-job session;
  - RSS polling over the whole process tree;
  - atomic result files;
  - in-worker source-lock verification.
- Per-trial atomic checkpoints with a manifest check on resume, and completion markers that bind CSV bytes.
- `planner_oracle_experiment`'s design:
  - matched problems;
  - the exact reference with an explicit scope string;
  - first-action loss defined as `V* − Σ π Q*`, which handles ties;
  - fresh output directories only.
