# Phase 8 — `benchmarks/` (scripts, shell runners, frozen contracts)

Files reviewed in full: 13 Python scripts and 3 shell scripts (2,168 lines). Also inspected the six frozen contract JSONs and the two panel directories: 24 files, 35 MB, tracked in git.

## Inventory by provenance

| Group | Files | Owner/era | Runnable after any `src` edit? |
|---|---|---|---|
| **Frozen-gate launchers** | `run_tail2_validation.py`, `run_root_chance_development.py`, `run_recursive_chance_development.py` + 6 contract JSONs + 24 panel JSONs | Codex/Antigravity, Sept 27–29 | **No.** Each refuses to start unless every `src/**/*.py` hash equals the frozen contract. |
| **Suite qualification** | `qualify_suite.py`, `tiger_pre_post_benchmark.py` | earlier review rounds | Yes, but they drive the default-profile suites (Phase 6 E1) |
| **L1/L3 study** | `l1_pomcp_vs_exact.py`, `l1_summary.py`, `exact_ipomdp_agent.py`, `validate_l3_l2_exact.py`, `l3_l2_convergence.py`, `l3_l2_ladder_analysis.py`, `l3_l2_figures.py`, 3 shell scripts | Claude, Sept 30 | Yes |
| **Read-only audit** | `audit_recursive_estimator_mixing.py` | Claude, Sept 30 | Yes (reads panel outputs) |

## Findings

### Correctness

- **B1 — Exact-audit time is counted as planning time in the L3 ladder.**
  - `l3_l2_convergence.ConvergenceRunner._instrument` replaces `planner.policy_for` with a wrapper that solves the exact reference *inside* the call (`l3_l2_convergence.py:191-213`).
  - The generic loop times `planner_i.get_action()` (Phase 6 B3), which calls the wrapper. For j, the L3 driver's own timing wrapper surrounds it.
  - Recorded `planning_time_i` and `planning_time_j` therefore include up to 120 s of exact solve per audited decision.
  - The "r4 averages about 2.3 h per trial" in the old handoff includes audit time. It must not be quoted as I-POMCP cost.
- **B2 — Pooled ladder intervals assume independent rows.**
  - `l3_l2_ladder_analysis.on_history` pools all audited (trial, step) rows into one t-interval (`:54-61`). Steps within a trial are correlated, so the intervals are too narrow.
  - Audits also cover early steps only (decision audit ≈ steps 1–3 for i), so pooled means are biased toward easy, early beliefs. Coverage is reported, but the pooled number has no interpretation beyond "early steps".
  - **Fix:** resample by trial (cluster bootstrap) and report per step only.
- **B3 — `tiger_pre_post_benchmark.py` reimplements the process supervisor** with `ThreadPoolExecutor` + `subprocess`, measuring the RSS of the top process only (`:256-275`). It also mutates the imported schedule dicts in place (`experiment.SIM_SCHEDULE[3] = …`, `:178-181`). Those are the same objects as `core.config.DEFAULT_SIM_SCHEDULE`. The mutation stays contained in a subprocess, but it is exactly the kind of global-state mutation the codebase forbids elsewhere.
- **B4 — `qualify_suite.py` imports the backward-compatibility proxy** `examples.tiger.runners.planner_comparison` (`:23-28`).
- **B5 — Inconsistent tie tolerances across references:**
  - `exact_ipomdp_agent` uses relative `1e-9` (`:25,50`);
  - `FixedTigerL1Policy` and `ExactIPOMDPSolver` use absolute `1e-10`;
  - `greedy_policy` uses exact equality;
  - the validation scripts use `1e-9` absolute.

  A shared tie rule should be part of the decision-problem definition.

### Process and structure

- **S1 — The frozen gates make the code unchangeable.**
  - `run_*_development.py` and `run_tail2_validation.py` hash all of `src` and refuse to run otherwise. Their contracts embed historical rows that must replay byte-for-byte (Phase 6 S5).
  - This is valid provenance for Codex's protocol, but it is exactly the versioned-history machinery the target codebase removes.
  - **Proposal:** after Codex's sweep is audited and its decision recorded:
    1. tag the commit (for example `archive/recursive-chance-gates`);
    2. move the conclusions into one results document;
    3. delete the launchers, contracts and panel JSONs (35 MB) from the tree.
- **S2 — `benchmarks/` is a second, unlocked `src/`.**
  - Scripts here import across each other (`l3_l2_ladder_analysis` imports `l3_l2_figures` and `l3_l2_convergence`) and from `src`, through bare-module imports that only work with `cwd=benchmarks`.
  - Source locks don't cover them (Phase 6 S2).
  - In the target layout, reusable logic (exact agent, nested-summary metrics, error decomposition, sunbursts) belongs in the package. `benchmarks/` keeps only thin, declarative study entry points, or is merged into one `experiments/` package.
- **S3 — Monkeypatching instance methods to instrument planners** (`l3_l2_convergence.py:189-243`, `tiger_l3_vs_l2_experiment.py:194-204`). That is the same pattern as the `unittest.mock` replays. An explicit observer or audit hook in the planner interface replaces all of them.
- **S4 — Reaching into private cache wrappers.** `exact_ipomdp_agent.build_exact_agent` rewraps `reference._q.__wrapped__` and `reference.branches.__wrapped__` (`:80-81`) to resize caches. Cache size should be a constructor argument.
- **S5 — The shell scripts are one-off operations** with hard-coded `results/tiger/ladder_20260930` paths and process kills by `pgrep -f`. Delete them once the ladder is closed.
- **S6 — Duplicate statistics helpers.** `l3_l2_figures.t_band` and `bootstrap_band`, plus `statistics.compute_statistical_significance`, form a third statistics implementation.

### What is good

- `l1_pomcp_vs_exact.py` is the cleanest convergence study in the repository:
  - a matched problem;
  - an exact alpha-vector reference;
  - a grid over budget × horizon × belief × seed;
  - loss, optimality, Q error and timing per row.

  It should become the template for the "small-scale first" qualification ladder. The same shape should be repeated at L2 against `FinitePolicyReference`, and then per domain.
- `validate_l3_l2_exact.py` evaluates real-agent beliefs (replayed histories) rather than hand-picked priors. That is the right complement to grid studies.
