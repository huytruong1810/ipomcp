# Phase 1 — `src/core/`

Files reviewed in full: `__init__.py`, `config.py`, `distribution.py`, `logger.py`,
`paths.py`, `pomdp_model.py`, `telemetry.py` (905 lines).

Severity: **B** = correctness/theory bug, **T** = theory or spec gap, **S** = structure
(DDD/DRY/KISS/SOLID, dead code, compatibility residue), **D** = docs/docstrings.

## Findings

### Correctness and theory

- **T1 — The model contract lets hidden state decide legal actions.**
  - The module docstring says action availability may depend only on the acting agent's information (`pomdp_model.py:9-11`).
  - The signature is `get_legal_actions(state, agent_id)` (`pomdp_model.py:160`), and its docstring encourages state-dependent pruning ("massively reduces MCTS branching", `:164-168`).
  - Callers guard this by requiring every particle to give the same legal set (`i_pomcp.py:74,195`; `finite_filter.py:128`; `rts_planner.py:55,116`). That is a runtime patch on a wrong interface.
  - **Fix direction:** the action set should be a function of the agent's history or observable state (or simply constant per frame), so the leak cannot be expressed. To be confirmed against UAV and Wumpus in Phase 5.
- **T2 — Rollout hook receives the hidden state.** `get_rollout_action(state, agent_id, belief)` (`pomdp_model.py:171`) receives the hidden state while forbidding its use. The same remedy applies as for T1: pass only private information.
- **T3 — Mixed RNG discipline.**
  - The model takes optional `rng=None` arguments.
  - `distribution.py`, `pomdp_model.py:183` and most solvers use the global `random` module.
  - `policy.py:65` calls `random.seed(seed)`, which reseeds the process-global RNG from inside a policy query. Phase 3 will check whether this couples search streams.
  - **Fix direction:** one explicit `random.Random` (or numpy `Generator`) per stream, owned by the caller, and no global RNG anywhere.
- **T4 — The default planner profile is known not to converge, but it is the default.**
  - `MCTSConfig` defaults to `backup="sampled"`, `exact_final_step=False`, `exact_history_rewards=False` (`config.py:48-55`).
  - The L1 study (`results/tiger/l1_vs_exact/`) shows this profile is 61–88% optimal at horizons 3–5, with Q error 40–72 that does not shrink with budget.
  - The validated profile is 100% optimal from 1k simulations.
  - `max_depth=20` and `n_sims=1000` are also arbitrary defaults that experiment code overrides.

### Structure

- **S1 — Flag explosion in `MCTSConfig`.**
  - Six interacting switches: `backup`, `exact_final_step`, `exact_tail_depth`, `exact_history_rewards`, `exact_root_chance`, `recursive_chance`.
  - Pairwise validity is checked by hand (`config.py:62-87`).
  - Each is a strategy (leaf evaluator, backup operator, final-decision scorer). Fix direction: typed strategy objects, not booleans with cross-constraints (SOLID: open/closed, single responsibility).
- **S2 — Compatibility residue in the serialised form.**
  - `IPOMCPConfig.to_dict` drops keys when they hold their default value, "to keep historical default policy seed/metadata identities unchanged" (`config.py:120-133`).
  - `to_search_dict` (`:135-140`) is a second identity used for seeding.
  - This is the schema versioning the project wants gone. Seeds should come from an explicit seed argument, not from hashing a config dict whose shape is kept frozen for old runs.
- **S3 — Dead code.**
  - `MemoryWatchdog`, `TelemetryLogger` and `SystemMonitor.capture_snapshot` are used only by `tests/test_telemetry.py`. The regime classification (`telemetry.py:293-302`) is therefore never consulted.
  - `DictDistribution` is used only by tests.
  - `ParticleDistribution` is used only by `rts_planner.py`.
  - `Distribution` (the ABC) has no production polymorphism.
  - The IPF claim in `distribution.py:5-6` is false: the interactive filter is `ipomdp/finite_filter.py`.
- **S4 — Process-global schedules.** `DEFAULT_SIM_SCHEDULE` and `DEFAULT_PARTICLE_SCHEDULE` (`config.py:15-18`) are level-indexed budgets with no stated derivation. They are imported by three experiment scripts and `qualify_suite.py`. They belong to an experiment design, not to `core`.
- **S5 — `RTSConfig` lives in `core` but is used by a single baseline.** Domain-agnostic `core` should not know about specific solvers. Configs belong with their solver; `core` keeps the model contract and probability types.
- **S6 — `ExperimentConfig` mixes the trial design** (`n_trials`, `max_steps`) **with execution limits** (`max_workers`, timeouts, RSS) **and output flags** (`export_trees`, `verbose`). Its `n_trials=1000` and `max_steps=10` defaults are never the right value for any study.
- **S7 — `paths.get_results_dir` creates directories as a side effect** and is called by 16 runner scripts. There are 16 scripts because there are many parallel runners (Phase 6).
- **S8 — `logger.get_logger` mutates handlers on every call** (it removes file handlers that point elsewhere). Whichever caller runs last wins. A single configuration at the entry point is simpler.

### Docs

- **D1 — Marketing tone instead of decisions.** Examples: "High-performance, zero-subprocess", "[ARCHITECTURAL MANDATE]", "microsecond-resolution", "strictly typed". Docstrings should state the contract and why each choice was made.
- **D2 — The `core/__init__.py` docstring** describes "abstract base classes for environments" but also re-exports configs and the logger.

## Proposed target (for the Phase 11 plan)

`core` (better named `ipomdp_core` or split into `model/` and `probability/`) holds:
- the physical model contract, fixed per T1 and T2;
- the probability primitives that are actually used;
- explicit RNG streams.

Configs move next to the components they configure. Telemetry keeps only the snapshot the batch runner records.
