# Phase 9 — `tests/`

Scope: 35 files, 329 tests (4,610 lines).
- Read in full: `reference_tiger.py`, `test_finite_filter.py` (setup and oracle), `test_integration.py`, `test_models.py`, `test_tiger_reference.py`, `test_uniform_matrix.py`, `test_audit_budgets.py`, `test_solvers.py`, `test_search_and_domain_contracts.py`, and the resampling tests in `test_l3_l2_experiment.py`.
- All other files: inventoried by test name, and scanned for monkeypatching, pinned digests and historical fixtures.

## What the suite does well

- **Independent oracles.**
  - `reference_tiger.py` enumerates the Tiger law with scalar arithmetic, independent of the filter.
  - `test_finite_filter.oracle_l2` checks every reachable L2 observation against a hand enumeration, over creak ∈ {0, .4, 1}, persistent/reset, and three private beliefs.
  - This is the right pattern, and it validates the core I-POMDP update (Phase 2).
- **Semantic contracts with names that state the property:**
  - "terminal probability is not renormalized away";
  - "nested private update does not condition on outer hidden action";
  - "search keeps all legal actions below root";
  - "uniform prior does not disclose actual opponent level".
- **Process tests:** timeouts, RSS limits, descendant reaping, source-lock violations, resume-manifest mismatch.

## Gaps (what is not tested and should be)

- **G1 — No convergence test of the planner against an exact reference beyond depth 1.**
  - `test_one_step_mcts_values_equal_exact_expected_rewards` covers H = 1 only.
  - The L1 study (`benchmarks/l1_pomcp_vs_exact.py`) shows the default profile fails at H = 3–5, yet the suite is green.
  - A small, fast qualification test is needed, for example H = 3, 1k simulations, five beliefs, the validated profile, loss ≤ ε. It would have caught the default-profile problem.
- **G2 — No sampler-vs-kernel agreement test for UAV and Wumpus.**
  - `sample_observation`/`sample_transition` and `get_observation_prob`/`transition_distribution` are separate code (Phase 5 S3).
  - Only Tiger at creak .9 is checked (`test_l3_l2_experiment.test_creak_point_nine_sensor_is_normalized_and_sampler_consistent`).
  - A generic frequency test per domain (χ² or a TV bound) belongs in a domain-contract suite run against every model.
- **G3 — No test that legal actions are constant over every reachable belief** in UAV and Wumpus (Phase 3 B2, Phase 5 T8).
- **G4 — No test of prior support adequacy.** For example, "the true initial state is in the prior support" or "zero-evidence rate under random play ≤ x". Phase 5 B1 went unnoticed.
- **G5 — The bounded-resampling tests cover only the easy regime.**
  - `test_systematic_resampling_bounds_each_mass_error_by_one_over_k` asserts `2/k` (its name says `1/k`) with 2 strata and k ≥ 8, where the bound happens to hold.
  - Order invariance is not tested (Phase 2 B1–B2).
- **G6 — The integration tests are smoke tests.** They check row counts and t = 0 zeros only. The Wumpus one takes **361 s** and asserts nothing about behaviour. It also runs the default profile.
- **G7 — Plotting has no column-name contract test.** That would have caught Phase 7 B1 (`n_particles_*` vs `belief_support_*`).

## Coupling and brittleness

- **C1 — White-box tests patch private methods.** At least 12 tests `monkeypatch` private methods or collaborators:
  - `planner._history_successor`, `planner._final_action_values`, `planner._rollout`;
  - `gen_model.tree_step`, `gen_model.sample_event`;
  - `bank.filter.update`, `bank.filter.action_distribution`, `random.random`.

  They test routing inside `IPOMCPPlanner` rather than contracts. They will all break under the strategy refactor (Phase 3 S1). They should be rewritten against the new seams (leaf evaluator, backup operator, belief propagator), each testable in isolation.
- **C2 — Tests for frozen-gate launchers** (`test_root_chance_development.py`, `test_recursive_chance_development.py`) and the replay harness (`test_recursive_chance.py::test_full_search_digest_rng_and_default_metadata_unchanged`) pin machinery slated for archival (Phase 8 S1). They go with it.
- **C3 — Tests import the compatibility proxy.** `test_solvers.py` imports `examples.tiger.runners.planner_comparison` (the proxy, Phase 6 S3).
- **C4 — The cross-test import** `from tests.test_finite_filter import l1` in `test_solvers.py` belongs in a shared fixture module.
- **C5 — Global RNG seeding** (`random.seed(...)`) in tests reflects Phase 1 T3.
- **C6 — Telemetry tests** (`test_telemetry.py`) and `test_distribution.py`'s `DictDistribution` tests keep dead production code alive (Phase 1 S3).

## Target structure

- `tests/contracts/`: one parametrized suite run against **every** domain:
  - kernels normalize;
  - the sampler agrees with the kernel;
  - legal actions depend on history only;
  - the initial prior is exact or has declared support.
- `tests/references/`: independent scalar oracles (`reference_tiger`, L2 enumeration).
- `tests/unit/`: one file per component seam.
- `tests/qualification/`: small, fast convergence checks against exact references (G1). Marked so they always run in CI, at under 60 s total.
- `tests/integration/`: one short episode per domain with behavioural asserts. Wumpus integration is made fast by fixing its prior (Phase 5 B1).
