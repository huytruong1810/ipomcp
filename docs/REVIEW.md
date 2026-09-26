# Current phased review

This ledger describes the current code rather than superseded implementations.
Git history retains earlier review notes. Inspection, a passing unit test, and a
scientific validation are distinct levels of evidence. Open items are explicit.

| Phase | Module | Current assessment |
|---|---|---|
| 1 | Layout, packaging, paths and entry points | Antigravity refactor inspected; source imports and wheel build verified. Four misleading oracle/report entry points removed and replaced by one matched experiment. |
| 2 | Probability, configurations, telemetry | Existing finite-mass checks retained. Modeled/real budgets remain explicit; no claim that 25 modeled simulations is sufficient. |
| 3 | Immutable beliefs, frames, bootstrap, bank | Joint private beliefs and pure cached policies retained. Conflicting repeated bootstrap configuration is still an open contract gap. |
| 4 | Recursive filtering and generative transitions | Shared strict kernel retained. Frozen opponent beliefs removed from nonterminal rollout transitions. No posterior repair on unsupported evidence. |
| 5 | MCTS, exploration and RTS | Unproven action pruning removed. Finite-budget accuracy remains a gate; correct state propagation does not imply a good approximation at available budgets. |
| 6 | Exact references | L2 hidden-state optimization replaced by observation-conditioned joint-belief recursion. L1 query validation, H0 and pruning numerical boundary repaired. |
| 7 | Domains | Tiger rollout memory handles reset/persistent physics and noisy creaks under declared L0. Domain integration checks are distinct from broad domain research validation. |
| 8 | Experiment execution and analysis | Oracle model mismatch and stale hard-coded artifact inputs removed. Supervision retains failures. Workers now use spawn; per-attempt resource provenance when limits change remains open. |
| 9 | Visualizations and demos | Existing interfaces import; semantic consolidation and native rendering remain incomplete. |
| 10 | Integrated scientific qualification | Thirty-seed before/after Tiger and multi-budget oracle panels provide evidence; full-suite optimality and reward equivalence are not established. |

## Demonstrated defects and their corrections

The old L2 reference used `max_a Q(s,b_j,a)` before averaging hidden states.
This allows different future actions for indistinguishable histories. With a
uniform initial tiger belief and H=2 it returned +8.5, whereas one 85%-accurate
growl leaves opening worth -6.5 and listening worth -1 on the final step. The
correct initial value is -1-.95=-1.95. The replacement marginalizes both hidden
actions and opponent observations, keeps correlations in the joint posterior,
and chooses continuation actions only after conditioning on the protagonist's
observation. No six-decimal belief merging or nearest-grid substitution remains.

This follows the belief-state control formulation, rather than a fully observable
state relaxation: [Cassandra, Kaelbling and Littman, AAAI 1994](https://cdn.aaai.org/AAAI/1994/AAAI94-157.pdf).
Reference assumptions include conditionally independent private Tiger sensors,
uniform L0, fixed physics, a common decreasing horizon, and numerical tie tolerance.
It is a bounded floating-point reference, not a symbolic exact arithmetic proof.

The former Tiger tree action mask asserted strict dominance from a fixed 85%
physical confidence threshold. No interactive, horizon-dependent dominance proof
supported that assertion. All legal actions now remain in the search tree.
The 92% leaf policy is explicitly a heuristic and cannot constrain tree actions.

The rollout speedup preserved the same opponent private model over future steps.
Intentional actions consequently stopped responding to simulated private history.
Rollouts now use the full generative transition except at the final reward step.
That restores the stated model and can increase runtime; low cost under different
dynamics is not an optimization of the same algorithm.

The removed optimal-report scripts used a fixed L1 oracle to label deeper-agent
returns, a hard-coded historical matrix, and a bespoke unbounded episode loop.
The replacement measures only a precisely matched L1 problem and records failure
and cost. It reports first-action loss under optimal continuation and Q errors;
it does not equate this loss with twenty-step policy regret.

## Verification and coverage inventory

165 tests passed: 161 unit and 4 domain integration. Lint, formatting, all 57
non-package source imports and clean wheel construction pass. The wheel contains
the new oracle module and no removed experiment modules. The audit's prior and
comparison CLIs and both spawn-based oracle workers pass targeted smoke runs.

Every current Python file is enumerated below for follow-up accountability.
Imports are a structural check, not a fresh line-by-line semantic certification
of every visualization and demo. Open semantic work remains in BACKLOG.md.

- `src/core/__init__.py`
- `src/core/config.py`
- `src/core/distribution.py`
- `src/core/logger.py`
- `src/core/paths.py`
- `src/core/pomdp_model.py`
- `src/core/telemetry.py`
- `src/examples/__init__.py`
- `src/examples/experiments/deep_hierarchy_prior_experiment.py`
- `src/examples/experiments/level_convergence_matrix_experiment.py`
- `src/examples/experiments/master_nested_ipomdp_benchmark.py`
- `src/examples/experiments/planner_comparison_experiment.py`
- `src/examples/experiments/planner_oracle_experiment.py`
- `src/examples/experiments/run_benchmarks.py`
- `src/examples/tiger/__init__.py`
- `src/examples/tiger/model/__init__.py`
- `src/examples/tiger/model/tiger_model.py`
- `src/examples/tiger/runners/__init__.py`
- `src/examples/tiger/runners/persistent_tiger_runner.py`
- `src/examples/tiger/runners/planner_comparison.py`
- `src/examples/tiger/runners/tiger_baseline_runner.py`
- `src/examples/tiger/runners/tiger_batch_runner.py`
- `src/examples/tiger/runners/tiger_level3_experiment.py`
- `src/examples/tiger/runners/tiger_mixture_experiment.py`
- `src/examples/uav/__init__.py`
- `src/examples/uav/model/__init__.py`
- `src/examples/uav/model/uav_model.py`
- `src/examples/uav/model/uav_viz.py`
- `src/examples/uav/runners/__init__.py`
- `src/examples/uav/runners/run_rts.py`
- `src/examples/uav/runners/run_uav.py`
- `src/examples/uav/runners/uav_baseline_runner.py`
- `src/examples/uav/runners/uav_batch_runner.py`
- `src/examples/wumpus/__init__.py`
- `src/examples/wumpus/model/__init__.py`
- `src/examples/wumpus/model/constants.py`
- `src/examples/wumpus/model/wumpus_model.py`
- `src/examples/wumpus/model/wumpus_state.py`
- `src/examples/wumpus/model/wumpus_viz.py`
- `src/examples/wumpus/runners/__init__.py`
- `src/examples/wumpus/runners/run_wumpus.py`
- `src/examples/wumpus/runners/wumpus_baseline_runner.py`
- `src/examples/wumpus/runners/wumpus_batch_runner.py`
- `src/ipomdp/__init__.py`
- `src/ipomdp/finite_belief.py`
- `src/ipomdp/finite_filter.py`
- `src/ipomdp/frame.py`
- `src/solvers/__init__.py`
- `src/solvers/exact/__init__.py`
- `src/solvers/exact/alpha_vector.py`
- `src/solvers/exact/ipomdp_exact_vi.py`
- `src/solvers/exact/pomdp_exact_vi.py`
- `src/solvers/exploration.py`
- `src/solvers/generative_model.py`
- `src/solvers/i_pomcp.py`
- `src/solvers/node.py`
- `src/solvers/planner.py`
- `src/solvers/policy.py`
- `src/solvers/random_planner.py`
- `src/solvers/rts_planner.py`
- `src/solvers/solver_bank.py`
- `src/solvers/solver_types.py`
- `src/utils/__init__.py`
- `src/utils/bootstrapper.py`
- `src/utils/generic_batch_runner.py`
- `src/utils/paper_plots.py`
- `src/utils/plot_metadata.py`
- `src/utils/plotting.py`
- `src/utils/process_supervisor.py`
- `src/utils/regenerate_all_experiment_plots.py`
- `src/utils/statistics.py`
- `src/utils/visualizer.py`
- `benchmarks/qualify_suite.py`
- `benchmarks/tiger_pre_post_benchmark.py`
- `tests/__init__.py`
- `tests/reference_tiger.py`
- `tests/test_audit_budgets.py`
- `tests/test_distribution.py`
- `tests/test_exact_vi_l1.py`
- `tests/test_exact_vi_l2.py`
- `tests/test_exploration.py`
- `tests/test_finite_filter.py`
- `tests/test_integration.py`
- `tests/test_models.py`
- `tests/test_node.py`
- `tests/test_online_filter.py`
- `tests/test_oracle_contracts.py`
- `tests/test_process_supervisor.py`
- `tests/test_review_regressions.py`
- `tests/test_rts.py`
- `tests/test_search_and_domain_contracts.py`
- `tests/test_solvers.py`
- `tests/test_telemetry.py`
- `tests/test_tiger_reference.py`
- `tests/test_uniform_matrix.py`
- `tests/test_visualization.py`


September 24 follow-up: reviewed Antigravity's 630 raw oracle cases with matching
source hashes; added an optional, documented remaining-horizon reward-bound UCB
and explicit exploration ablations in the matched runner. Mean backups and
production defaults remain unchanged. Seventy calibration cases all completed,
but every strategy tested retains action errors. See docs/BENCHMARK.md and
HANDOFF.md for evidence and the next runner protocol.


Calibration extension review (9815f47): independently checked coverage and source
hashes for all 3,150 cases, strategy binding and recomputed first-action losses.
Selected bounded c=1 for the fixed held-out protocol, without changing defaults.
Corrected unsupported asymptotic convergence and elapsed-time descriptions.
Added and tested explicit seed-range selection for fresh validation runs; 180
non-integration tests and lint/format pass. The held-out results are still pending.


Held-out review and exact-tail phase: all 4,320 validation cases verified; candidate
failed its unchanged gate. Reviewed the one-step belief-MDP boundary and added an
explicit finite-model tail option to core configuration, I-POMCP and the matched
runner. `tests/test_exact_final_step.py` covers hidden-state independence, private
history conditioning, impossible evidence, analytic H1/H2 values and strict option
validation. A 120-case development comparison fixes the tested H2 errors but not
H3; no new default or higher-level correctness claim follows.


Budget-curve review of 32a01ce: verified all 180 cases and source provenance.
Exact-tail H3 first passes the sampled grid at 200k, but every grid-optimal action
is listen; Q errors remain nonzero. Corrected convergence-rate/zero-value-error
claims and distinguished local ignored raw evidence from committed documentation.
No algorithm change or default promotion. HANDOFF.md now contains only the current
fresh-belief/seed validation protocol and active operational constraints.


Intermediate-backup phase: independently verified the 1,440-case failed gate;
corrected H1/H2 action classification. Reviewed and implemented empirical
chance-weighted history Bellman backups in node/search/configuration/runner.
`tests/test_bellman_backups.py` covers nonuniform chance weights, terminal mass,
current-child reevaluation, explicit frontier values and analytic Tiger boundaries.
Development evidence: 112 H2/H3 comparison cases and 20 H4/H5 smoke cases; strong
H3 improvement but remaining near-tie errors at H4/H5. No default promotion.

Engineering verification for source checkpoint 2230786: 200 non-integration tests
passed in 49.13 s, plus all four integration tests in 265.32 s; Final Ruff lint/format
passed across 106 Python files; the 31 focused tests also passed after that polish. Subsequent polish changes an option-validation
error message only. The user selected bounded first-action loss for future
validation, with the numerical threshold still pending; old failed gates stand.

Review of 933d917: independently audited all 1,800 development cases and reran
the exact reference at all 60 horizon/belief pairs. Corrected the incomplete
near-tie summary: three H5/50k decisions lose 0.530942 each. Clarified mean-max-Q
table headings and permanent historical gate status. Froze a 480-case larger-
budget development comparison; numerical validation tolerance remains pending.
No solver source changed, so previous engineering tests were not rerun.

Review of 33c3356: all 480 development cases audited, exact reference rerun at
24 horizon/belief pairs. Accuracy improvement confirmed; impossible panel
elapsed-time claims withdrawn, finite-budget convergence claims corrected.
Prepared 2000-case fresh L1 protocol at 1M traversals; verified no belief/seed
overlap across 31 manifests and 12,513 local case rows. No fresh solve performed.
Numerical loss bound still pending; no solver changes or default promotion.

September 25 review of 1b3527a: independently checked all 2000 cases and recomputed
the exact reference for 100 horizon/belief pairs. Confirmed zero frozen-gate
violations and zero strict first-action errors on the panel. Rejected universal
policy/convergence claims and global default promotion from this evidence.
Identified continuing disagreement between external elapsed and worker-duration
records. Next phase is consistent timing instrumentation and matched L2 policy
semantics; solver code and previous engineering test status are unchanged.


Fixed-depth L2 implementation phase: reviewed exact L2 recursion, policy bank,
private model representation, bootstrap and oracle runner together. Added an
explicit stationary exact-L1 opponent policy and optional fixed horizon in the
reference; retained shared countdown as a distinct mathematical model.
Reviewed supervisor/runner timing boundaries; worker intervals and parent
monotonic elapsed now permit consistency checks without rewriting raw evidence.
New test coverage includes analytic L2 values, horizon-semantic differences,
private-belief policy binding, invalid contracts, interval concurrency and
preserved timing on failure. See L2_CONTRACT.md and HANDOFF.md.

Final verification: all 217 tests passed in 303.62s, including domain integration;
Ruff lint/format and diff whitespace checks passed. The 18-case L2 smoke also
passed source-hash, coverage, contract metadata and worker-interval concurrency
checks. No production default was changed. Source checkpoint: 1fad300.


Independent review of 661fb59: verified all 900 fixed-depth L2 cases and
recomputed 90 reference configurations; confirmed the single .3377-loss case
and zero errors at 10k. Checked worker intervals, concurrency and aggregate
timing. Corrected denominator45, nonterminal opening semantics, initial d2
listening behavior and descriptive (not validation) .020 loss counts.
No solver changes. Next phase is exact L2 response to explicitly matched
finite-budget modeled policies, preserving private-model identity.


Finite-computation L2 phase: reviewed SolverBank policy identity, MCTS sampling,
recursive filter, generative transition and oracle runner together. Implemented
an exhaustive finite-policy response retaining private-model identity, with
independent H2 enumeration and joint posterior moment checks. Fixed reproduced
insertion-order dependence of modeled MCTS under equal FiniteBelief values.
The runner records complete real/modeled settings and seed identity. Preserved
exact-opponent controls; no silent policy-model substitution or default promotion.

Additional cross-check: with an exact depth2 opponent, the new full-model
reference matches the existing scalar reference on nine H1-H3 configurations
to within 8.9e-16 in every action value. This checks the outer recursion beyond
the independent H2 hand/enumeration tests.


Post-outage audit of bf1c225/123701b: confirmed saved225-test result; verified
all 900 finite-policy cases and recomputed 450 oracle configurations. Zero policy
loss and9/225 action-set changes confirmed. Monotonic concurrency checks pass,
but external timers differ by 1.32–3.06s per panel. Corrected repeated terminal-
opening language and separated54-case smoke from9-case reference cross-check.
A six-case depth 20 pilot completed with zero loss; fixed360-case development
extension prepared. No solver source change or default promotion.


Review of ee3d5b4: all 360 depth 20 cases audited and 180 distinct reference
problems recomputed. Confirmed zero first-action loss and monotonic interval
concurrency. Corrected independence/stability interpretations of duplicated
root-budget rows. A two-case H4/H5 resource pilot completed; next360-case
development protocol targets protagonist H4/H5. No solver changes.


Review of c68754b: independently verified 360 H4/H5 rows against 1359053 and
recomputed 120 reference configurations. Confirmed 6 material errors at 1k/10k,
zero first-action loss across 120 cases at 50k, and remaining maximum Q error
4.200977 at 50k. Corrected convergence, variance-causation and bias claims;
refreshed stale README coverage and BACKLOG audit status. Worker concurrency
checks pass; external versus monotonic discrepancy remains unresolved.

Ran six H6/H8 resource pilot cases atroot 50k against depth 20/budget 25.
All completed with zero first-action loss; maximum Q errors 7.042874/6.630328.
Prepared a fixed 120-case H6/H8 development panel for Antigravity. No solver
source or defaults changed; implementation tests were not redundantly rerun.
