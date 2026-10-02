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


Review of 03ca490: verified all 120 H6/H8 cases against e63db30 and recomputed
all 120 reference problems. Zero loss confirmed, but every optimum is listen
and minimum gap is .640013. Repeated six pilot rows reproduce exactly.
Corrected timing discrepancy endpoint and unsupported bias attribution.
Froze 1,680-case fresh L2 validation after checking physical-belief/seed
novelty against local manifests and 17,239 rows; no new validation solve.
Solver code/defaults unchanged; no redundant implementation test rerun.


Review of f0b98c5: audited the frozen 1,680-case L2 gate and independently
recomputed all reference configurations. Confirmed failure with 23 primary
violations and 26 strict errors. Reproduced two failing solves and decomposed
root Q errors into immediate, chance-frequency and child-value contributions.
Reviewed node Bellman backups, MCTS history propagation and bank reward
integration together. Planned an explicit conditional-reward ablation with
separate model-identity controls; no solver source or defaults changed.


Conditional-reward implementation: added opt-in config/runner metadata and
internal-tree reward integration using full private-history beliefs. Covered
analytic posterior rewards under both sampled hidden states, both backups,
terminal outcomes, independent tail/history flags and identical modeled laws
between arms. All 46 focused tests pass. A 12-pair new-source smoke improves
1 control violation to 0 candidate violations; full paired development remains.
Global defaults remain unchanged; new configuration changes seed identities,
so older source-bound evidence is not silently transferred.


Final implementation verification: all 241 tests passed in 392.64s, including
domain integration. Ruff lint and format checks passed across 103 Python files,
and diff whitespace checks passed. Full log:
results/history-rewards-20260925/tests.log.


Review of a7a8b75: verified all 3,360 paired development rows and source/manifests
against 4c17278, with independent reference recomputation for the seven residual
candidate errors. Confirmed gains and identified four new strict errors, two
new primary violations. Corrected strict-fix count, bias-elimination language,
invariant-RSS claim and isolated-overhead interpretation. Kept option opt-in.
Froze 3,360-case fresh candidate protocol after checking local manifests and
22,303 case rows for belief/seed novelty. No fresh solves or solver changes.


September 26 workflow: main fast-forwarded to bc35a16 and pushed to origin.
Deleted fix/tiger-policy-inversion locally and remotely after verifying both
local/remote main were ancestors. One checkout remains; continue on main,
creating branches only for separately justified side work as requested.
User reports concurrent CPU/GPU load from another codebase; isolate accuracy
reproducibility from resource and timing interpretation.


Review of bc35a16: checked all 3,360 fresh candidate rows and reproduced all 15
strict errors at the original 50k settings, with independent reference solves.
Verified Bellman error decompositions and child-error dominance in 14/15 action
ranking errors. Replayed eight H8 errors at 100k; all became optimal. Added
lower-horizon checks and prepared a full 1,440-case H5/H6/H8 budget experiment.
No solver/default changes; previous 241-test implementation result remains.
Concurrent workloads preclude isolated runtime comparisons, not preservation
of the failed gate.


Completed all 30 diagnostic jobs: 15 exact 50k replays and 15 selected 100k solves.
All 15 selected 100k cases meet tolerance; 13 are strictly optimal. Verified
unchanged reference values/model identity, source hashes and concurrency
checks. No inference of full-grid qualification follows from selected errors.


Review of the 100k report at aa8963b: audited all 1,440 cases, manifests/source
hashes, paired modeled-policy/Q identity, probability/loss arithmetic and
worker intervals. Independently recomputed all six error references. Confirmed
four new strict regressions, including all three tolerance violations; all old
H8 violations were corrected. Removed unsupported resolved-variance, isolated-
execution and invariant-memory interpretations from the authoritative summary.
Kept defaults unchanged; no solver edits or new validation launch.

Six-case 100k diagnostic artifact audit completed: source/script hashes, full
replay equality, branch probabilities/counts, Bellman reconstruction and worker
intervals checked. No new solver execution in this audit. Corrected attribution
of already-exact root rewards; continuation error dominance does not establish
its cause. Post-hoc exact root chance weights correct two violations, leaving
one, with child estimates fixed. Descendant diagnostics are the next checkpoint;
no production-default changes or additional experiment authorization.

Implemented the reusable post-search tree_diagnostics runner and completed all
six exact 100k replays with immediate-child decomposition. Independent algebra
tests cover unobserved branches, explicit rollout frontiers and uniform ties;
12 focused tests and Ruff pass. All competing-root-action children choose
optimal local actions, while grandchild value estimates dominate their error.
No production solver change or global correctness conclusion follows.


September 26 continuation implementation: inspected all six saved descendant
artifacts and ranked grandchild contributions with discounted empirical path
weights. The saved depth is insufficient to attribute those values internally;
no new six-case search/replay was performed. Added a bounded best-first trace
with signed/absolute unresolved error and reconstruction checks, plus an opt-in
two-step exact boundary using production reward, observation and filtering
kernels. The reference stays post-search and the finite opponent stays fixed.

The boundary is theoretically exact within two remaining decisions, not a proof
of the observed failures' cause or of improved H5/H6/H8 choices. Tests cover
analytic observation-conditioned Tiger values, survival mass, branch-limit
failure, uniform greedy ties, chance/local-loss/frontier attribution, and finite
opponent/reference identity. Default MCTS seed serialization retains the old
shape; nondefault tail depth is explicit. Antigravity control/candidate commands
and interpretation limits are in docs/CONTINUATION_PROTOCOL.md. No defaults,
failed gate status or frozen tolerance changed.


September 27 independent tail-two review: audited all 1,440 rows, source hashes,
coverage, full configurations, greedy policies, paired reference/model identity,
arithmetic and intervals. Recomputed all three residual-error references exactly.
Verified both six-case trace sets and rebuilt their weighted attribution. Chance
dominates five baseline cases; a rare poorly estimated H3 decision dominates the
sixth. Recorded two new small strict errors despite zero tolerance violations,
worse maximum Q error, source-working-tree provenance, and corrected RSS,
concurrency, estimated-margin and isolated-speedup claims. Full details are in
BENCHMARK.md. Raw artifacts and failed historical gates remain intact.

Frozen a fresh 3,360-case source-bound protocol after scanning local beliefs and
seeds; no validation case was solved. Added a launcher that verifies source and
runner hashes and refuses existing attempts. Global defaults remain unchanged.
Implementation checks: 254 tests passed in the full run (558.80s); after adding
four regressions, 17 focused tail/diagnostic tests pass and 258 tests are now
collected. Ruff/format and whitespace checks pass. This is engineering evidence,
not a substitute for the frozen accuracy gate.


September 28 fresh tail-two audit: verified all 3,360 rows and six freeze sidecars
against the original contract hash and complete source/configuration identities.
Checked exact coverage, greedy policies, loss/Q-error arithmetic, summaries,
worker intervals and sequential panels. Independently recomputed the sole error's
reference exactly; did not re-enumerate all other references. Gate FAILED with
one loss .062316559844 at budget25/b_j=.085/H8/seed8000/b_i=.966 (OR versus L).
H1–H6 have no observed errors. Defaults and historical failed gates are preserved.

Corrected report interpretation of b_i, old-study provenance, working-tree source,
case-versus-panel RSS ranges and unsupported clock-gap explanations. Prepared one
exact tail-two diagnostic replay for Antigravity with full-row equality required;
no source change, larger sweep or new gate. Only documentation and audit artifacts
changed, so the previous implementation test results remain applicable. Source
lock and whitespace checks pass; no redundant implementation-suite rerun.


September 28 diagnostic review: verified the new tail-two failure replay against
its full original row, input/frozen-source hashes, all 243 expanded histories,
weighted decomposition and resource records. Chance dominates the signed margin
error; local action loss/frontier terms are smaller but nonzero. Corrected the
walkthrough's categorical causal claims and distinguished incremental depth
contributions from recursively propagated value errors. Computed exact-root-
probability substitutions on seven saved tail-two trees: two error corrections,
five preserved optimal actions. This is a selected post-search counterfactual,
not a production candidate or a gate repair. Recorded the next implementation
design in ROOT_CHANCE_PLAN.md. No source, default, tolerance or frozen-contract
change; no redundant implementation test run.


## September 28: opt-in final root chance scoring

Implemented exact_root_chance=False by default; True requires empirical Bellman.
After identical seeded search, fully covered root actions use production expected
reward plus gamma times the unconditional observation/survival law weighted by
saved child estimates. Terminal probability is not renormalized. Missing positive
support causes whole-action empirical fallback with missing-mass telemetry;
enumeration failures propagate. Exact tail roots bypass scoring. The tree keeps
empirical Q; candidate evaluator rows separately record recommendation Q/policy,
search Q and coverage. The option is absent from search seeds, present in enabled
metadata, and omitted from default metadata. No modeled-opponent settings change.

For fixed child estimates this removes the direct root frequency discrepancy.
Adaptive child estimates still contain deeper chance, local action and frontier
errors; unbiasedness or universal variance reduction does not follow. The selected
post-hoc corrections motivated this ablation but are not candidate run outcomes.
The source-locked 14-pair pilot is docs/ROOT_CHANCE_PROTOCOL.md. Antigravity runs it;
Codex has not run the benchmark panel. All failed gates and .020 +1e-8 remain fixed.
No global-default promotion is justified.

Engineering validation for the root-chance implementation: 268 tests passed in
397.70s, including ten new coverage, terminal-mass, tie, inference-failure,
configuration, unchanged-tree/RNG and paired-L2 replay checks. Ruff lint/format
and git diff whitespace checks pass. All 14 saved pilot input hashes and rows,
and its 76-file Python source lock, were verified without running the panel.
The original failed tail-two contract SHA256 remains
f47c2375cfb6787c2b629291c83f44a6a4dfa33f82c41b8cbeea16c1dd050e85.
These are engineering checks, not a candidate accuracy qualification.

## September 28: independent root chance pilot audit

The locked 14-pair pilot completed. Independent source/contract/input/full-control
row, metadata, greedy-policy, loss/Q-error, coverage and timing-interval checks
pass. Violations (.020 +1e-8) fall 1 to 0; strict errors (>1e-8) fall 4 to 2.
Cases 0 and 6 correct to L and OR, with recommendation margins .008513028270675
and .011785735004079 respectively. Cases 3 and 9 retain losses
.00605795924416519 and .005155991132857807; their incorrect L-over-OR estimated
margins are .001785041113493 and .018846882683915. All eight neighboring controls
and H1/H2 boundary controls stay optimal, with no ties or regressions. All 36
non-boundary action scores have complete support, zero missing mass and no
fallback; six exact-tail scores bypass integration as intended.

Mean loss falls .005620464382397 to .000800996455502. Mean maximum Q error changes
only .692502900563 to .692290427176, and maximum Q error stays 3.849293549167.
This is a targeted ranking improvement, not uniformly accurate Q estimation.
Both corrected scores agree with previously saved-tree counterfactuals within
floating-point rounding. Residual error cannot be attributed to a missing root
branch; deeper child-value errors remain, without a new decomposition here.

Reporting corrections: actual contract SHA256 and recorded manifest hash both are
4ff704ee69aa369aeafb6ac3704ba4df0e26551be5b8195755115ffb7b11bc9b,
not the a93f4124... hash quoted in Antigravity's message. Full embedded contract
and all 76 source hashes match current files. No hash was refreshed by this audit.
External elapsed 439.81s agrees approximately with Unix elapsed 439.237792s but
not parent monotonic 468.416864s or serial worker sum 465.007998s. Monotonic
interval span 468.334885s and serial nonoverlap are internally consistent.
Preserve this unresolved clock discrepancy; do not infer its cause or repair logs.
GNU-time peak RSS is 138.816406 MiB; monitored case peaks range
77.398438–139.003906 MiB (different measurements). Mean pair-relative policy
runtime delta is +3.896339%, pooled delta +2.299937%, and mean across the 12
non-boundary pairs +2.419590%. Control-first ordering, varying load and clock
issues prevent interpreting these descriptive ratios as isolated scoring cost.

The saved runner records a common tree digest only after asserting equality of
two separately computed digests. This audit verified that locked implementation
and all recorded flags, not independently reconstructed both full trees. Oracle
vectors match preserved rows; no new reference solve or planner replay was run.
Audit script and JSON: results/oracle/l2_root_chance_review_20260928/.
Raw pilot: results/oracle/l2_root_chance_pilot_20260928/ and sibling .time.

Next scientific step: freeze a broader paired comparison across the already
observed 1,440 development and 3,360 formerly fresh cases before another solver
change or fresh qualification. Assess new regressions, fallback and costs, not
just these selected corrections. This audit does not launch or prescribe that
large run yet; its contract and resource plan remain to be prepared. Preserve
the original tail-two FAILED gate, all historical failures and frozen tolerance.
Global defaults remain unchanged. No source edits or test rerun were needed for
this artifact audit; the last implementation suite remains 268 passing tests.


September 28 broader development handoff: docs/ROOT_CHANCE_DEVELOPMENT.md now
prescribes the frozen 4,800-pair observed union in twelve sequential panels, one
worker, 600s/2048 MiB per pair. Master contract is
benchmarks/root_chance_development_20260928.json (SHA256
600caaae9425d12c225f3b96bcd79ed842ee39f201eaf616d1ca04b22e4a9927).
All saved rows/hashes and exact coverage pass preflight; 18 focused tests pass.
Source solver and paired runner remain identical to the audited pilot. Antigravity
runs this protocol; no experiments were launched by Codex. Report every regression
and fallback, cohort/stratum metrics and clock disagreements. Defaults, historical
failed gates and frozen tolerance remain unchanged; this cannot qualify defaults.

## September 29: broad root chance comparison independently audited

All 4,800 pairs and twelve source/contract/sidecar/input locks check out, including
exact full historical control-row equality, recorded tree equality invariants,
raw-search Q equality and unchanged task semantics. Greedy policies, losses,
Q errors, action coverage and complete Cartesian coverage were recomputed from
raw rows. All fourteen pilot rows and common tree digests reproduce exactly.
The saved digests attest the locked runner's comparison, not two independently
reconstructed full trees. No protagonist search was run by Codex in this audit.

The candidate does not improve the primary criterion: violations remain 1 -> 1;
strict errors worsen 4 -> 5, mean loss .0000163930211153 -> .0000225832314195,
and maximum loss .062316559843681674 -> .08501670289316543. Corrections are
P04/C117 and P06/C486; regressions P00/C103, P04/C122 and P11/C556. Unchanged
errors are P03/C86 and P04/C127. No ties or fallbacks: 11,520 action scores have
complete exact probability support and 2,880 bypass at an exact-tail boundary.
The candidate remains experimental; do not advance it to fresh qualification.

The new violation P11/C556 has reference L=10.36282005743891,
OR=10.277803354545744. Control L=10.32790908947335 versus OR=10.318029328458437
chooses L. Root weighting lowers L to10.31727558730152, leaves OR unchanged,
and selects OR by .000753741156917. The L adjustment is -.010633502171830.
The reference vector and loss were independently recomputed from a fresh bank
with the same finite modeled opponent and match exactly; no protagonist search
was used. The direct root frequency correction can remove an offsetting error
while deeper child estimation error remains. Its precise deeper decomposition
is pending, not established merely by this score change.

Corrections to the supplied report: maximum candidate loss is .08501670289316543,
not .085017255. Mean maximum Q error improves slightly from .303117595307246
 to .302387987996859, rather than both being .303387. Maximum Q error remains
6.225102823541306. P04/C122 loss is .009034698760331, so H6 errors are not all
<=.009 (all remain below .020). Improvements in mean Q error do not imply better
root decisions on near ties.

Timing: parent monotonic109932.542284s, worker sum109040.599543s,
parent Unix105472.017365s, external105477.0s. Within-panel worker intervals are
serial and fit parent monotonic elapsed. Cross-panel worker monotonic intervals
also support serial execution (all gaps positive); Unix endpoints show one
-.781879s apparent overlap between panels4/5. External/Unix and monotonic clocks
disagree substantially, especially earlier panels; do not infer the cause or
repair evidence. User/system CPU totals123847.21/1290.45s. GNU-time RSS147.496094
MiB differs from monitored case maximum147.769531 MiB. Search-only pooled policy
ratio1.017589 and mean-pair ratio1.019453 are descriptive, not causal overhead
estimates under control-first ordering and load/clock uncertainty.

Audit script/report: results/oracle/l2_root_chance_development_review_20260929/.
Reference-only script/result: reference_check.py and reference_check.json there.
The audit's reference limitation concerns its bulk row checks; the new violation
has the separate exact recomputation above. Solver/evaluator sources and all
historical contracts are unchanged; no full test rerun was needed for this audit.

Next Antigravity instructions: docs/ROOT_CHANCE_DIAGNOSIS.md, seven cases covering
all corrections, regressions and residuals, new H8 violation first. It explicitly
separates empirical trace attribution from the final root-scoring increment and
requires full replay equality and unresolved-term reporting. No tuned threshold,
blend, new solver candidate or fresh qualification is prescribed yet. Preserve
all historical failed gates and the .020 +1e-8 tolerance. Defaults stay unchanged.

## September 29: seven-case diagnostic audit

Verified unchanged source/input locks, all seven full row replays, independent
reconstruction of saved trace contributions and empirical-to-recommendation
margin identities. The recomputed identity residual is zero in floating-point
arithmetic for these rows; trace reconstruction residuals are at roundoff scale.
This is an algebraic audit, not an asymptotic sampling-rate certification.

All five nonzero traces resolve completely: 244/59/35/35/35 expanded histories,
408 total, no pending or unresolved terms. Their signed empirical margin errors
come from chance contributions; local action loss and rollout-frontier totals
are zero on the traced paths. Exact-boundary Q discrepancies are at most
1.4210854715202004e-14. The two corrected cases have zero candidate-versus-optimal
weights and no expanded paths; zero traces do not certify their deeper estimates.

For new violation P11/C556: empirical target .07513694187825237 plus scoring
increment .01063350217182979 equals recommendation margin error
.08577044405008216. Policy loss remains .08501670289316543; margin error is a
different quantity. Empirical signed chance contributions by remaining horizon:
H8 -.010538404257424, H7 +.013466187355586, H6 +.024069130820679,
H5 -.004161209683513, H4 +.005990855076252, H3 +.046310382566674.
These incremental terms may be summed; propagated child errors across depths
must not be summed again. Root chance uses exact child values in its diagnostic
term, whereas the actual score substitution weights estimated child values.
Consequently the scoring increment is not exactly minus that root chance term:
a .000095097914406 interaction remains under the changed weights.

The empirical OR continuation error is +.040225973912692 while L's is
-.045449372222988. Their mismatch survives root integration. These fixed traces
support deeper chance-estimation error as the mechanism in these five cases, not
a universal campaign-wide causal claim. No O(1/sqrt(N)) or O(1/sqrt(N_child)) rate
has been established for adaptive search values, maxima, correlated histories
and finite frontiers. Likewise OL/OR have one private-observation branch here,
not deterministic hidden transitions: opening resets Tiger Left/Right with
probability .5 each under this physics.

Diagnostic external123.57s, parent monotonic122.940329s, worker sum121.320996s;
serial intervals fit parent elapsed. Monitored RSS83.703125–131.703125 MiB,
GNU-time peak131.042969 MiB. Historical timing discrepancies are not erased.
Audit: results/oracle/l2_root_chance_diagnosis_review_20260929/audit.py and audit.json.
No new search/reference solve or solver source change in this audit.

Next implementation design: docs/RECURSIVE_CHANCE_PLAN.md. It proposes recursive
post-search integration on the same observed tree, with full-history production
kernels, explicit missing-support/frontier/action-coverage reporting and unchanged
search trajectories. It has only a conditional full-coverage Bellman guarantee;
it is not implemented or qualified yet. No new Antigravity experiment is ready
until engineering tests and a paired protocol are supplied. Preserve defaults,
all historical failed gates and the frozen loss tolerance.


September 29 implementation follow-up: recursive_chance is now an opt-in final
post-search evaluator; defaults and modeled opponents remain unchanged. It
integrates production chance laws recursively over supported observed histories,
propagates frontier/unevaluated-action/subtree uncertainty, and uses whole-action
empirical fallback for missing positive support. Search Q and all tree statistics
are preserved, and both final scoring flags are excluded from search seeds.

The source-frozen 23-case three-arm pilot is docs/RECURSIVE_CHANCE_PROTOCOL.md,
contract benchmarks/recursive_chance_pilot_20260929.json, SHA256
 a843f1aaca898e9652eb7302248c8aad473e23ece97864e57121de8f468b26a4.
It includes all seven diagnosed cases plus fourteen previously correct neighbors
and H1/H2 controls, with exact historical empirical and root-only row replay.
Codex has not run benchmark cases. Antigravity runs the declared pilot only;
source locks for prior completed experiments remain preserved and are not updated.
The root-only broad result still has one violation/five errors, and the original
fresh tail-two gate and all historical failures remain unchanged.

Engineering validation, September 29: full suite 292 passed in 391.76s, including
16 new recursive integration/coverage/fallback/configuration/tree-invariance
and three-arm L2 checks. Ruff lint/format and git diff whitespace checks pass.
The 23-case pilot input rows, source lock and four preserved historical contract
hashes were verified without running the pilot. Both CLI help and no-experiment
preflight were checked. Benchmark accuracy and resource costs remain pending
Antigravity's declared run; passing unit/integration tests is not qualification.

## September 29: recursive chance pilot review

All 23 three-arm cases completed. Source lock (78 Python files), contract/input
hashes, full empirical and root-only historical row replays, candidate task/config
semantics, separate three-arm tree digests and raw-search Q identity check out.
Greedy policies, losses and Q errors were recomputed from preserved rows. All
three arms have no greedy ties. This review also reconstructed every node/action
coverage counter, legal-action gap and subtree-flag dependency from the telemetry.
No new protagonist search or reference solve was run.

Recursive recommendations are optimal on 23/23: zero strict errors and zero
.020 +1e-8 violations, with no control regressions. Empirical baseline has four
errors/one violation; root-only has five errors/one violation. All seven diagnosed
cases and sixteen previously correct/boundary controls are optimal. This is
selected development evidence; it does not replace the root-only broad failure
or the original fresh tail-two FAILED gate.

Value accuracy is not uniformly improved: mean maximum Q error is
.569969262380566 (empirical), .569839930753683 (root-only), .662375580144034
(recursive). Maximum Q error increases from3.849293549166731 to4.322188276076147.
For P11/C556 the recursive L-over-OR margin is .094363867026054; L's value error
is +.011733109753155. The decision is corrected, but its subtree is approximate.

Coverage totals: 36,221 evaluated histories, 26,976 integrated action scores,
75,888 exact-tail scores, 2,830 whole-action empirical fallbacks, 515 rollout
frontiers, 1,433 histories with unevaluated legal actions, and 4,965 approximate
histories. Fallbacks occur at remaining horizons H5:37, H4:318, H3:2475; frontiers
at H4:41 and H3:474. These are local counts, not root probabilities. At the roots
there are 63 complete-support integrated action scores and six exact-tail scores,
with zero root-local fallbacks. Descendant approximation still propagates.

Only the H1/H2 cases (indices21/22) have fully exact root subtrees, and their
Q vectors agree with the finite reference to roundoff. Sixteen selected root
ACTIONS are marked exact and agree with their reference action Q to roundoff,
but only two complete roots are certified: exactness of the chosen action alone
does not bound approximate alternative actions. The other seven chosen actions
remain approximate, even where observed Q error happens to be zero. This supports
the conditional Bellman argument, not an unconditional exact-planning theorem.

Timing: external1237.60s, parent monotonic1236.994208s, parent Unix1236.994208s,
worker sum1228.132554s; serial intervals fit parent elapsed. Monitored RSS range
77.289063–139.234375 MiB; GNU-time peak147.117188 MiB is a different measure.
Policy-time sums empirical332.566311s, root-only335.841107s, recursive333.550568s;
recursive/empirical pooled ratio1.002960 and mean-pair ratio1.018294 are descriptive,
not isolated overhead estimates under fixed arm order and system load. Earlier
clock discrepancies remain preserved; these clocks are consistent for this run.

Review artifacts: results/oracle/l2_recursive_chance_codex_review_20260929/.
The audit extends the preserved supplied audit with full metadata and recursive
coverage checks; Antigravity's original audit remains unchanged at
results/oracle/l2_recursive_chance_review_20260929/. Both record that saved digests
are compared, not independently reconstructed full trees, and reference vectors
are historical rather than newly solved in this review.

Next: prepare the broader observed comparison over all 4,800 cases with the
recursive candidate frozen, including explicit coverage/regression/resource
reporting. Do not tune this candidate on the successful selected panel, start a
fresh qualification, or promote defaults. A new broad protocol and resource plan
must be prepared before Antigravity runs again. No solver edits or test reruns
were needed for this artifact review; the last full suite remains 292 passed.
All historical failed gates and the frozen tolerance remain unchanged.

## September 30: inferred recursive broad contract and panel0 reviewed

Antigravity completed the draft protocol by creating the exhaustive master and
running panel0 before Codex returned. Codex reviewed and accepts the preserved
master without regeneration: benchmarks/recursive_chance_development_20260929.json,
SHA256 7a1bb7bdf8e53a1d6c74664e66d94a572fa6132cced770c53e4ca7d9a1059c63.
All twelve contracts and 4,800 distinct inputs match the previously declared
observed union in order, with complete saved empirical/root-only rows. Source
hashes match the 78-file recursive pilot lock, launcher hash matches, and all five
historical contracts are unchanged. This is adoption of observed development
provenance after review, not a retrospectively declared held-out study.

Panel0 completed240/240 triples: recursive choices all optimal, zero regressions,
root-only case103 corrected. Empirical baseline0 errors; root-only1 error; no
violations in any arm. Full historical replays, candidate metadata, separate
three-arm digests, arithmetic and coverage pass audit. The three overlapping
pilot cases reproduce all rows/coverage and digests exactly. Reference vectors
were compared to historical rows, not newly recomputed in this review.

Mean max Q error: empirical.481425277942250, root-only.481380370176491,
recursive.394729824760483. Maximum worsens4.403166522319879 ->4.979874317778965.
The correct decisions do not certify all values. Recursive telemetry contains
434,061 histories;316,381 integrated actions;900,072 exact-tail actions;
42,239 fallback actions;7,530 frontiers;20,872 histories with unevaluated actions;
70,141 approximate histories. Seventy roots have fully exact subtrees and agree
with the reference to roundoff; remaining roots are not certified. All counters,
legal-action gaps and subtree dependencies were reconstructed from saved records.

External11639.0s / parent monotonic11584.556110s / Unix11638.194195s /
worker sum11492.706677s. Serial worker intervals fit monotonic parent elapsed;
Unix minus monotonic is53.638084s, unresolved. Monitored RSS80.078125–144.179688
MiB; GNU-time167.132813 MiB. Policy-time sums empirical2972.205404s,
root-only3050.953125s, recursive3056.657329s; ratios are descriptive under fixed
arm order and clock/load uncertainty. Disk usage813 MiB for panel0; WSL available
space926 GiB at review, not reserved capacity.

Codex review scripts/results: results/oracle/l2_recursive_chance_development_p00_codex_review_20260930/.
Antigravity's original audit and raw artifacts are preserved. No solver/source
change or benchmark launch by Codex. Launcher and recursive-focused tests24 passed
in .92s, lint and whitespace checks pass; previous full implementation suite292
passed, and the eight launcher tests are additional focused checks.

Next Antigravity instructions: docs/RECURSIVE_CHANCE_DEVELOPMENT.md now continues
from panel1 through11 only (4,560 triples). Do not rerun panel0, refresh contracts,
overwrite telemetry, or tune after seeing outcomes. Complete fixed coverage unless
an operational/provenance failure requires stopping. Preserve all regression and
coverage evidence; no default promotion or fresh qualification follows. All
historical failed gates and .020 +1e-8 remain unchanged.
