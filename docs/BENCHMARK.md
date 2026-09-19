# Matched Tiger benchmark and validation

## Matched Tiger experiment

Each run contains seeds 0–29, 20 environment decisions, depth-five planning, L3 vs
L2 with the requested 80% L2 prior. Root budgets are 20,000 and 15,000 simulations
per decision. Nominal particle requests are 2,000/1,500, capped at 1,000 retained
particles per node. Growl accuracy is 85%, creak accuracy 100%, discount 0.95.
Reported returns are undiscounted observed totals. All 30 trials have all 21 rows,
including the initial condition. Source manifests and raw trial JSON are retained.

| Source | Agent I mean return | Agent J mean return | Mean trial seconds | Mean I planning seconds/decision | Peak recorded worker RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Initial WSL snapshot | -30.63 | -19.63 | 177.51 | 5.90 | 2276.41 |
| Updated Antigravity | -10.10 | 0.90 | 249.70 | 8.10 | 2866.77 |
| Reviewed | 15.93 | 5.30 | 223.61 | 6.96 | 2461.83 |

The paired reviewed-minus-updated-Antigravity difference for Agent I is **+26.03**,
95% t interval **[−6.83, +58.90]**. Agent J's difference is **+4.40**, interval
**[−18.47, +27.27]**. These intervals cross zero; no statistical improvement or
equivalence is established. No equivalence margin was selected before the run.
The same seeds couple environment streams, but action-dependent draw consumption
means this is not perfect per-step counter-based CRN alignment.


Machine-readable results (exported audit artifact: `benchmark-comparison.json`) contain trial-level summary
statistics. The source directories are `wsl-tiger-baseline`,
`tiger-antigravity-20260917`, and `tiger-reviewed-20260917` beside this report.
The earlier Windows-clone experiment is excluded because it did not represent the
authoritative WSL local code.

## Efficiency: measured tradeoffs

The full reviewed runs recorded lower mean planning latency and peak sampled RSS.
Those wall-clock runs overlapped other audit workloads, so their difference is not
an isolated speedup estimate. Fresh-process, alternating-order CPU probes used
three additional seeds and three steps with the same full planning budgets:

| Probe | Updated Antigravity | Reviewed |
|---|---:|---:|
| Mean process CPU seconds | 42.22 | 49.53 |
| Mean process peak RSS (MiB) | 758.27 | 924.43 |

The short probes show **17.3% more CPU time** and
**21.9% more peak memory**. This is a small,
short-horizon sample, and policy/search trajectories differ. It prevents claiming
a general efficiency improvement from the full-run wall times alone.

The one-step profiles show nested generative simulation and rollout as the main
costs. The reviewed profile made roughly 1.19 million `tree_step` calls versus
1.10 million in the baseline; particle mapping protection also has measurable
allocation cost. Reservoir insertion calls fell from about 1.17 million to 0.73
million after removing duplicate/root routing. Optimize the specified belief/policy
semantics before assuming additional caches or model merging are safe.
Raw profiles and CPU probes are under `runtime-audit/`; aggregate values are in
runtime-summary.json (exported audit artifact: `runtime-summary.json`).

## Verification and Antigravity evidence

- **76 passed, 2 strict expected failures** in the full test suite. The xfails are
  the diagnostic-creak and nested-history theory counterexamples, not passing
  correctness tests. See pytest output (exported audit artifact: `pytest.txt`).
- Ruff checks and formatting checks pass; Git whitespace checks pass.
- The wheel builds, and all **61 packaged Python modules** import from the
  extracted wheel. See package verification (exported audit artifact: `package-verification.json`).
- Exact fixed-policy Tiger tests verify observation normalization, accumulated
  listening evidence, reset/type separation, and one-step Bellman values.
- Antigravity's saved N=50, T=10 CSV reproduces mean returns 0.78/1.00, positive
  trial counts 41/44, and treasure counts 139/149 and 113/120. This verifies that
  artifact's arithmetic, not an independently pinned source revision for that run.
  See note verification (exported audit artifact: `antigravity-notes-verification.json`).

## Integrated finite-filter qualification — September 18

The historical measurements above describe their source snapshots, not current
solver behavior. All new thirty-trial runs use twenty steps, depth five, L3/L2
real budgets 20k/15k and the 80% L2 prior. The new model uses shared empirical
physical priors, exact level weights, recursively evolving immutable beliefs and
ten-simulation modeled MCTS policies. Old node-capacity clipping no longer applies.

| Implementation | Mean I | Mean J | Mean wall seconds | Max worker MiB |
|---|---:|---:|---:|---:|
| Integrated filter | -33.93 | -2.03 | 21.38 | 194.95 |
| Exact root reward control variate | -21.10 | 8.23 | 24.61 | 190.96 |

Both panels contain all thirty seeds and twenty-one rows per seed. The second
run's mean CPU time is 24.36 seconds/trial. Paired second-minus-earlier-review
differences: I=-37.03 (95% t interval [-76.30,2.24]), J=2.93 ([-21.31,27.17]).
Versus Antigravity: I=-11.00 ([-54.13,32.13]), J=7.33 ([-10.30,24.97]).
These are descriptive exploratory comparisons on reused seeds, not confirmatory
equivalence tests. No equivalence margin was prespecified; timings overlapped
other audit work. The model changed, so this is not a pure optimization comparison.

All 39 depth-one, three-step smoke conditions pass. The full-budget seed-zero
L4/L3 depth-five twenty-step probe completes in 33.24 seconds, 182.25 MiB peak.
The RTS L2/MCTS L1 depth-three probe with 500 RTS particles and 50k real opponent
simulations fails after 7.09 seconds with unsupported right-creak evidence. Its
point prior models RTS rather than the executing MCTS kernel. Failure remains
recorded; it is not dropped from a successful-trial average. One-step RTS success
did not predict multi-step validity. Full-suite launch remains blocked.

## September 19 controlled-comparison qualification

The modeled planner family is now independent of protagonist search. The controlled
comparison declares the same MCTS L1 budget (50,000), exploration rule, initial
empirical prior (2,500 samples) and search seed as the executing opponent. Each
agent maintains isolated private beliefs thereafter. RTS lookahead remains 500
particles. MCTS protagonist uses 20,000 simulations in this qualification panel.
This changes the scientific configuration; it is not a pure speed optimization.

Ten seeds, twenty steps, depth three, two workers, 300-second/3,072-MiB per-trial
limits: RTS completes 10/10 and MCTS completes 10/10. RTS means are I=3.10,
J=-53.00, 34.61 wall seconds/trial,
34.58 CPU seconds/trial, max RSS 168.50 MiB.
MCTS means are I=3.10, J=-53.00,
38.86 wall seconds/trial, 38.83 CPU seconds/trial,
max RSS 169.79 MiB. Timing overlaps other qualification work.
These ten-seed panels demonstrate successful execution, not performance equivalence
or general optimality. All twenty-one rows per trial and source hashes were verified.

The earlier family/budget-only panel retained independent initial priors and seeds:
RTS completed 5/10 and MCTS
completed 8/10. Failed trials
are retained and excluded from no purported full-panel mean. Matching only a
planner name and simulation count was insufficient to specify the policy kernel.

All 122 tests pass across unit and integration invocations (no xfails), including
Wumpus; all 39 reduced-budget smoke conditions pass. The existing L3-vs-L2 prior
benchmark is unchanged in experimental design; no new equivalence claim is made.
Ordinary full-suite process supervision, remaining full-depth condition coverage,
demo consolidation and authoritative-checkout reconciliation are still open.

## Supervised full-depth qualification — September 19

Every batch trial now uses an isolated Linux process session, including runs with
one worker. Defaults are two workers, 900 seconds per trial and 3,072 MiB sampled
process-tree RSS. An explicit run_batch(max_workers=...) overrides the default
worker count. RSS sampling can miss short spikes; it is not an allocation ceiling.
The supervisor kills timed-out/over-budget sessions and reaps workers. Snapshot
episodes preceding suite batches use the same limits. Each attempt retains logs,
status, wall/RSS measurements and Python traceback where available. Successful
trial checkpoints survive failures; retries keep historical failure records.

The full-depth qualifier uses the actual condition tables with no reduced search
budgets. One seed per condition, twenty steps, depth five for prior/matrix and
depth three for comparison produced:

| Suite | Complete | Failed | Max wall seconds | Max sampled RSS MiB |
|---|---:|---:|---:|---:|
| Prior | 7 | 0 | 50.48 | 153.96 |
| Comparison | 7 | 0 | 76.34 | 134.58 |
| Matrix | 12 | 13 | 900.04 | 696.13 |

All 127 unit/integration tests pass, including fault injection for time, RSS,
process crashes, descendant cleanup and checkpoint recovery. The failed matrix
trials remain failures, not zero returns or omitted observations. Its so-called
exact prior is a point mass on a capped lower level, not a complete exact model
of the actual opponent policy. Strict inference cannot define a posterior on
zero-probability evidence. Choosing explicit uniform lower-level priors including
L0 is a separate experimental design and awaits user selection.

The prior and comparison families have a full-budget twenty-step execution check;
one seed per condition is not a powered validation of expected payoff. The full
suite remains unqualified while matrix conditions fail. Existing payoff/value
limitations and the unestablished L3-vs-L2 reward equivalence remain in force.
Runtime results are source-bound in results/full-depth-qualification-20260919.
