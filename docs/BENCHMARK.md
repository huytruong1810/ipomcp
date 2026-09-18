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
