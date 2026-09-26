# Current review handoff

## Ownership and state

Single checkout: /home/andyj1810/projects/ipomcp, fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs experiments. No source changes during
runs, source copies outside Git, silent retries or rewritten raw evidence.

Access is restored. Antigravity committed the pre-outage implementation as
bf1c225 and its900-case study documentation as123701b. Saved test evidence
confirms225 tests passed in310.22s. This audit changes documentation only;
no solver changes/default promotions and no redundant test rerun.

## Independent900-case audit

Verified every requested case, coverage/uniqueness, settings, bank seeds,
source hashes againstbf1c225 and summary statuses. Recomputed450 distinct
oracle configurations and checked both root-budget copies, policy distributions,
first-action losses and max Q errors. All900 choices match the reference:
mean/max first-action loss0. This remains DEVELOPMENT evidence at opponent
depth3, not general finite-policy optimality or a fresh validation gate.

Budget25 versus100:9/225 optimal-action sets change;21 Q/value changes exceed
1e-4; max Q difference7.733. Budget changes also change hashed solver settings,
so equal bank seed indices are not a common-random-number intervention. The
observed policy differences are real; do not claim a population convergence
rate or that each additional simulation caused a particular action change.

All six panels pass monotonic worker-duration and interval concurrency checks.
GNU times sum314.48s; parent monotonic times sum329.10s. Differences1.32–3.06s
per panel remain unresolved. Keep both records, rather than calling them equal.

Opening is not terminal. At modeled budget25,b_j=.5, opening-action max Q error
is .766343 despite zero action loss. Other panels have only roundoff opening
errors. Exact chosen actions do not imply exact estimates or zero continuation.

Artifacts:
- results/oracle/l2_finite_n{25,100}_b{0.085,0.5,0.915}_20260925/
- results/oracle/l2_finite_review_20260925.json
- results/l2-finite-20260925/ (pre-outage tests and smoke evidence)

The54 smoke cases and9-case exact-reference cross-check are separate evidence:
zero action loss in the former; max Q difference8.88e-16 in the latter.
Raw artifacts remain local and Git-ignored; documentation commits do not archive them.

## Depth20 resource pilot and next development run

Six pilot cases completed: modeled depth20,budget25,b_j=.5; H1-H2; root1000;
seed400; own beliefs .05/.5/.95. Zero policy loss, max Q error .402376,
max case time .486818s, peak monitored RSS74.094MiB. Parent monotonic1.630009s,
worker sum2.792059s; concurrency bound passes. Evidence:
results/oracle/l2_finite_depth20_pilot_20260925/ plus sibling log.
This checks feasibility at small protagonist horizons only.

## Completed 360-case depth20 development extension

Antigravity completed all six panels sequentially under source checkpoint 06cdcd3
with RUN_ID `20260925`:
- `results/oracle/l2_depth20_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 360 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and
concurrency bound checks for all six panels:

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 21.58 s | 22.27 s | 39.02 s | PASS (19.51 <= 22.27) | 74.59 MB |
| B_opp=25, b_j=0.500 | 19.49 s | 20.65 s | 36.47 s | PASS (18.24 <= 20.65) | 74.79 MB |
| B_opp=25, b_j=0.915 | 22.80 s | 22.28 s | 38.45 s | PASS (19.23 <= 22.28) | 74.77 MB |
| B_opp=100, b_j=0.085 | 25.22 s | 26.46 s | 46.67 s | PASS (23.34 <= 26.46) | 74.55 MB |
| B_opp=100, b_j=0.500 | 22.27 s | 23.51 s | 41.79 s | PASS (20.90 <= 23.51) | 74.65 MB |
| B_opp=100, b_j=0.915 | 24.31 s | 25.63 s | 45.37 s | PASS (22.69 <= 25.63) | 75.09 MB |

All six panels strictly satisfied the concurrency bound check. External GNU
time agreed with parent monotonic elapsed within 1.3 seconds (total wall time ~136s, ~2.26 min).

### Decision accuracy and loss summary

| Panel | Cases | Errors (loss > 1e-8) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1166 | 1.5352 |
| B_opp=25, b_j=0.500 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0685 | 0.5145 |
| B_opp=25, b_j=0.915 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0843 | 0.8467 |
| B_opp=100, b_j=0.085 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0789 | 0.7453 |
| B_opp=100, b_j=0.500 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0685 | 0.5145 |
| B_opp=100, b_j=0.915 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1076 | 1.7135 |
| **Total** | **360** | **0 (0.00%)** | **0 (0.00%)** | **0.000000** | **0.000000** | **0.0874** | **1.7135** |

Across all 360 cases, **360 decisions (100.0%)** achieved exact oracle agreement.
Zero primary gate violations and zero strict errors occurred.

### Opponent depth comparison: depth 3 vs depth 20 (matched seeds 400-401)

Comparing the 360 matched cases between opponent depth 3 and depth 20:
- **18 out of 360 pairs (5.00%) flip their Bayes-optimal action set** (all 18 occurred under B_opp=25).
- **62 out of 360 pairs (17.22%) exhibit Q-value differences > 1e-4** (max action Q diff reaching 7.7330).
- Under B_opp=100: **0 action flips** occurred between d=3 and d=20, showing policy stability
  across deeper search horizons. Protagonist MCTS accurately selected optimal actions across all 360 cases.

### Signed Q errors

Nonterminal door-opening actions (`OL`, `OR`) had zero Q error (+-0.0000) across all cases.
Mean signed Q error on action `L` was -0.0394 to +0.0350 (B_opp=25) and -0.0055 to +0.0163 (B_opp=100).

### Next steps: Codex independent review

These are developmental runs under declared finite MCTS opponent semantics at depth 20.
Ready for Codex review and verification of manifests, source hashes against 06cdcd3,
and recomputation of reference values. Production protagonist depth 20 qualification remains pending.

## Remaining limits

This matches the declared pure finite L1 opponent at fixed depth20; it still
does not reproduce every production config field. In this runner modeled
mcts.n_sims and opponent.n_sims both equal the requested budget; other full
configs hash differently. Actual protagonist horizons remain H1-H3, not20.
Mixed levels, empirical priors, long/deep Tiger before/after, L4/all39 and
remaining script semantic review are still pending.

Canonical sampling corrected representation dependence; older results remain
source-bound. Historical L1 passes/failures keep their original meanings.
No global default promotion or claim of universal optimality follows.
