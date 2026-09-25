# Current review handoff

## Ownership and completed implementation

Use only /home/andyj1810/projects/ipomcp on fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs experiments. No source changes during
runs or source copies; Git stores source history. Raw results remain local
and ignored unless explicitly archived elsewhere.

The fixed-depth L2 contract is implemented and documented in docs/L2_CONTRACT.md.
ExactIPOMDPSolver accepts an explicit opponent_horizon. The matched runner's
--level 2 --opponent-depth d uses an exact L1 opponent replanning at depth d
after every private update. Shared-countdown reference behavior is a separate
model. Global MCTS defaults and production opponent semantics are unchanged.

This is intentionally an exact-L1 opponent, not the production 25-simulation
modeled MCTS opponent. Both compared L2 solvers use this declared law. The root
MCTS receives no oracle protagonist Q values. Do not label results as production
L2 qualification or proceed to L4/all39 from this panel.

## Timing instrumentation

Worker outcomes now retain monotonic start/finish timestamps. Each oracle panel
writes timing.json even on a parent exception, with parent monotonic elapsed,
real-time start/finish, worker sums, coverage and the concurrency-bound check.
A worker sum larger than workers times parent elapsed raises an explicit error.
Intervals can also be checked for overlap and containment.

The 18-case L2 smoke has parent monotonic elapsed 5.329160s, worker sum 9.111192s,
two workers, and passes the concurrency bound. External GNU time recorded 5.29s,
while realtime endpoints differ by 4.796633s. The clock measurements differ;
the instrumentation exposes that discrepancy without rewriting measurements.
No universal cause is established and older timing records remain unresolved.

## Evidence and engineering checks

The L2 smoke used H1-H3, fixed opponent depth2, b_j=.085, own beliefs .1/.5/.9,
seeds300-301, 1000 traversals, exact tail, empirical Bellman, bounded c=1.
All 18 cases completed with zero first-action loss. Maximum Q error at H3 was
.722473, so exact action agreement does not imply exact value estimates.
This is smoke/development evidence, not a validation pass.

Source checkpoint: 1fad300. Evidence: results/l2-contract-20260925/smoke/,
smoke.log and smoke.time.
The focused reference/runner/supervisor tests passed (35 tests).
Full-suite verification: 217 tests passed in 303.62s, including all domain
integration and supervisor fault tests. Ruff lint and formatting passed.
Raw output: results/l2-contract-20260925/tests.log.

## Completed 900-case L2 fixed-depth development comparison

Antigravity completed all six panels sequentially under source checkpoint 01f36cf
(preceded by implementation at 1fad300) with RUN_ID `20260925`:
- `results/oracle/l2_fixed_d1_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d1_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d1_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 900 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and
concurrency bound checks for all six panels:

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| d=1, b_j=0.085 | 54.07 s | 54.57 s | 94.40 s | PASS (47.20 <= 54.57) | 74.68 MB |
| d=1, b_j=0.500 | 54.47 s | 54.87 s | 94.45 s | PASS (47.23 <= 54.87) | 74.79 MB |
| d=1, b_j=0.915 | 54.36 s | 54.79 s | 94.00 s | PASS (47.00 <= 54.79) | 74.72 MB |
| d=2, b_j=0.085 | 52.14 s | 52.07 s | 91.38 s | PASS (45.69 <= 52.07) | 74.52 MB |
| d=2, b_j=0.500 | 53.12 s | 53.66 s | 92.03 s | PASS (46.02 <= 53.66) | 74.51 MB |
| d=2, b_j=0.915 | 53.43 s | 53.88 s | 93.71 s | PASS (46.86 <= 53.88) | 75.01 MB |

All six panels strictly satisfied the concurrency bound check. External GNU
time agreed with parent monotonic elapsed within fractions of a second (total
wall time ~324s, ~5.4 min).

### Decision accuracy and loss summary

| Panel | Cases | Errors (loss > 1e-8) | Gate Fail (>0.02) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| d=1, b_j=0.085 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0227 | 0.3979 |
| d=1, b_j=0.500 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0707 | 0.8183 |
| d=1, b_j=0.915 | 150 | 1 (0.7%) | 1 | 0.002251 | 0.337700 | 0.0273 | 0.4615 |
| d=2, b_j=0.085 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1034 | 1.6567 |
| d=2, b_j=0.500 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0707 | 0.8183 |
| d=2, b_j=0.915 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0867 | 1.0469 |
| **Total** | **900** | **1 (0.11%)** | **1 (0.11%)** | **0.000375** | **0.337700** | **0.0636** | **1.6567** |

Across all 900 cases, **899 decisions (99.89%)** achieved exact oracle agreement.
Under Opponent Depth d=2: **450 / 450 decisions (100.0%)** were strictly optimal.
The single error occurred at d=1, b_j=0.915, H=3, B=1000, b_i=0.05, seed 301:
chosen `L` (est Q 2.7713) vs oracle Best `OL` (Q* 2.6475, oracle gap 0.3377 over `L`
at 2.3098), policy loss 0.33770. At budget 10,000 (seed 301), est Q(L)=2.3421 < 2.6475,
selecting `OL` with zero loss.

### Opponent replanning semantics: depth 1 vs depth 2 comparison

Comparing oracle evaluations across all 30 horizon/belief combinations:
- **8 out of 30 combinations flip their Bayes-optimal action** between d=1 and d=2:
  * At b_j=0.085: H=2 (b_i=0.05, 0.95) and H=3 (b_i=0.05, 0.95) flip from opening
    doors under d=1 (`OL` or `OR`) to listening (`L`) under d=2 (value gap up to 7.7330).
  * At b_j=0.915: Symmetrical flips occur at H=2 (b_i=0.05, 0.95) and H=3 (b_i=0.05, 0.95).
  * At b_j=0.500: 0 flips (uninformative opponent listens under both depths).
- **18 out of 30 combinations have oracle Q-value differences > 1e-4**.
This confirms that opponent replanning horizon alters the underlying game dynamics.
The fixed-depth contract correctly aligns the reference oracle with the tree planner.

### Signed Q errors

Terminal door-opening actions (`OL`, `OR`) had **identically zero Q error** (+-0.0000)
across all 900 cases. All estimation error was on `L` (recursive opponent rollout):
mean signed Q error on `L` ranged from +0.0001 to +0.0107 (d=1) and -0.0179 to +0.0107 (d=2).

### Next steps: Codex independent review

These are developmental runs under declared fixed-depth exact-opponent semantics.
Ready for Codex review and verification of manifests, source hashes against 01f36cf,
and recomputation of reference values. Production modeled-opponent matching remains pending.


## Remaining gates

Prior L1 2000-case frozen accuracy gate remains passed at its exact tested
settings. Old 50k/200k failures stay historical failures. No permanent
all-horizon resolution or global default promotion is claimed.

Still pending: matched finite-budget modeled opponents, deeper level mixtures,
long/deep Tiger before/after, L4/all39 resource qualification, finite-prior error
and remaining demo/visualization semantic review. The new exact-opponent L2
contract isolates one layer of these requirements; it does not discharge all.
