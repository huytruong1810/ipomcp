# Current review handoff

## Ownership and implementation checkpoint

Use only /home/andyj1810/projects/ipomcp on fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs experiments. No source changes during
runs, no source copies outside Git, no silent retries or overwritten evidence.

The finite-budget L2 reference and runner are implemented. The reference is
an exhaustive best response to the DECLARED finite MCTS L1 policy, not an
exact-L1 substitute. Complete immutable private models survive every update.
See docs/L2_CONTRACT.md and src/solvers/exact/finite_policy_l2.py.

A policy-identity bug was also fixed: equal beliefs with reversed insertion
order could produce OL versus L at the same seed. Canonical root mass ordering
now makes the deterministic sampling law respect order-independent belief
equality. This can change historical finite trajectories. Old results remain
source-bound; defaults are not promoted and old gates are not reclassified.

## Contract and recorded settings

--level 2 --opponent-depth d --opponent-budget B selects finite modeled MCTS.
Without opponent-budget, the distinct exact-L1 reference remains available.
Opponent backup defaults to sampled, tail=false, exploration=normalized,c=1.
Separate flags can explicitly change them. Invalid unused settings are rejected.

Each row records the actual planner config, full modeled config/exploration,
bank seed, private prior, and tie rule. The manifest records modeled config and
bank seed range as well as all CLI settings/source hashes. For these panels both
modeled mcts.n_sims and opponent.n_sims equal B; modeled policy calls use the
latter. All config fields enter policy identity, including otherwise inactive
ones; do not call other production configs bitwise equivalent without checking.

The root policy is chosen before the exhaustive protagonist reference is run.
Cached opponent policies are the shared environment law, not leaked oracle Q.
Total worker time includes both root planning and exact reference work.

## Verification and smoke evidence

Focused contract/runner tests: 31 passed. They cover cache eviction and reordered
beliefs, independent H2 state enumeration, joint posterior moments, retained
private posterior identity, information restrictions, full metadata and explicit
resource errors. The full suite is being finalized before handoff.

Three smoke panels completed, 18 cases each:
- modeled25 and modeled100: opponent depth3, b_j=.085, own beliefs .05/.5/.95,
  H1-H3, 1000 root traversals, seeds300-301;
- exact-control: depth2 exact opponent, b_j=.085, own beliefs .1/.5/.9,
  matching the saved earlier exact-opponent smoke.

All54 had zero first-action loss; finite-opponent maximum Q error .881825.
The exact-control policies and oracle values were unchanged from the saved
18-case pre-fix comparison. The new full-model reference also matched the scalar
exact-opponent reference on nine H1-H3 cases (max Q difference below1e-15). All source hashes and concurrency bounds were checked.
These are small development controls, not depth20 or production qualification.
Raw evidence: results/l2-finite-20260925/{modeled25,modeled100,exact-control}/,
their logs, audit.json and tests.log. Raw files are local and Git-ignored.

## Completed 900-case finite-budget modeled-opponent comparison

Antigravity completed all six panels sequentially under source checkpoint bf1c225
with RUN_ID `20260925`:
- `results/oracle/l2_finite_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 900 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and
concurrency bound checks for all six panels:

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 50.62 s | 53.62 s | 93.18 s | PASS (46.59 <= 53.62) | 74.49 MB |
| B_opp=25, b_j=0.500 | 54.21 s | 55.61 s | 95.38 s | PASS (47.69 <= 55.61) | 74.49 MB |
| B_opp=25, b_j=0.915 | 51.32 s | 54.28 s | 92.86 s | PASS (46.43 <= 54.28) | 74.72 MB |
| B_opp=100, b_j=0.085 | 51.74 s | 54.80 s | 94.96 s | PASS (47.48 <= 54.80) | 74.77 MB |
| B_opp=100, b_j=0.500 | 52.09 s | 53.41 s | 93.08 s | PASS (46.54 <= 53.41) | 74.54 MB |
| B_opp=100, b_j=0.915 | 54.50 s | 57.38 s | 98.26 s | PASS (49.13 <= 57.38) | 75.03 MB |

All six panels strictly satisfied the concurrency bound check. External GNU
time agreed with parent monotonic elapsed within fractions of a second (total
wall time ~314s, ~5.2 min).

### Decision accuracy and loss summary

| Panel | Cases | Errors (loss > 1e-8) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0721 | 0.8602 |
| B_opp=25, b_j=0.500 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0978 | 1.6300 |
| B_opp=25, b_j=0.915 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0744 | 1.0749 |
| B_opp=100, b_j=0.085 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0809 | 0.8602 |
| B_opp=100, b_j=0.500 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0693 | 0.6448 |
| B_opp=100, b_j=0.915 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0881 | 1.0749 |
| **Total** | **900** | **0 (0.00%)** | **0 (0.00%)** | **0.000000** | **0.000000** | **0.0804** | **1.6300** |

Across all 900 cases, **900 decisions (100.0%)** achieved exact oracle agreement.
Zero primary gate violations and zero strict errors occurred.

### Modeled opponent computation sensitivity (B_opp=25 vs B_opp=100)

Comparing oracle evaluations across all 225 matched (b_j, H, b_i, seed) configurations:
- **9 out of 225 configuration pairs (4.00%) flip their Bayes-optimal action set** between
  B_opp=25 and B_opp=100 (0 at H1, 4 at H2, 5 at H3), with max action Q diff reaching 7.7330.
- **21 out of 225 configuration pairs (9.33%) exhibit Q-value differences > 1e-4**.
- Finite sampling variance at 25 simulations occasionally prompts door opening where 100
  simulations favor listening. The reference oracle tracks this policy law exactly, and
  protagonist MCTS selects the optimal response in 100% of cases under both budgets.

### Signed Q errors

Terminal door-opening actions (`OL`, `OR`) had zero Q error (+-0.0000) across all cases
for b_j in {0.085, 0.915} and for B_opp=100 at b_j=0.5. Mean signed Q error on action `L`
ranged from +0.0146 to +0.0284 (B_opp=25) and -0.0048 to +0.0361 (B_opp=100).

### Next steps: Codex independent review

These are developmental runs under declared finite MCTS opponent semantics at depth 3.
Ready for Codex review and verification of manifests, source hashes against bf1c225,
and recomputation of reference values. Production depth 20 qualification remains pending.

## Remaining qualification

Finite opponent depth3 and budgets25/100 are development settings. Production
depth20, mixed levels, empirical priors, long/deep Tiger before/after, L4/all39
and remaining demo/visualization review remain unqualified. Keep the previous
900 exact-opponent cases and all L1 evidence unchanged. Prior frozen L1 passes
and failed gates retain their original source/configuration scope.
