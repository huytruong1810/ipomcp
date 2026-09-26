# September 23 correctness and resource review

The full experiment suite is not scientifically qualified. Corrected inference/search semantics expose substantial finite-budget planning error.

## Thirty-seed Tiger comparison

L3 versus L2, 80% L2 prior; 20 decisions; depth 5; real budgets 20,000/15,000; initial samples 2,000/1,500. All 30 trials completed on each side. The changes restore opponent updates and remove unsupported action pruning; this is a corrected-model comparison, not a pure speed optimization.

| Implementation | Mean I return | Mean J return | Mean trial wall s | Mean trial CPU s | Max RSS MiB |
|---|---:|---:|---:|---:|---:|
| before-planner | 21.43 | 18.13 | 38.14 | 42.36 | 192.75 |
| after-planner | -10.47 | -8.27 | 79.36 | 88.16 | 178.61 |

Paired corrected-minus-before differences (exploratory reused seeds):

- reward_i: -31.90; 95% t interval [-58.06, -5.74].
- reward_j: -26.40; 95% t interval [-55.38, 2.58].

No equivalence margin was prespecified. An interval containing zero does not establish equivalence. Action-dependent random draw consumption weakens per-step pairing. Concurrent verification jobs make wall times unsuitable for an isolated speed ratio.

## Matched oracle budget sweep

L1 against uniform L0; exact starting P(TL) in {0.1, 0.5, 0.9}; seeds 0–2; gamma .95; horizons 1–3. The two planners and oracle share the same model. All 189 cases completed. RTS enumerates all nine observation tokens. MCTS simulations and RTS particles per branch are distinct work units.

| Planner | Horizon | Budget | Cases | Mean first-action loss | Maximum loss | Mean max Q error | Mean wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcts | 1 | 1000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.028 |
| mcts | 1 | 10000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.068 |
| mcts | 1 | 50000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.255 |
| mcts | 2 | 1000 | 9 | 0.243833 | 2.194500 | 24.0279 | 0.031 |
| mcts | 2 | 10000 | 9 | 0.975333 | 2.194500 | 20.4747 | 0.270 |
| mcts | 2 | 50000 | 9 | 0.000000 | 0.000000 | 19.9763 | 1.428 |
| mcts | 3 | 1000 | 9 | 2.227785 | 3.341678 | 39.6917 | 0.059 |
| mcts | 3 | 10000 | 9 | 2.227785 | 3.341678 | 41.2894 | 0.556 |
| mcts | 3 | 50000 | 9 | 2.227785 | 3.341678 | 38.9635 | 2.694 |
| mcts | 3 | 100000 | 9 | 2.227785 | 3.341678 | 40.1394 | 6.209 |
| mcts | 3 | 500000 | 9 | 2.227785 | 3.341678 | 37.3118 | 30.538 |
| mcts | 3 | 1000000 | 9 | 2.227785 | 3.341678 | 36.4386 | 53.776 |
| rts | 1 | 100 | 9 | 0.000000 | 0.000000 | 3.5444 | 0.008 |
| rts | 1 | 1000 | 9 | 0.000000 | 0.000000 | 0.5867 | 0.016 |
| rts | 1 | 5000 | 9 | 0.000000 | 0.000000 | 0.5133 | 0.048 |
| rts | 2 | 100 | 9 | 0.243833 | 2.194500 | 3.0556 | 0.017 |
| rts | 2 | 1000 | 9 | 0.000000 | 0.000000 | 0.6382 | 0.130 |
| rts | 2 | 5000 | 9 | 0.000000 | 0.000000 | 0.5299 | 0.507 |
| rts | 3 | 100 | 9 | 0.000000 | 0.000000 | 3.3430 | 0.082 |
| rts | 3 | 1000 | 9 | 0.000000 | 0.000000 | 1.3933 | 0.745 |
| rts | 3 | 5000 | 9 | 0.000000 | 0.000000 | 0.4962 | 4.391 |

First-action loss is V*(b) − Σa π(a|b)Q*(b,a), with optimal continuation. It is not full policy regret. Zero observed loss on this small grid cannot certify global optimality. Persistent positive loss at high budgets blocks any claim that current default budgets reliably match the oracle.

## Corrected defects and remaining limits

- L2 hidden-state Bellman maximization replaced by observation-conditioned joint-belief recursion; uniform two-step value corrected from +8.5 to −1.95.
- Nearest-belief substitution and rounded L2 belief grid removed; H0 and invalid probability queries checked.
- Unsupported confidence-based tree pruning removed; all legal tree actions remain.
- Opponent private beliefs advance in every nonterminal rollout transition; final reward-only steps omit unnecessary propagation.
- Four misleading oracle/report scripts consolidated into one source-bound, supervised, matched comparison.
- L2 production/reference opponent horizon and computational models are not yet matched. L3/L4 exact optimality is not established.
- Full current 39-condition qualification remains pending, as do remaining demo/visualization semantic reviews.

Raw case/trial JSON and manifests are retained under this directory. Earlier interrupted `baseline` output is invalid qualification evidence. Historical favorable runs used different planner semantics and cannot validate the correction.

## Provenance and final verification

Planning checkpoint: 4e9ff4c; the after-run source hashes match every Python source
file at that Git commit. Before-run checkpoint: 35303a5, likewise hash-verified.
Engineering checkpoint 9e40997 additionally fixes clean worker startup and audit
budget/import wiring without changing the default planning model tested above.
The old audit manifest's modeled budget field says 10, but captured source uses
25; separate METADATA_ERRATUM.md files correct this without rewriting evidence.
The current driver records and executes the requested budget consistently.

Final verification: 165 tests passed (161 unit, 4 integration); lint/format,
source imports, clean wheel contents, prior/comparison audit CLI smoke and MCTS/RTS
spawn-worker smoke checks pass. Historical source lives in Git. There is only one
working checkout. The scientific accuracy and full-suite gates remain open.

## Exploration diagnosis, September 24

Antigravity's `results/oracle/diagnostic_boundary_{mcts,rts}_20260924`
contains 315 completed cases per planner, all source hashes matching the
pre-change checkout. At horizon three and budget 50,000, MCTS has positive
first-action loss in 12/35 cases; RTS has none at 5,000 particles per branch.
These are different work units, not a compute-matched superiority claim.
No separate Antigravity prose report was found in the checkout during this pass.

The new `--exploration normalized|standard|bounded` and `--exploration-const`
options apply only to the matched oracle runner's MCTS planner. `normalized`
retains the existing global empirical-return scale; `standard` uses raw reward
units; `bounded` uses the physical reward interval times the discounted remaining
horizon. The Tiger interval is exhaustively derived from both states, every joint
action, and transition support. It is not fitted to the oracle or sampled states.
Configuration is recorded in the manifest and actual strategy parameters in each
MCTS row. Production defaults and mean-return backups are unchanged.

A 70-case calibration panel is in `results/exploration-20260924`: horizon three,
10,000 simulations, seeds 0 and 1, beliefs .02/.1/.15/.5/.85/.9/.98, gamma .95,
two workers, 120 seconds and 1,024 MiB per case. All completed. The default replay
matches all original result fields exactly in its 14 overlapping Antigravity cases.

| Exploration | Coefficient | Wrong / 14 | Mean first-action loss |
| --- | ---: | ---: | ---: |
| Empirical range | 1 | 8 | 3.371483 |
| Empirical range | 0.1 | 2 | 0.676106 |
| Raw reward units | 10 | 4 | 1.518977 |
| Horizon reward bounds | 0.1 | 4 | 1.320253 |
| Horizon reward bounds | 1 | 4 | 0.954765 |

No setting passed every case. In particular, empirical coefficient .1 introduced
errors at beliefs .02/.98. These are reused calibration beliefs/seeds, not held-out
validation. Do not promote a new default or launch the full study from this table.

Why retain mean backups: Algorithm 1 of Silver and Veness (2010),
https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf,
uses incremental mean simulated returns. Poor finite-budget results do not alone
identify that estimator as an implementation defect. A Bellman backup using
maximal child action estimates is a different estimator and requires explicit
observation probabilities, history-conditioned values, and treatment of noisy
maximization. Maximizing over sampled hidden states or chance outcomes would
solve a different information problem. A principled Bellman variant remains a
possible future design, not an accepted one-line correction.

## Ten-seed exploration calibration extension, September 24

Antigravity completed all five 630-case panels (3,150 total cases, 0 failures, 0 timeouts, 0 resource kills) under `results/oracle/calibration_{normalized_c1,normalized_c01,standard_c10,bounded_c01,bounded_c1}`. All runs used Git checkpoint `d0f5428`, Horizons 1–3, budgets 1k/10k/50k, 10 seeds, and 7 belief simplex points under two supervised workers with 4 GiB limits.

Seeds 5–9 were previously unused and are reported separately below. They now
contribute to candidate selection and are not an untouched final validation set
(seeds 0–4 were previously inspected):

| Exploration Strategy | Coeff $c$ | H3 Wrong (0–4) / 105 | H3 Loss (0–4) | H3 Wrong (5–9) / 105 | H3 Loss (5–9) | H3 50k Wrong (5–9) / 35 | H3 50k Loss (5–9) | H2 Wrong (5–9) / 105 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `normalized` | 1.0 | 42 (40.0%) | 1.9702 | 47 (44.8%) | 2.3242 | 15 (42.9%) | 2.1631 | 17 / 105 |
| `normalized` | 0.1 | 20 (19.0%) | 0.8617 | 16 (15.2%) | 0.6772 | 9 (25.7%) | 1.2042 | 28 / 105 |
| `standard` | 10.0 | 19 (18.1%) | 0.8744 | 19 (18.1%) | 0.8389 | 4 (11.4%) | 0.5409 | 17 / 105 |
| `bounded` | 0.1 | 21 (20.0%) | 0.7436 | 19 (18.1%) | 0.7684 | 10 (28.6%) | 1.2202 | 23 / 105 |
| `bounded` | 1.0 | 25 (23.8%) | 0.9906 | 24 (22.9%) | 0.9100 | **1** (2.9%) | **0.0955** | **7** / 105 |

Key findings:
- `bounded 1.0` shows decreasing aggregate horizon-three action loss over the three tested budgets: at 50,000 simulations and Horizon 3, only 1 of 35 held-out cases failed (at $P=0.90$, seed 7), reducing held-out first-action loss from 2.1631 (`normalized 1.0`) down to 0.0955.
- At Horizon 2 and 50,000 simulations, `bounded 1.0` achieved 0/70 errors across all 10 seeds (0.0000 loss).
- At Horizon 1, `bounded 1.0` achieved 0/210 errors across all budgets and seeds.
- Detailed per-belief breakdown and run logs are in `results/oracle/CALIBRATION_EXTENSION_REPORT.md`.



## Independent calibration review and validation decision

Codex verified all 3,150 raw cases against their requested Cartesian products:
no missing/duplicate cases, all statuses complete, all recorded source hashes
match checkpoint d0f5428, and recorded strategy/coefficient settings match each
manifest. First-action losses were independently recomputed from policies and
oracle Q-values. The candidate is **bounded c=1**, with no production promotion.

The finite budget trend is not proof of asymptotic convergence or monotonicity
for individual cases. One bounded-c1 case (H2, seed 9, p=.9) has losses
0 / 2.1945 / 0 at 1k / 10k / 50k. At H3/50k on seeds 5-9, the candidate still
has a maximum first-action loss of 3.341678, and mean maximum absolute Q error
31.301129 despite only 1/35 wrong choices. Policy agreement does not establish
accurate action-value estimates.

The report's roughly 514-535 seconds per strategy are the **sum of case wall
times**, not measured elapsed panel times under two concurrent workers. The raw
case timings support those sums; they do not support the report's statement that
each panel took approximately 8.5 elapsed minutes.

The fixed held-out protocol is in HANDOFF.md: candidate bounded c=1 versus the
unchanged normalized-c1 baseline, new beliefs, seed indices 100-119, with 50k as
the primary budget and 10k/100k diagnostic budgets. Do not select a different
coefficient after inspecting that validation set. The gate concerns observed
first-action decisions on the specified L1 panel only, not L2/L3/L4 correctness,
value accuracy, statistical population guarantees, or full-suite qualification.

## Held-out L1 validation results, September 24

Antigravity completed both 2,160-case validation panels (4,320 total cases, 0 failures, 0 timeouts, 0 resource kills) under `results/oracle/heldout_bounded_c1_20260924` (candidate) and `results/oracle/heldout_normalized_c1_20260924` (baseline). Both runs used Git checkpoint `eef0e69`, Horizons 1–3, budgets 10k/50k/100k, seeds 100–119, and 12 fresh beliefs (.04/.06/.08/.12/.20/.35/.65/.80/.88/.92/.94/.96) under two supervised workers with 4 GiB limits. Total measured elapsed time was 3,155.8s for candidate and 2,943.1s for baseline; peak RSS was 75.8 MiB.

### Primary Decision Gate Outcome (50,000 simulations, N=720 cases: 240/horizon)

- **Gate criterion**: First-action loss $\le 10^{-8}$ for all 720 primary cases.
- **Outcome**: **FAILED** (Candidate: 80 / 720 wrong choices [11.11%], mean loss 0.0817; Baseline: 120 / 720 wrong choices [16.67%], mean loss 0.3811).
- Per protocol, candidate returns to development; this validation set becomes development evidence.

### Primary Budget Breakdown by Horizon (50,000 simulations)

| Horizon | Candidate Wrong / 240 | Candidate Mean Loss | Candidate Max Loss | Baseline Wrong / 240 | Baseline Mean Loss | Baseline Max Loss |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 / 240 (0.0%) | 0.0000 | 0.0000 | 0 / 240 (0.0%) | 0.0000 | 0.0000 |
| 2 | 40 / 240 (16.7%) | 0.0246 | 0.1478 | 40 / 240 (16.7%) | 0.0246 | 0.1478 |
| 3 | **40 / 240 (16.7%)** | **0.2205** | **1.3231** | **80 / 240 (33.3%)** | **1.1186** | **5.3884** |

### Per-Belief Breakdown at Primary Budget (50k simulations, N=20 seeds each)

- At Horizon 3:
  - Extreme safe door beliefs (.04, .06, .94, .96): Candidate 0/20 wrong; Baseline 0/20 wrong.
  - Intermediate beliefs (.12, .20, .35, .65, .80, .88): Candidate **0/20 wrong across all 6 beliefs (0.0000 loss)**; Baseline failed 20/20 at .12 and 20/20 at .88 (loss 5.3884 each).
  - Razor-thin boundary beliefs (.08 and .92): Both Candidate and Baseline failed 20/20 (Candidate loss 1.3231 vs Baseline loss 1.3231). At .08, $Q^*(L)=0.671$ vs $Q^*(OL)=-0.652$ (margin 1.3231), consistent with exploration costs depressing the listening estimate; root Q-values alone do not prove the cause.
- At Horizon 2:
  - Beliefs .04, .06, .12 through .88, .94, .96: Candidate 0/20 wrong.
  - Boundary beliefs .08 and .92: Both Candidate and Baseline failed 20/20 (margin $\Delta Q = 0.1478$).
- Diagnostic budget scaling (10k, 50k, 100k): Candidate Horizon 3 errors were 77/240 (10k) $\rightarrow$ 40/240 (50k) $\rightarrow$ 40/240 (100k).
- Full details in `results/oracle/HELDOUT_VALIDATION_REPORT.md`.


## Failed validation review and exact-tail development

Codex independently checked all 4,320 cases for complete requested coverage,
source-hash agreement, executed strategy and recomputed first-action loss. The
80/720 primary candidate failures are confirmed. These beliefs/seeds are now
explicitly development data. The full suite and default promotion remain blocked.
The causal description in the validation report was initially an inference from
root estimates; a separate tree-level calculation now gives supporting evidence.

A horizon-two diagnosis at p=.08, seed 100, 50k traversals, bounded c=1 found a
listening underestimate of 1.369056. Its additive decomposition is 1.281394 from
choosing suboptimal final actions, .058059 from final reward sampling, .029685
from observation-frequency sampling and -.000081 from first-expansion rollouts.
The true listening advantage is only .147767. This diagnosis used integer bound
parameters (-100,10,1), so its hash-derived stream differs from the floating-point
parameters in the validation panel; it is a separate diagnostic, not an exact
replay. Raw tree counts/values are in
`results/oracle/heldout_failure_diagnosis_20260924/h2-p008-seed100.json`.

The optional `--exact-final-step` evaluator implements the one-step Bellman
boundary on the full history-conditioned belief. Earlier backups remain sampled
means. There is no reference-solver call, hidden-state maximization, action mask,
or repaired posterior. THEORY.md defines the estimator and MODEL_SPECIFICATION.md
records the new configuration/seed implications. Production defaults are unchanged.

Development panel: H2/H3, budgets 10k/50k, seeds 100-102, beliefs
.08/.12/.5/.88/.92, gamma .95, bounded c=1, exact tail off/on. All 120 cases
completed with two supervised workers, 240-second and 1,024-MiB case limits.
Raw manifests/results: `results/exact-tail-20260924/{sampled,exact}`.

| Tail | H | Simulations | Wrong / 15 | Mean action loss | Mean max Q error | Mean case wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled | 2 | 10000 | 6 | 0.059107 | 13.821204 | 0.716 |
| sampled | 2 | 50000 | 6 | 0.059107 | 11.473404 | 1.684 |
| sampled | 3 | 10000 | 10 | 1.966139 | 32.865609 | 0.928 |
| sampled | 3 | 50000 | 6 | 0.529229 | 32.786549 | 2.532 |
| exact | 2 | 10000 | 0 | 0.000000 | 0.023249 | 0.734 |
| exact | 2 | 50000 | 0 | 0.000000 | 0.012855 | 1.636 |
| exact | 3 | 10000 | 6 | 0.529229 | 17.850371 | 0.936 |
| exact | 3 | 50000 | 6 | 0.529229 | 16.966682 | 2.646 |

The exact tail eliminates H2 decision errors on this small development panel.
At H3/50k, both configurations still fail every tested .08/.92 case: 6/15 errors.
The improved Q estimates do not meet the decision gate. Do not call this global
optimality, equivalence, or a successful held-out validation. Exact-tail rewards
are integrated, so equal traversal counts are not equal computational work.
The means above include worker startup; cost comparisons need larger panels.

Next: a bounded development budget curve on these already-inspected boundary
beliefs, with sampling and exact-tail controls at the same source checkpoint.
Investigate earlier-node exploration and chance-weighted Bellman estimators only
as explicit reviewed designs if those curves remain inadequate. Do not replace
mean returns by maxima over sampled states/outcomes to pass the oracle.


Verification of this implementation: 191 non-integration tests passed in 48.77s;
all four domain integration tests passed in 325.27s; Ruff lint and formatting
passed. The 120-case development panel completed with source hashes verified.
No production default has been promoted. Domain integration tests exercise the
default path; the new exact-tail path has analytic tests and L1 development cases,
not full L2/L3/L4 experimental qualification.

## Exact final-step budget curve results, September 24

Antigravity completed both 90-case budget panels (180 total cases, 0 failures, 0 timeouts, 0 resource kills) under `results/oracle/tail_budget_sampled_20260924` (Sampled Tail) and `results/oracle/tail_budget_exact_20260924` (Exact Final Step). Both runs used Git checkpoint `582bf1a`, Horizons 2–3, budgets 50k/200k/1M, seeds 100–104, boundary beliefs .08/.50/.92, and `bounded c=1.0` under two supervised workers with 4 GiB limits. Runner-reported elapsed times were 712.7s for sampled tail and 770.2s for exact tail (about 8.07% higher); peak RSS was 82.7 MiB. The saved case files independently record per-case timings, not panel start/end timestamps.

| Mode | Horizon | Budget | Wrong / 15 | Error Rate | Mean Loss | Max Loss | Mean Max Q Error | Mean Wall Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sampled Tail** | 2 | 50,000 | 10 / 15 | 66.7% | 0.0985 | 0.1478 | 11.51 | 1.44 s |
| **Sampled Tail** | 2 | 200,000 | 10 / 15 | 66.7% | 0.0985 | 0.1478 | 11.09 | 4.12 s |
| **Sampled Tail** | 2 | 1,000,000 | 8 / 15 | 53.3% | 0.0788 | 0.1478 | 10.12 | 18.45 s |
| **Exact Final Step** | 2 | 50,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | **0.01** | 1.34 s |
| **Exact Final Step** | 2 | 200,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | **0.01** | 3.88 s |
| **Exact Final Step** | 2 | 1,000,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | **0.00** | 17.25 s |
| **Sampled Tail** | 3 | 50,000 | 10 / 15 | 66.7% | 0.8820 | 1.3231 | 30.64 | 2.21 s |
| **Sampled Tail** | 3 | 200,000 | 10 / 15 | 66.7% | 0.8820 | 1.3231 | 28.59 | 7.49 s |
| **Sampled Tail** | 3 | 1,000,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | 26.03 | 35.97 s |
| **Exact Final Step** | 3 | 50,000 | 10 / 15 | 66.7% | 0.8820 | 1.3231 | **16.40** | 2.32 s |
| **Exact Final Step** | 3 | 200,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | **15.24** | 7.91 s |
| **Exact Final Step** | 3 | 1,000,000 | **0 / 15** | **0.0%** | **0.0000** | **0.0000** | **14.49** | 37.88 s |

Key findings:
- Exact Final Step completely eliminates Horizon 2 boundary errors across all tested budgets (0/15 errors at 50k, 200k, 1M sims; mean max Q errors 0.010421, 0.007358, 0.002522 respectively), whereas Sampled Tail still fails in 8/15 cases at 1,000,000 simulations.
- At Horizon 3, Exact Final Step achieves 0/15 errors at 200,000 simulations and maintains 0/15 at 1,000,000 simulations (the first tested zero-error budgets were 200k and 1M respectively; this coarse grid does not establish a fivefold convergence rate or speedup).
- Exact Final Step reduces Horizon 3 mean max Q error from 26.03 (Sampled Tail) down to 14.49 at 1M simulations.
- Detailed tables are in `results/oracle/TAIL_BUDGET_CURVE_REPORT.md`.



## Independent review of the 180-case budget panel

All 180 requested cases were verified: complete coverage without duplicates,
source hashes matching 582bf1a, executed mode/strategy matching the manifest,
row settings matching case keys, and first-action loss recomputed from saved
policies and reference Q-values. The reported action counts are supported.

The warranted conclusion is improved finite-budget action selection on the tested
L1 histories. Mean sampled-return backups and UCB exploration are part of standard
POMCP, not by themselves implementation defects (Silver and Veness 2010, sections
2.4 and 3.1). Exact final-step integration is an explicitly different finite-model
boundary evaluator. It removes final-step action exploration by construction,
while earlier sampling/exploration errors remain. These experiments do not prove
asymptotic convergence of this implementation or a structural error in POMCP.

Every reference-optimal action in this development grid (.08/.5/.92, H2/H3) is
**listen**. An always-listen policy would pass its decision test. Consequently the
zero-error cells do not establish correct opening decisions or global policy
quality. At H3/200k, mean max Q error remains 15.238546; at H3/1M it is 14.486203.
At H2/1M, the displayed 0.00 Q error is rounded: the actual mean is 0.002522351.
Zero first-action loss does not mean an exact value function or zero episode regret.

Verified cumulative case wall times are 1,045.348s sampled and 1,058.789s exact,
about 1.29% apart. This is a different quantity from runner-reported elapsed panel
time. Neither figure predicts overhead for deeper interactive posteriors. The
5:1 ratio of the first passing tested budgets is not a measured convergence rate.

Decision: freeze bounded c=1, exact_final_step=true, 200k traversals as a candidate
for a **new** held-out L1 action test on both opening and listening regions. The
previous 50k validation remains failed. No production default is promoted.
HANDOFF.md states fresh beliefs/seeds and the gate before execution. Higher-level
horizon matching and resource qualification remain separate unresolved work.

Evidence storage: 32a01ce commits the report text in HANDOFF.md and this file.
The raw result directories and manifests are present locally but ignored by Git;
`git ls-files results/oracle` is empty. Do not describe those raw files as committed.
Preserve them in place, retain their source manifests, and report any separate
archive location only after such an archive has actually been created.

## Frozen 200k L1 validation results, September 24

Antigravity completed both 720-case panels (1,440 total cases, 0 failures, 0 timeouts, 0 resource kills) under `results/oracle/tail_validation_exact_200k_20260924` (Candidate) and `results/oracle/tail_validation_sampled_200k_20260924` (Control). Both runs used Git checkpoint `ccf99b6`, Horizons 1–3, budget 200,000, seeds 200–219, and 12 fresh beliefs (.03/.07/.075/.085/.11/.25/.75/.89/.915/.925/.93/.97) under two supervised workers with 4 GiB limits. Measured elapsed panel times were 2,870.3s (Candidate) and 3,000.1s (Control); peak monitored RSS was 76.5 MiB.

### Primary Decision Gate Outcome (200,000 traversals, N=720 cases)

- **Gate criterion**: First-action loss $\le 10^{-8}$ for all 720 candidate cases.
- **Outcome**: **FAILED** (Candidate: 80 / 720 wrong choices [11.11%], mean loss 0.062899; Control: 85 / 720 wrong choices [11.81%], mean loss 0.075592).
- Per protocol, candidate returns to development; these validation points become development evidence.

### Breakdown by Horizon (200,000 traversals, N=240 cases each)

| Horizon | Candidate Wrong / 240 | Candidate Mean Loss | Candidate Max Loss | Candidate Mean Q Err | Control Wrong / 240 | Control Mean Loss | Control Max Loss | Control Mean Q Err |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **0 / 240 (0.0%)** | **0.0000** | **0.0000** | **0.00** | **0 / 240 (0.0%)** | 0.0000 | 0.0000 | 0.00 |
| **2** | **0 / 240 (0.0%)** | **0.0000** | **0.0000** | **0.01** | **0 / 240 (0.0%)** | 0.0000 | 0.0000 | 12.60 |
| **3** | **80 / 240 (33.3%)** | **0.1887** | **0.8184** | **16.85** | 85 / 240 (35.4%) | 0.2268 | 1.8277 | 32.00 |
| **Total** | **80 / 720 (11.11%)** | **0.0629** | **0.8184** | — | **85 / 720 (11.81%)** | 0.0756 | 1.8277 | — |

### Per-Belief Breakdown Across All Horizons (N=20 seeds per cell)

- **Horizons 1 and 2**: Both Candidate and Control achieved **0 errors out of 240 cases each (0.0000 loss)** across all 12 beliefs. At H1, opening is optimal at .03/.07/.075/.085/.915/.925/.93/.97, and listening at .11/.25/.75/.89. At H2, opening is optimal at .03/.07/.075/.925/.93/.97, and listening at .085/.11/.25/.75/.89/.915.
- **Horizon 3**:
  - Safe door-opening beliefs (.03, .97): Candidate 0/20 wrong; Control 0/20 wrong.
  - Interior listening beliefs (.085, .11, .25, .75, .89, .915): Candidate **0/20 wrong across all 6 beliefs (0.0000 loss)**; Control failed 2/20 at .085 and 3/20 at .915.
  - Razor-thin inflection beliefs (.070, .075, .925, .930): Both Candidate and Control failed 20/20 (Candidate loss 0.3138 at .070/.930 and 0.8184 at .075/.925). The reference optimal action flips from door-opening (H1/H2) to listening (H3), but the true margin ($\Delta Q \in [0.31, 0.82]$) is smaller than intermediate tree exploration suppression at depth 1.
- Detailed tables are in `results/oracle/TAIL_VALIDATION_200K_REPORT.md`.



## Independent review of the 1,440-case validation and next design

Codex verified all 1,440 requested cases: unique coverage, source hashes,
row/case settings, actual tail/exploration modes and recomputed first-action
losses. The 200k gate remains failed: exact tail 80/720 errors, sampled tail
85/720. These are measured outcomes at twelve belief points, not an exact global
mapping of the action boundary. The report incorrectly grouped the H1/H2 optimal
actions together; the corrected per-horizon action sets are above. In particular,
p=.085/.915 favors opening at H1 and listening at H2.

The next opt-in estimator is `empirical_bellman`: chance frequencies average
successor history values, while decisions maximize only at a private-history
node. Terminal samples remain in the action-count denominator. Frontier rollout
estimates are explicit and replaced when action values are available. See
THEORY.md for the formula, relation to prior work, and finite-budget limitations.
Production defaults remain sampled backups and a sampled final step.

As before, e7f27ff commits the documentation only; the raw result directories and
manifests are local and Git-ignored. Preserve those artifacts, and do not report
that they were committed. All previous gate failures remain failures.


## Empirical Bellman backup development comparison

Source checkpoint: 2230786. Exact final-step integration is enabled in both
controls; only the intermediate backup estimator differs. Both use bounded c=1,
gamma .95, H2/H3, budgets 10k/50k, inspected beliefs
.03/.07/.075/.5/.925/.93/.97 and seeds 200-201. All 112 cases completed under
two workers, 240-second and 1,024-MiB per-case limits. Source hashes were verified.
Raw cases and manifests: `results/bellman-backup-20260924/{sampled,empirical_bellman}`.

| Backup | H | Traversals | Errors / 14 | Mean first-action loss | Mean max Q error | Mean case wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sampled | 2 | 10000 | 0 | 0.000000 | 0.064570 | 0.710 |
| sampled | 2 | 50000 | 0 | 0.000000 | 0.017752 | 1.466 |
| sampled | 3 | 10000 | 8 | 0.323483 | 17.410287 | 0.915 |
| sampled | 3 | 50000 | 8 | 0.323483 | 17.635496 | 2.565 |
| empirical_bellman | 2 | 10000 | 0 | 0.000000 | 0.038409 | 0.717 |
| empirical_bellman | 2 | 50000 | 0 | 0.000000 | 0.023486 | 1.637 |
| empirical_bellman | 3 | 10000 | 0 | 0.000000 | 0.070815 | 1.029 |
| empirical_bellman | 3 | 50000 | 0 | 0.000000 | 0.033812 | 3.001 |

These are development points, not a new held-out pass. At H3/50k, empirical
Bellman removed the 8/14 observed errors and substantially reduced Q error;
it also cost more per case. Equal traversal counts are not equal work. No
convergence, population-error bound or production readiness follows from this
small panel. No confidence pruning or hidden-state maximization was introduced.

A further 20-case smoke panel used the same estimator and tail at H4/H5,
budgets 1k/10k, seed 200 and beliefs .03/.075/.5/.925/.97. All completed.

| Horizon | Traversals | Errors / 5 | Mean first-action loss | Mean max Q error |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1000 | 2 | 0.003970 | 6.904961 |
| 4 | 10000 | 1 | 0.001985 | 3.340033 |
| 5 | 1000 | 2 | 0.007254 | 6.878925 |
| 5 | 10000 | 2 | 0.007254 | 4.567808 |

Remaining wrong choices occur at .075/.925, with losses approximately .009924
at H4 and .018136 at H5. These are actual near-tie decision errors, not floating-
point ties. They fail strict action agreement; no tolerance is introduced after
observing them. A scientific loss tolerance, if desired, must be fixed before a
future validation and cannot reverse old gate failures. No default is promoted.

### 1,800-case H1–H5 development comparison (sampled vs empirical_bellman)

Source checkpoint: 7d8b6bd. Exact final-step integration is enabled in both
panels (`--exact-final-step`), along with bounded UCB ($c=1.0$). Both use $\gamma=0.95$,
Horizons 1–5, budgets 1,000, 10,000, and 50,000, seeds 200–204 (5 seeds), and the 12
development beliefs: 0.03, 0.07, 0.075, 0.085, 0.11, 0.25, 0.75, 0.89, 0.915, 0.925, 0.93, 0.97.
Both panels completed sequentially without failures, timeouts, or RSS kills (900 cases each,
1,800 cases total).

Raw cases and manifests:
- Control: `results/oracle/bellman_development_sampled_20260924/`
- Candidate: `results/oracle/bellman_development_empirical_20260924/`

#### Overall performance summary

| Backup | Total Cases | Errors (loss > 1e-8) | Error Rate | Mean First-Action Loss | Max First-Action Loss | Mean Max Q Error | Total Worker Wall (s) | Elapsed Panel Wall (s) | Monitored Peak RSS (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **sampled** | 900 | 206 | 22.89% | 0.36433 | 4.58069 | 22.04757 | 1181.29 | 602.29 | 82.42 |
| **empirical_bellman** | 900 | 89 | 9.89% | 0.02227 | 1.00748 | 1.79991 | 1292.68 | 657.76 | 84.34 |

#### Breakdown by horizon and budget

| H | Budget | Sampled Errs / 60 | Sampled Mean Loss | Sampled Mean Max Q Error | Empirical Errs / 60 | Empirical Mean Loss | Empirical Mean Max Q Error | Wall Ratio (Emp/Samp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 0 (0.0%) | 0.00000 | 0.0000 | 0 (0.0%) | 0.00000 | 0.0000 | 1.06x |
| 1 | 10,000 | 0 (0.0%) | 0.00000 | 0.0000 | 0 (0.0%) | 0.00000 | 0.0000 | 0.98x |
| 1 | 50,000 | 0 (0.0%) | 0.00000 | 0.0000 | 0 (0.0%) | 0.00000 | 0.0000 | 1.05x |
| 2 | 1,000 | 1 (1.7%) | 0.00607 | 0.1381 | 0 (0.0%) | 0.00000 | 0.1133 | 1.00x |
| 2 | 10,000 | 0 (0.0%) | 0.00000 | 0.0502 | 0 (0.0%) | 0.00000 | 0.0413 | 0.99x |
| 2 | 50,000 | 0 (0.0%) | 0.00000 | 0.0201 | 0 (0.0%) | 0.00000 | 0.0196 | 1.11x |
| 3 | 1,000 | 21 (35.0%) | 0.24439 | 20.3144 | 2 (3.3%) | 0.01046 | 0.4745 | 1.02x |
| 3 | 10,000 | 30 (50.0%) | 0.49332 | 18.6191 | 1 (1.7%) | 0.00523 | 0.0986 | 1.07x |
| 3 | 50,000 | 30 (50.0%) | 0.49332 | 17.6159 | **0 (0.0%)** | **0.00000** | **0.0388** | 1.12x |
| 4 | 1,000 | 20 (33.3%) | 0.70602 | 39.3047 | 20 (33.3%) | 0.08782 | 6.1670 | 0.98x |
| 4 | 10,000 | 20 (33.3%) | 0.76553 | 38.7978 | 9 (15.0%) | 0.00149 | 2.9157 | 1.04x |
| 4 | 50,000 | 20 (33.3%) | 0.76553 | 38.6097 | 5 (8.3%) | 0.00083 | 2.2934 | 1.13x |
| 5 | 1,000 | 24 (40.0%) | 0.46439 | 52.5918 | 19 (31.7%) | 0.10710 | 6.6125 | 1.03x |
| 5 | 10,000 | 20 (33.3%) | 0.76322 | 51.2494 | 20 (33.3%) | 0.09151 | 4.6968 | 1.02x |
| 5 | 50,000 | 20 (33.3%) | 0.76322 | 53.4023 | 13 (21.7%) | 0.02957 | 3.5270 | 1.16x |

#### Per-belief error counts (Sampled vs Empirical Bellman over 5 seeds)

| H | Budget | 0.030 | 0.070 | 0.075 | 0.085 | 0.110 | 0.250 | 0.750 | 0.890 | 0.915 | 0.925 | 0.930 | 0.970 |
| ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1,000 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 1 | 10,000 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 1 | 50,000 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 2 | 1,000 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 2 | 10,000 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 2 | 50,000 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 3 | 1,000 | 0/0 | 5/1 | 5/0 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 5/0 | 4/1 | 0/0 |
| 3 | 10,000 | 0/0 | 5/1 | 5/0 | 5/0 | 0/0 | 0/0 | 0/0 | 0/0 | 5/0 | 5/0 | 5/0 | 0/0 |
| 3 | 50,000 | 0/0 | 5/0 | 5/0 | 5/0 | 0/0 | 0/0 | 0/0 | 0/0 | 5/0 | 5/0 | 5/0 | 0/0 |
| 4 | 1,000 | 0/0 | 0/5 | 1/5 | 5/0 | 4/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/5 | 0/5 | 0/0 |
| 4 | 10,000 | 0/0 | 0/0 | 0/4 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/5 | 0/0 | 0/0 |
| 4 | 50,000 | 0/0 | 0/0 | 0/2 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/3 | 0/0 | 0/0 |
| 5 | 1,000 | 3/0 | 5/5 | 5/4 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 5/5 | 4/3 | 2/0 |
| 5 | 10,000 | 0/0 | 0/5 | 0/5 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/5 | 0/5 | 0/0 |
| 5 | 50,000 | 0/0 | 0/1 | 0/5 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/5 | 0/2 | 0/0 |

#### Key empirical conclusions

1. **Horizon 3 agreement on the tested 50k panel**: At $H=3$, at 50k sampled means failed on all 6 tested boundary beliefs ($p \in \{0.07, 0.075, 0.085, 0.915, 0.925, 0.93\}$) with a 50% error rate and mean Q error of 17.62 due to on-policy leaf exploration penalties dragging down $\hat{Q}(L)$. Empirical Bellman backups completely eliminated this error at 50,000 traversals: 0/60 errors, 0.00000 policy loss, and mean max Q error reduced 450x to 0.0388.
2. **Listen vs Open error asymmetry**: In Sampled Means, errors predominantly hit Listen cases (180/480 errors, 37.5%, mean loss 0.6243) where exploratory door openings drag down listening values. In Empirical Bellman, Listen cases achieved near-perfect accuracy (only 5/480 errors, 1.04%, mean loss 0.00616).
3. **Horizon 4 and 5 near-tie inflection errors**: At $H=4$ and $H=5$, the true optimal policy boundary shifts outward (opening doors becomes optimal at $p=0.070, 0.075, 0.925, 0.930$). At $p=0.075$ and $p=0.925$, the theoretical gap is miniscule ($\Delta Q^* = 0.00992$ at $H=4$ and $0.01814$ at $H=5$). Under Empirical Bellman, the solver favored Listen over Open, incurring losses of exactly 0.00992 at $H=4$ and 0.01814 at $H=5$. At H4/50k, all five errors have loss approximately 0.009924 (mean loss 0.00083 across all 60 cases). At H5/50k, three additional errors at p=.07/.93 have much larger loss 0.530942; the remaining errors cannot all be described as near ties.
4. **Computational footprint**: Empirical Bellman backups incurred a modest 9.2% wall-clock overhead across 900 cases (657.8s vs 602.3s elapsed panel time; mean 1.44s vs 1.31s per case) and virtually identical peak RSS (+1.92 MB max peak).
5. **Validation posture**: These findings represent development evidence on previously inspected points. They do not constitute a held-out validation pass or authorize changing default planner settings. The user-selected bounded first-action loss tolerance threshold remains pending numerical definition before fresh held-out validation can be frozen.



### Independent review of the 1,800-case report (933d917)

Codex verified all 1,800 case files, complete/unique requested coverage, manifest
source hashes against 7d8b6bd, settings and status. The exact L1 reference was
rerun for all 60 horizon/belief pairs; recorded oracle Q values, policy losses
and maximum absolute Q errors match. Audit aggregates are preserved locally in
results/oracle/bellman_development_review_20260924.json. This rerun checks the
report against the existing reference, not an independent proof of that solver.

The aggregate improvements above are confirmed. However, the earlier handoff's
near-tie takeaway omitted larger H5 errors. At H5/50k, ten candidate errors lose
about 0.018136 each, while three lose **0.53094248946**: p=.07/seed202 and
p=.93/seeds201,202. A proposed per-case bound of .020 would therefore still fail
3/60 cases at this horizon/budget. Lower-budget errors reach 1.00747779510.
The .020 suggestion is not an approved threshold. No previous failed gate is
reclassified, and no general action-boundary or convergence guarantee follows.

H3/50k agreement is an empirical result on 60 development cases. The change in
backup semantics supports the exploration-averaging diagnosis, but aggregate
action agreement alone does not isolate every source of finite-budget error.
The table's Q-error columns are means of per-case maximum absolute action-value
errors, not the maximum across all cases; headings have been clarified.

Next: fixed H4/H5 budgets 200k/1M, both backups, the same twelve development
beliefs and seeds 200-204 (480 cases total). See HANDOFF.md. This measures whether
larger computation reduces the material H5 losses before selecting a candidate
for fresh validation. No production default or theoretical claim is changed.

### 480-case H4–H5 larger-budget development comparison (200k and 1M traversals)

Source checkpoint: 66f9086. Evaluated both `backup="sampled"` and `backup="empirical_bellman"`
with `--exact-final-step`, bounded UCB ($c=1.0$), $\gamma=0.95$, across Horizons 4 and 5,
budgets of 200,000 and 1,000,000 traversals, seeds 200–204 (5 seeds), and all 12 development
beliefs: 0.03, 0.07, 0.075, 0.085, 0.11, 0.25, 0.75, 0.89, 0.915, 0.925, 0.93, 0.97 (240 cases
per mode, 480 total). Both panels executed sequentially under 2 supervised workers without
timeouts, crashes, or memory kills.

Raw cases and manifests:
- Control: `results/oracle/bellman_budget_sampled_20260924/`
- Candidate: `results/oracle/bellman_budget_empirical_20260924/`

#### Overall performance summary

| Backup | Total Cases | Errors (loss > 1e-8) | Error Rate | Mean First-Action Loss | Max First-Action Loss | Mean Max Q Error | Total Worker Wall (s) | Elapsed Panel Wall (s) | Monitored Peak RSS (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **sampled** | 240 | 60 | 25.00% | 0.46600 | 3.57187 | 44.81183 | 10,152.4 | unavailable; reported value invalid | 107.42 |
| **empirical_bellman** | 240 | 14 | **5.83%** | **0.00085** | **0.01814** | **2.47296** | 12,064.2 | unavailable; reported value invalid | 113.39 |

#### Breakdown by horizon and budget

| H | Budget | Sampled Errs / 60 | Sampled Mean Loss | Sampled Mean Max Q Error | Empirical Errs / 60 | Empirical Mean Loss | Empirical Mean Max Q Error | Wall Ratio (Emp/Samp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 200,000 | 10 (16.7%) | 0.16877 | 38.0633 | **5 (8.3%)** | **0.00083** | **1.7239** | 1.16x |
| 4 | 1,000,000 | 10 (16.7%) | 0.16877 | 36.6313 | **1 (1.7%)** | **0.00017** | **1.3603** | 1.19x |
| 5 | 200,000 | 20 (33.3%) | 0.76322 | 52.0554 | **5 (8.3%)** | **0.00151** | **3.3173** | 1.19x |
| 5 | 1,000,000 | 20 (33.3%) | 0.76322 | 52.4974 | **3 (5.0%)** | **0.00091** | **3.4903** | 1.19x |

#### Per-belief error counts (Sampled vs Empirical Bellman over 5 seeds)

| H | Budget | 0.030 | 0.070 | 0.075 | 0.085 | 0.110 | 0.250 | 0.750 | 0.890 | 0.915 | 0.925 | 0.930 | 0.970 |
| ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 200,000 | 0/0 | 0/0 | 0/3 | 5/0 | 0/0 | 0/0 | 0/0 | 0/0 | 5/0 | 0/2 | 0/0 | 0/0 |
| 4 | 1,000,000 | 0/0 | 0/0 | 0/1 | 5/0 | 0/0 | 0/0 | 0/0 | 0/0 | 5/0 | 0/0 | 0/0 | 0/0 |
| 5 | 200,000 | 0/0 | 0/0 | 0/3 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/2 | 0/0 | 0/0 |
| 5 | 1,000,000 | 0/0 | 0/0 | 0/2 | 5/0 | 5/0 | 0/0 | 0/0 | 5/0 | 5/0 | 0/1 | 0/0 | 0/0 |

#### Resolution of the three H5/50k material errors

The three large-loss cases identified at $H=5, 50\text{k}$ in the previous audit:
1. $p=0.070$, seed 202 (loss was 0.53094 at 1k, 10k, 50k): resolved to **0.00000** at 200k ($\hat{Q}(OL)=1.4083 > \hat{Q}(L)=0.9192$) and **0.00000** at 1M ($\hat{Q}(OL)=1.4177 > \hat{Q}(L)=0.8813$).
2. $p=0.930$, seed 201 (loss was 0.53094 at 1k, 10k, 50k): resolved to **0.00000** at 200k ($\hat{Q}(OR)=1.3995 > \hat{Q}(L)=0.9339$) and **0.00000** at 1M ($\hat{Q}(OR)=1.4603 > \hat{Q}(L)=0.9325$).
3. $p=0.930$, seed 202 (loss was 0.53094 at 10k, 50k): resolved to **0.00000** at 200k ($\hat{Q}(OR)=1.3576 > \hat{Q}(L)=0.9371$) and **0.00000** at 1M ($\hat{Q}(OR)=1.4234 > \hat{Q}(L)=0.8791$).

Across both 200k and 1M budgets, $p=0.070$ and $p=0.930$ achieved **0/5 errors** in Empirical Bellman. Added computation fully eliminated the material $0.53094$ errors.

#### Key findings and remaining errors

1. **Observed maximum loss across 240 cases**: The maximum first-action loss observed anywhere in the 240 Empirical Bellman cases is **0.01814** (at $H=5, p \in \{0.075, 0.925\}$). At $H=4$, the maximum loss is **0.00992**.
2. **Observed accuracy at H4/1M**: At 1,000,000 traversals, Empirical Bellman achieved **59/60 strictly optimal decisions (98.33%)**, with only 1 seed failing at $p=0.075$ with loss 0.00992 (mean loss across all 60 cases was 0.00017).
3. **Observed accuracy at H5/1M**: At 1,000,000 traversals, Empirical Bellman achieved **57/60 strictly optimal decisions (95.00%)**, with only 3 remaining cases failing at $p=0.075$ (2 seeds) and $p=0.925$ (1 seed), each with loss 0.01814 (mean loss across all 60 cases was 0.00091).
4. **Persistent sampled-mean errors at the tested budgets**: Even at 1,000,000 traversals, Sampled Means failed on 10/60 cases at $H=4$ and 20/60 cases at $H=5$ (50.0% failure on Listen optimal actions), with mean loss 0.46600 and mean max Q error of 44.81. The tested budgets retain these errors; this does not establish failure at all larger budgets or contradict asymptotic convergence under appropriate assumptions.
5. **Computational Cost**: Verified summed worker wall time increased by 18.83% (12,064.170s versus 10,152.416s), with peak monitored RSS 113.4 versus 107.4 MiB. The originally reported elapsed times are inconsistent with two workers and are withdrawn; see the audit below.



### Independent review of 33c3356

All 480 cases have verified coverage, status, source fingerprints against 66f9086
and settings. Exact oracle Q values were recomputed at all 24 horizon/belief
pairs, and all policy losses/Q errors recomputed. The reported accuracy totals
are confirmed. The three previously material H5 errors vanish at 200k and 1M
on the tested seeds; all candidate losses are at most .018135727943.

The report's panel elapsed times (4818.7/5708.4 s) are inconsistent with the raw
worker sums (10152.416/12064.170 s): two workers require at least half those sums,
5076.208/6032.085 s elapsed. Actual elapsed cannot be recovered from per-case
durations alone. The aggregate worker-time ratio is 1.1883; it is not verified
elapsed overhead or a measurement isolating one algorithm operation. Future
panels must capture a direct external elapsed timer. Raw cases remain unchanged.

The finite-budget accuracy result is not a convergence proof. H5 mean max Q
error increases from 3.317310 at 200k to 3.490327 at 1M, despite fewer decisions
being wrong. Do not conflate action selection with value-estimate accuracy.

Selected configuration for prepared fresh L1 validation: empirical Bellman,
exact final step, bounded c=1, gamma=.95, 1M traversals, H1-H5. HANDOFF.md specifies
2000 untouched cases using twenty new beliefs and seeds 1000-1019. No overlap
was found in 31 panel manifests or 12,513 local recorded case rows; no new case
has been evaluated. The numerical per-case tolerance remains pending user input,
at that preparation checkpoint. Threshold freezing and execution are recorded
below. Old gates stay failed, defaults unchanged.
Audit: results/oracle/bellman_budget_review_20260924.json (local, Git-ignored).

## Frozen 2,000-case fresh held-out Level-1 validation suite

Source checkpoint: a866e69. The numerical first-action loss tolerance threshold was
explicitly frozen at $\epsilon_{\text{loss}} = 0.020$ per user authorization before execution.
Evaluated candidate configuration: `backup="empirical_bellman"`, `--exact-final-step`,
bounded UCB ($c=1.0$), $\gamma=0.95$, fixed budget of 1,000,000 traversals at each of
Horizons 1–5. Evaluated across 20 strictly unseen beliefs:
`0.005, 0.035, 0.065, 0.0725, 0.0775, 0.0825, 0.095, 0.125, 0.225, 0.375, 0.625, 0.775, 0.875, 0.905, 0.9175, 0.9225, 0.9275, 0.935, 0.965, 0.995`
and 20 fresh seeds: `1000–1019` (2,000 cases total). Zero overlap exists with any previous
development run. Elapsed time was directly measured using GNU time.

Raw cases, manifest, and logs:
- Results directory: `results/oracle/bellman_validation_20260924/`
- Direct GNU time log: `results/oracle/bellman_validation_20260924.time` (`elapsed_seconds=43392.04`)
- Execution log: `results/oracle/bellman_validation_20260924.log`

### Validation Gate Verdict: PASSED

| Metric | Target Specification | Observed Result | Verdict |
| :--- | :---: | :---: | :---: |
| **Total Requested Cases** | 2,000 cases | 2,000 cases | Complete (100.0%) |
| **Incomplete / Timed Out Cases** | 0 allowed | 0 (0.00%) | **PASS** |
| **Primary Gate Violations ($\text{loss} > 0.020 + 10^{-8}$)** | 0 allowed | **0 / 2,000 (0.00%)** | **PASS** |
| **Strict Action Errors ($\text{loss} > 10^{-8}$)** | Diagnostic | **0 / 2,000 (0.00%)** | **100.0% Strict Pass** |
| **Mean First-Action Policy Loss** | Diagnostic | **0.000000** | Perfect first-action agreement |
| **Max First-Action Policy Loss** | Diagnostic | **0.000000** | Perfect first-action agreement |
| **Direct Elapsed Panel Wall Time** | Monitored | 43,392.04 s recorded; inconsistent | Unresolved; see audit |
| **Total Worker Wall Time** | Monitored | 90,054.3 s (25.02 h) | Average 45.03 s / case |
| **Monitored Peak RSS** | < 4,096 MB | **113.42 MB** | Well within memory ceiling |

### Breakdown by Horizon (400 cases per horizon)

| H | Cases | Gate Failures ($\text{loss} > 0.02$) | Strict Errors | Mean Loss | Max Loss | Mean Max Q Error | Max Max Q Error | Mean Case Wall (s) | Peak RSS (MB) |
| :---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 400 | 0 (0.0%) | 0 (0.0%) | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 0.83 | 82.0 |
| **2** | 400 | 0 (0.0%) | 0 (0.0%) | 0.000000 | 0.000000 | 0.0098 | 0.1087 | 19.50 | 82.2 |
| **3** | 400 | 0 (0.0%) | 0 (0.0%) | 0.000000 | 0.000000 | 0.0108 | 0.1012 | 44.47 | 82.8 |
| **4** | 400 | 0 (0.0%) | 0 (0.0%) | 0.000000 | 0.000000 | 1.3217 | 5.5954 | 68.80 | 87.2 |
| **5** | 400 | 0 (0.0%) | 0 (0.0%) | 0.000000 | 0.000000 | 3.6684 | 6.4767 | 91.54 | 113.4 |

### Optimal Action Category Breakdown

- **Optimal Action = Listen ($N=1,080$)**: 0 gate failures, 0 strict errors (100% agreement, loss = 0.000000).
- **Optimal Action = Open ($N=920$)**: 0 gate failures, 0 strict errors (100% agreement, loss = 0.000000).

### Key Takeaways

1. **Definitive Validation Pass**: The candidate planner (`backup="empirical_bellman"` + `--exact-final-step` + bounded UCB $c=1.0$ at 1,000,000 traversals) achieved a 100.0% success rate across all 2,000 held-out cases, meeting the frozen $\epsilon_{\text{loss}} \le 0.020$ primary criterion with 0 failures, and furthermore achieving 0 strict errors across the entire grid.
2. **Agreement across the five tested horizons and twenty belief points**: From $H=1$ through $H=5$, every single decision matched the exact oracle optimal policy across both extreme tails ($p=0.005, 0.995$), boundary inflection zones ($p \in [0.065, 0.095]$ and $p \in [0.905, 0.935]$), and uninformative interior beliefs ($p \in [0.125, 0.875]$).
3. **Execution Integrity**: The suite ran uninterrupted to completion. All recorded cases completed, and sampled peak RSS was 113.42 MiB. GNU time recorded 43,392.04 seconds, but this contradicts the worker-duration lower bound; elapsed-time integrity remains unresolved.
4. **Scope and Future Gates**: This result passes the finite L1 first-action validation panel at the evaluated settings; it does not certify all beliefs, horizons, full policies or value estimates. In accordance with the project charter, previous 50k and 200k validation failures remain historical failures, and Level-2/Level-4 opponent qualification and production default promotions remain separate pending milestones.



### Independent audit of 1b3527a (September 25)

All 2000 cases, source hashes against a866e69, exact requested coverage, settings
and summary statuses were checked. Exact reference Q values were recomputed for
100 horizon/belief pairs; all reported policy losses and maximum absolute Q
errors match. The frozen .020 + 1e-8 numerical gate passes with zero violations,
zero strict first-action errors and exactly zero recorded loss on this panel.

This is evidence for the specified L1 configuration (uniform L0 opponent, H1-H5,
twenty physical beliefs, seeds1000-1019, gamma=.95, bounded c=1, empirical Bellman,
exact final step, 1M traversals). It is not proof of permanent resolution or
agreement at untested horizons/beliefs/budgets. Prior 1M development cases still
include four action errors. H5 mean max Q error is 3.668401, with largest
per-case error 6.476666: selected actions match, estimated values are not exact.
Global defaults remain unchanged because their configuration and modeled-policy
resources differ. L2 matching precedes L4/all39 production qualification.

The external .time file does contain 43392.04 s. Verified worker durations sum
to 90054.254809 s, requiring at least 45027.127405 s with two workers on a
comparable elapsed clock. These records therefore conflict. The cause has not
been established; clock domains and timing provenance require investigation.
Do not label the 12.05 h figure verified or replace it with the lower bound as
an actual measurement. This discrepancy limits runtime conclusions without
changing the independently recomputed action-loss verdict. Raw files are retained.

Audit aggregates: results/oracle/bellman_validation_review_20260925.json
(local and Git-ignored). Next steps and the exact default-promotion limitations
are recorded in HANDOFF.md. No solver code changed in this review.


## Matched fixed-depth L2 implementation and smoke (September 25)

The comparison now declares an exact L1 opponent replanning at a fixed depth,
with a point prior on its private physical belief. Both reference and MCTS use
that same policy model; this is not a finite-budget production-opponent test.
See L2_CONTRACT.md for the complete joint-belief recursion and information limits.

An 18-case smoke used H1-H3, 1000 traversals, own beliefs .1/.5/.9,
b_j=.085, opponent depth2, seeds300-301, bounded c=1, empirical Bellman and
exact final-step integration. Every case completed with zero first-action loss.
Maximum Q error at H3 was .722473. This is development evidence only.

New timing records show parent monotonic elapsed 5.329160s, worker sum 9.111192s,
two workers, and a satisfied concurrency bound. GNU time reports 5.29s; the
saved realtime endpoints differ by 4.796633s. These clock differences are exposed,
not resolved. The runner retains parent timing on failures and rejects worker
durations inconsistent with its monotonic concurrency bound. Older raw records
are unchanged. Short-run timing under concurrent test load is not a speed claim.

Raw evidence: results/l2-contract-20260925/smoke/ plus smoke.log and smoke.time.
HANDOFF.md specifies the next 900-case development panel. Defaults remain unchanged.

## 900-case Level-2 fixed-depth contract development study (September 25)

Source checkpoint: 01f36cf (preceded by implementation at 1fad300).
Evaluated candidate MCTS (`backup="empirical_bellman"`, `--exact-final-step`, bounded UCB $c=1.0$, $\gamma=0.95$)
against the fixed-depth exact reference oracle (`ExactIPOMDPSolver` with explicit `opponent_horizon=d`).
Six panels were executed sequentially, crossing Opponent Replanning Depth $d \in \{1, 2\}$ with Opponent
Prior Physical Belief $b_j \in \{0.085, 0.5, 0.915\}$ across Horizons 1–3, budgets 1,000 and 10,000 traversals,
seeds 300–304 (5 seeds), and own physical beliefs $b_i \in \{0.05, 0.20, 0.50, 0.80, 0.95\}$ (150 cases per panel,
900 cases total). All cases ran under 2 supervised spawned workers with 240-second timeout and 2,048-MiB RSS ceiling.
Zero crashes, timeouts, or resource kills occurred.

Raw cases, manifests, and logs:
- `results/oracle/l2_fixed_d1_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d1_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d1_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fixed_d2_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

### Timing and concurrency instrumentation audit

Parent monotonic elapsed duration (`CLOCK_MONOTONIC`), realtime endpoints, individual worker wall times, and external
GNU `/usr/bin/time` measurements were captured. The monotonic concurrency bound
($\sum t_{\text{worker}} / 2 \le \Delta t_{\text{monotonic}}$) was strictly satisfied across all six panels:

| Panel | External GNU Elapsed (s) | Parent Monotonic Elapsed (s) | Worker Wall Sum (s) | Concurrency Bound ($\le 2 \times \text{Elapsed}$) | Monitored Peak RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $d=1, b_j=0.085$ | 54.07 | 54.57 | 94.40 | **PASS** ($47.20 \le 54.57$) | 74.68 |
| $d=1, b_j=0.500$ | 54.47 | 54.87 | 94.45 | **PASS** ($47.23 \le 54.87$) | 74.79 |
| $d=1, b_j=0.915$ | 54.36 | 54.79 | 94.00 | **PASS** ($47.00 \le 54.79$) | 74.72 |
| $d=2, b_j=0.085$ | 52.14 | 52.07 | 91.38 | **PASS** ($45.69 \le 52.07$) | 74.52 |
| $d=2, b_j=0.500$ | 53.12 | 53.66 | 92.03 | **PASS** ($46.02 \le 53.66$) | 74.51 |
| $d=2, b_j=0.915$ | 53.43 | 53.88 | 93.71 | **PASS** ($46.86 \le 53.88$) | 75.01 |
| **Total / Summary** | **321.59 s** | **323.84 s** | **560.00 s** | **PASS (6/6)** | **75.01 MB** |

External GNU elapsed time matches parent monotonic elapsed time within fractions of a second across all panels,
demonstrating measurement agreement under clean single-panel batch execution.

### Accuracy and policy loss performance

Evaluating first-action loss $\mathcal{L} = \max_{a} Q^*(b, a) - Q^*(b, a_{\text{chosen}})$:

| Panel | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error | Max Max Q Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $d=1, b_j=0.085$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0227 | 0.3979 |
| $d=1, b_j=0.500$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0707 | 0.8183 |
| $d=1, b_j=0.915$ | 150 | 1 (0.7%) | 1 | 0.002251 | 0.337700 | 0.0273 | 0.4615 |
| $d=2, b_j=0.085$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1034 | 1.6567 |
| $d=2, b_j=0.500$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0707 | 0.8183 |
| $d=2, b_j=0.915$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0867 | 1.0469 |
| **Total / Summary** | **900** | **1 (0.11%)** | **1** | **0.000375** | **0.337700** | **0.0636** | **1.6567** |

Across all 900 cases, **899 decisions (99.89%)** achieved exact oracle agreement. Under Opponent Depth $d=2$,
the solver achieved **450 / 450 (100.0%) strictly optimal decisions**.

### Breakdown by horizon and budget

| Horizon | Budget | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Mean Oracle Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **1** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **2** | 1,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0519 | 0.3199 | 15.6713 |
| **2** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0140 | 0.0891 | 15.6713 |
| **3** | 1,000 | 150 | 1 (0.7%) | 1 | 0.002251 | 0.337700 | 0.2556 | 1.6567 | 17.0110 |
| **3** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0599 | 0.2965 | 17.0110 |

The single observed error occurred at $d=1, b_j=0.915, H=3, B=1000, b_i=0.05$, seed 301. Oracle best was
`OL` ($Q^*=2.6475$, gap $0.3377$ over `L` at $2.3098$). Sampling noise at 1,000 traversals estimated $\hat{Q}(L)=2.7713$,
incurring loss $0.33770$. At budget 10,000 (seed 301), $\hat{Q}(L)=2.3421 < 2.6475 = \hat{Q}(OL)$, achieving **zero loss**.

### Opponent depth semantics: depth 1 vs depth 2 comparison

Holding physical beliefs and horizons identical, comparing oracle evaluations under $d=1$ vs $d=2$:
- **Oracle Action Flips**: **8 out of 45 opponent-belief/horizon/own-belief configurations (17.8%) flip their Bayes-optimal action**, with value discrepancies up to $\Delta Q^* = 7.7330$.
- **Oracle Value Differences**: **18 out of 45 configurations (40.0%) have value shifts** exceeding $10^{-4}$.
- **Flips at $b_j=0.085$**: At $H=2$ ($b_i=0.05, 0.95$) and $H=3$ ($b_i=0.05, 0.95$), the optimal action flips from opening doors under $d=1$ (`OL` or `OR`) to listening (`L`) under $d=2$. At these initial private beliefs the d=1 opponent opens and the d=2 opponent listens. Opening resets the state; this changes future private evidence and continuation values. The reward model has no collision penalty, so a collision-avoidance explanation is inappropriate.
- **Flips at $b_j=0.915$**: Symmetrical flips occur at $H=2$ ($b_i=0.05, 0.95$) and $H=3$ ($b_i=0.05, 0.95$).
- **$b_j=0.500$**: An uninformative opponent listens under both $d=1$ and $d=2$; 0 action flips occur.

This demonstrates that the opponent's replanning horizon changes its policy law and the induced planning problem; the physical transition and reward functions themselves are unchanged. MCTS with fixed opponent depth $d$ correctly tracks the reference oracle under that same model.

### Signed Q-error analysis

Door-opening action estimates matched the reference within floating-point error (maximum absolute discrepancy 1.43e-14). Opening is **not terminal**: it resets the physical state and continuation search remains active. For these H1-H3 cases, opening is followed optimally by listening through the remaining short horizon, so Q(open) equals its exact immediate expected reward minus the remaining discounted listen costs. This explains the matching opening estimates without implying that opening has no continuation or will remain exact at larger horizons. All material estimation error in this panel was on L. Mean signed Q error on `L` was $+0.0001$ to $+0.0107$ under $d=1$, and $-0.0179$ to $+0.0107$ under $d=2$.

### Development conclusions

These results support the fixed-depth contract implementation on the tested development grid. This is development evidence under declared exact-L1 opponent semantics. It does not qualify production Level-2 configurations (which feature finite-budget 25-simulation modeled opponents) or authorize global default changes.



### Independent audit of 661fb59

Verified all 900 cases and complete/unique requested coverage, six manifests
against 01f36cf, row settings, policies, losses and Q errors. Recomputed all 90
reference horizon/belief configurations (45 per opponent depth). Every worker
interval matches its duration; an interval sweep confirms no more than two
workers overlap, and each panel's worker sum satisfies its monotonic concurrency
bound. Saved summaries match coverage/status. Audit aggregates are local at
results/oracle/l2_fixed_review_20260925.json.

The sole strict error is confirmed: d1, b_j=.915, H3, 1k, b_i=.05, seed301,
loss .3377000000000012. All 450 cases at budget10k and all 450 d2 cases have zero
strict errors. This remains development evidence, with .020 loss counts purely
descriptive; no L2 validation gate was declared.

The depth comparison denominator is 3 opponent beliefs x 3 horizons x 5 own
beliefs = 45, not 30. Eight optimal-action sets change; eighteen configurations
have Q/value changes above 1e-4. The maximum action-value difference is 7.733.
Opening is not terminal, and the d2 initial opponent at .085/.915 listens.
Earlier causal explanations to the contrary have been corrected above.

Near agreement of external and monotonic durations on these short panels does
not resolve earlier long-run clock discrepancies. The stored parent timestamps
are realtime, not monotonic start/finish; worker intervals are monotonic and
parent monotonic elapsed is recorded separately.

No solver change follows from this audit. Next is a reference for the declared
finite-budget modeled MCTS policy, with explicit private-belief identity, fixed
depth, budget, seed and tie rule; see HANDOFF.md. Defaults remain unchanged.


## Finite-computation L2 reference implementation and smoke

The new reference exhaustively responds to the bank's declared finite MCTS L1
policy, preserving complete immutable private-model identity. It is not an
exact-opponent benchmark. Separate real/modeled configurations and seeds are
recorded; see L2_CONTRACT.md.

A canonical sampling fix addresses a reproduced identity bug: equal beliefs
in different insertion orders previously returned different modeled policies
at the same seed. Old evidence remains tied to its old code; no old validation
gate is silently extended to changed finite trajectories.

Three 18-case smoke panels completed. Modeled budgets 25/100 at depth3 used
H1-H3, own beliefs .05/.5/.95, b_j=.085, seeds300-301, root budget 1000,
empirical Bellman, exact tail, bounded c=1; modeled policies used sampled
backups/tail and normalized c=1. All36 decisions had zero first-action loss;
maximum Q error .881825. A further18-case exact-opponent control matched the
saved earlier smoke's policies and oracle Q values with no changes.
Source hashes and monotonic concurrency checks passed in all panels.
No runtime speed claim is made because the full test suite ran concurrently.

Raw evidence: results/l2-finite-20260925/ including audit.json.
HANDOFF.md defines the next900-case DEVELOPMENT panel. This is not production
depth 20 qualification, global default promotion or a new held-out validation.

## 900-case Level-2 finite-budget modeled-opponent development study (September 25)

Source checkpoint: bf1c225. Evaluated candidate protagonist MCTS (`backup="empirical_bellman"`,
`--exact-final-step`, bounded UCB $c=1.0$, $\gamma=0.95$) against the exhaustive best-response reference
(`FinitePolicyL2Reference`) conditional on the declared `SolverBank` modeled MCTS L1 policy (sampled backups,
sampled final step, normalized UCB $c=1.0$, fixed depth 3).

Six panels were executed sequentially, crossing Modeled Opponent Budget $B_{\text{opp}} \in \{25, 100\}$ with
Opponent Physical Belief $b_j \in \{0.085, 0.500, 0.915\}$ across Horizons 1–3, protagonist budgets 1,000
and 10,000 traversals, seeds 400–404 (5 seeds), and own physical beliefs $b_i \in \{0.05, 0.20, 0.50, 0.80, 0.95\}$
(150 cases per panel, 900 cases total). All cases ran under 2 supervised spawned workers with 240s timeout and
2,048 MiB RSS ceiling. Zero timeouts, crashes, or resource kills occurred.

Raw cases, manifests, and logs:
- `results/oracle/l2_finite_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_finite_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and concurrency bound checks:

| Panel | External GNU Elapsed (s) | Parent Monotonic Elapsed (s) | Worker Wall Sum (s) | Concurrency Bound ($\le 2 \times \text{Elapsed}$) | Monitored Peak RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 50.62 | 53.62 | 93.18 | **PASS** ($46.59 \le 53.62$) | 74.49 |
| $B_{\text{opp}}=25, b_j=0.500$ | 54.21 | 55.61 | 95.38 | **PASS** ($47.69 \le 55.61$) | 74.49 |
| $B_{\text{opp}}=25, b_j=0.915$ | 51.32 | 54.28 | 92.86 | **PASS** ($46.43 \le 54.28$) | 74.72 |
| $B_{\text{opp}}=100, b_j=0.085$ | 51.74 | 54.80 | 94.96 | **PASS** ($47.48 \le 54.80$) | 74.77 |
| $B_{\text{opp}}=100, b_j=0.500$ | 52.09 | 53.41 | 93.08 | **PASS** ($46.54 \le 53.41$) | 74.54 |
| $B_{\text{opp}}=100, b_j=0.915$ | 54.50 | 57.38 | 98.26 | **PASS** ($49.13 \le 57.38$) | 75.03 |
| **Total / Summary** | **314.48 s** | **329.10 s** | **567.72 s** | **PASS (6/6)** | **75.03 MB** |

The monotonic concurrency bound was satisfied across all six panels. External GNU elapsed totals314.48s, while parent monotonic elapsed totals329.10s.
Individual differences are1.32–3.06s; these are distinct measurements, not resolved
clock agreement. Monotonic worker consistency passes, but older timing concerns remain.

### Accuracy and policy loss performance

Evaluating first-action loss $\mathcal{L} = \max_{a} Q^*(b, a) - Q^*(b, a_{\text{chosen}})$:

| Panel | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error | Max Max Q Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0721 | 0.8602 |
| $B_{\text{opp}}=25, b_j=0.500$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0978 | 1.6300 |
| $B_{\text{opp}}=25, b_j=0.915$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0744 | 1.0749 |
| $B_{\text{opp}}=100, b_j=0.085$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0809 | 0.8602 |
| $B_{\text{opp}}=100, b_j=0.500$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0693 | 0.6448 |
| $B_{\text{opp}}=100, b_j=0.915$ | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0881 | 1.0749 |
| **Total / Summary** | **900** | **0 (0.00%)** | **0** | **0.000000** | **0.000000** | **0.0804** | **1.6300** |

Across all 900 cases, **900 decisions (100.0%)** achieved exact oracle agreement.

### Breakdown by horizon and protagonist budget

| Horizon | Root Budget | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Mean Oracle Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **1** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **2** | 1,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0790 | 0.4024 | 15.7799 |
| **2** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0243 | 0.1192 | 15.7799 |
| **3** | 1,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.2869 | 1.6300 | 17.3169 |
| **3** | 10,000 | 150 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0924 | 0.7663 | 17.3169 |

### Modeled opponent computation sensitivity ($B_{\text{opp}}=25$ vs $B_{\text{opp}}=100$)

Evaluating across the 225 matched configurations $(b_j, H, b_i, \text{seed})$:
- **Oracle Action Flips**: **9 out of 225 configuration pairs (4.00%) flip their Bayes-optimal action set** between $B_{\text{opp}}=25$ and $B_{\text{opp}}=100$ (0 at $H=1$, 4 at $H=2$, 5 at $H=3$).
- **Oracle Value Differences**: **21 out of 225 configuration pairs (9.33%) exhibit value shifts** $> 10^{-4}$ (max action Q difference $7.7330$).
- At 25 simulations, finite sampling noise causes the opponent to open a door in select boundary conditions where 100 simulations resolve to listening. The reference oracle tracks this policy change exactly, and protagonist MCTS selects the optimal response in 100% of cases under both budgets.

### Signed Q-error analysis

Nonterminal door-opening actions (`OL`, `OR`) had negligible Q error ($\pm 0.0000$) across all cases for $b_j \in \{0.085, 0.915\}$ and for $B_{\text{opp}}=100$ at $b_j=0.5$. Mean signed Q error on action `L` was $+0.0146$ to $+0.0284$ ($B_{\text{opp}}=25$) and $-0.0048$ to $+0.0361$ ($B_{\text{opp}}=100$).

### Development conclusions

These results show correct first-action choices on the tested finite-computation depth3 development grid; they do not establish exact values or accuracy at every belief. This is development evidence; production depth 20, mixed levels, and empirical priors remain pending.



### Independent review after the outage (bf1c225 / 123701b)

WSL access was restored and the clean checkout verified at123701b. The saved
pre-outage full-suite log confirms225 tests passed in310.22s. bf1c225 contains
the finite-policy reference, canonical sampling fix, tests and contract;123701b
contains the subsequent experiment documentation.

Independently verified all 900 raw cases: requested coverage/uniqueness, status,
source hashes againstbf1c225, complete modeled/root settings, seed identity,
policy distributions, first-action losses and max Q errors. Recomputed all450
distinct oracle configurations with the declared finite opponent and compared
both root-budget copies. Confirmed zero strict action errors and zero loss,
nine optimal-action-set changes out of225 modeled-budget pairs, and21 Q/value
changes above 1e-4 (maximum Q change7.733). The audit uses the existing exhaustive
reference independently of the report; it is not a separate proof of that code.

Worker interval durations and maximum concurrency of two passed for all panels.
Monotonic elapsed and GNU elapsed remain distinct:329.10s versus314.48s summed.
The largest per-panel difference is3.06s. Do not call the timing discrepancy solved.

Opening is nonterminal. In the25-simulation, b_j=.5 panel its maximum absolute
Q error is .766343, while selected actions still match. This directly cautions
against inferring accurate action values from zero first-action loss.
The previously drafted54-case smoke had zero policy errors; the separate9-case
reference cross-check had max Q difference8.88e-16. Those are different checks.

A six-case depth 20 resource pilot used modeled budget 25, b_j=.5, own beliefs
.05/.5/.95, H1-H2, root budget 1000, seed400, and the same root/ modeled estimators.
All cases completed with zero loss; max Q error .402376, max case time .486818s,
peak monitored RSS74.094MiB. Parent monotonic elapsed1.630009s and worker
sum2.792059s satisfy the two-worker bound. This is a small feasibility check,
not long-horizon production qualification.

Audit: results/oracle/l2_finite_review_20260925.json.
Pilot: results/oracle/l2_finite_depth20_pilot_20260925/ and sibling log.
No solver changes; HANDOFF.md defines a360-case depth 20 development extension.

## 360-case Level-2 depth-20 development extension (September 25)

Source checkpoint: 06cdcd3. Evaluated candidate protagonist MCTS (`backup="empirical_bellman"`,
`--exact-final-step`, bounded UCB $c=1.0$, $\gamma=0.95$) against the exhaustive best-response reference
(`FinitePolicyL2Reference`) conditional on the declared `SolverBank` modeled MCTS L1 policy at depth 20
(sampled backups, sampled final step, normalized UCB $c=1.0$, fixed depth 20).

Six panels were executed sequentially, crossing Modeled Opponent Budget $B_{\text{opp}} \in \{25, 100\}$ with
Opponent Physical Belief $b_j \in \{0.085, 0.500, 0.915\}$ across Horizons 1–3, protagonist budgets 1,000
and 10,000 traversals, seeds 400–401 (2 seeds), and own physical beliefs $b_i \in \{0.05, 0.20, 0.50, 0.80, 0.95\}$
(60 cases per panel, 360 cases total). All cases ran under 2 supervised spawned workers with 240s timeout and
2,048 MiB RSS ceiling. Zero timeouts, crashes, or resource kills occurred.

Raw cases, manifests, and logs:
- `results/oracle/l2_depth20_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_depth20_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and concurrency bound checks:

| Panel | External GNU Elapsed (s) | Parent Monotonic Elapsed (s) | Worker Wall Sum (s) | Concurrency Bound ($\le 2 \times \text{Elapsed}$) | Monitored Peak RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 21.58 | 22.27 | 39.02 | **PASS** ($19.51 \le 22.27$) | 74.59 |
| $B_{\text{opp}}=25, b_j=0.500$ | 19.49 | 20.65 | 36.47 | **PASS** ($18.24 \le 20.65$) | 74.79 |
| $B_{\text{opp}}=25, b_j=0.915$ | 22.80 | 22.28 | 38.45 | **PASS** ($19.23 \le 22.28$) | 74.77 |
| $B_{\text{opp}}=100, b_j=0.085$ | 25.22 | 26.46 | 46.67 | **PASS** ($23.34 \le 26.46$) | 74.55 |
| $B_{\text{opp}}=100, b_j=0.500$ | 22.27 | 23.51 | 41.79 | **PASS** ($20.90 \le 23.51$) | 74.65 |
| $B_{\text{opp}}=100, b_j=0.915$ | 24.31 | 25.63 | 45.37 | **PASS** ($22.69 \le 25.63$) | 75.09 |
| **Total / Summary** | **135.67 s** | **140.80 s** | **247.77 s** | **PASS (6/6)** | **75.09 MB** |

The monotonic concurrency bound was satisfied across all six panels. External GNU elapsed time agreed with
parent monotonic elapsed time on these short panels within 1.3 seconds (total elapsed ~2.26–2.35 min).

### Accuracy and policy loss performance

Evaluating first-action loss $\mathcal{L} = \max_{a} Q^*(b, a) - Q^*(b, a_{\text{chosen}})$:

| Panel | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error | Max Max Q Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1166 | 1.5352 |
| $B_{\text{opp}}=25, b_j=0.500$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0685 | 0.5145 |
| $B_{\text{opp}}=25, b_j=0.915$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0843 | 0.8467 |
| $B_{\text{opp}}=100, b_j=0.085$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0789 | 0.7453 |
| $B_{\text{opp}}=100, b_j=0.500$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0685 | 0.5145 |
| $B_{\text{opp}}=100, b_j=0.915$ | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.1076 | 1.7135 |
| **Total / Summary** | **360** | **0 (0.00%)** | **0** | **0.000000** | **0.000000** | **0.0874** | **1.7135** |

Across all 360 cases, **360 decisions (100.0%)** achieved exact oracle agreement.

### Breakdown by horizon and protagonist budget

| Horizon | Root Budget | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Mean Oracle Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **1** | 10,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0000 | 0.0000 | 15.4000 |
| **2** | 1,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0900 | 0.4024 | 15.8070 |
| **2** | 10,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0319 | 0.0632 | 15.8070 |
| **3** | 1,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.3290 | 1.7135 | 17.4422 |
| **3** | 10,000 | 60 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 0.0736 | 0.3935 | 17.4422 |

### Opponent depth comparison: depth 3 vs depth 20 (matched seeds 400–401)

Comparing the 360 matched cases between opponent depth 3 and depth 20 on seeds 400 and 401:
- **Oracle Action Flips**: **18 out of 360 pairs (5.00%) flip their Bayes-optimal action set** (all 18 occurred under $B_{\text{opp}}=25$).
- **Oracle Value Differences**: **62 out of 360 pairs (17.22%) exhibit value shifts** $> 10^{-4}$ (max action Q difference $7.7330$).
- Under $B_{\text{opp}}=100$: **0 action flips** occurred between $d=3$ and $d=20$, showing only that the protagonist's best action is unchanged on these pairs. Five distinct reference problems still change in Q/value at this budget (maximum Q change 2.264711); opponent-policy stability does not follow. Protagonist MCTS achieved 100% agreement with the reference under depth 20.

### Signed Q-error analysis

Nonterminal door-opening actions (`OL`, `OR`) had zero Q error ($\pm 0.0000$) across all cases. Mean signed Q error on action `L` was $-0.0394$ to $+0.0350$ ($B_{\text{opp}}=25$) and $-0.0055$ to $+0.0163$ ($B_{\text{opp}}=100$).

### Development conclusions

These results show matching first actions on the tested H1-H3 grid with a depth-20 modeled opponent; they do not establish accuracy at untested protagonist horizons. This is development evidence; production protagonist depth 20, mixed levels, and empirical priors remain pending.



### Independent review of ee3d5b4

Verified all 360 case files: complete/unique requested coverage, source hashes
against 06cdcd3, real/modeled configuration, seeds, statuses, valid policies,
losses and Q errors. Recomputed the 180 distinct oracle problems and checked
both root-budget copies. All first-action losses are zero; maximum Q error is
1.713519. Worker intervals match durations, no more than two workers overlap,
and each panel satisfies the parent monotonic concurrency bound.

The depth comparison includes two copies of each reference problem because
protagonist search budget does not enter the oracle. At the distinct-problem
level, 9/180 action sets change and 31/180 Q/value sets differ above 1e-4:
- modeled budget 25: 9 action changes, 26 Q/value changes out of 90;
- modeled budget 100: 0 action changes, 5 Q/value changes out of 90, with
  maximum Q change 2.264711.

Thus 18/360 and 62/360 are correct row counts, but not 360 independent oracle
comparisons. Unchanged best actions cannot establish unchanged opponent policy
or values. Depth changes also affect the hashed finite-search settings, so
equal seed indices do not isolate a common-random-number depth effect.

The external/parent-monotonic totals are 135.67/140.804341 seconds. Individual
differences reach 1.323687 seconds and change sign in one panel. Keep both
measurements; short-run concurrency consistency does not resolve older clock
discrepancies. No new statistical loss gate was declared for this development run.

A two-case protagonist H4/H5 feasibility pilot with opponent depth 20/budget 25,
b_j=.5, own belief .5, seed400 and root1000 completed with zero first-action
loss. H5 maximum Q error was 3.355773. Worker sum1.364052s fits two times parent
elapsed .800306s. A uniform-belief pilot does not qualify boundary decisions.

Audit: results/oracle/l2_depth20_review_20260925.json.
Pilot: results/oracle/l2_h45_resource_pilot_20260925/ and sibling log.
HANDOFF.md specifies the next 360-case H4/H5 development panel. No solver code
or default changed; the last implementation test result remains 225 passes.

## 360-case Level-2 H4/H5 development study (September 25)

Source checkpoint: 1359053. Evaluated candidate protagonist MCTS (`backup="empirical_bellman"`,
`--exact-final-step`, bounded UCB $c=1.0$, $\gamma=0.95$) against the exhaustive best-response reference
(`FinitePolicyL2Reference`) conditional on the declared `SolverBank` modeled MCTS L1 policy at fixed depth 20
(sampled backups, sampled final step, normalized UCB $c=1.0$) across longer protagonist horizons ($H \in \{4, 5\}$).

Six panels crossed Modeled Opponent Budgets $B_{\text{opp}} \in \{25, 100\}$ with Opponent Beliefs
$b_j \in \{0.085, 0.500, 0.915\}$ across Horizons 4–5, protagonist budgets 1,000, 10,000, and 50,000 traversals,
seeds 400–401 (2 seeds), and own physical beliefs $b_i \in \{0.050, 0.075, 0.500, 0.925, 0.950\}$ (60 cases per panel,
360 total). All cases ran under 2 supervised spawned workers with 240s timeout and 2,048 MiB RSS ceiling.
Zero timeouts, crashes, or resource kills occurred.

Raw cases, manifests, and logs:
- `results/oracle/l2_h45_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and concurrency bound checks:

| Panel | External GNU Elapsed (s) | Parent Monotonic Elapsed (s) | Worker Wall Sum (s) | Concurrency Bound ($\le 2 \times \text{Elapsed}$) | Monitored Peak RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 69.23 | 74.17 | 140.48 | **PASS** ($70.24 \le 74.17$) | 79.66 |
| $B_{\text{opp}}=25, b_j=0.500$ | 67.03 | 70.01 | 134.08 | **PASS** ($67.04 \le 70.01$) | 78.53 |
| $B_{\text{opp}}=25, b_j=0.915$ | 69.93 | 73.28 | 139.92 | **PASS** ($69.96 \le 73.28$) | 79.10 |
| $B_{\text{opp}}=100, b_j=0.085$ | 86.63 | 91.45 | 176.81 | **PASS** ($88.41 \le 91.45$) | 78.25 |
| $B_{\text{opp}}=100, b_j=0.500$ | 75.97 | 81.00 | 155.83 | **PASS** ($77.92 \le 81.00$) | 78.21 |
| $B_{\text{opp}}=100, b_j=0.915$ | 85.30 | 88.29 | 169.32 | **PASS** ($84.66 \le 88.29$) | 79.14 |
| **Total / Summary** | **454.09 s** | **478.20 s** | **916.44 s** | **PASS (6/6)** | **79.66 MB** |

The monotonic concurrency bound was satisfied across all six panels. External GNU elapsed time and parent monotonic elapsed time differ by 2.99–5.03s per panel (total elapsed ~7.57–7.97 min). Both measurements are preserved.

### Accuracy and policy loss performance

Evaluating first-action loss $\mathcal{L} = \max_{a} Q^*(b, a) - Q^*(b, a_{\text{chosen}})$:

| Panel | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error | Max Max Q Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 60 | 4 (6.67%) | 4 | 0.032912 | 0.954069 | 2.1765 | 9.5404 |
| $B_{\text{opp}}=25, b_j=0.500$ | 60 | 1 (1.67%) | 1 | 0.010268 | 0.616052 | 1.8571 | 14.1285 |
| $B_{\text{opp}}=25, b_j=0.915$ | 60 | 1 (1.67%) | 1 | 0.001048 | 0.062898 | 1.7438 | 9.0502 |
| $B_{\text{opp}}=100, b_j=0.085$ | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.7256 | 9.8171 |
| $B_{\text{opp}}=100, b_j=0.500$ | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.7341 | 5.4413 |
| $B_{\text{opp}}=100, b_j=0.915$ | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.6739 | 8.1720 |
| **Total / Summary** | **360** | **6 (1.67%)** | **6 (1.67%)** | **0.007371** | **0.954069** | **1.8185** | **14.1285** |

Across all 360 cases, **354 decisions (98.33%)** achieved exact oracle agreement. All 6 decision errors occurred under $B_{\text{opp}}=25$. Under $B_{\text{opp}}=100$, 180/180 decisions (100.0%) were strictly optimal.

### Breakdown by horizon and protagonist budget

| Horizon | Root Budget | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 1,000 | 60 | 1 (1.67%) | 1 | 0.002808 | 0.168457 | 1.7266 | 6.8501 |
| **4** | 10,000 | 60 | 2 (3.33%) | 2 | 0.003856 | 0.168457 | 0.8924 | 4.6836 |
| **4** | 50,000 | 60 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | **0.4920** | **2.2622** |
| **5** | 1,000 | 60 | 3 (5.00%) | 3 | 0.037564 | 0.954069 | 3.9125 | 14.1285 |
| **5** | 10,000 | 60 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | **2.2123** | **4.4549** |
| **5** | 50,000 | 60 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | **1.6753** | **4.2010** |

At root budget 50,000, **120/120 decisions (100.0%)** achieved exact oracle agreement
on this development grid. This does not establish convergence: H4 errors increase
from one at 1k to two at 10k before reaching zero at 50k. Budget is part of the
deterministic seed identity, so these runs are not nested prefixes of one search.

### Breakdown by own physical belief

| Belief ($b_i$) | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.050** | 72 | 3 (4.17%) | 3 | 0.013236 | 0.616052 | 2.0848 |
| **0.075** | 72 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | 2.0880 |
| **0.500** | 72 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | 1.0232 |
| **0.925** | 72 | **0 (0.0%)** | **0** | **0.000000** | **0.000000** | 1.8818 |
| **0.950** | 72 | 3 (4.17%) | 3 | 0.023621 | 0.954069 | 2.0148 |

Near-boundary transition points $b_i \in \{0.075, 0.925\}$ achieved 100.0% accuracy (144/144).

### Opening vs. listening breakdown

In `TigerModel`, opening is nonterminal; continuation paths extend 3–4 steps:
- **Opening Cases (oracle prefers `OL` or `OR`)**: 72 cases, 2 errors (2.78%; both at $B_{\text{opp}}=25, b_j=0.085, H=4, b_i=0.05$, resolved at 50k budget).
- **Listening Cases (oracle prefers `L`)**: 288 cases, 4 errors (1.39%; premature opening at 1k/10k; this panel does not isolate its statistical cause, all resolved at 50k budget).

### Signed Q-error analysis

- Action `L`: Mean signed error contracts from $+0.5653$ (1k) $\to +0.0464$ (10k) $\to -0.0082$ (50k); mean absolute error contracts from $0.7735 \to 0.0577$.
- Actions `OL` and `OR`: Mean signed errors under nonterminal continuation are respectively $+1.44$ and $+1.58$ at 1k, and $+0.39$ and $+0.53$ at 50k. These finite-panel means do not establish estimator bias.

### Opponent budget sensitivity: $B_{\text{opp}}=25$ vs. $B_{\text{opp}}=100$

Across 60 distinct $(H, b_j, \text{seed}, b_i)$ configurations:
- **6 action flips (10.00%)**: $H=4, b_j=0.085, s=400, b_i=0.95$ ($25 \to \text{L}$ vs $100 \to \text{OR}$); $H=4, b_j=0.915, s=400, b_i=0.05$ ($25 \to \text{L}$ vs $100 \to \text{OL}$); $H=4, b_j=0.915, s=400, b_i=0.95$ ($25 \to \text{L}$ vs $100 \to \text{OR}$); $H=5, b_j=0.085, s=400, b_i=0.05$ ($25 \to \text{OL}$ vs $100 \to \text{L}$); $H=5, b_j=0.915, s=400, b_i=0.05$ ($25 \to \text{OL}$ vs $100 \to \text{L}$); $H=5, b_j=0.915, s=400, b_i=0.95$ ($25 \to \text{OR}$ vs $100 \to \text{L}$).
- **42 Q-value shifts $> 10^{-4}$ (70.00%)**, with max Q shift 2.723554.

### Development conclusions

The candidate matches the reference on all tested H4/H5 cases at 50k and on all tested cases under $B_{\text{opp}}=100$. These results select a candidate resource budget for further testing; they do not certify every belief, seed, opponent configuration or horizon. Production qualification and empirical nested priors remain pending.


### Independent Codex audit of c68754b

Verified all 360 case identities, source hashes against 1359053, root and modeled
configurations, policy probabilities, statuses and recorded losses. Recomputed
all 120 distinct reference problems and checked all three root-budget copies.
Confirmed 6/360 strict errors, all greater than .020; that threshold remains
descriptive for this development study. At root 50k, maximum Q error is 4.200977
and mean maximum Q error is 1.083629 despite zero first-action loss. Accurate
action ranking is a different requirement from accurate values.

The 60 distinct opponent-budget comparisons reproduce six action-set changes,
42 Q changes above 1e-4, and maximum Q change 2.723554. Increasing modeled budget
changes the opponent's policy law and the decision problem; the zero-error
budget 100 group does not imply a better protagonist algorithm.

Worker endpoints reproduce durations and at most two simultaneous workers.
All monotonic concurrency bounds pass. External elapsed 454.09s remains smaller
than parent monotonic 478.198078s; clock/provenance discrepancies are unresolved.
Worker durations include both planner and reference, not planner-only latency.

Audit artifact: results/oracle/l2_h45_review_20260925.json. Raw evidence remains
local and Git-ignored. Source/configuration scope of older gates is unchanged.


### H6/H8 resource pilot

At source c68754b (solver files unchanged from 1359053), six cases crossed H6/H8
with own beliefs .05/.5/.95, root 50k, opponent depth 20/budget 25 and b_j=.5,
seed 500. All completed with zero first-action loss. Maximum Q errors were
7.042874 at H6 and 6.630328 at H8. Worker durations were 7.09–8.36s and 11.94–15.29s;
peak monitored RSS 85.78/125.63 MiB. Parent monotonic elapsed 15.802889/28.178427s
and worker sums 22.608093/39.254578s satisfy two-worker concurrency bounds.
These combined planner/reference costs show pilot feasibility, not broad
accuracy or planner-only efficiency. The next 120-case development protocol
is fixed in HANDOFF.md. Pilot raw data: results/oracle/l2_h{6,8}_resource_pilot_20260925/.

## 120-case Level-2 H6/H8 development study (September 25)

Source checkpoint: e63db30. Evaluated candidate protagonist MCTS (`backup="empirical_bellman"`,
`--exact-final-step`, bounded UCB $c=1.0$, $\gamma=0.95$) against the exhaustive best-response reference
(`FinitePolicyL2Reference`) conditional on the declared `SolverBank` modeled MCTS L1 policy at fixed depth 20
(sampled backups, sampled final step, normalized UCB $c=1.0$, $\epsilon=10^{-6}$) across deep protagonist horizons ($H \in \{6, 8\}$).

Six panels crossed Modeled Opponent Budgets $B_{\text{opp}} \in \{25, 100\}$ with Opponent Beliefs
$b_j \in \{0.085, 0.500, 0.915\}$ across Horizons 6 and 8, protagonist root budget fixed at 50,000 traversals,
seeds 500–501 (2 seeds), and own physical beliefs $b_i \in \{0.050, 0.075, 0.500, 0.925, 0.950\}$ (20 cases per panel,
120 total). All cases ran under 2 supervised spawned workers with 240s timeout and 2,048 MiB RSS ceiling.
Zero timeouts, crashes, or resource kills occurred.

Raw cases, manifests, and logs:
- `results/oracle/l2_h68_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

### Timing and concurrency instrumentation audit

Parent monotonic elapsed, worker wall sums, external GNU `/usr/bin/time`, and concurrency bound checks:

| Panel | External GNU Elapsed (s) | Parent Monotonic Elapsed (s) | Worker Wall Sum (s) | Concurrency Bound ($\le 2 \times \text{Elapsed}$) | Monitored Peak RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 89.97 | 95.08 | 184.03 | **PASS** ($92.02 \le 95.08$) | 132.08 |
| $B_{\text{opp}}=25, b_j=0.500$ | 83.42 | 89.08 | 172.46 | **PASS** ($86.23 \le 89.08$) | 125.92 |
| $B_{\text{opp}}=25, b_j=0.915$ | 87.33 | 93.02 | 179.43 | **PASS** ($89.72 \le 93.02$) | 132.38 |
| $B_{\text{opp}}=100, b_j=0.085$ | 98.75 | 104.32 | 202.57 | **PASS** ($101.29 \le 104.32$) | 132.00 |
| $B_{\text{opp}}=100, b_j=0.500$ | 94.29 | 99.97 | 193.67 | **PASS** ($96.84 \le 99.97$) | 129.29 |
| $B_{\text{opp}}=100, b_j=0.915$ | 101.73 | 107.39 | 208.76 | **PASS** ($104.38 \le 107.39$) | 126.76 |
| **Total / Summary** | **555.49 s** | **588.87 s** | **1140.93 s** | **PASS (6/6)** | **132.38 MB** |

The monotonic concurrency bound was satisfied across all six panels. External GNU elapsed time and parent monotonic elapsed time differ by 5.11–5.66s per panel (total elapsed ~9.26–9.81 min). Both measurements are preserved.

### Accuracy and policy loss performance

Evaluating first-action loss $\mathcal{L} = \max_{a} Q^*(b, a) - Q^*(b, a_{\text{chosen}})$:

| Panel | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Loss > 0.02 (diagnostic) | Mean Policy Loss | Max Policy Loss | Mean Max Q Error | Max Max Q Error | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $B_{\text{opp}}=25, b_j=0.085$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.4836 | 7.6314 | 100.0% |
| $B_{\text{opp}}=25, b_j=0.500$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 4.3732 | 7.0429 | 100.0% |
| $B_{\text{opp}}=25, b_j=0.915$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.5872 | 7.1004 | 100.0% |
| $B_{\text{opp}}=100, b_j=0.085$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.3236 | 8.5238 | 100.0% |
| $B_{\text{opp}}=100, b_j=0.500$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.7174 | 6.2539 | 100.0% |
| $B_{\text{opp}}=100, b_j=0.915$ | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 2.9572 | 6.2639 | 100.0% |
| **Total / Summary** | **120** | **0 (0.00%)** | **0 (0.00%)** | **0.000000** | **0.000000** | **3.5737** | **8.5238** | **100.0%** |

Across all 120 cases, **120 decisions (100.0%)** achieved exact oracle agreement.

### Breakdown by horizon and physical belief

| Horizon | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6** | 60 | 0 (0.0%) | 0.000000 | 0.000000 | 2.8410 | 8.5238 | 100.0% |
| **8** | 60 | 0 (0.0%) | 0.000000 | 0.000000 | 4.3064 | 7.6314 | 100.0% |

| Belief ($b_i$) | Cases | Errors ($\mathcal{L} > 10^{-8}$) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.050** | 24 | 0 (0.0%) | 0.000000 | 0.000000 | 3.3083 | 7.0429 | 100.0% |
| **0.075** | 24 | 0 (0.0%) | 0.000000 | 0.000000 | 3.8035 | 6.7721 | 100.0% |
| **0.500** | 24 | 0 (0.0%) | 0.000000 | 0.000000 | 2.7476 | 5.0341 | 100.0% |
| **0.925** | 24 | 0 (0.0%) | 0.000000 | 0.000000 | 3.8893 | 6.2539 | 100.0% |
| **0.950** | 24 | 0 (0.0%) | 0.000000 | 0.000000 | 4.1198 | 8.5238 | 100.0% |

### Action classification: uniform optimality of listening

In all 120 cases at $H \in \{6, 8\}$, the Bayes-optimal oracle action is **`L` (Listen)**:
- **Opening Cases (oracle prefers `OL` or `OR`)**: 0 / 120.
- **Listening Cases (oracle prefers `L`)**: 120 / 120 (100.0% accuracy).

At long horizons, information gathering strictly dominates door opening across all tested beliefs. Protagonist MCTS accurately selected `L` in 100% of cases.

### Signed Q-error analysis

- Action `L`: Mean signed error is $+0.298149$, mean absolute error is $0.332974$ (range $[-0.304545, +1.071613]$).
- Nonterminal door opening (`OL`, `OR`): Continuation search through 5–7 post-opening steps shows positive bias ($+2.156583$ on `OL`, $+2.031731$ on `OR`; max error $8.523833$). This continuation bias does not impair decision accuracy because $Q^*(L)$ exceeds opening values by a wide margin.

### Opponent budget sensitivity: $B_{\text{opp}}=25$ vs. $B_{\text{opp}}=100$

Across the 60 distinct $(H, b_j, \text{seed}, b_i)$ problems:
- **0 action flips (0.00%)**: Optimal policy uniformly listens under both modeled budgets.
- **60 out of 60 Q vectors shift $> 10^{-4}$ (100.0%)**, with max Q shift $1.616884$.

### Development conclusions

These results confirm that candidate protagonist MCTS at 50k budget accurately tracks the optimal listening policy across deep horizons ($H=6$ and $H=8$). This is development evidence; production qualification, empirical nested priors, and multi-step episode evaluations remain pending.
