# Current review handoff

## Decision and independently checked evidence

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs frozen experiments and retains raw evidence.

Do not promote exact_history_rewards to a Level-2 production default yet.
The candidate improves development performance substantially, but still has
three first-action losses above .020 and introduces some new errors. It is
the selected candidate for a separate fresh gate, not a qualified default.
Changing the global MCTSConfig default would also change modeled opponents;
the current evidence concerns only the protagonist estimator.

Codex audited a7a8b75 against source 4c17278: all 3,360 case identities, source
hashes, manifests, loss/Q-error arithmetic and worker intervals. All 1,680
paired modeled configurations and reference Q vectors agree exactly.
Independently recomputed the seven candidate error references within 1e-8;
the other reference values were checked for paired equality and arithmetic,
not independently recomputed in this audit.

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Cases | 1,680 | 1,680 |
| Loss > .020 + 1e-8 | 16 | 3 |
| Loss > 1e-8 | 19 | 7 |
| Mean loss | .001583807 | .000106598 |
| Maximum loss | .508595 | .094981 |
| Mean maximum Q error | 1.439959 | .389566 |
| Maximum Q error | 9.957050 | 5.799061 |

Paired transitions: 16 strict errors fixed, 4 new strict errors; 15 tolerance
violations fixed, 2 new violations. The report's 15-of-19 strict-fix count was
incorrect. This is improvement with regressions, not uniform dominance.

The three remaining violations are all H8:
- Opponent 25/b_j=.5, own .0325, seed 6002: loss .094981.
- Opponent 100/b_j=.085, own .9675, seed 6001: loss .022339.
- Opponent 100/b_j=.915, own .0325, seed 6002: loss .024690.

Positive mean signed opening errors shrink but are not eliminated. Finite-grid
means do not establish estimator bias or a universal causal mechanism.
Root seeds differ between arms; matching bank seed indices does not imply
common random numbers.

Raw external totals: 3970.96/4084.09s; parent monotonic: 4221.612305/4341.578523s;
worker sums: 8161.384092/8392.783595s (control/candidate). All internal two-worker
bounds pass. The observed worker-sum increase is 2.84%, but includes reference
work and clock/provenance limitations; it is not isolated planner overhead.
Peak per-case RSS varies: 73.58–139.67 MiB control, 73.59–138.98 MiB candidate.
The report's invariant 139.67 MiB claim is incorrect.

Artifacts: results/oracle/l2_rewards_{control,candidate}_n{25,100}_b{0.085,0.5,0.915}_20260925/.
Audit: results/oracle/l2_rewards_review_20260926.json.
Raw evidence is local and Git-ignored; documentation commits are not raw archives.

## Frozen fresh candidate validation

Freeze this design before any new solve. Run from the clean commit containing
this protocol; record the full commit ID and manifests. Solver source remains
4c17278. No fresh validation solve has been performed by Codex.

Six panels cross modeled budgets 25/100 with initial opponent beliefs .085/.5/.915.
Each panel contains 560 cases; total 3,360:

- Protagonist horizons 1,2,3,4,5,6,8; 50,000 simulations.
- Physical beliefs .002/.0175/.0275/.0375/.0875/
  .9125/.9625/.9725/.9825/.998.
- Seeds 7000–7007 inclusive.
- Root empirical_bellman, exact_final_step=true, exact_history_rewards=true,
  bounded UCB c=1, gamma=.95, node_capacity=200. The root's opponent-budget
  configuration field remains 25, as in the tested runner.
- Modeled L1 fixed depth 20, sampled backup and tail, exact_history_rewards=false,
  normalized UCB c=1, epsilon1e-6, node_capacity=500. Both modeled budget fields
  equal the declared 25 or 100. Preserve full configuration and exact model identity.
- Tiger dynamics/sensors and finite-policy reference as implemented at 4c17278:
  85% growl, perfect creak, nonterminal uniform reset after opening; pure L2
  point prior on L1 modeling uniform L0. Exhaustive reference branch limit 100,000,
  no rounding or approximate fallback, uniform exact-maximum finite-policy ties.
- Two workers, 240 seconds per case, 2,048 MiB per worker.

All ten physical beliefs and eight seeds are absent from inspected local
manifests and 22,303 local case rows. No new-grid reference values or planner
outputs were inspected. Opponent conditions and estimator were selected using
development data; this is a held-out finite-grid check, not a random population
sample or proof over all beliefs. Residual development failures mean this gate
can fail; neither 50k nor this design is a promise of success.

### Acceptance fixed before launch

Every requested case must complete validly and satisfy
first_action_loss <= .020 + 1e-8.
Loss is max_a Q*(b,a) - sum_a pi(a|b) Q*(b,a), with optimal continuation under
the declared finite-computation opponent. It is not episode regret.

Any completed violation fails the accuracy gate. Missing, timed-out, killed or
invalid cases prevent a pass and are reported separately. No selective retries,
case replacement, budget increases or tolerance changes may repair this gate.
Stop for execution failures and preserve all evidence; do not hide unfinished
panels. A changed design is a new study with a new frozen protocol.

Report strict errors separately (loss>1e-8), action gaps/sets, opening/listening
coverage per horizon and opponent, signed/absolute Q errors, every failure,
RSS and all recorded clocks. No action-based case screening. A successful
gate would support only this source/configuration/grid, not full L2 production,
nested priors, larger horizons, L4/all39 or universal reliability. Historical
ac3df1c validation remains failed regardless of this result.

## Antigravity completed run: 3,360-case fresh candidate validation

Antigravity completed the six sequential fresh validation panels on clean commit
f4d2aef (RUN_ID 20260926, supervisor script scratch/run_l2_fresh_candidate_validation.sh).
All 3,360 requested cases completed validly with exit code 0.

### Primary gate outcome: FAILED

Under the frozen zero-tolerance criterion (first_action_loss <= .020 + 1e-8), the
primary validation gate FAILED. Across the 3,360 cases, 9 cases exceeded .020
(maximum loss .205046). Compliance rate is 99.73% (3,351/3,360 passed).

### Summary metrics and comparison to prior fresh gate

| Metric | Prior Fresh Gate (ac3df1c, sampled rewards) | Candidate Fresh Gate (f4d2aef, --exact-history-rewards) | Relative change |
| --- | ---: | ---: | ---: |
| Cases evaluated | 1,680 | 3,360 | 2x scale |
| Strictly optimal (<= 1e-8) | 1,654 / 1,680 (98.45%) | **3,345 / 3,360 (99.55%)** | +1.10 percentage points |
| Strict errors (> 1e-8) | 26 / 1,680 (1.55%) | **15 / 3,360 (0.45%)** | 71% reduction |
| Tolerance violations (> .020) | 23 / 1,680 (1.37%) | **9 / 3,360 (0.27%)** | **5.1x reduction in violation rate** |
| Gate compliance rate (<= .020) | 98.63% | **99.73%** | +1.10 percentage points |
| Mean first-action policy loss | .002198 | **.000307** | **7.2x reduction** |
| Maximum policy loss | .741558 | **.205046** | **3.6x contraction** |
| Mean maximum Q error | 1.4400 | **.4124** | **3.5x reduction** |
| Peak absolute Q error | 8.5238 | **5.8211** | 31.7% contraction |

### Panel-by-panel breakdown

- **n25, b=0.085**: 559/560 optimal (99.82%), 1 violation (.205046), mean loss .000366, mean maxQ .3888.
- **n25, b=0.500**: 559/560 optimal (99.82%), 1 violation (.058131), mean loss .000104, mean maxQ .4060.
- **n25, b=0.915**: 557/560 optimal (99.46%), 3 violations (.163827, .147591, .065675), mean loss .000673, mean maxQ .4284.
- **n100, b=0.085**: 556/560 optimal (99.29%), 3 violations (.145390, .056220, .021715), mean loss .000410, mean maxQ .3816.
- **n100, b=0.500**: 556/560 optimal (99.29%), **0 violations (max loss .019711)**, mean loss .000077, mean maxQ .4603.
- **n100, b=0.915**: 558/560 optimal (99.64%), 1 violation (.113124), mean loss .000214, mean maxQ .4095.

### Failure anatomy and signed Q errors

1. **Horizon strata**:
   - H1–H4: 1,920 / 1,920 strictly optimal (100.0%), zero errors, zero violations.
   - H5: 479/480 optimal (99.79%), 1 violation (case-353, loss .058131).
   - H6: 474/480 optimal (98.75%), 1 violation (case-403, loss .021715), 5 near-ties (.0051–.0090).
   - H8: 472/480 optimal (98.33%), 7 violations (.0562–.2050), 1 near-tie (.0197).
2. **Belief localization**:
   - All 15 errors (9 violations, 6 near-ties) occurred strictly at the four frontier beliefs:
     b_i=.0275 (3 errors, 1 violation), b_i=.0375 (4 errors, 3 violations),
     b_i=.9625 (4 errors, 3 violations), b_i=.9725 (4 errors, 2 violations).
   - All six non-frontier beliefs (.0020, .0175, .0875, .9125, .9825, .9980) had zero errors
     across 100% of cases (2,016 / 2,016 strictly optimal).
3. **Signed Q errors**:
   - Listen (`L`): mean +.0024, std .0421, min -.3233, max +.2863, mean absolute error .0247.
   - Open Left (`OL`): mean +.0620, std .5450, min -5.4072, max +3.6596, mean absolute error .2107.
   - Open Right (`OR`): mean +.0572, std .5579, min -5.8211, max +3.5842, mean absolute error .2140.
4. **Action strata coverage**: Oracle best L: 1,233 (36.70%), OL: 1,058 (31.49%), OR: 1,069 (31.82%).
   Of the 15 errors, 12 were listening instead of opening; 3 were opening instead of listening.

### Timing and resources

- External wall sum: 10,327.93s (2.87 hours); parent monotonic: 10,436.89s (2.90 hours);
  worker wall sum: 20,220.20s (5.62 hours).
- Worker concurrency bounds passed on all 6 panels (sum t_worker / 2 <= delta t_parent).
- Monitored peak RSS ranged from 73.74 to 138.27 MiB across cases.
- Raw artifacts: results/oracle/l2_rewards_fresh_n{25,100}_b{0.085,0.5,0.915}_20260926/.

## Verification, status and handoff to Codex

All checks passed prior to launch; Ruff lint/format passed.
No global defaults have been modified.
The fresh validation gate failed under the frozen rule, though demonstrating substantial
progress over ac3df1c. Full report artifact: `l2_fresh_candidate_validation_report.md`.
Documentation updated in `HANDOFF.md`, `BACKLOG.md`, and `docs/BENCHMARK.md`.

Next decisions for Codex:
1. Conduct independent review and Bellman error decomposition of the 15 strict error cases
   (especially the 7 violations at H8).
2. Determine next algorithmic or architectural steps (e.g. higher root traversal budgets
   such as 100k at H >= 8 or boundary-targeted budget scaling).

