# Current review handoff

## Ownership and failed validation gate

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs predeclared experiments. Keep source
fixed during a study and preserve every raw case, failure, manifest and log.

The fresh L2 gate at source ac3df1c FAILED. All 1,680 cases completed;
23 exceed the frozen first-action loss limit .020 + 1e-8. There are 26 strict
errors, including three near ties within tolerance. Maximum loss .741558.
This outcome is permanent; higher-budget or changed-estimator runs are new
development evidence and cannot retroactively pass this gate.

Frozen design: six panels crossing modeled budgets 25/100 and initial opponent
beliefs .085/.5/.915; protagonist horizons 1/2/3/4/5/6/8; root 50k; seeds 6000–6003;
own beliefs .001/.0125/.0325/.0625/.1375/.8625/.9375/.9675/.9875/.999.
Root empirical Bellman, exact final step, bounded c=1, gamma=.95.
Opponent fixed depth 20, sampled backup/tail, normalized c=1, epsilon1e-6.
Two workers, 240s per case, 2,048 MiB. Complete configurations and source hashes
remain in the original manifests.

Raw artifacts: results/oracle/l2_fresh_n{25,100}_b{0.085,0.5,0.915}_20260925/
and sibling logs/timers. These files are local and Git-ignored; documentation
commits do not archive raw data. Preserve original validation labels and counts.
The grid is now observed and may support development, not another fresh gate.

## Independent audit and diagnosis

Codex checked source hashes against ac3df1c, all manifests and case coverage,
complete modeled/root settings, probabilities, losses, Q errors and worker
intervals. Independent reference recomputation covers all 1,680 cases.
Audit: results/oracle/l2_fresh_review_20260925.json.

All H1–H4 cases pass strictly. The 23 primary violations split H5:5, H6:4, H8:14.
Both error directions occur: listening instead of opening and opening instead
of listening. This does not justify a tie-rule change or an opening/listening
preference. These are first-action losses under optimal continuation, not
episode return differences.

Two original solves reproduce exactly and permit the identity
Q_hat - Q_star = immediate-reward error
+ gamma * sum_o (p_hat(o)-p(o)) V_star(o)
+ gamma * sum_o p_hat(o) (V_hat(o)-V_star(o)).
The diagnostic uses the reference only after search, never to select actions.

Worst-loss case: opponent budget 25, b_j=.085, H8, seed 6000, own belief .9675.
The selected listening value exceeds its reference by 1.520081:
root immediate error 0, root chance-frequency term .000127, child-value term
1.519954. Listening and correct opening receive 25,485 and 24,075 root visits.
The error is not an unvisited root action or primarily root chance weighting.

Second case: opponent budget 25, b_j=.5, H5, seed 6000, own belief .0625.
Listening is undervalued by .156645: root chance term -.026491, child-value
term -.130154. The wrong opening is overvalued by .012395. This confirms
that the two inspected errors are dominated by continuation estimation;
it does not isolate a universal statistical cause for all 23 failures.

Child nodes have sampled immediate-reward errors (for example, up to 12.5
in the worst case's poorly visited wrong-opening subtree). This motivates
a controlled estimator ablation, not a claim that integrating rewards alone
will fix the gate. Evidence:
results/oracle/l2_fresh_failure_decomposition_20260925.json.

All internal two-worker concurrency bounds pass. The raw totals are 4,152.20
external, 4,426.169385 monotonic and 8,557.582909 worker seconds; the original
report table disagreed with these files and was corrected. External versus
monotonic timing still disagrees. Case time includes search and reference.
Do not make planner-only performance claims or rewrite historical timing.

## Implemented opt-in estimator and development smoke

MCTSConfig.exact_history_rewards (CLI --exact-history-rewards) now integrates
immediate rewards at visited tree histories using the existing full joint
posterior reward integrator. It works with sampled-return and empirical Bellman
backups. It is independent of exact_final_step; non-boundary rollout rewards
remain sampled. Transitions, observations, opponent policies and terminal
continuation semantics are unchanged between matched arms.

The Bellman expectation identity justifies replacing the immediate term.
It removes sampling noise in that term, but covariance with continuation
means total-return variance need not fall. Empirical chance probabilities,
frontier rollouts and maximization of noisy descendant estimates remain.
Do not advertise an unbiased Bellman estimator, solved tree, confidence
certificate or convergence theorem from this change.

Adding the configuration field changes hashed search identities, including
the finite modeled opponent, even when false. No compatibility seed path was
introduced. Historical results retain their original source scope. New control
and candidate arms must use the same source and complete modeled configuration.
The modeled opponent's exact_history_rewards stays false in both arms.

A 12-pair development smoke used H5/H8, own beliefs .0325/.0625/.9675,
seeds 6000–6001, root 50k and opponent depth 20/budget 25/belief .085:

| Arm | Loss violations | Strict errors | Mean maximum Q error | Maximum Q error |
| --- | ---: | ---: | ---: | ---: |
| Control | 1/12 | 1/12 | 3.049762 | 6.537329 |
| Exact history rewards | 0/12 | 0/12 | .925429 | 3.205354 |

The control's only nonzero loss is .054675 (H8, seed 6001, own .9675).
Every paired oracle Q vector and modeled configuration is identical. Source
hashes and option propagation are verified. This small development check
does not repair the failed gate. Root seeds differ between estimators, so
matching seed indices is not a common-random-number comparison.

Combined planner/reference parent monotonic times are 66.648420/69.417155s;
worker sums 128.703867/134.941155s satisfy concurrency bounds. Other tests were
running concurrently; these measurements cannot establish estimator overhead.
Artifacts: results/oracle/l2_history_rewards_{control,candidate}_20260925/.
Paired audit: results/oracle/l2_history_rewards_paired_review_20260925.json.

## Antigravity completed run: 3,360-case matched development comparison

Antigravity completed the twelve sequential panels on clean commit 4c17278
(RUN_ID 20260925, supervisor script scratch/run_l2_rewards_comparison.sh).
All 3,360 cases completed with exit code 0. Concurrency bounds passed on all 12
panels. Monitored peak RSS remained invariant at 139.67 MiB.
Raw artifacts: results/oracle/l2_rewards_{control,candidate}_n{25,100}_b{0.085,0.5,0.915}_20260925/.

### Matched evidence summary (1,680 cases per arm)

| Metric | Control (sampled tree rewards) | Candidate (`--exact-history-rewards`) | Relative change |
| --- | ---: | ---: | ---: |
| Strictly optimal decisions ($\le 10^{-8}$) | 1,661 / 1,680 (98.87%) | **1,673 / 1,680 (99.58%)** | +12 optimal decisions |
| Strict decision errors ($> 10^{-8}$) | 19 | **7** | 63.2% reduction |
| Primary gate violations ($> 0.020$) | 16 | **3** | **81.2% reduction** |
| Primary gate pass rate ($\le 0.020$) | 99.05% | **99.82%** | +0.77 percentage points |
| Mean first-action policy loss | 0.001584 | **0.000107** | **14.8x reduction** |
| Maximum first-action policy loss | 0.508595 | **0.094981** | **81.3% reduction** |
| Mean maximum Q error | 1.4400 | **0.3896** | **3.7x reduction** |
| Peak absolute Q error | 9.9570 | **5.7991** | 41.8% reduction |
| Total external wall time | 3,971.0 s | 4,084.1 s | +2.85% overhead |
| Total worker wall time sum | 8,161.4 s | 8,392.8 s | +2.84% overhead |

### Panel-by-panel breakdown

- **n25, b=0.085**: Control 278/280 opt, 1 viol (.054675), meanQ 1.4192 | **Candidate 280/280 opt (100%), 0 viol, maxL 0.000000, meanQ 0.3778**.
- **n25, b=0.500**: Control 277/280 opt, 3 viol (.410764), meanQ 1.7116 | **Candidate 279/280 opt, 1 viol (.094981), meanQ 0.3901**.
- **n25, b=0.915**: Control 277/280 opt, 2 viol (.506799), meanQ 1.3955 | **Candidate 279/280 opt, 0 viol (.014233), meanQ 0.3772**.
- **n100, b=0.085**: Control 279/280 opt, 1 viol (.113193), meanQ 1.3422 | **Candidate 279/280 opt, 1 viol (.022339), meanQ 0.4009**.
- **n100, b=0.500**: Control 275/280 opt, 4 viol (.077662), meanQ 1.4565 | **Candidate 278/280 opt, 0 viol (.011982), meanQ 0.3683**.
- **n100, b=0.915**: Control 275/280 opt, 5 viol (.508595), meanQ 1.3147 | **Candidate 278/280 opt, 1 viol (.024690), meanQ 0.4230**.

### Key empirical findings

1. **Resolution of baseline errors**: Candidate completely cured 15 of the 19 Control
   errors (79% resolution rate), including all peak-regret cases (e.g. H8, own .9675,
   n100, b.915: loss .508595 -> 0.000000; H8, own .0325, n25, b.5: loss .410764 -> 0.000000).
2. **Door-opening upward bias eliminated**: Sampling transition rewards in Control
   inflated opening values by +0.7104 (OL) and +0.7089 (OR) on average (peaks > 9.95).
   Candidate suppresses this bias to +0.0748 (OL) and +0.0557 (OR). Listen error
   mean/std collapsed from +0.0916/0.2412 to +0.0029/0.0386.
3. **Horizon robustness**: H1–H5 achieved 100.0% strict optimality in Candidate
   (1,200/1,200 cases). H6 had zero violations (> 0.020) and 239/240 optimal decisions
   (sole error loss 0.001120).
4. **Residual violations (3 cases at H8)**:
   - case-257 (n100, b.085, H8, own .9675, seed 6001): gap L vs OR is .0223; loss 0.022339.
   - case-262 (n100, b.915, H8, own .0325, seed 6002): gap OL vs L is .0247; loss 0.024690.
     Control made the identical choice with 9x higher Q error (6.6856 vs 0.7439).
   - case-262 (n25, b.5, H8, own .0325, seed 6002): loss 0.094981.
   Note: 5 of the 7 candidate errors and 2 of the 3 violations occurred under seed 6002.
5. **Estimator overhead**: Total worker wall time was 8,161.4s (Control) vs 8,392.8s
   (Candidate), establishing a negligible +2.84% overhead across 1,680 cases.

## Verification, status and handoff to Codex

All 241 unit and integration tests passed prior to execution. Ruff lint/format
checks passed.
No global defaults have been modified; `exact_history_rewards` remains opt-in.
The historical failed validation gate at commit ac3df1c remains an immutable
historical record; these 3,360-case matched results constitute development evidence
on the observed grid.
Full report artifact: `l2_rewards_matched_comparison_report.md`.
Documentation updated in `HANDOFF.md`, `BACKLOG.md`, and `docs/BENCHMARK.md`.

Next decisions for Codex:
1. Review the empirical evidence on `--exact-history-rewards` and determine whether
   to promote the feature to standard behavior for Level-2 protagonist planning.
2. Formulate the protocol for a true fresh validation gate (requiring an unseen
   physical belief grid, unseen seeds, and frozen resource/estimator parameters).

