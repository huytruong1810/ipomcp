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

## Antigravity next run: matched development comparison

Tests have passed. From the clean committed implementation checkpoint,
Antigravity may run twelve sequential panels: both estimator arms on each of the six
modeled conditions from the failed gate. Reuse the observed grid explicitly
as development: H1/2/3/4/5/6/8, ten beliefs, seeds 6000–6003, root 50k.
Each arm has 1,680 cases; combined total 3,360. The .020 loss count is diagnostic,
not a new fresh validation gate. Keep all cases and report regressions as well
as improvements. Do not select only old failing rows or adapt resources.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    for arm in control candidate; do
      reward_option=()
      if [ "$arm" = candidate ]; then reward_option=(--exact-history-rewards); fi
      out=results/oracle/l2_rewards_${arm}_n${budget}_b${belief}_RUN_ID
      /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
        uv run python -m examples.experiments.planner_oracle_experiment \
        --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
        --opponent-budget "$budget" --opponent-backup sampled \
        --opponent-exploration normalized --opponent-exploration-const 1 \
        --planners mcts --horizons 1 2 3 4 5 6 8 --budgets 50000 \
        --seed-start 6000 --seeds 4 \
        --beliefs 0.001 0.0125 0.0325 0.0625 0.1375 0.8625 0.9375 0.9675 0.9875 0.999 \
        --backup empirical_bellman --exact-final-step "${reward_option[@]}" \
        --exploration bounded --exploration-const 1 --gamma 0.95 \
        --workers 2 --timeout 240 --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
    done
  done
done
```

Use Bash for the array expansion, replace RUN_ID uniquely, and start from a
clean fixed commit. Preserve both manifests and verify identical modeled
configurations and paired oracle Q values. Record action/gap strata, losses,
per-action Q errors, resource failures and timings. Do not infer improvements
from old-source versus new-source counts: the opponent seed identity changed.

## Verification and remaining gates

Focused analytic, terminal, option-independence, bank-law and runner tests:
46 passed. Full suite: 241 passed in 392.64s, including integration tests.
Ruff lint/format and diff whitespace checks passed. Test log:
results/history-rewards-20260925/tests.log.
No global defaults were promoted. The estimator remains opt-in. The historical
failed gate is preserved. Fresh validation after development needs unseen
beliefs/seeds and a separately frozen design; mixed/empirical nested priors,
L3-versus-L2 episodes, L4/all39, timing provenance and remaining script review
are still open.
