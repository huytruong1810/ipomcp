# Current review handoff

## Ownership and independent audit

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns coding/math; Antigravity runs the declared experiments. Keep source
fixed during runs and preserve raw cases, failures and manifests.

Codex independently audited 03ca490 against source e63db30. All 120 H6/H8
manifests/cases match the requested settings and source hashes; all 120
exhaustive reference recomputations agree within 1e-8. Recorded probabilities,
losses and Q errors check out. The six repeated pilot rows reproduce exactly.

All 120 chosen actions have zero first-action loss, but every optimum is L.
There are no opening-optimal cases. The smallest best-versus-second-best gap
is .640013; this grid does not test near ties. Maximum Q error is 8.523833,
and mean maximum Q error is 3.573702. Positive mean signed opening errors
are observed panel statistics, not an established estimator-bias mechanism.

The 60 distinct opponent-budget comparisons all change Q vectors by more than
1e-4 (maximum 1.616884), with no optimal action changes. A different modeled
budget changes the opponent policy law; it is not solely a protagonist resource
comparison.

Worker intervals reproduce durations with at most two concurrent workers;
every monotonic concurrency bound passes. External versus parent monotonic
elapsed totals are 555.49 versus 588.865233 seconds. Per-panel discrepancies
are 5.110796–5.693711 seconds, correcting the reported upper endpoint.
Timing provenance remains unresolved. Worker wall time includes the planner
and exhaustive reference, not planner-only latency.

Raw evidence: results/oracle/l2_h68_n{25,100}_b{0.085,0.5,0.915}_20260925/
and sibling timers/logs. Independent audit:
results/oracle/l2_h68_review_20260925.json.
These artifacts are local and Git-ignored; a documentation commit does not
archive the raw evidence.

## Next phase: fresh L2 validation protocol

This design is fixed before any new validation solve. Use the established
bounded-loss tolerance epsilon_loss=.020. This is a new L2 gate, not a
reinterpretation of any earlier development study or failed L1 gate.

Freeze the source at the commit containing this protocol; record its full
commit ID and source manifest before launch. Solver source is unchanged from
e63db30. Run six sequential panels crossing opponent budgets 25/100 with
initial private beliefs .085/.5/.915. Each panel has 280 cases:

- Protagonist horizons 1, 2, 3, 4, 5, 6, 8.
- Root simulations 50,000, empirical Bellman backups, exact final step,
  bounded UCB c=1, gamma=.95, node capacity 200.
- Ten physical beliefs .001/.0125/.0325/.0625/.1375/
  .8625/.9375/.9675/.9875/.999.
- Seeds 6000–6003 inclusive.
- Modeled L1: fixed depth 20, sampled backups and tail, normalized UCB c=1
  and epsilon=1e-6, node capacity 500. Both modeled configuration budget
  fields equal the declared 25 or 100. Retain every complete configuration
  field because configuration enters deterministic policy seed identity.
- Two workers, 240 seconds per case, 2,048 MiB per worker.

Total: 6 x 7 x 10 x 4 = 1,680 cases. These ten physical beliefs and four seed
indices were absent from all local oracle manifests and 17,239 local case
rows inspected before freezing. No reference values or sampled policies for
this new grid were computed. Opponent beliefs, architecture and resource
choices remain development-selected; freshness concerns physical beliefs
and seeds, not an independent random population sample.

More extreme physical beliefs broaden the design beyond the all-listen grid.
Do not screen cases by their oracle action or discard near ties after launch.
Report actual optimal-action coverage and minimum gaps separately for each
horizon; a pass applies only to the declared cases.

### Frozen acceptance and reporting

The primary accuracy gate requires every requested case to complete and
satisfy first_action_loss <= .020 + 1e-8. First-action loss is
max_a Q*(b,a) - sum_a pi(a|b) Q*(b,a), using the matched finite-policy
reference and optimal continuation. It is not cumulative episode regret.

Any completed case exceeding the bound fails the accuracy gate. Missing,
timed-out, resource-killed or invalid cases prevent a pass and are reported
separately from measured policy errors. No selective retries, seed replacement,
budget escalation or reference approximation may turn this run into a pass.
A changed design requires a new declared study and preserves this outcome.

Report strict errors (loss > 1e-8), primary violations, maximum/mean loss,
action gaps and sets, Q errors per action, and opening/listening strata per
horizon/opponent. Keep every failed case. Report external and monotonic timing,
worker intervals/sums and RSS. Timing discrepancies block runtime claims,
independently of the accuracy result. Four seed indices do not establish
population-wide success probabilities; do not treat cases as independent
Bernoulli trials for a universal reliability certificate.

### Antigravity launch command

Start only from the clean protocol commit. Replace RUN_ID uniquely; retain
the same source for all six panels. No new validation run has been launched
by Codex.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_fresh_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 1 2 3 4 5 6 8 --budgets 50000 \
      --seed-start 6000 --seeds 4 \
      --beliefs 0.001 0.0125 0.0325 0.0625 0.1375 0.8625 0.9375 0.9675 0.9875 0.999 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

## Completed 1,680-case fresh L2 validation suite

Antigravity completed all six panels sequentially under source checkpoint ac3df1c
with RUN_ID `20260925`:
- `results/oracle/l2_fresh_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fresh_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fresh_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fresh_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fresh_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_fresh_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 1,680 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 637.28 s | 663.85 s | 1297.07 s | PASS (648.54 <= 663.85) | 133.52 MB |
| B_opp=25, b_j=0.500 | 632.74 s | 660.18 s | 1289.44 s | PASS (644.72 <= 660.18) | 133.25 MB |
| B_opp=25, b_j=0.915 | 671.36 s | 700.17 s | 1369.31 s | PASS (684.66 <= 700.17) | 133.62 MB |
| B_opp=100, b_j=0.085 | 744.15 s | 773.30 s | 1515.15 s | PASS (757.58 <= 773.30) | 133.82 MB |
| B_opp=100, b_j=0.500 | 702.43 s | 730.83 s | 1432.22 s | PASS (716.11 <= 730.83) | 134.34 MB |
| B_opp=100, b_j=0.915 | 760.33 s | 790.35 s | 1548.88 s | PASS (774.44 <= 790.35) | 133.72 MB |
| **Total** | **4,148.29 s** | **4,318.68 s** | **8,452.07 s** | **PASS (6/6)** | **134.34 MB** |

All six panels passed the monotonic concurrency bound. External GNU time (4,148.29s,
~1.15h) and parent monotonic elapsed (4,318.68s, ~1.20h) differ by 26.57–30.02s
per panel (~4.1%); both measurements are preserved.

### Primary gate outcome: GATE FAILED

Under the frozen protocol ($\text{loss} \le 0.020 + 1e-8$ for every case without selective
retries or budget escalation), the validation gate **FAILS** due to 23 violations:
- **Primary Gate Passes ($\mathcal{L} \le 0.020$)**: **1,657 / 1,680 (98.63%)**
- **Primary Gate Violations ($\mathcal{L} > 0.020$)**: **23 / 1,680 (1.37%)**
- **Strict Optimal Choices ($\mathcal{L} \le 1e-8$)**: **1,654 / 1,680 (98.45%)**
- **Near-Tie Bounded Passes ($1e-8 < \mathcal{L} \le 0.020$)**: **3 / 1,680 (0.18%)**
- **Overall Policy Loss**: Mean 0.002198, Max 0.741558
- **Overall Q Errors**: Mean max Q error 1.4544, Max max Q error 9.4911

### Panel breakdown

| Panel | Cases | Gate Violations | Strict Errors | Pass Rate | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 280 | 5 | 6 | 98.21% | 0.005394 | 0.741558 | 1.3838 | 8.8418 |
| B_opp=25, b_j=0.500 | 280 | 3 | 3 | 98.93% | 0.000872 | 0.137071 | 1.5864 | 9.4911 |
| B_opp=25, b_j=0.915 | 280 | 3 | 3 | 98.93% | 0.002083 | 0.261441 | 1.4129 | 9.0953 |
| B_opp=100, b_j=0.085 | 280 | 3 | 3 | 98.93% | 0.000672 | 0.109319 | 1.5203 | 7.4656 |
| B_opp=100, b_j=0.500 | 280 | 3 | 5 | 98.93% | 0.001475 | 0.164706 | 1.3743 | 8.8610 |
| B_opp=100, b_j=0.915 | 280 | 6 | 6 | 97.86% | 0.002689 | 0.402415 | 1.4490 | 6.7104 |
| **Total** | **1,680** | **23** | **26** | **98.63%** | **0.002198** | **0.741558** | **1.4544** | **9.4911** |

### Horizon and belief localization

- **Horizons 1–4**: **960 / 960 cases (100.0%) passed** with zero gate violations and
  zero strict errors (mean loss 0.000000).
- **Horizons 5–8**: 23 violations across 720 cases (96.81% pass rate):
  - H5: 5 violations / 240 cases (97.92% pass rate; max loss 0.137071)
  - H6: 4 violations / 240 cases (98.33% pass rate; max loss 0.165231)
  - H8: 14 violations / 240 cases (94.17% pass rate; max loss 0.741558)
- **Belief Concentration**:
  - Boundary transition beliefs: $b_i = 0.0325$ (12 violations) and $b_i = 0.9675$ (10 violations),
    plus 1 violation at $b_i = 0.0625$.
  - Extreme ($0.0010, 0.0125, 0.9875, 0.9990$) and moderate ($0.1375, 0.8625, 0.9375$) beliefs:
    **1,176 / 1,176 cases (100.0%) passed** with zero violations.

### Action strata and signed Q errors

- **Opening Cases (oracle prefers OL/OR)**: 915 cases (54.46%), 12 violations (98.69% pass rate;
  mean loss 0.003004, max loss 0.741558).
- **Listening Cases (oracle prefers L)**: 765 cases (45.54%), 11 violations (98.56% pass rate;
  mean loss 0.001233, max loss 0.164706).
- **Signed Q Errors**:
  - Action `L`: Mean signed error +0.098280, mean absolute error 0.127690 (range [-0.885658, +1.520081]).
  - Actions `OL`, `OR`: Mean signed errors +0.721414 and +0.732191; mean absolute errors 0.766387 and 0.781955.

### Opponent budget sensitivity (25 vs 100 on 840 distinct problems)

Comparing the 840 distinct $(H, b_j, \text{seed}, b_i)$ problems between $B_{\text{opp}}=25$ and $B_{\text{opp}}=100$:
- **67 action flips (7.98%)**
- **490 Q-value shifts > 1e-4 (58.33%)**
- **Maximum Q-value shift**: 10.345500.

### Next steps: Codex independent audit

All 1,680 case files and manifests are stored in `results/oracle/l2_fresh_n{25,100}_b{0.085,0.5,0.915}_20260925/`.
Ready for Codex audit, manifest verification against ac3df1c, and recomputation of reference values.

## Remaining qualification

No solver source changed; the latest implementation test result remains 225
passes plus Ruff lint/format. Shell syntax and diff whitespace were checked
for this documentation-only checkpoint.

Production defaults remain unchanged. Even a fresh L2 pass would qualify
only this finite-policy point-prior contract and declared configuration.
It would not qualify empirical nested priors, mixed reasoning levels,
L3-versus-L2 episode returns, L4/all39 or all domains. Complete script review,
timing instrumentation and production configuration matching remain open.
