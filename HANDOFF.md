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

## Remaining qualification

No solver source changed; the latest implementation test result remains 225
passes plus Ruff lint/format. Shell syntax and diff whitespace were checked
for this documentation-only checkpoint.

Production defaults remain unchanged. Even a fresh L2 pass would qualify
only this finite-policy point-prior contract and declared configuration.
It would not qualify empirical nested priors, mixed reasoning levels,
L3-versus-L2 episode returns, L4/all39 or all domains. Complete script review,
timing instrumentation and production configuration matching remain open.
