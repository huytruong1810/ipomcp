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

Next:360-case depth20 DEVELOPMENT extension, six sequential panels crossing
modeled budgets25/100 and b_j=.085/.5/.915. Each panel: H1-H3,
root budgets1k/10k, seeds400-401, own beliefs .05/.2/.5/.8/.95
(60 cases per panel). Root empirical Bellman/exact tail/bounded c=1; modeled
sampled backups/sampled tail/normalized c=1. Retain gamma=.95 and all settings.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_depth20_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 1 2 3 --budgets 1000 10000 \
      --seed-start 400 --seeds 2 --beliefs 0.05 0.2 0.5 0.8 0.95 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID uniquely. Keep source/config fixed. Report every requested case
and failure; per-horizon/belief/budget loss, action sets and signed Q errors;
timing.json plus external timer, concurrency and RSS. Compare matched depth3
and depth20 rows using the already-inspected seeds. No unreported case replacement,
exact-L1 substitution or scalar-belief approximation. .020 counts are descriptive,
not a newly declared validation criterion.

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
