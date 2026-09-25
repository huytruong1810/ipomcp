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

Evidence: results/l2-contract-20260925/smoke/, smoke.log and smoke.time.
The focused reference/runner/supervisor tests passed (35 tests).
Full-suite verification: 217 tests passed in 303.62s, including all domain
integration and supervisor fault tests. Ruff lint and formatting passed.
Raw output: results/l2-contract-20260925/tests.log.

## Antigravity: next development protocol

After a clean committed checkout and completed engineering checks, run six
panels sequentially: opponent depth1/depth2 crossed with b_j=.085/.5/.915.
Each panel uses H1-H3, budgets1k/10k, seeds300-304 and own beliefs
.05/.2/.5/.8/.95: 150 cases per panel, 900 total. These are DEVELOPMENT runs.
There is no held-out gate or adaptive candidate promotion.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for depth in 1 2; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_fixed_d${depth}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth "$depth" --opponent-belief "$belief" \
      --planners mcts --horizons 1 2 3 --budgets 1000 10000 \
      --seed-start 300 --seeds 5 --beliefs 0.05 0.2 0.5 0.8 0.95 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID with a unique identifier; preserve source/configuration throughout.
Do not overwrite/retry failures invisibly or replace exact reference computation
on timeout. Report all coverage/failures; per-depth/private-belief/horizon/budget
action loss and oracle gaps; signed Q errors; both parent monotonic and external
elapsed; worker sums/concurrency checks; peak monitored RSS. Keep .020 and strict
loss summaries descriptive, not new validation gates. Compare both opponent
depths explicitly to confirm that replanning semantics affect the modeled task.

## Remaining gates

Prior L1 2000-case frozen accuracy gate remains passed at its exact tested
settings. Old 50k/200k failures stay historical failures. No permanent
all-horizon resolution or global default promotion is claimed.

Still pending: matched finite-budget modeled opponents, deeper level mixtures,
long/deep Tiger before/after, L4/all39 resource qualification, finite-prior error
and remaining demo/visualization semantic review. The new exact-opponent L2
contract isolates one layer of these requirements; it does not discharge all.
