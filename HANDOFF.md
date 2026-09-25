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

## Antigravity development protocol

After the final clean implementation commit, run six panels sequentially:
modeled budget25/100 crossed with initial b_j=.085/.5/.915, fixed depth3.
Each uses H1-H3, actual budgets1k/10k, seeds400-404 and own beliefs
.05/.2/.5/.8/.95: 150 cases per panel, 900 total.

Root uses empirical Bellman, exact tail and bounded c=1. The modeled opponent
uses sampled means, sampled tail, normalized UCB c=1. These are deliberately
different configured computations; record them separately.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_finite_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 3 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 1 2 3 --budgets 1000 10000 \
      --seed-start 400 --seeds 5 --beliefs 0.05 0.2 0.5 0.8 0.95 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID uniquely; source/config stay fixed across all panels. Report full
coverage and failures, per-opponent-budget/belief/horizon/root-budget losses,
strict action agreement, signed Q errors, modeled action-law differences,
monotonic/external elapsed and worker bounds, monitored RSS. .020 loss counts
are descriptive, not a new held-out gate. Do not replace resource failures with
an exact-L1 policy or scalar-belief approximation.

## Remaining qualification

Finite opponent depth3 and budgets25/100 are development settings. Production
depth20, mixed levels, empirical priors, long/deep Tiger before/after, L4/all39
and remaining demo/visualization review remain unqualified. Keep the previous
900 exact-opponent cases and all L1 evidence unchanged. Prior frozen L1 passes
and failed gates retain their original source/configuration scope.
