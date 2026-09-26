# Current review handoff

## Ownership and audited checkpoint

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs experiments. Keep source fixed during
runs; no source copies outside Git or overwritten raw evidence.

Codex independently reviewed ee3d5b4. All 360 cases have verified coverage,
source hashes against 06cdcd3, real/modeled settings, seeds and status.
Recomputed all 180 distinct reference problems and checked both root-budget
copies, policy probabilities, losses and maximum Q errors.

All 360 first actions match the reference; mean/max first-action loss is zero.
Maximum Q error is 1.713519. This is development evidence with protagonist
H1-H3 and a depth-20 modeled opponent. No new validation gate or global
default promotion follows.

## Depth comparison and timing corrections

The reported 18/360 action changes and 62/360 Q changes count both protagonist
budget copies of each oracle problem. There are 180 distinct comparisons:
- modeled budget25: 9 action changes and 26 Q/value changes out of90;
- modeled budget100: 0 action changes and 5 Q/value changes out of90.

At budget100, maximum Q change is2.264711 despite identical best actions.
Do not infer opponent-policy stability from unchanged protagonist decisions.
Changing depth also changes deterministic search settings/seeds; matching bank
seed indices is not a common-random-number depth experiment.

All worker intervals match durations, at most two overlap, and every panel
passes the monotonic concurrency bound. External elapsed sums135.67s versus
parent monotonic140.804341s. Differences reach1.323687s and change sign in one
panel. The discrepancy remains unresolved; preserve both measurements.

Artifacts: results/oracle/l2_depth20_n{25,100}_b{0.085,0.5,0.915}_20260925/
plus sibling logs/timers. Independent audit:
results/oracle/l2_depth20_review_20260925.json.
Raw artifacts remain local and Git-ignored; documentation commits are not raw
data archives. Opening remains nonterminal, and near-zero opening Q error in
this short panel must not be generalized to longer horizons.

## Next phase: protagonist H4/H5 development

A two-case resource pilot at H4/H5 completed with zero first-action loss.
It used root1000, opponent depth20/budget25, b_j=.5, own belief .5, seed400.
H5 maximum Q error was3.355773. Both worker durations fit the parent monotonic
concurrency bound. Evidence: results/oracle/l2_h45_resource_pilot_20260925/.
The pilot is only a feasibility check at one easy belief, not an accuracy gate.

Run six sequential panels crossing modeled budgets25/100 and b_j=.085/.5/.915.
Each uses protagonist H4/H5, budgets1k/10k/50k, seeds400-401, and own beliefs
.05/.075/.5/.925/.95: 60 cases per panel, 360 total. These are development
beliefs/seeds, with both opening and listening regions to be reported.

Root: empirical Bellman, exact tail, bounded c=1, gamma=.95.
Opponent: fixed depth20, sampled backup, sampled tail, normalized c=1.
No adaptive budget increases, reference substitutions or default changes.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_h45_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 4 5 --budgets 1000 10000 50000 \
      --seed-start 400 --seeds 2 --beliefs 0.05 0.075 0.5 0.925 0.95 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID uniquely. Preserve every failed/timed-out case and source manifest.
Report per-case and per-horizon/belief/root-budget loss, oracle gaps/action sets,
signed Q errors, external and parent monotonic elapsed, worker sums/intervals and
RSS. Loss>.020 counts remain descriptive, not a new held-out validation gate.
Report opening and listening cases separately. Do not call zero loss at one
budget or unchanged action labels general convergence or opponent stability.

## Completed 360-case H4/H5 development study

Antigravity completed all six panels sequentially under source checkpoint 1359053
with RUN_ID `20260925`:
- `results/oracle/l2_h45_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h45_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 360 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 69.23 s | 74.17 s | 140.48 s | PASS (70.24 <= 74.17) | 79.66 MB |
| B_opp=25, b_j=0.500 | 67.03 s | 70.01 s | 134.08 s | PASS (67.04 <= 70.01) | 78.53 MB |
| B_opp=25, b_j=0.915 | 69.93 s | 73.28 s | 139.92 s | PASS (69.96 <= 73.28) | 79.10 MB |
| B_opp=100, b_j=0.085 | 86.63 s | 91.45 s | 176.81 s | PASS (88.41 <= 91.45) | 78.25 MB |
| B_opp=100, b_j=0.500 | 75.97 s | 81.00 s | 155.83 s | PASS (77.92 <= 81.00) | 78.21 MB |
| B_opp=100, b_j=0.915 | 85.30 s | 88.29 s | 169.32 s | PASS (84.66 <= 88.29) | 79.14 MB |
| **Total** | **454.09 s** | **478.20 s** | **916.44 s** | **PASS (6/6)** | **79.66 MB** |

All six panels passed the monotonic concurrency bound. External GNU time (454.09s)
and parent monotonic elapsed (478.20s) differ by 2.99–5.03s per panel; both measurements
are preserved.

### Decision accuracy and loss summary

| Panel | Cases | Errors (loss > 1e-8) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 60 | 4 (6.67%) | 4 | 0.032912 | 0.954069 | 2.1765 | 9.5404 |
| B_opp=25, b_j=0.500 | 60 | 1 (1.67%) | 1 | 0.010268 | 0.616052 | 1.8571 | 14.1285 |
| B_opp=25, b_j=0.915 | 60 | 1 (1.67%) | 1 | 0.001048 | 0.062898 | 1.7438 | 9.0502 |
| B_opp=100, b_j=0.085 | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.7256 | 9.8171 |
| B_opp=100, b_j=0.500 | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.7341 | 5.4413 |
| B_opp=100, b_j=0.915 | 60 | 0 (0.00%) | 0 | 0.000000 | 0.000000 | 1.6739 | 8.1720 |
| **Total** | **360** | **6 (1.67%)** | **6 (1.67%)** | **0.007371** | **0.954069** | **1.8185** | **14.1285** |

Across all 360 cases, **354 decisions (98.33%)** were strictly optimal.

### Breakdown by root budget and horizon

- **Root Budget 1,000**: 4 errors / 120 cases (96.67% accuracy), mean loss 0.020186, max loss 0.954069.
- **Root Budget 10,000**: 2 errors / 120 cases (98.33% accuracy), mean loss 0.001928, max loss 0.168457.
- **Root Budget 50,000**: **0 errors / 120 cases (100.0% accuracy)**, mean loss 0.000000.
Every error observed at 1k or 10k is fully resolved when scaled to 50k traversals.
- **Horizon 4**: 3 errors / 180 cases (98.33% accuracy; 1 at 1k, 2 at 10k, 0 at 50k).
- **Horizon 5**: 3 errors / 180 cases (98.33% accuracy; 3 at 1k, 0 at 10k, 0 at 50k).

### Breakdown by physical belief and action class

- **Beliefs**: Near-boundary transition beliefs $b_i \in \{0.075, 0.925\}$ and symmetric belief $0.500$
  achieved **100.0% accuracy (216/216)**. All 6 errors were concentrated at extreme beliefs
  $b_i=0.05$ (3 errors) and $b_i=0.95$ (3 errors) under $B_{\text{opp}}=25$.
- **Opening Cases (oracle prefers OL/OR)**: 72 cases, 2 errors (2.78%, both in B_opp=25, bj=0.085,
  H=4, bi=0.05 where oracle prefers OL over L by margin 0.1685; resolved at 50k).
- **Listening Cases (oracle prefers L)**: 288 cases, 4 errors (1.39%, premature door opening due to
  Q variance at 1k/10k; all resolved at 50k).

### Signed Q errors and nonterminal door-opening dynamics

- Action `L`: Mean signed error contracts from +0.5653 (1k) to +0.0464 (10k) and -0.0082 (50k).
  Mean absolute error contracts from 0.7735 (1k) to 0.0577 (50k).
- Nonterminal door-opening actions (`OL`, `OR`): Continuation search through 3–4 post-opening steps
  shows positive bias at 1k (+1.44 / +1.58), contracting to +0.39 / +0.53 at 50k.

### Opponent budget sensitivity (25 vs 100 on 60 distinct problems)

Comparing the 60 distinct $(H, b_j, \text{seed}, b_i)$ problems between $B_{\text{opp}}=25$ and $B_{\text{opp}}=100$:
- **6 action flips (10.00%)**:
  - H=4, bj=0.085, s=400, bi=0.95: 25->L vs 100->OR
  - H=4, bj=0.915, s=400, bi=0.05: 25->L vs 100->OL
  - H=4, bj=0.915, s=400, bi=0.95: 25->L vs 100->OR
  - H=5, bj=0.085, s=400, bi=0.05: 25->OL vs 100->L
  - H=5, bj=0.915, s=400, bi=0.05: 25->OL vs 100->L
  - H=5, bj=0.915, s=400, bi=0.95: 25->OR vs 100->L
- **42 Q-value shifts > 1e-4 (70.00%)**, with max Q shift 2.723554.

### Next steps: Codex independent review

Development evidence at H4/H5 ready for Codex review, manifest/hash verification against 1359053,
and recomputation of reference values.

## Remaining qualification

No source changes in this audit; latest implementation verification remains
225 passing tests plus lint/format. Production defaults remain unchanged.
The runner declares modeled mcts.n_sims and opponent.n_sims equal to the chosen
budget; other complete production configs can hash differently. Model depth20
does not imply protagonist depth20, and the point prior is not an empirical
nested hierarchy. Longer-horizon/episode checks, mixtures, finite-prior error,
L4/all39, full script review and global default promotion remain separate gates.
Historical validation results retain their original source/configuration scope.
