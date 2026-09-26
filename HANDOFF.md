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

## Remaining qualification

No source changes in this audit; latest implementation verification remains
225 passing tests plus lint/format. Production defaults remain unchanged.
The runner declares modeled mcts.n_sims and opponent.n_sims equal to the chosen
budget; other complete production configs can hash differently. Model depth20
does not imply protagonist depth20, and the point prior is not an empirical
nested hierarchy. Longer-horizon/episode checks, mixtures, finite-prior error,
L4/all39, full script review and global default promotion remain separate gates.
Historical validation results retain their original source/configuration scope.
