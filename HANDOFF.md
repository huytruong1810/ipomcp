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

### Antigravity command

Use Bash, replace RUN_ID uniquely, and keep source fixed across all six panels.
Do not enable exact-history-rewards for the modeled opponent.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_rewards_fresh_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 1 2 3 4 5 6 8 --budgets 50000 \
      --seed-start 7000 --seeds 8 \
      --beliefs 0.002 0.0175 0.0275 0.0375 0.0875 0.9125 0.9625 0.9725 0.9825 0.998 \
      --backup empirical_bellman --exact-final-step --exact-history-rewards \
      --exploration bounded --exploration-const 1 --gamma 0.95 \
      --workers 2 --timeout 240 --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

## Status

No solver changes or default promotion in this audit. Last implementation
verification remains 241 tests passed plus Ruff lint/format. Documentation
and command syntax are checked; implementation tests were not redundantly rerun.
Timing provenance, production configuration matching, mixed/empirical priors,
L3-versus-L2 episode qualification and the remaining script review stay open.
