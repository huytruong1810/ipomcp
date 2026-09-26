# Current review handoff

## Working branch and ownership

Work in the single WSL checkout /home/andyj1810/projects/ipomcp on main.
The user requested direct main development; create another branch only for
separately justified side work. main was fast-forwarded to bc35a16 and pushed.
fix/tiger-policy-inversion was then deleted locally and remotely.
Codex owns coding/math; Antigravity runs declared experiments. Preserve raw
evidence, fixed source manifests and the outcomes of failed gates.

## Fresh candidate gate: audited failure

The f4d2aef candidate gate FAILED: all 3,360 requested cases completed,
15 strict errors and 9 losses above .020 + 1e-8, maximum loss .205046.
H1–H4 have no errors; H5 has 1 violation, H6 has 1 violation plus 5 near ties,
H8 has 7 violations plus 1 near tie. The gate is not zero-loss: its fixed
criterion allows .020 loss but zero cases exceeding that bound.

Codex verified every manifest/source hash against f4d2aef, requested case
coverage, full root/modeled configurations, policy probabilities, loss/Q-error
arithmetic and worker intervals. Replayed all 15 strict errors with the
original 50k budget and independently recomputed their reference values.
Every complete row, including policy and Q estimates, reproduces exactly.
Other references were checked arithmetically, not independently recomputed
in this audit.

Raw runs: results/oracle/l2_rewards_fresh_n{25,100}_b{0.085,0.5,0.915}_20260926/.
Audit: results/oracle/l2_rewards_fresh_review_20260926.json.
Raw data remain local and Git-ignored, not archived by documentation commits.
All tested beliefs/seeds are now development evidence.

## Competing workloads and timing

The user reports another codebase's agent consuming CPU/GPU resources.
This can affect elapsed time, throughput and wall/RSS failures. It does not
explain the 15 reproducible decision errors: those completed fixed-budget
seeded solves reproduce exactly under the current environment.
Do not dismiss gate failures as machine load or infer isolated algorithm
speed from these runs. Preserve timeouts as execution failures, not policy
errors, and do not replace them silently.

Raw fresh-gate totals: external 10,327.93s, parent monotonic 10,436.889919s,
worker sum 20,220.202898s. All two-worker consistency bounds pass; peak case
RSS ranges 73.74–138.27 MiB. Clock/provenance discrepancies remain unresolved.
Worker time includes reference computation. Earlier and current fresh gates
also differ in beliefs, seeds and modeled-policy identity; historical rate
comparisons are descriptive, not causal estimator comparisons.

## Bellman diagnosis

For each action, the post-search diagnostic verifies:
Q_hat-Q_star = immediate-reward error
+ gamma sum_o (p_hat(o)-p(o)) V_star(o)
+ gamma sum_o p_hat(o) (V_hat(o)-V_star(o)).
No reference value is supplied to search.

Compare each chosen action's error with the true best action's error.
In 14/15 cases, the child-value contribution is larger in magnitude than
the root chance-frequency contribution. Root immediate rewards are integrated.
Both competing root actions receive roughly 24,600–24,950 visits; none of these
failures is an unvisited root action.

| Case: B_opp, b_j, H, seed, b_i | Loss | Chosen / best | Chance contribution to ranking error | Child-value contribution |
| --- | ---: | --- | ---: | ---: |
| 25, .085, 8, 7004, .9625 | .205046 | OR / L | .011202 | .217004 |
| 25, .915, 8, 7004, .0375 | .163827 | L / OL | -.001593 | .174115 |
| 25, .915, 8, 7000, .9625 | .147591 | L / OR | .001423 | .154569 |
| 25, .5, 5, 7003, .0375 | .058131 | OL / L | .035497 | .024024 |

The H5 case is the exception: root chance error is larger, with a material
child-value contribution too. These identities localize estimation error;
they do not prove a universal bias mechanism or general convergence.

Full 15-case margin decomposition:
results/oracle/l2_fresh_margin_decomposition_20260926.json.
Replays and branches:
results/oracle/l2_rewards_fresh_diagnosis_20260926/.
Diagnostic source:
results/oracle/diagnose_fresh_gate_20260926.py.

## Root budget development check

The eight observed H8 error cases were rerun at 100k traversals. All 8 now
select an optimal first action (zero loss), including the seven tolerance
violations. The modeled opponent and reference Q vectors remain identical.
Root seeds change with budget, so these are not nested simulation prefixes.

Lower-horizon 100k checks also cover all seven observed H5/H6 errors:
results/oracle/l2_rewards_lower_horizon_100k_20260926/.
All seven complete within the .020 tolerance: five become strictly optimal;
two H6 cases retain losses .006699 and .005156. Combined with H8, all 15
selected cases pass the tolerance at 100k, with 13 strictly optimal. This is
not a pass of the full original gate. Aggregate audit:
results/oracle/l2_rewards_budget_probe_review_20260926.json.

These are selected observed cases, not a fresh gate or a regression check on
previously correct cases. The historical 50k gate stays failed. Increasing
only H8 cannot resolve the original H5/H6 violations without changing those
cases too.

## Next Antigravity phase: full observed H5/H6/H8 grid at 100k

Test 100k on every observed case at H5/H6/H8, including previously correct
decisions. Six panels x 3 horizons x 8 seeds x 10 beliefs = 1,440 cases.
Use the existing 50k rows as the source-matched baseline; the solver source is
unchanged. Report both fixed errors and new regressions, with exact equality
of paired opponent metadata and oracle Q vectors.

Keep all estimator, exploration and horizon semantics fixed:
empirical Bellman, exact final step and history rewards, bounded c=1,
gamma=.95, root node capacity 200. Modeled L1 stays depth 20 with 25/100
simulations, sampled rewards/backups/tail, normalized c=1, epsilon1e-6,
node capacity 500. No boundary-specific exceptions or use of oracle gaps
during search. Do not shrink max_depth to make the gate easier: that changes
the declared finite-horizon objective. Current evidence does not establish
that a different exploration rule is required.

This is development on the observed grid, with .020 counts descriptive.
A pass cannot relabel either failed fresh gate or qualify unseen cases.
Keep defaults opt-in. Any later validation needs a new frozen unused grid.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git branch --show-current
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_rewards_100k_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 5 6 8 --budgets 100000 \
      --seed-start 7000 --seeds 8 \
      --beliefs 0.002 0.0175 0.0275 0.0375 0.0875 0.9125 0.9625 0.9725 0.9825 0.998 \
      --backup empirical_bellman --exact-final-step --exact-history-rewards \
      --exploration bounded --exploration-const 1 --gamma 0.95 \
      --workers 2 --timeout 240 --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID uniquely and use a clean main commit. Keep raw timings but
record concurrent workload context; do not infer planner overhead from it.
No selective retries, budget escalation, case removal or overwritten artifacts.

## Scope and verification

No solver code changed in this audit; last implementation verification is
241 passing tests plus Ruff lint/format. This phase adds replay, reference,
decomposition and budget evidence, not a general correctness certificate.
Production defaults, empirical nested priors, mixed levels, L3-versus-L2
episodes, L4/all39 and remaining script-review gates stay open.
