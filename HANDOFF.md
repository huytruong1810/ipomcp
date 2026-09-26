# Current review handoff

## Ownership and audited checkpoint

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns code/math; Antigravity runs experiments. Keep source fixed during
runs. Preserve original raw evidence; use Git for source history, not additional
codebase copies.

Codex independently audited c68754b against source checkpoint 1359053.
Verified all 360 case identities, source hashes, complete real/modeled settings,
policy probabilities and statuses. Recomputed 120 distinct reference problems
and checked all three root-budget copies, losses and maximum Q errors.

| Root simulations | Cases | Strict errors | Maximum first-action loss | Maximum Q error |
| --- | --- | --- | --- | --- |
| 1,000 | 120 | 4 | .954069 | 14.128477 |
| 10,000 | 120 | 2 | .168457 | 4.683650 |
| 50,000 | 120 | 0 | 0 | 4.200977 |

All six errors exceed .020. This threshold is descriptive here, not a newly
declared validation gate. H4 errors increase from 1 at 1k to 2 at 10k before 0 at 50k.
Budget changes also change the deterministic search seed identity. Zero loss
at 50k is finite development agreement, not a convergence proof or accurate-Q
certificate. Loss measures one action followed by optimal continuation; it
does not measure the planner's full episode return.

Opponent budgets 25 versus 100 change 6/60 optimal action sets and 42/60 Q vectors
above 1e-4 (maximum change 2.723554). These are different modeled policy laws;
budget 100's zero errors do not demonstrate a better protagonist algorithm.

Artifacts: results/oracle/l2_h45_n{25,100}_b{0.085,0.5,0.915}_20260925/
and sibling logs/timers. Independent audit:
results/oracle/l2_h45_review_20260925.json.
Raw artifacts are local and Git-ignored; documentation commits do not archive
case data. Preserve raw runs with manifests when transferring evidence.

## Timing and interpretation

Worker endpoints match durations, at most two workers overlap, and every panel
passes the monotonic concurrency bound. External elapsed totals 454.09s versus
parent monotonic 478.198078s. Differences of 2.99–5.03s remain unexplained.
Keep both measurements; no precise speed comparison follows. Case wall time
includes planner and exhaustive reference. Future timing work should record
clock endpoints and execution provenance before attributing discrepancies.

## Next phase: H6/H8 development, fixed before launch

Codex ran six feasibility cases: protagonist H6/H8, root 50k, opponent
depth 20/budget 25, b_j=.5, own beliefs .05/.5/.95, seed 500. All completed with
zero first-action loss. Maximum Q errors were 7.042874 at H6 and 6.630328 at H8.
Worker durations were 7.09–8.36s at H6 and 11.94–15.29s at H8; peak monitored RSS
was 85.78 and125.63 MiB. Parent monotonic concurrency checks pass. This small
pilot checks resources and is development evidence, not a validation sample.

Pilot artifacts: results/oracle/l2_h{6,8}_resource_pilot_20260925/ and sibling logs.

Antigravity may run the following six sequential panels, 20 cases each, 120
total. Cross opponent budgets 25/100 and b_j=.085/.5/.915; protagonist H6/H8,
root 50k, seeds500–501, own beliefs .05/.075/.5/.925/.95. Keep repeated pilot
cases in the declared grid but do not count them as independent fresh evidence.
Root: empirical Bellman, exact tail, bounded c=1, gamma=.95.
Opponent: fixed depth 20, sampled backup/tail, normalized c=1, epsilon1e-6.
No adaptive budget increases or silent reference approximations. Failures,
including reference branch-budget limits, remain results to report.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
for budget in 25 100; do
  for belief in 0.085 0.5 0.915; do
    out=results/oracle/l2_h68_n${budget}_b${belief}_RUN_ID
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${out}.time" \
      uv run python -m examples.experiments.planner_oracle_experiment \
      --out "$out" --level 2 --opponent-depth 20 --opponent-belief "$belief" \
      --opponent-budget "$budget" --opponent-backup sampled \
      --opponent-exploration normalized --opponent-exploration-const 1 \
      --planners mcts --horizons 6 8 --budgets 50000 \
      --seed-start 500 --seeds 2 --beliefs 0.05 0.075 0.5 0.925 0.95 \
      --backup empirical_bellman --exact-final-step --exploration bounded \
      --exploration-const 1 --gamma 0.95 --workers 2 --timeout 240 \
      --max-rss-mb 2048 > "${out}.log" 2>&1 || exit 1
  done
done
```

Replace RUN_ID uniquely; start from a clean committed checkpoint. Preserve all
requested cases and failure records. Report per horizon, belief and opponent:
loss distribution, action sets/gaps, signed/absolute Q errors, external and
monotonic elapsed, worker intervals/sums and RSS. Separate opening and listening
regions. Do not call zero action loss value convergence or general optimality.

## Completed 120-case H6/H8 development study

Antigravity completed all six panels sequentially under source checkpoint e63db30
with RUN_ID `20260925`:
- `results/oracle/l2_h68_n25_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n25_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n25_b0.915_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.085_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.5_20260925/` (`.time`, `.log`, `timing.json`)
- `results/oracle/l2_h68_n100_b0.915_20260925/` (`.time`, `.log`, `timing.json`)

All 120 requested cases completed with status `complete` under two supervised
spawned workers with 240s timeout and 2,048 MiB memory limit. Zero timeouts,
crashes, or resource kills occurred.

### Timing and concurrency instrumentation audit

| Panel | GNU Elapsed | Parent Monotonic | Worker Wall Sum | Concurrency Bound | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 89.97 s | 95.08 s | 184.03 s | PASS (92.02 <= 95.08) | 132.08 MB |
| B_opp=25, b_j=0.500 | 83.42 s | 89.08 s | 172.46 s | PASS (86.23 <= 89.08) | 125.92 MB |
| B_opp=25, b_j=0.915 | 87.33 s | 93.02 s | 179.43 s | PASS (89.72 <= 93.02) | 132.38 MB |
| B_opp=100, b_j=0.085 | 98.75 s | 104.32 s | 202.57 s | PASS (101.29 <= 104.32) | 132.00 MB |
| B_opp=100, b_j=0.500 | 94.29 s | 99.97 s | 193.67 s | PASS (96.84 <= 99.97) | 129.29 MB |
| B_opp=100, b_j=0.915 | 101.73 s | 107.39 s | 208.76 s | PASS (104.38 <= 107.39) | 126.76 MB |
| **Total** | **555.49 s** | **588.87 s** | **1140.93 s** | **PASS (6/6)** | **132.38 MB** |

All six panels passed the monotonic concurrency bound. External GNU time (555.49s)
and parent monotonic elapsed (588.87s) differ by 5.11–5.66s per panel; both measurements
are preserved.

### Decision accuracy and policy loss summary

| Panel | Cases | Errors (loss > 1e-8) | Loss > 0.02 (diagnostic) | Mean Loss | Max Loss | Mean Max Q Err | Max Max Q Err | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B_opp=25, b_j=0.085 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.4836 | 7.6314 | 100.0% |
| B_opp=25, b_j=0.500 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 4.3732 | 7.0429 | 100.0% |
| B_opp=25, b_j=0.915 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.5872 | 7.1004 | 100.0% |
| B_opp=100, b_j=0.085 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.3236 | 8.5238 | 100.0% |
| B_opp=100, b_j=0.500 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 3.7174 | 6.2539 | 100.0% |
| B_opp=100, b_j=0.915 | 20 | 0 (0.0%) | 0 | 0.000000 | 0.000000 | 2.9572 | 6.2639 | 100.0% |
| **Total** | **120** | **0 (0.00%)** | **0 (0.00%)** | **0.000000** | **0.000000** | **3.5737** | **8.5238** | **100.0%** |

Across all 120 cases, **120 decisions (100.0%)** were strictly optimal (mean/max loss 0.000000).

### Action classification: uniform optimality of listening

- In all 120 cases at $H \in \{6, 8\}$, the Bayes-optimal oracle action is **`L` (Listen)**:
  - Opening cases (oracle prefers OL/OR): 0 / 120.
  - Listening cases (oracle prefers L): 120 / 120.
- With 6–8 steps of remaining horizon, information gathering dominates opening across all
  tested beliefs (including boundary beliefs $0.05$ and $0.95$). Protagonist MCTS accurately
  selected `L` across 100% of cases.

### Signed Q errors and deep nonterminal continuation

- Action `L`: Mean signed error is +0.298149, mean absolute error is 0.332974 (range [-0.304545, +1.071613]).
- Nonterminal door opening (`OL`, `OR`): Continuation search through 5–7 post-opening steps
  shows positive bias (+2.156583 on `OL`, +2.031731 on `OR`; max error 8.523833). This continuation
  bias does not impair decision accuracy because $Q^*(L)$ exceeds opening values by a wide margin.

### Opponent budget sensitivity (25 vs 100 on 60 distinct problems)

Comparing the 60 distinct $(H, b_j, \text{seed}, b_i)$ problems between $B_{\text{opp}}=25$ and $B_{\text{opp}}=100$:
- **0 action flips (0.00%)**: Optimal policy uniformly listens under both modeled budgets.
- **60 out of 60 Q vectors shift > 1e-4 (100.0%)**, with max Q shift 1.616884.
- This confirms that identical action choices do not imply invariant opponent policy distributions or values.

### Next steps: Codex independent review

Development evidence at H6/H8 ready for Codex review, manifest/hash verification against e63db30,
and recomputation of reference values.

## Remaining qualification

No solver source changed in this audit; latest implementation verification
remains 225 passing tests plus Ruff lint/format. Production defaults are unchanged.
A separately frozen fresh L2 validation design must follow development before
promotion. Earlier L1 passes and failures retain their source/configuration scope.

Modeled depth 20 is not protagonist depth 20. These point-prior L2 runs do not
qualify empirical nested hierarchies, mixed levels, L3-versus-L2 episode returns,
L4/all39 or the full production configuration. The runner's complete modeled
configuration can hash differently from other production configurations even
when nominal active budgets match. Full script review and long full-suite
qualification remain open; do not launch production L4/all39 from these results.
