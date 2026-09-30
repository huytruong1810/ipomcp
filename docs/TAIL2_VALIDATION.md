# Frozen fresh L2 two-step-tail validation — September 27, 2026

Status (September 28): **completed and independently audited; primary gate FAILED**.
All 3,360 cases completed; one loss .062316559843681674 exceeds .020 +1e-8.
See BENCHMARK.md and HANDOFF.md. Preserve the frozen contract and completed
attempts. The execution instructions below are historical; do not rerun the gate.
Antigravity executes experiments. Codex reviews the resulting evidence. Defaults
remain unchanged. Historical ac3df1c and f4d2aef gates remain failed.

## Fixed design and criterion

The immutable machine-readable contract is
`benchmarks/tail2_validation_20260927.json`. It binds the complete settings of
six panels and every current source hash, plus the launcher hash. The launcher
refuses changed source and refuses existing attempt directories/sidecars. Source
is the hashed working tree on main based on 62d6cd8; that commit alone does not
contain the candidate. Commit-only provenance is insufficient for these runs.

- 3,360 cases: six opponent conditions x seven horizons x eight seeds x ten beliefs.
- Opponent conditions: budget 25/100 crossed with b_j=.085/.5/.915; fixed depth20,
  sampled backup, normalized c=1, no exact opponent tail or history rewards.
- Protagonist: H1/H2/H3/H4/H5/H6/H8, root 100,000 traversals, gamma=.95,
  empirical Bellman, bounded c=1, exact history rewards, exact final step and
  exact_tail_depth=2. All root and modeled settings remain in the contract.
- Fresh bank seeds: 8000–8007. Fresh own beliefs:
  .004, .014, .024, .034, .084, .916, .966, .976, .986, .996.
- Per panel: 560 cases, two workers, 240 seconds per case, 2048 MiB per worker.
  Run panels sequentially; do not multiply total concurrency by launching them
  together. Record competing host workloads; do not claim isolated timings.

Primary acceptance requires every declared case complete and finite, matched
model/physics/source provenance, and **zero** first-action losses exceeding
.020 +1e-8. Keep the tolerance and numerical slack separate and unrounded.
Strict errors (loss >1e-8), signed/per-action Q errors, maximum Q error, and
costs are secondary reports; perfect strict agreement is not the frozen primary
criterion. First-action loss means optimal continuation after the returned
first action, not full episode regret.

Timeouts, memory failures, missing/duplicate rows, or provenance mismatches
preclude a qualification claim. Preserve attempts and failures. Do not silently
increase limits, replace rows, retune seeds/beliefs, exclude hard cases, or reuse
observed errors as fresh validation. If an infrastructure retry is needed,
review and record a separate attempt without erasing the original failure.
A primary accuracy violation is a failed gate even if later code fixes that case.

Novelty audit scanned 28,300 local oracle JSON files, including 91 settings
manifests and 28,221 row records (copies included). None of the proposed own
beliefs or bank seed indices appeared. The supporting audit is
`results/oracle/l2_tail2_review_20260927/novelty.json`. This establishes local
artifact novelty, not knowledge of unrecorded external experiments. No reference
or planner solve was performed at any proposed validation belief/seed.

## Antigravity execution (completed; historical instructions)

Use the existing checkout and environment, not a clone. Do not edit solver
source or this frozen design during execution. The runner writes a separate
freeze sidecar with contract hash, commit and working-tree status before each
panel, and the existing supervisor preserves all row outcomes and interval data.

```bash
cd /home/andyj1810/projects/ipomcp
set -euo pipefail
for panel in 0 1 2 3 4 5; do
  /usr/bin/time -v \
    -o results/oracle/l2_tail2_fresh_20260927_panel${panel}.time \
    .venv/bin/python benchmarks/run_tail2_validation.py --panel "${panel}" \
    > results/oracle/l2_tail2_fresh_20260927_panel${panel}.log 2>&1
done
```

Panel order is (25,.085), (25,.5), (25,.915), (100,.085), (100,.5), (100,.915).
Output directories are named in the frozen contract. Do not overwrite external
timers/logs if an attempt already exists. Record machine, UTC start/end, concurrent
workload, Python version and execution limits with the attempt. The frozen
launcher performs no tuning, automatic restart or automatic default promotion.

After completion, report exact Cartesian coverage, all completion/failure
statuses, source hashes, complete modeled/root configurations, normalized greedy
policies, independently recomputed loss and Q-error arithmetic, and all violations
without rounding. Give overall/per-panel/per-horizon first-action metrics and
signed Q-error distributions. Compare worker intervals against the actual worker
limit: sum(worker durations) <= 2 * parent monotonic elapsed, and observed
simultaneous intervals <=2. Sum(worker durations) >= parent elapsed is not the
relevant concurrency certificate. Preserve external elapsed, parent monotonic
and worker durations separately, including any discrepancy and peak worker RSS.

Fresh seeds also define new finite modeled policies. Do not treat validation
rows as paired controls of the old seeds or infer universal behavior from this
finite grid. A successful result qualifies only this declared distribution and
configuration. Default promotion and all untested horizons/levels require their
own decisions and evidence.
