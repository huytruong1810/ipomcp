# Tail-two fresh-gate failure: next diagnostic checkpoint

Status: completed by Antigravity and independently audited September 28.
Exact full-row replay and all source/input hashes pass. The 243-history trace is
exhausted; chance estimation dominates the signed error attribution. See the
latest BENCHMARK.md section and ROOT_CHANCE_PLAN.md. Commands below are the
historical protocol; preserve their completed artifacts.
Antigravity executes it in /home/andyj1810/projects/ipomcp on main. Do not clone,
change source, or rerun the completed validation launcher. This is post-failure
development analysis; the frozen 3,360-case gate remains FAILED.

## Fixed case and purpose

One case: panel0/case486, H8, seed8000, own belief .966, modeled budget25,
b_j=.085. Root100k, gamma .95, empirical Bellman, bounded c=1, exact history
rewards and tail depth two. Preserve every original setting and model identity.
First-action loss is .062316559843681674: selected OR versus oracle L.
The independently audited input includes the entire original row, not rounded Qs.

The current evidence gives OR/L ranking error +.065272689494 but no saved tree
for this new case. Trace weighted errors to distinguish chance estimation,
local action-ranking loss and rollout frontiers above the exact H2 boundary.
Do not attribute the failure to the large noncompeting OL error or infer that
increasing the budget or tail depth will fix it without testing regressions.

## Command

One worker, 600 seconds, 2048 MiB, at most 512 expanded trace histories. The
higher cap only affects post-search inspection, not the 100k search budget.
Use a new directory and preserve all failure/status records and external timing.

```bash
cd /home/andyj1810/projects/ipomcp
/usr/bin/time -v \
  -o results/oracle/l2_tail2_fresh_failure_diagnosis_20260928.time \
  .venv/bin/python -m examples.experiments.tree_diagnostics \
  --input results/oracle/l2_tail2_fresh_review_20260928/audit.json \
  --output results/oracle/l2_tail2_fresh_failure_diagnosis_20260928 \
  --tail-depth 2 --trace-nodes 512
```

Crucial: the diagnostic API's depth-two mode allows an explicit candidate and
asserts task identity rather than full row equality internally. Here the input
already uses tail two. Therefore **require exact full-row replay**, not merely
matching oracle/model metadata. Check this before interpreting any trace:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('results/oracle')
original = json.loads((root / 'l2_tail2_fresh_review_20260928/audit.json').read_text())['errors'][0]['row']
result = json.loads((root / 'l2_tail2_fresh_failure_diagnosis_20260928/case-0.json').read_text())
assert result['status'] == 'complete'
assert len(result['rows']) == 1
captured = result['rows'][0]
assert captured['exact_replay'] is True
assert captured['row'] == original
assert captured['baseline_row'] == original
assert captured['trace_policy'] == original['policy']
assert abs(captured['margin_trace']['reconstruction_residual']) < 1e-8
print('Exact full-row replay and trace reconstruction verified')
PY
```

Verify the diagnostic manifest source hashes match the frozen source contract
and its input hash matches the supplied audit file. Stop if replay differs;
preserve the result and investigate source/configuration identity rather than
changing seeds until it matches. Do not overwrite or silently retry a failed run.

## Required report and stopping point

Report root counts and OR-versus-L immediate/chance/continuation decomposition;
then signed total chance, local-loss, frontier, exact-boundary and unresolved
terms from the trace. Include both signed and absolute unresolved value error,
expanded/pending counts, and the largest weighted paths with remaining horizon,
visits, action counts, greedy choices, oracle choices and local loss. Group
contributions by remaining horizon. Check every inspected H2 exact boundary
against reference values. A zero unresolved value residual does not rule out
cancelling components within unexpanded histories.

Keep local loss and selected-action estimation error distinct. For a poor local
choice, inspect the estimated values/initialization of its competing alternatives,
not just the selected path; a bad rollout under an unselected alternative can
cause local action loss. Frontier error can mix rollout-policy loss and sampled
return noise, so do not rename it variance without further evidence.

Preserve external, Unix and monotonic timing and host workload/suspend provenance.
This replay includes reference diagnostics and is not a planning-cost comparison.
Return the artifacts for Codex review. No larger sweep, changed estimator, new
budget, increased tail depth or fresh validation gate is prescribed here.
