# Broader observed recursive chance comparison — September 29

Development comparison of 4,800 observed cases across three arms (empirical tail-two,
root-only exact chance, and recursive chance candidate). This is not held-out validation.
No defaults change follows.

## Frozen scope

Master contract: benchmarks/recursive_chance_development_20260929.json
SHA256: 7a1bb7bdf8e53a1d6c74664e66d94a572fa6132cced770c53e4ca7d9a1059c63

The twelve subordinate contracts are in
benchmarks/recursive_chance_development_20260929/panel-{0..11}.json.
Each stores complete historical rows and file hashes from the audited 4,800-case
development run. The master locks all twelve, the launcher and all 78 Python source files.

| Panel IDs | Historical cohort | Triples per panel | Total |
|---|---|---:|---:|
| 0–5 | September 26 tail-two development, H5/H6/H8, seeds 7000–7007 | 240 | 1,440 |
| 6–11 | September 27 formerly fresh failed gate, H1–H6/H8, seeds 8000–8007 | 560 | 3,360 |

Within each cohort the order is n25/b.085, n25/b.5, n25/b.915, n100/b.085,
n100/b.5, n100/b.915. Beliefs and all original settings are copied exactly.
Baseline: empirical tail-two (4 errors / 1 violation); root-only (5 errors / 1 violation).

## Resource and execution contract

One panel at a time, one supervised worker, 600 seconds and 2048 MiB RSS per
triple. Each triple computes fresh control, root-only, and recursive candidate searches,
with full historical control and root-only row identity, search-tree digest equality,
and task semantics checks.

First check inputs without starting an experiment:

```bash
cd /home/andyj1810/projects/ipomcp
.venv/bin/python benchmarks/run_recursive_chance_development.py --check-only
sha256sum benchmarks/recursive_chance_development_20260929.json
```

## September 30 continuation after panel 0 review

Antigravity created this exhaustive master from the draft intent and completed
panel0 before Codex's protocol review. Codex has now checked the existing master,
all twelve contracts, all 4,800 input rows/hashes, unchanged pilot solver sources,
and the full panel0 results. This adopts the preserved master as development
provenance; it does not retroactively make any cases held out. Do not regenerate
its hashes or rerun panel0. The existing master hash above remains unchanged.

Panel0: 240/240 optimal recursive choices, zero regressions; root-only case103
corrected. Seventy fully exact root subtrees agree with the reference. Other
roots retain approximation; 42,239 internal fallbacks and 7,530 frontiers were
verified. Maximum Q error worsens despite correct choices. Review artifacts:
results/oracle/l2_recursive_chance_development_p00_codex_review_20260930/.

Continue panels1–11 (4,560 triples, 13,680 root solves), sequentially. Solver and
three-arm runner are unchanged. Keep one worker, 600s and 2048 MiB per triple.
Panel0 used about3.23 external hours and813 MiB of disk. Remaining horizons and
opponent budgets differ, so do not multiply that duration as a reliable ETA;
plan for a multi-day run and preserve the full per-history telemetry. Filesystem
space was checked during review (926 GiB available in WSL); that is a snapshot,
not a reservation. Breaks are allowed between completed panels.

```bash
cd /home/andyj1810/projects/ipomcp
set -e
.venv/bin/python benchmarks/run_recursive_chance_development.py --check-only
for panel in {1..11}; do
  printf -v id '%02d' "$panel"
  base="results/oracle/l2_recursive_chance_development_p${id}_20260929"
  if [ -e "$base" ] || [ -e "${base}.freeze.json" ] || [ -e "${base}.time" ] || [ -e "${base}.log" ]; then
    echo "Existing attempt: preserve it and stop for review" >&2
    exit 1
  fi
  /usr/bin/time -v -o "${base}.time" \
    .venv/bin/python benchmarks/run_recursive_chance_development.py --panel "$panel" \
    > "${base}.log" 2>&1
done
```

The launcher preflights the whole collection and historical contracts on every
invocation. It records an exclusive freeze sidecar and refuses an existing
output/sidecar. If taking a break, restart with the first unstarted panel only;
do not repeat the loop from1 once that panel has completed. Existing partial or
failed outputs require review before retry. A provenance mismatch, replay/tree
inequality, timeout, RSS failure or exception stops execution. Do not increase
limits, repair beliefs, replace cases or overwrite logs. Accuracy regressions
alone do not justify stopping or retuning this fixed sweep: retain them and
complete coverage unless an operational/invariant failure occurs.

## Required audit and return

Return all twelve directories including preserved panel0, sidecars, .time and
.log files, and audit scripts/reports. Verify the unchanged master SHA256 and
complete 4,800-case Cartesian union, not merely the sum of file counts. All
23 pilot cases are included and should reproduce exactly; do not substitute
pilot outputs for their new three-arm cases.

For every case verify full empirical/root-only historical row identity, equal
three-arm tree digests, recursive search Q equal to empirical Q, unchanged task
and opponent settings, and greedy policies including ties. Recompute loss and
Q-error arithmetic. Report all three arms globally, by cohort, horizon, opponent
budget and opponent belief, including every correction/regression and all
recursive strict errors (>1e-8) or violations (> .020 +1e-8). Keep near-boundary
cases, mean/max loss and mean/max Q error visible. Zero selected-pilot errors
cannot hide broader regressions.

Audit the recursive per-history records: status counts, full legal-action gaps,
frontiers, fallback missing masses and propagated approximate-child masses;
distinguish local complete support from exact subtrees. Sum local counts only,
not missing masses across histories as though they were root probability.
Report coverage by remaining horizon and whether root/selected actions are
certified. Fully exact root Q vectors must agree with the finite reference to
roundoff; a material disagreement is an implementation issue. Identical digests
are evidence from the locked runner, not independently reconstructed full trees.

Check worker serial nonoverlap, parent elapsed bounds, limits, cross-panel
sequencing, RSS and per-arm policy times. Keep external, Unix and monotonic
clocks separate. Panel0 already has a53.638084s discrepancy between Unix and
parent monotonic elapsed; preserve it without hypothesizing a cause or repairing
logs. GNU-time and monitored RSS are different measurements. Report both pooled
and mean-pair timing ratios, including H1/H2 separately; fixed arm order and
load prevent interpreting these as causal overhead.

Codex audits before any fresh qualification or defaults change. The candidate
still has only conditional full-coverage Bellman exactness; approximate subtrees
are not certified by correct selected choices. Preserve all previous failed
gates, raw outputs, source contracts, defaults and the frozen tolerance.
