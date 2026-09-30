# Broader observed root chance comparison — September 28

COMPLETED and independently audited September 29. Do not rerun. Next instructions
are docs/ROOT_CHANCE_DIAGNOSIS.md. This is a development comparison of 4,800
already observed cases, not held-out validation. No defaults change follows.
Codex prepared and checked the contracts without launching any experiments.

## Frozen scope

Use the existing /home/andyj1810/projects/ipomcp checkout on main. Read HANDOFF.md
and the latest docs/REVIEW.md first. The audited pilot corrected two of four
selected errors, with no control regression; this panel tests whether that
improvement survives exhaustive observed coverage. Do not tune the candidate
or drop/reorder cases in response to results.

Master: benchmarks/root_chance_development_20260928.json
SHA256: 600caaae9425d12c225f3b96bcd79ed842ee39f201eaf616d1ca04b22e4a9927

The twelve subordinate contracts are in
benchmarks/root_chance_development_20260928/panel-{0..11}.json.
Each stores complete historical rows and file hashes; the master locks all twelve,
the launcher and all 76 Python source files. Solver, evaluator and paired runner
are unchanged from the audited pilot. The pilot and failed gate contracts are
preserved with their original hashes.

| Panel IDs | Historical cohort | Pairs per panel | Total |
|---|---|---:|---:|
| 0–5 | September 26 tail-two development, H5/H6/H8, seeds7000–7007 | 240 | 1,440 |
| 6–11 | September 27 formerly fresh failed gate, H1–H6/H8, seeds8000–8007 | 560 | 3,360 |

Within each cohort the order is n25/b.085, n25/b.5, n25/b.915, n100/b.085,
n100/b.5, n100/b.915. Beliefs and all original settings are copied exactly.
The union has 4,800 distinct opponent/horizon/seed/belief cases. All 14 pilot
cases are included again; do not substitute pilot outputs for new pairs.
Baseline: four strict errors (>1e-8), one violation (> .020 +1e-8), mean loss
.0000163930211153. Root budget100k, gamma.95, exact history rewards, tail depth2,
empirical Bellman, bounded c1; modeled depth20 and sampled normalized c1 remain
unchanged. Only final root chance scoring differs between arms.

## Resource and execution contract

One panel at a time, one supervised worker, 600 seconds and 2048 MiB RSS per
pair. Each pair computes fresh control and candidate searches and references,
with full historical control-row identity, search-tree digest equality, raw Q
identity and task/reference checks. Control runs first, as in the pilot.
9,600 root solves are prescribed. This is a many-hour run: twice the historical
single-arm worker totals is about 26 hours before paired fingerprint overhead
and differences in load/clocks; that is a planning scale, not an ETA or bound.
Do not raise worker count or limits silently. Avoid concurrent panels. Record
other substantial workloads; preserve all measured clock domains without repair.

First check inputs without starting an experiment:

```bash
cd /home/andyj1810/projects/ipomcp
.venv/bin/python benchmarks/run_root_chance_development.py --check-only
sha256sum benchmarks/root_chance_development_20260928.json
```

Then run the twelve panels in order, stopping on the first failure:

```bash
cd /home/andyj1810/projects/ipomcp
set -e
for panel in {0..11}; do
  printf -v id '%02d' "$panel"
  base="results/oracle/l2_root_chance_development_p${id}_20260928"
  if [ -e "${base}.time" ] || [ -e "${base}.log" ]; then
    echo "Existing telemetry: preserve this attempt and stop" >&2
    exit 1
  fi
  /usr/bin/time -v -o "${base}.time" \
    .venv/bin/python benchmarks/run_root_chance_development.py --panel "$panel" \
    > "${base}.log" 2>&1
done
```

Every invocation rechecks the entire input collection before starting its panel.
The launcher writes an exclusive .freeze.json sidecar before dispatch, records
the full contract, its hash, base Git commit and dirty status, and refuses an
existing output or sidecar. The unchanged worker runner creates its own manifest,
raw case files, logs and final summary. External time includes launcher preflight;
runner monotonic time starts after its own preflight. Preserve both.

A planned break is allowed between completed panels. Continue with the next
unstarted panel ID using the same command and unique telemetry paths; do not
rerun completed panels. Partial outputs, sidecars or failed pairs require Codex
review before retry; no automatic resume, overwriting, timeout exclusions or
replacement cases. A timeout, RSS failure, replay mismatch, changed hash, or other
exception stops the sequence. Record its reason and every existing artifact.
Accuracy regressions alone do not justify stopping or retuning this fixed sweep:
complete the declared comparison unless an operational/invariant failure occurs.

## Audit and return requirements

Return all twelve directories, .freeze.json sidecars, .time and .log files, plus
an audit script/report. Cite the actual master hash from the files. Preserve raw
rows rather than only aggregates. The following are required:

- Verify all 4,800 pairs, twelve panel sizes and exact historical Cartesian
  coverage. Check source/contract/input hashes, full control-row equality, task
  semantics and recorded tree/Q invariants. The saved common tree digest and
  equality flags attest the locked runner's comparison; they are not two saved
  full trees independently reconstructed by the audit.
- Recompute greedy policies including exact ties, losses, Q errors and margins.
  Report control/candidate totals, both cohorts separately, and strata by horizon,
  opponent budget and opponent belief. Use strict loss >1e-8 and violation
  > .020 +1e-8; never change tolerance, slack or tie rules after the run.
- List every correction and regression, every candidate strict error/violation,
  and any numerical boundary case. Compare the four known baseline errors and
  the fourteen pilot cases explicitly. Report mean/max loss and Q error as well
  as counts; improvement on known errors cannot hide newly regressed controls.
- Count coverage statuses for all actions, missing mass and branches, and whether
  a fallback affects the chosen or reference-best action. Keep exact-tail bypasses
  separate. A complete case with fallback remains part of the analysis, not an
  exclusion or a claim of exact chance weighting.
- Check serial worker interval nonoverlap, worker sum versus parent monotonic,
  panel sequencing, limits and RSS. Retain Unix and external clocks and flag any
  disagreement. Report per-pair timings, pooled ratios and mean pair-relative
  ratios; separate H1/H2 bypass cases. Control-first ordering and system load
  prevent a causal overhead claim even if clocks agree. Existing pilot clock
  discrepancies remain unresolved.

After completion Codex audits before prescribing any fresh qualification.
No claim of a passed fresh gate is available from this observed union, even
with zero violations. Preserve the tail-two FAILED gate and all older failures.
Defaults and the frozen loss tolerance remain unchanged.

## Engineering checks

All 4,800 full rows and input hashes, twelve contract hashes, source lock and
Cartesian coverage checked without running experiments. Eighteen focused tests
passed, including eight new launcher provenance/overwrite tests and ten existing
root scoring/pairing tests. Solver sources are unchanged; the prior full suite
remains 268 passing tests, with eight additional launcher tests now added.
