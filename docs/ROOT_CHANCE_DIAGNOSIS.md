# Root chance regression diagnosis — September 29

COMPLETED and independently audited. Preserve this historical protocol and its
artifacts; do not rerun. Next implementation design: docs/RECURSIVE_CHANCE_PLAN.md. Stay in /home/andyj1810/projects/ipomcp on main. Solver sources are
unchanged from the completed broad comparison. Do not retune seeds or tolerance.

Input: benchmarks/root_chance_diagnosis_20260929.json
SHA256: 6ccfe80e8bf0993ef1e5746f8d0b9277e66dfb56f8171b3c217cb8c91e20535d

Input order: P11/C556 (new violation), P00/C103 (regression), P03/C86 (residual),
P04/C117 (correction), P04/C122 (regression), P04/C127 (residual), P06/C486
(correction). Complete candidate and control rows, raw-pair hashes and source lock
are retained. Three regressions, two residuals and two corrections are all included.

Before running, verify the input hash, all source_sha256 entries and raw file
hashes against this input. The existing diagnostic CLI records source hashes but
does not enforce that input's source lock; stop on mismatch, never refresh hashes.
Preserve any existing output or timer; do not overwrite a partial attempt.

```bash
cd /home/andyj1810/projects/ipomcp
sha256sum benchmarks/root_chance_diagnosis_20260929.json
# Only run when the directory and .time path below do not exist.
/usr/bin/time -v -o results/oracle/l2_root_chance_diagnosis_20260929.time \
  .venv/bin/python -m examples.experiments.tree_diagnostics \
  --input benchmarks/root_chance_diagnosis_20260929.json \
  --output results/oracle/l2_root_chance_diagnosis_20260929 \
  --tail-depth 2 --trace-nodes 512
```

One supervised worker, 600 seconds and 2048 MiB per case. Preserve manifest,
full case JSON, logs, summary and timer. Require exact_replay=true and complete
row equality for all seven. The existing tail-depth-two diagnostic permits weaker
semantic equality for historical tail ablations, so a complete status alone does
not certify exact replay: explicitly audit this flag and row equality, and reject
interpretation on mismatch. Do not retry or increase limits without review.

## Correct interpretation

The diagnostic labels trace_value_source=empirical_search_q. Its margin_trace
uses the candidate's chosen policy against the oracle-best policy but decomposes
the EMPIRICAL search Q vector. It does not directly decompose recommendation Q.
For each case compute the root scoring increment separately:

    delta = sum_a (pi_candidate[a] - pi_oracle_best[a])
                  * (estimated_q[a] - search_estimated_q[a])

The candidate signed margin error equals the empirical trace target plus delta.
Use uniform exact ties for pi_oracle_best. Corrections are optimal at the root,
so their candidate-versus-optimal trace coefficients can be zero. Retain their
root action branch errors and scores; do not report a zero trace as proof that
all internal estimates are exact. No need to run another search to obtain control
Q: the paired outputs already preserve it and verify tree equality.

For regressions/residuals, report signed immediate, chance, local-action loss,
rollout-frontier, exact-tail and unresolved terms, plus delta. Show contributions
by remaining horizon and the largest weighted histories. Do not sum propagated
errors across depths (double-counting), or call a signed decomposition a variance
or causal proof. Check pending histories and residuals; 512 is a trace budget,
not a guarantee of full expansion. Preserve truncation instead of treating it as
zero. Report root probability mismatch separately from child estimation error.

Return artifacts to Codex. No new solver change, adaptive blend, tuned margin
threshold or fresh validation is prescribed until these traces are reviewed.
All defaults, historical failed gates and .020 +1e-8 remain unchanged.
