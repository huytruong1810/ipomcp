# Current review handoff

## Ownership and decision

Antigravity runs experiments; Codex owns code/math changes. Use only
/home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Do not change source during a run or make frozen source copies; Git is source
history. Raw artifacts remain local and Git-ignored unless separately archived.

The fresh held-out Level-1 validation suite has completed and **PASSED** the
frozen primary gate ($\epsilon_{\text{loss}} \le 0.020$) with **0 / 2,000 violations (100.0% pass rate)**.
Furthermore, all 2,000 cases achieved **0 strict errors** (100.0% exact first-action oracle
agreement; mean loss 0.000000, max loss 0.000000).
Previous 50k and 200k validation gate failures remain historical failures.
Production defaults remain unchanged pending promotion review.

## Verified fresh validation execution: 2,000 cases (PASSED)

Source checkpoint: a866e69. Evaluated candidate configuration: `backup="empirical_bellman"`,
`exact_final_step=true`, bounded UCB $c=1.0$, $\gamma=0.95$, fixed budget of 1,000,000
traversals across Horizons 1–5, 20 strictly unseen beliefs:
`0.005, 0.035, 0.065, 0.0725, 0.0775, 0.0825, 0.095, 0.125, 0.225, 0.375, 0.625, 0.775, 0.875, 0.905, 0.9175, 0.9225, 0.9275, 0.935, 0.965, 0.995`
and 20 fresh seeds: `1000–1019` (2,000 cases total). Zero overlap with previous development runs.

Raw cases, manifest, and logs:
- Results directory: `results/oracle/bellman_validation_20260924/`
- Direct GNU time log: `results/oracle/bellman_validation_20260924.time` (`elapsed_seconds=43392.04`)
- Execution log: `results/oracle/bellman_validation_20260924.log`

### Validation results summary

1. **Gate verdict**: **PASSED**.
   - Primary gate ($\text{loss} \le 0.020 + 10^{-8}$): **0 violations / 2,000 cases (0.00%)**.
   - Strict action agreement ($\text{loss} \le 10^{-8}$): **0 errors / 2,000 cases (0.00%)**.
   - Incomplete / killed / timed-out cases: **0 / 2,000 (0.00%)**.
   - Mean first-action policy loss: **0.000000**.
   - Max first-action policy loss: **0.000000**.
2. **Breakdown across Horizons (400 cases per horizon)**:
   - $H=1$: 0 errors, mean loss 0.000000, mean Q error 0.0000, mean wall 0.83 s.
   - $H=2$: 0 errors, mean loss 0.000000, mean Q error 0.0098, mean wall 19.50 s.
   - $H=3$: 0 errors, mean loss 0.000000, mean Q error 0.0108, mean wall 44.47 s.
   - $H=4$: 0 errors, mean loss 0.000000, mean Q error 1.3217, mean wall 68.80 s.
   - $H=5$: 0 errors, mean loss 0.000000, mean Q error 3.6684, mean wall 91.54 s.
3. **Action Category Breakdown**:
   - Optimal Listen ($N=1,080$): 0 errors, 100.0% exact agreement, loss = 0.000000.
   - Optimal Open ($N=920$): 0 errors, 100.0% exact agreement, loss = 0.000000.
4. **Computational Footprint**:
   - Directly measured elapsed panel wall time: **43,392.04 s (12.05 hours)** via GNU time.
   - Verified worker wall sum: **90,054.3 s (25.02 hours)**, average 45.03 s / case.
   - Monitored peak RSS: **113.42 MB** (well within 4,096 MB supervisor ceiling).

## Remaining engineering and research gates

No solver source changed in this review. Previous engineering checks remain
204 passing tests plus lint/format; those were not rerun for documentation edits.
Production defaults: sampled backups, sampled final steps, empirical-range UCB.

Still unresolved: matched L2 opponent horizon/computation semantics; long/deep
Tiger before/after performance; L4/all 39 conditions at intended resources;
finite-prior effects; remaining demo/visualization semantic review. L1 validation
cannot certify these or the default 25-simulation modeled-policy budget.

Keep HANDOFF.md, BACKLOG.md and docs/BENCHMARK.md consistent. Active instructions
are current-only; historical protocols belong in Git. Consult historical
METADATA_ERRATUM.md files when interpreting older modeled-budget fields.
