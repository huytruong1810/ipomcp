# Current review handoff

## Ownership and decision

Antigravity runs experiments; Codex owns code/math changes. Use only
/home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Do not change source during a run or make frozen source copies; Git is source
history. Raw artifacts remain local and Git-ignored unless separately archived.

The larger-budget evidence supports selecting empirical Bellman for fresh L1
validation. Production defaults remain unchanged. The numerical loss tolerance
has been explicitly selected by the user and frozen at **epsilon_loss = 0.020**.
The primary validation gate requires every requested case to satisfy
loss <= 0.020 + 1e-8. No fresh cases have been evaluated yet.

## Independent review of 33c3356

Verified all 480 cases: complete/unique coverage, settings, source hashes against
66f9086, and summary status. Recomputed exact L1 oracle Q values at all 24
horizon/belief pairs, policy losses and maximum absolute Q errors. This checks
the report against the existing reference, not a new proof of the reference.

Sampled: 60/240 errors, mean loss .465996077, mean max Q error 44.811826.
Candidate: 14/240 errors, mean loss .000852632, maximum loss .018135728,
mean max Q error 2.472959. All three H5/50k material errors disappear at both
200k and 1M in the inspected seeds. At 1M, candidate H4 has 1/60 errors and
H5 has 3/60. These are finite-budget results, not convergence proofs.
H5 mean max Q error rises from 3.317310 at 200k to 3.490327 at 1M despite
fewer action errors: correct action selection and accurate Q values differ.

Raw panels:
- results/oracle/bellman_budget_sampled_20260924/
- results/oracle/bellman_budget_empirical_20260924/

Independent audit: results/oracle/bellman_budget_review_20260924.json.

The previously reported elapsed panel times 4818.7/5708.4 seconds are invalid:
verified worker sums are 10152.416/12064.170 seconds, so two workers require
at least 5076.208/6032.085 seconds elapsed. Actual elapsed is unavailable from
the saved cases. The verified aggregate worker-time ratio is 1.1883, not an
independently verified elapsed-time ratio or isolated backup overhead.
Peak monitored RSS is 107.418/113.387 MiB. Preserve raw data; correct reports.

## Frozen fresh validation protocol -- epsilon_loss = 0.020

Candidate: backup=empirical_bellman, exact_final_step=true, bounded UCB c=1,
gamma=.95, one fixed budget of 1,000,000 traversals at each H1-H5. This selects
the largest tested budget for additional action accuracy; it is not proven
compute-optimal, and no equal-runtime comparison is claimed.

Use twenty beliefs:
.005 .035 .065 .0725 .0775 .0825 .095 .125 .225 .375
.625 .775 .875 .905 .9175 .9225 .9275 .935 .965 .995
and seeds 1000-1019. Total: 5 horizons x 20 beliefs x 20 seeds = 2000 cases.
They cover both tails, neighborhoods of inspected boundaries and interior
beliefs. This is a fixed stratified grid, not a random sample of all beliefs.
No proposed belief or seed occurs in 31 existing panel manifests or 12,513
recorded case rows checked locally. This does not certify unrecorded runs;
if Antigravity knows of any overlap, disclose it before starting. Do not
evaluate proposed cases during preparation.

Primary criterion: every requested case completes and has first-action loss
<= 0.020 + 1e-8. The numerical loss bound is explicitly frozen at **0.020** per
user authorization. Loss = V*(b) - sum_a pi(a|b) Q*(b,a), with optimal continuation
after the first choice. This is not complete-policy/episode regret. Report strict
action errors, maximum and mean loss, all signed Q errors and per-belief/horizon/seed
results. Q errors are diagnostics rather than an undisclosed secondary gate.

Freeze the number, this configuration, exact source HEAD and all case choices
before executing. Run once: no adaptive budget increases, early-success stopping,
or substitution of failing cases. Any failed case or resource kill fails the
panel; retain its evidence. No validation rerun after tuning remains held-out.
A pass applies only to this L1 finite panel and resource budget. Earlier gates
remain failures and production promotion/deeper qualification stay separate.

Once the numerical threshold is recorded, use an empty uniquely named output
directory. Measure elapsed time directly with GNU time; do not infer it from
first/last completion times or subtract startup from worker measurements.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
mkdir -p results/oracle
/usr/bin/time -f 'elapsed_seconds=%e' -o results/oracle/bellman_validation_RUN_ID.time \
  uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_validation_RUN_ID \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup empirical_bellman --horizons 1 2 3 4 5 --budgets 1000000 \
  --gamma 0.95 --seed-start 1000 --seeds 20 \
  --beliefs 0.005 0.035 0.065 0.0725 0.0775 0.0825 0.095 0.125 0.225 0.375 0.625 0.775 0.875 0.905 0.9175 0.9225 0.9275 0.935 0.965 0.995 \
  --workers 2 --timeout 2400 --max-rss-mb 4096 \
  > results/oracle/bellman_validation_RUN_ID.log 2>&1
```

Two 4-GiB RSS ceilings need parent/OS headroom. They are monitored limits,
not reservations. Capture failures and preserve original manifests unchanged.
If an operational interruption occurs, report the incomplete attempt rather
than silently recreating an apparently uninterrupted validation panel.

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
