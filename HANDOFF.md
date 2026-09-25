# Current review handoff

## Ownership and status

Antigravity runs experiments; Codex owns code/math changes. Work only in
/home/andyj1810/projects/ipomcp on fix/tiger-policy-inversion. Use Git for source
history; do not create source copies or edit source during experiments.

Next task: a fixed larger-budget development comparison at H4/H5. The completed
1,800-case panel improved accuracy but does not qualify production defaults.
Earlier 50k/200k gate failures remain historical failures regardless of future
results. No fresh validation protocol has yet been frozen.

## Independent review of 933d917

All 1,800 raw cases were checked for complete/unique coverage, status,
row/configuration consistency and source hashes against 7d8b6bd. Oracle Q values
were recomputed from the exact L1 solver for all 60 horizon/belief combinations,
and all policy losses and maximum absolute Q errors were recomputed. This
checks recorded evidence against the existing reference; it is not a new,
independent proof of that reference's theory.

Sampled backups: 206/900 errors, mean loss 0.3643336044.
Empirical Bellman: 89/900 errors, mean loss 0.0222665597.
At H3/50k the candidate has 0/60 errors. That is finite-panel agreement,
not a general convergence or boundary-resolution proof.

At H5/50k, 13/60 candidate decisions are wrong: ten have loss about 0.018136,
but three have loss 0.53094248946:
- p=.07, seed 202;
- p=.93, seeds 201 and 202.

Thus even the proposed, unapproved per-case bound 0.020 would fail three cases.
Do not describe all remaining errors as near ties. At H4/50k, the five errors
have loss about 0.009924; lower budgets retain larger errors.
The selected candidate can overvalue listening; this is not a certified
conservative policy or a proof of the cause of each error.

Raw panels:
- results/oracle/bellman_development_sampled_20260924/
- results/oracle/bellman_development_empirical_20260924/

Independent audit: results/oracle/bellman_development_review_20260924.json.
These artifacts are local and Git-ignored; documentation commits do not archive
the raw evidence. Preserve manifests and cases unchanged.

## Next runner protocol: larger-budget development

Compare both backups with exact final-step integration, bounded c=1, gamma=.95.
Use H4/H5, 200k/1M traversals, all twelve previously inspected beliefs and
seeds 200-204. There are 240 cases per mode, 480 total. Run modes sequentially,
two supervised workers per mode. These are development cases, not held-out data.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_budget_sampled_RUN_ID \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup sampled --horizons 4 5 --budgets 200000 1000000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_budget_empirical_RUN_ID \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup empirical_bellman --horizons 4 5 --budgets 200000 1000000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Replace RUN_ID with a unique identifier. Require a clean tree and unchanged source
through both panels. Do not change budgets or settings after inspecting partial
results. Resource kills/crashes/timeouts remain failures. Two 4-GiB RSS ceilings
require memory headroom for parent/OS; they are monitored limits, not reservations.

Report complete coverage, per-belief/horizon/budget errors, oracle action gaps,
all three signed Q errors, first-action loss distributions and maxima, wall time
and RSS. Explicitly track the three H5/50k failures above. Report policy accuracy
separately from Q accuracy and opening cases separately from listening cases.
Compare cost as well as traversal counts: neither equal traversals nor equal
seed indices imply equal work or common random numbers. Record panel elapsed
time from a timer/log, not the sum of worker times.

The question is whether added computation reduces the larger H5 errors, and at
what cost. If errors persist, inspect intermediate action visitation and
estimated values before changing the estimator. Do not tune an acceptance
threshold to make observed cases pass.

## Future validation criterion

The user requested bounded first-action loss; its numerical maximum is pending.
Antigravity suggested 0.020, but quoting that suggestion is not user acceptance.
The definition is V*(b) - sum_a pi(a|b) Q*(b,a), per case. It assumes optimal
continuation after the first action and is not whole-episode policy regret.
A numerical comparison allowance (currently 1e-8) is distinct from a scientific
loss tolerance. Freeze the numerical loss bound, candidate configuration,
resource budget and fresh belief/seed set before executing future validation.
A finite grid pass cannot certify every belief or all future random seeds.

## Engineering and remaining qualification

No solver changes in this evidence review. Source behavior remains 7d8b6bd:
204 tests previously passed, with final lint/format and focused checks passing.
Production defaults remain backup="sampled", exact_final_step=false and
empirical-range UCB. The optional combination has not been promoted.

Still unresolved: matched L2 opponent horizon/computation semantics, long/deep
Tiger before/after performance, L4/all 39 conditions at intended resources,
finite-prior effects and remaining demo/visualization review. L1 progress does
not qualify deeper agents or the default 25-simulation modeled-policy budget.

Keep HANDOFF.md, BACKLOG.md and docs/BENCHMARK.md consistent. Historical active
instructions belong in Git. Read METADATA_ERRATUM.md when using older run data.
