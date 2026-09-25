# Current review handoff

## Decision and ownership

Antigravity runs experiments; Codex owns code/math changes. Use the single
checkout `/home/andyj1810/projects/ipomcp`, branch `fix/tiger-policy-inversion`.
Do not change source during a run or create frozen source copies. Git holds
source history; raw results stay under results/ with source manifests.

Current task: broader **development** testing of empirical chance-weighted Bellman
backups against sampled means. Both use exact final-step evaluation. No production
default is promoted, and no new held-out validation or full-suite run is authorized
by these development results. Previous 50k and 200k validation gates remain failed.

## What was verified

The 1,440-case validation at ccf99b6 is independently checked: complete requested
coverage, source hashes, settings and recomputed loss. Exact-tail candidate failed
80/720 decisions, sampled-tail control 85/720. These beliefs/seeds are now
explicitly development data. H1 and H2 have different optimal action sets: .085
and .915 favor opening at H1 and listening at H2. Twelve sampled points do not
constitute an exact global boundary map.

New estimator source checkpoint: 2230786. Configuration is explicit:
`MCTSConfig.backup="empirical_bellman"`, independently of `exact_final_step`.
At each private-history decision node, maximize currently evaluated action values.
At chance branches, average current child values by observed nonterminal outcome
counts divided by **all** action visits. Terminal visits retain zero continuation
in that denominator. Explicit rollout estimates initialize unevaluated frontiers;
there is no implicit zero or maximization across hidden states or observations.
See docs/THEORY.md for the formula, prior work and limitations.

A 112-case development comparison at H2/H3 completed. With exact tail enabled
in both variants, H3 empirical Bellman had 0/14 errors at 10k and 50k; sampled
means had 8/14 at each budget. At 50k mean max Q error was .033812 versus 17.635496.
A further 20-case H4/H5 smoke panel retained near-tie errors (loss about .010-.018).
This is promising development evidence, not global optimality or a held-out pass.
Raw evidence is under `results/bellman-backup-20260924/`; see its REPORT.md and
analysis.json, plus docs/BENCHMARK.md.

Raw runs/manifests remain local and Git-ignored. Commits such as e7f27ff contain
documentation, not those raw artifacts. Do not claim a Git/remote archive exists
unless actually created. Preserve captured manifests and failed cases unchanged.

## Engineering verification

At source checkpoint 2230786, all 200 non-integration tests passed in 49.13 s
and all four domain integration tests passed in 265.32 s (204 total). Ruff lint
and formatting passed across 106 Python files. The follow-up changes only clarify
an MCTS-only option error message and documentation; planning behavior is unchanged.
These tests and the L1 panels do not certify L2/L3/L4 optimality or full-suite
readiness. Raw integration output is in
results/bellman-backup-20260924/integration-tests.log.

## Current runner protocol

Compare `sampled` and `empirical_bellman` with bounded c=1 and exact tail enabled
in **both**. Use H1-H5, budgets 1k/10k/50k, the twelve inspected beliefs below,
and seeds 200-204. Run panels sequentially, two supervised workers within each.
Expected coverage: 900 cases per mode, 1,800 total. This is development coverage,
not a new untouched validation set and not a relaxed version of either old gate.

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_development_sampled_RUN_ID \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup sampled --horizons 1 2 3 4 5 --budgets 1000 10000 50000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_development_empirical_RUN_ID \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup empirical_bellman --horizons 1 2 3 4 5 --budgets 1000 10000 50000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Replace RUN_ID with a unique identifier. Require a clean tree, record HEAD and
keep source/configuration fixed throughout both modes. Report every requested
case, all failures, per-belief/per-horizon oracle actions and gaps, chosen policies,
first-action losses, all Q errors, per-case wall/RSS and actual elapsed panel time.
Separate opening from listening cases; do not hide near-tie errors or average
away a failing belief. Use 1e-8 only for numerical loss comparison, not as a
scientific reward-equivalence margin. The user selected bounded first-action loss
for future validation; the numerical maximum is still pending. Define it per case
as max_a Q_oracle(a) - Q_oracle(chosen action), and freeze the threshold before
fresh validation. Do not infer a threshold from the observed .010-.018 losses.
The development panel can proceed while that number is clarified; no validation
pass can be declared without it. Previous failed gates cannot be changed.

Two 4-GiB sampled RSS ceilings need additional memory for parent and OS. They are
not instantaneous allocation reservations. Keep timeouts/resource kills as failed
cases. Extra child-value aggregation and finite posterior integration count as
computation; equal traversals do not imply equal runtime.

## Remaining gates and maintenance

Production defaults remain `backup="sampled"`, `exact_final_step=false`, and
empirical-range UCB. New config fields change deterministic search hashes even at
default values; old-source trajectories are not a bitwise control. Both modes must
use the same current checkpoint. Equal seed indices are not a promise of common
random numbers across configurations.

Still unresolved: matched L2 opponent horizon/computation semantics, long/deep
Tiger before/after performance, L4 and all 39 conditions at intended resources,
finite-prior effects and remaining demo/visualization review. L1 progress does not
qualify deeper agents or the default 25-simulation modeled-policy budget.

Update this current handoff, BACKLOG.md and docs/BENCHMARK.md together. Historical
instructions belong in Git, not an accumulating active handoff. Read prior audit
METADATA_ERRATUM.md files when interpreting old modeled-budget fields.
