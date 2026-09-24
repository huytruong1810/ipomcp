# Current review handoff

Current action: run the fixed held-out L1 validation at the end of this file.
Candidate: bounded c=1; production default unchanged. Do not rerun completed
calibration panels or launch the full suite.

Authoritative checkout: `/home/andyj1810/projects/ipomcp`. Antigravity's refactor
was committed at `8a96b85` before this pass. No uncommitted source was overwritten.
The September 23 review repairs reference planning, removes unsupported search
pruning, restores intentional-model evolution in rollouts, and consolidates the
oracle experiments. See `docs/REVIEW.md` for the current phase ledger and findings.

## Verified checkpoint

Engineering commit: `9e40997`. Planning/source checkpoint for the quantitative
panels: `4e9ff4c`; all recorded after-run source hashes match that commit.
The before-run source hashes match Git checkpoint `35303a5`. Source copies are
unnecessary: use Git history. Current checks: **165 tests passed** (161 unit and
4 domain integration), lint/format checks and a clean wheel build. The rebuilt
wheel excludes removed modules; generated build/extraction source copies were
removed after verification. Worker tests run without the earlier fork warning.

## Scientific status

The full long suite is **not qualified as an optimal-planning study**. Correct
Bayesian conditioning, successful execution, positive rewards, and optimal control
are different claims. Neither a confidence threshold nor a favorable payoff
establishes dominance, identification of reasoning level, or convergence.

The former L2 reference maximized independently over hidden states, returning
+8.5 at a uniform belief with two remaining steps. The correct observation-based
value is -1.95. Its rounded belief grid, nearest-belief substitution, and missing
continuation defaults are removed. The replacement retains joint physical-state /
opponent-belief uncertainty and conditions future choices on own observations.

MCTS now explores all legal tree actions and advances opponent private beliefs
during rollout transitions. Only a final reward-only step omits a useless update.
The Tiger rollout's physical memory is explicitly a uniform-L0 heuristic, not an
interactive posterior or an action-pruning certificate. No true hidden state or
opponent action is provided to that memory update. Impossible evidence raises.

The matched oracle suite is `examples.experiments.planner_oracle_experiment`.
It compares L1 versus uniform L0, with identical physical beliefs, horizons,
discount and sensor law. RTS retains all nine observation tokens. It records
per-case source hashes, policies, all action values, first-action loss, wall/RSS
cost and failures. MCTS simulations and RTS particles per branch are different
work units. L2 reference tests do not certify L3/L4 or a different horizon model.

## Evidence and next work

Current raw runs and verification logs: `results/review-20260923/`.
`before-planner` and `after-planner` are 30 seeds, 20 steps, depth five,
20,000/15,000 real simulations and 2,000/1,500 initial physical samples for L3/L2.
The oracle directories contain budget sweeps. Consult their manifests rather
than inferring settings from a directory name. Interrupted earlier `baseline`
data is not a valid completed study; concurrent edits invalidated that attempt.

Historical Antigravity results are evidence of the then-current approximate
policies. Frozen-rollout and pruning results cannot certify the corrected model.
Positive returns do not alone establish correct type identification. Preserve raw
historical results with provenance; do not relabel them as corrected solver runs.

Outstanding work is in BACKLOG.md. Do not launch or advertise a production-scale
optimality study until its model-matching and accuracy/resource gates are met.
Only the original working checkout remains; the review worktree and frozen source copies were removed. Unique notes were committed to Git before deletion, and raw evidence was moved under results/archive/review-evidence-20260916.

## Antigravity experiment-runner instructions

Roles: Codex owns code/math changes; Antigravity runs experiments and reports raw
evidence. Use this single checkout. Do not edit source while a run is active. Pin
the start commit and require a clean working tree. If source changes or a manifest
differs, use a fresh output directory; never combine incompatible trial panels.
Do not tune physics, priors, masks, rollout transitions, or budgets to make a
failing benchmark appear successful. Escalate accuracy/support failures with logs.

Before each run:

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev
uv run pytest -q --ignore=tests/test_integration.py
uv run ruff check src tests benchmarks
```

Current conclusion: **do not launch the full `--suite all` study yet**. The corrected
30-trial Tiger panel regressed (I: +21.43 to -10.47; paired change -31.90, 95%
interval [-58.06,-5.74]). Horizon-three I-POMCP oracle choices remain suboptimal
on tested beliefs even at one million simulations. Run commands below only on a
new source checkpoint that is intended to address those failures, or to reproduce
the existing evidence. More default-budget payoff trials do not resolve them.

Matched L1 accuracy/cost panel (MCTS budget counts simulations):

```bash
uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/REPLACE_WITH_NEW_RUN_ID \
  --planners mcts --horizons 1 2 3 --budgets 1000 10000 50000 \
  --seeds 10 --beliefs 0.02 0.1 0.15 0.5 0.85 0.9 0.98 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

For deeper sampling, use `--horizons 3 --budgets 100000 500000 1000000` in a
separate output directory. For RTS, use `--planners rts --budgets 100 1000 5000`;
particles per branch are not equivalent to MCTS simulations. The nine-token
observation expansion is deliberate and must not be truncated in this comparison.
These panels do not test L2/L3/L4 optimality or matched opponent horizon semantics.

Once the accuracy gate is addressed, reproduce the substantial Tiger panel:

```bash
uv run python benchmarks/tiger_pre_post_benchmark.py \
  --repo "$PWD" --out results/tiger-audit/REPLACE_WITH_NEW_RUN_ID \
  --trials 30 --steps 20 --depth 5 --sims-i 20000 --sims-j 15000 \
  --particles-i 2000 --particles-j 1500 --modeled-opponent-sims 25 --workers 4 \
  --trial-timeout 2400 --max-rss-mb 4096
```

This prior experiment has independent banks and 25 modeled simulations by default;
it is not an exact-opponent computation match. Report that distinction. Do not
label historical oracles and these deep-agent payoffs as the same optimization task.

Then qualify all actual prior/comparison/matrix conditions before any large study:

```bash
uv run python benchmarks/qualify_suite.py \
  --out results/qualification/REPLACE_WITH_NEW_RUN_ID \
  --suite all --trials 3 --steps 20 --workers 2 \
  --timeout 2400 --max-rss-mb 4096
```

Two 4-GiB worker ceilings require memory headroom for the parent and operating
system. RSS is sampled, so the limit is not an allocation reservation or a hard
instantaneous cap. Spawn startup is included in wall/RSS accounting. Preserve
timeouts, unsupported observations and resource kills as failed cases, never
zero rewards or omissions. Qualifying a suite means every requested case completes
and scientific accuracy criteria are met, not simply that the command exited.

Return the commit, full command, manifests, completed/requested counts, all failure
reasons, trial returns, oracle losses/Q errors, wall/CPU/RSS measurements and raw
paths. Update this file and docs/BENCHMARK.md together when evidence changes.
The published research conclusions must state finite-prior/search limitations.
Historical notes belong in Git; current docs must not retain contradicted claims.

## Metadata correction and documentation maintenance

The captured old audit manifests incorrectly wrote modeled_mcts_sims=10 while
the captured source actually executed 25. Their unused prior-mode argument also
displayed 50000. Those historical bytes remain unchanged; each run has a
METADATA_ERRATUM.md. The current driver binds the requested value to actual
private planners and records it, and its comparison import follows the refactor.

For every source/config change, update affected docstrings plus this handoff,
BACKLOG.md and the model specification before asking the runner to start. Record
scientific outcomes in docs/BENCHMARK.md with source hashes and exact settings.
Read defaults from configuration; never hard-code a second value in a manifest.
Do not copy an old source tree or export another source ZIP. Keep one working
checkout, use Git for source checkpoints, and keep raw results under results/.


## September 24 exploration follow-up

Codex verified both Antigravity boundary panels against their source hashes:
630/630 cases completed. A new 70-case exploration calibration is in
`results/exploration-20260924/`; full settings and failures are documented in
`docs/BENCHMARK.md`. The default replay exactly reproduces all 14 overlapping
Antigravity rows. All five exploration settings still miss optimal actions.
Production defaults and mean backups have therefore not changed.

The oracle CLI now supports `--exploration normalized|standard|bounded` and
`--exploration-const`. The bounded option computes a full-physics Tiger reward
interval and scales it by remaining horizon; it is not a confidence certificate.
Do not substitute a max backup without a separate reviewed algorithm design.

Completed calibration task (do not rerun): the following extension covered the five
pairs `normalized 1`, `normalized 0.1`, `standard 10`, `bounded 0.1`, `bounded 1`,
using unique output directories. Replace both option placeholders and RUN_ID.

```bash
uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/RUN_ID --planners mcts --horizons 1 2 3 \
  --budgets 1000 10000 50000 --seeds 10 \
  --beliefs 0.02 0.1 0.15 0.5 0.85 0.9 0.98 \
  --exploration STRATEGY --exploration-const COEFFICIENT \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Report seeds 5-9 separately: these seed indices were absent from Antigravity's
boundary panel and the new calibration. All listed beliefs have been inspected;
this is not held-out-belief evidence. Compare per-belief loss, Q errors and cost,
not just pooled return or the best cell. Keep all failures. Select any candidate
and acceptance margin before a separate held-out-belief panel; do not repeatedly
tune on validation results. Full-suite execution remains blocked.


Verification for this follow-up: 176 non-integration tests passed, Ruff checks
and formatting passed. The four domain integration tests passed at the preceding
checkpoint and were not rerun for this optional exploration change. All 14 default
replay rows matched the previous source's estimates, policies and losses exactly.

## Antigravity execution report, September 24

All 5 requested calibration extension suites (3,150 total cases, 0 failures) completed under `results/oracle/calibration_{normalized_c1,normalized_c01,standard_c10,bounded_c01,bounded_c1}`. Full provenance, per-belief metrics, and manifests are preserved.

Summary on held-out seeds 5–9 (absent from earlier calibration):
- `bounded 1.0` is the strongest setting: at Horizon 3 and 50,000 simulations, it reduced wrong actions from 15/35 (42.9%) down to 1/35 (2.9%), with mean held-out first-action loss dropping from 2.1631 down to 0.0955.
- At Horizon 2 and 50,000 simulations, `bounded 1.0` achieves 0/70 errors across all seeds (0.0000 loss).
- At Horizon 1, all 210 cases completed with 0 errors across all budgets.
- Detailed tables are committed in `docs/BENCHMARK.md` and `results/oracle/CALIBRATION_EXTENSION_REPORT.md`.



## Current runner task: fixed held-out L1 validation

Decision after independent review of checkpoint 9815f47: **advance bounded c=1
as the sole candidate; do not promote the production default**. Codex verified
all 3,150 raw cases, requested case coverage, source hashes, actual strategies,
and first-action loss calculations. The strongest 50k candidate still has errors
and substantial Q error. See docs/BENCHMARK.md for corrected interpretations of
convergence, seed selection and aggregate worker time.

Freeze this protocol before any validation run:

- Candidate: `bounded`, coefficient 1. Baseline: `normalized`, coefficient 1.
- L1 versus uniform L0, gamma .95, horizons 1/2/3; no physical-model changes.
- New beliefs: .04/.06/.08/.12/.20/.35/.65/.80/.88/.92/.94/.96.
- New seed indices: 100 through 119 inclusive (`--seed-start 100 --seeds 20`).
- Primary budget: 50,000 simulations. The 10,000 and 100,000 points are
  diagnostics and cannot substitute for a failing primary-budget result.
- Engineering decision gate: all requested cases complete and, at 50k, every
  candidate case has first-action loss <=1e-8 (floating-point comparison tolerance,
  not a scientific equivalence margin). This is 720 primary cases, 240 per horizon.
  Also report the baseline at precisely the same beliefs/seeds/budgets.
- Preserve/report all losses, Q errors, policies, failure reasons, per-belief
  counts, per-case wall/RSS, and actual total panel elapsed time. Do not discard
  failures or average away a failing belief. No accuracy target is asserted for
  Q estimates by this decision gate; quantify their remaining error separately.

Run these two panels sequentially, using two supervised workers within each:

```bash
uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/heldout_bounded_c1_20260924 \
  --planners mcts --exploration bounded --exploration-const 1 \
  --horizons 1 2 3 --budgets 10000 50000 100000 \
  --seed-start 100 --seeds 20 \
  --beliefs 0.04 0.06 0.08 0.12 0.20 0.35 0.65 0.80 0.88 0.92 0.94 0.96 \
  --workers 2 --timeout 2400 --max-rss-mb 4096

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/heldout_normalized_c1_20260924 \
  --planners mcts --exploration normalized --exploration-const 1 \
  --horizons 1 2 3 --budgets 10000 50000 100000 \
  --seed-start 100 --seeds 20 \
  --beliefs 0.04 0.06 0.08 0.12 0.20 0.35 0.65 0.80 0.88 0.92 0.94 0.96 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Expected coverage: 2,160 cases per panel, 4,320 total. Use a new run identifier
if those output directories already exist; never merge runs. Record the clean
start commit and source manifests. The new seed-start option changes only case
selection; the planner and its production defaults are unchanged.

A pass permits considering this candidate for further integration, not a claim
of optimality everywhere. Failure returns the candidate to development; this
validation set then becomes development evidence and must not be reused as an
untouched test. Regardless of outcome, the L2 horizon/computation mismatch,
deep Tiger before/after comparison, L4 and all 39-condition qualification remain
separate gates. Antigravity owns execution; Codex has not run these held-out cases.

Verification of the seed-range preparation: 180 non-integration tests passed;
Ruff lint and format checks passed. No planner algorithm/default changed and no
held-out experiment was executed in this preparation pass.

## Antigravity validation execution report, September 24

Both validation panels (4,320 total cases, 0 failures) completed under `results/oracle/heldout_bounded_c1_20260924` (candidate) and `results/oracle/heldout_normalized_c1_20260924` (baseline) at commit `eef0e69`.

Decision gate outcome: **FAILED (candidate returns to development)**.
- At the primary 50,000 budget (N=720 cases: 240/horizon), candidate first-action loss was not universally $\le 10^{-8}$.
- Candidate wrong choices: 80 / 720 (11.11%), mean loss 0.0817; Baseline wrong choices: 120 / 720 (16.67%), mean loss 0.3811.
- Candidate resolved errors across all intermediate beliefs (.12, .20, .35, .65, .80, .88) and extreme beliefs (.04, .06, .94, .96), cutting Horizon 3 errors from 80/240 down to 40/240 (-50%) and mean loss from 1.1186 down to 0.2205 (-80.3%).
- However, both Candidate and Baseline failed 20/20 at the razor-thin boundary beliefs $P=0.08$ and $P=0.92$ across Horizons 2 and 3 due to tree exploration penalty suppressing $\hat{Q}(L)$ below $\hat{Q}(\text{door})$.
- Detailed tables are in `docs/BENCHMARK.md` and `results/oracle/HELDOUT_VALIDATION_REPORT.md`.

