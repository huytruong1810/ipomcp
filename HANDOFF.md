# Current review handoff

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

Next runner task: run the following calibration extension separately for the five
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
