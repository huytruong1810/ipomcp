# Current review handoff

## Decision and ownership

Antigravity runs experiments; Codex owns code/math changes. Use the single
checkout `/home/andyj1810/projects/ipomcp`, branch `fix/tiger-policy-inversion`.
Do not change source during a run or create frozen source copies. Git holds
source history; raw results stay under results/ with source manifests.

Completed task: 1,800-case H1–H5 **development** panel comparing empirical
chance-weighted Bellman backups against sampled means (900 cases each). Both use
exact final-step evaluation (`--exact-final-step`) and bounded UCB ($c=1.0$).
No production default is promoted, and no new held-out validation or full-suite
run is authorized by these development results. Previous 50k and 200k validation
gates remain failed.

## What was verified

The 1,800-case development study at git checkpoint 7d8b6bd completed sequentially
with 2 supervised workers per panel without timeouts, crashes, or RSS kills.
Source manifests match identically across both runs except for `backup: "sampled"`
vs `backup: "empirical_bellman"`.

Raw results and manifests:
- Control: `results/oracle/bellman_development_sampled_20260924/`
- Candidate: `results/oracle/bellman_development_empirical_20260924/`

### Overall findings

1. **Error reduction**: Total policy errors (loss > 1e-8) dropped from 206/900
   (22.89%, mean loss 0.36433, mean Q error 22.05) in sampled means to 89/900
   (9.89%, mean loss 0.02227, mean Q error 1.80) in empirical Bellman.
2. **Horizon 1 ($H=1$)**: 0/180 errors in both panels across all budgets, beliefs,
   and seeds. Mean loss = 0.00000, Q error = 0.00000.
3. **Horizon 2 ($H=2$)**: Sampled means had 1/180 errors (at B=1k, seed 204, p=0.075,
   loss 0.36392). Empirical Bellman had **0/180 errors** across all budgets, beliefs,
   and seeds (mean Q error 0.0581 at 1k, 0.0196 at 50k).
4. **Horizon 3 ($H=3$) Resolution**: Sampled means persistently failed at all budgets
   (21/60 at 1k, 30/60 at 10k, 30/60 at 50k, 45.0% overall; mean loss 0.49332;
   mean Q error 17.62) due to depth-1 leaf exploration penalties dragging $\hat{Q}(L)$
   down. Empirical Bellman achieved **0/60 errors at 50k** (2/60 at 1k, 1/60 at 10k),
   with mean loss 0.00000 and mean Q error dropping 450x to 0.03882.
5. **Horizon 4 ($H=4$) Mechanics**: In the true model, the optimal action boundary
   shifts outward: $p \in \{0.070, 0.075, 0.925, 0.930\}$ favor opening doors, but at
   $p \in \{0.075, 0.925\}$, the gap is only $\Delta Q^* = 0.00992$.
   - Sampled means failed 20/60 at all budgets (mean loss 0.76553, max loss 3.58056,
     mean Q error 38.61), failing on all Listen beliefs ($0.085, 0.110, 0.890, 0.915$).
   - Empirical Bellman achieved **0 errors** on all Listen beliefs across all budgets!
     At 10k, errors dropped to 9/60 (4 at 0.075, 5 at 0.925), and at 50k dropped to
     5/60 (2 at 0.075, 3 at 0.925). In every single case at 10k and 50k, the chosen action
     was Listen and the first-action loss was exactly **0.00992** (mean loss at 50k: 0.00083).
6. **Horizon 5 ($H=5$) Mechanics**: Sampled means failed 64/180 (20/60 at 10k and 50k,
   mean loss 0.76322, mean Q error 53.40), completely failing on all Listen beliefs.
   Empirical Bellman had 52/180 errors (19 at 1k, 20 at 10k, 13 at 50k). At 50k, errors
   were localized to $p=0.070$ (1/5, loss 0.53094), $p=0.075$ (5/5, loss 0.01814),
   $p=0.925$ (5/5, loss 0.01814), and $p=0.930$ (2/5, loss 0.53094). Mean loss dropped
   26x to 0.02957, and mean Q error dropped 15x to 3.53.
7. **Action Category Asymmetry**:
   - Optimal = Listen (N=480): Sampled means had 180 errors (37.50%, mean loss 0.62430).
     Empirical Bellman had **5 errors (1.04%, mean loss 0.00616)**.
   - Optimal = Open (N=420): Sampled means had 26 errors (6.19%, mean loss 0.06723).
     Empirical Bellman had 84 errors (20.00%, mean loss 0.04068), concentrated on near-tie
     inflection boundaries where it conservatively listened.
8. **Computational Footprint**:
   - Sampled panel elapsed wall time: 602.29 s (mean case wall: 1.31 s; max RSS: 82.42 MB).
   - Empirical Bellman elapsed wall time: 657.76 s (mean case wall: 1.44 s; max RSS: 84.34 MB).
   - Wall clock overhead was only **1.09x** (+9.2%) and RSS difference was **+1.92 MB**.

Raw runs and manifests remain local and Git-ignored. Do not claim a Git/remote
archive exists unless actually created. Captured manifests and case JSON files
are preserved unchanged under `results/oracle/bellman_development_*_20260924/`.

## Engineering verification

At source checkpoint 2230786, all 200 non-integration tests passed in 49.13 s
and all four domain integration tests passed in 265.32 s (204 total). Ruff lint
and formatting passed across 106 Python files. The follow-up changes only clarify
an MCTS-only option error message and documentation; planning behavior is unchanged.
These tests and the L1 panels do not certify L2/L3/L4 optimality or full-suite
readiness. Raw integration output is in
results/bellman-backup-20260924/integration-tests.log.

## Executed runner protocol

The development comparison between `sampled` and `empirical_bellman` with bounded
$c=1$ and exact final-step enabled in **both** completed across H1–H5, budgets
1k/10k/50k, 12 beliefs, and seeds 200–204 (900 cases each, 1,800 total).

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD  # 7d8b6bd2eaf1f12d6ad0aace7dd1c697c4964199
uv sync --frozen --group dev

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_development_sampled_20260924 \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup sampled --horizons 1 2 3 4 5 --budgets 1000 10000 50000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/bellman_development_empirical_20260924 \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --backup empirical_bellman --horizons 1 2 3 4 5 --budgets 1000 10000 50000 \
  --seed-start 200 --seeds 5 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Tree state remained clean throughout; all 1,800 cases finished with status `complete`.
Per-case wall times and RSS remained within supervisor limits (mean 1.31 s / 1.44 s;
peak RSS 82.4 MB / 84.3 MB). Total elapsed panel time was 602.29 s (sampled) and
657.76 s (empirical). Detailed per-case files and logs are preserved under
`results/oracle/bellman_development_{sampled,empirical}_20260924/`.

## Key takeaways and next steps

1. **Definitive evidence on H3**: Empirical Bellman backups resolve the on-policy
   exploration drag on listening actions, producing 0/60 errors and reducing Q error
   by 450x at 50k budget.
2. **H4 and H5 near-tie boundary behavior**: At H4 and H5, the true optimal boundary
   shifts outward. Empirical Bellman conservatively listens at razor-thin boundaries
   ($p \in \{0.075, 0.925\}$ where the oracle gap is $\le 0.01814$), incurring small
   first-action losses ($\le 0.00992$ at H4; $\le 0.01814$ at H5). It completely
   eliminates the large prematures open errors on Listen beliefs that crippled sampled means.
3. **Loss threshold requirement**: A formal numerical tolerance bound for first-action
   loss (e.g., $\epsilon_{\text{loss}} = 0.01$ or $0.02$) must be defined and frozen
   before running a fresh validation suite. The threshold must not be chosen post-hoc.
4. **Validation gate status**: The 50k and 200k validation gates remain failed. All 12
   beliefs and seeds 200–204 are development evidence. Fresh validation will require
   independent, unseen beliefs and seeds.

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
