# Current review handoff

## Ownership and status

Antigravity runs experiments; Codex owns code/math changes. Work only in
/home/andyj1810/projects/ipomcp on fix/tiger-policy-inversion. Use Git for source
history; do not create source copies or edit source during experiments.

Next task: evaluate the candidate configuration and finalize the numerical loss
bound ($\epsilon_{\text{loss}}$) with the user before freezing a fresh held-out validation suite.
The 480-case H4/H5 budget scaling study is complete. Previous 50k and 200k validation
gate failures remain historical failures. No production default is promoted.

## What was verified: 480-case larger-budget development study

Source checkpoint: 66f9086. Evaluated both `backup="sampled"` and `backup="empirical_bellman"`
with `--exact-final-step`, bounded UCB ($c=1.0$), $\gamma=0.95$, across Horizons 4 and 5,
budgets 200k and 1M, seeds 200–204, and all 12 development beliefs (240 cases per mode,
480 total). Both panels executed sequentially under 2 supervised workers without timeouts,
crashes, or RSS kills. Source manifests match identically across both runs except for `backup`.

Raw results and manifests:
- Control: `results/oracle/bellman_budget_sampled_20260924/`
- Candidate: `results/oracle/bellman_budget_empirical_20260924/`

### Resolution of the three H5/50k material errors

The three large-loss cases identified at $H=5, 50\text{k}$ in the previous audit:
1. $p=0.070$, seed 202 (loss was 0.53094 at 1k, 10k, 50k):
   - At 200k: **chosen=OL (Best=OL), loss = 0.00000** ($\hat{Q}(OL)=1.4083 > \hat{Q}(L)=0.9192$)
   - At 1M: **chosen=OL (Best=OL), loss = 0.00000** ($\hat{Q}(OL)=1.4177 > \hat{Q}(L)=0.8813$)
2. $p=0.930$, seed 201 (loss was 0.53094 at 1k, 10k, 50k):
   - At 200k: **chosen=OR (Best=OR), loss = 0.00000** ($\hat{Q}(OR)=1.3995 > \hat{Q}(L)=0.9339$)
   - At 1M: **chosen=OR (Best=OR), loss = 0.00000** ($\hat{Q}(OR)=1.4603 > \hat{Q}(L)=0.9325$)
3. $p=0.930$, seed 202 (loss was 0.53094 at 10k, 50k):
   - At 200k: **chosen=OR (Best=OR), loss = 0.00000** ($\hat{Q}(OR)=1.3576 > \hat{Q}(L)=0.9371$)
   - At 1M: **chosen=OR (Best=OR), loss = 0.00000** ($\hat{Q}(OR)=1.4234 > \hat{Q}(L)=0.8791$)

Across both 200k and 1M budgets, $p=0.070$ and $p=0.930$ achieved **0/5 errors** in Empirical Bellman.
Added computation fully eliminated the material $0.53094$ errors.

### Comparative accuracy and error distributions

1. **Overall error reduction**: Total policy errors (loss > 1e-8) dropped from **60 / 240 (25.0%)**
   in Sampled Means to **14 / 240 (5.83%)** in Empirical Bellman. Mean loss dropped from 0.46600
   to **0.00085** (548x reduction), and mean max Q error dropped from 44.81 to **2.47** (18x reduction).
2. **Listen optimal actions ($N=120$)**:
   - Sampled Means: **60 / 120 errors (50.0% failure rate!)**, mean loss 0.93199, max loss 3.57187.
     Sampled means persistently fails on Listen actions even at 1M traversals.
   - Empirical Bellman: **0 / 120 errors (0.00% failure rate)**, mean loss 0.00000.
3. **Open optimal actions ($N=120$)**:
   - Sampled Means: 0 / 120 errors (it aggressively opens doors everywhere).
   - Empirical Bellman: 14 / 120 errors, exclusively localized to razor-thin inflection boundaries
     ($p \in \{0.075, 0.925\}$).
4. **Max Loss Bound across 240 cases**: The maximum first-action loss observed anywhere in the 240
   Empirical Bellman cases is **0.01814** (at $H=5, p \in \{0.075, 0.925\}$). At $H=4$, the maximum
   loss is **0.00992**.
5. **Convergence at 1,000,000 traversals**:
   - $H=4, 1\text{M}$: **59 / 60 strictly optimal decisions (98.33%)**. Only 1 error (seed 204 at $p=0.075$,
     loss 0.00992). Mean loss across all 60 cases: 0.00017.
   - $H=5, 1\text{M}$: **57 / 60 strictly optimal decisions (95.00%)**. Only 3 errors (seeds 200, 204 at
     $p=0.075$; seed 201 at $p=0.925$; loss 0.01814). Mean loss across all 60 cases: 0.00091.

### Computational footprint

- Sampled: elapsed panel wall time **4,818.7 s (80.31 min)**, worker sum 10,152.4 s, peak RSS 107.42 MB.
- Empirical Bellman: elapsed panel wall time **5,708.4 s (95.14 min)**, worker sum 12,064.2 s, peak RSS 113.39 MB.
- Elapsed runtime ratio (Emp/Samp) was **1.18x** (+18.5% overhead for chance-weighted Bellman backups at 1M).
- Peak RSS delta was **+5.97 MB**, well within the 4,096 MB monitor ceiling.

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
