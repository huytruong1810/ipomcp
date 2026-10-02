# Results

All results come from the current code (branch `claude/restructure`, October 1, 2026) and the published Tiger specification unless stated otherwise. **n** is the number of evaluated decisions or steps behind each number.

Terms:
- **"Optimal":** every action the planner can choose is an exact maximizer.
- **"Loss":** first-action loss (see [THEORY.md](THEORY.md)).
- **"TV":** total variation distance.

## 1. Planner vs exact value iteration (L1 Tiger)

**Setup.** Agent i against uniform-random L0 j. The reference is alpha-vector value iteration. Each cell covers 11 beliefs × 3 seeds (n = 33). Driver: `experiments.l1_qualification`.

| Simulations | H1 | H2 | H3 | H4 | H5 |
|---|---|---|---|---|---|
| 100: optimal | 100% | 100% | 100% | 100% | 97% |
| 1,000: optimal | 100% | 100% | 100% | 100% | 100% |
| 5,000: optimal | 100% | 100% | 100% | 100% | 100% |
| 100: mean max Q error | 0 | 0.27 | 0.48 | 1.06 | 2.94 |
| 1,000: mean max Q error | 0 | 0.08 | 0.13 | 0.54 | 2.14 |
| 5,000: mean max Q error | 0 | 0.05 | 0.06 | 0.43 | 0.88 |

CPU per root decision at 1,000 simulations ranges from 0.02 s at H2 to 0.10 s at H5.

**With IPF beliefs inside the search** (1,000 simulations; 9 beliefs × 3 seeds, n = 27 per cell):
- 100% optimal at H3 and H4 for 64, 256 and 1,024 particles.
- Mean max Q error at H3: 0.45, 0.18 and 0.17 at 64, 256 and 1,024 particles.
- Mean max Q error at H4: 0.71, 0.64 and 0.62.

## 2. Planner vs exact best response on a terminating domain (UAV, L1)

**Setup.** Pursuer on a 2 × 3 grid against a uniform-random evader. Beliefs are random over two target cells. n = 12 per cell.

| Horizon | Simulations | Optimal | Mean loss | Mean max Q error |
|---|---|---|---|---|
| 2 | 300 | 12/12 | 0 | 0.020 |
| 2 | 3,000 | 12/12 | 0 | 0.006 |
| 3 | 300 | 11/12 | 0.0003 | 0.046 |
| 3 | 3,000 | 11/12 | 0.0003 | 0.015 |

The single H3 miss is a near-tie: its loss is 0.004.

## 3. Interactive particle filter vs exact update (L2 Tiger)

**Setup.** i is L2 with level prior {L1 .5, L0 .5}. Modeled opponents are
I-POMCP with 50 simulations at depth 2. Each history is 8 steps (listen, with
one opening at step 4) generated against a uniform-random j; the IPF and the
exact update condition on the same history. The IPF resamples systematically.
Driver: `experiments.ipf_qualification`; data in
`results/ipf_resampling_probe_20261002`.

**All 12 histories** (n = 96 steps per cell):

| Particles N | Level-posterior TV | Physical-belief error | Nested-path TV (depth 2) | Max level TV | Steps with level TV > 0.2 |
|---|---|---|---|---|---|
| 16 | 0.206 | 0.101 | 0.103 | 0.91 | 39% |
| 64 | 0.062 | 0.041 | 0.031 | 0.68 | 6% |
| 256 | 0.040 | 0.020 | 0.020 | 0.65 | 4% |

**The 3 hardest histories at N = 256** (mean level TV 0.11, 0.12 and 0.09)
were re-run at larger N (n = 24 steps per cell):

| Particles N | Level-posterior TV | Max level TV | Physical-belief error | Nested-path TV |
|---|---|---|---|---|
| 1,024 | 0.020 | 0.067 | 0.010 | 0.010 |
| 4,096 | 0.014 | 0.060 | 0.005 | 0.007 |

**Interpretation:**
- Mean errors shrink with N on every measure. Resampling systematically
  instead of multinomially lowered the mean level TV at every N (paired over
  the same 12 histories: by 0.021, 0.052 and 0.014 at N = 16, 64 and 256;
  better on 6, 9 and 8 histories).
- **The worst case does not shrink like the mean.** After the opening, level
  TV of 0.04–0.06 persists at N = 4,096 on two of the hard histories. The
  modeled opponents act greedily, so a nested particle belief slightly
  different from the exact one can flip a modeled action. That is a discrete
  change in the likelihood of j's behaviour, and more particles make it rarer
  but cannot make it continuous.
- An earlier 3-history run (multinomial resampling) reported a level TV of
  0.019 at N = 1,024 and at most 0.004 at N = 4,096. Those histories were easy
  ones; the 12-history sample above replaces it.
- Raising the modeled budget from 50 to 400 did not change errors (earlier
  run), so the floor is not search noise in the modeled policies.
- The exact belief reaches 18,152 atoms by step 8.

**Gate.** Use N ≥ 1,024 where per-step level posteriors are reported, and
report the mean with the maximum. At L1, N = 64 already gave optimal decisions
inside the search (section 1).

## 4. Cost of an L3 decision

**Setup.** Tiger, i at L3. 2,000 root simulations, 25 modeled simulations, depth 3, after two observations. One decision each (n = 1), profiled.

| Beliefs | CPU (s) | Nested policy solves | Share of CPU in nested solves |
|---|---|---|---|
| Exact | 888 | 1,779 | 89% |
| IPF, 64 particles | 7.7 | 228 | 80% |

**Cross-step reuse.** Keeping the caches across real steps reduced CPU over six L3 decisions (IPF, 64 particles) from 26.6 s to 19.8 s. Decisions and the final belief were identical, which is what purity of the cached functions implies.

## 5. Root search vs opponent modeling (L2 Tiger, vs the exact I-POMDP)

**Setup.** i is L2 and models j as an I-POMCP L1 with `modeled` simulations
(depth equal to the horizon). The reference is the exact I-POMDP: j is an exact
L1 agent, so the loss includes both i's search error and its opponent-model
error. Each cell covers 5 beliefs of i × 3 beliefs of j × 2 seeds (n = 30).
CPU is per root decision, including the nested solves. Driver:
`experiments.l2_panel`.

Optimal rate (mean max Q error in brackets):

| Horizon | Root | Modeled 5 | Modeled 25 | Modeled 100 | Modeled 400 | Modeled 1,000 | Modeled 4,000 |
|---|---|---|---|---|---|---|---|
| 3 | 250 | 90% (0.77) | 90% (0.81) | 100% (0.33) | 100% (0.33) | | |
| 3 | 1,000 | 87% (0.70) | 90% (0.68) | 100% (0.15) | 100% (0.15) | | |
| 3 | 4,000 | 87% (0.62) | 87% (0.62) | 100% (0.06) | 100% (0.06) | | |
| 4 | 250 | 80% (2.38) | 80% (2.22) | 93% (1.56) | 93% (1.56) | 93% (1.55) | 100% (1.08) |
| 4 | 1,000 | 80% (2.08) | 80% (2.07) | 93% (1.21) | 93% (1.21) | 93% (1.20) | 100% (0.76) |
| 4 | 4,000 | 80% (2.12) | 80% (1.98) | 93% (1.03) | 93% (1.03) | | |

Mean CPU per decision: H3 0.07–1.2 s; H4 0.26–3.5 s up to modeled 400, 4.2–4.8 s
at modeled 1,000 and 18–22 s at modeled 4,000.

**Interpretation:**
- **The modeled budget decides optimality; the root budget only reduces Q
  error.** In every row the optimal rate is fixed by the modeled column. A
  16-fold larger root budget never removes a miss caused by a wrong opponent
  model.
- **The H4 ceiling of 93% is opponent-model error.** The two misses (max loss
  1.85) come from one modeled j-L1 that opens where the exact L1 listens
  (Q gap 0.66 at 400 simulations). It persists at 1,000 modeled simulations
  and disappears at 4,000, for both root budgets.
- **Allocation rule.** Give the modeled budget priority: enough that each
  modeled agent's decisions match its exact solution at the beliefs it is
  modeled at (here 100 simulations at depth 3 and 4,000 at depth 4). Section 1
  is not sufficient evidence for this, because it scores L1 decisions at
  i's beliefs, not at the beliefs j is modeled at. Spend what remains on root
  simulations, which lower Q error but do not change the decision once the
  opponent model is right.

## 6. I-POMCP vs IPF + RTS (L2 Tiger, vs the exact I-POMDP)

**Setup.** i is L2; j is modeled as an L1 agent solved by the *same kind* of
planner (I-POMCP with `modeled` simulations, or RTS with K samples per node, as
in IPF + RTS where every level is solved by RTS), at depth equal to the horizon.
The reference is the exact I-POMDP, solved on exact beliefs in its own bank.
Each cell covers 5 beliefs of i × 3 private beliefs of j × 2 seeds (n = 30).
CPU is per root decision, including nested solves. Driver:
`experiments.l2_panel`; data in `results/l2_baseline_20261002`.

**Both planners with IPF beliefs (N = 64)**: the full systems being compared.

| Horizon | Planner | Budget (modeled) | Optimal | Mean max Q error | CPU per decision |
|---|---|---|---|---|---|
| 3 | I-POMCP | 250 (100) | 100% | 0.64 | 2.9 s |
| 3 | I-POMCP | 1,000 (100) | 100% | 0.63 | 3.2 s |
| 3 | I-POMCP | 4,000 (400) | 100% | 0.59 | 7.5 s |
| 3 | IPF + RTS | K = 2 (2) | 93% | 2.19 | 1.0 s |
| 3 | IPF + RTS | K = 4 (4) | 97% | 1.55 | 2.1 s |
| 3 | IPF + RTS | K = 8 (8) | 97% | 1.27 | 4.1 s |
| 4 | I-POMCP | 250 (100) | 97% | 1.60 | 21 s |
| 4 | I-POMCP | 1,000 (100) | 90% | 1.54 | 22 s |
| 4 | I-POMCP | 4,000 (400) | 93% | 1.24 | 60 s |
| 4 | IPF + RTS | K = 2 (2) | 77% | 3.09 | 11 s |
| 4 | IPF + RTS | K = 4 (4) | 80% | 2.66 | 43 s |
| 4 | IPF + RTS | K = 8 (8) | 83% | 1.97 | 96 s |

The panel covers every I-POMCP cell (root 250, 1,000 and 4,000 × modeled 100
and 400). At H3 all are 100% optimal; at H4 they are 90–97% optimal.

**Control: the same cells on exact beliefs.** These separate belief error from search error.
- **I-POMCP:** H3 is 100% optimal at 0.4–1.9 s, and its Q error falls from 0.33 to 0.06 with the root budget. H4 is 93% optimal in every cell, at 1.6–6.5 s.
- **RTS:** H3 is 87%, 97% and 100% optimal at K = 2, 4 and 8, taking 0.04–0.41 s. H4 is 77%, 90% and 83%, taking 0.5–13 s.

**Interpretation:**
- **Accuracy at matched CPU.** I-POMCP is more accurate at every horizon.
  - At H4, I-POMCP with 250 root and 100 modeled simulations is 97% optimal in 21 s. IPF + RTS reaches 80–83% at 43–96 s.
  - RTS's tree grows as (|A| K)^(H−1) at every nesting level. Its CPU rose 11–24× from H3 to H4, while I-POMCP's rose 6–8×.
- **The I-POMCP H4 ceiling (93%) is the opponent-model error of Section 5.** It needs about 4,000 modeled simulations at depth 4, which this grid does not reach.
- **IPF costs more than exact beliefs on these problems.**
  - Each problem starts from a two-atom belief, so exact posteriors stay small within four steps, while the IPF always carries up to 64 particles per level.
  - The IPF earns its cost where exact supports grow: by step 8 of an episode, the exact L2 belief holds 18,152 atoms (Section 3).
  - So this panel compares the planners' accuracy with the IPF in place. It does not show that the IPF is faster here.
- **With the IPF, I-POMCP's Q error has a floor of about 0.6 at H3, whatever the root budget.**
  - The likely source is that each history's expected rewards are integrated over a 64-particle posterior, so particle error enters every backup. This has not been tested separately.
  - The floor never changed a decision at H3.
