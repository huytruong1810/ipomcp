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

## 6. I-POMCP vs RTS (L2 Tiger, vs the exact I-POMDP)

**Setup.** i is L2. Both planners are given the exact j-L1 (depth 3, belief
0.085), and both use exact beliefs, so the comparison varies only the
planner's own lookahead. The
reference is the exact I-POMDP at horizon 3. Each row covers 5 beliefs of i
× 3 seeds (n = 15). Budget is root simulations for I-POMCP and observation
samples per node (K) for RTS. Driver: `experiments.oracle --planners mcts rts`.

| Planner | Budget | Optimal | Mean max Q error | CPU per decision |
|---|---|---|---|---|
| I-POMCP | 250 | 100% | 0.39 | 0.13 s |
| I-POMCP | 1,000 | 100% | 0.12 | 0.23 s |
| I-POMCP | 4,000 | 100% | 0.06 | 0.55 s |
| RTS | K = 2 | 100% | 1.55 | 0.08 s |
| RTS | K = 4 | 87% | 1.23 | 0.09 s |
| RTS | K = 8 | 100% | 0.82 | 0.10 s |

**Interpretation:**
- At horizon 3, both planners usually choose correctly. I-POMCP's Q error is
  an order of magnitude lower at a few times the CPU, and it keeps falling
  with budget. RTS's error falls slowly with K, and its optimal rate is not
  monotone in K at this n (two misses at K = 4).
- RTS cost grows as (|A| K)^(H−1) per decision, while I-POMCP's grows linearly
  in simulations. The scaling comparison at longer horizons, and with IPF
  beliefs (IPF + RTS proper), is the next study.
