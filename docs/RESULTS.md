# Results

All results come from the current code (`main`; Sections 1–8 from October 1, 2026) and the published Tiger specification unless stated otherwise. **n** is the number of evaluated decisions or steps behind each number.

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

## 3. Sampled interactive particle filter vs exact update (L2 Tiger)

**Setup.** i is L2 with level prior {L1 .5, L0 .5}. Modeled opponents are
I-POMCP with 50 simulations at depth 2. Each history is 8 steps (listen, with
one opening at step 4) generated against a uniform-random j; the IPF and the
exact update condition on the same history. The IPF resamples systematically.
Driver: `experiments.belief_qualification --particle-filters sampled`; data in
`results/ipf_resampling_probe_20261002`. The compressed filter, measured the
same way, is in Section 8.

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

**Gate (sampled IPF).** Use N ≥ 1,024 where per-step level posteriors are
reported, and report the mean with the maximum. The compressed filter (Section
8) meets the same accuracy at N = 64.

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
`experiments.panel`.

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

## 6. I-POMCP vs IPF + RTS (L2 Tiger, vs the exact I-POMDP; sampled IPF)

**Setup.** i is L2; j is modeled as an L1 agent solved by the *same kind* of
planner (I-POMCP with `modeled` simulations, or RTS with K samples per node, as
in IPF + RTS where every level is solved by RTS), at depth equal to the horizon.
The reference is the exact I-POMDP, solved on exact beliefs in its own bank.
Each cell covers 5 beliefs of i × 3 private beliefs of j × 2 seeds (n = 30).
CPU is per root decision, including nested solves. Driver:
`experiments.panel`; data in `results/l2_baseline_20261002`.

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
- **On exact beliefs, the I-POMCP H4 ceiling (93%) is the opponent-model error of Section 5.**
  - With 4,000 modeled simulations it is 100% optimal at root 250 and at root 1,000 (n = 30 each; 38 s per decision; `results/l2_baseline_h4_modeled4000_20261002`).
- **With the IPF (N = 64), I-POMCP stays at 93% at H4 even with 4,000 modeled simulations (206 s per decision).**
  - Those misses are belief error, not search error: different problems from the exact-belief misses, and independent of both budgets.
  - At N = 64 the planner values opening at 6.2–6.5 against an exact 3.2–3.3.
  - Re-run at root 1,000 and modeled 100, the two missed problems become optimal at N = 256 and N = 1,024 respectively (`results/l2_ipf_miss_20261002`).
  - At L1, N = 64 was enough (Section 1); at L2 H4 it is not.
- **IPF costs more than exact beliefs on these problems.**
  - Each problem starts from a two-atom belief, so exact posteriors stay small within four steps, while the IPF always carries up to 64 particles per level.
  - The IPF earns its cost where exact supports grow: by step 8 of an episode, the exact L2 belief holds 18,152 atoms (Section 3).
  - So this panel compares the planners' accuracy with the IPF in place. It does not show that the IPF is faster here.
- **With the IPF, I-POMCP's Q error has a floor of about 0.6 at H3, whatever the root budget.**
  - The likely source is that each history's expected rewards are integrated over a 64-particle posterior, so particle error enters every backup. This has not been tested separately.
  - The floor never changed a decision at H3.

## 7. Exact vs sampled-IPF beliefs in mid-episode decisions (L2 Tiger, H3)

**Setup.**
- i decides after conditioning on the first k steps of a fixed history, generated by `oracle.sample_history` (i listens and opens once at step 4; j acts at random).
- j is the exact L1 agent, so only i's belief representation and search differ.
- Planner: I-POMCP with 1,000 root simulations.
- The reference is the exact I-POMDP at the exact posterior.
- Each cell covers 5 beliefs of i × 3 beliefs of j × 2 seeds (n = 30).
- Planning CPU is per decision; update CPU covers all k belief updates; support is the number of distinct interactive hypotheses i holds.
- Driver: `experiments.panel`; data in `results/l2_mid_episode_20261002`.

| History k | Beliefs | Optimal | Mean max Q error | Planning CPU | Update CPU | Support |
|---|---|---|---|---|---|---|
| 0 | exact | 100% | 0.15 | 0.3 s | 0 | 2 |
| 0 | IPF 64 | 100% | 0.63 | 0.8 s | 0 | 2 |
| 0 | IPF 256 | 100% | 0.15 | 1.7 s | 0 | 2 |
| 0 | IPF 1,024 | 100% | 0.13 | 4.8 s | 0 | 2 |
| 2 | exact | 97% | 0.14 | 2.1 s | 0.02 s | 23 |
| 2 | IPF 64 | 97% | 7.73 | 3.6 s | 0.06 s | 16 |
| 2 | IPF 256 | 97% | 1.89 | 8.2 s | 0.11 s | 26 |
| 2 | IPF 1,024 | 100% | 1.16 | 23 s | 0.45 s | 36 |
| 4 | exact | 100% | 0.11 | 26 s | 0.31 s | 241 |
| 4 | IPF 64 | 100% | 3.99 | 7.9 s | 0.32 s | 32 |
| 4 | IPF 256 | 100% | 2.77 | 23 s | 1.07 s | 81 |
| 4 | IPF 1,024 | 100% | 1.32 | 140 s | 1.83 s | 181 |

RTS (K = 4) on the same problems was 97–100% optimal. With exact beliefs it took 0.1, 1.4 and 17 s at k = 0, 2 and 4. With IPF 256 it took 0.9, 6.2 and 19 s, and its Q error was 1.3–2.9.

**Beyond k = 4** (one problem: i at 0.5, j at 0.5, seed 0):
- At k = 6, exact-belief I-POMCP did not finish one decision in 30 minutes.
- The exact I-POMDP reference exceeded its 200,000-branch enumeration budget.
- The IPF planners still run there, but they cannot be scored.

**Interpretation:**
- **Where the exact reference is tractable, exact beliefs dominate.** Up to k = 4, exact-belief I-POMCP is faster than the IPF at N = 1,024 and about as fast as at N = 256 (26 s against 23 s at k = 4). It is 8–25 times more accurate in Q than either. Only the IPF at N = 64 is clearly cheaper (8 s against 26 s at k = 4), at about 36 times the Q error, and it has one miss with a loss of 4.8 at k = 2.
- **Particle error accumulates with history.** At k = 0 the IPF matches exact beliefs (the prior has two atoms). By k = 2–4 its Q error is 1.2–7.7, because each in-tree update samples the opponent's private observations and actions afresh.
- **The compressed filter (Section 8) changes this conclusion.** The remaining points apply to the sampled IPF.
- **The IPF pays off only where exactness becomes intractable:** k ≥ 6 at L2 H3 here, or the 94,500-state Wumpus prior.
  - There, its accuracy can be checked only indirectly: by belief error against exact filtering along single histories (Section 3, up to 18,152 atoms), or against references that drop the nested enumeration.
  - Contribution 2 must therefore report IPF error where it can be measured and cost where it cannot. It must not claim measured accuracy in the regime that motivates the IPF.

## 8. The compressed filter: exact updates with supports bounded to N

**Why.** The diagnosis came from exact Bellman values over sampled-IPF beliefs, with no search involved.
- At N = 64, they reproduce the H4 misses of Section 6: Q(open) is 6.31 against an exact 3.32. At N = 256 they match the exact values to two decimals.
- The sampled IPF holds every belief, however small, as N equal-weight samples. A nested two-state belief therefore moves in steps of 1/N, and greedy modeled opponents flip their predicted action across thresholds.
- The compressed filter (`belief.compressed_filter`; see THEORY.md) enumerates each update exactly. It samples (systematically) only to bring a posterior with more than N atoms back to N.
- Prototype and diagnosis scripts are in `results/compressed_prototype_20261002`.

**Belief accuracy** (same 12 histories and modeled opponents as Section 3; n = 96 steps per cell; `results/belief_qualification_compressed_20261002`):

| N | Filter | Level-posterior TV | Physical-belief error | Nested-path TV | Max level TV | Steps with level TV > 0.2 |
|---|---|---|---|---|---|---|
| 16 | compressed | 0.049 | 0.013 | 0.025 | 0.43 | 4% |
| 16 | sampled | 0.206 | 0.101 | 0.103 | 0.91 | 39% |
| 64 | compressed | 0.008 | 0.001 | 0.004 | 0.09 | 0% |
| 64 | sampled | 0.062 | 0.041 | 0.031 | 0.68 | 6% |
| 256 | compressed | 0.002 | 0.0005 | 0.001 | 0.02 | 0% |
| 256 | sampled | 0.040 | 0.020 | 0.020 | 0.65 | 4% |

**Mid-episode decisions** (Section 7's problems: I-POMCP with 1,000 simulations, exact j-L1, H3; n = 30 per cell; `results/l2_compressed_20261002`):

| History k | Beliefs | Optimal | Mean max Q error | Planning CPU | Support |
|---|---|---|---|---|---|
| 2 | exact | 97% | 0.14 | 2.1 s | 23 |
| 2 | compressed 16 | 93% | 2.14 | 0.7 s | 9 |
| 2 | compressed 64 | 100% | 0.25 | 1.5 s | 23 |
| 2 | sampled IPF 64 | 97% | 7.73 | 3.6 s | 16 |
| 4 | exact | 100% | 0.11 | 26 s | 241 |
| 4 | compressed 16 | 100% | 0.68 | 0.8 s | 9 |
| 4 | compressed 64 | 100% | 0.18 | 1.9 s | 29 |
| 4 | compressed 256 | 100% | 0.12 | 4.6 s | 77 |
| 4 | sampled IPF 64 | 100% | 3.99 | 7.9 s | 32 |

RTS (K = 4) with compressed beliefs at N = 64:
- It was 97%, 97% and 100% optimal at k = 0, 2 and 4, with Q error 1.23, 1.00 and 0.48.
- It took 0.1, 0.8 and 1.3 s, against 0.1, 1.4 and 17 s on exact beliefs.

**From the prior at H4** (Section 6's problems, N = 64):

| Planner | Budget (modeled) | Compressed | Exact | Sampled IPF |
|---|---|---|---|---|
| I-POMCP | 250 (100) | 90%, 1.5 s | 93%, 1.6 s | 97%, 21 s |
| I-POMCP | 1,000 (100) | 93%, 2.1 s | 93%, 2.1 s | 90%, 22 s |
| I-POMCP | 250 (4,000) | 97%, 27 s | 100%, 38 s | 93%, 206 s |
| RTS | K = 2 | 77%, 0.4 s | 77%, 0.5 s | 77%, 11 s |
| RTS | K = 4 | 90%, 2.8 s | 90%, 3.1 s | 80%, 43 s |
| RTS | K = 8 | 83%, 9.9 s | 83%, 13 s | 83%, 96 s |

**Interpretation:**
- **At equal N, the compressed filter is far more accurate than the sampled IPF.**
  - Its belief errors are 4–40 times smaller. Its worst step at N = 64 (0.09) is close to the sampled filter's floor at N = 1,024–4,096 (0.06–0.07 on the hard histories).
  - In decisions, its Q error at N = 64 is close to exact (0.18 against 0.11 at k = 4).
- **It is also cheaper.**
  - Mid-episode at k = 4, N = 64 is 14 times faster than exact beliefs and 4 times faster than the sampled IPF.
  - From two-atom priors it costs the same as exact beliefs, because nothing exceeds N. The sampled IPF costs 5–24 times more there, because it always carries N samples.
- **So bounded beliefs now pay off inside the region where the exact I-POMDP is tractable,** and can be scored there. This replaces the Section 7 conclusion for the planner.
- **One problem differs from exact at H4 with 4,000 modeled simulations** (97% against 100%). Its loss, 1.53, is the exact-belief miss at low modeled budgets. The likely cause is that compressed nested beliefs change the modeled opponent's search seeds, so this is opponent-model noise; it has not been tested separately.
- **RTS with compressed beliefs keeps its exact-belief accuracy at exact-belief cost.** Its misses are therefore its own search error, and the sampled IPF added cost without changing them.

**Beyond the exact planner: k = 6** (same problems; n = 30 per cell; `results/l2_compressed_k6_20261002`):
- Raising the exact reference's enumeration guard to 50 million branches (`oracle.REFERENCE_BRANCHES`) made it complete at k = 6, in 146 s for one problem.
- Exact-belief I-POMCP did not finish one decision there in 30 minutes (n = 1).

| Planner | Beliefs | Optimal | Mean max Q error | Planning CPU | Support |
|---|---|---|---|---|---|
| I-POMCP 1,000 | compressed 64 | 100% | 1.02 | 2.9 s | 38 |
| I-POMCP 1,000 | compressed 256 | 100% | 0.22 | 7.6 s | 96 |
| RTS K = 4 | compressed 64 | 97% | 1.66 | 2.7 s | 38 |

- **Not yet measured:** L3.

## 9. Estimator revision: each step's state is drawn from b_h (October 2, 2026)

**What changed.** Sections 1–8 were run with root sampling: one particle drawn at the root and carried down the tree. Since commit `f793712`, every tree and rollout step draws its interactive state from the agent's history posterior b_h.
- With exact b_h the two have the same distribution.
- With bounded b_h only the new form is consistent: a carried particle can sit on an atom that compression removed from b_h. With deterministic sensors its observation then has zero evidence. This happened on Wumpus (Section 10).

**L1 gate** (Section 1's design, n = 33 per cell; `results/l1_qualification_belief_sampling_20261002`):
- 100% optimal at every budget and horizon, except H5 at 100 simulations: 94% (previously 97%).
- Mean max Q error at 1,000 simulations: 0.09, 0.11, 0.57 and 1.67 at H2–H5 (previously 0.08, 0.13, 0.54 and 2.14).

**L2 re-runs** (n = 30 per cell; `results/l2_revalidate_belief_sampling_20261002`):

| Problem | Beliefs | Optimal now (before) | Mean max Q error now (before) |
|---|---|---|---|
| H3, k = 0 | exact | 100% (100%) | 0.13 (0.15) |
| H3, k = 2 | exact | 100% (97%) | 0.12 (0.14) |
| H3, k = 4 | exact | 100% (100%) | 0.10 (0.11) |
| H3, k = 2 | compressed 64 | 97% (100%) | 0.26 (0.25) |
| H3, k = 4 | compressed 64 | 100% (100%) | 0.17 (0.18) |
| H3, k = 6 | compressed 64 | 97% (100%) | 1.03 (1.02) |
| H3, k = 2 | compressed 256 | 100% (97%) | 0.12 (0.15) |
| H3, k = 4 | compressed 256 | 100% (100%) | 0.11 (0.12) |
| H3, k = 6 | compressed 256 | 97% (100%) | 0.20 (0.22) |
| H4, root 250, modeled 100 | exact | 80% (93%) | 2.55 (1.56) |
| H4, root 250, modeled 100 | compressed 64 | 80% (90%) | 2.54 (1.51) |
| H4, root 1,000, modeled 100 | exact | 80% (93%) | 2.45 (1.21) |
| H4, root 250, modeled 4,000 | exact | 100% (100%) | 1.04 (1.08) |
| H4, root 250, modeled 4,000 | compressed 64 | 100% (97%) | 0.98 (0.97) |

**Interpretation:**
- **Within the sampling variation of n = 30, the revision reproduces every result except one.** The exception is the H4 rate at 100 modeled simulations, 93% → 80%.
- **That rate is a property of a few modeled opponents.** A modeled j-L1 at 100 simulations is one random search per distinct belief. The revision changed the random streams, so different modeled policies came out, and one wrong opponent policy is shared by several problems.
- **This strengthens Section 5's rule.** At 100 modeled simulations, H4 decisions depend on luck in the opponent models. At 4,000 they are 100% optimal for exact and compressed beliefs alike.
- **Compressed 64 now equals exact beliefs on every H4 cell** (same optimal rates and losses), consistent with nothing exceeding 64 atoms on those problems.

## 10. Wumpus pilot: exact real beliefs, compressed search (L1, H3)

**Setup (pilot).**
- The human (L1) against a uniform-random wumpus, on the full 4 × 4 prior (94,500 states).
- The human's real beliefs are exact; each search starts from the belief compressed to 64 or 256 atoms (`SolverBank(search_particles=N)`).
- The planner is I-POMCP with 1,000 simulations at depth 3, deciding after 0, 2 or 4 real steps of a scripted history.
- The reference is the exact best response at H3 on the exact belief.
- n = 18 decisions over 3 seeds. Data in `results/wumpus_split_pilot_20261002`.

**Measured:**
- **Exact filtering is cheap.** Along one history the support shrank from 94,500 to 46 atoms in 8 steps, at most 2.9 s per update (`results/wumpus_exact_filtering_20261002`).
- **The exact reference is tractable at H3:** 520–600 s at the prior, 92–142 s after 2–4 steps.
- **Planning took 1.2–34 s per decision.**
- **Losses were 0 in 5 of 18 decisions and 1.8–7.7 in the rest; max Q errors were 2.6–89** (rewards range over ±1,000).

**Status: not qualified.** The split removes the failure mode but is not accurate yet. The likely causes, not yet separated, are:
- compressing a 25,920–94,500-atom belief to 64–256 atoms;
- 1,000 simulations over Wumpus's 5 actions and 32 observations.

The next rung is a qualification grid over N and simulations against the exact reference.

## 11. Post-search chance scoring (October 2, 2026)

**What changed** (commit `192ea90`, `planning.chance_scoring`). After search, the planner rescores its tree bottom-up, weighting each observation by its exact probability P(o, alive | b_h, a) under the agent's belief at that history, instead of by how often search sampled it (THEORY.md). Two cases keep their approximate values:
- an action with an observation search never sampled keeps its **search value** (a "fallback");
- a branch reached once keeps its rollout estimate.

The rescored values decide, for real decisions and for modeled opponents alike.

**Design.** Search itself (exploration, backups, random streams) is unchanged, and every decision records both values:
- **"search":** the empirical-backup value, i.e. what the planner decided on before this change;
- **"scored":** the rescored value.

The "same-tree" columns compare the two on identical trees. The search columns reproduce Sections 1 and 9 exactly.

**L1 gate** (Section 1's design, n = 33 per cell; `results/l1_qualification_chance_scoring_20261002`). Mean max Q error, search → scored:

| Simulations | H2 | H3 | H4 | H5 |
|---|---|---|---|---|
| 100 | 0.29 → 0.02 | 0.45 → 0.38 | 1.16 → 1.17 | 2.43 → 3.52 |
| 1,000 | 0.09 → 0 | 0.11 → 0.004 | 0.57 → 0.61 | 1.67 → 1.57 |
| 5,000 | 0.05 → 0 | 0.05 → 0 | 0.34 → 0.38 | 1.46 → 1.50 |

("0" means below 1e-13.)
- **Decisions:** 100% optimal everywhere except H5 at 100 simulations, 94% → 91%. On the same trees, 0 decisions were corrected and 1 regressed, out of 297 at H3–H5.
- **Per action (H4, 5,000 simulations):** the mean Q error of listening, the most-sampled action, halves (0.078 → 0.038). The rarely sampled door openings get slightly worse (0.25 → 0.27, 0.17 → 0.19). Their children have only one or two visits, and exact weights give those noisy estimates their full probability.

**L2 panel** (Section 9's 15 cells, n = 30 per cell, against the exact I-POMDP; `results/l2_chance_scoring_20261002`). All values are search → scored:

| Problem | Beliefs | Optimal | Mean max Q error | CPU (s) |
|---|---|---|---|---|
| H3, k = 0 | exact | 100% → 100% | 0.13 → 0.05 | 0.33 → 0.25 |
| H3, k = 2 | exact | 100% → 100% | 0.12 → 0.08 | 2.1 → 1.9 |
| H3, k = 4 | exact | 100% → 100% | 0.10 → 0.04 | 23 → 22 |
| H3, k = 2 | compressed 64 | 97% → 97% | 0.26 → 0.24 | 1.5 → 1.4 |
| H3, k = 4 | compressed 64 | 100% → 100% | 0.17 → 0.14 | 1.6 → 1.6 |
| H3, k = 6 | compressed 64 | 97% → 93% | 1.03 → 1.01 | 2.7 → 4.2 |
| H3, k = 2 | compressed 256 | 100% → 100% | 0.12 → 0.08 | 2.3 → 2.2 |
| H3, k = 4 | compressed 256 | 100% → 100% | 0.11 → 0.05 | 4.4 → 4.1 |
| H3, k = 6 | compressed 256 | 97% → 100% | 0.20 → 0.16 | 5.9 → 8.0 |
| H4, root 250, modeled 100 | exact | 80% → 80% | 2.55 → 2.54 | 0.8 → 0.8 |
| H4, root 250, modeled 100 | compressed 64 | 80% → 80% | 2.55 → 2.58 | 0.9 → 0.9 |
| H4, root 1,000, modeled 100 | exact | 80% → 80% | 2.45 → 2.37 | 1.1 → 1.0 |
| H4, root 1,000, modeled 100 | compressed 64 | 80% → 80% | 2.45 → 2.39 | 1.2 → 1.2 |
| H4, root 250, modeled 4,000 | exact | 100% → 100% | 1.04 → 1.05 | 35 → 33 |
| H4, root 250, modeled 4,000 | compressed 64 | 100% → 100% | 0.98 → 0.98 | 34 → 33 |

CPU columns are the previous run (Section 9) and this run. The k = 6 runs shared the machine with unrelated jobs.

- **Same tree:** 1 decision corrected and 1 regressed, out of 450. Both are the same near-tie problem (k = 6, seed 1, p = 0.05: exact Q of listen 3.86, open-right 4.19). In both, open-right kept its search value (fallback) while listen was rescored, so the root compared a scored value against a search value.
- **Modeled opponents:** scoring them did not change any H4 result. Every search column equals Section 9 to the last digit, so no modeled policy's greedy action changed.
- **Exactness flag:** the scored root was never certified exact in these panels. Once-visited deep histories leave actions untried, so the flag is conservative.

**Rejected variant.** Instead of falling back, weight the sampled branches by their exact probabilities, renormalized to the continuation mass. On the L1 gate it gave the same decisions and more upward bias (listen at H4, 1,000 simulations: bias +0.04 → +0.13). Script: `rejected_renormalized_variant.py` in the L1 results folder.

**Verdict.**
- **Decisions are neutral within this n.** Across the two gates, 1 was corrected and 2 regressed out of 747 paired decisions. The legacy evidence (4 corrected, 0 regressed) was on a different estimator and does not transfer.
- **Q accuracy improves where branches are well sampled:** H2–H3 Q error falls 30–100%, and complete subtrees are exact.
- **It does not help at H4–H5**, where error is dominated by under-sampled actions and opponent models.

Kept as the default because it improves the accuracy of reported Q values at no decision cost. The comparison stays one commit away, since every row records both values.

## 12. Wider n (October 2, 2026)

Sections 1–11 rest on 30–33 decisions per cell. This section re-measures the questions those cells left open, with more problems.

### 12.1 L1 H5 Q error: no regression from per-step belief sampling

**Question.** At 5,000 simulations, L1 H5 mean max Q error was 0.88 in Section 1, 1.46 after per-step belief sampling (Section 9's code) and 1.50 with chance scoring (Section 11). Did `f793712` make long-horizon values worse?

**Design.** Section 1's problem (11 beliefs) at H5, 1,000 and 5,000 simulations, seeds 0–19: n = 220 per cell. Three arms, paired by (seed, belief):
- **before:** `f793712^` (one root particle carried down the tree);
- **after:** `f793712` (each step's state drawn from b_h);
- **HEAD:** `6f6c49d` (after + chance scoring).

The before and after arms ran from temporary worktrees of those commits. `results/gap2_belief_sampling_paired_20261002`.

**Results** (mean max Q error; differences are paired, with 95% t-intervals):

| Simulations | before | after | HEAD (scored) | after − before | HEAD − after |
|---|---|---|---|---|---|
| 1,000 | 1.71 | 1.47 | 1.52 | −0.24 [−0.53, +0.06] | +0.05 [−0.02, +0.12] |
| 5,000 | 1.05 | 1.03 | 1.14 | −0.02 [−0.22, +0.18] | +0.11 [+0.05, +0.17] |

Every decision in every arm is optimal (220/220; 95% Clopper–Pearson lower bound 98.3%).

- **No regression from `f793712`.** On seeds 0–2, the arms reproduce the earlier figures exactly (0.88, 1.46/1.67 and 1.50). Per-seed means at 5,000 simulations range from 0.56 to 2.15, so three-seed averages cannot resolve differences of this size.
- **Chance scoring slightly raises L1 H5 Q error at 5,000 simulations** (+0.11; Wilcoxon p = 0.0005). HEAD's search values equal the after arm's to the last digit, so this is scoring alone, on identical trees (`scoring_on_identical_trees.py`). Every one of the 440 roots has at least one action that falls back to its search value. This corrects Section 11's "flat at H4–H5": at H5 scoring is mildly harmful to Q values, though not to decisions.
- **The long-horizon plateau is real, and it is heavy-tailed.** Mean max Q error at H5 falls only from 1.47 to 1.03 between 1,000 and 5,000 simulations, and single decisions reach errors of 6–12. Section 12.2 locates it.

### 12.2 The long-horizon Q plateau is mostly the metric

**Question.** Mean max Q error barely falls with budget at long horizons (L1 H5: 1.47 → 1.03 from 1,000 to 5,000 simulations, Section 12.1; L2 H4 about 1.0 in 100%-optimal cells). Where does the error enter?

**Design.** L1 H5, seeds 0–4 × Section 1's 11 beliefs (n = 55 problems per budget). Each root action's search Q error is split exactly into immediate, chance and continuation error (`analysis.error_attribution.describe_node`). `results/q_plateau_diagnosis_20261002/root_attribution.py`.

**Findings.**
- **The error is continuation error.** Immediate error is 0, since E[R | b_h, a] is exact, and chance error is at most 0.007 in mean. Every root action sampled all 6 observation branches, and no child was a rollout frontier.
- **It sits on clearly wrong actions.** The eight largest errors (2.2–5.7) are all on the door the tiger is likely behind (exact Q −79 to −99). These eight had 121–205 of 5,000 visits (door openings average 800–1,200), as UCB should give them, so the value of the 4-step game after the reset is estimated from few samples. "max_a |Q̂ − Q*|" takes its maximum over these actions.
- **The decision-relevant value converges.** Q error of the optimal action(s) falls with budget, while the all-action maximum barely moves:

| L1 H5 (n = 55) | 1,000 simulations | 5,000 simulations |
|---|---|---|
| Max Q error over all actions | 1.57 | 1.35 |
| Q error of the optimal action | 0.40 | 0.12 |

- **L2 shows the same pattern** (Section 11's rows, n = 30 per cell). Optimal-action Q error vs the all-action maximum:
  - H4, modeled 4,000: 0.44 vs 1.05;
  - H3, k = 6, compressed 64: 0.30 vs 1.01;
  - H4, modeled 100: 0.85–1.09 vs 2.4–2.6, where opponent-model error also enters.
- **Chance scoring is neutral on the optimal action** (0.120 → 0.120 at 5,000 simulations). Its +0.11 in Section 12.1 is on the poorly sampled wrong actions, where every root falls back on at least one action.

**Consequence for contribution 2.** Error bounds should be reported on the decision-relevant value: optimal-action Q error, or the value error |max Q̂ − V*|. The all-action maximum is still recorded, but it measures how well search has valued actions it rightly neglects.

### 12.3 L2 decisions over 380 problems per cell

**Design.** Six cells of Sections 9 and 11, each over 19 beliefs of i (0.05, 0.10, …, 0.95) × 5 private beliefs of j (0.085, 0.3, 0.5, 0.7, 0.915) × seeds 0–3, so n = 380 per cell (was 30). With k > 0, each seed also draws a different history (states, j's actions, i's observations; i's actions follow `oracle.HISTORY_ACTIONS`). Reference: the exact I-POMDP. Intervals are exact 95% Clopper–Pearson. `results/l2_wide_n_20261002` (launcher inside).

Exact beliefs at H3, k = 4 were left out: one problem took 279 s of planner CPU (Section 8's mean was 22 s), and compressed 64 is the planner default.

| Problem | Beliefs | Optimal | 95% interval | Mean max Q error | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|---|
| H3, k = 0 | exact | 380/380 | 99.0–100% | 0.05 | 0.05 | 0.3 |
| H3, k = 2 | exact | 379/380 | 98.5–100% | 0.09 | 0.06 | 2.7 |
| H3, k = 2 | compressed 64 | 373/380 | 96.2–99.3% | 0.21 | 0.14 | 1.6 |
| H3, k = 4 | compressed 64 | 380/380 | 99.0–100% | 0.13 | 0.13 | 1.8 |
| H4, root 250, modeled 4,000 | exact | 380/380 | 99.0–100% | 1.00 | 0.48 | 36 |
| H4, root 1,000, modeled 100 | exact | 370/380 | 95.2–98.7% | 1.45 | 0.46 | 2.2 |

**Findings.**
- **H3 decisions are near-optimal at this n.** The 8 misses (H3, k = 2) are near ties: each lost exactly the gap to the second-best action, 0.11–0.40.
- **The H4 misses at modeled 100 are confined to the four "corner" problems,** where both i (0.05 or 0.95) and j (0.085 or 0.915) are confident:
  - corner problems: 6/16 optimal (95% interval 15–65%);
  - every other problem: 364/364 (99.0–100%).
  The old grid's 80% (Sections 9 and 11) came from its problem mix: corners were 4 of its 15 problems, but are 4 of 95 here. A headline rate over a belief grid depends on the grid, so this cell is reported split. At modeled 4,000 the corners are 16/16. This is Section 5's opponent-model error, localized: j-L1 at a confident belief is mis-solved by 100 simulations.
- **Chance scoring** corrected 5 decisions and regressed 1 out of 2,280, all near ties.
- **Optimal-action Q error is 2–3× below the all-action maximum at H4** (Section 12.2).

### 12.4 Modeled budget, qualified at the beliefs j is modeled at

**Question.** Section 12.3's H4 misses at modeled 100 sit on the corner problems. Are they wrong j-L1 decisions, and what modeled budget removes them?

**Probe** (`results/modeled_budget_probe_20261002/probe.py`). On the 16 corner problems (H4, root 1,000, modeled 100), every j-L1 belief that i's search asked a modeled policy for was logged: 480 beliefs, 12–60 per problem. Each was re-solved by j-L1 I-POMCP at several budgets and scored against the exact L1 agent (depth 4). Re-solving at 100 reproduced the in-search policy in 480/480 cases.

| Modeled simulations | Wrong j-L1 decisions (of 480) |
|---|---|
| 100 | 10 |
| 400 | 6 |
| 1,600 | 0 |
| 4,000 | 0 |

All 10 wrong decisions at 100 fall in the problems where i's decision was wrong. In the problems i got right, 0 of 360 were wrong. They occur where j is confident but not certain (P(tiger left) in 0.05–0.15 or 0.85–0.95), where listening and opening are close for j.

**End to end** (wide grid, n = 380 per cell, root 1,000; `results/l2_h4_modeled_sweep_20261002`, with modeled 100 from 12.3):

| Modeled simulations | Optimal | 95% interval | Corners | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|
| 100 | 370/380 | 95.2–98.7% | 6/16 | 0.46 | 2.2 |
| 400 | 374/380 | 96.6–99.4% | 10/16 | 0.35 | 4.9 |
| 1,600 | 380/380 | 99.0–100% | 16/16 | 0.19 | 13.7 |

**Conclusion.** Qualifying j-L1 at the beliefs it is modeled at predicts the budget that i's decisions need: 1,600 here, not the 4,000 used so far. H4 with root 1,000 and modeled 1,600 is 100% optimal at 2.6× less CPU than root 250 and modeled 4,000 (13.7 s vs 36 s), with lower optimal-action Q error (0.19 vs 0.48).

### 12.5 Modeled-agent qualification over the whole grid (October 3, 2026)

**Design.** `experiments.modeled_qualification` (the 12.4 probe as a driver). For each L2 problem (19 beliefs of i × 5 of j × seeds 0–1, n = 190 problems per horizon), i searches with root 1,000 and modeled 100. Every j-L1 belief that i's search queries is recorded and re-solved over a budget ladder, then scored against the exact L1 agent. On every query, the driver checks that the rung equal to 100 reproduces the policy i used. `results/modeled_qualification_20261003` and `results/modeled_qualification_h4_6400_20261003` (6,400 rung).

| Horizon | j-L1 simulations | Wrong / queries | Distinct beliefs wrong / queried | Wrong in problems i got wrong | Max j loss |
|---|---|---|---|---|---|
| 3 | 100 | 19 / 3,325 | 1 / 87 | 0 | 0.46 |
| 3 | 400 | 0 / 3,325 | 0 / 87 | 0 | 0 |
| 4 | 100 | 152 / 8,284 | 6 / 235 | 6 | 1.01 |
| 4 | 400 | 95 / 8,284 | 4 / 235 | 4 | 0.66 |
| 4 | 1,600 | 38 / 8,284 | 2 / 235 | 0 | 0.32 |
| 4 | 6,400 | 0 / 8,284 | 0 / 235 | 0 | 0 |

**Findings.**
- **Mistakes cluster on a few beliefs.** At H4 and 100 simulations, 152 wrong queries come from 6 distinct beliefs.
  - The harmful one is j's **prior** (P(tiger left) = 0.085 or 0.915). j opens a door where the exact choice is to listen (Q −0.67 vs −0.01). The prior is queried in every problem with that j.
  - The two left at 1,600 (P(tiger left) = 0.077 or 0.924) are near ties: j listens where opening is better by 0.20–0.32. i is unaffected.
- **j's own loss does not separate harmful from harmless mistakes.** Harmful ones lose 0.66; harmless ones lose 0.2–1.0. No loss tolerance selects the budget i needs.
- **Zero wrong modeled decisions is the qualifying rule.** It is sufficient for i to be unaffected, and it uses only the modeled level's exact reference. That makes it usable where the higher level's exact solution is out of reach (L3 and beyond). It certifies 400 at H3 and 6,400 at H4 (0/8,284; correct rate ≥ 99.96%).
- **It is conservative.** At H4, i's decisions are already 380/380 at 1,600 (12.4), with 2 harmless near-tie beliefs left. Where the higher level's exact reference exists, it can lower the budget, as in 12.4. Where it does not, the certified budget is the honest choice.

### 12.6 Why near-tie modeled decisions need thousands of simulations

**Question.** At H4, wrong j-L1 queries fall 152 → 95 → 38 → 0 from 100 to 6,400 simulations (12.5). That is a 64× budget for a few near-tie beliefs. Is this a bug, a poor allocation, or estimator variance?

**Design.** j-L1 (depth 4) at P(tiger left) = 0.0767: exact Q is open-left 0.246, listen 0.043, open-right −92.9. Budgets 100–6,400 × seeds 0–19 (n = 20 per budget). Root visits and Q estimates are logged per action. `results/near_tie_diagnosis_20261003/near_tie.py`.

| Simulations | Correct | Visits listen / open-left / open-right | Search Q(open-left): mean ± sd | Scored Q(open-left): mean ± sd |
|---|---|---|---|---|
| 100 | 11/20 | 42 / 43 / 15 | 0.35 ± 1.00 | 0.30 ± 1.09 |
| 400 | 12/20 | 181 / 183 / 36 | 0.30 ± 0.50 | 0.28 ± 0.43 |
| 1,600 | 18/20 | 761 / 769 / 70 | 0.26 ± 0.22 | 0.26 ± 0.19 |
| 6,400 | 20/20 | 3,120 / 3,169 / 112 | 0.24 ± 0.07 | 0.27 ± 0.03 |

**Findings.**
- **No bug.** Estimates centre on the exact values and their spread shrinks as 1/√n.
- **No allocation problem.** UCB splits visits evenly between the two contenders and spends little on the clearly wrong door. Simple-regret root selection, such as Sequential Halving (Karnin et al. 2013, as used at the root of Gumbel MuZero, Danihelka et al. 2022), would allocate the same way here, so it was not tried.
- **The floor is return variance.** About 6.5 per simulation means a 0.2 gap needs thousands of visits per action. Only variance reduction moves this floor. Chance scoring already cuts the spread 2.5× at 6,400 (0.07 → 0.03).
- **Consequence.** A flat modeled budget pays the near-tie price at every queried belief, even though most beliefs are clear. Adaptive per-belief budgets would cut that cost: stop when the decision is statistically separated, spend more on near ties. Such a rule is not designed or tested yet.

### 12.7 First level-3 decisions against the exact L3 I-POMDP (October 3, 2026)

**Design.** `experiments.panel` at level 3: i-L3 models j-L2, who models i-L1, who faces uniform L0 j. Point beliefs: 9 of i (0.1–0.9) × 3 of j × 3 of the modeled i-L1 (0.085, 0.5, 0.915) × seeds 0–1, n = 162 per cell. Root 1,000; modeled 0 means every lower level is exact. Before the run, the L3 reference was checked against `build_exact_agent`'s independently assembled hierarchy: equal to 0.0 at H3. `results/l3_panel_20261003`.

| Horizon | Modeled simulations | Optimal | 95% interval | Optimal-action Q error | Max Q error | CPU (s) |
|---|---|---|---|---|---|---|
| 2 | exact | 162/162 | 97.7–100% | 0 | 0 | 0.16 |
| 3 | exact | 162/162 | 97.7–100% | 0.06 | 0.42 | 17 |
| 3 | 100 | 162/162 | 97.7–100% | 0.17 | 0.48 | 14 |
| 3 | 400 | 162/162 | 97.7–100% | 0.06 | 0.42 | 26 |

- **Every L3 decision checked is optimal**, with exact or finite modeled opponents.
- **At 400 simulations, every modeled j-L2 policy equals the exact one.** i's trees and values are identical to the exact-opponent cell to the last digit. At 100, 144 of 162 problems are identical, and Q error rises (0.17 vs 0.06) without changing a decision.
- **Mid-episode exact checks reach their limit at L3.** All four H3, k = 2 jobs exceeded the 4 GB memory limit. One such problem alone ran over 30 minutes and reached a 5 GB peak. At H2, k = 2 takes 6–8 s, but i's support there is 32 atoms, so compressed 64 equals exact and nothing is tested. H2 at k = 4 with exact beliefs did not finish in 26 minutes.
- **Tractable frontier for exact L3 decision checks (Tiger): H3 at k = 0 and H2 at k ≤ 2.** Compressed-belief error at L3 must be measured as belief error against exact filtering (`belief_qualification`), not as decisions.

### 12.8 Sequential replicate searches for modeled policies (October 3, 2026)

*Superseded by 12.11: this rule is biased for j-L2 at L3 H4 (12.10) and was replaced by a growing-tree rule. The `6400s` stacks of 12.8–12.10 are this old rule.*

**Design change** (commit `c0cba21`, `planning.replicates`, opt-in through `PlannerConfig.modeled_stopping`). A modeled policy is solved by independent replicate searches of 200 simulations, a form of root parallelization (Chaslot et al. 2008). After at least 3 replicates, the solve stops once the two best mean scored values are 3 standard errors apart, which is fixed-confidence best-arm identification (cf. LUCB, Kalyanakrishnan et al. 2012). The cap is `modeled_simulations`. Search seeds exclude the rule, so single-search streams and all earlier results are unchanged. Motivation: 12.6 shows the near-tie floor is return variance, so spending adaptively is the lever.

**Offline A/B** (235 distinct j-L1 beliefs queried at L2 H4 × seeds 0–4, n = 1,175; `results/sequential_replicates_ab_20261003`):

| Modeled solve | Correct | Mean simulations |
|---|---|---|
| single search, 400 | 1,171/1,175 | 400 |
| single search, 1,600 | 1,175/1,175 | 1,600 |
| single search, 6,400 | 1,175/1,175 | 6,400 |
| replicates with stopping, cap 1,600 | 1,172/1,175 | 627 |
| replicates with stopping, cap 6,400 | 1,175/1,175 | 699 |

At an equal 1,600 cap, replicates are worse on near ties (17/20 vs 20/20), because small trees lose depth. The gain comes from adaptivity: about 4,070 simulations on the 4 near-tie beliefs, about 600 elsewhere.

**Gate on i's decisions** (L2 H4, root 1,000, the 380-problem grid of 12.3; `results/l2_h4_stopping_gate_20261003`):
- **Accuracy.** Stopping with cap 1,600 or 6,400 is 380/380 optimal (99.0–100%), the same as flat 1,600. Optimal-action Q error is 0.16 vs 0.19.
- **Cost, paired.** For 40 problems, both arms ran back to back in one process, in alternating order. i's decision CPU with stopping (cap 6,400) is 0.47× flat 1,600 (median 0.47; Wilcoxon p = 2×10⁻¹²): 9.8 s vs 20.6 s under that machine load. Per modeled solve it is 0.16 s vs 0.37 s, with 3.5 replicates on average and no per-replicate overhead.
- **The unpaired panel CPU is not comparable.** The gate panel showed 13.1–13.8 s against 13.7 s for arm A, but arm A ran on a quieter machine (another project's 8 jobs were running during the gate). CPU comparisons across runs at different load are invalid; compare only paired, same-process measurements.

**L3 gate** (L3 H3, root 1,000, stopping with cap 1,600 for both modeled levels, the 162 problems of 12.7; `results/l3_stopping_gate_20261003`):
- **Accuracy.** 162/162 optimal. Values are identical to the exact-opponent cell, so every modeled policy matched the exact one.
- **Cost, paired** (16 problems, alternating order). Stopping costs 1.30× flat 400 (median 1.23; Wilcoxon p = 3×10⁻⁵): 37.6 s vs 29.5 s. Flat 400 already reproduces the exact policies at L3 H3 (12.7), while the rule's floor is 3 replicates × 200 = 600 simulations. That floor applies at each nested level.

**Verdict.**
- **Use stopping where the qualified flat budget is well above its 600-simulation floor.** At L2 H4 it halves CPU at equal accuracy, and caps near-tie error at the level flat 6,400 reaches (12.5).
- **Use a flat budget where a small one already qualifies.** At L3 H3, flat 400 matches exact and the rule costs 1.3×.
- **It stays opt-in, chosen per modeled level.** It is not a default.

### 12.9 Per-level modeled solves, and the first level-3 modeled qualification (October 3–4, 2026)

**Design change** (commits `2700e7a`, and the oracle's `reference="none"`). `experiments.modeled_levels` names each modeled level's solve, top-down, with one label per level: `400` (one search of 400), `6400s` (sequential stopping, cap 6,400; 12.8), `exact`. At level 3, `6400s/100` solves j-L2 with stopping and the i-L1 that j models with a flat 100. `oracle`, `panel` and `modeled_qualification` build the hierarchy from that label. `modeled_qualification` now also runs at level 3: it scores j-L2 at the beliefs i-L3's search queries, against the exact L2 agent, and each ladder rung is a whole stack. Its qualifying rung is the cheapest one, by mean CPU per query, with zero wrong decisions. Only the modeled level's exact reference is needed, so `score_decision=False` drops i's level-3 reference where it is intractable.

**L3 H3 qualification** (12.7 grid: 9 beliefs of i × 3 of j × 3 of the modeled i-L1 × seeds 0–1, n = 162 problems; root 1,000; in-search stack `400`; 3,294 queried j-L2 models, 183 distinct). `results/l3_modeled_qualification_h3_20261003` and `results/l3_modeled_qualification_h3_ladder2_20261004`.

| j-L2 / i-L1 | Wrong / queries | Distinct beliefs wrong | 95% interval, correct | CPU per query (s) |
|---|---|---|---|---|
| 100 / 100 | 63 / 3,294 | 6 | — | 0.65 |
| 100 / 400 | 63 / 3,294 | 6 | — | 1.09 |
| 400 / 100 | 36 / 3,294 | 4 | 98.5–99.2% | 1.01 |
| 400 / 400 | 36 / 3,294 | 4 | 98.5–99.2% | 1.45 |
| 1,600 / 100 | 9 / 3,294 | 1 | 99.5–99.9% | 1.71 |
| **6,400 stopping / 100** | **0 / 3,294** | **0** | **99.9–100%** | **1.28** |

CPU is comparable only within a run: rows 1–5 ran in one run, the last two in the second.

- **i's decisions were optimal in every problem (162/162).** All the wrong j-L2 queries are harmless to i. This agrees with 12.7, which scored i's decisions, not j's.
- **i-L1's budget inside j-L2 does not matter at H3.** At 100 and at 400, every j-L2 policy is identical (3,294 of 3,294 queries), so `/100` is enough and costs about 30% less.
- **The j-L2 mistakes are near ties.** The 4 beliefs still wrong at 400 have P(tiger left) ≈ 0.03 or 0.97, with exact gaps of 0.08–0.15 between listening and opening the safe door. Re-solved over 10 seeds (`results/l3_near_tie_probe_20261003`), flat 400 is wrong 12/40, while flat 1,600 and flat 6,400 are 0/40, with i-L1 exact or at 400 alike. That is search noise, not a bug.
- **Certified L3 H3 stack: `6400s/100`.** It has zero wrong queries, and it costs 0.75× flat 1,600, which still has one wrong belief.

**Stopping floor A/B** (`results/stopping_floor_ab_20261004`). With 3 replicates, the stopping test's standard error has 2 degrees of freedom, so a 3-SE separation occurs by chance with two-sided probability ≈ 0.095. Is the floor too low?

| Minimum replicates | L2 H4 (n = 1,175): wrong, mean simulations | L3 H3 near ties (n = 40): wrong, mean simulations |
|---|---|---|
| 3 (current) | 0, 695 | 1, 4,570 |
| 4 | 0, 894 | 1, 4,955 |
| 5 | 0, 1,091 | 1, 5,245 |

The one wrong case (gap 0.083) reached the 6,400 cap in every arm, so it is a cap limit, not an early stop. Raising the floor costs 29–57% more simulations and corrects nothing, so **the floor stays at 3**. Lowering it to 2 replicates is unsound: the SE has 1 degree of freedom and a 3-SE separation occurs by chance about 20% of the time. It was not run.

### 12.10 Level-3 horizon-4 qualification: the stopping rule is biased for j-L2 (October 4–5, 2026)

**Design.** `modeled_qualification` at L3 H4 without i's exact L3 reference (`score_decision=False`), which is intractable here. Grid: 3 beliefs of i (0.1, 0.5, 0.9) × 3 of j × 3 of the modeled i-L1 × seed 0, n = 27 problems; root 1,000; in-search stack `400/100`. The 9 problems with nested belief 0.5 are in `results/l3_modeled_qualification_h4_20261004`. The 18 with a confident nested belief (0.085 or 0.915) are in `results/l3_modeled_qualification_h4_confident_20261004`, run with the two-phase driver below.

| Problems | j-L2 / i-L1 | Wrong / queries | Distinct beliefs wrong / queried | Max j loss |
|---|---|---|---|---|
| nested 0.5 (9) | 400 / 100 | 0 / 633 | 0 / 211 | 0 |
| nested 0.5 (9) | 6,400 stopping / 100 | 0 / 633 | 0 / 211 | 0 |
| nested 0.085, 0.915 (18) | 400 / 100 | 90 / 1,998 | 30 / 666 | 0.98 |
| nested 0.085, 0.915 (18) | 6,400 stopping / 100 | 24 / 1,998 | 8 / 666 | 0.30 |

Neither stack qualifies at L3 H4. Every wrong query is at a confident j belief (P(tiger left) 0.03–0.07 or 0.93–0.97), where listening and opening the safe door are close (gaps of 0.03–0.30). Confident nested beliefs make i's problem much larger: 103 queries of up to 86 atoms, with i's search taking 2,224 s and the exact L2 reference 140 s per belief, against 67 queries, 50 atoms, 600 s and 50 s at nested 0.5.

**Residual probe** (the 8 beliefs still wrong at `6400s/100`, re-solved over seeds 0–9, n = 80 per row; `results/l3_h4_residual_probe_20261005`):

| j-L2 / i-L1 | Wrong / 80 | Wrong on the three 0.19-gap beliefs / 30 | Mean CPU (s) |
|---|---|---|---|
| stopping, cap 6,400 / 100 | 54 | 18 | 112 |
| stopping, cap 6,400 / exact | 57 | 20 | 93 |
| stopping, cap 25,600 / exact | 51 | 16 | 149 |
| one search, 6,400 / exact | 14 | 0 | 157 |

- **The i-L1 budget is not the cause.** An exact i-L1 gives the same error rate as 100 simulations (57 vs 54 of 80).
- **The stopping rule is biased here, not merely noisy.** On the 0.19-gap beliefs one search of 6,400 is 0/30 wrong, while replicate searches stay 16–20/30 wrong even with a 4× higher cap. Each replicate is a 200-simulation tree, too shallow at depth 4 for j-L2. Averaging many of them converges to the shallow trees' value, and more replicates only make that biased answer more confident. 12.8 noted that small trees lose depth. At L2 H4 (j-L1) and L3 H3 (j-L2) the bias was too small to change a decision, but at L3 H4 it is not.
- **One search of 6,400 still misses the near ties** (gaps 0.03–0.04, 14/80). These are close to exact ties.

**Consequences.**
- The 12.8 rule (replicates of 200, cap 6,400) is not a consistent estimator, and must not be used beyond the cells where it was gated (L2 H4, L3 H3).
- L3 H4 has no certified modeled stack yet. Its j-L2 needs either a consistent adaptive rule or flat budgets above 6,400. Whether any of these j mistakes changes i's decision cannot be checked exactly at L3 H4.

**Engineering for this run** (commits `f2754fa`, `bf0608e` and later):
- `FiniteBelief` unpickling now restores masses bit for bit. Renormalizing them had changed 504 of 549 canonical digests, and so search seeds, which flipped 5 near-tie policies. Earlier results never sent beliefs between processes, so they are unaffected.
- `modeled_qualification` runs in two phases: a search per problem, then a job per distinct (horizon, seed, belief). This is exactly equivalent to the old driver (1,098 of 1,098 rows on 27 L3 H3 problems) at a third of the scoring work.
- The exact reference is built fresh per belief. A shared cache had grown to 9.6 GB.

### 12.11 A consistent stopping rule: one growing tree, calibrated by pilot trees (October 5, 2026)

**Question.** 12.10 showed that averaging 200-simulation replicates converges to a shallow tree's biased value. Can an adaptive modeled solve be consistent, as accurate as flat 6,400 at L3 H4, and cheaper?

**Diagnosis first** (`results/single_tree_bias_20261005`; the 8 residual beliefs of 12.10 × seeds 0–9, n = 80 per checkpoint). One j-L2 tree over an exact i-L1 was grown to 6,400 and chance-scored at 200, 400, …, 6,400:
- On the three 0.19-gap beliefs, wrong decisions fall 13/30 → 13/30 → 8/30 → 9/30 → 5/30 → 0/30.
- The spread of the scored gap across seeds falls from about 1.0 at 200 to about 0.06 at 6,400, roughly n^-0.8. Averaging k replicates reduces spread only as k^-0.5, so depth beats replication on both bias and variance.
- Replaying k = 2–4 independent growing trees on these beliefs: at an equal total budget they are worse than one tree (gap 0.19: 3–4% wrong with three trees of 3,200, against 0/30 with one tree of 6,400).

**Design change** (commit `f5287f5`, `planning.stopping`, which replaces `planning.replicates`):
- **One main tree.** It is seeded exactly as the flat search of the cap and grown by doubling from a pilot size `b`. The decision is always the main tree's, so at the cap the rule *is* the flat search, and it is consistent.
- **A calibrated test.** A tree with Bellman backups has no honest standard error, so `m − 1` independent pilot trees of `b` simulations measure one. With the main tree's checkpoint at `b`, they give σ, the spread over `m` trees of the scored difference between the main tree's two leading actions. At checkpoint n the main tree stops when its gap exceeds t(1 − α/2K, m − 1) · σ · sqrt(b/n). K is the number of checkpoints (Bonferroni). sqrt(b/n) assumes variance falls at least as 1/n (measured: about n^-1.6), so the test is conservative.
- **Defaults:** b = 200, m = 4, α = 0.01. Labels keep their form: `6400s` (default) or `6400s[200,4,0.01]`.

**Offline replay at L3 H4** (`results/l3_h4_tree_trajectories_20261005`). All 666 distinct j-L2 beliefs queried in the 18 confident-nested problems of 12.10 come from the 6 problems with i's belief 0.1, since i's own belief does not change which j beliefs it queries. Each belief has one main tree grown to 6,400 and 3 pilot trees of 200 and 400. Each problem's queries were solved in query order in one bank, as inside i's search, so CPU is the in-situ marginal cost. Rules replayed from these checkpoints give exactly the decisions they would make online with these seeds. n = 666; gap bands are exact gaps between the two best actions.

| Rule | Wrong | gap < 0.05 (8) | 0.05–0.3 (34) | 0.3–1 (68) | ≥ 1 (556) | Mean simulations | In-situ CPU per query (s) |
|---|---|---|---|---|---|---|---|
| flat 400 | 30 | 4 | 12 | 13 | 1 | 400 | 33 |
| flat 1,600 | 15 | 3 | 7 | 5 | 0 | 1,600 | 76 |
| flat 6,400 | 6 | 5 | 1 | 0 | 0 | 6,400 | 178 |
| **stopping, b 200, m 4, α 0.01** | **6** | 5 | 1 | 0 | 0 | 2,584 | **92** |
| stopping, b 200, m 4, α 0.05 | 6 | 5 | 1 | 0 | 0 | 2,211 | 83 |
| stopping, b 400, m 4, α 0.01 | 6 | 5 | 1 | 0 | 0 | 3,162 | 122 |
| stopping, b 200, m 3, α 0.01 | 6 | 5 | 1 | 0 | 0 | 3,189 | 109 |

- **Same decisions as flat 6,400 at 0.52× its CPU.** Every remaining miss runs to the cap, where the rule is flat 6,400. Five are near-exact ties (gaps 0.03–0.04); one has a gap of 0.14.
- **Where the main tree stops:** 334 beliefs at 200, 63 at 400, 47 at 800, 29 at 1,600, 28 at 3,200, and 165 at the cap.
- **CPU is front-loaded.** In situ, a query's first 200 simulations cost about 19 s, while its new nested i-L1 beliefs are solved. Flat 6,400 costs 178 s per query on average and up to 340 s, so it is about 5 CPU-hours per i decision at L3 H4 (about 100 queries). The earlier estimate of "hours" stands; a probe of the first 24 queries (16–20 s each, `results/shared_bank_cost_20261005`) had understated the later ones.

**Online checks of the implementation** (`results/stopping_redesign_gate_20261005`, `results/stopping_redesign_gates_20261005`):

| Set | n | New rule | Reference |
|---|---|---|---|
| L3 H4 residual beliefs × seeds 0–9 (12.10) | 80 | **14 wrong; 0/30 on the 0.19-gap beliefs** | old rule 57 (20/30); flat 6,400 14 (0/30), the same per belief |
| L2 H4 j-L1 beliefs × seeds 0–4 (12.8) | 1,175 | 0 wrong | flat 1,600 0 wrong |
| L3 H3 j-L2 near ties × seeds 0–9 (12.9) | 40 | 0 wrong | flat 1,600 0; old rule 1 |
| L3 H3 qualification, rung `6400s/100` (12.9 grid) | 3,294 queries (183 beliefs) | **0 wrong** (99.89–100%) | old rule 0 wrong |
| i's decisions, L2 H4, root 1,000, modeled `6400s` | 380 | **380/380** (99.0–100%); optimal-action Q error 0.155 | flat 1,600 380/380, 0.19 |
| i's decisions, L3 H3, root 1,000, stack `6400s/100` | 162 | **162/162** (97.7–100%); optimal-action Q error 0.06 | exact opponents 0.06 |

Paired CPU, measured in one process with alternating order and fresh banks:
- **L2 H4 j-L1** (n = 1,175): 0.37 s per solve against 0.44 s for flat 1,600, with 1,127 simulations on average. With α = 0.05 it is 0.28 s.
- **L3 H3 j-L2** (n = 183 beliefs, cold banks): 3.67 s against 3.77 s for flat 1,600/100 (median ratio 0.93, Wilcoxon p = 1×10⁻⁸) and 3.28 s for flat 400/100 (1.08×). In this cold setting flat 1,600/100 has 1 wrong belief and flat 400/100 has 3.

**Verdict.**
- **The rule replaces 12.8's.** It is consistent and as accurate as flat 6,400 wherever it was checked, and it costs 0.5× flat 6,400 in situ at L3 H4 and about 0.9× flat 1,600 at L2 H4 and L3 H3. It is still opt-in, chosen per level.
- **Certified stacks:** `6400s` at L2 H4 and `6400s/100` at L3 H3 (re-gated above).
- **L3 H4 still has no zero-wrong stack.** Its remaining misses are flat 6,400's misses: near-exact ties (gaps 0.03–0.04) and one 0.14 gap. Fixing them takes a larger cap or variance reduction (HANDOFF step 3), not a different stopping rule.

**Also measured, kept out of the code:**
- Memoizing the hashes of frames, models and atoms gives bit-identical results but costs 8% more CPU. Most of these objects are hashed only once or twice.
- A larger exact-reference cache changes nothing: every one of a solve's 173k i-L1 reference nodes is a distinct belief.
- Persistent nested trees (R6) cannot be reused exactly. Every modeled query is a fresh receding-horizon problem of depth H, while a subtree of an earlier j tree has fewer steps left.

### 12.12 Simulating minds, 2.2× faster in situ with outputs bit-identical (October 5, 2026)

**Profile first** (`results/w1_profile_20261005`, cProfile of one confident-nested L3 H4 problem's j-L2 queries in one bank, stack `1600/exact`). In situ, the exact i-L1 best response is 19% of CPU, not the 60% measured earlier on one cold solve. j-L2's own belief updates take 64%. On the later queries (21–40), one function took 33% on its own: `MentalModel` validation, which re-scanned the whole support on every one of 67M constructions.

**Changes** (commits `e17ab33`, `dd56728`, `5454665`):
- `AgentFrame` and `MentalModel` cache their hashes. The pickled state stays the declared fields, and hashes are recomputed on load.
- `FiniteBelief` summarizes its opponents once, so `MentalModel` validation is O(1).
- `InteractiveFilter.advance` memoizes successor models. Atoms built on one successor share it and compare by identity.
- `FiniteBelief` validation is one pass, and equality has an identity shortcut. Frozenset equality walks the support even for the same object.
- The enumerating filter groups rows by atom as it builds them (`FiniteBelief.from_groups`). This is bit-identical because `fsum` is correctly rounded.

Kept out: grouping inside the exact reference (no measurable gain), and memoizing Tiger's kernels (a lookup costs as much as the kernel).

**Gate** (`gate_w1.py`): every chance-scored tree's values, compared with `repr`. The old and new code ran concurrently under the same load.

| Set | n | Identical | CPU, new / old | Wilcoxon p |
|---|---|---|---|---|
| L3 H4 in situ, first 40 queries of problem 0, `1600/exact` | 40 queries | 40/40 | **0.45** (median 0.46; last 20 queries 0.43) | 2×10⁻¹² |
| L3 H3 cold, the 183 queried j-L2 beliefs, `6400s/100` | 183 (26,673 trees) | 183/183 | 0.84 (median 0.84) | 4×10⁻²⁷ |

The speedup grows with belief size, because the removed costs scaled with the support. An i-L3 H4 decision at `6400s/exact` should fall from about 2.5 CPU-hours to about 1.1. That figure is extrapolated from the per-query ratio, not measured on a whole decision.

### 12.13 Are near-tie mistakes of the modeled opponent harmful to i? (October 5, 2026)

**Design** (`results/w2_near_tie_harm_20261005/harm.py`). In i's exact I-POMDP, the directly modeled j level is perturbed adversarially: at every queried belief whose exact gap lies in (tie, ε), j plays its second-best action. i's exact decision under this hierarchy is then scored against i's true exact Q. Problems: L2 H4, the 95 distinct k = 0 problems of the wide grid; L3 H3, the 81 distinct problems of 12.7. ε ∈ {0.05, 0.15, 0.3}. A perturbed run is skipped, exactly, when no queried gap is below ε, since nothing can flip. 0.8 CPU-h.

| Problems | ε | Problems with a flip | Flipped j beliefs | i's decision changed | i's loss |
|---|---|---|---|---|---|
| L2 H4 (95) | 0.05, 0.15 | 0 | 0 | — | — |
| L2 H4 (95) | 0.3 | 38 | 38 | 0 | 0 |
| L3 H3 (81) | 0.05 | 0 | 0 | — | — |
| L3 H3 (81) | 0.15, 0.3 | 36 | 72 | 0 | 0 |

- **No flip of a j decision with gap up to 0.3 changed i's decision**, in 74 problems with 110 flipped beliefs. This holds even with all of a problem's near ties flipped at once.
- **Limit:** these exact hierarchies have no j gaps below 0.05. The 0.03–0.04 ties of L3 H4 (12.11) lie below anything tested here, so calling them harmless would extrapolate.
- **For the certification standard** (HANDOFF 0.1): the evidence supports a PAC-style standard with ε up to 0.3 at L2 H4 and L3 H3. Adopting it is the user's decision. 12.14 attacks the near ties directly.

### 12.14 The near-tie floor was the chance-scoring fallback: exact scoring with two steps left (October 5–6, 2026)

**Question** (PLAN W3). 12.6 attributed the near-tie floor of modeled solves to return variance and suggested variance reduction. Which part of a scored value's spread is it?

**Diagnosis on fixed trees** (`results/w3_variance_20261005`, `a0_frontiers.py`, `a0b_fallbacks.py`). j-L1 at L2 H4 (depth 4) at the 8 queried beliefs with exact gaps 0.2–1.0, seeds 0–29 (n = 60 per gap band). One tree per (belief, seed) at 400, 1,600 and 6,400 simulations. Each tree is scored several ways, so the comparison is exactly paired.

| Gap | Simulations | Spread of the scored gap: planner | frontiers exact | missing branches exact at two steps left | all missing branches exact (oracle) | Wrong: planner / two steps left |
|---|---|---|---|---|---|---|
| 0.205 | 400 | 0.61 | 0.55 | 0.55 | 0.36 | 18 / 13 of 60 |
| 0.205 | 1,600 | 0.25 | 0.25 | **0.03** | 0.02 | 15 / **0** of 60 |
| 0.205 | 6,400 | 0.053 | 0.053 | **0.000** | 0.000 | 0 / 0 of 60 |
| 0.320 | 1,600 | 0.25 | 0.26 | 0.02 | 0.02 | 3 / 0 of 60 |
| 0.656 | 1,600 | 0.26 | 0.26 | 0.02 | 0.02 | 0 / 0 of 60 |
| 1.008 | 1,600 | 0.23 | 0.23 | 0.02 | 0.01 | 0 / 0 of 60 |

- **Rollout noise is not the floor.** Valuing every frontier exactly leaves the spread unchanged from 1,600 simulations on.
- **The floor is whole-action fallbacks.** A tree keeps about 500 fallback (history, action) pairs at 1,600 simulations, nearly all with two steps left. An action falls back when one of its 6 observations was never sampled, and it then keeps its noisy search Q.
- With two steps left, every child is the last decision, whose exact value max_a E[R | b, a] needs only the child's posterior. Valuing those branches exactly cuts the spread 8.5× at 1,600, and the optimal-action Q error falls from 0.15 to 0.00. It is exact backward induction, so it adds no bias. This is unlike renormalizing over the sampled branches (Section 6).

**Design change** (`planning.chance_scoring`): an action with two steps left is scored over all its positive branches, sampled or not. Fallback remains only with three or more steps left. Search, UCB and random streams are unchanged.

**Gates** (new code against the code of 12.12):

| Set | n | Before | After |
|---|---|---|---|
| **L3 H4 residual beliefs × seeds 0–9, `6400s/exact`** (12.10) | 80 | 14 wrong (11/30 at gap 0.03, 3/10 at 0.039) | **4 wrong** (4/30 at gap 0.03, 0/10 at 0.039); 12 fixed, 2 broken, exact McNemar p = 0.013; max loss 0.030 |
| L3 H4 residual, flat `1600/exact` | 80 | — | 24 wrong |
| L2 H4 j-L1 beliefs × seeds 0–4 (12.8), every arm | 1,175 | 0 wrong | 0 wrong |
| L3 H3 j-L2 near ties × seeds 0–9 (12.9), every arm | 40 | 0 wrong | 0 wrong |
| L3 H3 183 queried j-L2 beliefs, `6400s/100`, cold | 183 | 0 wrong | 0 wrong, every policy unchanged |
| i's decisions, L2 H4, root 1,000, modeled `6400s` (`g3_decisions`) | 380 | 380/380; optimal-action Q error 0.155 | **380/380** (99.0–100%); optimal-action Q error **0.117** |
| i's decisions, L3 H3, root 1,000, stack `6400s/100` | 162 | 162/162; optimal-action Q error 0.06 | **162/162** (97.7–100%); optimal-action Q error **0.0006**; 83% of i's scored roots exact |

**Cost.** Scoring now needs posteriors for branches search never sampled. At L3 those posteriors bring new nested i-L1 beliefs to solve. Paired on the 183 L3 H3 beliefs (cold, same load): 1.18× CPU (median 1.20, Wilcoxon p = 5×10⁻²⁸). The residual set ran at 114 s per solve, unpaired, against 146 s before 12.12's speedups.

**Verdict.**
- **Kept as the planner's scoring.** It roughly halves L3 H4's residual misses (14 → 4 of 80, p = 0.013) and leaves no regression on any certified set. Q errors at i's root fall, at L3 H3 by 100×.
- **What remains at L3 H4:** misses only at the 0.03-gap beliefs, with loss ≤ 0.030. Every exact check (12.13) found such mistakes harmless to i, but those checks never contained a gap that small.
- **Follow-up: the stopping rule is now conservative.** Its pilots (200 simulations) calibrate σ and scale it as sqrt(200/n). Scored spreads now collapse much faster than 1/n, so near-tie solves still run to the cap (L2 H4 near ties: 6,760 simulations on average). A rule that also uses the scored values' exactness, or certified intervals (PLAN W6), could stop them far earlier.

### 12.15 The stopping rule scales with the measured rate, 1/n (October 6, 2026)

**Question.** After 12.14, near-tie modeled solves still ran to the cap. The rule scales the pilots' spread at 200 simulations to the main tree's size n as sqrt(200/n), the Monte Carlo rate. Is that still the right rate?

**Dataset** (`results/stopping_v2_20261006/dataset.py`). 2,016 solves, each in a fresh bank:
- L3 H4: all 110 queried j-L2 beliefs with exact gaps below 1, 110 clear ones, and the 8 residual beliefs × seeds 0–9;
- L3 H3: the 183 j-L2 beliefs × seeds 0–2;
- L2 H4: the 235 j-L1 beliefs × seeds 0–4.

Each solve has a main tree scored at 200–6,400 simulations, plus pilot trees. A rule replayed from these checkpoints (`replay.py`) decides exactly as it would online with these seeds.

**Measured rate.** Across 10 seeds at each of the 8 residual beliefs, the spread (sd) of the main tree's scored gap fell as n^−1.0 between 200 and 3,200 simulations (median; range 0.76–1.15).

**Replay** (wrong decisions / mean simulations; the threshold scales as (200/n)^β):

| Rule | L3 H4 (n = 292) | L2 H4 (n = 1,175) | L3 H3 (n = 549) |
|---|---|---|---|
| flat 6,400 | 4 / 6,400 | 0 / 6,400 | 0 / 6,400 |
| β = 0.5 (12.11's rule) | 4 / 5,081 | 0 / 1,424 | 0 / 1,283 |
| **β = 1.0 (measured)** | **4 / 4,332** | **0 / 1,103** | **0 / 1,107** |
| β = 1.25 | 4 / 3,622 | 0 / 1,033 | 0 / 1,000 |
| β = 1.5 | 7 / 2,971 | 0 / 980 | 0 / 937 |
| β = 2.0 | 22 / 2,000 | 0 / 934 | 0 / 879 |
| flat 1,600 | 37 / 1,600 | 0 / 1,600 | 0 / 1,600 |

The 4 L3 H4 misses are flat 6,400's own, all at exact gaps below 0.05. β = 1.25 is the steepest rate with no added mistakes, but choosing it would tune on the test data. The measured rate, β = 1.0, is adopted (commit `d7ba56c`).

**Online.**
- In situ, the first 40 queries of L3 H4 problem 0 (`6400s/exact`, the old and new rules run concurrently): 40/40 policies equal; CPU 0.88× in total, median 0.95 per query, Wilcoxon p = 0.09. In situ, CPU is dominated by solving new nested beliefs, so 15% fewer simulations save little CPU.
- The 12.11 near-tie sets, online (`gate/near_ties.py`): `6400s` is 0 wrong on the L2 H4 j-L1 set (n = 1,175) and on the L3 H3 near ties (n = 40), as in the replay. Mean simulations at L2 H4 fell from 1,199 (12.14) to 882, a 26% cut (23% in the replay). This harness counts simulations by tree `id()`, which freed pilot trees can reuse, so its absolute counts run about 220 below the replay's for both rules.

**Verdict:** same decisions, 14–23% fewer simulations, and a modest in-situ CPU saving that is not significant.

### 12.16 Gates for the episode study (October 6, 2026)

The study (`studies/tiger_l3_vs_l2.toml`) now runs at horizon 3 with root 1,000 and compressed 64-atom beliefs. Modeled levels are named per level (`AgentSpec.modeled`): `6400s/100` for i and `400` for j. Its mixed-level priors had never been gated.

- **Mixed priors vs the exact I-POMDP** (`results/w4_mixed_prior_gate_20261006`, seeds 0–19):
  - **Problems:** i-L3 at its prior; j-L2 at its prior and after 2 steps (4 histories).
  - **Result:** 120/120 optimal, with optimal-action Q error 0–0.09.
  - **Limit:** every exact gap here is at least 1.6, so this checks the hierarchy's construction, not near-tie accuracy.
- **Compressed beliefs at level 3** (`results/w4_belief_l3_20261006`, `belief_qualification --steps 5`): 12 histories under the study's priors, against exact beliefs whose supports reach 4,328 atoms by step 5.

  | N | Level TV, mean | Level TV, max | Steps above 0.2 | Nested TV, mean | Physical error, mean |
  |---|---|---|---|---|---|
  | 64 | 0.012 | 0.061 | 0 of 60 | 0.010 | 0.002 |
  | 256 | 0.003 | 0.017 | 0 of 60 | 0.003 | 0.0001 |

  At step 8, the exact L3 belief outgrew memory (stopped at 3.3 GB), so exact checks end at step 5.
- **Pilot** (`results/w4_episode_pilot_20261006`, 2 trials × 20 steps):
  - **CPU:** i's planning took 3,063 s and 480 s per trial, j's about 90 s. i's first steps cost 30–240 s each, while later steps reuse cached nested policies.
  - **Memory:** peak RSS 1.8 GB.
  - **Implication:** the full run (100 trials, 6 workers) needs about 50 CPU-h, within 3× of the plan's estimate.
  - **Bug found:** episode rows had empty `action_values`, because rows were written after the belief update cleared them. Fixed in commit `14ac0e2`, with a regression test.
- **Also recorded per step** (W6, W8): each decision's certified bounds, and each agent's predicted distribution of the opponent's next action (`docs/EQUILIBRIA.md`).

### 12.17 Certified bounds on every real decision (October 6, 2026)

`planning.chance_scoring(..., bounds=True)` returns, for each root action, an interval that contains the exact Q of the searched model, whatever the budget. Exact pieces enter exactly. Rollout frontiers, unsampled branches and untried actions are bounded by the declared reward bounds (THEORY's chance-scoring section; commit `95d11d5`).

- **Checks:** containment and tightness in 15 cases (L1 Tiger, depths 2–4, 3–800 simulations, 3 beliefs).
- **Where recorded:** `oracle` rows (`bound_width`, `bounds_contain_reference`) and episode rows (`bounds_<agent>`).
- **Scored values are unchanged,** and the cost is negligible, because successors are already cached.
- **What the bounds certify:** search error relative to the searched model. When the reference includes opponent-model or belief error, containment is not guaranteed and is reported as measured.
