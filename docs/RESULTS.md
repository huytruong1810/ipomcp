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
