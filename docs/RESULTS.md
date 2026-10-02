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

## 6. I-POMCP vs IPF + RTS (L2 Tiger, vs the exact I-POMDP; sampled IPF)

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
- Driver: `experiments.l2_panel`; data in `results/l2_mid_episode_20261002`.

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
- n = 11 decisions over 3 seeds; `results/wumpus_split_pilot_20261002`.

**Measured:**
- **Exact filtering is cheap.** Along one history the support shrank from 94,500 to 46 atoms in 8 steps, at most 2.9 s per update (`results/wumpus_exact_filtering_20261002`).
- **The exact reference is tractable at H3:** 520–600 s at the prior, 92–101 s after 2–4 steps.
- **Planning took 1.2–34 s per decision.**
- **Losses were 0 in 3 of 11 decisions and 1.8–7.7 in the rest; mean max Q errors were 3–89** (rewards range over ±1,000).

**Status: not qualified.** The split removes the failure mode but is not accurate yet. The likely causes, not yet separated, are:
- compressing a 25,920–94,500-atom belief to 64–256 atoms;
- 1,000 simulations over Wumpus's 5 actions and 32 observations.

The next rung is a qualification grid over N and simulations against the exact reference.
