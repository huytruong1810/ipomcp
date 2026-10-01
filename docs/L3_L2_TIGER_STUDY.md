# L3 vs L2 interactive Tiger: design, validation and convergence ladder (September 30)

Branch `claude/l3-l2-tiger-experiment`, worktree `../ipomcp-claude`. `main` is untouched.
All runs set `PYTHONPATH=$PWD/src`: the shared `.venv` otherwise imports `main`'s source.

## Declared design

- **Physics.** Two-agent Tiger with growl accuracy .85 for both agents and creak accuracy .90.
  - The creak law is this repository's: a listening opponent never creaks. An opening opponent creaks on the correct side .90, the wrong side .05, and is silent .05.
  - An agent that opens a door itself hears (S, S).
  - Opening resets the tiger uniformly. Rewards: -1 listen, +10 treasure, -100 tiger.
- **Agent i: L3.** Prior over j's level: L1 .8, L2 .1, L0 .1.
  - Its modeled j/L2 holds {L1 .5, L0 .5} over i, which matches the real j. This assumes i knows j's level prior.
  - Every modeled L1 holds uniform L0 opponents.
- **Agent j: L2.** Prior over i's level: L1 .5, L0 .5.
  - The real i (L3) lies outside this support, so L0 absorbs unexplained actions.
- **Physical priors.** Exactly 50/50 at every nesting level. `SolverBank.set_physical_prior` replaces the sampled 2,000-state fraction.
- **Planning.**
  - Fixed 5-step receding lookahead, gamma .95. The planners are not told the episode length.
  - 20 steps per episode. Reported cumulative rewards are undiscounted.
- **Planner profile: validated.** Empirical Bellman backups, exact final step, exact history rewards, horizon-bounded UCB with c = 1.

## Exact references

- **`FinitePolicyReference`** (`src/solvers/exact/finite_policy_reference.py`) is an exhaustive finite-horizon best response for any agent, level and mixture of opponent levels. It is exact conditional on the bank's policy providers.
  - It matches `FinitePolicyL2Reference` on the existing L2 cases.
  - It matches alpha-vector value iteration against uniform L0 at H = 1–4, for both agents and three beliefs.
- **Exact I-POMDP agent** (`benchmarks/exact_ipomdp_agent.py`). Every intentional agent, real or modeled, is exactly solved recursively down to uniform L0. It uses the same frames, priors, physics, fixed receding lookahead and filter as I-POMCP.
- **Filter check.** For both agents at all 8 supported (action, observation) pairs, the filter's level posteriors equal an independent one-step Bayes enumeration to 1e-12.

## Planner validation against the exact reference (pilot beliefs, H = lookahead)

| Profile | Agent | H | Optimal first actions | Max loss | Max Q error |
|---|---|---|---|---|---|
| default (sampled backups, normalized UCB) | i L3 | 3 | 3/5 | 8.26 | 48.2 |
| default | i L3 | 4 | 3/5 | 7.49 | 65.4 |
| default | j L2 | 3 | 4/5 | 8.26 | 47.2 |
| default | j L2 | 4 | 4/5 | 7.49 | 61.7 |
| validated | i L3 | 3 / 4 / 5 | 5/5 each | 0 | 0.02 / 0.49 / 0.81 |
| validated | j L2 | 3 / 4 / 5 | 5/5 each | 0 | 0.03 / 0.47 / 1.02 |

- **The default profile's failures.**
  - It opens a door after one growl (P(tiger) = .85). The exact answer is to listen: Q(L) = -0.09 versus Q(open) = -8.35 at H3.
  - In the pilot, agent i opened onto the tiger twice.
  - After an early opening, both agents' level posteriors collapsed to L0, because no intentional hypothesis predicts an opening that early.
- **The validated profile** was used for everything below.

## Problems found and how each is handled

1. **Exponential nested beliefs.** Exact interactive beliefs grow about x1.5 per step at L2.
   - j's exact support is 6 / 144 / 624 / 2,428 atoms at steps 1 / 8 / 12 / 16.
   - The growth is genuine: nested L1 beliefs depend on the order of private observations. Merging at a 1e-3 tolerance barely helps; only a 1e-2 grid caps it (about 180 atoms).
   - Consequence: exact 20-step L3 play is infeasible, for I-POMCP too, because it shares the exact filter. Every unbounded 20-step trial exceeded 3–4 GiB.
   - **Handled** by an opt-in bounded filter (`max_support` = K).
2. **Particle deprivation under plain resampling.**
   - At K = 32, resampling dropped every L0 atom. The next unexpected door opening then had zero evidence (`UnsupportedObservation`), which ended the trial.
   - **Handled** by stratified resampling within (opponent level, physical state) strata.
   - Level and physical marginals stay exact, and L0 is never dropped. With S level/state strata and K > S, the per-atom mass error of one update is below 1/(K − S); this does not bound error accumulated over steps or nesting levels. Resampling is independent of atom insertion order. Covered by tests. (An earlier claim of 2/K was wrong when S > K/2.)
3. **No false creaks at .90.** A single creak rules out every hypothesis that predicted listening, so level posteriors jump. This is kept by design and reported.
4. **Level is confounded with computation.** Modeled opponents are finite MCTS, and the posterior measures how well those finite models predict. The ladder quantifies this against exact models.
5. **Misspecification.** j's support excludes the real L3 i.
6. **Level information is sparse.** Level posteriors move only when the opponent opens a door while the observer listens.
7. **The exact L3 decision is itself exponential.** It takes 12 / 19 / 61 / 160 s at steps 1–4.
   - So exact audits are capped (next section), and coverage is reported rather than extrapolated.

## Convergence ladder

- **Rungs** (sims i, sims j, modeled sims, K):

  | Rung | Sims i | Sims j | Modeled sims | K |
  |---|---|---|---|---|
  | r1 | 2.5k | 2k | 10 | 32 |
  | r2 | 5k | 4k | 25 | 64 |
  | r3 | 10k | 7.5k | 50 | 128 |
  | r4 | 20k | 15k | 100 | 256 |

- **Trials.** Same trial ids across rungs, hence the same environment random streams.
- **Exact shadows.** Each real agent carries an unbounded exact I-POMDP shadow that conditions on the same own actions and observations. It supports two audits:
  - **Decision audit.** Exact regret, split into search regret (the agent's own approximate problem) and total regret (the exact I-POMDP). It stops once one audited decision exceeds 120 s.
  - **Belief audit.** Nested-belief total variation at every nesting depth, physical-belief error, and modeled-versus-exact opponent-policy total variation. It stops at 400 exact atoms, a 1.2 GiB process RSS, or 120 s per belief step.
- **Outcome convergence.** Cumulative rewards and mean hierarchies are compared with the top rung, paired by trial.
  - This is convergence evidence, not exact error.
  - It is the only evidence available after the exact audits stop.

### Probe results (2 trials per rung)

- **Coverage.** j's decision audit covered 27–40 of 40 decisions. i's covered 3 per trial, extended to about 7 steps by the belief audit.
- **At r1 (10 modeled sims),** i's level posterior diverged from exact by TV 0.47 at step 2 and 0.81–0.88 at steps 4–7.
  - Modeled-policy TV was 0.89 at step 1.
  - No decision error occurred on audited steps.
- **At r3,** j's exact regret was mean 0.057 (max 1.45; positive in 7.7% of decisions), while its search regret was 0.004. Most of j's loss comes from opponent modeling and belief bounds, not from search.

### Cost

- An L3 decision is dominated by the cascade of modeled-opponent solves for nested models reached in the tree. It scales with the modeled budget and support, much less with the root budget.
  - Examples at 24 atoms: 5k sims 143 s versus 20k sims 383 s (both modeled 100). Modeled 25: 115 s. K = 64: 164 s.
- **Probe trial times:** r1 7–11 min, r2 9–13 min, r3 about 20 min, r4 over 70 min.
- **Scheduling:** r1–r3 run 100 trials each. r4 runs as many trials as complete, taken in trial order and reported with their n.

## Results

TBD: filled from `results/tiger/ladder_20260930/` by `benchmarks/l3_l2_ladder_analysis.py` and `benchmarks/l3_l2_figures.py`.
