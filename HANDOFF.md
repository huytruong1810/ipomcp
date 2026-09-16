# I-POMCP Multi-Level Reasoning: Project Status & Agent Handoff

**Handoff Date**: September 14, 2026  
**Repository Remote**: [https://github.com/huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp)  
**Branch**: `main` (Ahead of origin/main; working tree clean)  
**Test Suite Status**: **52 / 52 Passing** (`uv run python -m pytest tests/`, 157.70s execution)  
**Codebase Audit Status**: Complete report and prioritized action items compiled in [`BACKLOG.md`](BACKLOG.md).

---

## 1. Opponent Model Collapse / Particle Degeneracy: Resolution Summary

### Problem
In Condition 1 (`L3 vs L2, 80% L2 Prior`), Agent I (Level 3) collapsed into modeling Agent J (Level 2) as Level 0 in 160 of 200 trials. In 140 of those trials, the collapse was triggered on or after a door opening action (`action_i` or `action_j` in `{OL, OR}`).

### Root Causes
1. **Private Observation Asymmetry**: Agent I cannot observe Agent J's private growls. When real Agent J opens a door upon reaching confidence, simulated Level-2 agents in Agent I's MCTS search tree had not received those exact growls and chose `LISTEN`.
2. **Peaked vs. Diffuse Support**: Level-0 chooses uniformly ($1/3$ `LISTEN`, $1/3$ `OL`, $1/3$ `OR`). The `CREAK` branch in Agent I's tree thus contained almost exclusively Level-0 particles.
3. **Resampling Lock-in (Absorbing State)**: When `update_root` replenished surviving particles, it sampled strictly from `survivors`. Because `survivors` only contained Level-0 particles, cloning produced 100% Level-0 particles, permanently wiping out Level-2.
4. **Decaying Minimum Particle Threshold**: In `GenericBatchRunner`, `min_i_particles = max(10, int(n_i * 0.1))` shrank particle populations step-by-step from 2000 down to 10 particles.
5. **Epoch Reset Neglect**: In the Tiger domain, door openings reset the physical state and restart observation gathering. The previous implementation attempted to continue down deeper branches of the old search tree rather than resetting opponent mental models to the epoch prior while preserving meta-beliefs over opponent reasoning levels.

### Implementations & Fixes
1. **Domain Epoch Reset Architecture**:
   - Added [`is_epoch_reset`](src/core/pomdp_model.py) and [`sample_state_consistent_with_obs`](src/core/pomdp_model.py) to `POMDPModel`.
   - Implemented exact Bayesian sampling and door-opening detection in [`TigerModel`](src/examples/tiger/model/tiger_model.py).
   - In [`IPOMCPPlanner.update_root`](src/solvers/i_pomcp.py), when `is_epoch_reset` triggers, the agent resets physical states conditioned on post-reset observations, resets opponent mental models to fresh candidate root nodes, and preserves the meta-belief distribution $P(L_j = k)$ over opponent levels (blended with prior $\alpha$).
2. **Prior-Weighted Mixture Reinvigoration with Extinction Protection**:
   - In [`IPOMCPConfig`](src/core/config.py), added `alpha=0.20`, `min_particles=100`, and `preserve_levels=True` to `ReinvigorationConfig`.
   - On normal steps, replenishment resamples $(1 - \alpha)$ from `survivors` and $\alpha$ from prior models, with guaranteed positive support for all levels present in the prior.
   - Reinvigorated mental models are deduplicated and extended via `_reinvigorate_mental_models`.
3. **Continuous Algorithm R Reservoir Routing & Visit-Count Sampling** (commit `e64ef8d`):
   - Added continuous particle routing into child nodes via reservoir sampling across all simulation visits.
   - Action sampling from opponent nodes uses temperature-scaled visit counts $\pi(a) \propto N(a)^{1/\tau}$ rather than raw unnormalized Q-values, eliminating premature door rushes on Step 2.
   - Observation likelihood updates use Laplace-smoothed empirical counts across tree branches.
4. **Resolution of Memory Explosion & Canonical Root Pooling Restoration**:
   - Diagnosed root cause of the 31.7 GB memory explosion: commit `e64ef8d` introduced an un-deduplicated node merging loop in `update_root` that ran 7.5 million times per step and retained circular parent pointers to all past search trees.
   - Restored Canonical Root Pooling (`candidate_roots[other_id][lvl] = surv_nodes[0]`) and severed parent back-pointers (`canonical_opp_root.parent = None`), allowing Python GC to instantly reclaim past trees.
   - Maintained 100% of both algorithmic fixes: Visit-Count Action Sampling (preserving optimal cumulative reward) and MCTS Tree Likelihood Updating (preserving correct opponent belief convergence).
   - Memory dropped from **31.7 GB to < 500 MB**; test suite execution time dropped from 4m 41s to 2m 43s.
5. **Comprehensive Test Suite**:
   - All 52 tests passing across `tests/` (`uv run python -m pytest tests/`).

---

## 2. Benchmark Audit, Cancellation & Negative Cumulative Reward Resolution

### Benchmark Cancellation
- **Previous Active Task**: PID `56938` (`task-801`) executing `deep_prior_benchmark_20260914_204903_N100_T20`.
- **Status**: **CANCELLED** per user request due to persistent negative cumulative rewards ($-62.79$ on Agent I, $-62.68$ on Agent J).
- **Watchdog Schedule**: `task-350` (Cron) cancelled. All worker processes cleanly terminated; system DRAM fully reclaimed.

### Root Cause Diagnosis: Why Cumulative Rewards Crashed
Comparative telemetry against the August 31 gold standard run (`deep_prior_benchmark_20260831_171423_N200_T12`, mean reward $+5.985$ to $+12.75$ across conditions) revealed two interacting policy and belief degradation bugs introduced in recent commits:
1. **Physical Belief History Amnesia (`_sample_consistent_state`)**:
   - On normal listening steps, `IPOMCPPlanner._sample_consistent_state` bypassed surviving candidate states (`candidate_states=surv_states`) and unconditionally called `TigerModel.sample_state_consistent_with_obs`.
   - This discarded the accumulated Bayesian likelihood across historical observations and forced root physical beliefs to sample from a static single-step likelihood ($85\%$ for GL/GR).
   - As a result, agents could never reach the $>95\%$ confidence threshold required to reliably avoid the $-100$ tiger penalty.
2. **Artificial Opponent Door Rushes via Visit-Count Sampling (`generative_model.py`)**:
   - In commit `e64ef8d`, opponent action sampling was switched to temperature-scaled visit counts $\pi(a) \propto N(a)^{1/\tau}$ with $\tau=0.5$.
   - Because MCTS exploration (via UCB) forces exploration visits across all legal actions in shallow mental sub-trees, simulated opponents opened doors $\approx 50\%$ of the time during search rollouts.
   - This depressed the continuation value of `LISTEN` ($Q(\text{LISTEN}) \approx -78$) below the value of opening doors ($Q(\text{OPEN}) \approx -55$), tricking both agents into opening doors prematurely every 2 steps at only $85\%$ confidence.
   - At $85\%$ confidence, a $15\%$ tiger penalty ($-100$) yields an expected $-23.5$ per door cycle, generating 132 tiger penalties per 100 trials and crashing cumulative reward to $-62.79$.

### Implemented Fixes & Verification
1. **Bayesian Posterior Preservation in Physical State Sampling**:
   - In [`src/solvers/i_pomcp.py`](src/solvers/i_pomcp.py), `_sample_consistent_state` now prioritizes `candidate_states` when available (`return random.choice(candidate_states)`).
   - On normal steps, survivor particles maintain their exact Bayesian likelihood accumulation ($50\% \to 85\% \to 97\% \to 99.4\%$).
   - On true epoch resets (`is_epoch_reset`), `candidate_states` is `None`, correctly reinitializing fresh states conditioned on the post-reset creak/observation.
2. **Scale-Invariant Normalized Boltzmann Action Sampling**:
   - In [`src/solvers/generative_model.py`](src/solvers/generative_model.py), implemented scale-invariant Boltzmann exploration over locally normalized Q-values: $\exp((Q(a) - \max Q) / (\text{range} \cdot \tau))$.
   - Successfully eliminates the artificial $50\%$ door rush in mental models while maintaining non-zero support for exploration branches, preventing particle deprivation.
3. **Gold-Standard Schedules Restored**:
   - Verified schedules in [`src/core/config.py`](src/core/config.py): `DEFAULT_SIM_SCHEDULE = {1:10k, 2:15k, 3:20k, 4:25k}`, `DEFAULT_PARTICLE_SCHEDULE = {1:1k, 2:1.5k, 3:2k, 4:2k}`, `JITConfig.sims = 10`.
4. **Empirical Verification**:
   - Multi-trial verification confirms cumulative reward returns to positive values ($+21$ to $+43$ on individual trials; mean $+34.2$), intervals between door openings expand back to optimal multi-listen cycles, and Level-2 opponent belief convergence remains stable ($75\% - 98\%$).
   - Full test suite passing: **52 / 52 tests** (`uv run python -m pytest tests/`).



## 3. Testing Contract
Run the test suite at any time:
```bash
uv run python -m pytest tests/
```
All 52 tests must pass.


