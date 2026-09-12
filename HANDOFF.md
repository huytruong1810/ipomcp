# I-POMCP Multi-Level Reasoning: Project Status & Agent Handoff

**Handoff Date**: September 12, 2026  
**Repository Remote**: [https://github.com/huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp)  
**Branch**: `main` (Ahead of origin/main; working tree clean)  
**Test Suite Status**: **50 / 50 Passing** (`uv run python -m pytest tests/`, 104.83s execution)  
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
3. **Stable Particle Population in Batch Runner**:
   - In [`GenericBatchRunner`](src/utils/generic_batch_runner.py), thresholds use initial particle counts `initial_n_i` and `initial_n_j`, preventing ratcheting down to 10 particles.
   - `_get_opponent_level_distribution` guarantees complete column representation ($0 \dots \text{max\_lvl}-1$).
4. **Comprehensive Test Suite**:
   - Added [`tests/test_reinvigoration.py`](tests/test_reinvigoration.py) covering consistent state sampling, opponent door opening preservation, agent door opening preservation, normal step extinction prevention, and multi-step end-to-end execution.
   - All 50 tests passing.

---

## 2. Benchmark Status & Execution

### Active Run Archival
- The contaminated run directory containing Level-0 collapsed trials was archived:
  `results/deep_prior/corrupted_collapse_deep_prior_benchmark_20260909_163530_N200_T20/`

### Launching Clean Master Benchmark
To execute the clean benchmark suite across all 3 suites (Deep Prior, Oracle RTS, Payoff Matrix):
```bash
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all
```
Or to run only the Deep Prior benchmark first:
```bash
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite prior
```

---

## 3. Testing Contract
Run the test suite at any time:
```bash
uv run python -m pytest tests/
```
All 50 tests must pass.


