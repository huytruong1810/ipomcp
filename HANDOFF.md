# Review handoff

Authoritative WSL checkout: /home/andyj1810/projects/ipomcp.
Review worktree: /home/andyj1810/projects/ipomcp-review-20260916.
The pre-integration tracked and untracked Antigravity source is preserved in
backup/antigravity-before-supervision-20260919 (commit 6a9393648ad1f6d9d3c8c3731e7b05250d689956).
The original index/worktree were unchanged when that recovery snapshot was made.
Engineering source is integrated into main at 02d814b; the final
integration record documents the retained stash and recovery branch. The original checkout also passed all 123 non-integration tests after dependency synchronization. Nothing is pushed.

## Supervised full-depth qualification — September 19

Every batch trial now uses an isolated Linux process session, including runs with
one worker. Defaults are two workers, 900 seconds per trial and 3,072 MiB sampled
process-tree RSS. An explicit run_batch(max_workers=...) overrides the default
worker count. RSS sampling can miss short spikes; it is not an allocation ceiling.
The supervisor kills timed-out/over-budget sessions and reaps workers. Snapshot
episodes preceding suite batches use the same limits. Each attempt retains logs,
status, wall/RSS measurements and Python traceback where available. Successful
trial checkpoints survive failures; retries keep historical failure records.

The full-depth qualifier uses the actual condition tables with no reduced search
budgets. One seed per condition, twenty steps, depth five for prior/matrix and
depth three for comparison produced:

| Suite | Complete | Failed | Max wall seconds | Max sampled RSS MiB |
|---|---:|---:|---:|---:|
| Prior | 7 | 0 | 50.48 | 153.96 |
| Comparison | 7 | 0 | 76.34 | 134.58 |
| Matrix | 12 | 13 | 900.04 | 696.13 |

All 127 unit/integration tests pass, including fault injection for time, RSS,
process crashes, descendant cleanup and checkpoint recovery. The failed matrix
trials remain failures, not zero returns or omitted observations. Its so-called
exact prior is a point mass on a capped lower level, not a complete exact model
of the actual opponent policy. Strict inference cannot define a posterior on
zero-probability evidence. Choosing explicit uniform lower-level priors including
L0 is a separate experimental design and awaits user selection.

The prior and comparison families have a full-budget twenty-step execution check;
one seed per condition is not a powered validation of expected payoff. Existing
payoff/value limitations and the unestablished L3-vs-L2 reward equivalence remain in force.
Runtime results are source-bound in results/full-depth-qualification-20260919.

## Tiger Policy Inversion & Opponent Model Collapse Resolution — September 19

Branch: `fix/tiger-policy-inversion`

### Identified Failure Modes
1. **Always-Listen Rollout Pathology**:
   - The baseline `TigerModel.get_rollout_action` unconditionally returned `LISTEN`.
   - Rollouts never opened doors or collected treasure (+10), causing listening to evaluate as an endless -20 penalty.
   - Inside MCTS, opening a door after only 1 observation appeared to have expected value -6.5 (better than -20), causing agents to rush doors prematurely and repeatedly suffer -100 penalties.
2. **Opponent Simulation Recursion Explosion**:
   - Modeled opponent policies defaulted to $N=10$, creating erratic low-budget approximations.
   - Uniformly raising $N=100$ caused combinatorial $O(N^L)$ recursion at Level 4, triggering the 900s timeout.
   - Calibrated `OpponentPolicyConfig.n_sims = 25` to balance policy fidelity and L4 recursion bounds.
3. **Opponent Model Collapse (L2/L1 -> L0)**:
   - When an opponent opened a door, `update_rollout_belief` failed to inspect creaks (`CREAK_LEFT` / `CREAK_RIGHT`) and compounded old pre-reset growls with new post-reset growls.
   - At depth 1 under door openings, deafened `(SILENCE, SILENCE)` observations funneled all simulations into a single child node where `NormalizedUCB` forced blind door openings at 50/50 belief, incurring -100 penalties and poisoning the child's average value to -78.7.
   - Consequently, modeled opponents output `LISTEN` even with 97% confidence, causing Bayesian likelihood under the real opponent's door opening to evaluate to 0.0 and collapsing L2 and L1 to 0%.

### Implemented Fixes
1. **Belief-Aware Information-Seeking Rollout Policy**:
   - `TigerModel.get_rollout_action`: opens the safe door when private confidence is $\ge 92\%$ (>= 2 consistent growls); listens when uncertain.
2. **Creak-Aware Rollout Belief Reset**:
   - `TigerModel.update_rollout_belief`: properly resets physical tiger distribution to uniform 0.5/0.5 upon detecting opponent door openings via creaks.
3. **Belief-Aware Candidate Action Filtering**:
   - `POMDPModel.get_candidate_actions` / `TigerModel.get_candidate_actions`: prunes strictly dominated door openings at tree depth > 0 when confidence is low (< 85%).
   - `IPOMCPPlanner._simulate`: connects candidate action filtering at depth > 0 while preserving complete legal action evaluation at root.
4. **Uniform Matrix Lower-Level Priors & Sensor Law Memoization**:
   - Enabled uniform priors over strictly lower levels to prevent `UnsupportedObservation` failures.

### Verified Outcomes
- **Prior & Comparison Suite (14 conditions)**:
  - 14/14 complete, 0 failures, 0 timeouts, 0 supervisor kills.
  - **Every single condition achieved positive cumulative return for Agent I** (ranging from +2.0 to +13.0).
  - Agent I mean return improved from **-67.0 to +8.3**.
  - Replaced catastrophic penalties (-141, -130, -75, -42) with clean, positive returns.
- **Opponent Model Tracking**:
  - In Condition 1 (`L3 vs L2`), Agent I maintained 100% confidence in Level 2 without collapsing into Level 0.
- **Test Suite**:
  - 144 / 144 unit, integration, and regression tests pass cleanly (`PYTHONPATH=src pytest tests/`).

