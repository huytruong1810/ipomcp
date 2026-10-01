# Phase 5 — Domains (Tiger, UAV, Wumpus)

Files reviewed in full:
- `examples/tiger/model/tiger_model.py`
- `examples/uav/model/uav_model.py`, `uav_viz.py`
- `examples/wumpus/model/constants.py`, `wumpus_state.py`, `wumpus_model.py`, `wumpus_viz.py` (render only)

Also read: `utils/bootstrapper.py`, because it constructs every domain's prior.

Probe: `scratchpad/wumpus_support_probe.py`.

## Tiger

**Checked against the source (September 30).** Gmytrasiewicz & Doshi (2005), *A Framework for Sequential Planning in Multi-Agent Settings*, JAIR 24, §7.1 and Table 1 ([arXiv 1109.2135](https://arxiv.org/abs/1109.2135)).

| Element | Published spec | This repository | Match |
|---|---|---|---|
| Transition | The tiger resets uniformly if either agent opens a door; otherwise it persists | Same | ✓ |
| Rewards | Each agent's reward depends only on its own action and the pre-transition state (−1 listen, +10 gold, −100 tiger) | Same | ✓ |
| Growl | 0.85 accuracy about the post-transition state, when the observer listens | Same | ✓ |
| Creak when the other agent opens a door | Correct side 0.9, wrong side 0.05, silence 0.05 | Same at creak 0.9 | ✓ |
| **Creak when the other agent listens** | **Silence 0.9, CL 0.05, CR 0.05 (false creaks)** | Silence with probability 1 | **✗ (T1 confirmed)** |
| **Observer opens a door itself** | **Uniform 1/6 over all six observations** | Deterministic (S, S) | **✗ (T2 confirmed)** |
| **Observation space** | **6 tokens, {GL, GR} × {CL, CR, S}** | 9 tokens: the growl can also be S | **✗** |
| Level 0 | An intentional POMDP agent (G&D's L0 has beliefs and optimises) | Uniform random | **✗ (documented deviation, Phase 2 T4)** |

**Consequences.**
- Without false creaks, one heard creak eliminates every hypothesis that predicted listening. That inflates how fast level posteriors move.
- The deterministic (S, S) for the opener, together with the 9-token space, gives the filter an exact signal that the agent itself opened a door. That signal is redundant, because agents know their own actions, but it changes the observation-space size used by RTS and by observation branching.

**Decision (my recommendation, adopted): implement the published Tiger exactly as the default spec.** Uniform-random L0 remains this project's declared convention, since IPF/I-NTMCP work commonly uses it, and it is stated explicitly in the Tiger spec document.

- **T4 — Default `creak_accuracy=1.0`**, while the L3 study used .90 and the README says 100%. Several runners rely on the default. One declared domain spec, with no default accuracies, avoids silent variant mixing.
- **S1 — Dead fallbacks:** `growl_acc.get(agent_id, 0.85)` (`:123,195`) after validating both keys; `joint_action.get(agent_id, LISTEN)` (`:107,182`).
- **S2 — A class-level `lru_cache` on a bound method** (`:166`) keeps every TigerModel instance alive. That's harmless at this scale but surprising.
- **S3 — Two kernels.** `sample_observation` and `get_observation_prob` are written separately (about 90 lines each) and must agree by hand. One finite `observation_distribution` with sampling derived from it removes the risk class entirely. The same applies to UAV.

## UAV

- **T5 — The UAV does not observe its own position.**
  - The initial UAV position is random, and the UAV's only sensor is the target's *row*, noisy at .85 when listening and uniform otherwise.
  - In the I-POMDP/I-DID UAV benchmarks (Doshi, Zeng and colleagues), the UAV knows its own cell and senses the fugitive's relative direction. Here the UAV cannot localise itself, and the column of the target is never observed.
  - In a 3×3 grid that makes capture mostly luck. Results on this domain may not say anything about planning quality.
  - **To be confirmed:** which published spec this domain is meant to reproduce.
- **T6 — Moving gives a uniformly random observation, not "no observation".** This is equivalent in information content but enlarges the observation branching for nothing. A dedicated null token is cleaner.
- **T7 — Capture requires landing on the same cell;** a swap of positions is not a capture. That is a defensible spec choice, but it is undocumented.
- **S4 — Agent roles are hard-coded:** `"i"` is the UAV and `"j"` the target, through `str(agent_id).lower()` checks (`:52,106,109,130,149`).
  - `joint_action.get("i")` returns `None` for a missing agent, and `_move` then silently stays in place.
- **S5 — `UAVState` overrides the `__eq__` and `__hash__` that `@dataclass(frozen=True)` already generates** (`:29-36`).
- **S6 — The 3×3 size and the role constants are module globals** rather than model parameters.

## Wumpus

- **B1 — The sampled prior never contains the true state (confirmed).**
  - The exact prior has 15 wumpus cells × 4 orientations × 15 gold cells × C(15,2) pit pairs = **94,500** states.
  - Runners sample 1,000 (`wumpus_batch_runner.py:41-64`) or 5,000 (`run_wumpus.py`) states from it (`solver_bank.initial_states`).
  - Sensors are deterministic, so the posterior is the prior restricted to observation-consistent states.
  - Probe: 20 episodes with random play at L1 against L0 and 1,000 states. The true state was **in support 0/20 times**, and **1/20** episodes hit `UnsupportedObservation` (at step 13).
  - In the other 19 episodes the belief is consistent but wrong by construction. For example, it places pits where there are none, because it can only represent 1,000 of the 94,500 worlds.
  - Purposeful exploration discriminates faster than random play, so failures should be more frequent under real planners.
  - **Fix direction:** an exact factored prior (pits ⊥ gold ⊥ wumpus pose) with lazy enumeration, or a particle filter with reinvigoration. That is the IPF contribution's job.
- **T8 — Legal actions depend on `has_arrow`.** This is part of the state, but it is a deterministic function of the human's own action history, so the per-belief "constant legal set" check holds.
  - The model contract should express "legal actions are a function of own history" (Phase 1 T1) instead of relying on that coincidence.
  - L0 still uses the full action list (Phase 2 T4).
- **T9 — The wumpus agent gets `Glitter`, `Breeze` and `Bump`.** These are odd percepts for a wumpus, but harmless. Its pursuit information is `Footstep` within Manhattan distance 2. Rewards are ±1000, −10 for shooting and −1 per step.
- **S7 — Agent roles are hard-coded** (`AGENT_HUMAN="i"`), with string comparisons. An `itertools` import sits inside a method (`:201`).
- **S8 — The observation probability is computed by regenerating the observation and comparing sets** (`:401-405`). That is fine for a deterministic sensor, but it again duplicates the kernel (see S3).

## Cross-domain

- **X1 — Every model has two sources of randomness:** an optional `rng` and the global `random` module (Phase 1 T3).
- **X2 — Each domain encodes "who is i" differently.** One `AgentID` role mapping per domain should be passed in.
- **X3 — Prior construction** (`utils/bootstrapper.py:230-238`).
  - The interactive prior is a **product**: the empirical physical sample × level weights × **one point belief per opponent level**. Each opponent level's belief is the registered solver's own initial belief.
  - So "i's model of j/L1" is exactly j/L1's own prior. That is a common-prior assumption, and it is undocumented in the experiment designs.
  - It also means nested models share the bank's single empirical physical sample (`solver_bank.py:32-49`), so all nesting levels share the same 1,000 Wumpus worlds and the same blind spot.
