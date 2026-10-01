# Phase 2 — `src/ipomdp/` (interactive beliefs and filtering)

Files reviewed in full: `__init__.py`, `frame.py`, `finite_belief.py`, `finite_filter.py` (427 lines).
Probe script: `scratchpad/resample_probe.py`; results are quoted below.

## What the module is

- `FiniteBelief`: an immutable, normalized, exact-float measure over `InteractiveState(state, MentalModel(frame, nested FiniteBelief))`.
- `FiniteInteractiveFilter.update`: the exact I-POMDP belief update of Gmytrasiewicz & Doshi (2005, eq. 3), enumerated:

  b'(s', θ_j') ∝ Σ_{s, θ_j} b(s, θ_j) Σ_{a_j} π_j(a_j | θ_j) T(s' | s, a) O_i(o_i | s', a) Σ_{o_j} O_j(o_j | s', a) [θ_j' = SE_θj(b_j, a_j, o_j)]

  The implementation matches this equation term for term (`finite_filter.py:158-205`). O_j enters once, through the enumeration of o_j, and the nested update recurses with j's own subjective model. This core is correct.

## Findings

### Correctness

- **B1 — Bounded resampling depends on insertion order (confirmed).**
  - `resample` walks `belief.mass` in insertion order (`finite_filter.py:231,257`), but `FiniteBelief` equality ignores order (`finite_belief.py:100,108-109`).
  - The probe built two equal beliefs (`a == b` is True) whose resamples differ (`resample(a,8) == resample(b,8)` is False).
  - Because `update` is `lru_cache`d on equality, the posterior returned for an equal input depends on which ordering was cached first, and so on query order and cache eviction. That contradicts the class docstring ("independent of query order", `:80-82`).
  - The seed hashes masses in insertion order too (`:245-247`).
  - `SolverBank._ordered_mass` already defines a canonical order (`solver_bank.py:112-121`), so the two modules disagree.
  - Affects only `max_support` runs, i.e. the L3-vs-L2 ladder.
  - **Fix:** order rows canonically within each stratum and seed from the canonical encoding.
- **B2 — The documented error bound is false (confirmed).**
  - The docstring claims per-atom mass error below 2/K (`finite_filter.py:226`).
  - The probe found error 0.838 at K = 3 with 26 atoms (ratio to 2/K: 1.26).
  - The correct per-stratum bound from systematic resampling is `total_k / n_k`. Since `n_k ≥ 1 + (K − S) total_k`, this gives a per-atom error of at most `1 / (K − S)`, where S is the number of strata. That is below 2/K only when S ≤ K/2.
  - The handoff, `L3_L2_TIGER_STUDY.md` and the note to Codex repeat the false bound.
  - **Fix:** state and test the correct bound.
- **B3 — Strata are keyed by `repr(state)`** (`:232`). Two distinct states with equal reprs would be merged, and the stratum order is lexicographic in the repr. Use the canonical `stable_value` encoding that `solver_bank` already uses.

### Theory gaps (they bear directly on contribution 1)

- **T1 — There is no interactive particle filter in the codebase.**
  - Contribution 1 is "I-NTMCP + IPF". The belief representation here is exact enumeration, plus my opt-in deterministic resampling.
  - The IPF of Doshi & Gmytrasiewicz (2009) propagates *sampled* (s, θ_j) particles, sampling a_j ~ π_j, s' ~ T and o_j ~ O_j, and weights only by O_i. Its cost is linear in the particle count per level.
  - Here, cost grows with the full branching over a_j × s' × o_j at every nesting level. That is why j's L2 support reaches 2,428 atoms by step 16 and every 20-step L3 trial ran out of memory.
  - `core/distribution.ParticleDistribution` claims to be "for IPF" but is used only by RTS (Phase 1 S3).
- **T2 — Error from bounded support has no propagated bound.**
  - Even with B2 corrected, the bound is per atom and per update.
  - The quantities that matter are bounds on level posteriors, nested beliefs at depth d, and value loss. Those compound across nesting levels (every nested posterior is resampled too) and across time.
  - Contribution 3 promises "quantifiable error bounds". Today only empirical TV against an exact shadow exists, and only while the exact shadow is tractable (about 7–8 steps for L3).
- **T3 — Termination conditioning is asymmetric.**
  - The nested update receives `terminal=ended`, so the opponent conditions on whether the episode ended (`:196`).
  - The top-level update in `i_pomcp.update_root` passes `terminal=False`. The base model (`:143`) defaults to `None`, meaning no conditioning.
  - Public termination is a modelling choice: it should be one declared observation component, not an optional argument.
- **T4 — L0 is uniform over `get_all_actions` while intentional levels use `get_legal_actions`** (`:113` vs `:128`). This is deliberate (documented) but makes L0 differ from the I-POMDP convention, where L0 is a POMDP agent. The README notes it. The project should state which definition the papers use and keep one.

### Structure

- **S1 — Two frame types.** `ipomdp.frame.AgentFrame` (agent, level, physics) and `solvers.solver_types.SolverKey` (agent, level) duplicate identity. `solver_types` re-exports `AgentFrame`. One value object suffices.
- **S2 — Frame equality uses object identity of the physics** (`frame.py:35,39`). That forces `__reduce__` workarounds (`finite_belief.py:111-114`) and a separate `stable_value(vars(physics))` encoding in `solver_bank`. Physics should be a value object with canonical equality.
- **S3 — `action_distribution(opponent, physical_state)` never uses `physical_state`** (`:103`). The parameter invites the very leak the docstring forbids.
- **S4 — `checked_distribution` caches validation with `lru_cache(1024)` on tuple keys** (`:60-73`). Validation becomes cache-sensitive in cost and hides repeated model errors after the first. Validating domain kernels once in domain tests is simpler.
- **S5 — Positional `terminal` argument on a cached function.** `update(model, a, o, terminal=...)` caches `terminal=False` and `terminal=None` as separate keys for the same posterior in non-terminating domains.

## Bugs fixed (per the agreed policy)

**Status: fixed on September 30.**

- **B1 and B3.**
  - The new `ipomdp/canonical.py` owns the canonical encoding of states, atoms and beliefs.
  - `resample` keys strata by the canonical state encoding, sorts atoms canonically within each stratum, breaks remainder ties by stratum key, and seeds from `canonical.belief_digest`.
  - `SolverBank` now delegates to the same module. The encoding is byte-identical to the bank's former private one, so search seeds and sampling order are unchanged. `stable_value`/`digest` moved there from `solvers/policy.py`.
- **B2.** The docstring now states the correct bound, `< 1/(K−S)` per atom per update for K > S, and says what it does not bound. `L3_L2_TIGER_STUDY.md` and `NOTE_TO_CODEX_ANTIGRAVITY.md` are corrected.
- **Regression tests** (`tests/test_l3_l2_experiment.py`):
  - `test_resampling_is_independent_of_atom_insertion_order` fails on the old code and passes now.
  - `test_systematic_resampling_keeps_marginals_and_per_atom_error_below_one_over_k_minus_s` covers more than 500 random cases with K close to S.
- **Side effect:** the stopped L3 ladder can no longer resume, because its source lock no longer matches. Data already collected is unaffected.

### Found while fixing

- **B4 — `FiniteBelief` normalization is not idempotent.**
  - Re-wrapping an already normalized belief divides by its max and by the fsum total again, which can change the last bit of each mass.
  - The new belief can then compare unequal to the original, so the caches miss.
  - Production code always builds beliefs from fresh unnormalized rows, so no current path is affected.
  - Fix in the restructure: normalize once, and treat an exactly normalized input as a fixed point.
