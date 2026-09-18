# Corrected inference specification

## Accepted model boundary

The user accepted retaining uniform-random level zero on September 18, 2026.
L0 samples uniformly from actions available on its information; it is a
subintentional model with no optimizing physical belief. An intentional Lk belief
is joint over physical state and complete private models at levels strictly below
k. This project currently supports two agents. Agent identity and reasoning level
do not alone identify a private model.

Keep the principal Tiger experiment's 0.85 growl accuracy, 1.0 creak accuracy,
simultaneous actions, pre-transition opening rewards, post-transition observations,
uniform physical reset on any opening, gamma 0.95, and always-listen rollout.
Persistent Tiger remains an explicitly different condition. A physical reset
never resets a fixed opponent type or erases its private history by itself.

Agents condition on their own actions and sensor observations. The runner logs
rewards but currently does not supply them as observations to the agent. This is
part of the implemented information structure, not a theorem that rewards carry
no information. An experiment that exposes realized rewards to belief updating
would require a corresponding observation kernel and is a different model.

## Recursive conditioning contract

For outer model i, enumerate or sample the opponent action from its subjective
policy. Propagate the physical state and opponent's private observation. Update
that opponent recursively using only its own action and private observation;
unknown actions are marginalized under its own subjective beliefs. Finally
condition the outer joint state/model distribution on i's observation.

The reference kernel in `ipomdp/finite_filter.py` enumerates both actions and
private observations, so each branch carries their probabilities once. A sampled
variant must not multiply again by the probability used to sample a branch.
Current domains assume sensor independence conditional on joint action and
post-transition state. Correlated observations would require a joint sensor law.

`FiniteBelief` normalizes and combines duplicate complete interactive states.
`MentalModel` enforces strictly decreasing nesting, and L0 has no belief payload.
These objects contain no mutable search tree, visit count or action-value estimate.
Caches compare complete immutable values without rounding. The physics and policy
providers must be fixed for a filter instance's lifetime; mutating a provider is
not a supported way to change an experiment.

The transition-support API supplies an actual probability law. Tiger enumerates
its two possible reset states; UAV and Wumpus enumerate their deterministic next
state. An unsupported domain raises explicitly instead of pretending one sampled
outcome is its whole transition law.

Zero evidence raises `UnsupportedObservation`. The message says zero support in
the supplied finite model: a sampled prior may have lost supported hypotheses, or
an intentional agent may hold a misspecified subjective model. Nested zero
evidence on a positive outer event is also reported, never silently dropped or
renormalized away. No type floor, epoch override, old-belief retention, or unrelated
initial-prior replacement appears in this kernel.

## Policy contract and remaining integration

The intentional policy provider returns a normalized distribution using only the
private model it is given. It receives no outer hidden state or action. The final
planner policy will distribute mass over maximizing action values, with uniform
tie handling. Every finite-search approximation must specify its horizon,
discount, simulation budget and randomization. Modeled and executing agents will
consume the same interface; differing solve budgets must be reported explicitly.

The exact-filter tests use a precisely specified one-step greedy Tiger policy or
a controlled fixed policy. They do not validate a finite-search MCTS policy.
Replacing the application's existing mutable-node models and JIT policy remains
necessary. The finite kernel is a correctness reference and migration foundation,
not a hidden alternative execution path selected when the existing solver fails.

## Evidence and limits

`tests/test_finite_filter.py` compares every L2 observation/action combination
against a separate scalar Tiger enumeration over three private prior probabilities,
three creak accuracies, and reset/persistent dynamics. Additional tests check
multi-step subjective marginalization, L3 recursive advancement, state/private-
belief correlation, diagnostic-creak type elimination, mutation, pickle hashes,
cache order/eviction, invalid probability laws, and unsupported private observations.

Numerically equivalent measures are compared within floating-point tolerance in
tests. This does not authorize rounding production belief keys. Finite enumeration
can grow exponentially with horizon and model level; it is not yet qualified for
the long full experiment suite. The two application-level counterexamples remain
strict expected failures until integration genuinely fixes both planners.
