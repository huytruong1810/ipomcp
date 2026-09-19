# Implemented model specification

The accepted base case is uniform-random L0 over the fixed action alphabet. Higher
levels contain immutable joint physical-state/opponent-model beliefs, with strictly
decreasing levels and exactly one opponent. Both MCTS and RTS now use the same
recursive finite filter; there is no legacy reconstruction path.

Principal Tiger conditions use simultaneous actions, pre-transition opening reward,
post-transition private sensors, growl accuracy 0.85, creak accuracy 1.0, uniform
physical reset on any opening, gamma 0.95 and an always-listen leaf rollout.
Persistent Tiger is a distinct configuration. Reset does not erase opponent type
or history. Agents observe their own actions, sensors and public survival; logged
rewards and the other agent's hidden action are not inference inputs.

Each bank fixes physics, solver settings and a reproducible seed. Nested models
share that bank's unconditional empirical physical prior. Prior level weights are
exact; diagnostic tree capacity does not clip the authoritative belief. Intentional
policies use uniform ties over maximum estimated Q. Modeled MCTS solves default
to ten simulations and are not identical to larger executing solves. RTS remains
a sampled, observation-truncated comparator. Full details and equations are in
THEORY.md; run manifests record effective budgets and source hashes.

Zero evidence and computation limits are explicit failures, never fallback updates.
The point-prior RTS/MCTS comparison has a demonstrated support mismatch at depth
three. Its scientific model must be resolved before long full-suite execution.
Finite exact reference tests validate bounded cases, not a general convergence
claim. See IMPLEMENTATION_PLAN.md for remaining qualification work.
