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
The prior RTS/MCTS comparison exposed a depth-three support mismatch. The
controlled comparison now declares matched computation, as specified below;
full-suite qualification still requires the remaining gates.
Finite exact reference tests validate bounded cases, not a general convergence
claim. See IMPLEMENTATION_PLAN.md for remaining qualification work.

## Controlled planner comparison: declared computation

The L2-vs-L1 comparison now separates the protagonist's planner from the modeled
opponent's planner. Both RTS and MCTS protagonists model MCTS L1 with the same
depth, exploration rule and default 50,000 simulations as the executing opponent.
The initial empirical physical prior (2,500 samples) and reproducible search seed
are common knowledge in this controlled design. Each bank constructs its own
objects and subsequently conditions only on its own private observations. No
current opponent belief, hidden action or physical state is read by the other
agent. RTS uses a separate 500-particle lookahead budget by default; this does not
change the initial prior sample count.

This is a declared matched-computation model, not evidence that arbitrary agents
know each other's internal randomness. If a search seed or subjective initial
prior is unknown, a faithful model needs a distribution over that uncertainty.
Matching only planner family and simulation count does not suffice: independent
initial priors/seeds produced unsupported observations in the qualification panel.
Lowering modeled_opponent_sims explicitly reintroduces computational mismatch and
may legitimately produce a recorded failure under strict inference.

The original L3-vs-L2 prior experiment keeps its independent-bank design and
ten-simulation modeled policies. Its previous reward evidence must not be presented
as evidence for this changed controlled comparison. All historical failed panels
remain preserved. These changes introduce no action noise or belief repair.
