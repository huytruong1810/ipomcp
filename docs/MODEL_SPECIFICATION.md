# Implemented model specification

The accepted base case is uniform-random L0 over the fixed action alphabet. Higher
levels contain immutable joint physical-state/opponent-model beliefs, with strictly
decreasing levels and exactly one opponent. Both MCTS and RTS now use the same
recursive finite filter; there is no legacy reconstruction path.

Principal Tiger conditions use simultaneous actions, pre-transition opening reward,
post-transition private sensors, growl accuracy 0.85, creak accuracy 1.0, uniform
physical reset on any opening, gamma 0.95 and a history-based physical-memory rollout heuristic.
Persistent Tiger is a distinct configuration. Reset does not erase opponent type
or history. Agents observe their own actions, sensors and public survival; logged
rewards and the other agent's hidden action are not inference inputs.

Each bank fixes physics, solver settings and a reproducible seed. Nested models
share that bank's unconditional empirical physical prior. Prior level weights are
exact; diagnostic tree capacity does not clip the authoritative belief. Intentional
policies use uniform ties over maximum estimated Q. Modeled MCTS solves default
to 25 simulations and are not identical to larger executing solves. RTS remains
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
25-simulation modeled policies. Its previous reward evidence must not be presented
as evidence for this changed controlled comparison. All historical failed panels
remain preserved. These changes introduce no action noise or belief repair.

## Payoff matrix and oracle scope

The matrix uses explicit uniform priors over strictly lower levels, including
random L0. This is an ex ante model choice, never a repair of a failed posterior.
The actual opponent level is not supplied to the subjective prior rule.

The exact L2 reference has a common decreasing finite horizon for both agents.
Production modeled planners use their configured depth at every private solve.
These must not be compared as if they were identical policy models. The matched
oracle experiment therefore makes numerical claims only for L1 against random L0.
Its supplied two-state physical prior is exact, rather than empirically sampled.


### Exploration ablations

The matched L1 oracle runner exposes empirical-range, raw-unit, and remaining-
horizon reward-bound UCB with an explicit coefficient. These change search
allocation, not the Tiger optimization problem or the intentional opponent model.
Production modeled/real planners keep their existing empirical-range default.
Coefficients have different units across strategies and must be reported alongside
simulation counts, remaining horizon, and wall/RSS costs. No current calibration
setting is certified by the oracle panel; consult docs/BENCHMARK.md.


### Exact final-step experiment

`MCTSConfig.exact_final_step=False` remains the default. The matched MCTS runner
exposes `--exact-final-step`; RTS rejects this MCTS-only option. The flag enters
configuration, result rows, and manifests. Enabled runs condition full private-
history finite beliefs and integrate only the last decision, as specified in
THEORY.md. This changes the leaf estimator and computation, not physical dynamics
or the reward function. It is not certified for production or deep hierarchies.

Adding this configuration field changes the deterministic configuration hash,
including the disabled setting. Old-source and new-source runs with the same
integer seed are not bitwise replays. Compare enabled/disabled runs at the same
new source checkpoint and preserve both manifests; do not pool old trajectories
as if they used the current stream. The default algorithmic rules remain sampled.


### Backup estimator experiment

`MCTSConfig.backup="sampled"` remains the production default. The matched MCTS
runner accepts `--backup empirical_bellman` independently of `--exact-final-step`.
The mode is recorded in configuration, result rows, detailed solver statistics
and manifests; RTS rejects this MCTS-only option. The empirical chance law,
frontier initialization and terminal treatment are specified in THEORY.md.

The additional config field changes deterministic search hashes, even for the
sampled setting. Both controls must be rerun at the same source checkpoint; old
trajectories are not a bitwise control. No compatibility hash or source snapshot
is retained. The physical model, observation law, reward law, real filtering and
private opponent evolution are unchanged. Their computation budgets still need
separate deep-hierarchy qualification.


## Explicit L2 reference opponent

The oracle comparison runner's level=2 mode requires a positive opponent-depth
and uses an exact L1 policy at that fixed depth. An opponent-belief argument
specifies the protagonist's point prior on j's initial private P(TL). Private
beliefs evolve under the same subjective uniform-L0 model in both solvers.
This reference model does not replace production's finite-budget modeled MCTS.
See [L2_CONTRACT.md](L2_CONTRACT.md) for the complete comparison contract.
