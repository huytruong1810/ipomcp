# Model contract and theoretical audit

## Information and recursive conditioning

This implementation supports two agents and finite physical transition laws.
An intentional model is an immutable frame and a joint subjective distribution
over physical states and complete lower-level opponent models. L0 is uniformly
random over the domain's fixed action alphabet, not an optimizing POMDP solver.
In Wumpus this includes ineffective shooting attempts. An intentional planner's
legal-action mask must be identical across its subjective support; otherwise it
would condition its action on hidden information.

For own action a, private observation o and public termination event d, the update is

$$
b_i'(s',m_j')\propto\sum_{s,m_j,a_j,o_j} b_i(s,m_j)\pi_j(a_j|m_j)
T(s'|s,a,a_j)O_i(o|s',a,a_j)O_j(o_j|s',a,a_j)
\mathbf1\{D(s')=d\}\mathbf1\{m_j'=U_j(m_j,a_j,o_j,d)\}.
$$

The sensors are conditionally independent given the joint action and successor
state in the supported domains. The inner update U uses only that agent's own
action, observation and the public termination event. It marginalizes hidden
actions under its own subjective models. The outer simulated action is not an
input to that subjective update. Recursion terminates at L0. Continuing episodes
condition on survival before the next policy query; terminal-containing policy
inputs are rejected. Rewards are logged, not supplied as observations.

The finite filter enumerates each probability factor once and normalizes complete
joint atoms, preserving correlations between physical state, type and private
history. Both online planners use it. Sampled search transitions sample private
observations and do not multiply their probability a second time. Search-node
reservoirs are diagnostics, never the authoritative posterior.

## Initial measure and model support

Initial physical states are sampled once per bank and shared as an unconditional
empirical prior by nested models. This avoids inconsistent independently sampled
supports without revealing the actual state. Separate real agents have separate
banks. Level-prior weights are exact, independent of search-node capacity.
The recursive filter is exact relative to this finite measure and its supplied
policy kernels, up to floating-point arithmetic; it is not an exact solution of
the original full-prior planning problem.

Zero finite-support evidence raises UnsupportedObservation. It can reflect a
misspecified subjective model or a sampled prior missing supported mass. It is
not proof that the event is impossible in the physical domain. No type floors,
old-belief retention, reset-to-prior repair or hidden action noise is implemented.
A branch budget raises InferenceBudgetExceeded instead of truncating the posterior.

The previous RTS comparison used a point-prior policy model that did not match
the executing opponent. It failed with unsupported evidence at depth three.
The controlled comparison now explicitly matches the modeled MCTS family, budget,
initial empirical prior and search seed, with isolated subsequent private beliefs.
See MODEL_SPECIFICATION.md for the common-knowledge assumption and its limits.
Misspecified conditions elsewhere remain subject to strict support failures.

## Search policy and reproducibility

Policies are uniform over maximal estimated Q values. MCTS uses the configured
real simulation budget, but modeled policies use 25 simulations by default;
these are different computational approximations even with the same decision rule.
RTS uses sampled lookahead, resampling and a top-k observation limit. Its omitted
continuation branches are not renormalized, but their error is not yet quantified.
Neither planner is an exact optimal-control oracle.

A fixed bank seed and complete unrounded belief encoding determine private search
randomness. Search restores the caller's Python random state. Immutable policy
results are cached; eviction reproduces the same query rather than changing its
policy. Physics and solver configuration must remain fixed for a bank's lifetime.
The current registration API still needs stronger rejection of conflicting reuse.

At the MCTS root, immediate reward is integrated exactly over the finite joint
belief, opponent actions and transitions. The backup is
E[R|b,a] + gamma times the sampled continuation. This is a control variate: root
UCB selects an action without inspecting the newly sampled hidden atom, so replacing
the sampled immediate reward preserves the expected backup target. It does not
remove finite-tree bias or make the continuation exact. Independent one-step
reference tests verify Q values across several budgets and beliefs.

## Evidence and boundaries

Current test results are recorded in the review verification logs; no theory defect is accepted as an expected failure.
An independent scalar L2 Tiger enumeration covers actions, observations, private
priors, sensor accuracies and reset/persistent dynamics. Other tests cover deeper
private-history advancement, diagnostic type elimination, joint correlations,
public survival, cache eviction, serialization and online integration.

These bounded checks do not prove general convergence, all-domain correctness or
uniform approximation quality. Finite support can grow rapidly with nesting and
history. Branch limits and bounded cache entry counts are not whole-process memory
limits. The audit and ordinary suite runners enforce wall/RSS limits through process
supervision; RSS is sampled, so transient allocation spikes can be missed. Tiger reward measurements do not certify
UAV or Wumpus research conclusions. See BENCHMARK.md and BACKLOG.md for measured
evidence and outstanding gates. The foundational interactive-filter literature
motivates subjective recursive propagation; its theorems are not inherited merely
by implementing similarly named classes.

## Reference horizon and rollout contracts

The L2 reference performs observation-conditioned Bellman recursion over joint
physical state and opponent private belief. Both exact agents use a common
decreasing finite horizon. Production nested MCTS replans at each modeled private
belief using its configured fixed depth and budget; comparing these policies
without reconciling that distinction is a model mismatch.

Tree and rollout generative transitions both advance intentional opponent private
beliefs. A final reward-only transition can omit the update because there is no
subsequent action. Tiger's rollout action uses physical-only memory under a stated
uniform-L0 reference model, including noisy creaks and persistent dynamics. That
is a heuristic policy memory, not authoritative Bayesian interactive inference.
It sees only its own action and observation and never prunes tree actions.

The matched L1 oracle experiment uses the exact two-state initial measure and the
same random L0, horizon, discount and sensors for all planners. RTS retains all
observation tokens there. Search budgets are accuracy/cost variables, not a proof
of optimality. Production top-k RTS and empirical initial priors remain separate
approximations whose errors require separate studies.

The POMCP baseline uses history-based rollout policies and a generative model in
both search and rollout phases; correct sampling does not remove finite-budget
error. See [Silver and Veness (2010)](https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf).
Recursive interactive-state transitions include the opponent's private update;
see [Doshi and Gmytrasiewicz (2009)](https://arxiv.org/pdf/1401.3455).
These references motivate the contracts; their theorems do not automatically
certify this finite-filter, normalized-UCB, bounded-computation implementation.


## Optional horizon-bound exploration

`HorizonBoundUCB` selects using Q(h,a) + c W_H sqrt(log N(h)/N(h,a)), where
W_H = (max(0,r_max)-min(0,r_min)) sum_{t=0}^{H-1} gamma^t. H is the remaining
search horizon, not the full root horizon at every node. The zero extension
accounts for early termination. Bounds must cover the entire reachable physical
model and every joint action; a particle-specific interval would leak information
or understate possible returns. The matched Tiger runner enumerates that support.

This interval bounds discounted reward sums. It does not turn UCT estimates,
whose continuation policy changes during search, into independently sampled
confidence intervals. The coefficient still requires calibration; neither
finite-budget optimality nor a new convergence theorem is claimed. Action sets,
generative transitions, private-model propagation, and arithmetic mean backups
are identical across these exploration strategies. The production default remains
empirical-range UCB. See docs/BENCHMARK.md for failed as well as improved cases.
