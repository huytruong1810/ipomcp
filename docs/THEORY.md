# Model contract and theoretical audit

## What is specified, and what is not proved

The physical kernels define simultaneous joint actions, rewards on the transition,
and private observations of the post-transition state. The mathematical target for
two-agent interactive filtering is a distribution over the *joint* physical and
opponent-model state. Write an opponent model as `theta_j=(b_j, frame_j)`.

For a fixed model frame and an explicit policy `pi_j`, the target update is:

$$
b_i'(s',\theta_j') = \frac{1}{Z}
\sum_{s,\theta_j,a_j,o_j}
b_i(s,\theta_j)\pi_j(a_j\mid\theta_j)
T(s'\mid s,a_i,a_j)
O_i(o_i\mid s',a_i,a_j)
O_j(o_j\mid s',a_i,a_j)
\mathbf{1}\{\theta_j'=U_j(\theta_j,a_j,o_j)\}.
$$

This expression assumes conditionally independent sensors given state and joint
action, as implemented in the domains. `U_j` is the opponent's *subjective* update:
it marginalizes unknown actions under that opponent's own models. Passing the
outer simulator's hidden action directly to `U_j` supplies information unavailable
to the opponent. For continuous model spaces the sums become integrals. The
normalizer must be positive; zero likelihood is a deprivation/model-support problem,
not permission to invent a uniform posterior.

The foundational framework treats nested models as part of interactive state and
updates them from private histories. Its convergence statements concern the
specified mathematical operators, not any implementation with similarly named
classes. [Gmytrasiewicz and Doshi, 2005](https://arxiv.org/pdf/1109.2135).

Doshi and Gmytrasiewicz's interactive particle filter propagates nested beliefs;
their reachability-tree method samples observations. This repository instead uses
top-k selection and incomplete nested propagation. It must not claim a faithful
implementation or an exact oracle based on that citation alone.
[Doshi and Gmytrasiewicz, 2009, Figures 4–5 and §11](https://arxiv.org/pdf/1401.3455).

POMCP's finite-horizon planning argument assumes the appropriate belief and
generative model. A fixed observation-safe rollout can aid finite-budget planning,
but does not validate a different interactive transition kernel.
[Silver and Veness, 2010](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf).

## Confirmed counterexamples and consequences

1. **Physical reset does not erase type evidence.** Let an L1 fixture always listen
   and L0 act uniformly. With prior masses 0.8 and 0.2, a perfect right creak has
   likelihood zero under the listener and one third under L0. The posterior mass
   on the listener is zero. Resetting the tiger changes the physical prior, not
   this likelihood ratio. `update_root` preserves approximately the prior instead.
   The strict xfail in `test_theory_gaps.py` and the passing exact enumeration in
   `test_tiger_reference.py` expose the discrepancy independently.
2. **The filter factorizes dependent quantities.** Rebuilding particles samples a
   physical state independently of an opponent level and associates it with one
   canonical node. In general `P(s,theta|h)` is not `P(s|h)P(theta|h)`. Separate
   marginal sampling loses correlations and private-history diversity.
3. **A capped reservoir is not a branch-frequency counter.** Suppose two observation
   branches receive 100 and 10,000 trajectories and both keep capacity 100. Stored
   lengths suggest equal likelihood despite a 100-fold difference in routed counts.
   Per-level likelihoods require counts from the predictive process, or an explicit
   importance-weighted estimator. Arbitrary pseudocounts and a 1% uniform type floor
   do not make these counts a Dirichlet-multinomial model of the actual observations.
4. **Private models do not fully transition.** `tree_step` inserts the outer sampled
   physical state into an opponent child and copies the nested model mapping from
   one old particle. The deeper nodes retain their histories. The second strict
   xfail demonstrates this. Mutating a shared search node also changes every
   particle pointing to that node: shallow mapping immutability is not immutable
   Bayesian model state.
5. **Policy semantics differ across components.** Real agents select maximum-visit
   actions, simulated intentional opponents sample range-normalized Q-softmax,
   and JIT entropy uses unnormalized Q-softmax. At temperature 0.5, the ratio of
   worst to best action weight under a nondegenerate normalized range is `exp(-2)`
   regardless of reward scale. This is an assumed bounded-rational policy, not
   an optimal-action distribution. Shared search updates make it nonstationary
   during simulation. No inherited UCT theorem has been established for this setup.
6. **Deprivation proposals lack history.** Tiger's one-growl 85/15 sampler is exact
   from a uniform prior, including a known physical reset. Two quiet left growls
   instead yield `0.85²/(0.85²+0.15²) ≈ 0.9698` under a fixed listening opponent.
   The generic proposal simply samples the initial prior. Neither can stand in for
   a general history-conditioned filter. RTS can still retain an old belief after
   zero weights; that unresolved behavior must not be interpreted as inference.

## Bounded fixes with direct justification

Search no longer resamples the root into itself. With a capacity-K reservoir,
reinserting its own draws performs neutral drift without conditioning on evidence.
It also routes each child trajectory once, including first expansion. Tests assert
root preservation and equality between first-level routed counts and action counts.

RTS terminal trajectories contribute their immediate reward but zero continuation.
If only half the mass survives, the observation continuation retains that half
mass instead of renormalizing to one. A two-step deterministic test with reward 2
and discount 0.5 has value `2 + 0.5*0.5*2 = 2.5`. This correction does not remove
finite-particle bias, top-k omitted mass, or nested-model errors.

Wumpus stores bump and scream events in the post-state, because location/aliveness
alone cannot distinguish attempted blocked movement from arrival at a wall or a
new kill from an old one. Its deterministic likelihood uses exactly those events.
UAV retains clamped moves at boundaries because action availability must not reveal
unobserved position. These fixes alter their respective domain semantics and need
new domain baselines before reuse of historical UAV/Wumpus performance claims.

## Reference scope and a principled replacement boundary

`tests/reference_tiger.py` enumerates the two physical states, opponent actions,
and observations for fixed state-independent opponent policies. It validates
one-step values, normalization, accumulated evidence, and reset/type separation.
It is deliberately not described as an oracle for L3 versus L2.

A Bayesian replacement should separate immutable model beliefs from mutable search
statistics. Define and test a policy distribution interface; condition each
subjective filter only on that agent's action and observation; preserve complete
joint particles; make type dynamics explicit; and cache only equivalent model
states with compatible remaining horizons and configurations. Validate tiny
enumerated L1/L2 cases before optimizing shared storage or trying L3-scale runs.
Fixed finite capacity, bounded computation, and history merging are approximations
that need error measurements, not unqualified correctness claims.

The existing uniform-random L0 convention and normalized-softmax opponent model
are retained as experimental assumptions. Changing them changes the research
question. An always-listen rollout is similarly a named leaf-value heuristic, not
an algorithmic proof or evidence that reported type probabilities are calibrated.

## Corrected finite-support kernel checkpoint

The accepted random-L0 contract and recursive conditioning specification now live
in [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md). `ipomdp/finite_belief.py`
separates immutable subjective models from search state, and
`ipomdp/finite_filter.py` enumerates the joint Bayesian update without epoch/type
overrides. Independent small L2 references and recursive L3 tests validate this
kernel. It is not yet used by MCTS or RTS; their defects documented above and the
two strict expected failures remain open. Exact finite support does not imply
tractable cost at the requested experiment horizon or high nesting levels.
