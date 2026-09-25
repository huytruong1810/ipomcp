# Matched Level-2 Tiger reference contract

## Scope and purpose

The first matched L2 experiment isolates the protagonist's approximation while
holding its intentional opponent policy fixed. The opponent is an exact
finite-horizon L1 agent, replanning at a fixed positive depth d, which models
the protagonist as uniform L0. It is not the production finite-budget modeled
MCTS opponent. Matching a reference requires matching the entire policy law,
not merely assigning the same initial tree depth.

The existing shared-countdown reference remains a distinct mathematical model:
ExactIPOMDPSolver(opponent_horizon=None) queries j with i's remaining horizon.
An explicit positive opponent_horizon instead queries j at d on every step.
Changing i's remaining horizon or calling solve() never changes this d.
There is no silent substitution of one interpretation for the other.

## Information, dynamics and objective

Let s be the physical Tiger state and b_j the opponent's private probability
of Tiger Left. The protagonist's belief B is JOINT over (s,b_j). Initially the
runner uses B(s,b_j)=b_i(s) times a point mass at the declared opponent belief.
Point support initially does not imply independence after observations.

Both agents use the same TigerModel transition, reward and private observation
laws, gamma=.95 unless explicitly changed. The action alphabet is L, OL, OR.
Opening resets the physical state according to the model; it does not terminate
the Tiger episode. Private observation draws are conditionally independent given
the joint action and post-transition state. Neither the other agent's action,
private observation nor true hidden state is exposed to a decision policy.

The opponent law is pi_j(a_j | b_j;d), obtained from ExactPOMDPSolver at fixed d
under uniform L0 for i. Equal action values within absolute tolerance 1e-10
receive uniform probability. The horizon and tie rule are model parameters,
not adaptively selected per outer state or benchmark result.

The opponent updates its own belief by

    tau_j(b_j,a_j,z_j)(s')
      proportional to sum_s sum_a_i b_j(s) (1/3)
                      T(s'|s,a_i,a_j) O_j(z_j|s',a_i,a_j).

It uses its subjective uniform-L0 law, not i's actual action. The protagonist's
private action/observation branch has unnormalized next joint mass

    W_a,z_i(s',b'_j)
      = sum_s,b_j,a_j,z_j B(s,b_j) pi_j(a_j|b_j;d)
          T(s'|s,a,a_j) O_i(z_i|s',a,a_j) O_j(z_j|s',a,a_j)
          1[b'_j = tau_j(b_j,a_j,z_j)].

Normalize W only after marginalizing all hidden events. Zero-evidence branches
are not legal posteriors; no nearest-belief or uniform fallback is used.
Both the independent scalar-belief reference recursion and the recursive finite
filter implement this information contract.

With h protagonist decisions remaining,

    Q_h(B,a) = E_B,pi,T[R_i(s,a,a_j,s')]
               + gamma sum_z_i P(z_i|B,a) max_a' Q_(h-1)(B_a,z_i,a'),
    Q_0 = 0.

There is one maximization per observable posterior, never per hidden state,
opponent action or opponent observation. Discounting and horizon apply to i;
d remains fixed for the opponent. This is a best response to the specified
intentional policy model, not a game equilibrium or mutual-rationality proof.

## Implementation boundaries

- ExactIPOMDPSolver enumerates the joint belief recursion without belief
  rounding or probability cutoffs. Its exact L1 policy uses alpha-vector
  finite-model backups with floating-point numerical tolerances.
- FixedTigerL1Policy is registered for j at level 1 in SolverBank. Its input is
  j's MentalModel, with Tiger physics and an i/L0 opponent checked explicitly.
  It integrates j's own physical marginal; no hidden-state argument is present.
- The runner constructs explicit two-state priors with no empirical bootstrap.
  The real L2 MCTS root uses its requested search budget/horizon and can use
  empirical Bellman backups, exact tail and bounded UCB.
- The protagonist never receives oracle Q values. The exact opponent policy is
  part of the declared environment model and is used by both compared solvers.
  Reference protagonist Q values are used only to score the returned policy.
- SolverBank caches are scoped to fixed policy providers. Do not replace the
  registered opponent or mutate its settings during a run.
- CLI level=2 requires opponent-depth; RTS is outside this first L2 protocol.
  The reference and MCTS opponent share d, discount, subjective L0 and tie rule.
  Manifests/rows record the intentional policy type and its parameters.

## Validation and limits

The analytic uniform two-step case has both agents initially listening. One
85%-accurate growl makes the preferred opening worth -6.5, below listening -1;
therefore the optimal value is -1-.95=-1.95. At b_j=.085, the L1 one-step policy
opens, while its two-step policy listens: tests ensure fixed-depth planning does
not become countdown planning when i reaches its final step. Further tests
check hidden-belief-independent immediate rewards and strict input rejection.

A passing L2 panel under this contract does not qualify finite-budget modeled
MCTS opponents, mixtures over levels, empirical physical priors or deeper
hierarchies. The next production comparison must match those additional policy
parameters explicitly. Exact recursion grows rapidly with horizon; resource
failures must remain reported outcomes, never triggers for a cheaper reference.


## Finite-computation modeled L1 contract

Supplying --opponent-budget selects a second explicit opponent law: the registered
IPOMCPPlanner at j/L1, called through SolverBank.policy with modeled=True.
The depth remains fixed on every call. Its full IPOMCPConfig, exploration
strategy and constant, and bank seed determine its deterministic finite policy.
The uniform-exact-maximum tie rule of production greedy_policy applies, rather
than the exact-L1 reference's 1e-10 tie tolerance. No exact-L1 substitution is
made when the finite policy is expensive or inaccurate.

FinitePolicyL2Reference exhaustively computes the protagonist's best response
to this law. Its state is the complete InteractiveState(s,MentalModel_j), not
(s,scalar_j). It retains the immutable nested belief objects returned by the
bank's subjective update. Rounding or reconstructing a mathematically close
scalar belief could change the deterministic search seed and hence the opponent
action, so such reconstruction is outside this contract.

The joint Bellman equation above is unchanged, but pi_j is now the declared
finite-computation policy. Outer transition/observation enumeration is implemented
separately from the production filter. The opponent's subjective update is shared
because it defines which private model the policy receives. Independent tests
enumerate H2 physical posteriors without that filter and compare joint posterior
moments against scalar subjective Bayes updates. Probability support is retained;
exceeding the explicit branch budget raises InferenceBudgetExceeded.

The reference is exact relative to this specified opponent and finite probability
model, up to floating arithmetic. It does not certify that j's finite policy is
optimal. Its caches are bounded and instance-owned; the bank's physics/config/
seed must remain fixed. The sampled root policy is computed before the exhaustive
reference, so its oracle Q values cannot guide the root search. Reference and
planner may reuse cached opponent-policy evaluations: these are part of their
common modeled dynamics, not information about optimal protagonist actions.
Worker elapsed includes both computations, so it is not planner-only latency.

## Order-independent finite policy identity

FiniteBelief equality ignores insertion order, and search seeds already hash a
canonical encoding. Sampling previously used insertion order, however, allowing
equal models with identical seeds to select different finite-budget actions.
The reproduced case was seed4, b_j=.085, depth3, modeled budget25: OL versus L
under reversed input order.

SolverBank now supplies canonical ordered masses to MCTS root sampling and
physical-marginal accumulation. Ordering uses full immutable atom encodings,
including exact mass hex values and nested beliefs; nothing is rounded or
projected to a scalar. The regression checks direct solves and cache eviction.
This fixes the policy-function contract but may change finite trajectories on
old input orderings. Historical results remain tied to their recorded source;
do not silently relabel an older validation as validation of changed code.

The runner records complete actual and modeled configurations in each row,
the modeled configuration and bank seed range in the manifest, and explicit
policy type. For this benchmark modeled mcts.n_sims and opponent.n_sims are both
set to the requested opponent budget; policy_for(modeled=True) uses the latter.
Production configurations with other inactive fields can hash differently and
must be recorded separately rather than assumed bitwise equivalent.

## Current finite-opponent experiment limits

The initial test contract is Tiger i/L2 against pure j/L1 under uniform i/L0,
fixed point prior on j's private belief, and explicit two-state physical priors.
It does not yet qualify mixed reasoning levels, empirical priors, RTS opponents,
L4 or all production conditions. Canonical private-policy identity and exact
small-case enumeration are prerequisites, not substitutes for those gates.
