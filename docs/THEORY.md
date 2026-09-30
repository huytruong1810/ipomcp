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


## Experimental exact final-step evaluation

`MCTSConfig.exact_final_step` defaults to false. When enabled, every simulated
own action/observation history carries a joint finite posterior computed by the
shared recursive filter. Conditioning includes public survival; it never receives
the sampled physical state or the actual opponent action. The posterior is
separate from the physical rollout heuristic and the node's particle reservoir.

For one remaining decision, let b_h be that posterior, theta the opponent model,
and pi_theta its fixed modeled policy. The boundary condition is

Q_1(b_h,a) = sum_{s,theta} b_h(s,theta) sum_{a_j,s'}
            pi_theta(a_j) T(s'|s,a,a_j) R_i(s,a,a_j,s'),
V_1(b_h) = max_a Q_1(b_h,a).

The maximization is outside every hidden-state/model/action sum. The existing
reward integrator computes these expectations from the supplied model; the
planner does not import or query the Tiger reference solver. This is the ordinary
one-step belief-MDP boundary, not max-value backups at all tree depths. See
[Cassandra, Kaelbling and Littman (1994)](https://cdn.aaai.org/AAAI/1994/AAAI94-157.pdf)
for the belief-state expectation and control formulation.

The tail applies in both tree and rollout paths. Earlier tree nodes retain their
incremental sampled-return means, root reward integration, legal action sets and
opponent private-model transitions. Exact one-step tree nodes store action values
but no fictitious per-action simulation counts. Their visit count records how
often the integrated value is requested. The simulation budget therefore counts
root traversals, not an identical amount of work across configurations.

This is an explicit hybrid of root sampling and finite-model tail integration;
it requires more than a black-box simulator. Exactness is relative to the supplied
finite belief and modeled policies. It does not resolve empirical-prior error,
subjective opponent mismatch, or finite-budget exploration bias at earlier nodes.
Enumeration can be costly in deep hierarchies. Unsupported observations and
resource-limit exceptions remain failures; there is no sampled-posterior fallback.


## Experimental empirical chance-weighted Bellman backups

`MCTSConfig.backup` is `sampled` by default. The `empirical_bellman` option
changes the action-value estimator, independently of exploration and exact-tail
choices. For a private history h and action a, let N(h,a) count all sampled
outcomes, and C(h,a,o,alive) count continuing outcomes with observation o:

Qhat(h,a) = Rbar(h,a) + gamma * sum_o
            C(h,a,o,alive) / N(h,a) * Vhat(h,a,o,alive).

Rbar is the sample mean immediate reward, except at the root where the existing
full-belief expected reward is used. Vhat is the maximum of currently evaluated
action estimates at that history. A frontier with no evaluated actions uses its
explicit first rollout estimate until expansion; an absent estimate raises, never
silently becomes zero. Untried actions are still scheduled by UCB and are not
assigned artificial zero-valued alternatives. Exact one-step nodes use all their
integrated action values. Every ancestor on the sampled path recomputes its
selected action from current child values, not an average of old child estimates.

These are **empirical** chance probabilities. The denominator includes terminal
samples, with zero continuation. Outcomes are neither maximized nor given equal
weights, and surviving outcomes are not renormalized to probability one. The
probability count is independent of descendant action counts and reservoir size.
The maximizing decision is always attached to a private history, never a sampled
hidden state. Parent reward and future value can be added by linearity of
expectation; no independence between reward and next observation is assumed.

The backup structure is related to the Max-Monte-Carlo backups in Keller and
Helmert (2013), section MaxUCT:
https://ai.dmi.unibas.ch/papers/keller-helmert-icaps2013.pdf . That work describes
MDP decision/chance nodes; this implementation applies the construction to
private-history nodes with sampled immediate rewards, public survival and an
optional finite-belief tail. Its results are not automatically a convergence
proof for this implementation or its nested finite-budget opponent models.

For a fixed finite-horizon model, a consistency argument would require reward
and outcome estimates to converge at every relevant history and enough visits
for every available action and positive-probability branch. Backward induction
then makes the weighted backup approach the history Bellman equation. This
conditional argument is not a finite-budget guarantee. Maxima of noisy estimates
can be optimistic; rare or unseen histories and rollout initialization can still
matter. No solved-node labels, confidence certificates or action elimination are
introduced. Compare action loss and value error separately, and charge the extra
per-backup work (linear in the number of discovered continuing observations).


## Fixed-depth L2 comparison contract

The matched L2 runner now explicitly models an exact L1 opponent that replans
at a fixed depth, distinct from both shared-countdown planning and production
finite-budget modeled MCTS. See [L2_CONTRACT.md](L2_CONTRACT.md) for its joint
belief Bellman equation, observation/subjective update law, tie convention and
analytic tests. This is a scoped best-response comparison, not an equilibrium
claim or a change to global production semantics.


## L2 response to finite-computation modeled policies

The finite-opponent reference treats SolverBank's deterministic modeled MCTS
policy as part of the transition law of the interactive state. It maximizes
only after marginalizing hidden events at a private observation history.
Full immutable opponent beliefs and the declared subjective update are retained;
no scalar rounding or exact-policy substitution is allowed. See L2_CONTRACT.md.

A representation-invariance defect was corrected: equal order-independent
beliefs previously shared a search seed but could sample different atoms from
the same random variate due to insertion order. Canonical mass ordering restores
the requirement that cached finite policies are functions of the immutable
model and settings. This changes some finite trajectories, not the physical
model or Bellman objective; old empirical evidence retains its source scope.


## Optional conditional rewards at tree histories

MCTSConfig.exact_history_rewards integrates each visited tree action's immediate
reward against the full finite joint posterior b_h and modeled policy:
r_bar(h,a) = sum_x b_h(x) sum_events P(event | x,a) R(x,a,event).
The production reward integrator enumerates physical transitions and opponent
actions. It receives a private-history model, never an oracle Q or the sampled
hidden state. Own action, private observation and public continuation determine
the next posterior. Root integration already applies independently of this option.

Linearity gives Q(h,a) = r_bar(h,a) + gamma E[V(next history) | h,a].
Replacing a sampled immediate term therefore preserves the target under the
declared model. It does not make adaptive tree estimates unbiased or guarantee
a lower variance for their sum: reward and continuation can be correlated.
Empirical chance frequencies, noisy child maxima and frontier rollouts remain.

The option works with sampled-return and empirical Bellman backups. It is
independent of exact_final_step: the latter solves the last decision's complete
action set, whereas exact_history_rewards still visits and counts actions.
Non-boundary rollout rewards remain sampled; they use the declared domain
rollout policy. Terminal events retain zero continuation and remain in empirical
action-count denominators. No hidden-state maximization is introduced.

Full private-history propagation can be expensive for larger hierarchies.
Both options default to false. Configuration enters deterministic policy seed
identity, so adding a field can change finite trajectories even when it is false.
Compare control/candidate arms on the same source and same complete modeled
configuration; historical source-bound evidence cannot be transferred silently.


## Optional two-decision exact boundary

With `exact_final_step=True`, `exact_tail_depth=2` replaces the last two
protagonist decisions by the finite-belief Bellman boundary:

Q2(b,a) = r(b,a) + gamma sum_o P(o,alive | b,a) max_u r(U(b,a,o,alive),u).

The production bank enumerates the unconditional observation/survival law and
expected rewards; the recursive filter constructs the full joint posterior.
Private opponent actions/observations are marginalized, never disclosed to the
protagonist maximization. Terminal mass is zero continuation, not discarded by
renormalizing survivor probabilities. The modeled opponent's fixed computation
and replanning depth remain unchanged. No benchmark reference enters the solver.

This follows by conditioning the second reward on own observation and public
survival, then optimizing the final action separately at each resulting belief.
It removes chance, action-ranking and rollout errors inside the two-step boundary
relative to the finite model, up to floating-point arithmetic. Empirical chance
errors, action allocation and frontier policy errors above it remain. New search
trajectories can regress at finite budget; this is not a global error certificate.
Enumeration is bounded by the filter branch budget and propagates failures.
Tail and observation-law caches have at most 1024 entries each, not a byte bound.

Only depths one and two are accepted. One is the unchanged default and is omitted
from canonical MCTS config serialization to preserve historical policy seeds;
two is explicit. This compatibility convention supersedes the earlier general
warning about automatically inserting new default fields into seed identities.
Protagonist tail-two comparisons change protagonist seeds, not the finite modeled
opponent. See CONTINUATION_PROTOCOL.md for the required empirical checks.


## September 28: opt-in final root chance scoring

Implemented exact_root_chance=False by default; True requires empirical Bellman.
After identical seeded search, fully covered root actions use production expected
reward plus gamma times the unconditional observation/survival law weighted by
saved child estimates. Terminal probability is not renormalized. Missing positive
support causes whole-action empirical fallback with missing-mass telemetry;
enumeration failures propagate. Exact tail roots bypass scoring. The tree keeps
empirical Q; candidate evaluator rows separately record recommendation Q/policy,
search Q and coverage. The option is absent from search seeds, present in enabled
metadata, and omitted from default metadata. No modeled-opponent settings change.

For fixed child estimates this removes the direct root frequency discrepancy.
Adaptive child estimates still contain deeper chance, local action and frontier
errors; unbiasedness or universal variance reduction does not follow. The selected
post-hoc corrections motivated this ablation but are not candidate run outcomes.
The source-locked 14-pair pilot is docs/ROOT_CHANCE_PROTOCOL.md. Antigravity runs it;
Codex has not run the benchmark panel. All failed gates and .020 +1e-8 remain fixed.
No global-default promotion is justified.

## September 29: seven-case diagnostic audit

Verified unchanged source/input locks, all seven full row replays, independent
reconstruction of saved trace contributions and empirical-to-recommendation
margin identities. The recomputed identity residual is zero in floating-point
arithmetic for these rows; trace reconstruction residuals are at roundoff scale.
This is an algebraic audit, not an asymptotic sampling-rate certification.

All five nonzero traces resolve completely: 244/59/35/35/35 expanded histories,
408 total, no pending or unresolved terms. Their signed empirical margin errors
come from chance contributions; local action loss and rollout-frontier totals
are zero on the traced paths. Exact-boundary Q discrepancies are at most
1.4210854715202004e-14. The two corrected cases have zero candidate-versus-optimal
weights and no expanded paths; zero traces do not certify their deeper estimates.

For new violation P11/C556: empirical target .07513694187825237 plus scoring
increment .01063350217182979 equals recommendation margin error
.08577044405008216. Policy loss remains .08501670289316543; margin error is a
different quantity. Empirical signed chance contributions by remaining horizon:
H8 -.010538404257424, H7 +.013466187355586, H6 +.024069130820679,
H5 -.004161209683513, H4 +.005990855076252, H3 +.046310382566674.
These incremental terms may be summed; propagated child errors across depths
must not be summed again. Root chance uses exact child values in its diagnostic
term, whereas the actual score substitution weights estimated child values.
Consequently the scoring increment is not exactly minus that root chance term:
a .000095097914406 interaction remains under the changed weights.

The empirical OR continuation error is +.040225973912692 while L's is
-.045449372222988. Their mismatch survives root integration. These fixed traces
support deeper chance-estimation error as the mechanism in these five cases, not
a universal campaign-wide causal claim. No O(1/sqrt(N)) or O(1/sqrt(N_child)) rate
has been established for adaptive search values, maxima, correlated histories
and finite frontiers. Likewise OL/OR have one private-observation branch here,
not deterministic hidden transitions: opening resets Tiger Left/Right with
probability .5 each under this physics.

Diagnostic external123.57s, parent monotonic122.940329s, worker sum121.320996s;
serial intervals fit parent elapsed. Monitored RSS83.703125–131.703125 MiB,
GNU-time peak131.042969 MiB. Historical timing discrepancies are not erased.
Audit: results/oracle/l2_root_chance_diagnosis_review_20260929/audit.py and audit.json.
No new search/reference solve or solver source change in this audit.

Next implementation design: docs/RECURSIVE_CHANCE_PLAN.md. It proposes recursive
post-search integration on the same observed tree, with full-history production
kernels, explicit missing-support/frontier/action-coverage reporting and unchanged
search trajectories. It has only a conditional full-coverage Bellman guarantee;
it is not implemented or qualified yet. No new Antigravity experiment is ready
until engineering tests and a paired protocol are supplied. Preserve defaults,
all historical failed gates and the frozen loss tolerance.


September 29 implementation follow-up: recursive_chance is now an opt-in final
post-search evaluator; defaults and modeled opponents remain unchanged. It
integrates production chance laws recursively over supported observed histories,
propagates frontier/unevaluated-action/subtree uncertainty, and uses whole-action
empirical fallback for missing positive support. Search Q and all tree statistics
are preserved, and both final scoring flags are excluded from search seeds.

The source-frozen 23-case three-arm pilot is docs/RECURSIVE_CHANCE_PROTOCOL.md,
contract benchmarks/recursive_chance_pilot_20260929.json, SHA256
 a843f1aaca898e9652eb7302248c8aad473e23ece97864e57121de8f468b26a4.
It includes all seven diagnosed cases plus fourteen previously correct neighbors
and H1/H2 controls, with exact historical empirical and root-only row replay.
Codex has not run benchmark cases. Antigravity runs the declared pilot only;
source locks for prior completed experiments remain preserved and are not updated.
The root-only broad result still has one violation/five errors, and the original
fresh tail-two gate and all historical failures remain unchanged.

## September 29: recursive chance pilot review

All 23 three-arm cases completed. Source lock (78 Python files), contract/input
hashes, full empirical and root-only historical row replays, candidate task/config
semantics, separate three-arm tree digests and raw-search Q identity check out.
Greedy policies, losses and Q errors were recomputed from preserved rows. All
three arms have no greedy ties. This review also reconstructed every node/action
coverage counter, legal-action gap and subtree-flag dependency from the telemetry.
No new protagonist search or reference solve was run.

Recursive recommendations are optimal on 23/23: zero strict errors and zero
.020 +1e-8 violations, with no control regressions. Empirical baseline has four
errors/one violation; root-only has five errors/one violation. All seven diagnosed
cases and sixteen previously correct/boundary controls are optimal. This is
selected development evidence; it does not replace the root-only broad failure
or the original fresh tail-two FAILED gate.

Value accuracy is not uniformly improved: mean maximum Q error is
.569969262380566 (empirical), .569839930753683 (root-only), .662375580144034
(recursive). Maximum Q error increases from3.849293549166731 to4.322188276076147.
For P11/C556 the recursive L-over-OR margin is .094363867026054; L's value error
is +.011733109753155. The decision is corrected, but its subtree is approximate.

Coverage totals: 36,221 evaluated histories, 26,976 integrated action scores,
75,888 exact-tail scores, 2,830 whole-action empirical fallbacks, 515 rollout
frontiers, 1,433 histories with unevaluated legal actions, and 4,965 approximate
histories. Fallbacks occur at remaining horizons H5:37, H4:318, H3:2475; frontiers
at H4:41 and H3:474. These are local counts, not root probabilities. At the roots
there are 63 complete-support integrated action scores and six exact-tail scores,
with zero root-local fallbacks. Descendant approximation still propagates.

Only the H1/H2 cases (indices21/22) have fully exact root subtrees, and their
Q vectors agree with the finite reference to roundoff. Sixteen selected root
ACTIONS are marked exact and agree with their reference action Q to roundoff,
but only two complete roots are certified: exactness of the chosen action alone
does not bound approximate alternative actions. The other seven chosen actions
remain approximate, even where observed Q error happens to be zero. This supports
the conditional Bellman argument, not an unconditional exact-planning theorem.

Timing: external1237.60s, parent monotonic1236.994208s, parent Unix1236.994208s,
worker sum1228.132554s; serial intervals fit parent elapsed. Monitored RSS range
77.289063–139.234375 MiB; GNU-time peak147.117188 MiB is a different measure.
Policy-time sums empirical332.566311s, root-only335.841107s, recursive333.550568s;
recursive/empirical pooled ratio1.002960 and mean-pair ratio1.018294 are descriptive,
not isolated overhead estimates under fixed arm order and system load. Earlier
clock discrepancies remain preserved; these clocks are consistent for this run.

Review artifacts: results/oracle/l2_recursive_chance_codex_review_20260929/.
The audit extends the preserved supplied audit with full metadata and recursive
coverage checks; Antigravity's original audit remains unchanged at
results/oracle/l2_recursive_chance_review_20260929/. Both record that saved digests
are compared, not independently reconstructed full trees, and reference vectors
are historical rather than newly solved in this review.

Next: prepare the broader observed comparison over all 4,800 cases with the
recursive candidate frozen, including explicit coverage/regression/resource
reporting. Do not tune this candidate on the successful selected panel, start a
fresh qualification, or promote defaults. A new broad protocol and resource plan
must be prepared before Antigravity runs again. No solver edits or test reruns
were needed for this artifact review; the last full suite remains 292 passed.
All historical failed gates and the frozen tolerance remain unchanged.
