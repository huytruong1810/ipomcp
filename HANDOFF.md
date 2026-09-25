# Current review handoff

## Ownership and status

One checkout: /home/andyj1810/projects/ipomcp, branch fix/tiger-policy-inversion.
Codex owns code/math; Antigravity owns experiment execution. Do not edit source
during experiments or create source snapshots outside Git. Raw cases remain
local and Git-ignored unless explicitly archived elsewhere.

The 900-case exact-opponent L2 development study has been independently audited.
No solver/default change is justified by this audit. Antigravity should retain
all evidence and await the finite-budget modeled-opponent comparison contract
before further qualification runs.

## Verified results at 661fb59

Six panels at source01f36cf: d1/d2 x b_j=.085/.5/.915; H1-H3; budgets1k/10k;
seeds300-304; own beliefs .05/.2/.5/.8/.95. Exact tail, empirical Bellman,
bounded c=1, gamma=.95. Two workers, timeout240s, RSS limit2048MiB.

All 900 case files have complete/unique coverage, matching source fingerprints,
settings and status. Recomputed exact reference values for all 90 configurations
(45 per depth), every policy loss and max Q error. All recorded policies are
valid distributions. Summaries agree with case coverage/status.

899/900 first actions match the reference. The sole error:
d1, b_j=.915, H3, budget1000, b_i=.05, seed301, chosen L rather than OL,
loss .3377000000000012. Its 10k case has zero loss. All 450 budget10k cases
and all 450 d2 cases have zero strict errors. Maximum Q error remains 1.656668.
These are development results; .020 loss counts are diagnostics, not a declared
L2 validation gate, and cannot qualify untested beliefs/budgets/deeper agents.

For depth comparisons there are 45 configurations, not 30:
3 opponent beliefs x 3 horizons x 5 own beliefs. Eight optimal-action sets
change; eighteen configurations have Q/value changes above1e-4; maximum action
Q difference7.733. Physics is unchanged; the opponent policy law changes.

Opening is NOT terminal. TigerModel resets the physical state after opening
and continues. Opening Q errors here are at most1.43e-14 because, over H1-H3,
the post-reset short-horizon optimum is to listen. The continuation is present:
Q(open)=E[R(open)]-sum_(t=1)^(H-1) gamma^t. At b_j=.085/.915 the d1 initial
opponent opens and d2 listens; the report's former safe-opening/collision-risk
explanation was incorrect. No collision penalty exists in this reward model.

## Timing and provenance

Each panel's worker sum fits twice parent monotonic elapsed. All worker
start/finish intervals match durations, with at most two overlapping workers.
External timers and parent monotonic elapsed differ by fractions of a second
on these short panels. This does not resolve earlier long-run discrepancies.

timing.json stores parent monotonic ELAPSED plus realtime endpoints, not parent
monotonic start/finish timestamps. Per-worker endpoints are monotonic.
Do not invent stronger endpoint checks than the available metadata supports.

Raw directories: results/oracle/l2_fixed_d{1,2}_b{0.085,0.5,0.915}_20260925/.
External .time/.log files are siblings. Independent audit:
results/oracle/l2_fixed_review_20260925.json. Preserve original artifacts.
Documentation commits do not constitute a Git archive of these raw cases.

## Next coding phase: match finite-budget modeled opponents

The existing contract in docs/L2_CONTRACT.md uses an exact fixed-depth L1 policy.
Production instead asks SolverBank for a finite-budget private MCTS solve.
Its budget, depth, estimator, tail setting, exploration, deterministic search
seed and tie convention are all part of the opponent model. Replacing that
policy by exact L1 is not the same decision problem.

Plan before implementation:
1. Define the reference as an exhaustive L2 best response to the declared,
   deterministic finite-computation L1 policy, rather than to an exact L1.
   Keep the current exact-L1 experiment explicit as a distinct model.
2. Preserve complete immutable opponent MentalModel/FiniteBelief identity
   through reference transitions. Do not round or reconstruct a scalar belief
   and assume the resulting modeled policy is identical: search seeds hash the
   private belief representation and settings, so floating differences can
   change a finite-budget action. Define precisely how private updates and
   policy identity are shared across the reference and tree planner.
3. Fix opponent configuration and policy provider for a run. Record both real
   and modeled budgets/depths/estimators/exploration plus bank seed in manifests.
   Shared policy calls are legitimate model definitions; protagonist oracle
   values must remain outside its action-selection path.
4. Test H1 immediate reward, uniform H2 information limits, posterior
   correlation and private-information restrictions. Check that reference and
   planner query identical opponent policies at identical private models.
   Use independent tiny enumerations so a shared filtering bug is not accepted
   merely because both implementations use the same kernel.
5. Establish small H1-H3 development comparisons before increasing depth and
   modeled simulation budgets. Preserve resource failures; no cheaper reference
   or nearest-belief fallback on timeout.

This plan preserves fixed-depth production replanning rather than silently
changing it to shared countdown. Do not launch a production L2/L4/all39 suite
until that comparison and resource contract exist.

## Engineering and remaining gates

No source changes in this audit. Latest implementation verification remains
217 passing tests plus lint/format at the preceding implementation checkpoint;
tests were not rerun for these documentation edits.

The earlier 2000-case L1 accuracy gate remains passed for its specific settings.
Older 50k/200k gates remain failures. Global defaults remain sampled backups,
sampled final steps and empirical-range UCB. Long/deep Tiger before/after,
finite-prior error, production modeled-policy resources and remaining
demo/visualization semantic review are still pending.
