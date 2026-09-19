# Review handoff

Authoritative checkout: /home/andyj1810/projects/ipomcp, unchanged from the captured
Antigravity source. Review worktree: /home/andyj1810/projects/ipomcp-review-20260916,
branch review/rigor-20260916. Nothing is merged or pushed into the original.

The September 18 immutable-filter integration is followed by separate modeled
planner configuration, corrected frame binding, and a declared matched-computation
RTS/MCTS comparison. See docs/MODEL_SPECIFICATION.md for the public initialization
assumption. No posterior fallback, action lapse or live private-state sharing was
introduced. Solver lookup rejects mismatched physics identity; bootstrap retains
the registered private solver's frame when reusing it.

## September 19 controlled-comparison qualification

The modeled planner family is now independent of protagonist search. The controlled
comparison declares the same MCTS L1 budget (50,000), exploration rule, initial
empirical prior (2,500 samples) and search seed as the executing opponent. Each
agent maintains isolated private beliefs thereafter. RTS lookahead remains 500
particles. MCTS protagonist uses 20,000 simulations in this qualification panel.
This changes the scientific configuration; it is not a pure speed optimization.

Ten seeds, twenty steps, depth three, two workers, 300-second/3,072-MiB per-trial
limits: RTS completes 10/10 and MCTS completes 10/10. RTS means are I=3.10,
J=-53.00, 34.61 wall seconds/trial,
34.58 CPU seconds/trial, max RSS 168.50 MiB.
MCTS means are I=3.10, J=-53.00,
38.86 wall seconds/trial, 38.83 CPU seconds/trial,
max RSS 169.79 MiB. Timing overlaps other qualification work.
These ten-seed panels demonstrate successful execution, not performance equivalence
or general optimality. All twenty-one rows per trial and source hashes were verified.

The earlier family/budget-only panel retained independent initial priors and seeds:
RTS completed 5/10 and MCTS
completed 8/10. Failed trials
are retained and excluded from no purported full-panel mean. Matching only a
planner name and simulation count was insufficient to specify the policy kernel.

All 122 tests pass across unit and integration invocations (no xfails), including
Wumpus; all 39 reduced-budget smoke conditions pass. The existing L3-vs-L2 prior
benchmark is unchanged in experimental design; no new equivalence claim is made.
Ordinary full-suite process supervision, remaining full-depth condition coverage,
demo consolidation and authoritative-checkout reconciliation are still open.

Raw source-bound evidence: results/comparison-{rts,mcts}-public-20260919.
Earlier incomplete panels: results/comparison-{rts,mcts}-matched-20260919.
Smoke: results/suite-smoke-public-20260919. Test logs: results/verification.
The audit driver supports --condition prior, comparison-rts or comparison-mcts.
Comparison requires --sims-j 50000 --particles-j 2500; --particles-i sets RTS
lookahead only in the RTS condition, while both comparison initial priors use
2,500 samples. Effective budgets are explicit in the manifest. All new outputs
must use empty directories. Native Graphviz dot remains unavailable.

Next: full-depth coverage of other conditions, explicit treatment of intentional
model misspecification, full-suite resource supervision, conflicting bootstrap
registration guards, demo consolidation and safe integration. See BACKLOG.md.
Do not describe the long full suite as qualified yet.
