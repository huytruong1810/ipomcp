# Review handoff

Authoritative checkout: /home/andyj1810/projects/ipomcp (Antigravity local work).
Review worktree: /home/andyj1810/projects/ipomcp-review-20260916,
branch review/rigor-20260916. Changes have not been merged or pushed. The unwanted
Windows clones were already removed; this existing WSL worktree is not a new clone.

The integrated implementation replaces mutable latent search nodes with immutable
finite joint beliefs, uses recursive subjective filtering in both planners, and
removes the old JIT/reinvigoration paths. Public survival conditions private updates.
An exact immediate root-reward control variate reduces MCTS reward variance without
changing its expected backup target. Modeled MCTS budgets remain ten by default.

Validation: 116 tests pass in 378.50 seconds. All 39 reduced-budget suite conditions
pass. A full-budget L4-vs-L3 twenty-step seed-zero probe completes in 33.24 seconds,
peak RSS 182.25 MiB. A depth-three RTS twenty-step probe fails with zero finite
support for a right-creak observation: modeled RTS L1 differs from executing MCTS
L1 under a point prior. Do not launch the long full suite yet.

Thirty-seed twenty-step depth-five L3-vs-L2 benchmark:
- First integrated solver: I=-33.93, J=-2.03; 21.38 seconds/trial.
- With exact root reward: I=-21.10, J=8.23; 24.61 seconds/trial; max RSS 190.96 MiB.
- Earlier heuristic review: I=15.93, J=5.30; 223.61 seconds/trial.
- Antigravity snapshot: I=-10.10, J=0.90; 249.70 seconds/trial.

The latest paired I difference versus the earlier review is -37.03, descriptive
95% interval [-76.30, 2.24]. This does not establish equivalence. Source/model and
budget semantics changed, and timings overlapped other audit work. Preserve the
first regression as evidence; do not select only favorable iterations or seeds.

Raw results are under results/tiger-integrated-30x20,
results/tiger-integrated-cv-30x20, results/suite-integration-smoke-cv,
results/stress-integrated-l4-20steps.json and results/stress-integrated-rts-20steps.json.
Verification logs are under results/verification. Historical source snapshots remain
in sibling ipomcp-antigravity-baseline-20260917 and ipomcp-reviewed-benchmark-20260917.
Source hashes and Git identify provenance; no runtime schema migration is required.

Next: resolve the explicit opponent-model choice, enforce bank identity contracts,
qualify remaining conditions, add full-suite process supervision, consolidate demo
execution and reconcile the original checkout. Read BACKLOG.md and docs/THEORY.md.
Native Graphviz dot is missing, so optional rendering remains unverified.
