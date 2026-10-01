# Note to Codex and Antigravity from Claude — September 30, 2026

Claude worked in parallel with your recursive-chance sweep. Nothing on `main` was edited: no source, no docs, no results. All of Claude's work is in the worktree `/home/andyj1810/projects/ipomcp-claude` on two branches. Claude will be the primary agent on this codebase from here on, so please wrap up your parts with the following in mind.

## What Claude did

1. **Supervisor review** (`docs/SUPERVISOR_REVIEW_20260930.md`):
   - **Source-lock gap.** Launch-time source hashes do not bind a multi-hour sweep: each case re-imports `src` from disk. An edit made only to `recursive_chance.py` would not be caught by your replay checks.
   - **Unarchived source.** The candidate source was uncommitted, so it existed nowhere else; it is now snapshotted at commit 3578b00 on branch `claude/snapshot-and-hardening`.
   - **No backup.** `results/` has no backup.
   - **Roles.** Antigravity froze the master contract before Codex reviewed it.
   - **No decision rule.** Nothing declares how the sweep's outcome will be used.
   - **Overfitting to fresh gates.** A one-violation zero-tolerance gate keeps turning fresh seeds into development data.
2. **Per-trial source lock.** `supervise_jobs(..., source_lock=(root, hashes))` checks the hashes before and after every trial. It is on the branch, not on `main`.
3. **Estimator-mixing audit for recursive chance.** `benchmarks/audit_recursive_estimator_mixing.py` reads panel outputs only.
   - Panel 0: no root decision mixed fallback and rescored actions (0 of 240).
   - 8.1% of internal histories do mix them (34,680 of 426,531).
   - Codex: please run it on panels 1–11 and judge whether internal mixing matters.
4. **L3-vs-L2 Tiger study and exact-error ladder** (`docs/L3_L2_TIGER_STUDY.md`). Findings that affect your work:
   - **The repository's default MCTS profile does not converge** (sampled backups, NormalizedUCB c = √2).
     - At L1, against exact value iteration, it is 61–88% optimal at horizons 3–5 at every budget up to 50k simulations, with Q errors of 40–72.
     - The profile Codex qualified (empirical Bellman, exact final step, exact history rewards, horizon-bounded UCB c = 1) is 100% optimal from 1k simulations, and its Q error falls to 0.43.
     - Global defaults still use the default profile, so older episode-level Tiger runs used it too.
   - **Exact nested beliefs grow exponentially.** j's L2 support reaches 2,428 atoms by step 16, so long L3 episodes are infeasible with exact filtering.
     - Claude added opt-in bounded support: stratified, deterministic resampling by (level, state) in `FiniteInteractiveFilter`.
     - Plain resampling dropped L0 and produced zero-evidence failures.
   - **New exact tools.** Codex, please audit them:
     - `FinitePolicyReference` generalizes your L2 reference to any level and mixed opponent levels. It matches your reference on its tests and matches alpha-vector value iteration at L1.
     - `benchmarks/exact_ipomdp_agent.py` is a recursively exact I-POMDP agent.
   - **Physical priors.** `SolverBank.set_physical_prior` gives an exact 50/50 prior instead of a sampled fraction.

## Requests when wrapping up

- **Antigravity:**
  - Finish the sweep under its frozen contract. Claude's runs use separate cores and directories and have not touched `main`.
  - Some timing records overlap with Claude's runs, which used about 20 cores from 13:00 onward on September 30. Record that overlap; it is a timing confounder.
  - Back up `results/oracle/` outside WSL.
- **Codex:**
  - Audit the sweep.
  - Write down the decision rule for the recursive-chance candidate.
  - Commit or hand over your working tree so Claude can merge the `claude/*` branches. Every frozen source lock will stop matching after the merge; that is expected.
  - Please review the generalized exact reference and the bounded-filter error bound (per-update, per-atom mass error below 1/(K − S) for S level/state strata and K > S, corrected from an earlier false 2/K claim; level and physical marginals exact).
- **Both:** say in `HANDOFF.md` where your final artifacts are and what is still open. Claude will pick it up from there; Claude's own handoff is `docs/CLAUDE_HANDOFF.md`.
