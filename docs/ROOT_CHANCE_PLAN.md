# Next candidate design: exact root chance scoring

Status: implemented opt-in candidate, September 28. The preceding tail-two gate
remains FAILED. Run instructions and the fixed 14-pair development selection are
in docs/ROOT_CHANCE_PROTOCOL.md. The pilot is now complete: 1 -> 0 violations, 4 -> 2 strict errors, no
regressions. See latest docs/REVIEW.md for the independent audit and next step.

## Evidence and narrow scope

The failed case's signed chance contribution is +.064423122 of a +.065272689
margin error. Exact probability weighting of saved root child values corrects
that choice; it also corrects one older selected tail-two near tie without
changing five other available optimal choices. Deeper chance errors remain.
This supports a minimal final-root-scoring ablation, not a claim that all internal
backups or rollout policies should be changed, or that seven trees qualify it.

## Implementation requirements

- Add an explicit opt-in final recommendation rule; keep the existing tree search,
  traversal budget, UCB, tail depth, history rewards, modeled policy and defaults.
  Use the production bank's continuation_law and full root private-history model,
  never reference probabilities or Q values in the solver.
- For complete root support, score each evaluated action as expected reward plus
  gamma times sum P(observation,alive | root,action) * saved child value. Public
  terminal probability contributes zero. Do not renormalize the survivor law.
  Preserve the empirical search Q vector and record the recommendation Q vector
  separately, so diagnostics cannot mistake a rescore for an empirical backup.
- Require values for every positive-probability continuing branch. Define and
  test an explicit coverage/fallback contract before coding the option. Never
  silently drop unseen probability, invent a zero continuation, or reconstruct
  missing values from an oracle. A conservative first ablation can use the
  empirical score for an action with incomplete support and report its fallback
  and missing mass explicitly; such an action is not exactly chance-weighted.
- Preserve the same search trajectory between the empirical and rescored arms.
  This option acts after search, so its metadata must be recorded while search
  seed derivation depends only on actual search settings. Keep modeled-opponent
  configuration and historical default seed identity unchanged. Test both the
  trajectory identity and the distinct final recommendation metadata.
- Keep diagnostics explicit about which Q vector they decompose. Verify source
  and full control-row identity against preserved rows before interpreting
  candidate results. Preserve inference-limit failures rather than falling back
  to repaired beliefs or a changed opponent law.

For fixed child estimates, the rule substitutes the model expectation over
observations for an empirical frequency sum. It removes that direct root
frequency discrepancy. Child estimates depend on the adaptive search, so this
is not an unconditional variance-reduction or unbiasedness theorem. It does not
eliminate chance error deeper in the tree and may regress on other cases.

## Checks and later experiment design

Test nonuniform observation probabilities, terminal mass, missing-branch behavior,
uniform greedy ties, source/seed and modeled-policy invariance, preservation of
empirical tree statistics, and agreement with a hand-computed final rescore.
Do not use a production oracle dependency to make those checks pass.

The declared paired development panel contains
the known failure, all three tail-two H6 development errors, and previously
correct controls. Then review accuracy, fallback coverage and resource costs
before extending to all observed cases. All observed validation cases now belong
to development evidence. Any subsequent fresh qualification requires a separate
unseen design; the .020 +1e-8 zero-violation rule and failed gates remain fixed.
