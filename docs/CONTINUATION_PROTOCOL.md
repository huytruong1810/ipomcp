# Deeper continuation diagnosis and bounded-tail candidate

Execution update (September 27): Antigravity completed both diagnostic arms and
the full development grid. Codex independently audited them; see the latest
BENCHMARK.md section and HANDOFF.md. The commands below are the historical
protocol, not a request to overwrite or rerun completed artifacts. The active
next experiment is TAIL2_VALIDATION.md.

Checkpoint: September 26, 2026. Codex implements and checks algebra; Antigravity
runs experiments. Continue in the existing WSL checkout on main. No experiment
in this protocol was run by Codex at this checkpoint. Historical failed gates,
raw artifacts and the frozen loss tolerance remain unchanged.

## What the saved evidence establishes

Read all six completed rows under
`results/oracle/l2_descendant_diagnosis_20260926/`. Their trees contain root and
immediate-child decompositions, but grandchild values without grandchild action
counts, Q vectors or deeper trees. Consequently these files cannot determine
whether a grandchild error is caused by chance, local action loss or a frontier.
Do not infer a deeper cause from its magnitude or sign.

For selected root action minus an oracle-best action, multiply each child action's
branch error by gamma times the root empirical frequency, averaging exact greedy
ties. Ranking all these signed contributions gives the following largest paths.
The sign includes the root action contrast; it is not the raw value-error sign.

| Case | Root action / observation; child action / observation | Grandchild arrivals | Signed margin contribution |
| --- | --- | ---: | ---: |
| 0 | OL / (S,S); L / (GR,S) | 24,156 | +.018289539 |
| 1 | OR / (S,S); L / (GR,S) | 23,852 | +.022058852 |
| 2 | OL / (S,S); L / (GL,S) | 23,188 | +.072410186 |
| 3 | L / (GL,S); OR / (S,S) | 22,006 | +.022677756 |
| 4 | L / (GR,S); OL / (S,S) | 22,551 | +.021266289 |
| 5 | L / (GL,S); L / (GL,S) | 1,223 | +.029286855 |

Case 2's first path estimates 5.469092286 versus reference 5.629440428;
its competing L/(GR,S), OL/(S,S) path estimates 4.692419835 versus 4.600601624,
contributing +.068716516. Negative paths cancel part of these: the signed sum
of all grandchild contributions is +.084761663. Across cases 0–5 the corresponding
sums are .021525395, .038406859, .084761663, .038047356, .012659232, .036092312,
reproducing the previous contribution table. These are read-only calculations
from existing artifacts, not new planner executions. Arrival counts do not reveal
allocation among grandchild actions, still less their descendants.

## Implemented plan and theory

1. Extend the post-search diagnostic to trace the largest absolute *weighted*
   residual first, down the current uniform-greedy continuation policy. Each
   action error splits into immediate reward, empirical chance and successor
   value terms. At each decision subtract local oracle action loss. A frontier's
   entire remaining error stays explicitly attributed to its rollout estimate;
   it is not called action loss or sampling variance. That frontier residual may
   combine rollout policy suboptimality and sampled-return noise; this trace does
   not distinguish those further. Exact boundaries are separate.
2. Bound work at 256 expanded histories per case. Record the complete path,
   belief digest, horizon, visits, action counts, estimates and reference values.
   Preserve both signed and absolute unresolved contributions; a small signed
   residual can conceal cancellation. Absolute unresolved value error is not a
   bound on cancelling components *inside* an unexpanded history. Every result
   checks additive reconstruction
   to 1e-8. This is an attribution of an adaptive saved tree, not a causal test.
3. Implement an opt-in exact two-decision boundary. With a full history belief b,
   Q2(b,a) = r(b,a) + gamma sum_o P(o,alive|b,a) max_u r(b[a,o,alive],u).
   Production reward and observation kernels supply the expectation; the shared
   recursive filter supplies the full posterior, including private opponent
   models. Maximize only after conditioning on the protagonist observation.
   Terminal probability contributes zero and is not renormalized away. No
   reference code or oracle values enter planning.
4. Use `exact_final_step=True, exact_tail_depth=2` in both tree and rollout
   boundaries. Larger tail depths are rejected to bound the new algorithmic
   scope. Non-boundary rollouts and empirical backups remain as declared.
   Exact two-step values remove all three error mechanisms *within that boundary*
   up to floating-point arithmetic. No proof says these are the dominant errors
   in the six cases, or that a new adaptive search improves every root action.
   This candidate tests a mechanism, not a known successful repair of the gate.

The finite opponent's depth/budget/policy and the evaluator's first-action loss
are unchanged. Default tail depth is one and is omitted from serialized MCTS
seed/config identity, preserving historical identities; nondefault depth is
explicit. This also preserves the modeled opponent between arms. The candidate
changes the protagonist seed and trajectory, so the experiment is not a nested
prefix or common-random-number comparison. New enumeration and bounded caches
cost time/memory; traversal counts are not equal-compute guarantees.

## Antigravity: historical execution protocol (completed)

Use the checkout's `.venv/bin/python -m ...`; its installed `pytest` entry-point
has a stale interpreter path, so tests use `python -m pytest` as well. Do not
recreate the environment just for that launcher. Use new output directories;
never overwrite the September 26 artifacts or substitute candidate rows for them.

The input is the audited six-error JSON. Its SHA256 is
`d273deb4c7464eeeea9f9fa3251a9bbeccffe5691749406e13e926b1295e3ab9`.
The diagnostic manifest records all current source hashes, source commit, input
hash, exact rows and limits, including uncommitted code. Capture host/workload
provenance and external timing separately. One worker, 600 seconds per case,
2048 MiB cap; failures remain saved failures, never omitted from a denominator.

First run the control. All six entire evaluator rows must equal their originals:

```bash
cd /home/andyj1810/projects/ipomcp
/usr/bin/time -v -o results/oracle/l2_deeper_control_20260926.time \
  .venv/bin/python -m examples.experiments.tree_diagnostics \
  --input results/oracle/l2_rewards_100k_review_20260926.json \
  --output results/oracle/l2_deeper_control_20260926 \
  --trace-nodes 256 --tail-depth 1
```

If any control replay differs, stop and inspect identity/provenance; do not
interpret candidate improvements. If controls match, run the candidate:

```bash
/usr/bin/time -v -o results/oracle/l2_deeper_tail2_20260926.time \
  .venv/bin/python -m examples.experiments.tree_diagnostics \
  --input results/oracle/l2_rewards_100k_review_20260926.json \
  --output results/oracle/l2_deeper_tail2_20260926 \
  --trace-nodes 256 --tail-depth 2
```

The runner asserts unchanged oracle Q vectors and modeled-opponent records.
Candidate rows are labeled `exact_replay=false` when changed, never rejected for
failing to reproduce the control choice. Both traces use the *original* root
policy contrasted with uniform oracle-best actions, so their attribution refers
to the same competing decisions even if the candidate corrects the action.
The candidate's actual returned policy, first-action loss and every root Q remain
in its row. Report any newly selected third action separately.

Report, per case: original/candidate policy, loss, strict error (>1e-8), violation
(>.020 +1e-8), maximum Q error; signed immediate/chance/local-loss/frontier/tail
contributions; unresolved signed AND absolute error and pending history count;
top weighted paths with remaining horizons and per-action counts. Keep raw local
loss distinct from its signed margin contribution. Compare contributions grouped
by remaining horizon, especially H2, to assess whether the tail targets an observed
source. If unresolved absolute contributions materially affect interpretation,
request a declared larger trace cap with new artifacts; do not silently treat
unresolved terms as rollout effects or rerun with an unrecorded resource limit.

Review these selected-case results before any large sweep. If they support the
candidate, the next development comparison must include all 1,440 already observed
H5/H6/H8 cases (six conditions: modeled budgets 25/100 x b_j .085/.5/.915,
seeds 7000–7007, own beliefs .002/.0175/.0275/.0375/.0875/.9125/.9625/.9725/.9825/.998),
including previously correct cases. Preserve root 100k, gamma .95, bounded c=1,
empirical Bellman, exact history rewards, opponent depth20/sample backup/normalized
c=1/no exact opponent tail. Copy each historical manifest's complete settings and
change only protagonist exact_tail_depth to two; confirm control identity before
using the historical rows as a control. Report resources with concurrent workload
caveats. This larger study is a later review checkpoint, not launched here.

Selected improvements cannot qualify defaults. The all-case .020 criterion,
1e-8 numerical slack, historical ac3df1c/f4d2aef failed gates and unchanged global
defaults remain binding. A future validation requires a separately frozen unseen
design; neither these six cases nor the observed 1,440-case grid is fresh evidence.
