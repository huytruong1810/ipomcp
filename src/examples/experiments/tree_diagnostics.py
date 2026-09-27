"""Post-search Tiger L2 tree diagnostics against the finite-policy reference.

This development tool replays saved evaluator rows exactly before inspecting the
tree. Reference values are never supplied to planning. It deliberately keeps
the opponent model, horizon and RNG contract unchanged. A value error at a
history is split into the estimation error of its greedy action(s), minus their
reference action loss; those two terms must not be conflated with statistical
bias or a proof of inadequate exploration.
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

from examples.experiments.planner_oracle_experiment import evaluate_case
from solvers.exact.finite_policy_l2 import FinitePolicyL2Reference
from solvers.i_pomcp import IPOMCPPlanner
from utils.process_supervisor import supervise_jobs


def describe_node(node, model, horizon, reference, identity, levels=1, history=()):
    """Inspect a history and a bounded number of descendant layers.

    Full immutable successor models come from reference conditioning, never a
    rounded physical-belief reconstruction. The history and canonical belief
    digest identify each node relative to the manifest's physics and policy.
    Unexpanded histories and frontier rollouts are explicitly distinguished.
    Empirical action errors use all observation branches, including zero-count
    branches; no absent child is assigned an invented value.
    """
    oracle = reference.q_values(model, horizon)
    result = dict(
        history=list(history),
        belief_digest=identity(model),
        remaining_horizon=horizon,
        oracle_q=oracle,
    )
    if node is None:
        return dict(result, kind="unexpanded")
    value = node.value_estimate()
    result.update(
        value=value,
        value_error=value - max(oracle.values()),
        visits=node.visit_count,
        action_counts=node.action_counts.copy(),
        estimated_q=node.action_values.copy(),
    )
    if not node.action_values:
        return dict(result, kind="rollout_frontier")
    best = max(node.action_values.values())
    chosen = [a for a, q in node.action_values.items() if q == best]
    chosen_exact = math.fsum(oracle[a] for a in chosen) / len(chosen)
    regret = max(oracle.values()) - chosen_exact
    estimation = best - chosen_exact
    assert abs(value - max(oracle.values()) - (estimation - regret)) < 1e-8
    result.update(
        kind="action_node",
        greedy_actions=chosen,
        local_action_loss=regret,
        greedy_estimation_error=estimation,
        unevaluated_actions=[a for a in oracle if a not in node.action_values],
    )
    if horizon == 1:
        # Exact integrated tails have Q values but intentionally no action counts.
        return dict(result, kind="final_step", actions={})
    actions = {}
    for action, estimate in node.action_values.items():
        reward, branches = reference.branches(model, action)
        n = node.action_counts[action]
        assert n > 0
        chance = continuation = 0.0
        details = []
        assert sum(node.continuation_counts.get(action, {}).values()) == n
        assert abs(math.fsum(p for _, p, _ in branches) - 1) < 1e-8
        for observation, probability, successor in branches:
            count = node.continuation_counts.get(action, {}).get(observation, 0)
            frequency = count / n
            exact = max(reference.q_values(successor, horizon - 1).values())
            child = node.get_child(action, observation)
            child_value = child.value_estimate() if child is not None else None
            assert not count or child_value is not None
            chance += reference.gamma * (frequency - probability) * exact
            contribution = reference.gamma * frequency * (child_value - exact) if count else 0.0
            continuation += contribution
            branch = dict(
                observation=str(observation),
                probability=probability,
                frequency=frequency,
                count=count,
                value=child_value,
                exact_value=exact,
                weighted_value_error=contribution,
            )
            if levels:
                branch["child"] = describe_node(
                    child,
                    successor,
                    horizon - 1,
                    reference,
                    identity,
                    levels - 1,
                    (*history, dict(action=str(action), observation=str(observation))),
                )
            details.append(branch)
        immediate = node.immediate_reward_means[action] - reward
        residual = estimate - oracle[action] - immediate - chance - continuation
        assert abs(residual) < 1e-8, (history, action, residual)
        actions[action] = dict(
            immediate_error=immediate,
            chance_error=chance,
            continuation_error=continuation,
            reconstruction_residual=residual,
            branches=details,
        )
    result["actions"] = actions
    return result


def diagnose(original):
    """Replay one declared row; capture only the completed protagonist tree.

    Patching is scoped to a single supervisor child process and is restored even
    on exceptions. It records the returned planner, not simulation events, so it
    consumes no random numbers and changes no search decisions.
    """
    captured = []
    method = IPOMCPPlanner.policy_for

    def record(self, model, modeled=False):
        policy = method(self, model, modeled)
        if self.key.level == 2 and not modeled:
            captured.append((self, model))
        return policy

    op = original["opponent_model"]
    cfg = op["config"]["mcts"]
    with patch.object(IPOMCPPlanner, "policy_for", record):
        row = evaluate_case(
            "mcts",
            original["horizon"],
            original["budget"],
            original["seed"],
            original["belief_p"],
            original["gamma"],
            exploration=original["exploration"]["strategy"],
            exploration_const=original["exploration"]["c"],
            exact_final_step=original["exact_final_step"],
            exact_history_rewards=original["exact_history_rewards"],
            backup=original["backup"],
            level=2,
            opponent_depth=cfg["max_depth"],
            opponent_belief=op["initial_belief_p"],
            opponent_budget=cfg["n_sims"],
            opponent_backup=cfg["backup"],
            opponent_exact_final_step=cfg["exact_final_step"],
            opponent_exploration=op["exploration"]["strategy"],
            opponent_exploration_const=op["exploration"]["c"],
        )[0]
    assert row == original, "Full seeded replay differs; do not interpret diagnostics"
    assert len(captured) == 1
    planner, model = captured[0]
    reference = FinitePolicyL2Reference(planner.pomdp_model, planner.solver_bank, original["gamma"])
    tree = describe_node(
        planner.root,
        model,
        original["horizon"],
        reference,
        lambda m: planner.solver_bank._belief_digest(m.belief),
    )
    return [dict(row=row, exact_replay=True, tree=tree)]


def main():
    """Run an immutable, resource-supervised development diagnostic batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Audited JSON with errors entries containing full row dictionaries",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    cases = [entry["row"] for entry in json.loads(raw)["errors"]]
    assert cases and all(r["level"] == 2 and r["backup"] == "empirical_bellman" for r in cases)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    manifest = dict(
        scope="Selected development errors; no validation gate",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").rglob("*.py"))
        },
        input_sha256=hashlib.sha256(raw).hexdigest(),
        cases=cases,
        workers=1,
        timeout_seconds=600,
        max_rss_mb=2048,
        descendant_layers=1,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    statuses = []
    jobs = {i: partial(diagnose, row) for i, row in enumerate(cases)}
    for i, result in supervise_jobs(
        jobs, workers=1, timeout_seconds=600, max_rss_mb=2048, log_directory=args.output / "logs"
    ):
        (args.output / f"case-{i}.json").write_text(json.dumps(result, indent=2))
        statuses.append(dict(id=i, status=result["status"], wall_seconds=result["wall_seconds"]))
        print(i, result["status"], flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(results=statuses, elapsed_monotonic_seconds=time.monotonic() - start), indent=2
        )
    )
    assert len(statuses) == len(cases) and all(r["status"] == "complete" for r in statuses)


if __name__ == "__main__":
    main()
