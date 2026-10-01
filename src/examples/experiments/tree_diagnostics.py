"""Post-search Tiger L2 tree diagnostics against the finite-policy reference.

This development tool replays saved evaluator rows exactly before inspecting the
tree. Reference values are never supplied to planning. The optional candidate
changes only the protagonist tail depth; control replays preserve the original
RNG contract and both arms preserve the opponent model and horizon. A value error at a
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
from ipomdp.canonical import belief_digest
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
    if not node.action_counts:
        # Exact integrated tails have Q values but intentionally no action counts.
        return dict(result, kind="final_step" if horizon == 1 else "exact_tail", actions={})
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


def trace_margin(node, model, horizon, reference, identity, policy, max_nodes=256):
    """Best-first additive attribution of a root estimated-versus-exact margin.

    Expand the largest absolute weighted residual, not the largest raw error.
    Continue along uniform greedy actions only, subtracting local action loss.
    Keep both signed and absolute unresolved mass: cancellation is not coverage.
    This is an algebraic attribution of one saved tree, not a causal ablation.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    exact = reference.q_values(model, horizon)
    best = max(exact.values())
    optimal = [a for a, q in exact.items() if q == best]
    coefficients = {
        a: policy.get(a, 0.0) - (1 / len(optimal) if a in optimal else 0) for a in exact
    }
    target = math.fsum(w * (node.action_values[a] - exact[a]) for a, w in coefficients.items())
    totals = dict(
        immediate=0.0, chance=0.0, local_action_loss=0.0, rollout_frontier=0.0, exact_tail=0.0
    )
    pending, paths = [], []

    def add_actions(current, current_model, remaining, description, weights, history):
        for action, weight in weights.items():
            if not weight:
                continue
            data = description["actions"][action]
            totals["immediate"] += weight * data["immediate_error"]
            totals["chance"] += weight * data["chance_error"]
            branches = reference.branches(current_model, action)[1]
            for (observation, _, successor), branch in zip(branches, data["branches"]):
                if not branch["count"]:
                    continue
                child_weight = weight * reference.gamma * branch["frequency"]
                child = current.get_child(action, observation)
                child_history = (*history, dict(action=str(action), observation=str(observation)))
                error = branch["value"] - branch["exact_value"]
                pending.append(
                    (
                        abs(child_weight * error),
                        child_weight * error,
                        child_weight,
                        child,
                        successor,
                        remaining - 1,
                        child_history,
                    )
                )

    root = describe_node(node, model, horizon, reference, identity, levels=0)
    if root["kind"] in ("final_step", "exact_tail"):
        totals["exact_tail"] = target
    else:
        add_actions(node, model, horizon, root, coefficients, ())
    while pending and len(paths) < max_nodes:
        index = max(range(len(pending)), key=lambda i: pending[i][0])
        _, contribution, weight, child, successor, remaining, history = pending.pop(index)
        description = describe_node(
            child, successor, remaining, reference, identity, levels=0, history=history
        )
        paths.append(dict(weight=weight, weighted_value_error=contribution, **description))
        kind = description["kind"]
        if kind == "rollout_frontier":
            totals["rollout_frontier"] += contribution
        elif kind in ("final_step", "exact_tail"):
            totals["exact_tail"] += contribution
        else:
            totals["local_action_loss"] -= weight * description["local_action_loss"]
            chosen = description["greedy_actions"]
            add_actions(
                child,
                successor,
                remaining,
                description,
                {a: weight / len(chosen) for a in chosen},
                history,
            )
    unresolved = math.fsum(item[1] for item in pending)
    residual = target - math.fsum([*totals.values(), unresolved])
    assert abs(residual) < 1e-8, residual
    return dict(
        target_margin_error=target,
        contributions=totals,
        unresolved_signed=unresolved,
        unresolved_absolute=math.fsum(item[0] for item in pending),
        pending_histories=len(pending),
        expanded_histories=len(paths),
        reconstruction_residual=residual,
        paths=paths,
    )


def diagnose(original, trace_nodes=256, tail_depth=1):
    """Replay one declared control or evaluate its explicit tail candidate.

    Capture only the completed protagonist tree.

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
            exact_tail_depth=tail_depth,
            recursive_chance=original["planner_config"]["mcts"].get("recursive_chance", False),
            exact_root_chance=original["planner_config"]["mcts"].get("exact_root_chance", False),
            level=2,
            opponent_depth=cfg["max_depth"],
            opponent_belief=op["initial_belief_p"],
            opponent_budget=cfg["n_sims"],
            opponent_backup=cfg["backup"],
            opponent_exact_final_step=cfg["exact_final_step"],
            opponent_exploration=op["exploration"]["strategy"],
            opponent_exploration_const=op["exploration"]["c"],
        )[0]
    if tail_depth == 1:
        assert row == original, "Full seeded replay differs; do not interpret diagnostics"
    else:
        for key in ("opponent_model", "oracle_q", "horizon", "budget", "seed", "belief_p", "gamma"):
            assert row[key] == original[key], f"Candidate changed task semantics: {key}"
    assert len(captured) == 1
    planner, model = captured[0]
    reference = FinitePolicyL2Reference(planner.pomdp_model, planner.solver_bank, original["gamma"])
    tree = describe_node(
        planner.root,
        model,
        original["horizon"],
        reference,
        lambda m: belief_digest(m.belief),
    )
    trace = trace_margin(
        planner.root,
        model,
        original["horizon"],
        reference,
        lambda m: belief_digest(m.belief),
        original["policy"],
        max_nodes=trace_nodes,
    )
    return [
        dict(
            row=row,
            exact_replay=row == original,
            tree=tree,
            margin_trace=trace,
            trace_value_source="empirical_search_q",
            trace_policy=original["policy"],
            baseline_row=original,
        )
    ]


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
    parser.add_argument("--trace-nodes", type=int, default=256)
    parser.add_argument("--tail-depth", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.trace_nodes < 1:
        parser.error("--trace-nodes must be positive")
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
        trace_nodes=args.trace_nodes,
        exact_tail_depth=args.tail_depth,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    statuses = []
    jobs = {
        i: partial(diagnose, row, args.trace_nodes, args.tail_depth) for i, row in enumerate(cases)
    }
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
