"""Additive attribution of a search tree's value error against an exact reference.

For a solved tree and an exact reference that can evaluate any history, the
error of an estimated action value splits exactly (zero residual) into:

- immediate error: estimated mean immediate reward minus the exact one;
- chance error: gamma * sum_o (empirical frequency - exact probability) * V*(h a o);
- continuation error: gamma * sum_o frequency * (V_hat(h a o) - V*(h a o)).

``describe_node`` computes these per history; ``trace_margin`` follows the
largest weighted residuals best-first to explain a root decision margin. This
is an algebraic attribution of one saved tree, not a causal ablation, and it
does not establish statistical bias.
"""

import math


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
