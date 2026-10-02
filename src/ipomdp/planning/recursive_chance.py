"""Final Bellman integration on an observed tree, without mutating search.

Exactness requires all legal actions and all positive continuation branches down
to exact boundaries. Local support alone does not certify a subtree. No oracle
or sampled hidden state enters scoring; finite inference errors propagate.
"""

import math
from collections import Counter


def recursive_recommendation(planner, model, root):
    cfg = planner.config.mcts
    records, cache = [], {}

    def visit(node, history_model, remaining, history):
        key = (node, history_model, remaining)
        if key in cache:
            return cache[key]
        rewards = planner._final_action_values(history_model)
        legal = set(rewards)
        values = dict(node.action_values)
        if not set(values) <= legal:
            raise ValueError("Search contains an illegal action")
        record = dict(
            history=history,
            remaining_horizon=remaining,
            unevaluated_actions=sorted(legal - set(values), key=str),
            actions={},
            subtree_exact=False,
        )
        records.append(record)
        if not values:
            value = node.value_estimate()
            if not math.isfinite(value):
                raise ValueError("Nonfinite frontier value")
            record.update(kind="rollout_frontier", value=value, estimated_q={})
            result = (value, False, values)
            cache[key] = result
            return result
        record["kind"] = "action_node"
        boundary = cfg.exact_final_step and remaining <= cfg.exact_tail_depth
        for action in values:
            if not math.isfinite(values[action]):
                raise ValueError("Nonfinite search value")
            if boundary or remaining == 1 or cfg.gamma == 0:
                if not boundary:
                    values[action] = rewards[action]
                record["actions"][action] = dict(
                    status="exact_tail" if boundary else "no_continuation",
                    subtree_exact=True,
                    missing_probability=0.0,
                    missing_branches=[],
                    approximate_child_probability=0.0,
                )
                continue
            law = tuple(
                (o, p)
                for o, p in planner.solver_bank.continuation_law(history_model, action)
                if p > 0
            )
            missing = [
                dict(observation=repr(o), probability=p)
                for o, p in law
                if node.get_child(action, o) is None
                or (
                    not node.get_child(action, o).action_values
                    and node.get_child(action, o).rollout_value is None
                )
            ]
            info = dict(
                status="empirical_fallback" if missing else "exact_probabilities",
                missing_probability=math.fsum(b["probability"] for b in missing),
                missing_branches=missing,
                continuation_probability=math.fsum(p for _, p in law),
                approximate_child_probability=None,
                subtree_exact=False,
            )
            record["actions"][action] = info
            if missing:
                # Whole-action fallback. Do not mix original and rescored branches.
                continue
            terms, approximate_mass = [], []
            for observation, probability in law:
                child_model = planner._history_successor(history_model, action, observation)
                child_value, exact, _ = visit(
                    node.get_child(action, observation),
                    child_model,
                    remaining - 1,
                    history + [dict(action=str(action), observation=repr(observation))],
                )
                terms.append(probability * child_value)
                if not exact:
                    approximate_mass.append(probability)
            values[action] = rewards[action] + cfg.gamma * math.fsum(terms)
            info.update(
                approximate_child_probability=math.fsum(approximate_mass),
                subtree_exact=not approximate_mass,
            )
        exact = not record["unevaluated_actions"] and all(
            a["subtree_exact"] for a in record["actions"].values()
        )
        record.update(subtree_exact=exact, estimated_q=values, value=max(values.values()))
        result = (record["value"], exact, values)
        cache[key] = result
        return result

    _, exact, scores = visit(root, model, cfg.max_depth, [])
    statuses = Counter(a["status"] for n in records for a in n["actions"].values())
    details = dict(
        root_subtree_exact=exact,
        nodes=records,
        action_status_counts=dict(statuses),
        visited_histories=len(records),
        rollout_frontiers=sum(n["kind"] == "rollout_frontier" for n in records),
        histories_with_unevaluated_actions=sum(bool(n["unevaluated_actions"]) for n in records),
        approximate_histories=sum(not n["subtree_exact"] for n in records),
    )
    return scores, details
