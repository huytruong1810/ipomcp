"""Read-only audit: do recursive-chance decisions compare different estimators?

Recursive scoring keeps the empirical search Q for any action with missing
positive continuation mass (whole-action fallback) while rescoring its siblings
with production chance probabilities. A greedy choice between a fallback and a
rescored action therefore ranks two different estimators. Global fallback
counts do not reveal whether this happens at the decisions that matter.

For every completed triple this reports, at the root and at internal histories:
the estimator class of each action, whether the greedy action and its closest
competitor use different classes, and the resulting estimated margin. Root
choices are also scored against the reference for all three arms. Nothing is
recomputed; saved rows are only read. This is descriptive telemetry, not a
causal attribution of any error.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

RESCORED = {"exact_probabilities", "exact_tail", "no_continuation"}
TOLERANCE = 1e-8
VIOLATION = 0.020 + 1e-8


def estimator_class(status):
    if status in RESCORED:
        return "rescored"
    if status == "empirical_fallback":
        return "fallback"
    raise ValueError(f"Unknown recursive action status {status!r}")


def decision(q, classes):
    """Greedy set, closest competitor and whether their estimator classes differ."""
    best = max(q.values())
    greedy = sorted((a for a, v in q.items() if v >= best - TOLERANCE), key=str)
    rest = {a: v for a, v in q.items() if a not in greedy}
    if not rest:
        return dict(greedy=greedy, competitor=None, margin=None, mixed=False)
    competitor = max(rest, key=rest.get)
    greedy_classes = {classes[a] for a in greedy}
    return dict(
        greedy=greedy,
        competitor=competitor,
        margin=best - rest[competitor],
        mixed=greedy_classes != {classes[competitor]} or len(greedy_classes) > 1,
    )


def row_outcome(row):
    loss = row["first_action_loss"]
    return dict(strict_error=loss > TOLERANCE, violation=loss > VIOLATION, loss=loss)


def audit_case(path):
    case = json.loads(path.read_text())
    if case["status"] != "complete":
        return dict(file=path.name, status=case["status"])
    row = case["rows"][0]
    candidate = row["candidate"]
    nodes = candidate["recursive_chance"]["nodes"]
    root = nodes[0]
    if root["history"] or root["remaining_horizon"] != candidate["horizon"]:
        raise ValueError(f"{path}: first record is not the root history")

    internal = Counter()
    for node in nodes:
        if node["kind"] != "action_node" or not node["actions"]:
            continue
        classes = {a: estimator_class(info["status"]) for a, info in node["actions"].items()}
        kinds = set(classes.values())
        internal["action_histories"] += 1
        internal["mixed_histories"] += len(kinds) > 1
        internal["all_fallback_histories"] += kinds == {"fallback"}
        if len(kinds) > 1:
            internal["mixed_decisions"] += decision(node["estimated_q"], classes)["mixed"]

    classes = {a: estimator_class(info["status"]) for a, info in root["actions"].items()}
    root_decision = decision(candidate["estimated_q"], classes)
    oracle = candidate["oracle_q"]
    best = max(oracle.values())
    return dict(
        file=path.name,
        status="complete",
        horizon=candidate["horizon"],
        belief_p=candidate["belief_p"],
        seed=candidate["seed"],
        opponent_budget=candidate["opponent_model"]["config"]["mcts"]["n_sims"],
        opponent_belief=candidate["opponent_model"].get("initial_belief_p"),
        root_subtree_exact=candidate["recursive_chance"]["root_subtree_exact"],
        root_unevaluated=root["unevaluated_actions"],
        root_classes=classes,
        root_fallback_missing_mass={
            a: info["missing_probability"]
            for a, info in root["actions"].items()
            if classes[a] == "fallback"
        },
        root_decision=root_decision,
        oracle_optimal=sorted((a for a, v in oracle.items() if v >= best - TOLERANCE), key=str),
        arms={arm: row_outcome(row[arm]) for arm in ("control", "root_only", "candidate")},
        internal=dict(internal),
    )


def summarize(cases):
    done = [c for c in cases if c["status"] == "complete"]
    root_kinds = Counter(
        "mixed"
        if len(set(c["root_classes"].values())) > 1
        else next(iter(set(c["root_classes"].values())))
        for c in done
    )
    mixed_root = [c for c in done if c["root_decision"]["mixed"]]
    by_horizon = {}
    for c in done:
        h = by_horizon.setdefault(
            str(c["horizon"]), Counter(cases=0, mixed_root_decisions=0, mixed_root_actions=0)
        )
        h["cases"] += 1
        h["mixed_root_decisions"] += c["root_decision"]["mixed"]
        h["mixed_root_actions"] += len(set(c["root_classes"].values())) > 1
    internal = Counter()
    for c in done:
        internal.update(c["internal"])
    errors = {
        arm: [c["file"] for c in done if c["arms"][arm]["strict_error"]]
        for arm in ("control", "root_only", "candidate")
    }
    margins = [c["root_decision"]["margin"] for c in mixed_root if c["root_decision"]["margin"]]
    return dict(
        cases=len(cases),
        complete=len(done),
        root_action_estimators=dict(root_kinds),
        root_decisions_mixing_estimators=len(mixed_root),
        mixed_root_decision_cases=[
            dict(
                file=c["file"],
                horizon=c["horizon"],
                belief_p=c["belief_p"],
                classes=c["root_classes"],
                decision=c["root_decision"],
                oracle_optimal=c["oracle_optimal"],
                candidate_loss=c["arms"]["candidate"]["loss"],
            )
            for c in mixed_root
        ],
        smallest_mixed_root_margin=min(margins) if margins else None,
        by_horizon={k: dict(v) for k, v in sorted(by_horizon.items(), key=lambda kv: int(kv[0]))},
        internal_histories=dict(internal),
        strict_errors=errors,
        candidate_errors_with_mixed_root_decision=sorted(
            set(errors["candidate"]) & {c["file"] for c in mixed_root}
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path, nargs="+", help="Completed panel directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for panel in args.panels:
        files = sorted(panel.glob("case-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        if not files:
            raise FileNotFoundError(f"No case files in {panel}")
        cases = [audit_case(f) for f in files]
        report[str(panel)] = dict(summary=summarize(cases), cases=cases)
        s = report[str(panel)]["summary"]
        print(
            f"{panel.name}: {s['complete']}/{s['cases']} complete; root estimators "
            f"{s['root_action_estimators']}; mixed root decisions "
            f"{s['root_decisions_mixing_estimators']}; candidate errors {s['strict_errors']['candidate']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
