"""Paired development replay of frozen saved cases; never a qualification gate."""

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

from examples.experiments.planner_oracle_experiment import evaluate_case
from ipomdp.canonical import atom_row, stable_value
from solvers.i_pomcp import IPOMCPPlanner
from utils.process_supervisor import supervise_jobs
from utils.source_lock import source_hashes


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tree_digest(planner):
    """All search statistics plus ordered reservoir contents, across fresh banks."""
    digest = hashlib.sha256()

    def visit(node):
        # Serialize node-local statistics only; recurse without truncation.
        state = {
            k: v
            for k, v in vars(node).items()
            if k not in ("children", "belief_particles", "_parent_ref")
        }
        state["particles"] = [atom_row(p, 1.0) for p in node.belief_particles]
        digest.update(json.dumps(stable_value(state), sort_keys=True, allow_nan=False).encode())
        for action, children in sorted(node.children.items()):
            for obs, child in sorted(children.items()):
                digest.update(json.dumps([action, obs]).encode())
                visit(child)

    visit(planner.root)
    return digest.hexdigest()


def replay(original, enabled, *, recursive=False):
    captures = []
    method = IPOMCPPlanner.policy_for

    def capture(self, model, modeled=False):
        start = time.monotonic()
        policy = method(self, model, modeled)
        seconds = time.monotonic() - start
        if self.key.level == 2 and not modeled:
            captures.append((tree_digest(self), seconds))
        return policy

    op = original["opponent_model"]
    cfg = op["config"]["mcts"]
    with patch.object(IPOMCPPlanner, "policy_for", capture):
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
            exact_tail_depth=original["planner_config"]["mcts"].get("exact_tail_depth", 1),
            exact_root_chance=enabled,
            recursive_chance=recursive,
            level=2,
            opponent_depth=cfg["max_depth"],
            opponent_belief=op["initial_belief_p"],
            opponent_budget=cfg["n_sims"],
            opponent_backup=cfg["backup"],
            opponent_exact_final_step=cfg["exact_final_step"],
            opponent_exploration=op["exploration"]["strategy"],
            opponent_exploration_const=op["exploration"]["c"],
        )[0]
    assert len(captures) == 1
    return row, captures[0]


def paired(original):
    control, (control_tree, control_seconds) = replay(original, False)
    assert control == original, "Historical control row changed; stop interpretation"
    candidate, (candidate_tree, candidate_seconds) = replay(original, True)
    assert control_tree == candidate_tree, "Search trajectory changed"
    assert candidate["search_estimated_q"] == control["estimated_q"]
    for key in ("oracle_q", "opponent_model", "horizon", "budget", "seed", "belief_p", "gamma"):
        assert candidate[key] == control[key], f"Task semantics changed: {key}"
    return [
        dict(
            control=control,
            candidate=candidate,
            tree_sha256=control_tree,
            exact_control_replay=True,
            identical_search_tree=True,
            control_policy_seconds=control_seconds,
            candidate_policy_seconds=candidate_seconds,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    raw = args.input.read_bytes()
    contract = json.loads(raw)
    for entry in contract["cases"]:
        data = (root / entry["file"]).read_bytes()
        assert sha(data) == entry["sha256"], "Preserved input changed"
        assert json.loads(data)["rows"][0] == entry["row"]
    source = source_hashes(root)
    assert source == contract["source_sha256"], "Pilot source changed; review before rerunning"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        contract=contract,
        input_sha256=sha(raw),
        source_sha256=source,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        workers=1,
        timeout_seconds=600,
        max_rss_mb=2048,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start, unix_start = time.monotonic(), time.time()
    statuses = []
    jobs = {i: partial(paired, entry["row"]) for i, entry in enumerate(contract["cases"])}
    try:
        for i, result in supervise_jobs(
            jobs,
            workers=1,
            timeout_seconds=600,
            max_rss_mb=2048,
            log_directory=args.output / "logs",
            source_lock=(root, contract["source_sha256"]),
        ):
            (args.output / f"case-{i}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            statuses.append(
                dict(id=i, status=result["status"], wall_seconds=result["wall_seconds"])
            )
            print(i, result["status"], flush=True)
            if result["status"] != "complete":
                raise RuntimeError("Pair failed; preserve outputs and report before extending")
    finally:
        (args.output / "summary.json").write_text(
            json.dumps(
                dict(
                    results=statuses,
                    elapsed_monotonic_seconds=time.monotonic() - start,
                    started_unix=unix_start,
                    finished_unix=time.time(),
                ),
                indent=2,
            )
        )
    assert len(statuses) == len(jobs)


if __name__ == "__main__":
    main()
