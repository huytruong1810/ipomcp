"""Three-arm recursive chance development pilot; not a qualification gate."""

import argparse
import json
import subprocess
import time
from functools import partial
from pathlib import Path

from examples.experiments.root_chance_experiment import replay, sha
from utils.process_supervisor import supervise_jobs


def triple(original, root_only_original):
    control, (control_tree, control_seconds) = replay(original, False)
    assert control == original, "Historical control replay changed"
    root_only, (root_tree, root_seconds) = replay(original, True)
    assert root_only == root_only_original, "Historical root-only replay changed"
    candidate, (candidate_tree, candidate_seconds) = replay(original, False, recursive=True)
    assert control_tree == root_tree == candidate_tree, "Search trajectory changed"
    assert candidate["search_estimated_q"] == control["estimated_q"]
    for key in ("oracle_q", "opponent_model", "horizon", "budget", "seed", "belief_p", "gamma"):
        assert candidate[key] == control[key], f"Task semantics changed: {key}"
    return [
        dict(
            control=control,
            root_only=root_only,
            candidate=candidate,
            control_tree_sha256=control_tree,
            root_only_tree_sha256=root_tree,
            candidate_tree_sha256=candidate_tree,
            exact_control_replay=True,
            exact_root_only_replay=True,
            identical_search_tree=True,
            control_policy_seconds=control_seconds,
            root_only_policy_seconds=root_seconds,
            candidate_policy_seconds=candidate_seconds,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    raw = args.input.read_bytes()
    contract = json.loads(raw)
    for entry in contract["cases"]:
        data = (root / entry["file"]).read_bytes()
        assert sha(data) == entry["sha256"], "Preserved input changed"
        saved = json.loads(data)["rows"][0]
        assert saved["control"] == entry["row"]
        assert saved["candidate"] == entry["root_only_row"]
    source = {
        str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / "src").rglob("*.py"))
    }
    assert source == contract["source_sha256"], (
        "Recursive pilot source changed; review before rerunning"
    )
    if args.check_only:
        print(f"Verified {len(contract['cases'])} triples; no experiment started")
        return
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
    jobs = {
        i: partial(triple, entry["row"], entry["root_only_row"])
        for i, entry in enumerate(contract["cases"])
    }
    try:
        for i, result in supervise_jobs(
            jobs,
            workers=1,
            timeout_seconds=600,
            max_rss_mb=2048,
            log_directory=args.output / "logs",
        ):
            (args.output / f"case-{i}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            statuses.append(
                dict(id=i, status=result["status"], wall_seconds=result["wall_seconds"])
            )
            print(i, result["status"], flush=True)
            if result["status"] != "complete":
                raise RuntimeError("Triple failed; preserve outputs and report before extending")
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
