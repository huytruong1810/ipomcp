"""Run one source-frozen validation panel; intended for Antigravity execution."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=int, choices=range(6), required=True)
    args = parser.parse_args()
    path = ROOT / "benchmarks/tail2_validation_20260927.json"
    raw = path.read_bytes()
    frozen = json.loads(raw)
    actual = {
        str(p.relative_to(ROOT / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    if actual != frozen["source_sha256"]:
        raise RuntimeError("Source differs from frozen validation; do not run or retune this gate")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != frozen["runner_sha256"]:
        raise RuntimeError("Validation runner differs from the frozen version")
    panel = frozen["panels"][args.panel]
    directory = ROOT / panel["out"]
    if directory.exists():
        raise FileExistsError("Preserve the existing attempt; do not overwrite or silently resume")
    sidecar = directory.with_name(directory.name + ".freeze.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("x") as stream:
        json.dump(
            {
                "freeze_sha256": hashlib.sha256(raw).hexdigest(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--short"], cwd=ROOT, text=True
                ),
                "panel": args.panel,
                "contract": frozen,
            },
            stream,
            indent=2,
        )
    from examples.experiments.planner_oracle_experiment import run_oracle_comparison

    run_oracle_comparison(directory, **panel["settings"])


if __name__ == "__main__":
    main()
