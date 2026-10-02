"""Source-frozen broader observed development comparison; Antigravity executes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/root_chance_development_20260928.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(root=ROOT, contract_path=CONTRACT):
    frozen = json.loads(contract_path.read_text())
    if sha(root / "benchmarks/run_root_chance_development.py") != frozen["launcher_sha256"]:
        raise RuntimeError("Launcher changed; preserve the frozen comparison")
    source = {str(p.relative_to(root)): sha(p) for p in sorted((root / "src").rglob("*.py"))}
    if source != frozen["source_sha256"]:
        raise RuntimeError("Source changed; do not refresh frozen hashes")
    count = 0
    for panel in frozen["panels"]:
        path = root / panel["contract"]
        if sha(path) != panel["sha256"]:
            raise RuntimeError("Panel contract changed")
        data = json.loads(path.read_text())
        if data["source_sha256"] != source or len(data["cases"]) != panel["pairs"]:
            raise RuntimeError("Panel configuration differs")
        for entry in data["cases"]:
            raw = root / entry["file"]
            if sha(raw) != entry["sha256"]:
                raise RuntimeError("Historical input changed")
            saved = json.loads(raw.read_text())
            if saved["status"] != "complete" or saved["rows"] != [entry["row"]]:
                raise RuntimeError("Historical row differs")
        count += len(data["cases"])
    if count != frozen["total_pairs"]:
        raise RuntimeError("Incomplete coverage")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--panel", type=int, choices=range(12))
    args = parser.parse_args()
    frozen = preflight()
    if args.check_only:
        print(f"Verified {frozen['total_pairs']} pairs; no experiment started", flush=True)
        return
    panel = frozen["panels"][args.panel]
    directory = ROOT / panel["out"]
    sidecar = directory.with_name(directory.name + ".freeze.json")
    if directory.exists() or sidecar.exists():
        raise FileExistsError("Preserve existing attempt; no overwrite or automatic resume")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("x") as stream:
        json.dump(
            dict(
                contract=frozen,
                contract_sha256=sha(CONTRACT),
                panel=args.panel,
                git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                git_status=subprocess.check_output(
                    ["git", "status", "--short"], cwd=ROOT, text=True
                ),
            ),
            stream,
            indent=2,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.experiments.root_chance_experiment",
            "--input",
            str(ROOT / panel["contract"]),
            "--output",
            str(directory),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
