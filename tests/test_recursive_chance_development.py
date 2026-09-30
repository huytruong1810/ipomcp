"""Frozen launcher rejects provenance changes without running experiments."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recursive_chance_launcher",
    Path(__file__).resolve().parents[1] / "benchmarks/run_recursive_chance_development.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def fixture_contract(root):
    (root / "src").mkdir()
    (root / "benchmarks").mkdir()
    (root / "src/example.py").write_text("# frozen\n")
    (root / "benchmarks/run_recursive_chance_development.py").write_text("# launcher\n")
    row = {"seed": 1}
    raw = root / "raw.json"
    raw.write_text(json.dumps(dict(status="complete", rows=[dict(control=row, candidate=row)])))
    source = {"src/example.py": launcher.sha(root / "src/example.py")}
    panel = root / "panel.json"
    panel.write_text(
        json.dumps(
            dict(
                source_sha256=source,
                cases=[dict(file="raw.json", sha256=launcher.sha(raw), row=row, root_only_row=row)],
            )
        )
    )
    master = root / "master.json"
    master.write_text(
        json.dumps(
            dict(
                source_sha256=source,
                total_triples=1,
                historical_contracts={},
                launcher_sha256=launcher.sha(
                    root / "benchmarks/run_recursive_chance_development.py"
                ),
                panels=[dict(contract="panel.json", sha256=launcher.sha(panel), triples=1)],
            )
        )
    )
    return master


def test_preflight_only_reads(tmp_path):
    master = fixture_contract(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert launcher.preflight(tmp_path, master)["total_triples"] == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "file",
    [
        "src/example.py",
        "src/new.py",
        "raw.json",
        "panel.json",
        "benchmarks/run_recursive_chance_development.py",
    ],
)
def test_changed_evidence_rejected(tmp_path, file):
    master = fixture_contract(tmp_path)
    with (tmp_path / file).open("a") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError):
        launcher.preflight(tmp_path, master)


@pytest.mark.parametrize("existing", ["out", "out.freeze.json"])
def test_existing_attempt_refused(tmp_path, monkeypatch, existing):
    target = tmp_path / existing
    target.write_text("preserve me")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "preflight", lambda: {"panels": [{"out": "out"}]})
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", "--panel", "0"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not start a process")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(FileExistsError):
        launcher.main()
    assert target.read_text() == "preserve me"
