"""A declared source lock must bind every supervised trial, not only launch."""

from functools import partial

import pytest

from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import SourceLockViolation, source_hashes, verify_source_lock


def _success():
    return [{"ok": True}]


def _edit(path):
    with open(path, "a") as stream:
        stream.write("# edited mid-run\n")
    return [{"ok": True}]


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "module.py").write_text("VALUE = 1\n")
    (tmp_path / "src" / "notes.txt").write_text("not source\n")
    return tmp_path


def test_hashes_use_repository_relative_python_paths(tree):
    assert list(source_hashes(tree)) == ["src/pkg/module.py"]


def test_violation_names_added_removed_and_changed_paths(tree):
    lock = source_hashes(tree)
    (tree / "src" / "pkg" / "module.py").write_text("VALUE = 2\n")
    (tree / "src" / "extra.py").write_text("")
    with pytest.raises(
        SourceLockViolation, match=r"added=\['src/extra.py'\].*changed=\['src/pkg/module.py'\]"
    ):
        verify_source_lock(tree, lock)


def test_matching_lock_completes(tree, tmp_path):
    outcomes = dict(
        supervise_jobs(
            {0: _success},
            workers=1,
            timeout_seconds=60,
            max_rss_mb=2048,
            source_lock=(tree, source_hashes(tree)),
        )
    )
    assert outcomes[0]["status"] == "complete"


def test_stale_lock_is_rejected_before_any_worker_starts(tree):
    lock = source_hashes(tree)
    (tree / "src" / "pkg" / "module.py").write_text("VALUE = 2\n")
    with pytest.raises(SourceLockViolation):
        dict(
            supervise_jobs(
                {0: _success},
                workers=1,
                timeout_seconds=60,
                max_rss_mb=2048,
                source_lock=(tree, lock),
            )
        )


def test_edit_during_run_fails_that_trial_and_every_later_one(tree):
    target = tree / "src" / "pkg" / "module.py"
    outcomes = dict(
        supervise_jobs(
            {0: partial(_edit, str(target)), 1: _success},
            workers=1,
            timeout_seconds=60,
            max_rss_mb=2048,
            source_lock=(tree, source_hashes(tree)),
        )
    )
    for identifier in (0, 1):
        assert outcomes[identifier]["status"] == "failed"
        assert outcomes[identifier]["error_type"] == "SourceLockViolation"
        assert "src/pkg/module.py" in outcomes[identifier]["reason"]


def test_empty_lock_is_invalid(tree):
    with pytest.raises(ValueError, match="at least one file"):
        dict(
            supervise_jobs(
                {0: _success},
                workers=1,
                timeout_seconds=60,
                max_rss_mb=2048,
                source_lock=(tree, {}),
            )
        )
