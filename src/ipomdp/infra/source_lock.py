"""Source locks that hold for every trial, not only at launch.

Supervised workers are spawned interpreters that import ``src`` from disk when
each trial starts. A launch-time hash therefore does not bind a multi-hour
panel: an edit made mid-run silently reaches every later trial. A lock is
checked in the worker before and after its job, so a trial either ran entirely
under the declared bytes or fails visibly. Hashes verify bytes; they do not
archive them, so keep the locked tree reachable from version control.
"""

import hashlib
from pathlib import Path


class SourceLockViolation(RuntimeError):
    """The files under a declared lock differ from their recorded hashes."""


def source_hashes(root):
    """SHA256 of every src/**/*.py file, keyed by its path relative to ``root``."""
    root = Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src").rglob("*.py"))
    }


def verify_source_lock(root, expected):
    """Raise with the added, removed and changed paths if the tree differs."""
    actual = source_hashes(root)
    if actual == expected:
        return
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = sorted(p for p in set(actual) & set(expected) if actual[p] != expected[p])
    raise SourceLockViolation(
        f"Source lock violated: added={added} removed={removed} changed={changed}"
    )
