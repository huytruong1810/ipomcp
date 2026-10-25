"""Execute a study: supervised trials, atomic checkpoints, verifiable completion.

``run_study(spec, output)``:

1. writes ``manifest.json`` with the specification and the hashes of every
   source file; an existing output directory is resumed only if both match;
2. runs each pending trial in its own supervised process (wall-time and
   process-tree memory limits, source lock verified before and after);
3. validates each returned panel (every step 0..steps present, finite returns)
   and writes it atomically to ``trials/<id>.csv``;
4. records every attempt, failed or not, under ``attempts/<run>/``;
5. when all trials are complete, writes ``episodes.csv`` and a completion marker
   binding the trial count, step count and the CSV's SHA-256.

A failed trial never becomes data; completed trials stay checkpointed so a
rerun only executes what is missing.

Command line: ``python -m ipomdp.experiments.runner study.toml --output DIR``.
"""

import argparse
import hashlib
import json
import logging
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import StudySpec
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes

logger = logging.getLogger(__name__)


def run_study(spec: StudySpec, output, workers=None, episode=run_episode) -> pd.DataFrame:
    """Run (or resume) every trial of ``spec`` into ``output``; return all rows.

    ``episode`` is the per-trial function (module-level, picklable); tests
    substitute it to inject failures.
    """
    root = Path(output)
    lock = source_hashes(REPOSITORY_ROOT)
    manifest = {"study": spec.to_dict(), "source_sha256": lock}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError(
                "Existing output was produced by a different study or source; use a new directory"
            )
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty and has no manifest")
    (root / "trials").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    frames, pending = [], []
    for trial in range(spec.trials):
        path = root / "trials" / f"{trial:06}.csv"
        if path.exists():
            frame = _read(path)
            if not valid_panel(frame, [trial], spec.steps):
                raise ValueError(f"Invalid trial checkpoint: {path}")
            frames.append(frame)
        else:
            pending.append(trial)

    failures = []
    attempt = root / "attempts" / uuid4().hex
    jobs = {trial: partial(episode, spec, trial) for trial in pending}
    for trial, outcome in supervise_jobs(
        jobs,
        workers=min(workers or spec.workers, max(len(pending), 1)),
        timeout_seconds=spec.trial_timeout_seconds,
        max_rss_mb=spec.max_trial_rss_mb,
        log_directory=attempt,
        source_lock=(REPOSITORY_ROOT, lock),
    ):
        rows = outcome.pop("rows", None)
        if outcome["status"] == "complete":
            frame = pd.DataFrame(rows)
            if valid_panel(frame, [trial], spec.steps):
                _write_atomic(frame, root / "trials" / f"{trial:06}.csv")
                frames.append(frame)
            else:
                outcome.update(status="failed", reason="incomplete step panel")
        (attempt / f"trial-{trial}.json").write_text(json.dumps(outcome, indent=2, default=str))
        if outcome["status"] != "complete":
            failures.append((trial, outcome["reason"]))
            logger.error("Trial %s failed: %s", trial, outcome["reason"])
    if failures:
        raise RuntimeError(
            f"{len(failures)} trials failed; completed trials are checkpointed: {failures}"
        )
    episodes = pd.concat(frames, ignore_index=True).sort_values(["trial", "step"])
    path = root / "episodes.csv"
    _write_atomic(episodes, path)
    marker = dict(
        trials=spec.trials, steps=spec.steps, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (root / "episodes.complete.json").write_text(json.dumps(marker))
    return episodes


def is_complete(output, spec: StudySpec) -> bool:
    """True when ``episodes.csv`` is a full panel whose bytes match the completion marker."""
    root = Path(output)
    path, marker = root / "episodes.csv", root / "episodes.complete.json"
    if not path.is_file() or not marker.is_file():
        return False
    expected = dict(
        trials=spec.trials, steps=spec.steps, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return json.loads(marker.read_text()) == expected and valid_panel(
        _read(path), range(spec.trials), spec.steps
    )


def valid_panel(frame, trials, steps) -> bool:
    """Every (trial, step) present exactly once, finite returns, known statuses."""
    if not {"trial", "step", "cum_reward_i", "cum_reward_j", "status"} <= set(frame.columns):
        return False
    expected = pd.MultiIndex.from_product([list(trials), range(steps + 1)])
    actual = pd.MultiIndex.from_frame(frame[["trial", "step"]])
    return (
        actual.is_unique
        and set(actual) == set(expected)
        and np.isfinite(frame[["cum_reward_i", "cum_reward_j"]].to_numpy(dtype=float)).all()
        and frame["status"].isin(["Active", "Terminal"]).all()
    )


def _read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _write_atomic(frame, path):
    """Replace within the same directory so readers never see a partial file."""
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", type=Path, help="TOML study specification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    frame = run_study(StudySpec.from_toml(args.study), args.output, args.workers)
    print(f"{frame['trial'].nunique()} trials, {len(frame)} rows -> {args.output}")


if __name__ == "__main__":
    main()
