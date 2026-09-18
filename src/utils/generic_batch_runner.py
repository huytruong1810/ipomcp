import concurrent.futures
import gc
import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import ExperimentConfig
from core.logger import get_logger
from core.pomdp_model import POMDPModel, State
from core.telemetry import SystemMonitor
from solvers.planner import Planner

logger = get_logger("BatchRunner")


def is_batch_complete(csv_path: str, expected_trials: int, expected_steps: int) -> bool:
    """Accept only a complete rectangular trial-by-step panel and matching digest.

    Counting trial identifiers alone accepts truncated episodes. The completion
    marker is written last, after atomic CSV replacement, and binds the actual
    bytes. A marker is evidence of completion, not of matching experimental design;
    run_batch additionally checks the run manifest before resuming trial files.
    """
    path = Path(csv_path)
    marker = path.with_suffix(".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata != dict(
            trials=expected_trials,
            steps=expected_steps,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ):
            return False
        return _valid_panel(
            pd.read_csv(path, keep_default_na=False, na_values=[""]),
            range(expected_trials),
            expected_steps,
        )
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False


def _valid_panel(df, trials, steps):
    expected = pd.MultiIndex.from_product([trials, range(steps + 1)], names=["trial", "step"])
    if not {"trial", "step", "cum_reward_i", "cum_reward_j", "status"} <= set(df.columns):
        return False
    actual = pd.MultiIndex.from_frame(df[["trial", "step"]])
    return (
        actual.is_unique
        and len(actual) == len(expected)
        and set(actual) == set(expected)
        and np.isfinite(df[["cum_reward_i", "cum_reward_j"]].to_numpy()).all()
        and df["status"].isin(["Active", "Terminal"]).all()
    )


def _write_csv_atomic(df, path):
    """Replace within the same directory so readers never see a partial CSV."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _get_belief_support(planner: Planner) -> int:
    """Number of distinct weighted interactive hypotheses, not search trajectories."""
    return len(planner.belief.mass) if getattr(planner, "belief", None) is not None else 0


def _get_opponent_level_distribution(planner: Any, opponent_id: str) -> Dict[str, float]:
    belief = getattr(planner, "belief", None)
    if belief is None:
        return {}
    masses = {level: 0.0 for level in range(planner.key.level)}
    for atom, weight in belief.mass:
        frame = atom.opponent.frame
        if frame.agent_id == opponent_id:
            masses[frame.level] += weight
    return {f"prob_l{level}_{opponent_id}": mass for level, mass in sorted(masses.items())}


def _extract_recursive(
    node,
    curr_agent,
    curr_path_id,
    curr_weight,
    depth,
    ids,
    labels,
    parents,
    values,
    hover_texts,
    levels,
    agent_names,
):
    """Integrate every weighted private belief, preserving its conditional mass."""

    def descend(weighted_beliefs, agent, path, mass, remaining):
        if remaining <= 0 or mass <= 0:
            return
        groups = {}
        for belief, parent_mass in weighted_beliefs:
            if belief is None:
                continue
            for atom, weight in belief.mass:
                model = atom.opponent
                group = groups.setdefault((model.frame.agent_id, model.frame.level), {})
                group[model.belief] = group.get(model.belief, 0.0) + parent_mass * weight
        for (opponent, level), children in sorted(groups.items()):
            branch_mass = sum(children.values())
            child_id = f"{path}/{opponent}_L{level}"
            ids.append(child_id)
            labels.append(f"{opponent} (L{level})")
            parents.append(path)
            values.append(branch_mass)
            levels.append(level)
            agent_names.append(opponent)
            hover_texts.append(
                f"Modeled by: {agent}<br>Conditional mass: {branch_mass / mass:.1%}<br>Joint mass: {branch_mass:.6f}"
            )
            if level > 0:
                descend(list(children.items()), opponent, child_id, branch_mass, remaining - 1)

    descend([(node, curr_weight)], curr_agent, curr_path_id, curr_weight, depth)


def extract_nested_belief_hierarchy(
    planner: Any, agent_id: str = "i", agent_level: int = 0, max_depth: int = 8
) -> Dict[str, Any]:
    """
    Recursively extracts the full multi-level nested mental model belief tree for an I-POMDP agent.

    Returns structured tabular data compatible with Plotly Sunburst, Icicle, and Treemap:
    {
        "agent_id": str,
        "agent_level": int,
        "ids": List[str],
        "labels": List[str],
        "parents": List[str],
        "values": List[float],
        "hover_texts": List[str],
        "levels": List[int],
        "agent_names": List[str]
    }
    """
    if getattr(planner, "belief", None) is None:
        return {}

    lvl = getattr(planner.key, "level", agent_level) if hasattr(planner, "key") else agent_level
    root_id = f"{agent_id}_L{lvl}"
    root_label = f"{agent_id} (Level-{lvl})"
    hover = (
        f"<b>Protagonist Agent {agent_id}</b><br>Reasoning Level: {lvl}<br>Total Belief Mass: 100%"
    )

    ids = [root_id]
    labels = [root_label]
    parents = [""]
    values = [1.0]
    hover_texts = [hover]
    levels = [lvl]
    agent_names = [agent_id]

    _extract_recursive(
        node=planner.belief,
        curr_agent=agent_id,
        curr_path_id=root_id,
        curr_weight=1.0,
        depth=max_depth,
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        hover_texts=hover_texts,
        levels=levels,
        agent_names=agent_names,
    )

    return {
        "agent_id": agent_id,
        "agent_level": lvl,
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values,
        "hover_texts": hover_texts,
        "levels": levels,
        "agent_names": agent_names,
    }


class GenericBatchRunner(ABC):
    def __init__(self, config: ExperimentConfig, log_dir: str = None):
        self.config = config
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

            # Attach file logger to record execution traces alongside artifacts
            global logger
            logger = get_logger("BatchRunner", log_file=os.path.join(self.log_dir, "execution.log"))

    @abstractmethod
    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        pass

    def _get_custom_metrics(
        self,
        state: State,
        next_state: State,
        env: POMDPModel,
        planner_i: Planner,
        planner_j: Planner,
    ) -> Dict[str, Any]:
        return {}

    def _run_single_trial_parallel(self, trial_id: int) -> List[Dict[str, Any]]:
        return self._run_episode(trial_id)

    def _run_episode(self, trial_id: int, snapshots=None) -> List[Dict[str, Any]]:
        # Common Random Numbers (CRN): Dedicated isolated RNGs for transition and per-agent observations
        env_rng_trans = random.Random(trial_id * 10000 + 42)
        env_rng_obs_i = random.Random(trial_id * 10000 + 1042)
        env_rng_obs_j = random.Random(trial_id * 10000 + 2042)

        # Seed global RNG for planners
        planner_seed = trial_id * 10000 + 1337
        random.seed(planner_seed)
        np.random.seed(planner_seed)

        env, planner_i, planner_j, true_state = self._setup_domain()

        true_state = env.get_initial_state(rng=env_rng_trans)

        trial_records = []
        cum_reward_i = 0.0
        cum_reward_j = 0.0
        is_terminal = env.is_terminal(true_state)

        # Step 0: Initial condition prior to taking any action
        initial_stats_i = (
            planner_i.get_detailed_stats() if hasattr(planner_i, "get_detailed_stats") else {}
        )
        initial_stats_j = (
            planner_j.get_detailed_stats() if hasattr(planner_j, "get_detailed_stats") else {}
        )
        n_i = _get_belief_support(planner_i)
        n_j = _get_belief_support(planner_j)

        init_record = {
            "trial": trial_id,
            "step": 0,
            "action_i": None,
            "action_j": None,
            "obs_i": None,
            "obs_j": None,
            "reward_i": 0.0,
            "cum_reward_i": 0.0,
            "planning_time_i": 0.0,
            "reward_j": 0.0,
            "cum_reward_j": 0.0,
            "belief_support_i": n_i,
            "belief_support_j": n_j,
            "status": "Active",
            "action_values_i": initial_stats_i.get("action_values"),
            "action_values_j": initial_stats_j.get("action_values"),
        }
        init_custom = self._get_custom_metrics(true_state, true_state, env, planner_i, planner_j)
        init_record.update(init_custom)
        init_record.update(_get_opponent_level_distribution(planner_i, "j"))
        trial_records.append(init_record)
        last_custom_metrics = init_custom
        if snapshots is not None:
            snapshots[0] = extract_nested_belief_hierarchy(planner_i, "i")

        # Decision steps: t = 1 ... max_steps
        for t in range(1, self.config.max_steps + 1):
            if not is_terminal:
                start_time_i = time.perf_counter()
                a_i = planner_i.get_action()
                plan_time_i = time.perf_counter() - start_time_i

                a_j = planner_j.get_action()
                joint_action = {"i": a_i, "j": a_j}

                next_state = env.sample_transition(true_state, joint_action, rng=env_rng_trans)
                o_i = env.sample_observation(next_state, joint_action, "i", rng=env_rng_obs_i)
                o_j = env.sample_observation(next_state, joint_action, "j", rng=env_rng_obs_j)

                r_i = env.get_reward(true_state, joint_action, next_state, "i")
                r_j = env.get_reward(true_state, joint_action, next_state, "j")

                cum_reward_i += r_i
                cum_reward_j += r_j
                is_terminal = env.is_terminal(next_state)

                stats_i = planner_i.get_detailed_stats()
                stats_j = planner_j.get_detailed_stats()

                n_i = _get_belief_support(planner_i)
                n_j = _get_belief_support(planner_j)

                record = {
                    "trial": trial_id,
                    "step": t,
                    "action_i": str(a_i),
                    "action_j": str(a_j),
                    "obs_i": str(o_i),
                    "obs_j": str(o_j),
                    "reward_i": r_i,
                    "cum_reward_i": cum_reward_i,
                    "planning_time_i": plan_time_i,
                    "reward_j": r_j,
                    "cum_reward_j": cum_reward_j,
                    "belief_support_i": n_i,
                    "belief_support_j": n_j,
                    "status": "Terminal" if is_terminal else "Active",
                    "action_values_i": stats_i.get("action_values"),
                    "action_values_j": stats_j.get("action_values"),
                    "process_rss_mb": round(
                        SystemMonitor.get_process_memory()[0] / (1024.0 * 1024.0), 2
                    ),
                }

                if self.log_dir and self.config.verbose:
                    step_dir = os.path.join(self.log_dir, f"trial_{trial_id}", f"step_{t}")
                    os.makedirs(step_dir, exist_ok=True)

                    with open(os.path.join(step_dir, "stats_i.json"), "w") as f:
                        json.dump(stats_i, f, indent=2)
                    with open(os.path.join(step_dir, "stats_j.json"), "w") as f:
                        json.dump(stats_j, f, indent=2)

                    if self.config.export_trees:
                        planner_i.visualize(os.path.join(step_dir, "tree_i"), t)
                        planner_j.visualize(os.path.join(step_dir, "tree_j"), t)

                if not is_terminal:
                    planner_i.update_root(a_i, o_i)
                    planner_j.update_root(a_j, o_j)
                last_custom_metrics = self._get_custom_metrics(
                    true_state, next_state, env, planner_i, planner_j
                )
                record.update(last_custom_metrics)
                record.update(_get_opponent_level_distribution(planner_i, "j"))
                record["belief_support_i"] = _get_belief_support(planner_i)
                record["belief_support_j"] = _get_belief_support(planner_j)
                trial_records.append(record)
                true_state = next_state
                if snapshots is not None:
                    snapshots[t] = extract_nested_belief_hierarchy(planner_i, "i")
            else:
                record = {
                    "trial": trial_id,
                    "step": t,
                    "reward_i": 0.0,
                    "cum_reward_i": cum_reward_i,
                    "planning_time_i": 0.0,
                    "reward_j": 0.0,
                    "cum_reward_j": cum_reward_j,
                    "belief_support_i": None,
                    "belief_support_j": None,
                    "status": "Terminal",
                    "action_values_i": None,
                    "action_values_j": None,
                }
                record.update(last_custom_metrics)
                trial_records.append(record)
                if snapshots is not None:
                    snapshots[t] = snapshots[t - 1]

        # Nested fields have exactly one wire representation: JSON, not Python repr.
        for record in trial_records:
            for key in ("action_values_i", "action_values_j"):
                record[key] = json.dumps(record[key], allow_nan=False)
        del planner_i, planner_j, env
        gc.collect()
        return trial_records

    def run_batch(self, max_workers: Optional[int] = None) -> pd.DataFrame:
        """Execute independent trials and checkpoint each successful result.

        Resume is allowed only with identical source and runner configuration.
        Worker failures are collected, completed trials stay available, and no
        completion marker is written for an incomplete experiment. Worker recycling
        bounds allocator retention across episodes, not the live memory of one tree.
        """
        source_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in sorted(source_root.rglob("*.py")):
            digest.update(str(source.relative_to(source_root)).encode())
            digest.update(source.read_bytes())
        manifest = dict(
            source_sha256=digest.hexdigest(),
            runner=type(self).__module__ + "." + type(self).__qualname__,
            experiment=asdict(self.config),
            parameters={k: v for k, v in vars(self).items() if k not in {"config", "log_dir"}},
        )
        # Canonical JSON converts integer mapping keys consistently before comparison.
        manifest = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
        root = Path(self.log_dir) if self.log_dir else None
        if root:
            path = root / "run_manifest.json"
            if not path.exists() and (root / "batch_results.csv").exists():
                raise ValueError(
                    "Existing results lack a matching manifest; use a new output directory."
                )
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Resume configuration or source differs from the saved manifest.")
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (root / "trials").mkdir(exist_ok=True)
        frames, pending = [], []
        for trial in range(self.config.n_trials):
            path = root / "trials" / f"{trial:06}.csv" if root else None
            if path and path.exists():
                frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
                if not _valid_panel(frame, [trial], self.config.max_steps):
                    raise ValueError(f"Invalid trial checkpoint: {path}")
                frames.append(frame)
            else:
                pending.append(trial)
        if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
            raise ValueError("max_workers must be a positive integer")
        workers = min(max_workers or (os.cpu_count() or 1), len(pending) or 1)
        failures = []
        if pending and workers == 1:
            for trial in pending:
                frame = pd.DataFrame(self._run_single_trial_parallel(trial))
                if not _valid_panel(frame, [trial], self.config.max_steps):
                    raise ValueError(f"Trial {trial} returned an incomplete panel.")
                if root:
                    _write_csv_atomic(frame, root / "trials" / f"{trial:06}.csv")
                frames.append(frame)
        elif pending:
            for start in range(0, len(pending), workers * 20):
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self._run_single_trial_parallel, trial): trial
                        for trial in pending[start : start + workers * 20]
                    }
                    for future in concurrent.futures.as_completed(futures):
                        trial = futures[future]
                        try:
                            frame = pd.DataFrame(future.result())
                            if not _valid_panel(frame, [trial], self.config.max_steps):
                                raise ValueError(f"Trial {trial} returned an incomplete panel.")
                            if root:
                                _write_csv_atomic(frame, root / "trials" / f"{trial:06}.csv")
                            frames.append(frame)
                            logger.info("Completed %s/%s trials", len(frames), self.config.n_trials)
                        except Exception as exc:
                            # Aggregate worker failures deliberately; never reinterpret them as data.
                            failures.append((trial, str(exc)))
                            logger.exception("Trial %s failed", trial)
        df = (
            pd.concat(frames, ignore_index=True).sort_values(["trial", "step"])
            if frames
            else pd.DataFrame()
        )
        if failures:
            if root and not df.empty:
                _write_csv_atomic(df, root / "batch_results_partial.csv")
            raise RuntimeError(f"Batch failed; completed trials are checkpointed: {failures}")
        if root:
            path = root / "batch_results.csv"
            _write_csv_atomic(df, path)
            marker = dict(
                trials=self.config.n_trials,
                steps=self.config.max_steps,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            temporary = path.with_suffix(".complete.tmp")
            temporary.write_text(json.dumps(marker), encoding="utf-8")
            temporary.replace(path.with_suffix(".complete.json"))
        return df

    def run_single_trial_with_snapshots(self, trial_id: int = 0):
        """Run the batch episode semantics with deterministic, RNG-neutral snapshots."""
        snapshots = {}
        records = self._run_episode(trial_id, snapshots)
        if self.log_dir:
            path = Path(self.log_dir) / f"nested_belief_snapshots_trial_{trial_id}.json"
            path.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        return pd.DataFrame(records), snapshots
