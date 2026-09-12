# Absolute Path: <project_root>/utils/generic_batch_runner.py

import gc
import time
import pandas as pd
import os
import json
import random
import numpy as np
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

from core.pomdp_model import POMDPModel, State
from core.config import ExperimentConfig
from core.logger import get_logger
from core.telemetry import SystemMonitor, MemoryWatchdog
from solvers.planner import Planner

logger = get_logger("BatchRunner")


def is_batch_complete(csv_path: str, expected_trials: int) -> bool:
    """Verifies that a batch results CSV exists and contains the expected number of completed trials."""
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if "trial" not in df.columns:
            return False
        return int(df["trial"].nunique()) >= expected_trials
    except Exception:
        return False


def _get_n_particles(planner: Planner) -> int:
    if hasattr(planner, 'root') and hasattr(planner.root, 'belief_particles'):
        return len(planner.root.belief_particles)
    if hasattr(planner, 'belief'):
        return len(planner.belief)
    return -1


def _get_opponent_level_distribution(planner: Any, opponent_id: str) -> Dict[str, float]:
    """Computes online posterior distribution P(l_j = k | h_i^t) over opponent levels in the active belief."""
    particles = None
    if hasattr(planner, 'root') and hasattr(planner.root, 'belief_particles') and planner.root.belief_particles:
        particles = planner.root.belief_particles
    elif hasattr(planner, 'belief') and planner.belief:
        particles = planner.belief

    max_lvl = 1
    if hasattr(planner, 'key') and hasattr(planner.key, 'level'):
        max_lvl = max(1, planner.key.level)
    elif hasattr(planner, 'level'):
        max_lvl = max(1, planner.level)

    res = {f"prob_l{lvl}_{opponent_id}": 0.0 for lvl in range(max_lvl)}

    if not particles:
        return res
    total = len(particles)
    if total == 0:
        return res
    counts: Dict[int, int] = {}
    for p in particles:
        if hasattr(p, 'models') and opponent_id in p.models:
            frame, _ = p.models[opponent_id]
            lvl = frame.level
            counts[lvl] = counts.get(lvl, 0) + 1
    for lvl, count in counts.items():
        res[f"prob_l{lvl}_{opponent_id}"] = count / total
    return dict(sorted(res.items()))


def _extract_recursive(node: Any, curr_agent: str, curr_path_id: str, curr_weight: float, depth: int,
                       ids: List[str], labels: List[str], parents: List[str], values: List[float],
                       hover_texts: List[str], levels: List[int], agent_names: List[str]):
    """Recursively walks particle tree and accumulates nested path probabilities."""
    if node is None or not hasattr(node, 'belief_particles') or not node.belief_particles or depth <= 0:
        return

    particles = node.belief_particles
    total_p = len(particles)
    if total_p == 0 or curr_weight <= 0:
        return

    sample_p = particles[0]
    if not hasattr(sample_p, 'models'):
        return

    for opp_id in sample_p.models.keys():
        level_groups: Dict[int, List[Any]] = {}
        for p in particles:
            if hasattr(p, 'models') and opp_id in p.models:
                frame, sub_node = p.models[opp_id]
                level_groups.setdefault(frame.level, []).append((frame, sub_node))

        for opp_level, sub_entries in sorted(level_groups.items()):
            sub_count = len(sub_entries)
            local_prob = sub_count / total_p
            branch_weight = curr_weight * local_prob

            child_id = f"{curr_path_id}/{opp_id}_L{opp_level}"
            child_label = f"{opp_id} (L{opp_level})"
            hover = (f"<b>Agent {opp_id} (Level-{opp_level})</b><br>"
                     f"Modeled by: {curr_agent}<br>"
                     f"Conditional Belief: {local_prob:.1%}<br>"
                     f"Joint Mass: {branch_weight:.3f}<br>"
                     f"Particles: {sub_count}/{total_p}")

            ids.append(child_id)
            labels.append(child_label)
            parents.append(curr_path_id)
            values.append(branch_weight)
            hover_texts.append(hover)
            levels.append(opp_level)
            agent_names.append(opp_id)

            if opp_level >= 1:
                representative_sub_node = None
                for _, sub_node in sub_entries:
                    if sub_node is not None and hasattr(sub_node, 'belief_particles') and sub_node.belief_particles:
                        representative_sub_node = sub_node
                        break

                if representative_sub_node is not None:
                    _extract_recursive(
                        node=representative_sub_node,
                        curr_agent=opp_id,
                        curr_path_id=child_id,
                        curr_weight=branch_weight,
                        depth=depth - 1,
                        ids=ids, labels=labels, parents=parents, values=values,
                        hover_texts=hover_texts, levels=levels, agent_names=agent_names
                    )


def extract_nested_belief_hierarchy(planner: Any, agent_id: str = "i", agent_level: int = 0, max_depth: int = 8) -> Dict[str, Any]:
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
    if not hasattr(planner, 'root') or not hasattr(planner.root, 'belief_particles') or not planner.root.belief_particles:
        return {}

    lvl = getattr(planner.key, 'level', agent_level) if hasattr(planner, 'key') else agent_level
    root_id = f"{agent_id}_L{lvl}"
    root_label = f"{agent_id} (Level-{lvl})"
    hover = f"<b>Protagonist Agent {agent_id}</b><br>Reasoning Level: {lvl}<br>Total Belief Mass: 100%"

    ids = [root_id]
    labels = [root_label]
    parents = [""]
    values = [1.0]
    hover_texts = [hover]
    levels = [lvl]
    agent_names = [agent_id]

    _extract_recursive(
        node=planner.root,
        curr_agent=agent_id,
        curr_path_id=root_id,
        curr_weight=1.0,
        depth=max_depth,
        ids=ids, labels=labels, parents=parents, values=values,
        hover_texts=hover_texts, levels=levels, agent_names=agent_names
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
        "agent_names": agent_names
    }


class GenericBatchRunner(ABC):
    def __init__(self, config: ExperimentConfig, log_dir: str = None):
        self.config = config
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.config.save(os.path.join(self.log_dir, "experiment_config.json"))

            # Attach file logger to record execution traces alongside artifacts
            global logger
            logger = get_logger("BatchRunner", log_file=os.path.join(self.log_dir, "execution.log"))

    @abstractmethod
    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        pass

    def _get_custom_metrics(self, state: State, next_state: State, env: POMDPModel,
                            planner_i: Planner, planner_j: Planner) -> Dict[str, Any]:
        return {}

    def _run_single_trial_parallel(self, trial_id: int) -> List[Dict[str, Any]]:
        # Common Random Numbers (CRN): Dedicated isolated RNGs for transition and per-agent observations
        env_rng_trans = random.Random(trial_id * 10000 + 42)
        env_rng_obs_i = random.Random(trial_id * 10000 + 1042)
        env_rng_obs_j = random.Random(trial_id * 10000 + 2042)

        # Seed global RNG for planners
        planner_seed = trial_id * 10000 + 1337
        random.seed(planner_seed)
        np.random.seed(planner_seed)

        env, planner_i, planner_j, true_state = self._setup_domain()

        # Sample initial true state using CRN generator if supported
        if hasattr(env, 'get_initial_state'):
            try:
                true_state = env.get_initial_state(rng=env_rng_trans)
            except TypeError:
                true_state = env.get_initial_state()

        trial_records = []
        cum_reward_i = 0.0
        cum_reward_j = 0.0
        is_terminal = False

        # Step 0: Initial condition prior to taking any action
        initial_stats_i = planner_i.get_detailed_stats() if hasattr(planner_i, 'get_detailed_stats') else {}
        initial_stats_j = planner_j.get_detailed_stats() if hasattr(planner_j, 'get_detailed_stats') else {}
        n_i = _get_n_particles(planner_i)
        n_j = _get_n_particles(planner_j)
        initial_n_i = n_i
        initial_n_j = n_j

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
            "n_particles_i": n_i,
            "n_particles_j": n_j,
            "status": "Active",
            "action_values_i": initial_stats_i.get("action_values"),
            "action_values_j": initial_stats_j.get("action_values")
        }
        init_custom = self._get_custom_metrics(true_state, true_state, env, planner_i, planner_j)
        init_record.update(init_custom)
        init_record.update(_get_opponent_level_distribution(planner_i, 'j'))
        trial_records.append(init_record)

        # Decision steps: t = 1 ... max_steps
        for t in range(1, self.config.max_steps + 1):
            if not is_terminal:
                start_time_i = time.perf_counter()
                a_i = planner_i.get_action()
                plan_time_i = time.perf_counter() - start_time_i

                a_j = planner_j.get_action()
                joint_action = {'i': a_i, 'j': a_j}

                # Sample transition using isolated CRN generator
                try:
                    next_state = env.sample_transition(true_state, joint_action, rng=env_rng_trans)
                except TypeError:
                    next_state = env.sample_transition(true_state, joint_action)

                # Sample observations using isolated per-agent CRN generators
                try:
                    o_i = env.sample_observation(next_state, joint_action, 'i', rng=env_rng_obs_i)
                except TypeError:
                    o_i = env.sample_observation(next_state, joint_action, 'i')

                try:
                    o_j = env.sample_observation(next_state, joint_action, 'j', rng=env_rng_obs_j)
                except TypeError:
                    o_j = env.sample_observation(next_state, joint_action, 'j')

                r_i = env.get_reward(true_state, joint_action, next_state, 'i')
                r_j = env.get_reward(true_state, joint_action, next_state, 'j')

                cum_reward_i += r_i
                cum_reward_j += r_j
                is_terminal = env.is_terminal(next_state)

                stats_i = planner_i.get_detailed_stats()
                stats_j = planner_j.get_detailed_stats()

                n_i = _get_n_particles(planner_i)
                n_j = _get_n_particles(planner_j)
                min_i_particles = max(100, int(initial_n_i * 0.1)) if initial_n_i > 0 else 0
                min_j_particles = max(100, int(initial_n_j * 0.1)) if initial_n_j > 0 else 0

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
                    "n_particles_i": n_i,
                    "n_particles_j": n_j,
                    "status": "Active",
                    "action_values_i": stats_i.get("action_values"),
                    "action_values_j": stats_j.get("action_values"),
                    "process_rss_mb": round(SystemMonitor.get_process_memory()[0] / (1024.0 * 1024.0), 2)
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

                last_custom_metrics = self._get_custom_metrics(true_state, next_state, env, planner_i, planner_j)
                record.update(last_custom_metrics)
                record.update(_get_opponent_level_distribution(planner_i, 'j'))

                trial_records.append(record)

                if not is_terminal:
                    planner_i.update_root(a_i, o_i, min_i_particles)
                    planner_j.update_root(a_j, o_j, min_j_particles)
                    true_state = next_state
            else:
                record = {
                    "trial": trial_id, "step": t,
                    "reward_i": 0.0, "cum_reward_i": cum_reward_i,
                    "planning_time_i": 0.0,
                    "reward_j": 0.0, "cum_reward_j": cum_reward_j,
                    "n_particles_i": None, "n_particles_j": None,
                    "status": "Terminal",
                    "action_values_i": None, "action_values_j": None
                }
                trial_records.append(record)

        del planner_i, planner_j, env
        gc.collect()
        return trial_records

    def run_batch(self, max_workers: Optional[int] = None) -> pd.DataFrame:
        all_records = []
        # OS-level memory guardrail verification
        watchdog = MemoryWatchdog()
        sentry = watchdog.check_and_enforce()
        if not sentry["safe"]:
            logger.warning(f"OS Resource Warning prior to batch dispatch: {sentry.get('reason')}")

        logger.info(f"Initialized ProcessPoolExecutor with {max_workers} worker processes.")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_single_trial_parallel, t_id)
                       for t_id in range(self.config.n_trials)]

            # Note: Removed standard tqdm so progress bars don't conflict with structured logs.
            # Instead, we emit an INFO log every 10% completion.
            completed = 0
            log_interval = max(1, self.config.n_trials // 10)

            failed_trials = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    records = future.result()
                    all_records.extend(records)
                    completed += 1
                    if completed % log_interval == 0 or completed == self.config.n_trials:
                        logger.info(f"Progress: {completed}/{self.config.n_trials} trials completed.")
                except Exception as exc:
                    failed_trials += 1
                    logger.error(f"Trial failed with exception: {exc}", exc_info=True)

        df = pd.DataFrame(all_records)
        if completed < self.config.n_trials:
            if self.log_dir and not df.empty:
                partial_path = os.path.join(self.log_dir, "batch_results_partial.csv")
                df.to_csv(partial_path, index=False)
                logger.warning(f"Batch incomplete ({completed}/{self.config.n_trials} trials). Saved to {partial_path}")
            raise RuntimeError(f"Batch execution incomplete: {completed}/{self.config.n_trials} completed, {failed_trials} failed.")

        if self.log_dir and not df.empty:
            csv_path = os.path.join(self.log_dir, "batch_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Aggregation complete. Results successfully saved to {self.log_dir}")

        return df

    def run_single_trial_with_snapshots(self, trial_id: int = 0) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
        """
        Executes a single detailed episode trial, capturing step telemetry and full
        multi-level nested mental model belief snapshots at every timestep t = 0 ... T.

        Returns:
            df: DataFrame containing the episode step-by-step metrics.
            snapshots_by_step: Dictionary mapping step index t -> nested belief hierarchy dict.
        """
        random.seed(trial_id)
        np.random.seed(trial_id)

        env, planner_i, planner_j, true_state = self._setup_domain()
        trial_records = []
        snapshots_by_step: Dict[int, Dict[str, Any]] = {}

        cum_reward_i = 0.0
        cum_reward_j = 0.0
        is_terminal = False

        # Step 0: Initial condition snapshot
        initial_stats_i = planner_i.get_detailed_stats() if hasattr(planner_i, 'get_detailed_stats') else {}
        initial_stats_j = planner_j.get_detailed_stats() if hasattr(planner_j, 'get_detailed_stats') else {}
        n_i = _get_n_particles(planner_i)
        n_j = _get_n_particles(planner_j)
        initial_n_i = n_i
        initial_n_j = n_j

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
            "n_particles_i": n_i,
            "n_particles_j": n_j,
            "status": "Active",
            "action_values_i": initial_stats_i.get("action_values"),
            "action_values_j": initial_stats_j.get("action_values")
        }
        init_custom = self._get_custom_metrics(true_state, true_state, env, planner_i, planner_j)
        init_record.update(init_custom)
        init_record.update(_get_opponent_level_distribution(planner_i, 'j'))
        trial_records.append(init_record)

        snapshots_by_step[0] = extract_nested_belief_hierarchy(planner_i, 'i')

        # Decision steps: t = 1 ... max_steps
        for t in range(1, self.config.max_steps + 1):
            if not is_terminal:
                start_time_i = time.perf_counter()
                a_i = planner_i.get_action()
                plan_time_i = time.perf_counter() - start_time_i

                a_j = planner_j.get_action()
                joint_action = {'i': a_i, 'j': a_j}

                next_state = env.sample_transition(true_state, joint_action)
                o_i = env.sample_observation(next_state, joint_action, 'i')
                o_j = env.sample_observation(next_state, joint_action, 'j')

                r_i = env.get_reward(true_state, joint_action, next_state, 'i')
                r_j = env.get_reward(true_state, joint_action, next_state, 'j')

                cum_reward_i += r_i
                cum_reward_j += r_j
                is_terminal = env.is_terminal(next_state)

                stats_i = planner_i.get_detailed_stats()
                stats_j = planner_j.get_detailed_stats()

                n_i = _get_n_particles(planner_i)
                n_j = _get_n_particles(planner_j)
                min_i_particles = max(100, int(initial_n_i * 0.1)) if initial_n_i > 0 else 0
                min_j_particles = max(100, int(initial_n_j * 0.1)) if initial_n_j > 0 else 0

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
                    "n_particles_i": n_i,
                    "n_particles_j": n_j,
                    "status": "Terminal" if is_terminal else "Active",
                    "action_values_i": stats_i.get("action_values"),
                    "action_values_j": stats_j.get("action_values"),
                    "process_rss_mb": round(SystemMonitor.get_process_memory()[0] / (1024.0 * 1024.0), 2)
                }
                custom_metrics = self._get_custom_metrics(true_state, next_state, env, planner_i, planner_j)
                record.update(custom_metrics)
                record.update(_get_opponent_level_distribution(planner_i, 'j'))
                trial_records.append(record)

                # Particle filter updates
                if hasattr(planner_i, 'update_root'):
                    planner_i.update_root(a_i, o_i, min_particles=min_i_particles)
                if hasattr(planner_j, 'update_root'):
                    planner_j.update_root(a_j, o_j, min_particles=min_j_particles)

                true_state = next_state
                snapshots_by_step[t] = extract_nested_belief_hierarchy(planner_i, 'i')
                gc.collect()

        df = pd.DataFrame(trial_records)
        if self.log_dir:
            json_path = os.path.join(self.log_dir, f"nested_belief_snapshots_trial_{trial_id}.json")
            with open(json_path, "w") as f:
                json.dump(snapshots_by_step, f, indent=2)

        del planner_i, planner_j, env
        gc.collect()
        return df, snapshots_by_step