"""I-POMCP versus the exact I-POMDP on the L3-vs-L2 Tiger design, over a resource ladder.

Each rung fixes the real search budgets, the modeled-opponent budget and the
interactive-belief support bound K (all nesting levels, systematic resampling).
Every rung uses the same trial ids, hence the same environment random streams.

Exact nested beliefs grow exponentially with the episode (about x1.5 per step at
L2), so a fully exact 20-step game is infeasible. Each real agent instead carries
an exact *shadow*: an unbounded exact I-POMDP agent that conditions on the same
own actions and observations, so exact quantities are evaluated on the history
that actually occurred. A shadow is audited until one of its declared caps is
exceeded (decision time or belief support). Later steps are recorded as not
audited, and coverage is reported rather than extrapolated.

Per audited step, for each agent:
- ``search_regret``: exact loss of the chosen policy in the agent's own
  approximate decision problem (bounded belief, finite modeled policies). This
  is search error only.
- ``exact_regret``: exact loss of the chosen action in the exact I-POMDP at the
  shadow's belief. It combines search, opponent-model and belief-bound error.
- ``tv_depth{d}``: total variation of joint nested-model path masses at nesting
  depth d, against the shadow. ``phys_err_depth{d}`` is the physical-belief error
  at that depth.
- ``policy_tv``: belief-weighted total variation between each finite modeled
  opponent policy and the exact policy of the same model.
Outcome-level convergence across rungs (cumulative rewards, mean hierarchies) is
computed by the analysis script.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from exact_ipomdp_agent import build_exact_agent

from core.config import ExperimentConfig
from examples.experiments.tiger_l3_vs_l2_experiment import TigerL3vsL2Runner
from examples.tiger.model.tiger_model import TIGER_LEFT
from solvers.exact.finite_policy_reference import FinitePolicyReference

RUNGS = {
    # name: (sims_i, sims_j, modeled_sims, belief support bound K)
    "r1": (2500, 2000, 10, 32),
    "r2": (5000, 4000, 25, 64),
    "r3": (10000, 7500, 50, 128),
    "r4": (20000, 15000, 100, 256),
}


SHADOW_CACHE = 20000


def current_rss_mb():
    """Current (not peak) resident memory of this process, from /proc."""
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def p_tl(belief):
    return math.fsum(w for atom, w in belief.mass if atom.state == TIGER_LEFT)


def nested_summary(planner, agent_id):
    """{path: [joint mass, mass-weighted P(tiger left) of that model's belief]}.

    Identical private beliefs within a path are merged exactly. L0 leaves have
    no belief and report None for the physical component.
    """
    out = {}

    def walk(weighted, path):
        total = math.fsum(m for _, m in weighted)
        out[path] = [total, math.fsum(m * p_tl(b) for b, m in weighted) / total]
        groups = {}
        for belief, mass in weighted:
            for atom, weight in belief.mass:
                frame = atom.opponent.frame
                key = f"{path}/{frame.agent_id}_L{frame.level}"
                children = groups.setdefault((key, frame.level), {})
                children[atom.opponent.belief] = (
                    children.get(atom.opponent.belief, 0.0) + mass * weight
                )
        for (key, level), children in sorted(groups.items()):
            if level == 0:
                out[key] = [math.fsum(children.values()), None]
            else:
                walk(list(children.items()), key)

    walk([(planner.belief, 1.0)], f"{agent_id}_L{planner.key.level}")
    return out


def summary_errors(approx, exact):
    """Per nesting depth: TV of joint path masses; physical-belief errors."""
    depths = {}
    for path in set(approx) | set(exact):
        depth = path.count("/")
        a = approx.get(path, [0.0, None])
        e = exact.get(path, [0.0, None])
        row = depths.setdefault(depth, {"tv": 0.0, "phys_weighted": 0.0, "phys_mass": 0.0})
        row["tv"] += 0.5 * abs(a[0] - e[0])
        if a[1] is not None and e[1] is not None:
            weight = min(a[0], e[0])
            row["phys_weighted"] += weight * abs(a[1] - e[1])
            row["phys_mass"] += weight
    result = {}
    for depth, row in sorted(depths.items()):
        result[f"tv_depth{depth}"] = row["tv"]
        if row["phys_mass"] > 0:
            result[f"phys_err_depth{depth}"] = row["phys_weighted"] / row["phys_mass"]
    return result


def policy_tv(planner, shadow):
    """Belief-weighted TV between finite modeled and exact opponent policies."""
    total, weight = 0.0, 0.0
    for atom, mass in planner.belief.mass:
        opponent = atom.opponent
        if opponent.frame.level == 0:
            continue
        finite = planner.solver_bank.policy(opponent)
        exact = shadow.solver_bank.policy(opponent)
        actions = set(finite) | set(exact)
        total += mass * 0.5 * math.fsum(abs(finite.get(a, 0) - exact.get(a, 0)) for a in actions)
        weight += mass
    return total / weight if weight else 0.0


def _loss(q, policy):
    best = max(q.values())
    return math.fsum(p * (best - q[a]) for a, p in policy.items())


class ConvergenceRunner(TigerL3vsL2Runner):
    def __init__(
        self,
        config,
        log_dir=None,
        rung="r3",
        depth=5,
        gamma=0.95,
        audit_seconds=120.0,
        audit_support=400,
        audit_rss_mb=1200,
    ):
        sims_i, sims_j, modeled, support = RUNGS[rung]
        super().__init__(
            config,
            log_dir,
            profile="validated",
            sims_i=sims_i,
            sims_j=sims_j,
            modeled_sims=modeled,
            depth=depth,
            gamma=gamma,
            max_support=support,
        )
        self.rung = rung
        self.audit_seconds, self.audit_support = audit_seconds, audit_support
        self.audit_rss_mb = audit_rss_mb

    def _instrument(self, planner, agent):
        """Two audits on the exact shadow, each with its own declared caps.

        The decision audit (exact and search regret) stops once one audited
        decision exceeds ``audit_seconds``. The cheaper belief audit (nested TV,
        physical error, policy TV) continues on the exact filter until the shadow
        support exceeds ``audit_support`` or one belief step exceeds ``audit_seconds``.
        """
        state = self.audits[agent] = {
            "shadow": build_exact_agent(
                agent,
                planner.key.level,
                self.depth,
                self.gamma,
                cache=SHADOW_CACHE,
                physics=planner.pomdp_model,
            ),
            "search": FinitePolicyReference(planner.pomdp_model, planner.solver_bank, self.gamma),
            "decision_active": True,
            "belief_active": True,
            "decision_stop": "",
            "belief_stop": "",
            "decision": {},
        }
        original_policy_for, original_update = planner.policy_for, planner.update_root

        def policy_for(model, modeled=False):
            policy = original_policy_for(model, modeled)
            if modeled or not state["decision_active"]:
                return policy
            start = time.perf_counter()
            own_q = state["search"].q_values(model, self.depth)
            shadow = state["shadow"]
            exact_q = shadow.q_values(shadow.model())
            elapsed = time.perf_counter() - start
            state["decision"] = {
                f"search_regret_{agent}": _loss(own_q, policy),
                f"exact_regret_{agent}": _loss(exact_q, policy),
                f"policy_{agent}": json.dumps({str(a): p for a, p in policy.items()}),
                f"exact_q_{agent}": json.dumps({str(a): q for a, q in exact_q.items()}),
                f"search_q_{agent}": json.dumps({str(a): q for a, q in own_q.items()}),
                f"estimated_q_{agent}": json.dumps(
                    {str(a): q for a, q in planner.recommendation_values.items()}
                ),
                f"audit_seconds_{agent}": elapsed,
            }
            if elapsed > self.audit_seconds:
                state.update(decision_active=False, decision_stop="decision time cap", search=None)
            return policy

        def update_root(action, observation):
            original_update(action, observation)
            if not state["belief_active"]:
                return
            start = time.perf_counter()
            state["shadow"].update_root(action, observation)
            state["belief_update_seconds"] = time.perf_counter() - start
            if not state["decision_active"]:
                # Only exact policies are still needed; release decision-solve caches.
                state["shadow"].reference._q.cache_clear()
                state["shadow"].reference.branches.cache_clear()
            if len(state["shadow"].belief.mass) > self.audit_support:
                reason = "support cap"
            elif current_rss_mb() > self.audit_rss_mb:
                reason = "memory cap"
            elif state["belief_update_seconds"] > self.audit_seconds:
                reason = "belief time cap"
            else:
                return
            state.update(
                belief_active=False,
                belief_stop=reason,
                decision_active=False,
                decision_stop=state["decision_stop"] or reason,
                shadow=None,
                search=None,
            )

        planner.policy_for, planner.update_root = policy_for, update_root

    def _setup_domain(self):
        env, planner_i, planner_j, state = super()._setup_domain()
        self.audits = {}
        self._instrument(planner_i, "i")
        self._instrument(planner_j, "j")
        return env, planner_i, planner_j, state

    def _get_custom_metrics(self, true_state, next_state, env, planner_i, planner_j):
        metrics = super()._get_custom_metrics(true_state, next_state, env, planner_i, planner_j)
        for agent, planner in (("i", planner_i), ("j", planner_j)):
            state = self.audits[agent]
            summary = nested_summary(planner, agent)
            metrics[f"nested_{agent}"] = json.dumps(summary)
            metrics.update(state["decision"])
            metrics[f"decision_audited_{agent}"] = bool(state["decision"])
            state["decision"] = {}
            metrics[f"belief_audited_{agent}"] = state["belief_active"]
            metrics[f"decision_stop_{agent}"] = state["decision_stop"]
            metrics[f"belief_stop_{agent}"] = state["belief_stop"]
            if not state["belief_active"]:
                continue
            start = time.perf_counter()
            shadow = state["shadow"]
            exact_summary = nested_summary(shadow, agent)
            metrics[f"nested_exact_{agent}"] = json.dumps(exact_summary)
            metrics[f"exact_support_{agent}"] = len(shadow.belief.mass)
            for name, value in summary_errors(summary, exact_summary).items():
                if name != "tv_depth0":  # the root always has mass one
                    metrics[f"{name}_{agent}"] = value
            metrics[f"policy_tv_{agent}"] = policy_tv(planner, shadow)
            metrics[f"belief_audit_seconds_{agent}"] = (
                state.get("belief_update_seconds", 0.0) + time.perf_counter() - start
            )
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rung", choices=sorted(RUNGS), required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=21)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--audit-seconds", type=float, default=120.0)
    parser.add_argument("--audit-support", type=int, default=400)
    parser.add_argument("--audit-rss-mb", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner = ConvergenceRunner(
        ExperimentConfig(
            n_trials=args.trials,
            max_steps=args.steps,
            max_workers=args.workers,
            trial_timeout_seconds=args.timeout,
            max_trial_rss_mb=args.max_rss_mb,
            export_trees=False,
            verbose=False,
        ),
        log_dir=str(args.output),
        rung=args.rung,
        audit_seconds=args.audit_seconds,
        audit_support=args.audit_support,
        audit_rss_mb=args.audit_rss_mb,
    )
    start = time.monotonic()
    df = runner.run_batch()
    print(f"{len(df)} rows in {time.monotonic() - start:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
