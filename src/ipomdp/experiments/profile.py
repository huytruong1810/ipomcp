"""Planner profiles, defined once for ``reachable_decisions decide``, the panel and study specs.

A profile names a full I-POMCP configuration, so a single decision and an episode that
state the same profile run the same planner. ``certified`` is certified search directed by
its bounds: belief-dependent frontier bounds, probability-weighted scoring, the exact
two-step tail, pruning and IEMax selection; each simulation refines, at the root, the
largest upper bound among the actions that may still be optimal and whose interval is open
(``root_rule="open"``), follows the open branch with the largest probability-weighted gap,
and expands one history (``outcome="gap"``; THEORY.md, "Termination of the gap-directed
rules"). Every modeled level runs the same search, resolved to its budget as a cap, and is
predicted by its plausible set.

Selecting ``"ucb"`` instead (ablation of bound-directed search) also returns the descent to
the sampled outcome rule, since the gap rule chooses by bounds and needs IEMax.
"""

from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.stopping import ModeledResolution

PROFILES = ("certified",)
MODELED_PILOT = 50


def own_options(profile, selection=None):
    """``PlannerConfig`` options of an agent's own planner; ``selection`` overrides the profile's."""
    _check(profile)
    selection = selection or "iemax"
    directed = dict(outcome="gap", root_rule="open") if selection == "iemax" else dict(outcome="sample", root_rule="upper")
    return dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                leaf_eval="bound", lazy_expansion=True, selection=selection) | directed


def modeled_options(profile, cap, selection=None, pilot=MODELED_PILOT):
    """``PlannerConfig`` options of a modeled level whose budget (the resolution's cap) is ``cap``."""
    return own_options(profile, selection) | dict(
        modeled_resolution=ModeledResolution(pilot=min(pilot, cap)), modeled_prediction="plausible")


# Components of the certified profile that can be switched off one at a time (the ablation rows of the paper).
# "gap" returns the descent to the sampled outcome rule (outcome "sample", root rule "upper"); the gap rule needs
# pruning and IEMax and makes leaf evaluation and lazy expansion irrelevant, so "prune" and "bound_leaves" are
# accepted only with "gap" switched off as well. "resolution" acts on modeled levels only: each searches its whole
# budget and is predicted by its argmax.
OFF_COMPONENTS = ("gap", "exact_tail", "prune", "bound_leaves", "pw_scoring", "resolution")


def switch_off(options, components, modeled=False):
    """``options`` (a profile's ``PlannerConfig`` options) with the named ``components`` switched off."""
    components = set(components or ())
    unknown = components - set(OFF_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown profile components {sorted(unknown)}; choose from {OFF_COMPONENTS}")
    needs_gap_off = components & {"prune", "bound_leaves"}
    if needs_gap_off and "gap" not in components and options.get("outcome") == "gap":
        raise ValueError(f"Switching off {sorted(needs_gap_off)} needs 'gap' off too: the gap rule prunes and "
                         "does not evaluate leaves")
    options = dict(options)
    if "gap" in components:
        options |= dict(outcome="sample", root_rule="upper")
    if "exact_tail" in components:
        options["exact_tail"] = False
    if "prune" in components:
        options["prune"] = False
    if "bound_leaves" in components:
        options |= dict(leaf_eval="rollout", lazy_expansion=False)
    if "pw_scoring" in components:
        options["scoring"] = "legacy"
    if "resolution" in components and modeled:
        options.pop("modeled_resolution", None)
        options["modeled_prediction"] = "argmax"
    return options


def planner_options(profile, selection=None, off=None):
    """``own_options`` with the components ``off`` switched off (an agent's own planner)."""
    return switch_off(own_options(profile, selection), off)


def level_options(profile, cap, selection=None, off=None, pilot=MODELED_PILOT):
    """``modeled_options`` with the components ``off`` switched off (a modeled level of budget ``cap``)."""
    return switch_off(modeled_options(profile, cap, selection, pilot), off, modeled=True)


def recorded(options):
    """``options`` as a row records them: the fields that differ from ``PlannerConfig``'s defaults."""
    defaults = PlannerConfig(1, 1, 1)
    return {k: (str(v) if k == "modeled_resolution" else v)
            for k, v in options.items() if getattr(defaults, k) != v}


def _check(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}")
