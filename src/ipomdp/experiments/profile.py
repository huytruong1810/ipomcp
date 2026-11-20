"""Planner profiles, defined once for ``reachable_decisions decide`` and for study specs.

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


def _check(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}")
