"""Planner profiles, defined once for ``reachable_decisions decide`` and for study specs.

A profile names a full I-POMCP configuration, so a single decision and an episode that
state the same profile run the same planner. ``certified`` is certified search with
bound-directed selection (RESULTS 12.127, 12.129): belief-dependent frontier bounds,
probability-weighted scoring, the exact two-step tail, pruning, bound leaves, lazy
expansion and IEMax selection for an agent's own decision; every modeled level runs
the same search, resolved to its budget as a cap, and is predicted by its plausible set.
"""

from ipomdp.planning.stopping import ModeledResolution

PROFILES = ("certified",)
MODELED_PILOT = 50


def own_options(profile, selection=None):
    """``PlannerConfig`` options of an agent's own planner; ``selection`` overrides the profile's."""
    _check(profile)
    return dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                leaf_eval="bound", lazy_expansion=True, selection=selection or "iemax")


def modeled_options(profile, cap, selection=None, pilot=MODELED_PILOT):
    """``PlannerConfig`` options of a modeled level whose budget (the resolution's cap) is ``cap``."""
    return own_options(profile, selection) | dict(
        modeled_resolution=ModeledResolution(pilot=min(pilot, cap)), modeled_prediction="plausible")


def _check(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}")
