"""
ipomdp — Interactive POMDP belief representation.

This package contains the data structures that extend a standard POMDP
belief to the *interactive* (multi-agent) setting:

* :class:`InteractiveParticle` — A single particle that bundles a physical
  state with the nested *mental states* (MCTS-tree pointers) of modelled
  opponents.  This is the fundamental unit of the Interactive Particle
  Filter (IPF).

Theoretical background
----------------------
In I-POMDPs (Gmytrasiewicz & Doshi, 2005), the state space is
*augmented* with the intentional models of other agents.  A belief
particle therefore contains not just "where is the world?" but also
"what does the opponent think?".  Level-k reasoning truncates this
recursion: a Level-0 agent has no mental model, a Level-1 agent models
opponents as Level-0, and so on.
"""

from ipomdp.belief import InteractiveParticle

__all__ = ["InteractiveParticle"]
