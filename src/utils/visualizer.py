"""Render the real search tree and its immutable private-belief hypotheses.

Opponent beliefs are probability measures, not search trees. The diagram labels
them as beliefs and never invents visit counts or Q-values for them. Display depth
is bounded independently of inference; shared immutable values are deduplicated.
"""

import graphviz


class ForestVisualizer:
    def __init__(self, solver_bank):
        self.bank = solver_bank

    def export_forest(self, root_planner, filename="forest", step=0):
        dot = graphviz.Digraph(comment=f"Interactive search and beliefs, step {step}")
        dot.attr(rankdir="LR")

        def tree(node, depth):
            key = "search_" + str(id(node))
            dot.node(key, f"Search visits={node.visit_count}", shape="box")
            if depth:
                for action, observations in node.children.items():
                    for observation, child in observations.items():
                        dot.edge(key, tree(child, depth - 1), label=f"{action}/{observation}")
            return key

        seen = {}

        def belief(value, depth):
            if value in seen:
                return seen[value]
            key = "belief_" + str(len(seen))
            seen[value] = key
            dot.node(key, f"Belief: {len(value.mass)} weighted hypotheses", shape="ellipse")
            if depth:
                groups = {}
                for atom, mass in value.mass:
                    groups[atom.opponent] = groups.get(atom.opponent, 0) + mass
                for model, mass in groups.items():
                    if model.belief is not None:
                        target = belief(model.belief, depth - 1)
                        dot.edge(
                            key,
                            target,
                            label=f"{model.frame.agent_id} L{model.frame.level}: {mass:.3f}",
                        )
            return key

        search_root = tree(root_planner.root, 3)
        if root_planner.belief is not None:
            dot.edge(
                search_root,
                belief(root_planner.belief, 3),
                label="authoritative belief",
                style="dashed",
            )
        return dot.render(filename, view=False, format="png", cleanup=True)
