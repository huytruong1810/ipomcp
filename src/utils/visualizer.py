import graphviz
from typing import Set, Dict, List
from solvers.solver_bank import SolverBank
from solvers.i_pomcp import IPOMCPPlanner
from solvers.node import POMCPNode


class ForestVisualizer:
    """
    Exports the I-POMCP Solver Forest to Graphviz format.
    Visualize the Protagonist's Tree, the referenced Opponent Nodes, and the links between them.
    """

    def __init__(self, solver_bank: SolverBank):
        self.bank = solver_bank

    def export_forest(self, root_planner: IPOMCPPlanner, filename: str = "forest", step: int = 0):
        """
        Generates a .dot and .png image of the current mental state.

        Args:
            root_planner: The Level 2 (Protagonist) planner.
            filename: Output filename prefix.
            step: Current simulation step (for labeling).
        """
        # Create Digraph
        dot = graphviz.Digraph(comment=f'I-POMCP Forest Step {step}')
        dot.attr(rankdir='LR')  # Left-to-Right layout

        # We track visited IDs to handle the DAG nature (merging paths)
        # and prevent infinite recursion if loops existed (shouldn't in trees).
        visited: Set[int] = set()

        # 1. Draw Protagonist Tree (Limit depth 3 for readability)
        self._add_tree_to_dot(dot, root_planner.root, f"I", visited, depth_limit=3, is_root=True)

        # 2. Draw Opponent Sub-Trees
        # We only draw nodes that are actually referenced by the protagonist's current belief.
        referenced_nodes = self._collect_referenced_nodes(root_planner.root)

        for agent_id, nodes in referenced_nodes.items():
            for i, node in enumerate(nodes):
                # Unique prefix for opponent nodes to avoid collision
                self._add_tree_to_dot(dot, node, f"J_{agent_id}", visited, depth_limit=2,
                                      label_prefix=f"Opp({agent_id})")

        # 3. Draw Belief Links (The "Interactive" part)
        # We sample particles from the protagonist's root to show what they are pointing to.
        if root_planner.root.belief_particles:
            # We group edges to avoid drawing 1000 lines.
            # Logic: "N particles point to Node X"

            link_counts: Dict[int, int] = {}  # Map OpponentNodeID -> Count

            for p in root_planner.root.belief_particles:
                for other_id, (_, node_ptr) in p.models.items():
                    if node_ptr:
                        nid = id(node_ptr)
                        link_counts[nid] = link_counts.get(nid, 0) + 1

            # Draw edges with weights
            start_node_id = str(id(root_planner.root))
            for target_nid, count in link_counts.items():
                if target_nid in visited:  # Only link if we drew the node
                    label = f"{count} parts"
                    dot.edge(start_node_id, str(target_nid), style="dashed", color="red", label=label, penwidth="2.0")

        # Render
        try:
            output_path = dot.render(filename, view=False, format='png', cleanup=True)
            print(f"[Viz] Forest visualized to {output_path}")
        except Exception as e:
            print(f"[Viz] Visualization failed (Graphviz installed?): {e}")

    def _add_tree_to_dot(self, dot, node: POMCPNode, name_prefix: str, visited: Set[int], depth_limit: int,
                         is_root=False, label_prefix=""):
        node_id = str(id(node))

        # If we already drew this node, skip to avoid duplicates (DAG structure)
        if id(node) in visited: return
        visited.add(id(node))

        if depth_limit < 0: return

        # Node Style
        # Green = Visited/Expanded, Grey = Unvisited
        color = "lightblue" if is_root else ("lightgreen" if node.visit_count > 0 else "lightgrey")
        shape = "box" if is_root else "ellipse"

        # Label Construction
        # Shows N (Visits) and Best Action
        label = f"{label_prefix}\nN={node.visit_count}"
        if node.visit_count > 0 and node.action_values:
            best_a = max(node.action_values, key=node.action_values.get)
            val = node.action_values[best_a]
            label += f"\nBest={best_a}\nV={val:.2f}"

        dot.node(node_id, label=label, style="filled", fillcolor=color, shape=shape)

        # Draw Children
        for action, obs_map in node.children.items():
            for obs, child in obs_map.items():
                # Only draw path if it has been visited or is the immediate next step
                if child.visit_count > 0 or is_root:
                    child_id = str(id(child))
                    self._add_tree_to_dot(dot, child, name_prefix, visited, depth_limit - 1)
                    dot.edge(node_id, child_id, label=f"{action}/{obs}")

    def _collect_referenced_nodes(self, root: POMCPNode) -> Dict[str, List[POMCPNode]]:
        """
        Scans the belief to find which opponent nodes are currently 'active' in the agent's mind.
        """
        refs = {}
        if not root.belief_particles: return refs

        for p in root.belief_particles:
            for aid, (_, node_ptr) in p.models.items():
                if node_ptr:
                    if aid not in refs: refs[aid] = []
                    # We use object identity to deduplicate
                    if node_ptr not in refs[aid]:
                        refs[aid].append(node_ptr)
        return refs