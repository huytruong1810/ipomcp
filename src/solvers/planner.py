"""Common online decision interface; belief estimates belong to intentional planners."""

import abc


class Planner(abc.ABC):
    @abc.abstractmethod
    def get_action(self, belief=None):
        """Sample the next action from the planner's specified policy."""

    def update_root(self, action, observation):
        """Condition on one's own action and observation; L0 has no belief to update."""

    def get_detailed_stats(self):
        return {}

    def visualize(self, filename, step):
        """Render decision statistics for planners without a persistent search tree."""
        import json

        import graphviz

        dot = graphviz.Digraph(comment=f"Planner statistics at step {step}")
        dot.node("planner", json.dumps(self.get_detailed_stats(), indent=2), shape="box")
        return dot.render(filename, view=False, format="png", cleanup=True)
