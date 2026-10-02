"""History-node structure and serialization (Bellman backups: test_bellman_backups)."""

import gc

from ipomdp.planning.node import POMCPNode


def test_node_creation_and_linking():
    root = POMCPNode()
    assert root.visit_count == 0
    child = root.create_child("A1", "O0")
    assert child.parent is root
    assert root.create_child("A1", "O0") is child
    assert root.get_child("A1", "O0") is child
    assert root.get_child("A2", "O0") is None


def test_parent_links_do_not_keep_discarded_roots_alive():
    root = POMCPNode()
    child = root.create_child("A1", "O0")
    del root
    gc.collect()
    assert child.parent is None


def test_node_serialization():
    root = POMCPNode()
    root.visit_count = 10
    root.action_counts["A1"] = 6
    root.action_values["A1"] = 2.5
    root.create_child("A1", "O0").visit_count = 5
    data = root.to_dict(max_depth=2)
    assert data["visit_count"] == 10
    assert data["children"]["A1"]["O0"]["visit_count"] == 5
