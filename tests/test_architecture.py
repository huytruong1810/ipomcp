"""The package's dependency rules, enforced from its import statements.

``docs/ARCHITECTURE.md`` draws the bounded contexts and the direction in which
they may depend on each other. A rule that only a document states erodes one
convenient import at a time (the analysis and experiment layers imported each
other, and the exact references imported a domain, before this test existed).
Here every ``import`` in the package, including those inside functions, is read
from the source and checked against the table below, so adding a dependency
means changing the table in the same commit, where a reviewer sees it.

The table, bottom-up:

- ``model``: the domain contract; depends on nothing.
- ``domains``: implementations of the contract.
- ``belief``: interactive states, beliefs and their updates; knows the contract, no domain.
- ``planning``: the planner and the solver bank; no domain, no reference.
- ``reference``: exact solutions, built on the planner's interfaces; no domain.
- ``baselines``: other planners behind the same interfaces.
- ``analysis``: reads beliefs, planners and references; never the experiment drivers.
- ``infra``: process supervision and paths; depends on nothing in the package.
- ``experiments``: the only layer that names domains and wires everything together.
"""

import ast
from pathlib import Path

import ipomdp

ALLOWED = {
    "model": set(),
    "domains": {"model"},
    "belief": {"model"},
    "planning": {"belief", "model"},
    "reference": {"belief", "planning"},
    "baselines": {"belief", "planning"},
    "analysis": {"belief", "planning", "reference"},
    "infra": set(),
    "experiments": {"analysis", "baselines", "belief", "domains", "infra", "planning", "reference"},
}
ROOT = Path(ipomdp.__file__).parent


def imports_by_context():
    """{context: {(imported context, importing file, line), ...}} for every import of the package itself."""
    found = {}
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if len(relative.parts) == 1:
            continue  # the package's own __init__ re-exports names and belongs to no context
        context = relative.parts[0]
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                parts = module.split(".")
                if parts[0] == "ipomdp" and len(parts) > 1 and parts[1] != context:
                    found.setdefault(context, set()).add((parts[1], str(relative), node.lineno))
    return found


def test_every_context_is_in_the_table():
    contexts = {path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith("_")}
    assert contexts == set(ALLOWED)


def test_contexts_depend_only_on_what_the_architecture_allows():
    violations = [
        f"{file}:{line} ({context} imports {imported})"
        for context, edges in imports_by_context().items()
        for imported, file, line in sorted(edges)
        if imported not in ALLOWED[context]
    ]
    assert not violations, "\n".join(violations)


def test_the_allowed_dependencies_have_no_cycle():
    """A table with a cycle would permit two contexts to import each other in turn."""
    order, remaining = [], dict(ALLOWED)
    while remaining:
        free = sorted(context for context, needs in remaining.items() if needs <= set(order))
        assert free, f"cycle among {sorted(remaining)}"
        order += free
        for context in free:
            del remaining[context]
