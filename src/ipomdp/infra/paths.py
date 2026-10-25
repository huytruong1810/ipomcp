"""Checkout layout: the one place that knows where source and repository roots are.

``<repository>/src/ipomdp/infra/paths.py``. Every module that needs these roots
imports the constants instead of counting parent directories from its own
location, so moving a module cannot silently change what a source lock or a
run manifest covers.
"""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SOURCE_ROOT.parent
