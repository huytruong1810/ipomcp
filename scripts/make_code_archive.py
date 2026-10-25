"""Build the anonymized code archive for the double-blind supplement (IJCAI-27).

Contents: the package (src/ipomdp), the tests, pyproject.toml, the theory, architecture and domain notes, and the
scripts and pre-registered endpoints of every experiment folder that docs/RESULTS.md cites (no data, no source
snapshots).
Every file is scrubbed of local paths and user names, and the result is checked: any remaining identifying string
aborts the build.

Usage: python scripts/make_code_archive.py <out.zip>
"""

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"src", "__pycache__", "trials", "rows", "rows_offline", "refs", "refs_k5_price", "curves", "replays",
             "venv", "third_party", "run", "attempts", "diag", "price", "pilot_out", "full_out", "gate", "score_src"}
SCRIPT_SUFFIXES = {".py", ".sh"}
NOTES = {"ENDPOINTS.md", "KILL_CRITERION.txt"}
SCRUB = [
    (re.compile(r"/home/[A-Za-z0-9_.-]+/projects/ipomcp"), "<repo>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "<home>"),
    (re.compile(r"/tmp/claude-\d+/[^\s'\"]*"), "<tmp>"),
]
FORBIDDEN = re.compile(r"andyj1810|truong|gmail|uic\.edu|piotr|claude\.ai|anthropic|codex|antigravity", re.I)
ANON_README = """# I-POMCP (anonymized code for review)

Online Monte Carlo planning in finitely nested I-POMDPs, the exact references, the baselines' adapters and the
domains of the submission. Install with `pip install -e .` (Python >= 3.10); run the tests with `pytest`.
I-NTMCP is its authors' public implementation (github.com/RDLLab/i-ntmcp, commit 63deec3); put it on PYTHONPATH
for the I-NTMCP adapter and its tests. `experiments/` holds, per experiment, the driver scripts and the endpoints
fixed before each run (`<repo>` stands for the repository root).
"""


def scrub(text):
    for pattern, replacement in SCRUB:
        text = pattern.sub(replacement, text)
    return text


def files():
    for path in sorted((ROOT / "src" / "ipomdp").rglob("*.py")):
        yield path, Path("src/ipomdp") / path.relative_to(ROOT / "src" / "ipomdp")
    for path in sorted((ROOT / "tests").rglob("*.py")):
        yield path, Path("tests") / path.relative_to(ROOT / "tests")
    yield ROOT / "pyproject.toml", Path("pyproject.toml")
    for name in ("THEORY.md", "ARCHITECTURE.md", "DOMAINS.md"):
        yield ROOT / "docs" / name, Path("docs") / name
    cited = set(re.findall(r"results/([A-Za-z0-9_.-]+)", (ROOT / "docs" / "RESULTS.md").read_text()))
    for folder in sorted(ROOT / "results" / name for name in cited if (ROOT / "results" / name).is_dir()):
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if not path.is_file() or any(part in SKIP_DIRS for part in rel.parts[:-1]):
                continue
            if path.suffix in SCRIPT_SUFFIXES or path.name in NOTES:
                yield path, Path("experiments") / folder.name / rel


def main(out):
    seen = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", ANON_README)
        for source, name in files():
            text = scrub(source.read_text(errors="replace"))
            hit = FORBIDDEN.search(text)
            if hit:
                raise SystemExit(f"identifying string {hit.group(0)!r} left in {name}")
            archive.writestr(str(name), text)
            seen.append(name)
    print(f"{out}: {len(seen)} files")


if __name__ == "__main__":
    main(sys.argv[1])
