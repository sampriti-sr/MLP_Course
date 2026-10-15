#!/usr/bin/env python3
"""Create a new course page (a notebook) from templates/page-template.ipynb.

    python tools/new_page.py docs/lifecycle/01-problem-framing.ipynb "Problem Framing"

Fills in the title and the 'Open in Colab' / install links for the right file and git ref
(default: the current branch), then reminds you to add the page to `nav:` in mkdocs.yml.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
REPO = "sampriti-sr/MLP_Course"


def current_branch() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        return out if out and out != "HEAD" else "main"
    except (OSError, subprocess.CalledProcessError):
        return "main"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="notebook path inside docs/, e.g. docs/lifecycle/01-framing.ipynb")
    ap.add_argument("title", help="page title (the H1)")
    ap.add_argument("--ref", default=None, help="git ref for Colab links (default: current branch)")
    a = ap.parse_args()

    dest = (ROOT / a.path).resolve() if not Path(a.path).is_absolute() else Path(a.path)
    try:
        rel = dest.relative_to(ROOT).as_posix()
    except ValueError:
        ap.error("path must be inside the repository")
    if not rel.startswith("docs/") or dest.suffix != ".ipynb":
        ap.error("path must look like docs/<section>/<name>.ipynb")
    if dest.exists():
        ap.error(f"{rel} already exists; refusing to overwrite")

    ref = a.ref or current_branch()
    nb = nbformat.read(ROOT / "templates" / "page-template.ipynb", as_version=4)
    for c in nb.cells:
        c.source = (c.source.replace("@@REPO@@", REPO).replace("@@REF@@", ref)
                    .replace("@@PATH@@", rel).replace("@@TITLE@@", a.title))
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dest)
    nav_path = rel.removeprefix("docs/")
    print(f"Created {rel}\nNext:\n  1. add `- {a.title}: {nav_path}` under nav: in mkdocs.yml"
          f"\n  2. write the page, then run: python tools/execute_notebooks.py --write {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
