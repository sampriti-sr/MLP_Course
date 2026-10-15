#!/usr/bin/env python3
"""Run and lint the course notebooks (the notebooks in docs/ ARE the site pages).

    python tools/execute_notebooks.py --lint            # static checks only, no kernel
    python tools/execute_notebooks.py --check           # execute everything; fail on any error
    python tools/execute_notebooks.py --write           # execute and save outputs into the files
    python tools/execute_notebooks.py --ref main --write-ref   # retarget Colab links + install cell

Defaults to every docs/**/*.ipynb. Pass paths to limit the run.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
REPO = "sampriti-sr/MLP_Course"
COLAB = re.compile(r"colab\.research\.google\.com/github/(?P<repo>[^/]+/[^/]+)/blob/(?P<ref>[^/]+)/(?P<path>[^)\s]+\.ipynb)")
INSTALL = re.compile(r"mlp-course @ git\+https://github\.com/(?P<repo>[^@\"]+)@(?P<ref>[^\"]+)\"")


def find_notebooks(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return sorted(p for p in (ROOT / "docs").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def lint(nb_path: Path, nb) -> list[str]:
    problems = []
    rel = nb_path.relative_to(ROOT).as_posix()
    first = nb.cells[0].source if nb.cells else ""
    m = COLAB.search(first)
    if nb.cells and nb.cells[0].cell_type != "markdown" or not m:
        problems.append("first cell must be the 'Open in Colab' badge (markdown)")
    elif m["path"] != rel:
        problems.append(f"Colab badge points at '{m['path']}', expected '{rel}'")
    if not any(c.cell_type == "code" and INSTALL.search(c.source) for c in nb.cells):
        problems.append("missing the `%pip install mlp-course @ git+...` setup cell")
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if c.get("execution_count") is None:
            problems.append(f"cell {i} has not been executed (run with --write before committing)")
        if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
            problems.append(f"cell {i} contains an error output")
    return problems


def execute(nb_path: Path, nb, timeout: int):
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}}, record_timing=False)
    client.execute()
    for c in nb.cells:  # keep diffs small and deterministic
        c.get("metadata", {}).pop("execution", None)
    nb.metadata.pop("widgets", None)
    return nb


def retarget(nb, ref: str) -> bool:
    changed = False
    for c in nb.cells:
        new = COLAB.sub(lambda m: m.group(0).replace(f"/blob/{m['ref']}/", f"/blob/{ref}/"), c.source)
        new = INSTALL.sub(lambda m: m.group(0).replace(f"@{m['ref']}\"", f"@{ref}\""), new)
        if new != c.source:
            c.source, changed = new, True
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--lint", action="store_true", help="static checks only")
    ap.add_argument("--check", action="store_true", help="execute; do not modify files")
    ap.add_argument("--write", action="store_true", help="execute and save outputs")
    ap.add_argument("--ref", help="git ref to use in Colab links / install cell")
    ap.add_argument("--write-ref", action="store_true", help="apply --ref to the files")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if not (a.lint or a.check or a.write or a.write_ref):
        ap.error("choose one of --lint, --check, --write, --write-ref")

    failed = False
    for p in find_notebooks(a.paths):
        rel = p.relative_to(ROOT).as_posix()
        nb = nbformat.read(p, as_version=4)
        if a.write_ref:
            if not a.ref:
                ap.error("--write-ref needs --ref")
            if retarget(nb, a.ref):
                nbformat.write(nb, p)
                print(f"[ref ] {rel} -> {a.ref}")
            continue
        if a.check or a.write:
            print(f"[exec] {rel} ...", flush=True)
            try:
                nb = execute(p, nb, a.timeout)
            except Exception as e:  # nbclient raises CellExecutionError with the traceback in str(e)
                print(f"[FAIL] {rel}\n{e}", file=sys.stderr)
                failed = True
                continue
            if a.write:
                nbformat.write(nb, p)
        problems = lint(p, nb)
        for msg in problems:
            print(f"[lint] {rel}: {msg}", file=sys.stderr)
        failed |= bool(problems)
        if not problems:
            print(f"[ ok ] {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
