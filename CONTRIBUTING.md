# Contributing

Thanks for helping improve the course. The README covers setup and commands; this page covers the rules a change has to meet.

## Workflow

1. Branch from the current working branch (never commit to `main` directly).
2. One logical change per commit; say what changed and why.
3. Open a pull request. CI runs the unit tests, lints and executes every notebook, and builds the site with `--strict`. A change is ready when that run is green.

## Before you push

```bash
pytest -q
python tools/execute_notebooks.py --lint
python tools/execute_notebooks.py --write docs/path/to/page.ipynb   # re-run and save outputs
mkdocs build --strict
```

## Rules for pages

- **The notebooks are the pages.** Edit the `.ipynb`, re-run it, and commit the saved outputs; the site build does not execute code.
- **Use real data** through the `mlp_data` loaders. Do not invent a DataFrame.
- **Check every number in prose** against the executed output, and guard important claims with an `assert` in the notebook.
- **No Material syntax in notebooks.** `!!!`, `???` and `===` do not render there; use the HTML callouts in [`templates/AUTHORING.md`](templates/AUTHORING.md).
- **Start new pages from the template:** `python tools/new_page.py <path> "<title>"`, then add the page to `nav:` in `mkdocs.yml`.
- **No emojis;** use Material icons on the site and plain text elsewhere.

## Rules for datasets

- Raw data is never committed. Add a loader to `mlp_data/loaders.py` that downloads from the original publisher, with a checksum.
- Record the source, licence, size and known caveats in [`data/README.md`](data/README.md) before using a dataset in a page.

## Rules for helper code

Code in `mlp_data/` is imported by the notebooks, so it needs offline unit tests in `tests/`.
