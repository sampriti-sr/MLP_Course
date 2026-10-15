# Writing a course page

Every page is a **Jupyter notebook in `docs/`**. The site shows it with its saved outputs, and Colab opens the same file, so the code, the output and the "Open in Colab" button can never disagree.

## The workflow

```bash
pip install -r requirements-notebooks.txt            # once
python tools/new_page.py docs/<section>/<name>.ipynb "Page Title"
# add the page under `nav:` in mkdocs.yml
# write the page in Jupyter / VS Code
python tools/execute_notebooks.py --write docs/<section>/<name>.ipynb   # runs it, saves outputs
python tools/execute_notebooks.py --lint                                # same checks CI runs
mkdocs serve                                                            # look at it
```

Commit the notebook **with its outputs**. CI re-runs every notebook from scratch and fails if any cell errors or any cell was committed unexecuted.

## What every page must have

See `templates/page-template.ipynb`. In short: objectives, prerequisites, a real *situation*, a **Maths ↔ code** note (never a re-derivation, as students know the maths), a worked example on the course data, *what goes wrong* with evidence, a decision guide, check-yourself questions, exercises with hidden solutions, a recap.

**Rules of thumb**

- Use the course datasets via `mlp_data` (see `docs/getting-started/datasets.ipynb`). Do not invent a `df`.
- Compute numbers in code; never type them into prose by hand. If the data changes, the page should change with it.
- After each output, say in words what it *means*. A table with no interpretation is not teaching.
- Objectives start with a testable verb ("compute", "choose", "diagnose"), not "understand".
- Keep a page to roughly 20 to 30 minutes. If it is longer, split it.
- Keep runtime reasonable: heavy work belongs on a sample, and the page should say so.

## Formatting: this is the one thing that is different from normal MkDocs

Notebook Markdown is rendered by **nbconvert, not by MkDocs**. The Material extensions used on the plain `.md` pages (`!!! tip`, `??? question`, `=== "Tab"`) are **not applied** in notebooks and will show up as literal text. Use plain HTML instead. Material's stylesheet styles it identically.

**Callout** (kinds: `note`, `info`, `tip`, `success`, `question`, `warning`, `failure`, `danger`, `bug`, `example`, `quote`):

```html
<div class="admonition warning">
<p class="admonition-title">Watch out</p>
<p>Text of the callout.</p>
</div>
```

**Collapsible answer** (note the blank lines, which let Markdown and code fences render inside):

```html
<details class="question">
<summary>Question text. Try it first, then open the answer.</summary>

The answer, in **Markdown**, with code if needed:

```python
df["x"].mean()
```

</details>
```

**Math** works with `$...$` and `$$...$$`. **Tables** and **images** (`![alt](path.png)`, or just plot in a code cell) work normally. Give every figure alt text. **Tabs** are not available: use sub-headings.

## Setup cell and Colab

The second code cell installs the helper if it is missing (this is what runs on Colab). `new_page.py` writes it with the right git ref; `tools/execute_notebooks.py --ref main --write-ref` retargets every notebook's Colab link and install cell (do this when merging to `main`).

## Dependencies

If a page needs a new library, add it to `requirements-notebooks.txt` so CI and students get it. Keep the list small.
