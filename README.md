# Machine Learning Practice (MLP) Course

A course site for learning the whole machine learning project lifecycle on real data. Every page is a Jupyter notebook you can read in the browser, download, or open in Colab. The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Live site: <https://sampriti-sr.github.io/MLP_Course/>

## What is in it

| Part | Folder | Pages | Data |
|------|--------|-------|------|
| Start here | `docs/getting-started/` | How to use the site, learning path, the datasets, environment setup | |
| A. The ML project lifecycle | `docs/lifecycle/` | 14: framing, validation, cleaning, EDA, splitting, baselines, features, modelling, evaluation, tuning, interpretation, serving, monitoring, reproducibility | Online Retail II (one running project) |
| B/C. Toolkit | `docs/toolkit/` | 13: NumPy, pandas, Matplotlib/Seaborn, scikit-learn | Retail, Airbnb, MovieLens, NYC taxi, credit default |
| D. Time series | `docs/timeseries/` | 8: decomposition, baselines, classical and ML forecasting, backtesting, metrics, holidays, regime change | NYC taxi demand |
| E. Recommender systems | `docs/recsys/` | 9: framing, splitting and metrics, baselines, neighbourhood methods, factorisation, implicit retail, cold start, beyond accuracy, two-stage | MovieLens, Online Retail II |
| Resources | `docs/workflows/`, `docs/resources/` | EDA checklist, end-to-end pipeline, common errors, links | |

Planned: exercises and projects, a capstone. See [`PLAN.md`](PLAN.md) for the roadmap and what is done.

---

## For students

You do not need to install anything to read the course. To run the notebooks:

- **In Colab:** every page has an "Open in Colab" badge at the top. The first cell installs what it needs.
- **Locally:**

```bash
git clone https://github.com/sampriti-sr/MLP_Course.git
cd MLP_Course
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-notebooks.txt
jupyter lab docs/
```

Datasets are downloaded from their original publishers the first time a notebook needs them and cached in `~/.cache/mlp_course`. Sizes, licences and caveats are in [`data/README.md`](data/README.md).

---

## For instructors and contributors

### Preview the site

```bash
pip install -r requirements.txt
mkdocs serve                     # http://127.0.0.1:8000, reloads on save
```

### Tested with

| What | Versions |
|------|----------|
| Site build (`requirements.txt`) | mkdocs 1.6.1, mkdocs-material 9.7.7, mkdocs-jupyter 0.26.3, pymdown-extensions 12.3; a fresh venv builds with `mkdocs build --strict` |
| Notebooks | pandas 2.3 and 3.0 (both exercised); Python 3.11 in CI |

`requirements.txt` uses `~=` ranges on these versions, so patch updates are accepted but the next major version is not. `requirements-notebooks.txt` still uses open lower bounds; the CI run on every push is what tells us when a new release breaks a page.

### Add a page

The notebooks in `docs/` are the pages (rendered by `mkdocs-jupyter`). Start from the template:

```bash
python tools/new_page.py docs/lifecycle/15-my-page.ipynb "My Page Title"
```

Then add the file to `nav:` in `mkdocs.yml`. Authoring rules (callouts, exercises, how numbers in prose are checked against output) are in [`templates/AUTHORING.md`](templates/AUTHORING.md). Material syntax such as `!!!` and `===` does **not** work inside notebooks; use the HTML patterns in that guide. Plain `.md` pages (the older reference pages) can still use Material syntax.

### Check your work

```bash
pytest -q                                          # offline unit tests for mlp_data
python tools/execute_notebooks.py --lint           # static checks, no kernel
python tools/execute_notebooks.py --check          # run every notebook from scratch (downloads data)
python tools/execute_notebooks.py --write docs/lifecycle/03-cleaning.ipynb   # run and save outputs
mkdocs build --strict                              # a warning fails the build
```

The site build never runs code: outputs are saved inside the notebooks by `--write`, so re-run and commit a notebook after changing it. CI (`.github/workflows/notebooks.yml`) runs all of the above on every push and pull request.

### Retarget the Colab links when merging

Colab and install links name a git ref (currently the working branch). When merging to `main`:

```bash
python tools/execute_notebooks.py --ref main --write-ref
```

### Deploy

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push to `main`. In the repository settings, set **Pages > Source** to the `gh-pages` branch. Manual deploy: `mkdocs gh-deploy`.

### Repository layout

```text
MLP_Course/
├── docs/                    # the site: notebooks (.ipynb) and a few .md pages
│   ├── getting-started/  lifecycle/  toolkit/  timeseries/  recsys/  workflows/  resources/
│   ├── stylesheets/extra.css
│   └── javascripts/mathjax.js
├── mlp_data/                # tested helper package used by the notebooks
│   ├── loaders.py           #   one-line, cached dataset loaders
│   ├── retail.py            #   cleaning, snapshots and splits for Online Retail II
│   ├── timeseries.py  forecasting.py   # taxi demand, baselines, backtesting
│   ├── recsys.py            #   splits, ranking metrics, kNN / SVD / ALS models, cold-start and diversity helpers
│   └── metrics.py  models.py  pipeline.py  tracking.py  serving.py  style.py
├── tests/                   # offline unit tests
├── tools/                   # execute_notebooks.py, new_page.py
├── templates/               # page template and AUTHORING.md
├── data/README.md           # dataset audit: URLs, licences, checksums
├── mkdocs.yml
├── requirements.txt         # to build the site
├── requirements-notebooks.txt  # to run the notebooks and checks
└── PLAN.md                  # roadmap and decisions
```
