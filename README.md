# Machine Learning Practice (MLP) Course Documentation

A modern, responsive, and searchable course documentation website and reference portal for students taking **Machine Learning Practice (MLP)**.

Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), this repository gives students a single, clean place to look up API references, code patterns, EDA workflows, and debugging tips across:

- **NumPy**: Vectorization, broadcasting rules, matrix math, array indexing.
- **Pandas**: DataFrames, cleaning missing data, types, groupby aggregations, merges.
- **Matplotlib & Seaborn**: Object-Oriented plotting, subplots, statistical distributions, heatmaps.
- **Scikit-Learn**: Estimator design, preprocessing scalers/encoders, `ColumnTransformer`, `Pipeline`, model evaluation & hyperparameter tuning.
- **Practical Workflows**: Step-by-step EDA checklist and an end-to-end runnable tabular ML pipeline template.
- **Debugging Guide**: The top 10 most common runtime errors and pitfalls with immediate fixes.

---

## Quickstart: Running Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd MLP_Course
```

### 2. Set up virtual environment & install requirements
```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start local live-reloading dev server
```bash
mkdocs serve
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. Any edits you make to the `.md` files in `docs/` will automatically refresh the browser in real time!

---

## Deploying to GitHub Pages (Free Hosting)

### Method 1: Automatic via GitHub Actions (Recommended)
This repository includes `.github/workflows/deploy.yml`. 
1. Push this repo to GitHub (`main` branch).
2. Go to your repository settings on GitHub: **Settings > Pages > Build and deployment**.
3. Under **Source**, select **GitHub Actions** (or `gh-pages` branch if deploying manually).
4. Every push to `main` will build and publish your course website automatically!

### Method 2: One-Command CLI Deployment
With your virtual environment active and Git configured:
```bash
mkdocs gh-deploy
```

---

## Repository Directory Structure

```text
MLP_Course/
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD automated deployment
├── mkdocs.yml                      # MkDocs Material theme & site configuration
├── requirements.txt                # Python documentation dependencies
├── README.md                       # Instructor & student guide
└── docs/                           # Markdown content files
    ├── index.md                    # Course portal homepage
    ├── getting-started/
    │   ├── environment-setup.md    # venv, conda, VS Code, health-check script
    │   └── quickstart-cheatsheet.md# 1-page survival cheatsheet & golden rules
    ├── libraries/
    │   ├── numpy/
    │   │   ├── index.md            # ndarray architecture, memory layout
    │   │   ├── arrays-indexing.md  # Creation, slicing, boolean masking, reshape
    │   │   └── vectorization-math.md # Broadcasting, axis stats, matrix multiplication
    │   ├── pandas/
    │   │   ├── index.md            # Series/DataFrame, .loc vs .iloc, SettingWithCopy
    │   │   ├── data-cleaning.md    # Missing values, type fixing, duplicates, filtering
    │   │   └── manipulation.md     # GroupBy named agg, merge, concat, pivot
    │   ├── visualization/
    │   │   ├── matplotlib.md       # Figure vs Axes OO interface, subplots, export
    │   │   └── seaborn.md          # Distributions, pairplots, correlation & CM heatmaps
    │   └── scikit-learn/
    │       ├── index.md            # Estimators, Transformers, Predictors API design
    │       ├── preprocessing.md    # Scalers (Standard/MinMax/Robust), Encoders, Imputers
    │       ├── pipelines.md        # ColumnTransformer, leak-free Pipelines, CV tuning
    │       ├── models.md           # Model taxonomy (Linear, Trees, Ensembles, SVM, KNN)
    │       └── evaluation-tuning.md# Stratified splits, ROC-AUC, F1, Grid/Random Search
    ├── workflows/
    │   ├── eda-checklist.md        # 7-step rigorous EDA workflow
    │   └── end-to-end-ml.md        # Full runnable tabular pipeline template with joblib export
    ├── resources/
    │   ├── common-errors.md        # Top 10 runtime & logic errors with fixes
    │   └── external-links.md       # Official docs, printable cheatsheets, visualizers
    └── stylesheets/
        └── extra.css               # Clean typography, badge, and card styles
```

---

## Adding New Topics or Lab Assignments

To add a new lecture or lab module:
1. Create a new markdown file inside `docs/` (e.g., `docs/labs/lab-01-data-cleaning.md`).
2. Add the path to `mkdocs.yml` under the `nav:` section.
3. Save the file; `mkdocs serve` will reload instantly.
