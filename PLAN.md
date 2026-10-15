# MLP Course: Improvement Plan (v2)

**Goal:** turn the site from a syntax cheat-sheet into a resource a student can *learn the whole ML project lifecycle from*: read in order, run every snippet on real data, see real output, practise, and check understanding.

**Decisions locked in**

| # | Decision |
|---|----------|
| Scope | Entire ML project lifecycle · classical ML · **time series** · **recommender systems** |
| Audience | Knows Python **and the maths behind the algorithms**; new to *applying* them to messy real data |
| Data | **Real-world datasets**, not toy ones, used as connected case studies |
| Notebooks | Downloadable / "Open in Colab" for every page |
| Workflow | Work on a branch of `sampriti-sr/MLP_Course` (we have write access); **one commit per task**; no merge to `main` without review |

## Progress

| Phase | Status | Notes |
|---|---|---|
| 0. Quick fixes | **Not started** | polyfill.io is still in `mkdocs.yml`; repo URL, README and CSS fixes pending. `mkdocs<2` pin was pulled forward into Phase 1. |
| 1. Foundations | **Done** (branch `kshitij-variant`) | See below. |
| 2. Lifecycle (Part A) | **Done** (14 pages) | See below. |
| 3-7 | Not started | Phase 3 (reference pages on real data), Part D (time series), Part E (recommenders), exercises/projects, release. |

**Phase 1 outcomes**

| Task | Result |
|---|---|
| 1.1 Dataset audit | `data/README.md`: all five candidates verified (URL, licence, size, schema, quality claims, checksums). Open items: TLC states no licence; Airbnb snapshots rotate. |
| 1.2 Loader | `mlp_data` package: one-line loaders, cache, checksum warnings, safe zip handling, certifi for macOS. Cold and warm runs tested. |
| 1.3 Template | `templates/page-template.ipynb`, `templates/AUTHORING.md`, `tools/new_page.py`. |
| 1.4 Notebooks + CI | Notebooks in `docs/` are the pages (mkdocs-jupyter). `tools/execute_notebooks.py` lints and runs them; `.github/workflows/notebooks.yml` runs it on every push/PR. Notebooks tested on pandas 2.3 and 3.0. |
| 1.5 Orientation pages | "How to use this site", "Learning path", and the first real page, "The Datasets". |

**Phase 2 outcomes (Part A, the lifecycle on the retail data)**

Fourteen executed, tested notebooks in `docs/lifecycle/`, one running project ("which customers will buy again in the next 90 days?"), each page ending in exercises with hidden answers. Supporting, tested code: `mlp_data.retail` (cleaning, snapshots, splits), `metrics`, `models`, `tracking`, `serving`, `pipeline`, `style` (45 offline unit tests).

What the real data taught (all asserted in the notebooks, not just stated):

- The 34,335 "duplicates" are 22,523 overlap rows (two Excel sheets overlapping) plus 11,812 genuine repeats.
- Random splits flatter a *flexible* model on rich features by ~+0.07 AUC; a simple model barely notices.
- The accuracy-optimised rankings protected *less* revenue than random calling; the decision rule mattered far more than the model.
- Extra features and tuning added almost nothing; no model was distinguishable from another on money.
- The test verdict: ranking held, probabilities collapsed (predicted 32%, actual 50%) in the autumn peak; the model is only on par with a free rule.
- Monitoring: input drift (PSI) did not see the shift; an early 30-day partial-label rate did, ~60 days before the full outcome.

**A defect found late, and fixed at the source.** Reading the final call list showed rank 1 was a customer whose GBP 77k order had been cancelled sixteen minutes later. Page 3 had tested only whether such orders flipped labels (they did not) and called them harmless; they distort *spend*, which a value-ranked list depends on. Fixed in `clean_retail` (step 6, with tests), and every page re-run and its prose corrected where a conclusion moved (the extra features' money gain and boosting's edge over the forest both disappeared; the model's lead over the rule shrank to "not demonstrable"). Lesson recorded on Page 3: *test a cleaning decision against every way the data is used.*

**Design decisions made during Phase 1 (they change later phases)**

- **Notebooks are the pages**, not companions to Markdown pages. A spike showed notebook Markdown is rendered by nbconvert, so Material's `!!!`, `???` and `===` do not work in notebooks. Pages use HTML callouts and `<details>` instead (verified in a browser). Tabs are not available in notebooks. The older `.md` reference pages are unaffected. Details in `templates/AUTHORING.md`.
- **MkDocs is pinned `<2`**: Material warns that MkDocs 2.0 removes the plugin system.
- **Colab/install links name a git ref** (currently `kshitij-variant`). At merge time run `python tools/execute_notebooks.py --ref main --write-ref`.
- Raw data is never committed; loaders download from the original publishers. A GitHub Release mirror remains an open item (Airbnb link rot, 35 s Excel parse).

**Principle:** since students already know the maths, pages do **not** re-derive algorithms. Each page answers: *When do I use this? How do I apply it correctly to real data? What goes wrong in practice? How do I know it worked?* A short "Maths ↔ code" box links each equation to the line that implements it.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done

---

## 1. Real-world data strategy

Real data brings real constraints, so this comes first.

**Selection criteria:** (a) permissive licence we are allowed to redistribute or link; (b) reachable without a login (Kaggle-only data breaks Colab flow); (c) small enough for free Colab (<~500 MB raw, ideally sampled to <50 MB for the repo/release); (d) genuinely messy; (e) supports a *question a business would ask*.

**Proposed case-study spine** (candidates, **none verified yet**: Task 1.1 checks current URLs, licences and sizes before we commit to any):

| Track | Candidate dataset | Why it teaches well | Concepts it carries |
|---|---|---|---|
| **Retail thread** (main spine) | UCI **Online Retail II** (UK e-commerce transactions) | One dataset, four problems: churn/propensity, customer segmentation (RFM), demand forecasting, "customers also bought" | Cleaning (cancellations, nulls, duplicates), feature engineering from transactions, **time-aware splits**, clustering, hierarchical forecasting, item-based recommenders |
| **Time-series** | NYC **TLC taxi trips** (monthly parquet) aggregated to hourly demand; weather from a free open API as exogenous feature | Strong seasonality, holidays, anomalies (e.g. the 2020 shock = regime change) | Decomposition, stationarity, ARIMA/ETS, lag-feature GBMs, backtesting, drift |
| **Recommender** | **MovieLens** (explicit ratings) + retail purchases (implicit feedback) | Contrasts explicit vs implicit feedback on real sparsity | Collaborative filtering, matrix factorisation, ranking metrics, cold start |
| **Tabular, messy, with geography/text** | **Inside Airbnb** listings (one city) | Price regression with outliers, missing data, text and lat/long | Feature engineering, regression diagnostics, interpretation |
| **Imbalanced classification** | A public credit-default or fraud dataset (to be chosen in 1.1) | Class imbalance, cost-sensitive thresholds | PR curves, calibration, threshold choice, fairness check |

**Delivery mechanics:** `data/README.md` records source, licence, date fetched and column dictionary for every dataset. Large files are *not* committed: a `fetch_data.py` helper downloads, caches and (where licence allows) uses a fixed sample so every student gets identical numbers. The same helper runs in Colab.

---

## 2. Revised site structure

```
Start here
  ├─ How to use this site · Learning path · Environment & Colab setup
  └─ Datasets (the case studies)
Part A: The ML project lifecycle   (retail thread)
  1 Problem framing & success metrics → 2 Data acquisition & validation
  → 3 Cleaning → 4 EDA → 5 Splitting strategy (random / grouped / time-based)
  → 6 Baselines → 7 Feature engineering → 8 Modelling → 9 Evaluation & error analysis
  → 10 Tuning & experiment tracking → 11 Interpretation & fairness
  → 12 Packaging & serving → 13 Monitoring & drift → 14 Reproducibility & project hygiene
Part B: Core toolkit               (NumPy · pandas · plotting · sklearn API/pipelines) — reference + exercises
Part C: Classical ML in practice   (linear/regularised · trees & boosting · SVM/KNN · imbalance · clustering/PCA)
Part D: Time series                (NYC taxi)
Part E: Recommender systems        (MovieLens + retail)
Part F: Capstone & projects
Resources: debugging guide · glossary · cheatsheets · further reading
```

The existing NumPy/pandas/sklearn pages are **kept and upgraded**, not thrown away. They move into Part B and are rewritten on real data.

---

## 3. Phased task list

### Phase 0: Quick fixes (about 1 hr; no content decisions)

| # | Task | Done when |
|---|------|-----------|
| 0.1 | Create branch `content-overhaul` | Branch exists |
| 0.2 | Remove `polyfill.io` from `mkdocs.yml` (compromised domain; MathJax 3 doesn't need it) | No request to polyfill.io |
| 0.3 | Set `repo_url`, `site_url`, real `site_author`, `edit_uri` | GitHub link + "edit this page" work |
| 0.4 | README: replace `<your-repo-url>`, separate student vs instructor sections | Reads correctly to both |
| 0.5 | Pin versions in `requirements.txt`; add "tested with" versions table | Fresh venv builds site |
| 0.6 | CSS: wrap/scroll wide code blocks cleanly; enlarge Mermaid text | Nothing clipped |
| 0.7 | CI: `mkdocs build --strict` | CI fails on a broken link |
| 0.8 | Add LICENSE + CONTRIBUTING | Present |

### Phase 1: Foundations: data, template, tooling

| # | Task | Done when |
|---|------|-----------|
| 1.1 | **Dataset audit**: verify URL, licence, size, login-free access, redistributability for every candidate; finalise the list *with you* | Signed-off table in `data/README.md` |
| 1.2 | `fetch_data.py` + caching + fixed samples; test in Colab and locally | One-line load works everywhere |
| 1.3 | Page template (below) + shared setup snippet | Template in repo |
| 1.4 | Notebook pipeline: `.ipynb` per page, "Open in Colab" badge, CI executes notebooks | CI fails if a notebook errors |
| 1.5 | "How to use this site" + learning-path page | Linked from home |

**Page template**

```
# Title
Objectives (3) · Prerequisites (links) · Dataset used
## The situation        (a real question from the case study)
## Maths ↔ code         (equation → the line that implements it; no re-derivation)
## Worked example       (real data, real output and plots)
## What goes wrong      (pitfalls found *in this dataset*, with evidence)
## Decision guide       (when to use / not use; alternatives)
## Check yourself       (quiz, collapsible answers)
## Exercises            (2-3, collapsible solutions)
## Recap · Next
Open in Colab ▸
```

### Phase 2: Part A, the lifecycle (retail thread). Highest-value content

One page + notebook each, in order: 2.1 Problem framing & metrics · 2.2 Data acquisition & validation (schema/range checks) · 2.3 Cleaning (cancellations, duplicates, nulls) · 2.4 EDA with interpretations · 2.5 **Splitting strategy** (random vs grouped vs time-based; show leakage inflating scores) · 2.6 Baselines · 2.7 Feature engineering (RFM, aggregates, windows) · 2.8 Modelling · 2.9 Evaluation & error analysis · 2.10 Tuning & experiment tracking · 2.11 Interpretation & fairness · 2.12 Packaging & serving (joblib → small API) · 2.13 Monitoring & drift · 2.14 Reproducibility & hygiene.

### Phase 3: Part B/C, rework existing pages on real data

| # | Task |
|---|------|
| 3.1 | NumPy (3 pages): vectorisation benchmark, broadcasting shape drills |
| 3.2 | pandas (3 pages): real cleaning and merge/groupby on retail data |
| 3.3 | Matplotlib/Seaborn: every plot shown; "which chart for which question" |
| 3.4 | sklearn core + Pipelines: leakage demonstrated with real numbers |
| 3.5 | Models page → split into: regularised linear · trees/boosting · SVM/KNN, each with *when it fails on this data* |
| 3.6 | Evaluation: confusion matrix, PR vs ROC, calibration, threshold selection, imbalance, cost-sensitive |
| 3.7 | Unsupervised: clustering (RFM segments) + PCA, how to *validate* clusters |

### Phase 4: Part D, time series (NYC taxi)

4.1 Time-series data handling (datetime index, resampling, missing timestamps, time zones) · 4.2 Decomposition & stationarity · 4.3 Baselines (naïve, seasonal naïve) · 4.4 ETS/ARIMA/SARIMAX · 4.5 ML on lag/rolling/calendar features · 4.6 **Backtesting** (`TimeSeriesSplit`, rolling origin) · 4.7 Forecast metrics (MAE, RMSE, MAPE pitfalls, MASE) & prediction intervals · 4.8 Holidays, exogenous variables, regime change (2020), drift.

### Phase 5: Part E, recommender systems

5.1 Problem framing: explicit vs implicit feedback, sparsity · 5.2 Popularity & item-item baselines · 5.3 Neighbourhood CF · 5.4 Matrix factorisation (SVD/ALS) · 5.5 Implicit feedback on retail purchases · 5.6 **Ranking metrics** (Precision/Recall@k, MAP, NDCG), correct train/test splitting for recsys · 5.7 Cold start & content features · 5.8 Beyond accuracy (coverage, diversity, popularity bias) · 5.9 Two-stage retrieval → ranking overview.

### Phase 6: Practice & assessment

6.1 Exercise bank (≥3 per page, solutions hidden) · 6.2 Quizzes incl. "predict the output" and "spot the leakage" · 6.3 Three mini-projects with rubrics (one tabular, one forecasting, one recommender) · 6.4 Capstone brief + rubric · 6.5 Debugging guide v2 (10 → ~25 entries incl. time-series and recsys failures, how to read a traceback).

### Phase 7: Polish & release

7.1 Glossary · 7.2 Printable cheatsheets · 7.3 Curated further reading (one book chapter / paper / video per topic with a one-line "why") · 7.4 Accessibility + mobile pass · 7.5 "Was this helpful?" feedback + issue template · 7.6 Clean-machine run of every notebook · 7.7 **Pilot with 2-3 real students** on one module, revise · 7.8 Merge to `main`, tag release.

---

## 4. Suggested order and effort

| Phase | Effort | Notes |
|---|---|---|
| 0 | ~1 hr | Start immediately |
| 1 | ~2-3 days | Dataset audit is the real work; everything depends on it |
| 2 | ~1.5 weeks | The core differentiator |
| 3 | ~1 week | Mostly rewrites of existing pages |
| 4 | ~1 week | |
| 5 | ~1 week | |
| 6 | ~1 week | Can run in parallel with 4-5 |
| 7 | ~3-4 days | |

Rough total: **6-8 weeks** for the full vision. **Minimum viable (about 2-3 weeks):** Phases 0-1, lifecycle pages 2.3-2.5 and 2.9, the leakage demo (3.4), one time-series page on backtesting (4.6), and one recommender page on ranking metrics (5.6). That covers the topics where students most often go wrong.

## 5. How we'll execute

- Branch `content-overhaul`; **one task = one commit** with a clear message; box ticked here in the same commit.
- For each task I show you the diff (and a rendered screenshot for visual pages) before committing.
- I run `mkdocs build --strict` and execute each notebook before calling a task done.
- Nothing is pushed or merged to `main` without your explicit go-ahead.

## 6. Risks

- **Data availability/licensing changes**, so Task 1.1 must verify rather than assume.
- **Notebook maintenance** as library versions move, mitigated by pinned versions + CI execution.
- **Scope**, so the minimum-viable cut above is the fallback.
- **Real data is messy** and may give "boring" model results. That is a teaching feature, but each page must say honestly what the result means.

## 7. Open questions

1. Retail thread as the spine: OK, or is there a domain closer to your students' interests (finance, healthcare, mobility)?
2. Should deployment (2.12) stay light (joblib + a small API) or go further (Docker, cloud)?
3. Do you want the deep-learning-based recommenders and forecasters (e.g. neural CF, LSTM/Transformer) covered, or kept classical only?
4. Who is the first reviewer for each phase, you or the repo owner as well?
