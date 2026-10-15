# How to Use This Site

This course teaches the **whole machine-learning project lifecycle on real data**: from a vague business question to a model you can defend, ship and monitor. It also covers time-series forecasting and recommender systems.

It assumes you already know Python and the mathematics behind the main algorithms. So pages do **not** re-derive them. They show you how to *apply* methods to messy data, what goes wrong, and how to tell.

---

## Read it in order, run it as you read

Course pages are **notebooks**. Each one shows code *and* its real output. You get the most from a page by running it yourself and changing things.

=== "Google Colab (easiest)"

    1. Click the **Open in Colab** badge at the top of any course page.
    2. Run the first code cell. It installs the course helper for you.
    3. Run the rest top to bottom.

    Colab has no setup and no installs, but datasets are re-downloaded each new session.

=== "On your own computer"

    ```bash
    git clone https://github.com/sampriti-sr/MLP_Course
    cd MLP_Course
    python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements-notebooks.txt
    jupyter lab docs/
    ```

    Datasets are cached in `~/.cache/mlp_course`, so each one downloads once.

!!! tip "Load any dataset in one line"
    ```python
    from mlp_data import load_retail, load_movielens, load_taxi, load_airbnb, load_credit
    retail = load_retail()
    ```
    See [The Datasets](datasets.ipynb) for what each one is and what is wrong with it.

---

## What a page looks like

Every page follows the same shape, so you always know where to look:

| Section | What it is for |
|---|---|
| **What you will be able to do** | The objectives, as things you can *test* yourself on |
| **The situation** | A real question we are trying to answer with the data |
| **Maths ↔ code** | Which line of code implements which formula you already know |
| **Worked example** | The method on real data, with real output and a plain-words reading of it |
| **What goes wrong** | Pitfalls that actually occur in *this* data, shown with evidence |
| **Decision guide** | When to use the method, when not to, and the alternatives |
| **Check yourself** and **Exercises** | Try first, then open the hidden answer |
| **Recap** | The few things to remember |

## Exercises: try before you peek

Answers are folded away on purpose. Open one only after you have tried the question, because struggling with a problem for five minutes teaches more than reading its solution.

## When something breaks

1. **Download failed or shows a certificate error:** the message says why and what to try. On macOS with Python from python.org, run *Install Certificates.command* once, or `pip install -U certifi`.
2. **A number differs slightly from the page:** the publisher may have updated a dataset (you will see a checksum warning), or your library version differs. Small differences in display are normal; large ones are worth investigating.
3. **A page's code errors on a clean install:** that is a bug in the course. Please open an issue on the repository with the page name and the error.
4. **Stuck on a concept:** see the [debugging guide](../resources/common-errors.md) and the [learning path](learning-path.md) for prerequisites.

## The reference pages

The library pages (NumPy, pandas, Matplotlib/Seaborn, scikit-learn) are a quick **lookup reference**, not a teaching sequence. Use them when you know what you want and need the syntax. The lifecycle chapters in the [learning path](learning-path.md) are where the teaching happens.
