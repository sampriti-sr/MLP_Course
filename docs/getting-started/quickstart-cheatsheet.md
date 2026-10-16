# Quickstart Cheatsheet

A concise, single-page reference for common imports, data loading, basic model training, and key rules.

---

## :material-pin-outline:{ .mlp-icon } Standard Library Import Conventions

Always adhere to standard community naming conventions:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Common Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score
```

---

## :material-timer-outline:{ .mlp-icon } 60-Second Machine Learning Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv("dataset.csv")

# 2. Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# 3. Train-test split (BEFORE any transformation to prevent leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define column transformations
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 5. Build Pipeline: Preprocessing + Model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 6. Fit on training data & evaluate on test data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

## :material-format-list-checks:{ .mlp-icon } The 4 Golden Rules of Machine Learning Practice

!!! warning "Rule #1: Never fit transformers on test data!"
    Always `fit_transform()` on **`X_train`**, but only `transform()` on **`X_test`**. Using `fit` on test or full dataset causes **data leakage**, resulting in falsely optimistic validation scores that fail in production.

!!! tip "Rule #2: Always wrap preprocessing in a Pipeline"
    Manual scaling before `train_test_split` leaks information. Placing scalers and models inside a `Pipeline` guarantees transformations happen strictly inside each cross-validation fold.

!!! note "Rule #3: Check class balance before picking metrics"
    If 95% of records are class `0`, an unlearned dummy model predicting `0` will boast 95% accuracy. Use **F1-Score**, **ROC-AUC**, and **PR-AUC** for imbalanced datasets.

!!! tip "Rule #4: Fix random seeds for reproducibility"
    Always pass `random_state=42` (or any integer) in `train_test_split`, cross-validators, and stochastic algorithms (`RandomForest`, `KMeans`, etc.) so lab results are reproducible.
