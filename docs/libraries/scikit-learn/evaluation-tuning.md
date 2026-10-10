# Scikit-Learn: Evaluation & Tuning

Evaluating machine learning models requires more than reporting simple training accuracy. Proper validation techniques and metric selection protect against overfitting and deceptive evaluation.

---

## ✂️ Data Splitting Best Practices

```python
from sklearn.model_selection import train_test_split

# 1. Stratified split (preserves target class proportion in both train and test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # CRITICAL for classification datasets!
)
```

---

## 🔁 Cross-Validation

Cross-validation computes performance over multiple disjoint test folds to reduce variance.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Configure 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using specific metric (e.g. 'roc_auc', 'f1_macro', 'neg_mean_squared_error')
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

print(f"5-Fold F1 Macro: {scores.mean():.3f} (± {scores.std():.3f})")
```

---

## 🎯 Classification Evaluation Metrics

| Metric | Formula | When to Prioritize |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Balanced classes only (dangerous if imbalanced) |
| **Precision** | $\frac{TP}{TP + FP}$ | When False Positives are costly (e.g. spam filter, fraud alert) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | When False Negatives are catastrophic (e.g. cancer diagnosis, defect detection) |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance of Precision and Recall on skewed classes |
| **ROC-AUC** | Area under TPR vs FPR curve | Evaluates model discrimination across all decision thresholds |

```python
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

# Detailed precision, recall, and f1 breakdown per class
print(classification_report(y_test, y_pred, digits=3))

# Area Under the ROC Curve (requires prediction probabilities)
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc_score:.4f}")

# Plotting ROC Curve directly
RocCurveDisplay.from_estimator(model, X_test, y_test)
```

---

## 📏 Regression Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Ground truth y_test vs predictions y_pred
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f} (Average absolute error in target units)")
print(f"RMSE: {rmse:.2f} (Penalizes large errors heavily)")
print(f"R²:   {r2:.3f} (Proportion of variance explained by model)")
```

---

## 🔍 Hyperparameter Tuning: Grid vs Random Search

=== "`GridSearchCV` (Exhaustive Search)"

    Tests every single parameter combination. Best for small parameter grids (under 50 combinations).

    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"]
    }

    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    ```

=== "`RandomizedSearchCV` (Sampling Distributions)"

    Samples random configurations from probability distributions. Dramatically faster and often finds better optima across large hyperparameter spaces.

    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform

    param_dist = {
        "classifier__n_estimators": randint(50, 300),
        "classifier__max_depth": randint(3, 15),
        "classifier__learning_rate": uniform(0.01, 0.2)
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,             # Number of random parameter sets to evaluate
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    ```
