# Scikit-Learn: Models & Estimators

Scikit-learn provides algorithms across supervised classification, regression, and unsupervised clustering. Choosing the right algorithm depends on sample size, interpretability requirements, and feature characteristics.

---

## 🧭 Model Selection Decision Matrix

| Model Family | Key Algorithms | Best Suited For | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Linear Models** | `Ridge`, `Lasso`, `LogisticRegression` | Baseline benchmarks, high-dimensional text data, interpretability | ➕ Fast, explainable weights<br>➖ Assumes linear feature relationships |
| **Tree Ensembles** | `RandomForest`, `HistGradientBoosting` | Tabular data, non-linear boundaries, mixed feature scales | ➕ Top tabular performance, handles outliers<br>➖ Less interpretable, slower inference |
| **Distance-Based** | `KNeighbors`, `SVC` | Small-to-medium clean datasets, non-linear geometric boundaries | ➕ Effective in high dimensions with RBF kernels<br>➖ Requires feature scaling; $O(N^2)$ to $O(N^3)$ complexity |
| **Probabilistic** | `GaussianNB`, `MultinomialNB` | Text classification (spam detection), bag-of-words | ➕ Extremely fast, minimal memory<br>➖ Strong feature independence assumption |

---

## 📈 1. Linear Models & Regularization

Regularization penalizes large coefficients to prevent overfitting:

- **Ridge Regression ($L_2$ Regularization):** Shrinks weights toward zero; handles collinearity.
- **Lasso Regression ($L_1$ Regularization):** Drives unimportant weights strictly to zero (automatic feature selection).
- **Logistic Regression:** Log-odds linear boundary for binary and multiclass probabilities.

```python
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

# Ridge Regression (alpha is the penalty strength lambda)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression for sparse feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
zero_coef_count = np.sum(lasso.coef_ == 0)

# Logistic Regression for Classification
log_reg = LogisticRegression(
    penalty="l2",
    C=1.0,               # Inverse regularization strength: smaller C = stronger penalty
    max_iter=1000,       # Increase if 'ConvergenceWarning' occurs
    random_state=42
)
log_reg.fit(X_train, y_train)
```

---

## 🌲 2. Tree-Based Models & Ensembles

Tree algorithms partition feature space with orthogonal decision boundaries. They do **not** require feature scaling.

=== "Random Forest (Bagging)"

    Averages predictions across hundreds of de-correlated decision trees:

    ```python
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=200,      # Number of decision trees
        max_depth=10,          # Depth limit to curb overfitting
        min_samples_split=5,   # Minimum samples required to split an internal node
        min_samples_leaf=2,    # Minimum samples required at leaf node
        max_features="sqrt",   # Features considered per split
        n_jobs=-1,             # Use all CPU cores in parallel
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Feature Importance analysis
    importances = rf.feature_importances_
    ```

=== "`HistGradientBoosting` (Boosting)"

    Scikit-learn's modern histogram-based boosting algorithm (inspired by LightGBM). Extremely fast on large datasets with native support for missing values:

    ```python
    from sklearn.ensemble import HistGradientBoostingClassifier

    hgb = HistGradientBoostingClassifier(
        max_iter=150,          # Equivalent to n_estimators (number of boosting rounds)
        learning_rate=0.08,    # Shrinkage factor
        max_leaf_nodes=31,
        early_stopping=True,   # Stops automatically when validation score plateaus
        random_state=42
    )
    hgb.fit(X_train, y_train)
    ```

---

## 📐 3. Support Vector Machines & K-Nearest Neighbors

!!! warning "Crucial Requirement"
    Distance-based algorithms **strictly require feature scaling** (such as `StandardScaler`). Features with larger numerical ranges will otherwise dominate distance calculations.

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Classifier with Non-linear RBF Kernel
svm = SVC(
    C=1.0,             # Margin slack penalty
    kernel="rbf",      # Radial Basis Function
    gamma="scale",     # Kernel coefficient
    probability=True   # Enables .predict_proba() output
)

# K-Nearest Neighbors
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance", # Closer neighbors contribute higher weight
    metric="minkowski", # p=2 is standard Euclidean distance
    p=2
)
```
