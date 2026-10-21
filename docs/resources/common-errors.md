# Top 10 Common Mistakes & Debugging Guide

A rapid diagnostic guide for the most frequent runtime errors and logical bugs students encounter during Machine Learning Practice labs.

---

## 1. `ValueError: Found input variables with inconsistent numbers of samples`
- **Symptom:** `ValueError: Found input variables with inconsistent numbers of samples: [1000, 950]`
- **Root Cause:** $X$ and $y$ have differing numbers of rows (usually caused by filtering or dropping `NaN` from $X$ without applying the same filter to $y$).
- **Solution:** Always drop missing values or filter rows from the combined DataFrame **before** separating $X$ and $y$:
  ```python
  # Fix:
  df = df.dropna(subset=["critical_column"])
  X = df.drop(columns=["target"])
  y = df["target"]
  ```

---

## 2. `ValueError: Expected 2D array, got 1D array instead`
- **Symptom:** Calling `.predict()` or `StandardScaler.fit()` on a single 1D feature array crashes.
- **Root Cause:** Scikit-Learn estimators expect feature matrices to be 2-dimensional `(n_samples, n_features)`.
- **Solution:** Reshape with `.reshape(-1, 1)` or slice using double brackets `df[['feature']]`:
  ```python
  # Fix for NumPy:
  single_feature_2d = feature_arr.reshape(-1, 1)

  # Fix for Pandas:
  single_feature_df = df[["single_col"]]  # Note double brackets!
  ```

---

## 3. Data Leakage (The "Overly Perfect" Test Score)
- **Symptom:** Your model gets 99.8% accuracy in cross-validation, but plummets to 65% when tested on new data.
- **Root Cause:** Scalers, imputers, or target statistics were computed on the entire dataset prior to `train_test_split`.
- **Solution:** Encapsulate transformers inside a `Pipeline`:
  ```python
  # Fix:
  pipeline = Pipeline([
      ("scaler", StandardScaler()),
      ("model", LogisticRegression())
  ])
  cross_val_score(pipeline, X_train, y_train, cv=5)
  ```

---

## 4. `ConvergenceWarning: lbfgs failed to converge`
- **Symptom:** `ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.`
- **Root Cause:** Optimization solver did not reach minimum before running out of iterations. Typically happens when features are not scaled or `max_iter` is too low.
- **Solution:** Scale numerical features with `StandardScaler` and increase `max_iter`:
  ```python
  # Fix:
  model = LogisticRegression(max_iter=1000)
  ```

---

## 5. `ValueError: Found unknown categories in column ... during transform`
- **Symptom:** `OneHotEncoder` crashes when evaluating test data because a new unseen category appeared.
- **Root Cause:** Default `OneHotEncoder(handle_unknown='error')` raises an error on new categories.
- **Solution:** Set `handle_unknown="ignore"`:
  ```python
  # Fix:
  encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  ```

---

## 6. `SettingWithCopyWarning` in Pandas
- **Symptom:** `A value is trying to be set on a copy of a slice from a DataFrame.`
- **Root Cause:** Assigning values to a chained slice (e.g. `df[df['a'] > 5]['b'] = 10`).
- **Solution:** Use `.loc[]` or explicitly call `.copy()`:
  ```python
  # Fix:
  df.loc[df["a"] > 5, "b"] = 10
  ```

---

## 7. The Accuracy Paradox on Imbalanced Classes
- **Symptom:** The model achieves 96% accuracy, but has 0% recall on the positive minority class (e.g. fraud detection).
- **Root Cause:** When 96% of labels are negative, predicting negative for all samples yields 96% accuracy.
- **Solution:** Use `f1_score`, `roc_auc_score`, or `classification_report`, and enable `class_weight='balanced'`:
  ```python
  # Fix:
  model = RandomForestClassifier(class_weight="balanced")
  print(classification_report(y_test, y_pred))
  ```

---

## 8. Forgetting to Scale Before Distance / Kernel Algorithms
- **Symptom:** KNN, SVM, or PCA performance is completely dominated by one feature with a huge numerical range (e.g. income 100,000 vs age 25).
- **Root Cause:** Euclidean distance gives $(100000)^2$ vastly more weight than $(25)^2$.
- **Solution:** Always use `StandardScaler` or `MinMaxScaler` prior to distance-based estimators.

---

## 9. Dimension Broadcasting Errors in NumPy
- **Symptom:** `ValueError: operands could not be broadcast together with shapes (100, 3) (100,)`
- **Root Cause:** Subtracting a 1D vector `(100,)` from a 2D matrix column-wise instead of row-wise.
- **Solution:** Ensure correct trailing dimensions with `np.newaxis`:
  ```python
  # Fix:
  diff = matrix_100x3 - vector_100[:, np.newaxis]
  ```

---

## 10. `NotFittedError: This instance is not fitted yet`
- **Symptom:** Calling `.predict()` or `.transform()` before `.fit()`.
- **Root Cause:** Attempting inference on an uninitialized or newly reinstantiated model.
- **Solution:** Verify that `.fit(X_train, y_train)` has completed before `.predict()`.
