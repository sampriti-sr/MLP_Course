# Scikit-Learn: Preprocessing & Encoders

Raw tabular data typically features diverse scales, categorical strings, and missing values. Preprocessing transforms these features into numerical representations suitable for machine learning algorithms.

---

## ⚖️ Numerical Scalers: Comparison & Selection

| Scaler | Formula | Sensitive to Outliers? | Best Used For |
| :--- | :--- | :---: | :--- |
| **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ | Yes | Linear models, Logistic regression, SVMs, Neural networks |
| **`MinMaxScaler`** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Highly | Image pixel values $[0, 1]$, algorithms requiring positive inputs |
| **`RobustScaler`** | $x' = \frac{x - \text{median}}{\text{IQR}}$ | **No (Outlier-safe)** | Datasets with heavy tails or uncleaned extreme values |

=== "`StandardScaler` (Zero Mean, Unit Variance)"

    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

=== "`MinMaxScaler` (Bounded [0, 1])"

    ```python
    from sklearn.preprocessing import MinMaxScaler

    minmax = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = minmax.fit_transform(X_train)
    X_test_scaled = minmax.transform(X_test)
    ```

=== "`RobustScaler` (Median & Interquartile Range)"

    ```python
    from sklearn.preprocessing import RobustScaler

    robust = RobustScaler()
    X_train_scaled = robust.fit_transform(X_train)
    X_test_scaled = robust.transform(X_test)
    ```

---

## 🔠 Categorical Encoders

=== "`OneHotEncoder` (Nominal / Unordered Categories)"

    Converts each distinct category into a binary column ($0$ or $1$). Always pass `handle_unknown="ignore"` to handle unseen categories during test time without crashing.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(
        handle_unknown="ignore", # Returns all zeros for unseen test categories
        sparse_output=False      # Returns a dense NumPy array instead of scipy sparse matrix
    )

    cats = [["Red"], ["Green"], ["Blue"], ["Green"]]
    encoded = ohe.fit_transform(cats)
    print(ohe.get_feature_names_out())
    # ['x0_Blue' 'x0_Green' 'x0_Red']
    ```

=== "`OrdinalEncoder` (Ordered Categories)"

    Encodes rank-ordered categories into integer levels (e.g. Low < Medium < High).

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    education_levels = [["High School"], ["BSc"], ["MSc"], ["PhD"]]
    encoder = OrdinalEncoder(categories=[["High School", "BSc", "MSc", "PhD"]])
    
    encoded = encoder.fit_transform([["BSc"], ["PhD"]])
    # [[1.], [3.]]
    ```

=== "`TargetEncoder` (High Cardinality)"

    Replaces category levels with the expected target value for that category. Includes shrinkage regularization to prevent overfitting.

    ```python
    from sklearn.preprocessing import TargetEncoder

    # Ideal when a column has 100+ unique zip codes or cities
    te = TargetEncoder(smooth="auto", cv=5)
    X_train_encoded = te.fit_transform(X_train_cat, y_train)
    X_test_encoded = te.transform(X_test_cat)
    ```

---

## 🩹 Missing Value Imputation

=== "`SimpleImputer` (Univariate)"

    ```python
    from sklearn.impute import SimpleImputer

    # Numerical imputation with median
    num_imputer = SimpleImputer(strategy="median")
    X_num_imp = num_imputer.fit_transform(X_num)

    # Categorical imputation with most frequent (mode)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_cat_imp = cat_imputer.fit_transform(X_cat)
    ```

=== "`KNNImputer` (Multivariate Distance-Based)"

    Imputes missing values using the mean of the $k$-nearest neighbors based on other available features.

    ```python
    from sklearn.impute import KNNImputer

    knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_imputed = knn_imputer.fit_transform(X_scaled)
    ```
