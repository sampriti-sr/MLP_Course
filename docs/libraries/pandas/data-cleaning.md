# Pandas: Data Cleaning & Filtering

Real-world datasets are messy: values are missing, types are wrong, and duplicate rows exist. Machine learning models in Scikit-Learn will raise errors if passed raw `NaN` values or unparsed strings.

---

## 🧼 Detecting & Handling Missing Values (`NaN`)

```python
import pandas as pd
import numpy as np

# 1. Count missing values per column
missing_counts = df.isna().sum()
missing_percent = (df.isna().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    "Missing Count": missing_counts,
    "Percentage (%)": missing_percent
})
print(missing_summary[missing_summary["Missing Count"] > 0])
```

### Strategies for Missing Data:

=== "Dropping Missing Rows / Cols"

    ```python
    # Drop rows where ANY column is NaN
    df_clean = df.dropna()

    # Drop rows where a specific critical column is NaN
    df_clean = df.dropna(subset=["target", "user_id"])

    # Drop columns that exceed 40% missing values
    threshold = 0.6 * len(df)
    df_clean = df.dropna(thresh=threshold, axis=1)
    ```

=== "Imputing in Pandas"

    ```python
    # Fill with a fixed value or mean/median
    median_val = df["salary"].median()
    df["salary"] = df["salary"].fillna(median_val)

    # Forward fill (propagate last valid observation forward - common in time series)
    df["stock_price"] = df["stock_price"].ffill()

    # Backward fill
    df["stock_price"] = df["stock_price"].bfill()
    ```

!!! tip "ML Pipeline Recommendation"
    For production ML models, do **not** hardcode manual mean imputation in Pandas before splitting. Use Scikit-learn's `SimpleImputer` inside a `Pipeline` to prevent test-set leakage.

---

## 🪞 Detecting & Dropping Duplicates

```python
# Check how many duplicated rows exist
num_duplicates = df.duplicated().sum()

# Drop exact duplicate rows (keeps the first occurrence)
df = df.drop_duplicates()

# Drop duplicates based on a subset of unique keys (e.g., student_id)
df = df.drop_duplicates(subset=["student_id"], keep="first")
```

---

## 🏷️ Fixing Data Types

When numbers are mistakenly loaded as strings (e.g., `"1,200.50"` or `"?"`):

```python
# 1. Clean dirty string numbers and convert
df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
# Note: errors="coerce" turns unparseable values (like 'N/A') into np.nan!

# 2. Date parsing
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

# Extract date features for ML:
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 3. Categorical types (saves huge memory for repeated text columns)
df["category"] = df["category"].astype("category")
```

---

## 🎯 Filtering & Querying

```python
# 1. Multiple conditions (parentheses are MANDATORY!)
seniors_in_tech = df[(df["age"] >= 35) & (df["department"] == "Engineering")]

# 2. Using .isin() for multi-value matching
target_depts = ["Engineering", "Product", "Design"]
tech_staff = df[df["department"].isin(target_depts)]

# 3. Range checking with .between()
mid_career = df[df["age"].between(25, 40, inclusive="both")]

# 4. String pattern matching
gmail_users = df[df["email"].str.contains(r"@gmail\.com$", na=False)]

# 5. Using .query() for readable string syntax
high_salary = df.query("salary > 75000 and department == 'Engineering'")
```

---

## ✂️ Outlier Trimming (IQR & Clipping)

Extreme outliers can distort linear models and distance calculations.

```python
# Quantile-based clipping (Winsorizing)
q_low = df["income"].quantile(0.01)
q_high = df["income"].quantile(0.99)

df["income_clipped"] = df["income"].clip(lower=q_low, upper=q_high)

# Interquartile Range (IQR) Filtering
Q1 = df["income"].quantile(0.25)
Q3 = df["income"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove IQR outliers
df_no_outliers = df[(df["income"] >= lower_bound) & (df["income"] <= upper_bound)]
```
