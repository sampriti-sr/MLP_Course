# Pandas: Aggregations & Merging

Advanced feature engineering often requires aggregating statistics across subgroups or joining disparate datasets.

---

## 📊 GroupBy & Named Aggregations

The `groupby()` operation follows the **Split-Apply-Combine** pattern.

```python
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    "department": ["Engineering", "Sales", "Engineering", "Sales", "HR"],
    "level": ["Senior", "Junior", "Mid", "Senior", "Junior"],
    "salary": [120000, 60000, 95000, 110000, 55000],
    "experience": [8, 2, 4, 7, 1]
})

# 1. Simple aggregation
dept_mean = df.groupby("department")["salary"].mean()

# 2. Modern Named Aggregations (cleanest, best practice!)
dept_stats = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    median_exp=("experience", "median"),
    employee_count=("salary", "count")
).reset_index()

print(dept_stats)
```

### Aggregation vs `transform()`:

- **`agg()`**: Collapses rows into a single summary row per group.
- **`transform()`**: Returns an array with the **same row count** as the original DataFrame, allowing group-level features to be added directly back to samples!

```python
# Create a feature: difference between employee salary and departmental average
df["dept_avg_salary"] = df.groupby("department")["salary"].transform("mean")
df["salary_vs_dept_mean"] = df["salary"] - df["dept_avg_salary"]
```

---

## 🔗 Merging DataFrames (`pd.merge`)

Use `pd.merge()` for SQL-like joins across primary / foreign keys:

```python
users = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "Diana"]
})

transactions = pd.DataFrame({
    "trans_id": [101, 102, 103, 104],
    "user_id": [1, 2, 2, 5],
    "amount": [250.0, 45.0, 99.0, 120.0]
})

# 1. Inner Join (only rows with keys in both tables)
inner_df = pd.merge(users, transactions, on="user_id", how="inner")

# 2. Left Join (keep all users, fill missing transactions with NaN)
left_df = pd.merge(users, transactions, on="user_id", how="left")

# 3. Merging on different column names
# pd.merge(df_a, df_b, left_on="customer_id", right_on="user_id", how="left")
```

---

## 🧱 Concatenation (`pd.concat`)

Use `pd.concat()` to stitch DataFrames along an axis:

```python
# Vertical Stacking (combining monthly data logs)
q1_data = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
q2_data = pd.DataFrame({"id": [3, 4], "val": [30, 40]})

all_quarters = pd.concat([q1_data, q2_data], axis=0, ignore_index=True)

# Horizontal Stacking (combining features side-by-side)
features_part1 = pd.DataFrame({"feat_1": [1.1, 2.2]})
features_part2 = pd.DataFrame({"feat_2": [5.5, 6.6]})

full_features = pd.concat([features_part1, features_part2], axis=1)
```

---

## 🔄 Pivot Tables & Reshaping

Transforming between long and wide formats:

```python
# Create a pivot table: department vs level with mean salary
pivot = pd.pivot_table(
    df,
    values="salary",
    index="department",
    columns="level",
    aggfunc="mean",
    fill_value=0
)
print(pivot)

# Melting wide data into long format (useful for plotting)
melted = pd.melt(
    pivot.reset_index(),
    id_vars=["department"],
    var_name="level",
    value_name="avg_salary"
)
```
