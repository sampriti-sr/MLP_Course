# Pandas: Series & DataFrames

[Pandas](https://pandas.pydata.org/) provides intuitive data structures (`Series` and `DataFrame`) for tabular, labeled, and relational datasets. In practical ML workflows, 80% of data prep occurs in Pandas.

---

## 🏗️ Core Structures: `Series` vs `DataFrame`

- **`Series`**: A 1-dimensional labeled array capable of holding any data type. (Like a single column).
- **`DataFrame`**: A 2-dimensional labeled table of heterogeneous columns. (Like a spreadsheet or SQL table).

```python
import pandas as pd

# Creating a Series
labels = pd.Series([1, 0, 1, 1], name="target")

# Creating a DataFrame from a dictionary
data = {
    "age": [25, 30, 45, 22],
    "salary": [50000, 64000, 120000, 48000],
    "department": ["Sales", "Engineering", "Management", "Engineering"]
}
df = pd.DataFrame(data)
```

---

## 📥 Loading & Exporting Data

```python
# Reading CSV files
df = pd.read_csv(
    "dataset.csv",
    sep=",",                   # Delimiter (default is ',')
    na_values=["?", "NA", ""], # Strings to interpret as NaN
    parse_dates=["date_col"],  # Automatically convert date columns
    usecols=["age", "salary"]  # Load only required columns (memory saver!)
)

# Reading Parquet (faster & compressed, standard in modern big data)
df_parquet = pd.read_parquet("dataset.parquet")

# Saving results
df.to_csv("clean_data.csv", index=False)       # Always omit index unless meaningful
df.to_parquet("clean_data.parquet", index=False)
```

---

## 🔍 Fast Dataset Inspection

Run these four commands immediately upon receiving any new dataset:

```python
# 1. Quick preview of first/last rows
df.head(5)
df.tail(3)

# 2. Dimensions (rows, columns)
print("Shape:", df.shape)

# 3. Data types, non-null counts, and exact memory usage
df.info(memory_usage="deep")

# 4. Summary statistics for numerical columns
df.describe()

# 5. Summary statistics for categorical / text columns
df.describe(include=["object", "category"])
```

---

## 🎯 Indexing: `.loc[]` vs `.iloc[]`

Avoid raw bracket indexing (e.g. `df['col'][0]`) because it causes subtle bugs and chained indexing warnings.

| Method | Indexing Basis | Syntax Example |
| :--- | :--- | :--- |
| **`.loc[]`** | **Label-based** (row index names & column names) | `df.loc[0:5, ['age', 'salary']]` *(endpoint inclusive)* |
| **`.iloc[]`** | **Integer position-based** (0-indexed integer coordinates) | `df.iloc[0:5, 0:2]` *(endpoint exclusive, like Python)* |

```python
# Accessing row index label 0, column 'salary'
val = df.loc[0, "salary"]

# Accessing first row, second column by position
val_pos = df.iloc[0, 1]

# Conditional selection using .loc
high_earners = df.loc[df["salary"] > 60000, ["age", "department"]]
```

---

## ⚠️ The `SettingWithCopyWarning` Explained

Students often see: `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.`

### Why it happens:
Chained indexing (e.g. `df[df['age'] > 30]['salary'] = 70000`) creates an ambiguous intermediate copy. Pandas cannot guarantee if the assignment updates the original DataFrame or a temporary view.

### The Fix:
Always use `.loc[]` for assignment, and call `.copy()` when creating filtered sub-DataFrames:

```python
# ❌ WRONG (May not modify original, triggers warning):
sub_df = df[df["age"] > 30]
sub_df["bonus"] = 5000 

# ✅ CORRECT OPTION A: Explicit copy
sub_df = df[df["age"] > 30].copy()
sub_df["bonus"] = 5000

# ✅ CORRECT OPTION B: In-place modification on parent via .loc
df.loc[df["age"] > 30, "bonus"] = 5000
```
