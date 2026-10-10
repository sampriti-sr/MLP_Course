# Seaborn: Statistical Data Visualization

[Seaborn](https://seaborn.pydata.org/) builds on top of Matplotlib, integrating seamlessly with Pandas DataFrames. It is optimized for statistical exploration, handling confidence intervals, distributions, and multi-variable groupings with concise syntax.

---

## 🎨 Theme & Aesthetic Setup

At the top of your notebook or script:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Apply clean statistical styling
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
```

---

## 📊 Distribution Plots

Inspecting feature normality, skewness, and outliers:

=== "`sns.histplot` (Distribution + KDE)"

    ```python
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="feature_value",
        kde=True,               # Kernel Density Estimate curve
        hue="target_class",     # Breakdown by target class
        bins=30,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Feature Distribution across Classes")
    plt.show()
    ```

=== "`sns.boxplot` & `sns.violinplot` (Outliers & Quartiles)"

    ```python
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot shows median, IQR, and outlier points
    sns.boxplot(data=df, x="department", y="salary", ax=axes[0], palette="Set2")
    axes[0].set_title("Salary Distribution by Department")

    # Violinplot reveals multi-modal probability density
    sns.violinplot(data=df, x="department", y="salary", hue="gender", split=True, ax=axes[1])
    axes[1].set_title("Salary Density by Department and Gender")

    plt.tight_layout()
    plt.show()
    ```

---

## 🔍 Multi-Variable Exploration (`pairplot`)

For initial exploratory data analysis on 3 to 6 numerical features:

```python
# Pairwise relationships and diagonal univariate distributions
sns.pairplot(
    data=df,
    vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    hue="species",
    diag_kind="kde",
    corner=True  # Renders only the lower triangle to reduce clutter!
)
plt.show()
```

---

## 🔥 Correlation Matrix Heatmap

Linear models suffer when collinear features are present. Check correlation matrices:

```python
# Compute Pearson correlation for numerical columns
corr_matrix = df.select_dtypes(include="number").corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,              # Show numerical values inside cells
    fmt=".2f",               # 2 decimal places
    cmap="coolwarm",         # Diverging colormap (-1 blue, 0 white, +1 red)
    vmin=-1.0, vmax=1.0,     # Fixed color scale bounds
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.show()
```

---

## 🎯 Confusion Matrix Heatmap for Classification

Evaluating a Scikit-Learn classifier's predictions:

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# y_true and y_pred from test set
cm = confusion_matrix(y_true, y_pred)
class_labels = ["Negative (0)", "Positive (1)"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",                 # Display as integer counts
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax
)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Class", fontsize=11)
ax.set_ylabel("True Class", fontsize=11)
plt.show()
```
