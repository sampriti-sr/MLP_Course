# Matplotlib: The Object-Oriented Interface

[Matplotlib](https://matplotlib.org/) is Python's foundational plotting library. While quick scripts often use `plt.plot()`, practical ML workflows demand the **Object-Oriented (OO) API** for total layout control, reproducible subplots, and publication-ready figures.

---

## 🧠 The Mental Model: `Figure` vs `Axes`

- **`Figure` (`fig`)**: The overarching canvas or window holding all visual elements, titles, and legends.
- **`Axes` (`ax`)**: An individual coordinate subplot residing inside a `Figure`. All plots, ticks, axis labels, and grids belong to an `Axes`.

```text
+-------------------------------------------------------+
| Figure (fig)                                          |
|                                                       |
|  +------------------------+  +---------------------+  |
|  | Axes 1 (ax1)           |  | Axes 2 (ax2)        |  |
|  | [Loss Curves]          |  | [Accuracy Curves]   |  |
|  +------------------------+  +---------------------+  |
+-------------------------------------------------------+
```

---

## 📐 Standard Subplot Template

Always start plots using `plt.subplots()`:

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. Create canvas with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=100)

# Generate dummy training history
epochs = np.arange(1, 21)
train_loss = np.exp(-epochs / 5) + 0.1
val_loss = np.exp(-epochs / 5) + 0.15 + 0.05 * np.random.rand(20)

# Subplot 1: Loss curves
axes[0].plot(epochs, train_loss, label="Train Loss", color="royalblue", lw=2)
axes[0].plot(epochs, val_loss, label="Val Loss", color="crimson", lw=2, linestyle="--")
axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch", fontsize=10)
axes[0].set_ylabel("Loss", fontsize=10)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(frameon=True)

# Subplot 2: Residual distribution
residuals = np.random.normal(0, 1, 500)
axes[1].hist(residuals, bins=25, color="seagreen", edgecolor="black", alpha=0.7)
axes[1].set_title("Model Residuals Distribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Prediction Error", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].axvline(0, color="black", linestyle="--", lw=1.5)

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()
plt.show()
```

---

## 🎨 Common Plot Types

### 1. Scatter Plots (Feature Relationships & Decision Boundaries)
```python
fig, ax = plt.subplots(figsize=(7, 5))

scatter = ax.scatter(
    x=df["feature_1"], 
    y=df["feature_2"], 
    c=df["target"],           # Color points by class
    cmap="coolwarm",          # Color map
    alpha=0.8, 
    edgecolors="w", 
    s=50                      # Marker size
)

ax.set_title("Feature 1 vs Feature 2 by Class")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
fig.colorbar(scatter, ax=ax, label="Target Class")
plt.show()
```

### 2. Bar Charts with Error Bars
```python
models = ["Linear Reg", "Ridge", "Random Forest", "GBoost"]
rmse = [45.2, 42.1, 31.8, 29.4]
std_err = [2.1, 1.9, 1.2, 1.0]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models, rmse, yerr=std_err, capsize=5, color="teal", alpha=0.85)

ax.set_title("Model Comparison (RMSE with 5-Fold Std Dev)")
ax.set_ylabel("Root Mean Squared Error")
ax.set_ylim(0, 55)

# Annotate values above bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval:.1f}", ha="center", va="bottom")

plt.show()
```

---

## 💾 Exporting Publication-Ready Figures

Always save figures at 300 DPI with tight bounding boxes for reports and presentations:

```python
fig.savefig(
    "model_evaluation.png",
    dpi=300,                  # High print resolution
    bbox_inches="tight",      # Trims whitespace padding
    transparent=False         # Solid background
)
```
