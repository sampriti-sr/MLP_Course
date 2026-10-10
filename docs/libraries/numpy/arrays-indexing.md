# NumPy: Arrays, Slicing & Reshaping

Efficient manipulation of feature matrices and label vectors requires fluency in indexing, slicing, and reshaping `ndarray` objects.

---

## 🛠️ Array Creation Functions

```python
import numpy as np

# Zeros and Ones
zeros = np.zeros((3, 4))            # 3x4 matrix filled with 0.0
ones = np.ones((2, 3), dtype=int)   # 2x3 matrix filled with 1
constant = np.full((2, 2), 7.0)     # 2x2 matrix filled with 7.0
identity = np.eye(3)                # 3x3 Identity matrix

# Ranges & Spacings
range_arr = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8] (step = 2)
linear_arr = np.linspace(0, 1, 5)   # 5 evenly spaced points between 0 and 1: [0., 0.25, 0.5, 0.75, 1.]

# Random Number Generation (Modern Generator API)
rng = np.random.default_rng(seed=42)
rand_uniform = rng.uniform(low=0.0, high=1.0, size=(3, 2)) # Uniform [0, 1)
rand_normal  = rng.normal(loc=0.0, scale=1.0, size=(3, 2))  # Standard normal N(0, 1)
rand_integers = rng.integers(low=0, high=10, size=(4,))     # Random integers in [0, 10)
```

---

## ✂️ Indexing & Slicing

Remember: Slices in NumPy are **views** into original memory, not copies! Modifying a slice modifies the original array unless `.copy()` is called.

```python
matrix = np.array([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]
])

# Single element access: [row, col]
val = matrix[0, 2]         # 12

# Slicing: [row_start:row_end, col_start:col_end]
sub = matrix[0:2, 1:3]     # [[11, 12], [21, 22]]

# Select entire row or column
first_row = matrix[0, :]   # [10, 11, 12, 13] (shape: (4,))
second_col = matrix[:, 1]  # [11, 21, 31]     (shape: (3,))
```

!!! danger "View vs Copy Trap"
    ```python
    a = np.array([1, 2, 3, 4])
    b = a[0:2]     # b is a VIEW of a
    b[0] = 999
    print(a)       # Output: [999, 2, 3, 4]  <-- Original was modified!

    # To create an independent copy:
    c = a[0:2].copy()
    ```

---

## 🎯 Boolean Masking & Conditional Filtering

Boolean masking is how you filter samples, detect outliers, or threshold predictions.

```python
scores = np.array([55, 82, 91, 40, 78, 95, 62])

# 1. Create a boolean mask
mask = scores >= 75
print(mask) # [False,  True,  True, False,  True,  True, False]

# 2. Filter using the mask
passing_scores = scores[mask]
print(passing_scores) # [82, 91, 78, 95]

# 3. Multiple logical conditions (use bitwise & and | with parentheses)
honors = scores[(scores >= 80) & (scores < 95)]  # [82, 91]

# 4. Conditional assignment with np.where
# Syntax: np.where(condition, value_if_true, value_if_false)
labels = np.where(scores >= 60, "Pass", "Fail")
```

---

## 🔄 Reshaping & Dimension Manipulation

In Machine Learning, dimension mismatches (e.g. expecting `(N, 1)` but receiving `(N,)`) are common.

```python
x = np.arange(12)  # shape (12,)

# 1. Reshape
mat = x.reshape(3, 4)   # shape (3, 4)
mat2 = x.reshape(3, -1) # -1 infers column dimension automatically -> (3, 4)

# 2. Add a new axis (1D vector -> 2D column/row matrix)
col_vec = x[:, np.newaxis] # shape: (12, 1)
row_vec = x[np.newaxis, :] # shape: (1, 12)
# Equivalent alternative:
col_vec2 = np.expand_dims(x, axis=1) # shape: (12, 1)

# 3. Squeeze (remove singleton dimensions)
squeezed = np.squeeze(col_vec) # shape: (12,)

# 4. Flattening
flat_copy = mat.flatten() # Returns a copy
flat_view = mat.ravel()   # Returns a flattened view (faster)
```

---

## 🧱 Stacking & Concatenating

Combine separate feature matrices or append labels:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Vertical stacking (row-wise concatenation) -> shape (4, 2)
v_stack = np.vstack([A, B])
v_concat = np.concatenate([A, B], axis=0)

# Horizontal stacking (column-wise concatenation) -> shape (2, 4)
h_stack = np.hstack([A, B])
h_concat = np.concatenate([A, B], axis=1)
```
