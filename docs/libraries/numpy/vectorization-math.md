# NumPy: Vectorization & Mathematics

Vectorization allows mathematical operations to be expressed cleanly and executed at compiled C-speeds, eliminating slow explicit Python `for` loops.

---

## ⚡ Vectorization vs Loops

Consider calculating Euclidean distances between samples:

=== "Vectorized NumPy (Fast & Clean)"

    ```python
    import numpy as np

    # Vectorized element-wise calculation
    X = np.random.randn(100_000, 10)
    mean = np.mean(X, axis=0)

    # Centering data (automatically broadcasts)
    X_centered = X - mean
    ```

=== "Python Loop (Slow & Clunky)"

    ```python
    # Avoid this in machine learning pipelines:
    X_centered = []
    for row in X:
        row_centered = []
        for i, val in enumerate(row):
            row_centered.append(val - mean[i])
        X_centered.append(row_centered)
    # 50x - 100x slower!
    ```

---

## 📡 Broadcasting Mechanics

Broadcasting defines how NumPy handles arithmetic operations between arrays of different shapes without copying data in memory.

### The Broadcasting Rule:
Starting from trailing (rightmost) dimensions and moving backwards:
1. Dimensions are compatible if they are **equal**, or
2. One of them is **1**.

### Visual Example:
```text
Array A (2D):   (3, 4)
Array B (1D):      (4,)   --> stretched along axis 0 to match (3, 4)
Result:         (3, 4)

Array A (2D):   (4, 1)
Array B (2D):   (1, 3)
Result:         (4, 3)
```

```python
A = np.array([[10, 20, 30],
              [40, 50, 60]])   # Shape: (2, 3)

bias = np.array([1, 2, 3])      # Shape: (3,)

# Bias is broadcast across each row of A
result = A + bias
# [[11, 22, 33],
#  [41, 52, 63]]
```

---

## 🧭 Axis-Wise Aggregations

In 2D matrices (samples $\times$ features):
- `axis=0`: Collapse across rows (computes statistics **per feature / column**)
- `axis=1`: Collapse across columns (computes statistics **per sample / row**)

```python
X = np.array([
    [1.0, 5.0],
    [3.0, 7.0],
    [2.0, 9.0]
])  # Shape: (3 samples, 2 features)

# Feature means (used in StandardScaler)
feature_means = np.mean(X, axis=0)      # [2.0, 7.0]   Shape: (2,)
feature_stds  = np.std(X, axis=0)       # [0.816, 1.633]

# Per-sample sum
sample_sums   = np.sum(X, axis=1)       # [6.0, 10.0, 11.0] Shape: (3,)

# Finding index of maximum value
best_sample_idx = np.argmax(X, axis=0)  # Indices of highest value per column
```

---

## 🧮 Linear Algebra & Matrix Multiplication

Machine learning models (Linear Regression, PCA, Neural Networks) rely on matrix operations:

```python
A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[5, 6], 
              [7, 8]])

# 1. Element-wise multiplication (Hadamard product)
elementwise = A * B
# [[ 5, 12],
#  [21, 32]]

# 2. Matrix Multiplication (Dot product) - use the @ operator!
matmul = A @ B
# [[19, 22],
#  [43, 50]]
# Equivalent to: np.matmul(A, B) or np.dot(A, B)

# 3. Transpose
A_T = A.T

# 4. Matrix Inverse & Determinant
A_inv = np.linalg.inv(A)
det_A = np.linalg.det(A)

# 5. Vector & Matrix Norms (Euclidean / L2 norm)
v = np.array([3.0, 4.0])
l2_norm = np.linalg.norm(v)     # 5.0
l1_norm = np.linalg.norm(v, ord=1) # 7.0
```

---

## 🛡️ Numerical Stability Functions in ML

Loss functions and gradient updates often run into numerical overflow ($e^{1000} \to \infty$) or underflow ($\log(0) \to -\infty$). Use these safe functions:

```python
# 1. Clip extreme values to prevent exploding gradients or division by zero
clipped = np.clip(arr, a_min=1e-15, a_max=1.0 - 1e-15)

# 2. Safe log calculation for small numbers: log(1 + x)
# Accurate when x is very close to 0
safe_log = np.log1p(1e-8)

# 3. Safe exp calculation for small numbers: exp(x) - 1
safe_expm1 = np.expm1(1e-8)
```
