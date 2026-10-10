# NumPy: Overview & Core Concepts

[NumPy](https://numpy.org/) (**Numerical Python**) is the foundational package for scientific computing and machine learning in Python. Most ML libraries—including Pandas, Scikit-learn, PyTorch, and TensorFlow—either build directly on NumPy arrays or interoperate natively with them.

---

## 🚀 Why NumPy Instead of Python Lists?

| Feature | Standard Python List | NumPy `ndarray` |
| :--- | :--- | :--- |
| **Memory Storage** | Array of pointers to boxed Python objects | Contiguous block of raw C memory |
| **Data Types** | Heterogeneous (mixed types allowed) | Homogeneous (single fixed type) |
| **Execution Speed** | Iteration via Python bytecode interpreter | Compiled C / BLAS / LAPACK SIMD vector instructions |
| **Memory Footprint** | Large (significant object overhead) | Minimal (exact byte layout) |
| **Syntax** | Requires list comprehensions or loops | Natural mathematical vector notation |

---

## 🧩 The Anatomy of an `ndarray`

Every NumPy array has core attributes you should inspect regularly:

```python
import numpy as np

arr = np.array([[1.0, 2.0, 3.0], 
                [4.0, 5.0, 6.0]], dtype=np.float32)

print("Array:\n", arr)
print("Shape:       ", arr.shape)    # (2, 3) -> 2 rows, 3 columns
print("Dimensions:  ", arr.ndim)     # 2 dimensions
print("Data Type:   ", arr.dtype)    # float32
print("Item Size:   ", arr.itemsize) # 4 bytes per float32 element
print("Total Size:  ", arr.size)     # 6 total elements
```

---

## 🗂️ Common Data Types (`dtype`)

In Machine Learning, choosing the right dtype impacts memory usage and execution speed:

- `np.float32` / `np.float64`: Default floating point numbers. Deep learning frameworks usually prefer `float32` for speed and GPU compatibility.
- `np.int32` / `np.int64`: Standard integers for counts and discrete labels.
- `np.bool_`: Boolean values (`True` / `False`), essential for logical masks.
- `np.uint8`: Unsigned 8-bit integer (0 to 255), standard for image pixel representations.

Converting types with `.astype()`:

```python
# Convert float to int (truncates decimals)
int_arr = arr.astype(np.int32)

# Memory optimization check: float64 takes 8 bytes/element vs float32 4 bytes/element
x = np.ones((1000, 1000), dtype=np.float64)  # ~8 MB
y = np.ones((1000, 1000), dtype=np.float32)  # ~4 MB
```
