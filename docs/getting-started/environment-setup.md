# Environment Setup Guide

Before beginning Machine Learning assignments, having a clean, reproducible Python environment is essential. Never install course packages into your global system Python.

---

## 1. Choosing Your Environment Tool

=== "Python `venv` (Standard & Recommended)"

    Python's built-in virtual environment tool requires no external package managers.

    ```bash
    # 1. Create a virtual environment named .venv
    python3 -m venv .venv

    # 2. Activate the virtual environment
    # On macOS / Linux:
    source .venv/bin/activate
    # On Windows PowerShell:
    # .venv\Scripts\Activate.ps1

    # 3. Upgrade pip
    pip install --upgrade pip
    ```

=== "Conda / Mamba (Alternative)"

    Recommended if you prefer Anaconda or Miniconda:

    ```bash
    # 1. Create a conda environment with Python 3.11 or 3.12
    conda create -n mlp-course python=3.11 -y

    # 2. Activate the environment
    conda activate mlp-course
    ```

---

## 2. Installing the Core ML Stack

Run the following command within your activated environment:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel
```

To register your environment kernel so that VS Code and JupyterLab can select it:

```bash
python -m ipykernel install --user --name=mlp-course --display-name="Python (MLP Course)"
```

---

## 3. Environment Verification Script

Create a file named `check_env.py` and run it:

```python
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns

print("=== MLP Environment Health Check ===")
print(f"Python:       {sys.version.split()[0]}")
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Seaborn:      {sns.__version__}")
print("====================================")
print("Environment is ready for Machine Learning Practice!")
```

Run:
```bash
python check_env.py
```

---

## 4. Recommended VS Code Extensions

If you use Visual Studio Code, install these extensions:

1. **Python** (`ms-python.python`) — IntelliSense, linting, debugging.
2. **Jupyter** (`ms-toolsai.jupyter`) — Interactive notebooks within VS Code.
3. **Data Wrangler** (`ms-toolsai.datawrangler`) — Visual data exploration tool for Pandas DataFrames.
