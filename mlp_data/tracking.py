"""A tiny experiment tracker: enough to make every result reproducible and comparable.

Industrial tools (MLflow, Weights & Biases) do this at scale; the habit is what matters. Each run
records WHAT was tried (name, parameters), WHAT happened (metrics), and WHERE it came from (data
fingerprint, code version, library versions, seed, time), so a number in a report can be traced
back and re-created.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(df: pd.DataFrame, *, length: int = 12) -> str:
    """A short hash of a table's shape, column names and every value. Changes if any of them changes."""
    h = hashlib.sha256()
    h.update(json.dumps([list(map(str, df.columns)), list(df.shape)]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()[:length]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict:
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "git_commit": _git_commit()}


class ExperimentLog:
    """Append-only record of runs. In memory, and optionally mirrored to a JSON-lines file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.runs: list[dict] = []
        if self.path and self.path.exists():
            self.runs = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def log(self, name: str, params: dict, metrics: dict, *, seed: int | None = None,
            data: str | None = None, notes: str = "") -> dict:
        run = {"run_id": uuid.uuid4().hex[:8], "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "name": name, "params": _plain(params), "metrics": _plain(metrics), "seed": seed, "data": data,
               "notes": notes, "env": environment()}
        self.runs.append(run)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(run) + "\n")
        return run

    def table(self, sort_by: str | None = None, *, ascending: bool = False) -> pd.DataFrame:
        rows = [{"run_id": r["run_id"], "name": r["name"], **{f"param.{k}": v for k, v in r["params"].items()},
                 **{f"metric.{k}": v for k, v in r["metrics"].items()}, "seed": r["seed"], "data": r["data"], "notes": r["notes"]}
                for r in self.runs]
        df = pd.DataFrame(rows)
        return df.sort_values(sort_by, ascending=ascending) if sort_by and sort_by in df else df

    def best(self, metric: str, *, higher_is_better: bool = True) -> dict:
        scored = [r for r in self.runs if metric in r["metrics"]]
        if not scored:
            raise KeyError(f"no run has the metric {metric!r}")
        return (max if higher_is_better else min)(scored, key=lambda r: r["metrics"][metric])


def _plain(d: dict) -> dict:
    """Make numpy scalars JSON-friendly."""
    def one(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)): return bool(v)
        return v
    return {str(k): one(v) for k, v in d.items()}
