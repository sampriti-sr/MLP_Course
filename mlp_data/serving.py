"""Packaging and scoring: turning a validated model into something other code can call safely.

A saved model is a file plus a *card* beside it. The card records what the model is, how it was
trained and how good it was, and carries a SHA-256 of the model file so that corruption or an
accidental swap is detected at load time.

SECURITY NOTE: model files are pickles. Loading a pickle can run arbitrary code, so only load
files you created or trust. The hash check catches accidents and one-sided tampering; it is NOT a
substitute for trusting where the file came from (store the expected hash separately, or sign it).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import models, retail
from .tracking import environment

CARD_VERSION = 1
SALES_COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id", "country", "revenue"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model, directory, name: str, *, metrics: dict, training: dict, notes: str = "") -> dict:
    """Write `<name>.joblib` and `<name>.card.json` into `directory`; return the card."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{name}.joblib"
    joblib.dump(model, model_path)
    card = {"card_version": CARD_VERSION, "name": name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": model_path.name, "sha256": _sha256(model_path), "size_bytes": model_path.stat().st_size,
            "features": list(models.FEATURES), "metrics": metrics, "training": training, "notes": notes, "env": environment()}
    (directory / f"{name}.card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def load_model(directory, name: str):
    """Load a model after verifying its hash against the card. Returns (model, card)."""
    directory = Path(directory)
    card = json.loads((directory / f"{name}.card.json").read_text())
    model_path = directory / card["file"]
    if _sha256(model_path) != card["sha256"]:
        raise ValueError(f"{model_path.name} does not match the hash in its card: the file was changed or corrupted; refusing to load it")
    if card["features"] != list(models.FEATURES):
        raise ValueError("the model was trained on a different feature list than this code expects")
    here = environment()
    if card["env"].get("scikit-learn") != here["scikit-learn"]:
        warnings.warn(f"model was saved with scikit-learn {card['env'].get('scikit-learn')}, you have {here['scikit-learn']}; "
                      "pickles are not guaranteed to be compatible across versions", stacklevel=2)
    return joblib.load(model_path), card


def validate_inputs(sales: pd.DataFrame, returns: pd.DataFrame, cutoff) -> pd.Timestamp:
    """Fail early, with a readable message, instead of deep inside feature code."""
    for label, frame in (("sales", sales), ("returns", returns)):
        missing = [c for c in SALES_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(frame["invoice_date"]):
            raise ValueError(f"{label}['invoice_date'] must be a datetime column")
        if frame[["customer_id", "invoice_date", "revenue"]].isna().any().any():
            raise ValueError(f"{label} has missing customer_id, invoice_date or revenue values")
    if len(sales) == 0:
        raise ValueError("sales is empty")
    try:
        cutoff = pd.Timestamp(cutoff)
    except (ValueError, TypeError) as e:
        raise ValueError(f"cutoff {cutoff!r} is not a date") from e
    if pd.isna(cutoff):
        raise ValueError("cutoff is not a date")
    if cutoff < sales["invoice_date"].min():
        raise ValueError(f"cutoff {cutoff.date()} is before the first record ({sales['invoice_date'].min().date()})")
    if cutoff > sales["invoice_date"].max() + pd.Timedelta(days=1):
        raise ValueError(f"cutoff {cutoff.date()} is after the last record ({sales['invoice_date'].max().date()}); the data is stale")
    return cutoff


def score_customers(model, sales: pd.DataFrame, returns: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Score every active customer as of `cutoff`, using only events before it.

    Returns one row per customer, most urgent first: the probability of buying again, the value at
    risk (chance of lapsing times past-year spend) and the call rank (1 = call first).
    """
    cutoff = validate_inputs(sales, returns, cutoff)
    snap = retail.make_snapshot(sales, returns, cutoff, features="full")      # features use only the past; its label is meaningless here
    p = model.predict_proba(snap[models.feature_columns(snap)])[:, 1]
    out = pd.DataFrame({"p_return": p, "value_at_risk": (1 - p) * snap["monetary"].to_numpy(),
                        "past_year_spend": snap["monetary"].to_numpy(), "orders": snap["frequency"].to_numpy(),
                        "days_since_last_order": snap["recency_days"].to_numpy()}, index=snap.index)
    out = out.sort_values("value_at_risk", ascending=False, kind="stable")
    out["call_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index()
