"""The course's champion and challenger models for the repurchase task.

Page 8 builds these in the open and checks that they match; later pages import them. Everything
that *learns from data* (medians, scales) lives inside the pipeline, so it is fitted on training
rows only. Transformers are named module-level functions (not lambdas) so that a fitted model can
be saved with joblib, which Page 12 does.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

NOT_FEATURES = ("customer_id", "cutoff", "y")

HEAVY_TAILED = ["recency_days", "frequency", "monetary", "monetary_90d", "avg_order_value", "items_per_order",
                "n_products", "freq_30d", "freq_90d", "freq_180d", "freq_prev_90d", "n_returns", "tenure_days", "avg_gap_days"]
RATIO = ["recency_over_gap"]
SIGNED = ["trend_90d"]
OTHER = ["return_value_share", "is_uk"]
FEATURES = HEAVY_TAILED + RATIO + SIGNED + OTHER      # fixed order: the model is trained and scored on exactly these columns


def _log1p(x):
    return np.log1p(x)


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _capped_log(x, cap=50):
    return np.log1p(np.clip(x, 0, cap))     # the overdue ratio can explode: cap, then log


def feature_columns(frame) -> list[str]:
    """The model inputs of a snapshot table, in the course's fixed order. Raises if one is missing."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"snapshot table is missing feature columns: {missing}")
    return FEATURES


def make_logistic(C: float = 1.0, overdue_cap: float = 50):
    """Champion: logistic regression with heavy-tail handling, imputation + missing indicators, scaling."""
    prepare = ColumnTransformer([
        ("tails", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), FunctionTransformer(_log1p, feature_names_out="one-to-one"), StandardScaler()), HEAVY_TAILED),
        ("ratio", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), FunctionTransformer(_capped_log, kw_args={"cap": overdue_cap}, feature_names_out="one-to-one"), StandardScaler()), RATIO),
        ("change", make_pipeline(FunctionTransformer(_signed_log, feature_names_out="one-to-one"), StandardScaler()), SIGNED),
        ("rest", StandardScaler(), OTHER),
    ])
    return make_pipeline(prepare, LogisticRegression(C=C, max_iter=2000))


def make_boosting(**overrides):
    """Challenger: gradient-boosted trees (handles missing values natively; no scaling needed)."""
    params = dict(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)
