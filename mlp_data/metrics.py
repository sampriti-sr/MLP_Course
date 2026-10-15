"""The yardstick used by every modelling page, so numbers are comparable across the course.

The decision behind it (Page 1): account managers can phone K customers per cycle, so we rank
customers by risk of NOT buying again and call the top K. We therefore report ranking quality
(AUC, average precision) AND decision quality (precision@K, revenue protected@K).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

K = 200  # calls per cycle: an assumption made in Page 1


def at_risk_report(y_returns, p_returns, spend, *, k: int = K, risk=None, probabilities: bool = False, tie_seed: int = 0) -> dict:
    """Evaluate a scoring of customers on the fixed-call-budget decision.

    y_returns : 1 if the customer bought again, else 0 (so a "lapse" is 1 - y)
    p_returns : None, or the model's score that the customer buys again (higher = more likely to return). It
                only needs to rank customers, unless probabilities=True
    probabilities : set True only if p_returns really is a probability; then Brier score and log
                loss are reported too. Never compute them on an arbitrary score.
    spend     : each customer's past-year revenue (the value at stake)
    tie_seed  : seed for breaking ties between equal risk scores (reproducible, order-independent)
    risk      : optional ranking score where higher = call this customer first. Defaults to
                1 - p_returns (call the least likely to return). Use it to rank by value-at-risk.
    """
    y = np.asarray(y_returns).astype(int)
    spend = np.asarray(spend, dtype=float)
    lapse = 1 - y
    if p_returns is None:
        if risk is None or probabilities:
            raise ValueError("p_returns=None needs risk=... and cannot report probabilities")
        p = None
    else:
        p = np.asarray(p_returns, dtype=float)
    risk = (1 - p) if risk is None else np.asarray(risk, dtype=float)
    n = len(y)
    # Ties (e.g. thousands of customers with exactly one order) must be broken at random, not by the
    # order rows happen to appear in the file: shuffle with a fixed seed, then do a stable sort.
    shuffle = np.random.default_rng(tie_seed).permutation(n)
    order = shuffle[np.argsort(-risk[shuffle], kind="stable")]
    top = order[:k]
    lapse_rate = lapse.mean()
    revenue_lapsers = spend[lapse == 1]
    out = {
        "AUC": roc_auc_score(lapse, risk),
        "avg precision": average_precision_score(lapse, risk),
        f"precision@{k}": lapse[top].mean(),
        f"lift@{k}": lapse[top].mean() / lapse_rate if lapse_rate > 0 else np.nan,
        f"revenue@{k}": float(spend[top][lapse[top] == 1].sum()),
        f"revenue@{k} random": float(k / n * revenue_lapsers.sum()),                 # expected value, no simulation noise
        f"revenue@{k} oracle": float(np.sort(revenue_lapsers)[::-1][:k].sum()),      # best possible with perfect foresight
    }
    out[f"revenue@{k} % of oracle"] = out[f"revenue@{k}"] / out[f"revenue@{k} oracle"] if out[f"revenue@{k} oracle"] else np.nan
    if probabilities:
        if not ((p >= 0) & (p <= 1)).all():
            raise ValueError("probabilities=True but p_returns is outside [0, 1]")
        out["brier"] = brier_score_loss(y, np.clip(p, 0, 1))
        out["log loss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    return out


def report_table(rows: dict, *, k: int = K) -> pd.DataFrame:
    """rows: name -> dict from at_risk_report. Returns a tidy comparison table."""
    return pd.DataFrame(rows).T


def random_calling_report(y_returns, spend, *, train_rate: float, k: int = K) -> dict:
    """What calling customers at random achieves, as exact expectations (no simulation noise).

    A constant score cannot be ranked, so this is computed analytically rather than by letting a
    tie-break pick arbitrary customers. Brier/log-loss use a constant prediction of `train_rate`.
    """
    y = np.asarray(y_returns).astype(int)
    spend = np.asarray(spend, dtype=float)
    lapse = 1 - y
    n = len(y)
    rand = k / n * spend[lapse == 1].sum()
    oracle = np.sort(spend[lapse == 1])[::-1][:k].sum()
    const = np.full(n, train_rate)
    return {"AUC": 0.5, "avg precision": lapse.mean(), f"precision@{k}": lapse.mean(), f"lift@{k}": 1.0,
            f"revenue@{k}": float(rand), f"revenue@{k} random": float(rand), f"revenue@{k} oracle": float(oracle),
            f"revenue@{k} % of oracle": float(rand / oracle), "brier": brier_score_loss(y, const), "log loss": log_loss(y, const)}


def bootstrap_interval(y_returns, p_returns, spend, *, metric: str, risk=None, k: int = K,
                       n_boot: int = 300, seed: int = 0, level: float = 0.90) -> tuple[float, float]:
    """Percentile interval for one metric, resampling customers with replacement.

    Heavy-tailed revenue means a handful of customers can swing a metric. Differences between two
    models that are smaller than this noise are not evidence.
    """
    y, spend = np.asarray(y_returns), np.asarray(spend)
    p = None if p_returns is None else np.asarray(p_returns)
    risk = None if risk is None else np.asarray(risk)
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(at_risk_report(y[i], None if p is None else p[i], spend[i], k=k, risk=None if risk is None else risk[i])[metric])
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_difference(y_returns, spend, risk_a, risk_b, *, metric: str, k: int = K,
                         n_boot: int = 300, seed: int = 0, level: float = 0.90) -> dict:
    """Paired bootstrap of metric(A) - metric(B): both methods are scored on the SAME resampled customers.

    Two methods evaluated on the same customers share most of their noise, so the difference is far
    better determined than the two separate intervals suggest. Returns the mean difference, an
    interval, and the share of resamples in which A beats B.
    """
    y, spend = np.asarray(y_returns), np.asarray(spend)
    ra, rb = np.asarray(risk_a), np.asarray(risk_b)
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a = at_risk_report(y[i], None, spend[i], k=k, risk=ra[i])[metric]
        b = at_risk_report(y[i], None, spend[i], k=k, risk=rb[i])[metric]
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"mean": float(diffs.mean()), "low": float(lo), "high": float(hi), "share_a_better": float((diffs > 0).mean())}
