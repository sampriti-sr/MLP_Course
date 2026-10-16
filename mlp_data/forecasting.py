"""Forecasting yardsticks shared by the time-series pages: error metrics, baselines, safe lag features
and a rolling-origin backtest.

Everything assumes a regular ``DatetimeIndex`` (call ``.asfreq("h")`` first) and a *forecast origin*:
the moment we stand at, knowing the past and nothing after it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# --- error metrics ------------------------------------------------------------------------------
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mape(y, p) -> float:
    """Mean absolute % error. Blows up (or is undefined) when y is near zero: see the metrics page."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs((y - p) / y)))


def smape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))))


def mase(y, p, y_train, m: int = 1) -> float:
    """MAE scaled by the in-sample MAE of the seasonal-naive forecast with period ``m``.
    Below 1 means "better than seasonal naive on the training data"."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        raise ValueError("The training series is perfectly seasonal, so the MASE scale is zero.")
    return mae(y, p) / float(scale)


# --- baselines ----------------------------------------------------------------------------------
def seasonal_naive(history: pd.Series, horizon_index: pd.DatetimeIndex, period: int) -> pd.Series:
    """Forecast each future hour with the value ``period`` steps earlier, repeating the last full cycle
    when the horizon is longer than the period. ``period=1`` is the plain naive forecast."""
    last = history.iloc[-period:].to_numpy()
    reps = int(np.ceil(len(horizon_index) / period))
    return pd.Series(np.tile(last, reps)[: len(horizon_index)], index=horizon_index, name="forecast")


# --- features -----------------------------------------------------------------------------------
def make_features(y: pd.Series, *, min_lag: int, lags=(), windows=()) -> pd.DataFrame:
    """Calendar + lag + rolling features for forecasting ``min_lag`` steps ahead.

    Every lag is at least ``min_lag``; every rolling statistic is computed on ``y.shift(min_lag)`` so that the
    row for time *t* only uses data up to *t - min_lag*, i.e. data that exists at the forecast origin.
    """
    if any(l < min_lag for l in lags):
        raise ValueError(f"Lags must be >= min_lag={min_lag}, otherwise the feature is unknown at forecast time.")
    X = pd.DataFrame(index=y.index)
    X["hour"] = y.index.hour
    X["dayofweek"] = y.index.dayofweek
    for l in lags:
        X[f"lag_{l}"] = y.shift(l)
    base = y.shift(min_lag)
    for w in windows:
        X[f"mean_{w}"] = base.rolling(w).mean()
    return X


# --- backtesting --------------------------------------------------------------------------------
def rolling_origins(index: pd.DatetimeIndex, *, first_origin, last_origin=None, step: str = "D") -> list[pd.Timestamp]:
    """Forecast origins from ``first_origin`` on, every ``step``, never past ``last_origin``."""
    last = index[-1] if last_origin is None else pd.Timestamp(last_origin)
    return list(pd.date_range(pd.Timestamp(first_origin), last, freq=step))


def backtest(y: pd.Series, forecast: Callable[[pd.Series, pd.DatetimeIndex], pd.Series], *,
             origins, horizon: int) -> pd.DataFrame:
    """Rolling-origin evaluation. At each origin ``o`` the forecaster sees ``y[:o)`` only (strictly before ``o``)
    and must return ``horizon`` values for the steps starting at ``o``. Returns one row per forecast step:
    ``origin, step, time, actual, forecast``. Steps that fall beyond the end of the data are dropped."""
    rows = []
    step = pd.tseries.frequencies.to_offset(y.index.freq or pd.infer_freq(y.index))
    for o in origins:
        history = y.loc[: o - pd.Timedelta(1, "ns")]
        target_index = pd.date_range(o, periods=horizon, freq=step)
        target_index = target_index[target_index <= y.index[-1]]
        if len(target_index) == 0:
            continue
        f = forecast(history, target_index)
        f = pd.Series(np.asarray(f, float), index=target_index)
        rows.append(pd.DataFrame({"origin": o, "step": np.arange(1, len(target_index) + 1),
                                  "time": target_index, "actual": y.loc[target_index].to_numpy(), "forecast": f.to_numpy()}))
    return pd.concat(rows, ignore_index=True)
