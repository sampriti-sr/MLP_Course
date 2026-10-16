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
    """Symmetric MAPE, between 0 and 2. A step where both the actual and the forecast are 0 counts as a perfect 0."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2 * np.abs(y - p) / denom)
    return float(np.mean(terms))


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


def seasonal_mean(history: pd.Series, horizon_index: pd.DatetimeIndex, *, weeks: int = 4, statistic: str = "mean") -> pd.Series:
    """Forecast each future hour with the mean (or median) of the same weekday-and-hour in the last ``weeks`` weeks.
    Needs ``horizon_index`` to start no later than one week after the end of ``history``."""
    stack = np.column_stack([history.reindex(horizon_index - pd.Timedelta(weeks=k)).to_numpy() for k in range(1, weeks + 1)])
    values = np.nanmedian(stack, axis=1) if statistic == "median" else np.nanmean(stack, axis=1)
    return pd.Series(values, index=horizon_index, name="forecast")


def holiday_aware_mean(history: pd.Series, horizon_index: pd.DatetimeIndex, holidays, *, weeks: int = 4,
                       holiday_profile: bool = True) -> pd.Series:
    """Seasonal mean that knows the holiday calendar.

    * Holiday dates are masked out of the history, so a holiday never contaminates the average used for an ordinary day.
    * With ``holiday_profile=True``, a target hour that falls on a holiday is forecast with the average of the same hour on
      *earlier holidays* in the history (falling back to the ordinary forecast when there is none yet).

    ``holidays`` is any collection of ``datetime.date`` objects known in advance (e.g. from a calendar).
    """
    holidays = set(holidays)
    is_holiday = pd.Series([d in holidays for d in history.index.date], index=history.index)
    clean = history.where(~is_holiday)
    past_holidays = sorted({d for d in holidays if pd.Timestamp(d) + pd.Timedelta(days=1) <= history.index[-1]
                            and pd.Timestamp(d) >= history.index[0].normalize()})
    values = []
    for t in horizon_index:
        if holiday_profile and t.date() in holidays and past_holidays:
            same_hour = [history.get(pd.Timestamp(d) + pd.Timedelta(hours=t.hour), np.nan) for d in past_holidays]
            values.append(np.nanmean(same_hour))
            continue
        ordinary = [clean.get(t - pd.Timedelta(weeks=k), np.nan) for k in range(1, weeks + 4)]   # look a few weeks further back
        ordinary = [v for v in ordinary if not np.isnan(v)][:weeks]                            # to replace masked holiday weeks
        values.append(np.mean(ordinary) if ordinary else np.nan)
    return pd.Series(values, index=horizon_index, name="forecast")


def score(bt: pd.DataFrame, y_train, *, m: int = 168) -> dict:
    """The standard scorecard for a backtest frame: MAE, RMSE, sMAPE and MASE (scaled by seasonal-naive period ``m``)."""
    return {"MAE": mae(bt["actual"], bt["forecast"]), "RMSE": rmse(bt["actual"], bt["forecast"]),
            "sMAPE": smape(bt["actual"], bt["forecast"]), "MASE": mase(bt["actual"], bt["forecast"], y_train, m)}


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
