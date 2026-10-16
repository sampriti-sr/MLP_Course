"""Hourly / daily taxi demand: the series used by the time-series track.

``taxi_hourly`` does what Page 1 of the track teaches (keep the month's own timestamps, count pickups
per hour), but reads only one column of each monthly file and caches the tiny result, so a student can
work with several months without holding 10 million trips in memory.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from ._download import data_dir, fetch
from .loaders import _TAXI_SHA, _TAXI_URL


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(month) != 7 or month[4] != "-":
        raise ValueError(f"Month must look like '2023-01', got {month!r}.")
    start = pd.Timestamp(month + "-01")
    return start, start + pd.offsets.MonthBegin(1)


def taxi_hourly(months: str | Sequence[str] = "2023-01") -> pd.Series:
    """Yellow-taxi pickups per hour, as a Series named ``pickups`` on a ``DatetimeIndex``.

    * Timestamps are the meter's local New York wall-clock time, *naive* (no time zone), as published.
    * Only trips whose pickup falls inside the requested month are counted (the raw files contain
      strays dated years away).
    * Hours with no trips are simply absent: nothing is filled in. Use ``.asfreq("h")`` to see the gaps.
    """
    months = [months] if isinstance(months, str) else list(months)
    parts = []
    for m in months:
        start, end = _month_bounds(m)
        cache = data_dir() / f"taxi_hourly_{m}.parquet"
        if cache.exists():
            parts.append(pd.read_parquet(cache)["pickups"])
            continue
        path = fetch(_TAXI_URL.format(month=m), data_dir() / "raw" / f"yellow_tripdata_{m}.parquet",
                     sha256=_TAXI_SHA.get(m), label=f"NYC taxi {m}")
        ts = pd.read_parquet(path, columns=["tpep_pickup_datetime"])["tpep_pickup_datetime"]
        ts = ts[(ts >= start) & (ts < end)]
        counts = ts.dt.floor("h").value_counts().sort_index().rename("pickups")
        counts.index.name = "hour"
        counts.to_frame().to_parquet(cache)
        parts.append(counts)
    out = pd.concat(parts).sort_index()
    out.index.name = "hour"
    return out
