"""One loader per course dataset. Data is returned *raw* on purpose: cleaning is a lesson.

The only changes made here are purely technical (e.g. casting mixed int/str columns to a
single text dtype so the file can be cached as parquet). Provenance, licences and known
quirks of each dataset are recorded in data/README.md.
"""
from __future__ import annotations

import zipfile
from typing import NamedTuple, Sequence

import pandas as pd

from ._download import data_dir, extract_member, fetch, verify

# --- Pinned sources (checksums: see data/README.md) ---------------------------------
_RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
_RETAIL_ZIP_SHA = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
_RETAIL_XLSX_SHA = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"

_ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
_ML_SHA = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"

_TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
_TAXI_SHA = {"2023-01": "32df6f67578fa86c484a6b5ef23a5281992ff085521082340b0f9e5889e9a572"}

AIRBNB_SNAPSHOT = "2026-09-17"
_AIRBNB_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    "{snapshot}/data/listings.csv.gz"
)
_AIRBNB_SHA = {"2026-09-17": "8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c"}

_CREDIT_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
_CREDIT_ZIP_SHA = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
_CREDIT_XLS_SHA = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


def _raw(name: str):
    return data_dir() / "raw" / name


def load_retail() -> pd.DataFrame:
    """UCI Online Retail II: 1,067,371 invoice lines, Dec 2009 - Dec 2011.

    First call downloads ~46 MB and converts the Excel file (~35 s); later calls read a
    cached parquet copy in about a second.
    """
    cache = data_dir() / "retail.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zpath = fetch(_RETAIL_URL, _raw("online_retail_ii.zip"), sha256=_RETAIL_ZIP_SHA, label="Online Retail II")
    xlsx = extract_member(zpath, "online_retail_II.xlsx", _raw("online_retail_II.xlsx"))
    verify(xlsx, _RETAIL_XLSX_SHA, "online_retail_II.xlsx")
    print("Reading Excel (one-off, ~35 s) ...", flush=True)
    sheets = pd.read_excel(xlsx, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    # Invoice / StockCode / Description mix ints and strings; one text dtype makes them cacheable.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string")
    df.to_parquet(cache)
    return df


class MovieLens(NamedTuple):
    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame
    tags: pd.DataFrame


def load_movielens() -> MovieLens:
    """MovieLens `ml-latest-small`: 100,836 ratings, 610 users, 9,742 movies.

    Non-commercial use only; see the GroupLens terms in data/README.md.
    """
    zpath = fetch(_ML_URL, _raw("ml-latest-small.zip"), sha256=_ML_SHA, label="MovieLens small")
    tables = {}
    with zipfile.ZipFile(zpath) as zf:
        for name in ("ratings", "movies", "links", "tags"):
            with zf.open(f"ml-latest-small/{name}.csv") as f:
                tables[name] = pd.read_csv(f)
    return MovieLens(**tables)


def load_taxi(months: str | Sequence[str] = "2023-01") -> pd.DataFrame:
    """NYC TLC Yellow Taxi trips for one or more months, e.g. ``"2023-01"`` or ``["2023-01", "2023-02"]``.

    Each month is roughly 3 million rows (~47 MB download). The data is raw, including
    trips dated outside their month and negative fares.
    """
    months = [months] if isinstance(months, str) else list(months)
    frames = []
    for m in months:
        if len(m) != 7 or m[4] != "-":
            raise ValueError(f"Month must look like '2023-01', got {m!r}.")
        path = fetch(_TAXI_URL.format(month=m), _raw(f"yellow_tripdata_{m}.parquet"),
                     sha256=_TAXI_SHA.get(m), label=f"NYC taxi {m}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_airbnb() -> pd.DataFrame:
    """Inside Airbnb London listings, snapshot 2026-09-17: ~92k rows x 90 columns.

    `price` is returned as text with a currency symbol and is ~35% null, on purpose.
    Attribution: Inside Airbnb (CC BY 4.0).
    """
    path = fetch(_AIRBNB_URL.format(snapshot=AIRBNB_SNAPSHOT), _raw(f"airbnb_london_{AIRBNB_SNAPSHOT}_listings.csv.gz"),
                 sha256=_AIRBNB_SHA.get(AIRBNB_SNAPSHOT), label=f"Inside Airbnb London {AIRBNB_SNAPSHOT}")
    return pd.read_csv(path, low_memory=False)


def load_credit() -> pd.DataFrame:
    """UCI Default of Credit Card Clients (Taiwan, 2005): 30,000 rows x 25 columns.

    The target column is ``default payment next month``. Returned raw, including the
    undocumented category codes in EDUCATION and MARRIAGE.
    """
    zpath = fetch(_CREDIT_URL, _raw("default_of_credit_card_clients.zip"), sha256=_CREDIT_ZIP_SHA, label="Credit default")
    xls = extract_member(zpath, "default of credit card clients.xls", _raw("default_of_credit_card_clients.xls"))
    verify(xls, _CREDIT_XLS_SHA, "default of credit card clients.xls")
    return pd.read_excel(xls, header=1)  # the real header is on the second row


DATASETS = {
    "retail": load_retail,
    "movielens": load_movielens,
    "taxi": load_taxi,
    "airbnb": load_airbnb,
    "credit": load_credit,
}
