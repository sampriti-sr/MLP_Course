"""Cleaning and snapshot building for the Online Retail II case study (lifecycle chapters).

The lifecycle pages build every step in the open, in the notebook. This module holds the
same logic, tested, so that later pages can reuse it without repeating 200 lines. Each page
that builds a step ends with a cell asserting that its own result equals this module's, so
the two cannot drift apart.

The running question: *which customers who bought in the last year will buy again in the
next 90 days?*  One row = one customer at one **snapshot date** (the "cutoff"). Features may
only use what happened *before* the cutoff; the label looks at what happens *after* it.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# A real product code is 5 digits plus an optional 1-2 letter variant suffix (85123A, 15056BL).
# 99.4% of raw lines match; the rest are postage, manual entries, bank charges, test items, ...
PRODUCT_CODE = r"\d{5}(?:[A-Za-z]{1,2})?"

HORIZON_DAYS = 90      # label window: "buys again within 90 days of the cutoff"
LOOKBACK_DAYS = 365    # who counts as an "active" customer, and how far features look back

# Snapshot dates. Cutoffs start 2010-12-01 so every customer has a full year of observable
# history (the data begins 2009-12-01); the last label window must end before the data does.
TRAIN_CUTOFFS = tuple(pd.to_datetime(["2010-12-01", "2011-01-01", "2011-02-01", "2011-03-01"]))
VALID_CUTOFF = pd.Timestamp("2011-06-01")
TEST_CUTOFF = pd.Timestamp("2011-09-01")

_RENAME = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "unit_price",
    "Customer ID": "customer_id", "Country": "country",
}


class Cleaned(NamedTuple):
    sales: pd.DataFrame    # genuine purchase lines from identified customers
    returns: pd.DataFrame  # cancellation lines from identified customers
    log: pd.DataFrame      # row accounting: what each step removed and why


REVERSAL_WINDOW = pd.Timedelta(days=1)   # a cancellation within a day of an identical purchase undoes it


def fully_reversed_invoices(sales: pd.DataFrame, returns: pd.DataFrame, window: pd.Timedelta = REVERSAL_WINDOW):
    """Find purchases that were cancelled almost immediately.

    A purchase line is "reversed" when the same customer returns the same product, in the same
    quantity, within `window` after buying it. An invoice is *fully reversed* when every product on it
    was. Returns (invoice numbers, the matching cancellation lines as (invoice, stock_code) pairs).
    """
    if len(returns) == 0 or len(sales) == 0:
        return set(), pd.DataFrame(columns=["invoice", "stock_code"])
    r = returns.assign(qty=-returns["quantity"])
    m = r.merge(sales[["customer_id", "stock_code", "quantity", "invoice_date", "invoice"]],
                left_on=["customer_id", "stock_code", "qty"], right_on=["customer_id", "stock_code", "quantity"], suffixes=("_ret", "_sale"))
    m = m[(m["invoice_date_sale"] <= m["invoice_date_ret"]) & (m["invoice_date_ret"] - m["invoice_date_sale"] <= window)]
    products_on_invoice = sales.groupby("invoice")["stock_code"].nunique()
    products_reversed = m.groupby("invoice_sale")["stock_code"].nunique()
    full = products_reversed[products_reversed >= products_on_invoice.reindex(products_reversed.index)].index
    pairs = m[m["invoice_sale"].isin(full)][["invoice_ret", "stock_code"]].drop_duplicates().rename(columns={"invoice_ret": "invoice"})
    return set(full), pairs


def find_sheet_boundary(raw: pd.DataFrame) -> int | None:
    """Row where the second Excel sheet starts, found from the data itself.

    The file is two sheets stacked. Within a sheet time only moves forward, so the single
    place where it jumps backwards is the seam. Returns None if there is no seam.
    """
    jumps = np.flatnonzero((raw["InvoiceDate"].diff() < pd.Timedelta(0)).to_numpy())
    return int(jumps[0]) if len(jumps) == 1 else None


def clean_retail(raw: pd.DataFrame) -> Cleaned:
    """Apply the course's cleaning decisions and account for every row removed."""
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    log = [("raw file", 0, len(df), "")]

    def step(name: str, keep: pd.Series, why: str) -> None:
        nonlocal df
        before = len(df)
        df = df[keep]
        log.append((name, before - len(df), len(df), why))

    # 1. The two sheets overlap for nine days; the overlap is stored twice, identically.
    seam = find_sheet_boundary(raw)
    if seam is not None:
        second_start = raw["InvoiceDate"].iloc[seam]
        in_overlap_copy = (df.index < seam) & (df["InvoiceDate"] >= second_start)
        step("sheet overlap", ~in_overlap_copy, "same rows stored in both sheets: would double-count")
    # 2. Postage, manual entries, bank charges, test items are not products.
    step("non-product codes", df["StockCode"].str.fullmatch(PRODUCT_CODE, na=False),
         "not products: postage, manual, bank charges, tests")
    # 3. No customer id: guest checkouts. Cannot be tied to a person across time.
    step("no customer id", df["Customer ID"].notna(), "guest checkouts: cannot track repeat behaviour")

    df = df.rename(columns=_RENAME)
    df["customer_id"] = df["customer_id"].astype(int)
    df["revenue"] = df["quantity"] * df["unit_price"]
    is_cancel = df["invoice"].str.startswith("C")

    returns = df[is_cancel & (df["quantity"] < 0) & (df["unit_price"] > 0)].copy()
    log.append(("set aside cancellations", int(is_cancel.sum()), len(df) - int(is_cancel.sum()),
                "kept separately as returns: they are behaviour, not purchases"))
    df = df[~is_cancel]
    step("zero price (free items)", (df["quantity"] > 0) & (df["unit_price"] > 0),
         "gifts and samples at price 0: no revenue, not purchases")

    # 6. A purchase cancelled within a day is not a purchase. Net the pair out, so it neither counts as
    #    "bought again" nor inflates spend (one such order was worth GBP 77,000 and ranked a customer #1).
    reversed_ids, pairs = fully_reversed_invoices(df, returns)
    sales_before, returns_before = len(df), len(returns)
    df = df[~df["invoice"].isin(reversed_ids)]
    matched = returns.merge(pairs.assign(_hit=True), on=["invoice", "stock_code"], how="left")["_hit"].fillna(False).to_numpy(dtype=bool)
    returns = returns[~matched]
    removed = (sales_before - len(df)) + (returns_before - len(returns))
    log.append(("orders cancelled within a day", removed, len(df),
                "purchase and its cancellation netted out: neither is behaviour that happened"))

    return Cleaned(
        sales=df.reset_index(drop=True),
        returns=returns.reset_index(drop=True),
        log=pd.DataFrame(log, columns=["step", "rows_removed", "rows_remaining", "why"]),
    )


def invoices(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice: customer, timestamp, revenue, item count, country."""
    return (sales.groupby(["customer_id", "invoice"], as_index=False)
            .agg(date=("invoice_date", "min"), revenue=("revenue", "sum"),
                 items=("quantity", "sum"), country=("country", "first")))


def make_snapshot(sales: pd.DataFrame, returns: pd.DataFrame, cutoff, *, features: str = "rfm",
                  horizon_days: int = HORIZON_DAYS, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Features and label for every active customer at one snapshot date.

    * population: customers with at least one purchase in the `lookback_days` before `cutoff`
    * features: computed ONLY from events strictly before `cutoff`
    * label `y`: 1 if the customer buys again in [cutoff, cutoff + horizon_days)
    * features="rfm": recency, frequency, monetary.  features="full": adds the richer set
    """
    if features not in {"rfm", "full"}:
        raise ValueError("features must be 'rfm' or 'full'")
    cutoff = pd.Timestamp(cutoff)
    day = pd.Timedelta(days=1)
    inv = invoices(sales)
    past = inv[inv["date"] < cutoff]
    look = past[past["date"] >= cutoff - lookback_days * day]
    pop = pd.Index(look["customer_id"].unique(), name="customer_id")
    g = look.groupby("customer_id")

    out = pd.DataFrame(index=pop)
    last = past.groupby("customer_id")["date"].max()
    out["recency_days"] = (cutoff - last.reindex(pop)).dt.total_seconds() / 86400
    out["frequency"] = g.size().reindex(pop)
    out["monetary"] = g["revenue"].sum().reindex(pop)

    if features == "full":
        def n_inv(lo: int, hi: int) -> pd.Series:
            w = past[(past["date"] >= cutoff - lo * day) & (past["date"] < cutoff - hi * day)]
            return w.groupby("customer_id").size().reindex(pop).fillna(0)

        out["freq_30d"], out["freq_90d"], out["freq_180d"] = n_inv(30, 0), n_inv(90, 0), n_inv(180, 0)
        out["freq_prev_90d"] = n_inv(180, 90)
        out["trend_90d"] = out["freq_90d"] - out["freq_prev_90d"]
        w90 = past[past["date"] >= cutoff - 90 * day]
        out["monetary_90d"] = w90.groupby("customer_id")["revenue"].sum().reindex(pop).fillna(0)
        out["avg_order_value"] = out["monetary"] / out["frequency"]
        out["items_per_order"] = g["items"].mean().reindex(pop)
        first = past.groupby("customer_id")["date"].min()
        out["tenure_days"] = (cutoff - first.reindex(pop)).dt.total_seconds() / 86400
        gaps = look.sort_values("date").groupby("customer_id")["date"].diff().dt.total_seconds() / 86400
        out["avg_gap_days"] = gaps.groupby(look.sort_values("date")["customer_id"]).mean().reindex(pop)
        out["recency_over_gap"] = (out["recency_days"] / out["avg_gap_days"]).replace([np.inf, -np.inf], np.nan)
        lines = sales[(sales["invoice_date"] < cutoff) & (sales["invoice_date"] >= cutoff - lookback_days * day)]
        out["n_products"] = lines.groupby("customer_id")["stock_code"].nunique().reindex(pop).fillna(0)
        ret = returns[(returns["invoice_date"] < cutoff) & (returns["invoice_date"] >= cutoff - lookback_days * day)]
        out["n_returns"] = ret.groupby("customer_id")["invoice"].nunique().reindex(pop).fillna(0)
        out["return_value_share"] = (-ret.groupby("customer_id")["revenue"].sum().reindex(pop).fillna(0)
                                     / out["monetary"]).clip(lower=0) + 0.0   # + 0.0 turns -0.0 into 0.0
        country = past.sort_values("date").groupby("customer_id")["country"].last().reindex(pop)
        out["is_uk"] = (country == "United Kingdom").astype(int)

    future = inv[(inv["date"] >= cutoff) & (inv["date"] < cutoff + horizon_days * day)]
    out["y"] = out.index.isin(future["customer_id"]).astype(int)
    return out


def make_snapshots(sales: pd.DataFrame, returns: pd.DataFrame, cutoffs, **kw) -> pd.DataFrame:
    """Stack several snapshots into one table with `customer_id` and `cutoff` columns."""
    parts = []
    for c in cutoffs:
        snap = make_snapshot(sales, returns, c, **kw).reset_index()
        snap.insert(1, "cutoff", pd.Timestamp(c))
        parts.append(snap)
    return pd.concat(parts, ignore_index=True)


class Splits(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def check_split_dates(data_end, *, horizon_days: int = HORIZON_DAYS) -> None:
    """Raise unless the chronological split is leak-free.

    Every label window must end before the next split begins (an "embargo" of one horizon),
    and the last label window must end before the data does.
    """
    h = pd.Timedelta(days=horizon_days)
    if max(TRAIN_CUTOFFS) + h > VALID_CUTOFF:
        raise ValueError("training label windows run into the validation period")
    if VALID_CUTOFF + h > TEST_CUTOFF:
        raise ValueError("validation label windows run into the test period")
    if TEST_CUTOFF + h > pd.Timestamp(data_end):
        raise ValueError("test label window runs past the end of the data")


def build_splits(sales: pd.DataFrame, returns: pd.DataFrame, *, features: str = "full") -> Splits:
    """The course's train / validation / test snapshots, after checking they cannot leak."""
    check_split_dates(sales["invoice_date"].max())
    return Splits(
        train=make_snapshots(sales, returns, TRAIN_CUTOFFS, features=features),
        valid=make_snapshots(sales, returns, [VALID_CUTOFF], features=features),
        test=make_snapshots(sales, returns, [TEST_CUTOFF], features=features),
    )
