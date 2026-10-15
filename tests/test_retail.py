"""Offline unit tests for mlp_data.retail: tiny hand-made tables, no downloads."""
import numpy as np
import pandas as pd
import pytest

from mlp_data import retail


def raw_frame(rows):
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def line(inv, code, qty, when, price=2.0, cust=1.0, country="United Kingdom"):
    return (inv, code, "thing", qty, when, price, cust, country)


def test_sheet_boundary_found_and_overlap_removed_once():
    # Like the real file: sheet 1 runs to 12-09, sheet 2 restarts at 12-05, so time jumps backwards
    # once and the 12-05..12-09 rows are stored in both sheets.
    sheet1 = [line("1", "10001", 1, "2010-12-01 09:00"), line("2", "10001", 1, "2010-12-05 09:00"),
              line("3", "10001", 1, "2010-12-09 09:00")]
    sheet2 = [line("2", "10001", 1, "2010-12-05 09:00"), line("3", "10001", 1, "2010-12-09 09:00"),
              line("4", "10001", 1, "2010-12-20 09:00")]
    raw = raw_frame(sheet1 + sheet2)
    assert retail.find_sheet_boundary(raw) == 3
    sales = retail.clean_retail(raw).sales
    assert sorted(sales["invoice"]) == ["1", "2", "3", "4"]   # each invoice exactly once
    assert retail.clean_retail(raw).log.set_index("step").loc["sheet overlap", "rows_removed"] == 2


def test_no_seam_means_no_overlap_step():
    raw = raw_frame([line("1", "10001", 1, "2010-12-01"), line("2", "10001", 1, "2010-12-02")])
    assert retail.find_sheet_boundary(raw) is None
    assert "sheet overlap" not in retail.clean_retail(raw).log["step"].tolist()


def test_cleaning_rules_and_row_accounting():
    raw = raw_frame([
        line("10", "10001", 3, "2011-01-01"),                      # good sale
        line("11", "POST", 1, "2011-01-01"),                        # not a product
        line("12", "10002", 1, "2011-01-01", cust=np.nan),          # guest checkout
        line("C13", "10001", -1, "2011-01-02"),                     # return
        line("14", "10003", -5, "2011-01-03"),                      # negative qty without a C invoice
        line("15", "10004", 2, "2011-01-03", price=0.0),            # free / adjustment
        line("16", "85123A", 1, "2011-01-04"),                      # letter-suffixed product is kept
    ])
    c = retail.clean_retail(raw)
    assert sorted(c.sales["invoice"]) == ["10", "16"]
    assert c.returns["invoice"].tolist() == ["C13"]
    assert c.log.iloc[0]["rows_remaining"] == 7
    # every removed row is accounted for: raw - removed (excluding the moved returns) == sales
    removed = c.log[c.log["step"] != "set aside cancellations"]["rows_removed"].sum()
    assert 7 - removed - len(c.returns) == len(c.sales)


def make_sales():
    rows = [
        line("1", "10001", 1, "2010-06-01", cust=1), line("2", "10001", 1, "2010-12-15", cust=1),
        line("3", "10001", 2, "2011-01-10", cust=1),      # inside the label window of cutoff 2011-01-01
        line("4", "10001", 1, "2010-12-20", cust=2),      # active, never returns
        line("5", "10001", 1, "2009-12-02", cust=3),      # last purchase > 365d before cutoff: inactive
    ]
    raw = raw_frame(rows)
    c = retail.clean_retail(raw)
    return c.sales, c.returns


def test_population_label_and_no_future_in_features():
    sales, returns = make_sales()
    snap = retail.make_snapshot(sales, returns, "2011-01-01")
    assert set(snap.index) == {1, 2}                          # customer 3 is outside the lookback
    assert snap.loc[1, "y"] == 1 and snap.loc[2, "y"] == 0
    assert snap.loc[1, "frequency"] == 2                      # the 2011-01-10 order is the future
    assert snap.loc[1, "recency_days"] == pytest.approx(17.0, abs=1e-6)


def test_features_do_not_change_when_future_data_changes():
    sales, returns = make_sales()
    base = retail.make_snapshot(sales, returns, "2011-01-01", features="full")
    extra = sales.iloc[[0]].copy()
    extra["invoice"], extra["invoice_date"], extra["customer_id"] = "999", pd.Timestamp("2011-02-01"), 2
    sales2 = pd.concat([sales, extra], ignore_index=True)
    after = retail.make_snapshot(sales2, returns, "2011-01-01", features="full")
    pd.testing.assert_frame_equal(base.drop(columns="y"), after.drop(columns="y"))
    assert base.loc[2, "y"] == 0 and after.loc[2, "y"] == 1   # only the label moved


def test_label_window_is_half_open():
    sales, returns = make_sales()
    on_cutoff = sales.iloc[[0]].copy()
    on_cutoff["invoice"], on_cutoff["invoice_date"], on_cutoff["customer_id"] = "888", pd.Timestamp("2011-01-01"), 2
    snap = retail.make_snapshot(pd.concat([sales, on_cutoff], ignore_index=True), returns, "2011-01-01")
    assert snap.loc[2, "y"] == 1                              # on the cutoff counts as the future...
    assert snap.loc[2, "frequency"] == 1                      # ...and is not in the features


def test_make_snapshots_stacks_with_cutoff_column():
    sales, returns = make_sales()
    both = retail.make_snapshots(sales, returns, ["2011-01-01", "2011-02-01"])
    assert set(both["cutoff"]) == {pd.Timestamp("2011-01-01"), pd.Timestamp("2011-02-01")}
    assert {"customer_id", "cutoff", "y"} <= set(both.columns)


def test_bad_feature_set_rejected():
    sales, returns = make_sales()
    with pytest.raises(ValueError):
        retail.make_snapshot(sales, returns, "2011-01-01", features="nope")


def test_course_split_dates_are_leak_free_for_the_real_data_end():
    retail.check_split_dates("2011-12-09")           # must not raise


def test_split_check_rejects_overlapping_windows(monkeypatch):
    monkeypatch.setattr(retail, "VALID_CUTOFF", pd.Timestamp("2011-04-15"))   # inside the last train window
    with pytest.raises(ValueError, match="training label windows"):
        retail.check_split_dates("2011-12-09")
    monkeypatch.setattr(retail, "VALID_CUTOFF", pd.Timestamp("2011-06-01"))
    with pytest.raises(ValueError, match="test label window"):
        retail.check_split_dates("2011-11-15")        # data ends before the test window does


def test_build_splits_have_identical_columns_and_disjoint_dates():
    rows = [line(str(i), "10001", 1, d, cust=float(c))
            for i, (c, d) in enumerate([(1, "2010-03-01"), (1, "2011-02-20"), (1, "2011-05-20"), (1, "2011-08-20"), (1, "2011-11-01"),
                                        (2, "2010-03-01"), (2, "2011-08-25"), (2, "2011-12-05")])]   # data runs past the test window
    rows.sort(key=lambda r: r[4])                 # the real file is time-ordered; an unsorted fixture would look like a sheet seam
    c = retail.clean_retail(raw_frame(rows))
    sp = retail.build_splits(c.sales, c.returns, features="full")
    assert list(sp.train.columns) == list(sp.valid.columns) == list(sp.test.columns)
    assert set(sp.valid["cutoff"]) == {retail.VALID_CUTOFF} and set(sp.test["cutoff"]) == {retail.TEST_CUTOFF}
    assert sp.train["cutoff"].max() < sp.valid["cutoff"].min() < sp.test["cutoff"].min()
