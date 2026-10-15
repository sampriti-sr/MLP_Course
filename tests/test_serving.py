import numpy as np
import pandas as pd
import pytest

from mlp_data import models, retail, serving


def synthetic_sales(n_customers=250, seed=0):
    """Cleaned-schema sales for random customers over 2010-2011 (time-ordered, like the real file)."""
    rng = np.random.default_rng(seed)
    rows = []
    for cust in range(1, n_customers + 1):
        n = rng.integers(2, 14)
        dates = pd.to_datetime("2010-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 700, n)), unit="D")
        for k, d in enumerate(dates):
            rows.append((f"{cust}-{k}", "10001", int(rng.integers(1, 20)), d + pd.Timedelta(hours=10), float(rng.uniform(1, 9)), cust, "United Kingdom"))
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id", "country"])
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["description"] = "thing"
    return df.sort_values("invoice_date").reset_index(drop=True)


@pytest.fixture(scope="module")
def world():
    sales = synthetic_sales()
    returns = sales.iloc[:0].copy()
    train = pd.concat([retail.make_snapshot(sales, returns, c, features="full").reset_index() for c in ["2010-12-01", "2011-01-01"]])
    model = models.make_logistic().fit(train[models.FEATURES], train["y"])
    return sales, returns, model


def test_scores_are_ranked_probabilities(world):
    sales, returns, model = world
    out = serving.score_customers(model, sales, returns, "2011-06-01")
    assert out["call_rank"].tolist() == list(range(1, len(out) + 1))
    assert out["value_at_risk"].is_monotonic_decreasing and ((out["p_return"] >= 0) & (out["p_return"] <= 1)).all()
    assert out["customer_id"].is_unique


def test_data_after_the_cutoff_cannot_change_a_score(world):
    sales, returns, model = world
    cutoff = pd.Timestamp("2011-06-01")
    full = serving.score_customers(model, sales, returns, cutoff)
    past_only = serving.score_customers(model, sales[sales["invoice_date"] < cutoff], returns, cutoff)
    pd.testing.assert_frame_equal(full, past_only)


def test_save_load_round_trip_and_tamper_detection(world, tmp_path):
    sales, returns, model = world
    card = serving.save_model(model, tmp_path, "m", metrics={"auc": 0.8}, training={"rows": 10})
    again, card2 = serving.load_model(tmp_path, "m")
    a = serving.score_customers(model, sales, returns, "2011-06-01"); b = serving.score_customers(again, sales, returns, "2011-06-01")
    pd.testing.assert_frame_equal(a, b)
    assert card["sha256"] == card2["sha256"] and card["features"] == models.FEATURES
    (tmp_path / "m.joblib").write_bytes((tmp_path / "m.joblib").read_bytes() + b"x")          # tamper
    with pytest.raises(ValueError, match="does not match the hash"):
        serving.load_model(tmp_path, "m")


@pytest.mark.parametrize("mutate, message", [
    (lambda s, r, c: (s.drop(columns=["revenue"]), r, c), "missing required columns"),
    (lambda s, r, c: (s.assign(invoice_date=s["invoice_date"].astype(str)), r, c), "datetime"),
    (lambda s, r, c: (s, r, "not a date"), "not a date"),
    (lambda s, r, c: (s, r, "2005-01-01"), "before the first record"),
    (lambda s, r, c: (s, r, "2030-01-01"), "stale"),
    (lambda s, r, c: (s.iloc[:0], r, c), "empty"),
])
def test_bad_inputs_fail_early_with_a_clear_message(world, mutate, message):
    sales, returns, model = world
    s, r, c = mutate(sales, returns, "2011-06-01")
    with pytest.raises(ValueError, match=message):
        serving.score_customers(model, s, r, c)
