import pandas as pd

from mlp_data import retail
from mlp_data.pipeline import Config, run_pipeline

from test_serving import synthetic_sales


def synthetic_cleaned():
    sales = synthetic_sales(n_customers=300, seed=3)
    # make sure the data runs past the end of the test window so the split check passes
    last = sales.iloc[[0]].copy(); last["invoice"] = "last"; last["invoice_date"] = pd.Timestamp("2011-12-05 10:00"); last["customer_id"] = 1
    sales = pd.concat([sales, last], ignore_index=True).sort_values("invoice_date").reset_index(drop=True)
    return retail.Cleaned(sales=sales, returns=sales.iloc[:0].copy(), log=pd.DataFrame())


def test_same_config_and_data_give_identical_fingerprints_and_metrics():
    cleaned = synthetic_cleaned()
    a, b = run_pipeline(Config(), cleaned), run_pipeline(Config(), cleaned)
    assert a["predictions"] == b["predictions"] and a["metrics"] == b["metrics"] and a["data"] == b["data"]


def test_changing_the_config_changes_the_predictions_fingerprint():
    cleaned = synthetic_cleaned()
    assert run_pipeline(Config(C=1.0), cleaned)["predictions"] != run_pipeline(Config(C=0.001), cleaned)["predictions"]


def test_manifest_describes_what_was_run():
    m = run_pipeline(Config(k=50), synthetic_cleaned())
    assert m["config"]["k"] == 50 and {"cleaned_sales", "train", "valid", "test"} <= set(m["data"])
    assert "revenue@50" in m["metrics"] and "scikit-learn" in m["env"]
