import io

import joblib
import numpy as np
import pandas as pd

from mlp_data import models


def toy_snapshot(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.lognormal(2, 1, n) for c in models.HEAVY_TAILED})
    df["recency_over_gap"] = rng.lognormal(0, 2, n)
    df.loc[rng.random(n) < 0.3, ["avg_gap_days", "recency_over_gap"]] = np.nan       # single-order customers have no rhythm
    df["trend_90d"] = rng.integers(-5, 6, n).astype(float)
    df["return_value_share"] = rng.random(n) * 0.2
    df["is_uk"] = rng.integers(0, 2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(np.log1p(df["frequency"]) - 2)))).astype(int)
    return df, y


def test_both_models_fit_handle_missing_values_and_give_probabilities():
    X, y = toy_snapshot()
    for model in (models.make_logistic(), models.make_boosting()):
        p = model.fit(X[models.FEATURES], y).predict_proba(X[models.FEATURES])[:, 1]
        assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all() and not np.isnan(p).any()


def test_fitted_pipeline_survives_a_save_and_load_round_trip():
    X, y = toy_snapshot()
    model = models.make_logistic().fit(X[models.FEATURES], y)
    buf = io.BytesIO(); joblib.dump(model, buf); buf.seek(0)          # lambdas would fail here
    again = joblib.load(buf)
    assert np.allclose(model.predict_proba(X[models.FEATURES]), again.predict_proba(X[models.FEATURES]))


def test_missing_feature_is_reported_by_name():
    import pytest
    with pytest.raises(KeyError, match="frequency"):
        models.feature_columns(pd.DataFrame({"recency_days": [1.0]}))


def test_overdue_cap_is_a_parameter_and_still_picklable():
    X, y = toy_snapshot()
    a = models.make_logistic(overdue_cap=50).fit(X[models.FEATURES], y).predict_proba(X[models.FEATURES])[:, 1]
    b = models.make_logistic(overdue_cap=5).fit(X[models.FEATURES], y)
    buf = io.BytesIO(); joblib.dump(b, buf); buf.seek(0)
    assert not np.allclose(a, joblib.load(buf).predict_proba(X[models.FEATURES])[:, 1])      # the cap changes the model


def test_every_model_input_has_a_readable_name_for_interpretation():
    X, y = toy_snapshot()
    model = models.make_logistic().fit(X[models.FEATURES], y)
    names = list(model[0].get_feature_names_out())
    assert len(names) == model[-1].coef_.shape[1]                        # one name per coefficient
    assert "tails__frequency" in names and "rest__is_uk" in names
    assert any("missingindicator" in n for n in names)                   # the missing-value indicators are visible too
