import numpy as np
import pandas as pd
import pytest

from mlp_data.forecasting import backtest, make_features, mase, mape, rolling_origins, seasonal_naive, smape


def _series(n=96):
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_mape_is_undefined_at_zero_and_smape_is_bounded():
    assert np.isinf(mape([0.0, 10.0], [5.0, 10.0]))
    assert 0 <= smape([0.0, 10.0], [5.0, 10.0]) <= 2


def test_mase_of_seasonal_naive_on_its_own_training_data_is_one():
    y = np.random.default_rng(0).normal(size=60)
    assert mase(y[3:], y[:-3], y, m=3) == pytest.approx(1.0)


def test_seasonal_naive_repeats_the_last_cycle():
    y = _series()
    h = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=30, freq="h")
    f = seasonal_naive(y, h, 24)
    assert f.iloc[0] == y.iloc[-24] and f.iloc[24] == y.iloc[-24]


def test_features_refuse_lags_shorter_than_the_horizon():
    with pytest.raises(ValueError):
        make_features(_series(), min_lag=24, lags=(1, 24))


def test_features_use_only_data_available_at_the_origin():
    y = _series()
    X = make_features(y, min_lag=24, lags=(24, 48), windows=(24,))
    t = y.index[80]
    assert X.loc[t, "lag_24"] == y.loc[t - pd.Timedelta(hours=24)]
    assert X.loc[t, "mean_24"] == y.loc[t - pd.Timedelta(hours=47): t - pd.Timedelta(hours=24)].mean()


def test_backtest_never_shows_the_forecaster_the_future():
    y = _series()
    seen = []

    def fc(hist, idx):
        seen.append((hist.index[-1], idx[0]))
        return pd.Series(0.0, index=idx)

    origins = rolling_origins(y.index, first_origin="2023-01-03", step="D")
    out = backtest(y, fc, origins=origins, horizon=24)
    assert all(last < first for last, first in seen)
    assert out["step"].max() == 24 and len(out) == 48
