import numpy as np
import pandas as pd
import pytest

from mlp_data.forecasting import seasonal_mean as fc_seasonal_mean
from mlp_data.forecasting import backtest, holiday_aware_mean, make_features, mase, mape, rolling_origins, score, seasonal_mean, seasonal_naive, smape


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


def test_seasonal_mean_averages_the_same_hour_of_previous_weeks():
    idx = pd.date_range("2023-01-01", periods=24 * 7 * 3, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    horizon = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    f = seasonal_mean(y, horizon, weeks=2)
    t = horizon[0]
    assert f[t] == (y[t - pd.Timedelta(weeks=1)] + y[t - pd.Timedelta(weeks=2)]) / 2
    assert seasonal_mean(y, horizon, weeks=3, statistic="median")[t] == y[t - pd.Timedelta(weeks=2)]


def test_score_reports_four_metrics():
    bt = pd.DataFrame({"actual": [10.0, 20.0, 30.0], "forecast": [12.0, 18.0, 30.0]})
    out = score(bt, np.arange(1, 400, 3.0), m=1)
    assert set(out) == {"MAE", "RMSE", "sMAPE", "MASE"} and out["MAE"] == pytest.approx(4 / 3)


def test_smape_treats_zero_zero_as_perfect_and_zero_vs_positive_as_the_worst_case():
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert smape([0.0], [5.0]) == 2.0


def test_holiday_aware_mean_masks_holidays_and_uses_the_holiday_profile():
    idx = pd.date_range("2023-01-02", periods=24 * 7 * 5, freq="h")                 # five weeks from a Monday
    y = pd.Series(100.0, index=idx)
    y[y.index.date == pd.Timestamp("2023-01-09").date()] = 10.0                      # a holiday Monday is much quieter
    holidays = {pd.Timestamp("2023-01-09").date(), pd.Timestamp("2023-02-13").date()}
    horizon = pd.date_range("2023-02-06", periods=24, freq="h")                      # an ordinary Monday, 4 weeks after the holiday
    plain = fc_seasonal_mean(y, horizon)
    aware = holiday_aware_mean(y, horizon, holidays)
    assert plain.iloc[0] < 100 and aware.iloc[0] == 100.0                            # masked: the 10s no longer pull the average down
    on_holiday = pd.date_range("2023-02-13", periods=24, freq="h")
    longer = y.reindex(pd.date_range(idx[0], on_holiday[0] - pd.Timedelta(hours=1), freq="h"))
    assert holiday_aware_mean(longer, on_holiday, holidays).iloc[0] == 10.0           # the earlier holiday's level
    assert holiday_aware_mean(longer, on_holiday, holidays, holiday_profile=False).iloc[0] == 100.0
