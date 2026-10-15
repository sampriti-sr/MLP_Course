import numpy as np
import pytest

from mlp_data.metrics import at_risk_report


def test_perfect_ranking_hits_the_oracle_and_random_matches_expectation():
    y = np.array([1, 1, 0, 0, 0, 0])               # customers 2-5 lapse
    spend = np.array([5, 5, 100, 80, 60, 10.0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4])    # lapsers are the lowest p: perfect ranking
    r = at_risk_report(y, p, spend, k=2)
    assert r["AUC"] == 1.0 and r["precision@2"] == 1.0
    assert r["revenue@2"] == 180.0                  # calls customers with p = 0.1 and 0.2
    assert r["revenue@2 oracle"] == 180.0 and r["revenue@2 % of oracle"] == 1.0
    assert r["revenue@2 random"] == pytest.approx(2 / 6 * 250)


def test_risk_override_ranks_by_value_not_by_probability():
    y = np.array([1, 0, 0, 0])
    spend = np.array([1, 5, 500, 6.0])
    p = np.array([0.9, 0.2, 0.3, 0.1])
    by_prob = at_risk_report(y, p, spend, k=1)["revenue@1"]          # calls p=0.1 (spend 6)
    by_value = at_risk_report(y, p, spend, k=1, risk=(1 - p) * spend)["revenue@1"]   # calls spend 500
    assert (by_prob, by_value) == (6.0, 500.0)


def test_ties_are_broken_deterministically():
    y = np.array([0, 1, 0, 1]); p = np.full(4, 0.5); spend = np.array([1, 2, 3, 4.0])
    assert at_risk_report(y, p, spend, k=2) == at_risk_report(y, p, spend, k=2)


def test_non_probability_scores_skip_calibration_metrics():
    r = at_risk_report([1, 0, 1, 0], [10.0, -2.0, 7.0, 0.5], [1, 1, 1, 1], k=1)
    assert "brier" not in r and "AUC" in r


def test_calibration_metrics_only_on_request_and_validated():
    y, p, spend = [1, 0, 1, 0], [0.8, 0.3, 0.6, 0.2], [1, 1, 1, 1]
    assert "brier" not in at_risk_report(y, p, spend, k=1)
    r = at_risk_report(y, p, spend, k=1, probabilities=True)
    assert 0 < r["brier"] < 0.25 and r["log loss"] > 0
    with pytest.raises(ValueError):
        at_risk_report(y, [5.0, -1, 2, 0], spend, k=1, probabilities=True)


def test_random_calling_is_exact_and_not_tie_dependent():
    from mlp_data.metrics import random_calling_report
    y = np.array([1, 0, 0, 0]); spend = np.array([9, 10, 20, 30.0])
    r = random_calling_report(y, spend, train_rate=0.25, k=2)
    assert r["revenue@2"] == pytest.approx(2 / 4 * 60) and r["revenue@2 oracle"] == 50.0
    assert r["precision@2"] == 0.75 and r["lift@2"] == 1.0


def test_bootstrap_interval_brackets_the_point_estimate_and_is_reproducible():
    from mlp_data.metrics import bootstrap_interval
    rng = np.random.default_rng(1); n = 400
    y = rng.integers(0, 2, n); p = np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1); spend = rng.lognormal(5, 1, n)
    point = at_risk_report(y, p, spend, k=40)["AUC"]
    lo, hi = bootstrap_interval(y, p, spend, metric="AUC", k=40, n_boot=100)
    assert lo < point < hi and (lo, hi) == bootstrap_interval(y, p, spend, metric="AUC", k=40, n_boot=100)


def test_rule_based_baselines_can_pass_only_a_risk_score():
    y, spend = [1, 0, 0, 1], [1, 100, 50, 1]
    r = at_risk_report(y, None, spend, risk=[0.1, 0.9, 0.8, 0.2], k=2)
    assert r["precision@2"] == 1.0 and r["revenue@2"] == 150.0
    with pytest.raises(ValueError):
        at_risk_report(y, None, spend, k=2)


def test_ties_do_not_depend_on_row_order():
    rng = np.random.default_rng(0); n = 500
    y = rng.integers(0, 2, n); spend = rng.lognormal(4, 1, n); risk = np.zeros(n)          # everything tied
    a = at_risk_report(y, None, spend, risk=risk, k=50)["revenue@50"]
    perm = rng.permutation(n)
    b = at_risk_report(y[perm], None, spend[perm], risk=risk[perm], k=50)["revenue@50"]
    # With random tie-breaking the answer is a draw around the random-calling expectation either way,
    # not a systematic artefact of how the file was sorted (sorted-by-spend input used to give the maximum).
    order = np.argsort(-spend)
    sorted_input = at_risk_report(y[order], None, spend[order], risk=risk[order], k=50)["revenue@50"]
    expected = at_risk_report(y, None, spend, risk=risk, k=50)["revenue@50 random"]
    assert abs(sorted_input - expected) < 0.6 * expected and abs(a - expected) < 0.6 * expected and abs(b - expected) < 0.6 * expected


def test_paired_bootstrap_detects_a_clearly_better_method():
    from mlp_data.metrics import bootstrap_difference
    rng = np.random.default_rng(2); n = 600
    y = rng.integers(0, 2, n); spend = rng.lognormal(5, 1, n)
    good = (1 - y) * 1.0 + rng.random(n) * 0.3            # nearly separates lapsers
    bad = rng.random(n)                                    # noise
    d = bootstrap_difference(y, spend, good, bad, metric="AUC", k=60, n_boot=80)
    assert d["low"] > 0 and d["share_a_better"] == 1.0
