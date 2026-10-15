import numpy as np
import pandas as pd
import pytest

from mlp_data.tracking import ExperimentLog, fingerprint


def test_fingerprint_is_stable_and_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 0.25, 0.125]})
    assert fingerprint(df) == fingerprint(df.copy())
    changed = df.copy(); changed.loc[1, "b"] = 0.26
    assert fingerprint(changed) != fingerprint(df)
    assert fingerprint(df.rename(columns={"a": "z"})) != fingerprint(df)
    assert fingerprint(df.iloc[:2]) != fingerprint(df)


def test_log_records_provenance_and_survives_numpy_types():
    log = ExperimentLog()
    run = log.log("demo", {"depth": np.int64(4), "lr": np.float64(0.1)}, {"auc": np.float32(0.8)}, seed=0, data="abc123")
    assert run["params"] == {"depth": 4, "lr": 0.1} and run["data"] == "abc123" and run["seed"] == 0
    assert {"python", "numpy", "pandas", "scikit-learn", "git_commit"} <= set(run["env"])
    assert len(run["run_id"]) == 8 and run["time_utc"].endswith("+00:00")


def test_table_and_best():
    log = ExperimentLog()
    for name, auc in [("a", 0.70), ("b", 0.80), ("c", 0.75)]:
        log.log(name, {"x": 1}, {"auc": auc})
    t = log.table(sort_by="metric.auc")
    assert list(t["name"]) == ["b", "c", "a"] and "param.x" in t
    assert log.best("auc")["name"] == "b" and log.best("auc", higher_is_better=False)["name"] == "a"
    with pytest.raises(KeyError):
        log.best("nonexistent")


def test_jsonl_file_round_trip_appends_across_sessions(tmp_path):
    path = tmp_path / "runs.jsonl"
    ExperimentLog(path).log("first", {}, {"auc": 0.5})
    again = ExperimentLog(path)                       # a new session sees the earlier run
    again.log("second", {}, {"auc": 0.6})
    assert [r["name"] for r in ExperimentLog(path).runs] == ["first", "second"]
