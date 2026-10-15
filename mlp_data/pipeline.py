"""The whole project as one function: raw data in, a fingerprinted result out.

    python -m mlp_data.pipeline            # prints the run manifest as JSON

Everything that can change a number lives in `Config`; everything that identifies what was run goes
in the manifest. Two runs with the same config, data and library versions must give the same
`predictions` fingerprint, which the Reproducibility page checks, including from a fresh process.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from . import models, retail
from .loaders import load_retail
from .metrics import at_risk_report
from .tracking import environment, fingerprint


@dataclass(frozen=True)
class Config:
    C: float = 1.0                 # logistic regularisation strength
    overdue_cap: float = 50        # cap on the exploding overdue ratio
    k: int = 200                   # call budget
    seed: int = 0                  # seeds the tie-breaking in the ranking metrics


def run_pipeline(config: Config = Config(), cleaned: retail.Cleaned | None = None) -> dict:
    """Clean, split, fit on train + validation, score the test snapshot, and describe exactly what happened."""
    cleaned = cleaned or retail.clean_retail(load_retail())
    sp = retail.build_splits(cleaned.sales, cleaned.returns, features="full")
    labelled = pd.concat([sp.train, sp.valid], ignore_index=True)
    model = models.make_logistic(C=config.C, overdue_cap=config.overdue_cap).fit(labelled[models.FEATURES], labelled["y"])
    p = model.predict_proba(sp.test[models.FEATURES])[:, 1]
    ranking = at_risk_report(sp.test["y"], p, sp.test["monetary"], k=config.k, probabilities=True, tie_seed=config.seed)           # how good are the scores?
    money = at_risk_report(sp.test["y"], p, sp.test["monetary"], k=config.k, risk=(1 - p) * sp.test["monetary"], tie_seed=config.seed)   # how good is the call list?
    k = config.k
    metrics = {"AUC": ranking["AUC"], "Brier score": ranking["brier"], "log loss": ranking["log loss"],
               f"precision@{k}": money[f"precision@{k}"], f"revenue@{k}": money[f"revenue@{k}"], f"revenue@{k} % of oracle": money[f"revenue@{k} % of oracle"]}
    predictions = pd.DataFrame({"customer_id": sp.test["customer_id"].to_numpy(), "p_return": p})
    return {"config": asdict(config),
            "data": {"cleaned_sales": fingerprint(cleaned.sales), "train": fingerprint(sp.train), "valid": fingerprint(sp.valid), "test": fingerprint(sp.test)},
            "predictions": fingerprint(predictions),
            "metrics": {name: float(v) for name, v in metrics.items()},
            "env": {k: v for k, v in environment().items() if k != "git_commit"}}


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2, default=float))
