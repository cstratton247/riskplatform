"""Purged expanding-window walk-forward evaluation (spec section 8.1).

For test year Y the training set is every eligible row whose LABEL was fully known strictly
before Jan 1 of Y (this purges the h-day overlap). Rows on or after `holdout_start` are
removed before anything else, so the final holdout can never influence an experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskplatform.models.benchmarks import ModelSpec

from .metrics import classification_metrics


@dataclass
class WalkForwardResult:
    preds: pd.DataFrame
    fold_info: dict
    trials: pd.DataFrame          # every hyperparameter configuration tried (inner CV)


@dataclass(frozen=True)
class Fold:
    year: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"wf_{self.year}"


def make_folds(first_year: int, last_year: int) -> list[Fold]:
    return [
        Fold(y, pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))
        for y in range(first_year, last_year + 1)
    ]


def eligible_rows(
    frame: pd.DataFrame, required_features: list[str], modeling_start, holdout_start
) -> pd.DataFrame:
    rows = frame[(frame.index >= pd.Timestamp(modeling_start)) & (frame.index < pd.Timestamp(holdout_start))]
    ok = rows["y"].notna() & rows[required_features].notna().all(axis=1)
    rows = rows[ok].copy()
    rows["y"] = rows["y"].astype(int)
    rows["label_known_date"] = pd.to_datetime(rows["label_known_date"])
    return rows


def split_fold(rows: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = rows[rows["label_known_date"] < fold.test_start]
    test = rows[(rows.index >= fold.test_start) & (rows.index <= fold.test_end)]
    return train, test


def run_walk_forward(
    frame: pd.DataFrame,
    models: dict[str, ModelSpec],
    folds: list[Fold],
    required_features: list[str],
    modeling_start,
    holdout_start,
) -> WalkForwardResult:
    """Every model sees exactly the same train/test rows."""
    rows = eligible_rows(frame, required_features, modeling_start, holdout_start)
    preds, info, trials = [], {}, []
    for fold in folds:
        train, test = split_fold(rows, fold)
        if train.empty or test.empty:
            continue
        info[fold.label] = {
            "train_start": train.index.min(), "train_end": train.index.max(),
            "n_train": len(train), "n_test": len(test),
        }
        for name, spec in models.items():
            n_before = len(spec.trial_log)
            p = spec.fit_predict(train, test)
            for t in spec.trial_log[n_before:]:
                trials.append({"model": name, "fold": fold.label, **t})
            preds.append(pd.DataFrame(
                {"model": name, "fold": fold.label, "as_of_date": test.index,
                 "prob": np.asarray(p, float), "y": test["y"].to_numpy()}
            ))
    trial_cols = ["model", "fold", "params", "metric", "score"]
    return WalkForwardResult(
        pd.concat(preds, ignore_index=True), info, pd.DataFrame(trials, columns=trial_cols)
    )


def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics per (model, scope), where scope is each fold plus 'pooled'."""
    out = []
    for model, g in preds.groupby("model", sort=False):
        for fold, gf in g.groupby("fold", sort=True):
            out.append({"model": model, "scope": fold, **classification_metrics(gf["y"], gf["prob"])})
        out.append({"model": model, "scope": "pooled", **classification_metrics(g["y"], g["prob"])})
    return pd.DataFrame(out)
