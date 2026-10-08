"""Tuned models (spec section 7, rungs 5-7) with purged inner cross-validation.

Hyperparameters are chosen ONLY inside each outer training window, using expanding-window
inner folds that validate on each of the last `n_val_years` calendar years of that window;
inner training rows must have labels known before the validation year starts (purging).
The inner metric is log loss (a proper scoring rule). Every configuration tried is logged
in the spec's `trial_log`, and later written to ml.model_trial.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from riskplatform.features.compute import BASE_FEATURES

from .benchmarks import ModelSpec, _design

SEED = 0


def inner_splits(train: pd.DataFrame, n_val_years: int = 3, min_train_rows: int = 500):
    """Expanding purged splits inside a training frame."""
    years = sorted(set(train.index.year))[-n_val_years:]
    for y in years:
        val_start = pd.Timestamp(y, 1, 1)
        tr = train[train["label_known_date"] < val_start]
        va = train[train.index.year == y]
        if len(tr) >= min_train_rows and len(va) > 0 and tr["y"].nunique() == 2:
            yield tr, va


def _grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]


def tuned_model(
    name: str, family: str, features: list[str], make_estimator: Callable[[dict], object],
    space: dict, log_transform: bool, n_val_years: int = 3,
) -> ModelSpec:
    grid = _grid(space)

    def X(frame):
        return _design(frame, features) if log_transform else frame[features].astype(float).to_numpy()

    def fit_predict(train, test):
        splits = list(inner_splits(train, n_val_years))
        if not splits:
            raise RuntimeError("no valid inner splits; training window too short to tune")
        scored = []
        for params in grid:
            losses = []
            for tr, va in splits:
                est = make_estimator(params).fit(X(tr), tr["y"].astype(int).to_numpy())
                p = np.clip(est.predict_proba(X(va))[:, 1], 1e-6, 1 - 1e-6)
                losses.append(log_loss(va["y"].astype(int), p, labels=[0, 1]))
            score = float(np.mean(losses))
            scored.append((score, params))
            spec.trial_log.append({"params": params, "metric": "log_loss", "score": score})
        best = min(scored, key=lambda t: t[0])[1]
        final = make_estimator(best).fit(X(train), train["y"].astype(int).to_numpy())
        return final.predict_proba(X(test))[:, 1]

    spec = ModelSpec(
        name, family, features,
        {"grid": space, "inner_metric": "log_loss", "inner_val_years": n_val_years}, fit_predict,
    )
    return spec


def logit_tuned() -> ModelSpec:
    return tuned_model(
        "logit_tuned", "logistic", list(BASE_FEATURES),
        lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000)),
        {"C": [0.01, 0.1, 1.0]}, log_transform=True,
    )


def hist_gb() -> ModelSpec:
    return tuned_model(
        "hist_gb", "gradient_boosting", list(BASE_FEATURES),
        lambda p: HistGradientBoostingClassifier(
            max_depth=p["max_depth"], learning_rate=p["learning_rate"], max_iter=150,
            min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=SEED),
        {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]}, log_transform=False,
    )


def random_forest() -> ModelSpec:
    return tuned_model(
        "random_forest", "random_forest", list(BASE_FEATURES),
        lambda p: RandomForestClassifier(
            n_estimators=300, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            max_features="sqrt", random_state=SEED, n_jobs=-1),
        {"max_depth": [4, 8], "min_samples_leaf": [20, 50]}, log_transform=False,
    )


def full_ladder() -> dict[str, ModelSpec]:
    from .benchmarks import default_ladder

    ladder = default_ladder()
    for spec in (logit_tuned(), hist_gb(), random_forest()):
        ladder[spec.name] = spec
    return ladder
