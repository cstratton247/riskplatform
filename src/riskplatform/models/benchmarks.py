"""Benchmark ladder, rungs 0-5 (spec section 7).

All hyperparameters are fixed and pre-declared; nothing is tuned on test folds. Strictly
positive volatility-type inputs are log-transformed (a monotone transform), then standardized
using the TRAINING window only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from riskplatform.features.compute import BASE_FEATURES

LOG_FEATURES = {"rv_5d", "rv_21d", "rv_63d", "vix_level", "parkinson_21d", "garman_klass_21d"}


@dataclass
class ModelSpec:
    name: str
    family: str                      # must match ml.model_version.family
    features: list[str]
    hyperparameters: dict
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def _design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = frame[features].astype(float).copy()
    for c in features:
        if c in LOG_FEATURES:
            x[c] = np.log(x[c].clip(lower=1e-8))
    return x.to_numpy()


def prevalence_model() -> ModelSpec:
    def fit_predict(train, test):
        return np.full(len(test), train["y"].astype(int).mean())
    return ModelSpec("prevalence", "prevalence", [], {}, fit_predict)


def logit_model(name: str, family: str, features: list[str], C: float = 1.0) -> ModelSpec:
    def fit_predict(train, test):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(_design(train, features), train["y"].astype(int).to_numpy())
        return model.predict_proba(_design(test, features))[:, 1]
    return ModelSpec(name, family, features, {"C": C, "penalty": "l2"}, fit_predict)


def default_ladder() -> dict[str, ModelSpec]:
    specs = [
        prevalence_model(),
        logit_model("persistence_logit", "persistence", ["rv_5d"]),
        logit_model("har_logit", "har", ["rv_5d", "rv_21d", "rv_63d"]),
        logit_model("vix_logit", "vix", ["vix_level"]),
        logit_model("logit_full", "logistic", list(BASE_FEATURES)),
    ]
    return {s.name: s for s in specs}
