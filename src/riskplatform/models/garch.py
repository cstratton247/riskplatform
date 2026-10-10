"""GJR-GARCH(1,1), zero mean, Gaussian quasi-maximum-likelihood. Implemented with scipy so it can be
tested without extra dependencies (a cross-check against `arch` lives in tests/test_garch.py).

All series inside this module are in PERCENT returns (r * 100) for numerical stability:
    h_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * h_{t-1}
h_t is the variance of r_t given information through t-1, so every quantity here is causal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import ModelSpec


@dataclass(frozen=True)
class GJRParams:
    omega: float
    alpha: float
    gamma: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.persistence)


def variance_path(r: np.ndarray, p: GJRParams, h0: float) -> np.ndarray:
    """h[t] = variance of r[t] given information through t-1; h[0] = h0."""
    r = np.asarray(r, float)
    n = len(r)
    if n == 1:
        return np.array([h0])
    shock = (p.alpha + p.gamma * (r[:-1] < 0)) * r[:-1] ** 2
    u = p.omega + shock                                        # u[t-1] drives h[t]
    h_rest = lfilter([1.0], [1.0, -p.beta], u, zi=[p.beta * h0])[0]
    return np.r_[h0, h_rest]


def neg_loglik(theta: np.ndarray, r: np.ndarray, h0: float) -> float:
    p = GJRParams(*theta)
    h = variance_path(r, p, h0)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(h) + r ** 2 / h))


def fit_gjr(r_pct: np.ndarray) -> GJRParams:
    r = np.asarray(r_pct, float)
    h0 = float(np.var(r))
    best, best_val = None, np.inf
    for a0, g0, b0 in ((0.05, 0.05, 0.88), (0.03, 0.10, 0.90), (0.08, 0.02, 0.85)):
        w0 = h0 * (1 - (a0 + g0 / 2 + b0))
        res = minimize(
            neg_loglik, x0=np.array([w0, a0, g0, b0]), args=(r, h0), method="SLSQP",
            bounds=[(1e-6, None), (0.0, 0.5), (0.0, 0.8), (0.0, 0.9999)],
            constraints=[{"type": "ineq", "fun": lambda t: 0.9995 - (t[1] + t[2] / 2 + t[3])}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if res.fun < best_val and res.success is not False:
            best, best_val = res.x, res.fun
    if best is None:
        raise RuntimeError("GJR-GARCH fit failed to converge")
    return GJRParams(*best)


def next_variance(r: np.ndarray, p: GJRParams, h: np.ndarray) -> np.ndarray:
    """h_{t+1} given information through t, for every origin t."""
    r = np.asarray(r, float)
    return p.omega + (p.alpha + p.gamma * (r < 0)) * r ** 2 + p.beta * h


def avg_variance_forecast(r: np.ndarray, p: GJRParams, h0: float, horizon: int = 5) -> np.ndarray:
    """Mean of the 1..horizon-step-ahead variance forecasts from each origin t (analytic)."""
    h = variance_path(r, p, h0)
    v = next_variance(r, p, h)
    total = v.copy()
    for _ in range(2, horizon + 1):
        v = p.omega + p.persistence * v
        total += v
    return total / horizon


def garch_logit(returns: pd.Series, horizon: int = 5) -> ModelSpec:
    """Benchmark rung 3: logistic calibration of the GJR-GARCH 5-day volatility forecast.

    Parameters are estimated ONLY on returns through the last training day; the variance
    recursion is then run causally over the whole series, so a forecast at date t uses
    returns through t only. `returns` must already exclude the final holdout.
    """
    r_all = returns.dropna().astype(float)

    def fit_predict(train, test):
        fit = r_all.loc[: train.index.max()] * 100
        params = fit_gjr(fit.to_numpy())
        h0 = float(fit.var())
        r_pct = r_all * 100
        var5 = avg_variance_forecast(r_pct.to_numpy(), params, h0, horizon)
        score = pd.Series(np.log(np.sqrt(252 * var5 / 1e4)), index=r_all.index)
        cal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cal.fit(score.loc[train.index].to_numpy().reshape(-1, 1), train["y"].astype(int).to_numpy())
        return cal.predict_proba(score.loc[test.index].to_numpy().reshape(-1, 1))[:, 1]

    return ModelSpec(
        "garch_logit", "garch", [],
        {"model": "GJR-GARCH(1,1) Gaussian QMLE", "horizon": horizon,
         "calibration": "logistic on log annualized forecast volatility"},
        fit_predict,
    )
