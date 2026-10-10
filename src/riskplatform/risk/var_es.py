"""One-day VaR and Expected Shortfall (spec section 10). Database-free.

Conventions
  * Returns are simple or log daily returns; VaR and ES are reported as POSITIVE LOSSES.
  * Every method returns series indexed by the TARGET day: the number applied to day t's return
    was computed using information through day t-1 only.
  * Fixed, pre-declared settings (window, EWMA lambda, t degrees of freedom); nothing is tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from riskplatform.models.garch import (
    GJRParams, fit_gjr, next_variance, variance_path,
)


def _tail(var: pd.Series, es: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"var": var, "es": es})


def historical(r: pd.Series, conf: float, window: int = 500) -> pd.DataFrame:
    x = r.to_numpy(float)
    var = np.full(len(x), np.nan)
    es = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        w = x[t - window:t]
        q = np.quantile(w, 1 - conf)
        var[t] = -q
        es[t] = -w[w <= q].mean()
    return _tail(pd.Series(var, r.index), pd.Series(es, r.index))


def ewma_sigma(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """Daily volatility known at the close of t (includes r_t)."""
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def normal(r: pd.Series, conf: float, lam: float = 0.94) -> pd.DataFrame:
    sigma = ewma_sigma(r, lam).shift(1)                     # applied to the NEXT day
    z = stats.norm.ppf(conf)
    return _tail(z * sigma, sigma * stats.norm.pdf(z) / (1 - conf))


def _t_scale(nu: float) -> float:
    return np.sqrt(nu / (nu - 2))


def t_multipliers(conf: float, nu: float = 5.0) -> tuple[float, float]:
    """(VaR, ES) multipliers of sigma for UNIT-VARIANCE Student-t shocks."""
    q = stats.t.ppf(conf, nu)
    var_mult = q / _t_scale(nu)
    es_mult = stats.t.pdf(q, nu) / (1 - conf) * (nu + q ** 2) / (nu - 1) / _t_scale(nu)
    return float(var_mult), float(es_mult)


def student_t(r: pd.Series, conf: float, lam: float = 0.94, nu: float = 5.0) -> pd.DataFrame:
    """Unit-variance Student-t shocks scaled by EWMA volatility (fixed degrees of freedom)."""
    sigma = ewma_sigma(r, lam).shift(1)
    var_mult, es_mult = t_multipliers(conf, nu)
    return _tail(var_mult * sigma, es_mult * sigma)


def filtered_historical(
    r: pd.Series, confs: list[float], window: int = 500, min_fit_obs: int = 500,
    start: str | None = None,
) -> dict[float, pd.DataFrame]:
    """GJR-GARCH filtered historical simulation.

    Parameters are re-estimated on January 1 of each year using returns through the prior
    December only, then the variance recursion runs causally. The VaR for day t is
    sqrt(h_t) times the empirical quantile of the previous `window` standardized residuals.
    """
    x = r.to_numpy(float) * 100
    n = len(x)
    out = {c: (np.full(n, np.nan), np.full(n, np.nan)) for c in confs}
    years = sorted(set(r.index.year))
    first_eval = pd.Timestamp(start) if start else r.index[0]
    for year in years:
        in_year = np.flatnonzero(r.index.year == year)
        if len(in_year) == 0 or r.index[in_year[-1]] < first_eval:
            continue
        fit_end = in_year[0]                                  # estimation uses days strictly before this year
        if fit_end < min_fit_obs:
            continue
        fit = x[:fit_end]
        params = fit_gjr(fit)
        h0 = float(np.var(fit))
        h = variance_path(x, params, h0)
        z = x / np.sqrt(h)
        for t in in_year:
            if t < window or r.index[t] < first_eval:
                continue
            w = z[t - window:t]
            sigma = np.sqrt(h[t]) / 100
            for c in confs:
                q = np.quantile(w, 1 - c)
                out[c][0][t] = -sigma * q
                out[c][1][t] = -sigma * w[w <= q].mean()
    return {c: _tail(pd.Series(v, r.index), pd.Series(e, r.index)) for c, (v, e) in out.items()}


def gjr_monte_carlo(
    params: GJRParams, h_next_pct2: float, std_resid: np.ndarray, conf: float,
    horizon: int = 10, paths: int = 100_000, seed: int = 0,
) -> tuple[float, float]:
    """Horizon VaR/ES (as decimal loss) of the cumulative return by simulating GJR-GARCH paths.

    `h_next_pct2` is the next-day variance (percent^2); shocks are bootstrapped from
    standardized residuals. Fully vectorized across paths.
    """
    rng = np.random.default_rng(seed)
    h = np.full(paths, h_next_pct2, float)
    total = np.zeros(paths)
    for _ in range(horizon):
        z = rng.choice(std_resid, size=paths, replace=True)
        ret = np.sqrt(h) * z
        total += ret
        h = params.omega + (params.alpha + params.gamma * (ret < 0)) * ret ** 2 + params.beta * h
    total /= 100
    q = np.quantile(total, 1 - conf)
    return float(-q), float(-total[total <= q].mean())
