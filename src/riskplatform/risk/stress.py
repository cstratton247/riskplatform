"""Stress testing and portfolio metrics (spec section 10). Database-free."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from riskplatform.strategy.overlay import max_drawdown


def portfolio_returns(asset_rets: pd.DataFrame, weights: dict) -> pd.Series:
    """Daily-rebalanced portfolio return."""
    w = pd.Series(weights, dtype=float)
    return asset_rets[w.index].mul(w, axis=1).sum(axis=1, min_count=len(w))


def replay(port: pd.Series, market: pd.Series, windows: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in windows.items():
        p, m = port.loc[pd.Timestamp(a):pd.Timestamp(b)], market.loc[pd.Timestamp(a):pd.Timestamp(b)]
        rows.append({"scenario": name, "days": len(p),
                     "portfolio_return": float((1 + p).prod() - 1), "portfolio_max_dd": max_drawdown(p),
                     "portfolio_worst_day": float(p.min()),
                     "market_return": float((1 + m).prod() - 1), "market_max_dd": max_drawdown(m)})
    return pd.DataFrame(rows).set_index("scenario")


def betas(asset_rets: pd.DataFrame, market: pd.Series) -> pd.Series:
    df = asset_rets.join(market.rename("_mkt")).dropna()
    var_m = df["_mkt"].var()
    return pd.Series({c: df[c].cov(df["_mkt"]) / var_m for c in asset_rets.columns})


def equity_shock_loss(weights: dict, beta: pd.Series, shock: float) -> float:
    """Instantaneous portfolio return if the market falls by `shock` (e.g. -0.20) and each asset
    moves by beta * shock. A linear scenario, so it understates convexity and a correlation
    breakdown; the stressed-covariance scenario below addresses the latter."""
    return float(sum(w * beta[a] * shock for a, w in weights.items()))


def stressed_normal_var_es(
    cov: pd.DataFrame, weights: dict, conf: float, vol_multiplier: float = 1.0,
    corr_blend: float = 0.0,
) -> dict:
    """Gaussian one-day VaR/ES after scaling volatilities and blending correlations toward 1."""
    names = list(weights)
    c = cov.loc[names, names].to_numpy()
    sd = np.sqrt(np.diag(c))
    rho = c / np.outer(sd, sd)
    rho = rho + corr_blend * (1 - rho)
    np.fill_diagonal(rho, 1.0)
    sd = sd * vol_multiplier
    w = np.array([weights[n] for n in names])
    sigma = float(np.sqrt(w @ (np.outer(sd, sd) * rho) @ w))
    z = stats.norm.ppf(conf)
    return {"sigma": sigma, "var": z * sigma, "es": sigma * stats.norm.pdf(z) / (1 - conf)}


def portfolio_metrics(
    asset_rets: pd.DataFrame, weights: dict, market: pd.Series, cash: pd.Series, confs=(0.95, 0.99),
) -> pd.Series:
    p = portfolio_returns(asset_rets, weights).dropna()
    df = pd.concat([p.rename("p"), market.rename("m"), cash.rename("c")], axis=1).dropna()
    excess = df["p"] - df["c"]
    out = {
        "ann_return": float((1 + df["p"]).prod() ** (252 / len(df)) - 1),
        "ann_vol": float(df["p"].std() * np.sqrt(252)),
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(252)),
        "max_drawdown": max_drawdown(df["p"]),
        "beta_to_market": float(df["p"].cov(df["m"]) / df["m"].var()),
        "corr_to_market": float(df["p"].corr(df["m"])),
    }
    for c in confs:
        q = df["p"].quantile(1 - c)
        out[f"hist_var_{int(c * 100)}"] = float(-q)
        out[f"hist_es_{int(c * 100)}"] = float(-df["p"][df["p"] <= q].mean())
    return pd.Series(out)
