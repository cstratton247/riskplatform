"""Decision-level evaluation: does the forecast improve a portfolio decision? (spec section 9)

Timeline for every strategy: the exposure decision is made with information through the CLOSE of
day t and is applied to the return of day t + lag (base case lag = 2, one day of execution delay).
Uninvested capital earns the fed funds rate known BEFORE the day (as of the previous day).
Costs are charged on turnover in the weight actually held. No leverage. Database-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform.validation.stats import paired_sharpe_bootstrap, sharpe
from riskplatform.validation.stats import holm

TD = 252
BASELINES = ["buy_hold", "static_matched", "ewma_vol_target", "vix_target"]


def ewma_vol(ret: pd.Series, lam: float = 0.94, td: int = TD) -> pd.Series:
    """RiskMetrics volatility, known at the close of t (includes r_t)."""
    return np.sqrt(td * ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def build_weights(
    spy_ret: pd.Series, vix: pd.Series, probs: dict[str, pd.Series],
    target_vol: float, w_max: float = 1.0, lam: float = 0.94,
) -> pd.DataFrame:
    """Exposure decided at the close of each day t (not yet lagged)."""
    idx = spy_ret.index
    w = {
        "buy_hold": pd.Series(1.0, index=idx),
        "ewma_vol_target": (target_vol / ewma_vol(spy_ret, lam)).clip(lower=0, upper=w_max),
        "vix_target": (target_vol / (vix.reindex(idx) / 100)).clip(lower=0, upper=w_max),
    }
    for name, p in probs.items():
        w[f"model_{name}"] = (1 - p.reindex(idx)).clip(lower=0, upper=w_max)
    return pd.DataFrame(w)


def run_backtest(
    weights: pd.DataFrame, spy_ret: pd.Series, cash: pd.Series, lag: int, cost_bps: float,
    start, end, static_from: str,
) -> dict:
    held = weights.shift(lag)                              # weight in force on day s was decided at s - lag
    idx = weights.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    mask &= held.notna().all(axis=1).to_numpy() & spy_ret.notna().to_numpy() & cash.notna().to_numpy()
    held = held[mask].copy()
    held["static_matched"] = held[static_from].mean()      # constant exposure with the same average
    held = held[BASELINES + [c for c in held.columns if c not in BASELINES]]
    r, c = spy_ret[mask], cash[mask]
    turnover = held.diff().abs()
    turnover.iloc[0] = held.iloc[0].abs()                  # entering from cash is charged too
    net = held.mul(r, axis=0) + (1 - held).mul(c, axis=0) - turnover * cost_bps / 10_000
    return {"returns": net, "held": held, "turnover": turnover, "cash": c}


def max_drawdown(r: pd.Series) -> float:
    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def performance(bt: dict) -> pd.DataFrame:
    rows = {}
    years = len(bt["returns"]) / TD
    for col in bt["returns"].columns:
        r = bt["returns"][col]
        excess = (r - bt["cash"]).to_numpy()
        ann_ret = float((1 + r).prod() ** (1 / years) - 1)
        mdd = max_drawdown(r)
        cutoff = r.quantile(0.05)
        rows[col] = {
            "ann_return": ann_ret, "ann_vol": float(r.std() * np.sqrt(TD)), "sharpe": sharpe(excess),
            "max_drawdown": mdd, "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
            "cvar95_daily": float(r[r <= cutoff].mean()),
            "avg_exposure": float(bt["held"][col].mean()),
            "turnover_per_year": float(bt["turnover"][col].sum() / years),
        }
    return pd.DataFrame(rows).T


def episode_table(bt: dict, episodes: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in episodes.items():
        sl = bt["returns"].loc[pd.Timestamp(a):pd.Timestamp(b)]
        for col in sl.columns:
            rows.append({"episode": name, "strategy": col,
                         "total_return": float((1 + sl[col]).prod() - 1),
                         "max_drawdown": max_drawdown(sl[col])})
    return pd.DataFrame(rows).pivot(index="strategy", columns="episode", values=["total_return", "max_drawdown"])


def sharpe_tests(bt: dict, model_col: str, vs: list[str], stats_cfg: dict) -> pd.DataFrame:
    rows = []
    cash = bt["cash"].to_numpy()
    a = bt["returns"][model_col].to_numpy() - cash
    for other in vs:
        b = bt["returns"][other].to_numpy() - cash
        res = paired_sharpe_bootstrap(a, b, stats_cfg["bootstrap_draws"],
                                      stats_cfg["mean_block_length"], stats_cfg["seed"])
        rows.append({"model_rule": model_col, "vs": other, "sharpe_diff": res["obs_diff"],
                     "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_boot": res["p"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_boot"])
    return out


def sensitivity(
    spy_ret, vix, probs, cash, ov: dict, window: tuple, primary: str,
) -> pd.DataFrame:
    """Net Sharpe for every strategy across lag, cost and target-volatility settings."""
    rows = []
    for lag in ov["lag_sensitivity"]:
        for cost in ov["cost_bps_sensitivity"]:
            for tv in ov["target_vol_sensitivity"]:
                w = build_weights(spy_ret, vix, probs, tv, ov["max_weight"], ov["ewma_lambda"])
                perf = performance(run_backtest(w, spy_ret, cash, lag, cost, *window, f"model_{primary}"))
                rows.append({"lag": lag, "cost_bps": cost, "target_vol": tv, **perf["sharpe"].to_dict()})
    return pd.DataFrame(rows)
