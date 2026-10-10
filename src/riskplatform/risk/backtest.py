"""VaR / ES backtests: Kupiec, Christoffersen, Basel traffic light, ES ratio. Database-free."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlog1py, xlogy


def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 when the loss exceeded VaR (r < -VaR)."""
    return (r < -var).astype(int)


def kupiec_pof(x: int, n: int, p: float) -> dict:
    """Proportion-of-failures LR test of the exception rate against p."""
    phat = x / n
    ll_null = xlog1py(n - x, -p) + xlogy(x, p)
    ll_alt = xlog1py(n - x, -phat) + xlogy(x, phat)
    lr = float(max(-2 * (ll_null - ll_alt), 0.0))
    return {"lr": lr, "p": float(stats.chi2.sf(lr, 1))}


def christoffersen_independence(i: np.ndarray) -> dict:
    """LR test that exceptions do not cluster (first-order Markov independence)."""
    i = np.asarray(i, int)
    prev, cur = i[:-1], i[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    ll_null = xlog1py(n00 + n10, -pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlog1py(n00, -pi01) + xlogy(n01, pi01) + xlog1py(n10, -pi11) + xlogy(n11, pi11))
    lr = float(max(-2 * (ll_null - ll_alt), 0.0))
    return {"lr": lr, "p": float(stats.chi2.sf(lr, 1)), "n01": n01, "n11": n11}


def conditional_coverage(i: np.ndarray, p: float) -> dict:
    kup = kupiec_pof(int(np.sum(i)), len(i), p)
    ind = christoffersen_independence(i)
    lr = kup["lr"] + ind["lr"]
    return {"lr": lr, "p": float(stats.chi2.sf(lr, 2))}


def basel_zone(exceptions_in_250: int) -> str:
    """Basel traffic light for 99% VaR over 250 trading days."""
    if exceptions_in_250 <= 4:
        return "green"
    return "yellow" if exceptions_in_250 <= 9 else "red"


def traffic_light_shares(i: np.ndarray, window: int = 250) -> dict:
    i = np.asarray(i, int)
    if len(i) < window:
        return {"green": np.nan, "yellow": np.nan, "red": np.nan}
    counts = np.convolve(i, np.ones(window, int), mode="valid")
    zones = pd.Series([basel_zone(c) for c in counts]).value_counts(normalize=True)
    return {z: float(zones.get(z, 0.0)) for z in ("green", "yellow", "red")}


def es_ratio(r: pd.Series, es: pd.Series, exc: pd.Series) -> float:
    """Mean realized loss on exception days over mean predicted ES on those days (~1 if ES is right)."""
    mask = exc.astype(bool)
    if mask.sum() == 0:
        return np.nan
    return float((-r[mask]).mean() / es[mask].mean())


def backtest_table(r: pd.Series, estimates: dict, start, end) -> pd.DataFrame:
    """`estimates` maps (method, confidence) -> DataFrame with 'var' and 'es' columns."""
    rows = []
    idx = r.loc[start:end].index
    for (method, conf), est in estimates.items():
        e = est.reindex(idx)
        ok = e["var"].notna() & r.reindex(idx).notna()
        rr, vv, ee = r.reindex(idx)[ok], e["var"][ok], e["es"][ok]
        exc = exceptions(rr, vv)
        n, x, p = len(rr), int(exc.sum()), 1 - conf
        kup = kupiec_pof(x, n, p)
        ind = christoffersen_independence(exc.to_numpy())
        cc = conditional_coverage(exc.to_numpy(), p)
        shares = traffic_light_shares(exc.to_numpy()) if abs(conf - 0.99) < 1e-9 else {}
        rows.append({
            "method": method, "conf": conf, "n": n, "expected": round(n * p, 1), "exceptions": x,
            "rate": x / n, "kupiec_p": kup["p"], "indep_p": ind["p"], "cc_p": cc["p"],
            "es_ratio": es_ratio(rr, ee, exc), "avg_var": float(vv.mean()),
            "basel_green": shares.get("green", np.nan), "basel_yellow": shares.get("yellow", np.nan),
            "basel_red": shares.get("red", np.nan),
        })
    return pd.DataFrame(rows)
