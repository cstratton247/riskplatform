"""Feature computation (spec section 6).

Every feature at day t uses information available at the close of t only: backward-looking
windows, and macro values read through the point-in-time functions. The truncation-invariance
test (tests/test_features.py) enforces this for the whole set.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .pit import pit_daily, pit_monthly

TD = 252
SECTORS = ["XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
PRICE_TICKERS = ["^GSPC", "^VIX", "SPY", "HYG", "LQD"] + SECTORS
DAILY_MACRO = ["DFF", "DGS10", "DGS2", "T10Y2Y"]
MONTHLY_MACRO = ["UNRATE", "CPIAUCSL"]
MACRO_SERIES = DAILY_MACRO + MONTHLY_MACRO
OPEN_ENDED = date(9999, 12, 31)


@dataclass(frozen=True)
class FeatureDef:
    name: str
    group: str
    description: str
    lookback_days: int
    formula: str
    extended: bool = False


FEATURE_DEFS: list[FeatureDef] = [
    FeatureDef("ret_1d", "returns", "S&P 500 log return, 1 day", 1, "ln(C_t / C_{t-1})"),
    FeatureDef("ret_5d", "returns", "S&P 500 log return, 5 days", 5, "ln(C_t / C_{t-5})"),
    FeatureDef("ret_21d", "returns", "S&P 500 log return, 21 days", 21, "ln(C_t / C_{t-21})"),
    FeatureDef("ret_63d", "returns", "S&P 500 log return, 63 days", 63, "ln(C_t / C_{t-63})"),
    FeatureDef("rv_5d", "backward_vol", "Realized vol, 5d (HAR weekly-ish)", 5, "sqrt(252 * mean(r^2, 5))"),
    FeatureDef("rv_21d", "backward_vol", "Realized vol, 21d (HAR monthly)", 21, "sqrt(252 * mean(r^2, 21))"),
    FeatureDef("rv_63d", "backward_vol", "Realized vol, 63d", 63, "sqrt(252 * mean(r^2, 63))"),
    FeatureDef("semidev_21d", "downside", "Downside semi-deviation, 21d", 21, "sqrt(252 * mean(min(r,0)^2, 21))"),
    FeatureDef("drawdown_252d", "downside", "Drawdown from 252d high", 252, "C_t / max(C, 252) - 1"),
    FeatureDef("parkinson_21d", "range_vol", "Parkinson range volatility, 21d", 21, "sqrt(252/(4 ln2) * mean(ln(H/L)^2, 21))"),
    FeatureDef("garman_klass_21d", "range_vol", "Garman-Klass volatility, 21d", 21, "sqrt(252 * mean(0.5 ln(H/L)^2 - (2 ln2 - 1) ln(C/O)^2, 21))"),
    FeatureDef("vix_level", "implied_vol", "VIX close", 0, "VIX_t"),
    FeatureDef("vix_chg_1d", "implied_vol", "VIX change, 1 day", 1, "VIX_t - VIX_{t-1}"),
    FeatureDef("vix_chg_5d", "implied_vol", "VIX change, 5 days", 5, "VIX_t - VIX_{t-5}"),
    FeatureDef("vrp_proxy", "implied_vol", "VIX minus trailing 21d realized vol (decimal units)", 21, "VIX_t/100 - rv_21d"),
    FeatureDef("spy_volume_ratio_21d", "volume", "Log of SPY volume vs its 21d mean", 21, "ln(V_t / mean(V, 21))"),
    FeatureDef("dff_chg_21d", "rates", "Fed funds rate change, 21 trading days", 21, "DFF_known_t - DFF_known_{t-21}"),
    FeatureDef("dgs10_chg_5d", "rates", "10y yield change, 5 days", 5, "DGS10_known_t - DGS10_known_{t-5}"),
    FeatureDef("dgs10_chg_21d", "rates", "10y yield change, 21 days", 21, "DGS10_known_t - DGS10_known_{t-21}"),
    FeatureDef("dgs2_chg_5d", "rates", "2y yield change, 5 days", 5, "DGS2_known_t - DGS2_known_{t-5}"),
    FeatureDef("t10y2y_level", "rates", "10y minus 2y spread", 0, "T10Y2Y_known_t"),
    FeatureDef("unrate_chg_1m", "macro", "Unemployment change vs prior month, same vintage", 0, "UNRATE_latest - UNRATE_prev (same snapshot)"),
    FeatureDef("cpi_yoy", "macro", "CPI 12-month change, same vintage", 0, "CPI_latest / CPI_12m_earlier - 1 (same snapshot)"),
    FeatureDef("sector_dispersion_1d", "cross_section", "Cross-sectional std of 9 sector ETF returns", 1, "std_i(r_i,t)"),
    FeatureDef("sector_avgcorr_21d", "cross_section", "Average pairwise correlation of sector ETFs, 21d", 21, "mean_{i<j} corr(r_i, r_j; 21)"),
    FeatureDef("hyg_lqd_chg_5d", "credit", "Log change in HYG/LQD ratio, 5d (HYG starts 2007)", 5, "ln(HYG/LQD)_t - ln(HYG/LQD)_{t-5}", extended=True),
]
BASE_FEATURES = [d.name for d in FEATURE_DEFS if not d.extended]
ALL_FEATURES = [d.name for d in FEATURE_DEFS]


@dataclass
class FeatureInputs:
    prices: dict[str, pd.DataFrame]   # ticker -> frame indexed by date: open high low close adj_close volume
    macro: dict[str, pd.DataFrame]    # series_id -> vintages: observation_date valid_from valid_to value


def truncate_inputs(inputs: FeatureInputs, as_of: pd.Timestamp) -> FeatureInputs:
    """The dataset as it stood at the close of `as_of`: no later prices, no later vintages."""
    as_of = pd.Timestamp(as_of)
    d = as_of.date()
    prices = {k: v.loc[:as_of] for k, v in inputs.prices.items()}
    macro = {}
    for k, v in inputs.macro.items():
        known = v[v["valid_from"].map(lambda x: x <= d)].copy()
        known["valid_to"] = known["valid_to"].map(lambda x: OPEN_ENDED if x > d else x)
        macro[k] = known.reset_index(drop=True)
    return FeatureInputs(prices, macro)


def _avg_pairwise_corr(rets: pd.DataFrame, window: int) -> pd.Series:
    x = rets.to_numpy(dtype=float)
    n_obs, k = x.shape
    out = np.full(n_obs, np.nan)
    if n_obs >= window:
        win = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)   # (n, k, w)
        ok = ~np.isnan(win).any(axis=(1, 2))
        cen = win - win.mean(axis=2, keepdims=True)
        std = np.sqrt((cen ** 2).sum(axis=2))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.einsum("nkw,njw->nkj", cen, cen) / (std[:, :, None] * std[:, None, :])
            avg = (corr.sum(axis=(1, 2)) - k) / (k * (k - 1))
        avg[~ok] = np.nan
        out[window - 1:] = avg
    return pd.Series(out, index=rets.index)


def compute_features(inputs: FeatureInputs) -> pd.DataFrame:
    for t in PRICE_TICKERS:
        if t not in inputs.prices:
            raise KeyError(f"missing price series {t}")
    for s in MACRO_SERIES:
        if s not in inputs.macro:
            raise KeyError(f"missing macro series {s}")

    spx = inputs.prices["^GSPC"]
    idx = spx.index
    c, o, hi, lo = spx["close"], spx["open"], spx["high"], spx["low"]
    lc = np.log(c)
    r = lc.diff()
    f: dict[str, pd.Series] = {}

    f["ret_1d"] = r
    for w in (5, 21, 63):
        f[f"ret_{w}d"] = lc.diff(w)
    for w in (5, 21, 63):
        f[f"rv_{w}d"] = np.sqrt(TD * r.pow(2).rolling(w).mean())
    f["semidev_21d"] = np.sqrt(TD * r.clip(upper=0).pow(2).rolling(21).mean())
    f["drawdown_252d"] = c / c.rolling(252).max() - 1

    u = np.log(hi / lo)
    cc = np.log(c / o)
    f["parkinson_21d"] = np.sqrt(TD / (4 * np.log(2)) * u.pow(2).rolling(21).mean())
    gk = (0.5 * u.pow(2) - (2 * np.log(2) - 1) * cc.pow(2)).rolling(21).mean()
    f["garman_klass_21d"] = np.sqrt(TD * gk.clip(lower=0))

    vix = inputs.prices["^VIX"]["close"].reindex(idx)
    f["vix_level"] = vix
    f["vix_chg_1d"] = vix.diff()
    f["vix_chg_5d"] = vix.diff(5)
    f["vrp_proxy"] = vix / 100 - f["rv_21d"]

    vol = inputs.prices["SPY"]["volume"].reindex(idx)
    ratio = vol / vol.rolling(21).mean()
    f["spy_volume_ratio_21d"] = np.log(ratio.where(ratio > 0))

    dff = pit_daily(inputs.macro["DFF"], idx)
    d10 = pit_daily(inputs.macro["DGS10"], idx)
    d2 = pit_daily(inputs.macro["DGS2"], idx)
    f["dff_chg_21d"] = dff - dff.shift(21)
    f["dgs10_chg_5d"] = d10 - d10.shift(5)
    f["dgs10_chg_21d"] = d10 - d10.shift(21)
    f["dgs2_chg_5d"] = d2 - d2.shift(5)
    f["t10y2y_level"] = pit_daily(inputs.macro["T10Y2Y"], idx)

    ur = pit_monthly(inputs.macro["UNRATE"], idx, lag_months=1)
    f["unrate_chg_1m"] = ur["latest"] - ur["lagged"]
    cpi = pit_monthly(inputs.macro["CPIAUCSL"], idx, lag_months=12)
    f["cpi_yoy"] = cpi["latest"] / cpi["lagged"] - 1

    sec = pd.DataFrame({t: inputs.prices[t]["adj_close"].reindex(idx) for t in SECTORS})
    sec_r = np.log(sec).diff()
    f["sector_dispersion_1d"] = sec_r.std(axis=1).where(sec_r.notna().all(axis=1))
    f["sector_avgcorr_21d"] = _avg_pairwise_corr(sec_r, 21)

    hyg = inputs.prices["HYG"]["adj_close"].reindex(idx)
    lqd = inputs.prices["LQD"]["adj_close"].reindex(idx)
    f["hyg_lqd_chg_5d"] = np.log(hyg / lqd).diff(5)

    out = pd.DataFrame(f, index=idx)[ALL_FEATURES]
    return out.replace([np.inf, -np.inf], np.nan)


def to_long(features: pd.DataFrame) -> pd.DataFrame:
    """Wide feature frame -> long (as_of_date, name, value), keeping only finite values.

    Missing values (warm-up, pre-listing history) are simply absent from the long form;
    they are never stored as NaN. (pandas >= 3 `stack()` keeps NaN, so filter explicitly.)
    """
    long = features.stack(future_stack=True).reset_index()
    long.columns = ["as_of_date", "name", "value"]
    return long[np.isfinite(long["value"].astype(float))].reset_index(drop=True)
