"""Target construction (spec section 2).

Leakage rule: RV_fwd(s) is only fully known once day s+h has closed, so the
label threshold at day t may use RV_fwd(s) only for s <= t-h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def forward_realized_vol(
    returns: pd.Series, horizon: int = 5, trading_days: int = 252
) -> pd.Series:
    """RV_fwd(t) = sqrt(trading_days/h * sum_{i=1..h} r_{t+i}^2); NaN for the last h days."""
    window_sum = returns.pow(2).rolling(horizon).sum()   # at k: sum of r_{k-h+1..k}
    return np.sqrt(trading_days / horizon * window_sum.shift(-horizon))


def make_labels(
    returns: pd.Series,
    horizon: int = 5,
    quantile: float = 0.80,
    lookback: int = 252,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Return rv_fwd, threshold, y and label_known_date indexed like `returns`.

    threshold(t) = quantile of the `lookback` most recent RV_fwd values that are
    fully observed at t, i.e. RV_fwd(t-h) and earlier.
    """
    rv = forward_realized_vol(returns, horizon, trading_days)
    observed_at_t = rv.shift(horizon)                      # value at t is RV_fwd(t-h)
    threshold = observed_at_t.rolling(lookback).quantile(quantile)

    valid = rv.notna() & threshold.notna()
    y = (rv > threshold).astype("float").where(valid).astype("Int8")

    dates = pd.Series(returns.index, index=returns.index)
    label_known_date = dates.shift(-horizon).where(valid)  # trading day on which y is determined

    return pd.DataFrame(
        {"rv_fwd": rv, "threshold": threshold, "y": y, "label_known_date": label_known_date}
    )
