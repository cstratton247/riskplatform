"""Point-in-time lookups of macro vintages (vectorized mirror of macro.latest_known_as_of).

A vintage row says: observation_date's value was `value`, publicly known from
valid_from through valid_to inclusive. At as-of date t, the "latest known" value is the
one with the greatest observation_date among rows with valid_from <= t <= valid_to.

All arithmetic uses integer day/month numbers so the open-ended 9999-12-31 sentinel
never has to become a pandas Timestamp (which cannot represent it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ordinals(dates) -> np.ndarray:
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


def _index_ordinals(idx: pd.DatetimeIndex) -> np.ndarray:
    return _ordinals(idx.date)


def pit_daily(vintages: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Daily, unrevised series (one row per observation): latest value known on each date."""
    vf = _ordinals(vintages["valid_from"])
    obs = _ordinals(vintages["observation_date"])
    val = vintages["value"].to_numpy(dtype=float)
    order = np.lexsort((obs, vf))                       # by valid_from, then observation_date
    vf, obs, val = vf[order], obs[order], val[order]
    last_of_each = np.r_[vf[1:] != vf[:-1], True]       # latest observation per release date
    vf, val = vf[last_of_each], val[last_of_each]
    t = _index_ordinals(idx)
    pos = np.searchsorted(vf, t, side="right") - 1
    out = np.where(pos >= 0, val[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=idx)


def pit_monthly(vintages: pd.DataFrame, idx: pd.DatetimeIndex, lag_months: int) -> pd.DataFrame:
    """Monthly revised series. Returns columns `latest` and `lagged`.

    `lagged` is the value of the observation `lag_months` before the latest one, read from
    the SAME snapshot (same as-of date), so a revision never leaks into a difference or ratio.
    """
    vf = _ordinals(vintages["valid_from"])
    vt = _ordinals(vintages["valid_to"])
    val = vintages["value"].to_numpy(dtype=float)
    month = np.array(
        [d.year * 12 + d.month for d in vintages["observation_date"]], dtype=np.int64
    )
    t_all = _index_ordinals(idx)
    latest = np.full(len(t_all), np.nan)
    lagged = np.full(len(t_all), np.nan)

    for start in range(0, len(t_all), 1000):                        # chunked to bound memory
        t = t_all[start:start + 1000]
        valid = (vf[None, :] <= t[:, None]) & (vt[None, :] >= t[:, None])
        m = np.where(valid, month[None, :], -1)
        j = m.argmax(axis=1)
        rows = np.arange(len(t))
        best = m[rows, j]
        has = best >= 0
        latest[start:start + len(t)][has] = val[j[has]]

        match = valid & (month[None, :] == (best - lag_months)[:, None])
        k = match.argmax(axis=1)
        ok = has & match[rows, k]
        lagged[start:start + len(t)][ok] = val[k[ok]]

    return pd.DataFrame({"latest": latest, "lagged": lagged}, index=idx)
