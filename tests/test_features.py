from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from riskplatform.features.compute import (
    ALL_FEATURES, BASE_FEATURES, FEATURE_DEFS, OPEN_ENDED, SECTORS,
    FeatureInputs, compute_features, to_long, truncate_inputs,
)
from riskplatform.features.pit import pit_daily, pit_monthly
from riskplatform.ingest.fred import apply_rule_lag

N = 900
IDX = pd.bdate_range("2015-01-01", periods=N)


def _walk(rng, n, vol=0.01, start=100.0):
    return start * np.exp(np.cumsum(rng.normal(0, vol, n)))


def _ohlcv(rng, volume=True):
    c = _walk(rng, N)
    df = pd.DataFrame(
        {"open": c * (1 + rng.normal(0, 0.002, N)), "high": c * 1.01, "low": c * 0.99,
         "close": c, "adj_close": c}, index=IDX)
    df["volume"] = rng.integers(1_000_000, 5_000_000, N) if volume else 0
    return df


def _daily_macro(rng, level, vol):
    vals = level + np.cumsum(rng.normal(0, vol, N))
    rows = [{"observation_date": d.date(), "valid_from": d.date(), "valid_to": OPEN_ENDED, "value": float(v)}
            for d, v in zip(IDX, vals)]
    return pd.DataFrame(apply_rule_lag(rows))


def _monthly_macro(rng, level, first_lag=35, revise_lag=120):
    rows = []
    months = pd.date_range("2014-01-01", "2018-12-01", freq="MS")
    for i, m in enumerate(months):
        v = level + i * 0.05 + rng.normal(0, 0.05)
        first = m.date() + timedelta(days=first_lag)
        revised = m.date() + timedelta(days=revise_lag)
        rows.append({"observation_date": m.date(), "valid_from": first,
                     "valid_to": revised - timedelta(days=1), "value": v})
        rows.append({"observation_date": m.date(), "valid_from": revised,
                     "valid_to": OPEN_ENDED, "value": v + rng.normal(0, 0.03)})
    return pd.DataFrame(rows)


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    prices = {"^GSPC": _ohlcv(rng), "SPY": _ohlcv(rng), "HYG": _ohlcv(rng), "LQD": _ohlcv(rng)}
    prices["^VIX"] = _ohlcv(rng, volume=False)
    prices["^VIX"]["close"] = 15 + np.abs(np.cumsum(rng.normal(0, 0.3, N)))
    for s in SECTORS:
        prices[s] = _ohlcv(rng)
    macro = {"DFF": _daily_macro(rng, 1.0, 0.01), "DGS10": _daily_macro(rng, 3.0, 0.02),
             "DGS2": _daily_macro(rng, 2.0, 0.02), "T10Y2Y": _daily_macro(rng, 1.0, 0.01),
             "UNRATE": _monthly_macro(rng, 5.0), "CPIAUCSL": _monthly_macro(rng, 250.0)}
    return FeatureInputs(prices, macro)


def _assert_invariant(fn, inputs, T):
    full = fn(inputs)
    trunc = fn(truncate_inputs(inputs, IDX[T]))
    pd.testing.assert_frame_equal(full.iloc[: T + 1], trunc, check_freq=False, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------- the leakage tests
@pytest.mark.parametrize("T", [400, 700, 850])
def test_truncation_invariance_of_every_feature(T):
    """Features at T must be identical with or without any data after T."""
    _assert_invariant(compute_features, _inputs(), T)


def test_future_shocks_do_not_change_past_features():
    inputs = _inputs()
    T = 700
    base = compute_features(inputs).iloc[: T + 1]
    shocked = FeatureInputs(
        {k: v.copy() for k, v in inputs.prices.items()},
        {k: v.copy() for k, v in inputs.macro.items()},
    )
    for v in shocked.prices.values():
        v.iloc[T + 1:] = v.iloc[T + 1:] * 3
    cutoff = IDX[T].date()
    for v in shocked.macro.values():
        later = v["valid_from"].map(lambda x: x > cutoff)
        v.loc[later, "value"] = v.loc[later, "value"] * 10
    pd.testing.assert_frame_equal(base, compute_features(shocked).iloc[: T + 1],
                                  check_freq=False, rtol=1e-9, atol=1e-12)


def test_the_invariance_test_has_teeth():
    """A centered (look-ahead) window must FAIL the same check."""
    def leaky(inp):
        return pd.DataFrame({"x": inp.prices["^GSPC"]["close"].rolling(5, center=True).mean()})
    with pytest.raises(AssertionError):
        _assert_invariant(leaky, _inputs(), 700)


# ---------------------------------------------------------------- point-in-time lookups
def _bruteforce_latest(v, t, lag_months):
    snap = v[(v["valid_from"] <= t) & (v["valid_to"] >= t)]
    if snap.empty:
        return np.nan, np.nan
    latest = snap.loc[snap["observation_date"].idxmax()]
    target = latest["observation_date"]
    y, m = divmod(target.year * 12 + target.month - 1 - lag_months, 12)
    prior = snap[(snap["observation_date"].map(lambda d: (d.year, d.month)) == (y, m + 1))]
    return float(latest["value"]), (float(prior["value"].iloc[0]) if len(prior) else np.nan)


def test_pit_monthly_matches_bruteforce_including_revisions():
    v = _monthly_macro(np.random.default_rng(1), 5.0)
    out = pit_monthly(v, IDX, lag_months=1)
    for t in IDX[::37]:
        exp_latest, exp_lag = _bruteforce_latest(v, t.date(), 1)
        got = out.loc[t]
        assert (np.isnan(exp_latest) and np.isnan(got["latest"])) or np.isclose(got["latest"], exp_latest)
        assert (np.isnan(exp_lag) and np.isnan(got["lagged"])) or np.isclose(got["lagged"], exp_lag)


def test_pit_daily_matches_bruteforce_and_respects_release_lag():
    v = _daily_macro(np.random.default_rng(2), 1.0, 0.01)
    s = pit_daily(v, IDX)
    for t in IDX[::29]:
        known = v[v["valid_from"] <= t.date()]
        if known.empty:
            assert np.isnan(s.loc[t])
        else:
            expected = known.loc[known["observation_date"].idxmax(), "value"]
            assert np.isclose(s.loc[t], expected)
    # nothing is visible before the first release date
    assert s.loc[: pd.Timestamp(v["valid_from"].min()) - pd.Timedelta(days=1)].isna().all()
    # on every date, the visible value never comes from an observation dated after that day
    # (rule-lag: observation d is released after d)
    assert (v["valid_from"] > v["observation_date"]).all()


# ---------------------------------------------------------------- sanity
def test_feature_registry_and_sanity():
    feats = compute_features(_inputs())
    assert list(feats.columns) == ALL_FEATURES
    assert len(BASE_FEATURES) == len(ALL_FEATURES) - 1
    assert len({d.name for d in FEATURE_DEFS}) == len(FEATURE_DEFS)
    late = feats.iloc[400:]
    assert (late["drawdown_252d"] <= 1e-12).all()
    assert (late["rv_21d"] >= 0).all()
    assert late["sector_avgcorr_21d"].between(-1, 1).all()
    assert late[BASE_FEATURES].notna().all().all()           # base set fully populated after warm-up


def test_to_long_drops_missing_and_nonfinite_values():
    feats = compute_features(_inputs())
    long = to_long(feats)
    assert np.isfinite(long["value"].to_numpy(dtype=float)).all()
    assert len(long) == int(np.isfinite(feats.to_numpy(dtype=float)).sum())
    assert len(long) < feats.shape[0] * feats.shape[1]          # warm-up gaps are not stored
