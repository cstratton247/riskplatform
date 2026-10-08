import numpy as np
import pandas as pd

from riskplatform.features.labels import forward_realized_vol, make_labels
from riskplatform.validation.purging import train_mask

H, Q, LOOKBACK = 5, 0.80, 252


def _returns(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))   # slowly varying volatility
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(0, vol), index=idx)


def test_forward_rv_known_value():
    r = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=20))
    rv = forward_realized_vol(r, horizon=H)
    expected = np.sqrt(252 * 0.01**2)
    assert np.isclose(rv.iloc[0], expected)
    assert rv.iloc[-H:].isna().all()          # last h days have no complete forward window
    assert rv.iloc[:-H].notna().all()


def test_forward_rv_uses_only_future_returns():
    r = _returns(100)
    rv_a = forward_realized_vol(r, H)
    r2 = r.copy()
    r2.iloc[50] *= 100                        # shock on day 50
    rv_b = forward_realized_vol(r2, H)
    changed = np.flatnonzero(~np.isclose(rv_a.fillna(0), rv_b.fillna(0)))
    assert changed.min() == 45 and changed.max() == 49   # only days 45..49 look forward to day 50


def test_label_known_date_and_tail():
    lab = make_labels(_returns(), H, Q, LOOKBACK)
    ok = lab["y"].notna()
    assert (lab.loc[ok, "label_known_date"] > lab.index[ok]).all()
    assert lab["y"].iloc[-H:].isna().all()
    assert lab["y"].iloc[: LOOKBACK + H - 1].isna().all()   # warm-up produces no labels


def test_prevalence_is_close_to_one_minus_quantile():
    lab = make_labels(_returns(4000), H, Q, LOOKBACK)
    assert abs(lab["y"].dropna().astype(float).mean() - (1 - Q)) < 0.05


def test_threshold_truncation_invariance():
    """Threshold at T must be identical whether or not data after T exists (leakage test)."""
    r = _returns()
    T = 1000
    full = make_labels(r, H, Q, LOOKBACK)
    trunc = make_labels(r.iloc[: T + 1], H, Q, LOOKBACK)
    pd.testing.assert_series_equal(full["threshold"].iloc[: T + 1], trunc["threshold"])


def test_threshold_ignores_future_shock():
    r = _returns()
    T = 1000
    r_shocked = r.copy()
    r_shocked.iloc[T + 1 :] *= 10
    a = make_labels(r, H, Q, LOOKBACK)["threshold"].iloc[: T + 1]
    b = make_labels(r_shocked, H, Q, LOOKBACK)["threshold"].iloc[: T + 1]
    pd.testing.assert_series_equal(a, b)


def test_naive_threshold_would_fail_the_leakage_test():
    """Demonstrates the test has teeth: a threshold built from RV_fwd(s<=t) leaks the future."""
    r = _returns()
    T = 1000
    naive_full = forward_realized_vol(r, H).rolling(LOOKBACK).quantile(Q)
    naive_trunc = forward_realized_vol(r.iloc[: T + 1], H).rolling(LOOKBACK).quantile(Q)
    assert not np.isnan(naive_full.iloc[T])
    assert np.isnan(naive_trunc.iloc[T])      # depends on returns after T, so truncation changes it


def test_purging_excludes_overlapping_labels():
    lab = make_labels(_returns(), H, Q, LOOKBACK)
    test_start = lab.index[1100]
    mask = train_mask(lab["label_known_date"], test_start)
    assert mask.any()
    assert (lab.loc[mask, "label_known_date"] < test_start).all()
    # the last H samples before the test start must be purged
    assert not mask.iloc[1100 - H + 1 : 1100].any()
