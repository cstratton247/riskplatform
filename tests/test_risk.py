import time

import numpy as np
import pandas as pd
from scipy import stats

from riskplatform.models.garch import GJRParams, variance_path
from riskplatform.risk import backtest as bt
from riskplatform.risk import var_es as ve
from riskplatform.risk.stress import (
    betas, equity_shock_loss, portfolio_metrics, portfolio_returns, replay, stressed_normal_var_es,
)
from test_garch import TRUE, simulate


def _series(arr, start="2010-01-04"):
    return pd.Series(arr, index=pd.bdate_range(start, periods=len(arr)))


# ------------------------------------------------------------------ backtest statistics
def test_kupiec_known_values():
    assert bt.kupiec_pof(10, 1000, 0.01)["lr"] < 1e-9 and bt.kupiec_pof(10, 1000, 0.01)["p"] > 0.99
    assert bt.kupiec_pof(25, 1000, 0.01)["p"] < 0.001
    zero = bt.kupiec_pof(0, 250, 0.01)                                   # too FEW exceptions is also rejected
    assert np.isclose(zero["lr"], -2 * 250 * np.log(0.99)) and np.isclose(zero["p"], 0.025, atol=0.001)


def test_christoffersen_flags_clustering_not_spread_out_exceptions():
    spread = np.zeros(1000, int)
    spread[::50] = 1
    clustered = np.zeros(1000, int)
    for s in range(0, 1000, 200):
        clustered[s:s + 4] = 1
    ok, bad = bt.christoffersen_independence(spread), bt.christoffersen_independence(clustered)
    assert ok["p"] > 0.2 and ok["n11"] == 0
    assert bad["p"] < 1e-6 and bad["n11"] > bad["n01"]


def test_basel_zones_and_shares():
    assert [bt.basel_zone(k) for k in (0, 4, 5, 9, 10, 15)] == ["green", "green", "yellow", "yellow", "red", "red"]
    i = np.zeros(600, int)
    i[300:310] = 1
    shares = bt.traffic_light_shares(i)
    assert np.isclose(sum(shares.values()), 1.0) and shares["red"] > 0 and shares["green"] > 0


def test_es_ratio():
    r = _series([-0.05, 0.01, -0.03, 0.0])
    es = _series([0.04, 0.04, 0.04, 0.04])
    exc = _series([1, 0, 1, 0])
    assert np.isclose(bt.es_ratio(r, es, exc), 1.0)                      # mean loss 0.04 / ES 0.04


# ------------------------------------------------------------------ VaR / ES methods
def test_historical_uses_only_prior_days():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0, 0.01, 800))
    base = ve.historical(r, 0.99, window=250)
    r2 = r.copy()
    r2.iloc[600] = -0.5                                                    # shock on day 600
    shocked = ve.historical(r2, 0.99, window=250)
    assert np.isclose(base["var"].iloc[600], shocked["var"].iloc[600])    # day-600 VaR cannot see r[600]
    assert shocked["var"].iloc[601] > base["var"].iloc[601] or shocked["es"].iloc[601] > base["es"].iloc[601]
    assert (base["es"].dropna() >= base["var"].dropna()).all()


def test_t_multipliers_match_simulation_and_normal_limit():
    nu = 5.0
    z = stats.t.rvs(nu, size=2_000_000, random_state=1) / np.sqrt(nu / (nu - 2))
    for conf in (0.95, 0.99):
        v, e = ve.t_multipliers(conf, nu)
        q = np.quantile(z, 1 - conf)
        assert abs(v - (-q)) / v < 0.02 and abs(e - (-z[z <= q].mean())) / e < 0.03
    v, e = ve.t_multipliers(0.99, 1e6)                                      # huge df -> normal
    assert abs(v - stats.norm.ppf(0.99)) < 0.01


def test_normal_method_is_calibrated_on_gaussian_returns():
    r = _series(np.random.default_rng(1).normal(0, 0.01, 6000))
    est = ve.normal(r, 0.99, lam=0.99)
    burn = 1000                                  # the EWMA starts from one squared return; let it forget
    exc = bt.exceptions(r.iloc[burn:], est["var"].iloc[burn:])
    assert 0.007 < exc.mean() < 0.015
    assert (est["es"].dropna() > est["var"].dropna()).all()


def test_filtered_historical_is_causal_and_roughly_calibrated():
    r = _series(simulate(TRUE, 2600, 7) / 100, start="2008-01-01")
    full = ve.filtered_historical(r, [0.99], window=250, min_fit_obs=300)[0.99]
    T = 2100
    cut = ve.filtered_historical(r.iloc[:T + 1], [0.99], window=250, min_fit_obs=300)[0.99]
    pd.testing.assert_frame_equal(full.iloc[: T + 1], cut, check_freq=False)       # no look-ahead
    ev = full.dropna()
    rate = bt.exceptions(r.loc[ev.index], ev["var"]).mean()
    assert 0.003 < rate < 0.025


def test_monte_carlo_matches_filtered_quantile_and_is_fast():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(5000)
    h1 = 1.0                                                                 # percent^2 -> 1% daily
    v1, e1 = ve.gjr_monte_carlo(TRUE, h1, z, 0.99, horizon=1, paths=200_000, seed=1)
    expected = -np.quantile(z, 0.01) * 0.01
    assert abs(v1 - expected) / expected < 0.03 and e1 > v1
    t0 = time.perf_counter()
    v10, _ = ve.gjr_monte_carlo(TRUE, h1, z, 0.99, horizon=10, paths=100_000, seed=1)
    assert time.perf_counter() - t0 < 2.0 and v10 > v1


# ------------------------------------------------------------------ stress and portfolio
def _assets(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({"SPY": mkt, "HI": 2 * mkt + rng.normal(0, 0.002, n),
                         "BOND": -0.2 * mkt + rng.normal(0, 0.004, n)}, index=idx)


def test_stressed_covariance_scales_and_blends():
    cov = pd.DataFrame(np.diag([0.0001, 0.0004]), index=["a", "b"], columns=["a", "b"])
    w = {"a": 0.5, "b": 0.5}
    base = stressed_normal_var_es(cov, w, 0.99)
    assert np.isclose(base["sigma"], np.sqrt(0.25 * 0.0001 + 0.25 * 0.0004))
    doubled = stressed_normal_var_es(cov, w, 0.99, vol_multiplier=2)
    assert np.isclose(doubled["var"], 2 * base["var"])
    perfect = stressed_normal_var_es(cov, w, 0.99, corr_blend=1.0)           # correlation -> 1
    assert np.isclose(perfect["sigma"], 0.5 * 0.01 + 0.5 * 0.02)
    assert base["es"] > base["var"] and perfect["var"] > base["var"]


def test_betas_replay_and_portfolio_metrics():
    a = _assets()
    b = betas(a, a["SPY"])
    assert abs(b["HI"] - 2) < 0.1 and abs(b["BOND"] + 0.2) < 0.1 and np.isclose(b["SPY"], 1)
    w = {"SPY": 0.6, "BOND": 0.4}
    assert np.isclose(equity_shock_loss(w, b, -0.2), (0.6 * 1 + 0.4 * b["BOND"]) * -0.2)
    port = portfolio_returns(a, w)
    assert np.allclose(port, 0.6 * a["SPY"] + 0.4 * a["BOND"])
    win = {"w": [str(a.index[100].date()), str(a.index[160].date())]}
    rp = replay(port, a["SPY"], win).loc["w"]
    manual = (1 + port.iloc[100:161]).prod() - 1
    assert np.isclose(rp["portfolio_return"], manual) and rp["days"] == 61
    cash = pd.Series(0.0001, index=a.index)
    m = portfolio_metrics(a, w, a["SPY"], cash)
    assert 0.3 < m["beta_to_market"] < 0.7 and m["hist_es_99"] >= m["hist_var_99"] > 0
