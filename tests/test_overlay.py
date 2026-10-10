import numpy as np
import pandas as pd

from riskplatform.strategy.overlay import (
    BASELINES, build_weights, episode_table, ewma_vol, max_drawdown, performance,
    run_backtest, sharpe_tests,
)
from riskplatform.validation.stats import paired_sharpe_bootstrap, sharpe

IDX = pd.bdate_range("2020-01-01", periods=12)


def _flat_weights(model_col="model_x"):
    return pd.DataFrame({"buy_hold": 1.0, "ewma_vol_target": 1.0, "vix_target": 1.0, model_col: 1.0}, index=IDX)


def _bt(weights, spy_ret, cash=None, lag=2, cost=0.0):
    cash = cash if cash is not None else pd.Series(0.0, index=IDX)
    return run_backtest(weights, spy_ret, cash, lag, cost, IDX[0], IDX[-1], "model_x")


def test_exposure_decided_at_close_applies_two_days_later():
    spy = pd.Series(0.0, index=IDX)
    spy.iloc[6] = -0.10                                    # crash on day 6
    w = _flat_weights()
    w.loc[IDX[5], "model_x"] = 0.0                         # de-risk decided at the close of day 5
    bt = _bt(w, spy, lag=2)
    assert bt["held"].loc[IDX[6], "model_x"] == 1.0        # still fully exposed on the crash day
    assert bt["held"].loc[IDX[7], "model_x"] == 0.0
    assert np.isclose(bt["returns"].loc[IDX[6], "model_x"], -0.10)
    fast = _bt(w, spy, lag=1)                              # with a 1-day lag the same decision would dodge it
    assert np.isclose(fast["returns"].loc[IDX[6], "model_x"], 0.0)


def test_turnover_costs_and_cash():
    spy = pd.Series(0.0, index=IDX)
    w = _flat_weights()
    w.loc[IDX[5]:, "model_x"] = 0.0                        # de-risk at day 5 and stay out
    cash = pd.Series(0.001, index=IDX)
    bt = _bt(w, spy, cash, cost=10)
    first = bt["returns"].index[0]
    assert np.isclose(bt["turnover"]["buy_hold"].sum(), 1.0)           # entry only
    assert np.isclose(bt["turnover"]["model_x"].sum(), 2.0)            # entry + exit
    assert np.isclose(bt["returns"].loc[first, "buy_hold"], -0.001)    # 10 bps on entering
    assert np.isclose(bt["returns"].loc[IDX[8], "model_x"], 0.001)     # flat position earns cash
    assert np.isclose(bt["held"]["static_matched"].iloc[0], bt["held"]["model_x"].mean())


def test_all_strategies_share_one_window_and_baselines_come_first():
    bt = _bt(_flat_weights(), pd.Series(0.001, index=IDX))
    assert list(bt["returns"].columns[:4]) == BASELINES
    assert bt["returns"].notna().all().all()


def _market(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    ret = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    vix = pd.Series(15 + np.abs(np.cumsum(rng.normal(0, 0.3, n))), index=idx)
    p = pd.Series(rng.random(n), index=idx)
    return ret, vix, {"x": p}


def test_weights_use_only_information_through_day_t():
    ret, vix, probs = _market()
    T = 300
    full = build_weights(ret, vix, probs, 0.10)
    cut = build_weights(ret.iloc[: T + 1], vix.iloc[: T + 1], {"x": probs["x"].iloc[: T + 1]}, 0.10)
    pd.testing.assert_frame_equal(full.iloc[: T + 1], cut, check_freq=False)


def test_weight_rules():
    ret, vix, probs = _market()
    w = build_weights(ret, vix, probs, 0.10, w_max=1.0)
    assert w[["ewma_vol_target", "vix_target", "model_x"]].dropna().le(1.0).all().all()
    assert np.allclose(w["model_x"], 1 - probs["x"])
    wild = ret * 10                                         # huge volatility -> weight below the cap
    ww = build_weights(wild, vix, probs, 0.10)
    sigma = ewma_vol(wild)
    sel = ww["ewma_vol_target"].dropna()
    assert (sel < 1).all() and np.allclose(sel, (0.10 / sigma).loc[sel.index])


def test_performance_known_values():
    r = pd.Series([0.10, -0.20, 0.05] + [0.0] * 3, index=pd.bdate_range("2020-01-01", periods=6))
    assert np.isclose(max_drawdown(r), -0.20)
    flat = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=252))
    bt = {"returns": pd.DataFrame({"s": flat}), "held": pd.DataFrame({"s": 1.0}, index=flat.index),
          "turnover": pd.DataFrame({"s": 0.0}, index=flat.index), "cash": pd.Series(0.0, index=flat.index)}
    perf = performance(bt)
    assert np.isclose(perf.loc["s", "ann_return"], 1.01 ** 252 - 1)
    rng = np.random.default_rng(1)
    noisy = pd.Series(rng.normal(0.001, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    bt["returns"], bt["held"], bt["turnover"], bt["cash"] = (
        pd.DataFrame({"s": noisy}), pd.DataFrame({"s": 1.0}, index=noisy.index),
        pd.DataFrame({"s": 0.0}, index=noisy.index), pd.Series(0.0, index=noisy.index))
    expected = noisy.mean() / noisy.std(ddof=1) * np.sqrt(252)
    assert np.isclose(performance(bt).loc["s", "sharpe"], expected)


def test_sharpe_bootstrap_null_and_real_difference():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0004, 0.01, 1500)
    null = paired_sharpe_bootstrap(base, base, draws=200, mean_block=20, seed=1)
    assert null["obs_diff"] == 0 and null["ci_low"] <= 0 <= null["ci_high"]
    real = paired_sharpe_bootstrap(base + 0.002, base, draws=200, mean_block=20, seed=1)
    assert real["obs_diff"] > 1 and real["ci_low"] > 0 and real["p"] < 0.05
    assert np.isclose(sharpe(base), base.mean() / base.std(ddof=1) * np.sqrt(252))


def test_sharpe_tests_and_episode_table_run():
    ret, vix, probs = _market(700)
    idx = ret.index
    w = build_weights(ret, vix, probs, 0.10)
    bt = run_backtest(w, ret, pd.Series(0.0001, index=idx), 2, 5, idx[0], idx[-1], "model_x")
    cfg = {"bootstrap_draws": 100, "mean_block_length": 20, "seed": 1}
    t = sharpe_tests(bt, "model_x", ["buy_hold", "ewma_vol_target", "vix_target"], cfg)
    assert len(t) == 3 and (t["p_holm"] >= t["p_boot"]).all()
    ep = episode_table(bt, {"window": [str(idx[300].date()), str(idx[400].date())]})
    assert ("total_return", "window") in ep.columns and "model_x" in ep.index
