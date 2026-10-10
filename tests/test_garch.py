import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from riskplatform.models.garch import (
    GJRParams, avg_variance_forecast, fit_gjr, garch_logit, next_variance, variance_path,
)
from riskplatform.validation.walkforward import make_folds, run_walk_forward
from riskplatform.features.compute import BASE_FEATURES
from test_walkforward import HOLDOUT, MODELING_START, _frame

TRUE = GJRParams(0.02, 0.03, 0.10, 0.90)


def simulate(p, n, seed):
    rng = np.random.default_rng(seed)
    r, h = np.empty(n), p.unconditional_variance
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = p.omega + (p.alpha + p.gamma * (r[t] < 0)) * r[t] ** 2 + p.beta * h
    return r


def test_fit_recovers_known_parameters():
    for seed in (1, 2):
        p = fit_gjr(simulate(TRUE, 8000, seed))
        assert abs(p.persistence - TRUE.persistence) < 0.02
        assert abs(p.unconditional_variance - 1.0) < 0.2
        assert p.gamma > p.alpha + 0.03                       # leverage effect is detected


def test_variance_is_causal():
    r = simulate(TRUE, 500, 3)
    h = variance_path(r, TRUE, 1.0)
    r2 = r.copy()
    r2[300:] *= 5
    h2 = variance_path(r2, TRUE, 1.0)
    assert np.allclose(h[:301], h2[:301])                      # h[t] uses returns through t-1 only
    assert not np.allclose(h[301:], h2[301:])


def test_multi_step_forecast_matches_manual_recursion():
    r = simulate(TRUE, 400, 4)
    h = variance_path(r, TRUE, 1.0)
    t = 250
    v = TRUE.omega + (TRUE.alpha + TRUE.gamma * (r[t] < 0)) * r[t] ** 2 + TRUE.beta * h[t]
    steps = [v]
    for _ in range(4):
        steps.append(TRUE.omega + TRUE.persistence * steps[-1])
    assert np.isclose(avg_variance_forecast(r, TRUE, 1.0, 5)[t], np.mean(steps))
    assert np.isclose(next_variance(r, TRUE, h)[t], steps[0])


def _returns():
    f = _frame()
    return f["ret_1d"][f.index < pd.Timestamp(HOLDOUT)]


def test_garch_logit_ranks_volatility_and_respects_time():
    rets = _returns()
    frame = _frame()
    fold = make_folds(2016, 2016)
    run = lambda r: run_walk_forward(frame, {"g": garch_logit(r)}, fold, list(BASE_FEATURES),
                                     MODELING_START, HOLDOUT).preds
    base = run(rets)
    assert base["prob"].between(0, 1).all() and roc_auc_score(base["y"], base["prob"]) > 0.65
    scrambled = rets.copy()
    scrambled.loc["2016-07-01":] = np.random.default_rng(0).normal(0, 0.05, (scrambled.index >= "2016-07-01").sum())
    again = run(scrambled)
    early = base["as_of_date"] < pd.Timestamp("2016-07-01")
    np.testing.assert_allclose(base.loc[early, "prob"].to_numpy(), again.loc[early, "prob"].to_numpy())


def test_matches_the_arch_library_when_installed():
    arch = pytest.importorskip("arch")
    r = simulate(TRUE, 4000, 5)
    res = arch.arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
    ours = fit_gjr(r)
    arch_rho = res.params["alpha[1]"] + res.params["gamma[1]"] / 2 + res.params["beta[1]"]
    assert abs(ours.persistence - arch_rho) < 0.02
    assert abs(ours.unconditional_variance - res.params["omega"] / (1 - arch_rho)) < 0.15
