import numpy as np
import pandas as pd

from riskplatform.validation.report import fold_mean_table, pairwise_tests
from riskplatform.validation.stats import (
    dm_test, holm, long_run_variance, paired_fold_bootstrap, stationary_bootstrap_indices,
)


def test_stationary_bootstrap_shape_range_and_block_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(20000, 20, rng)
    assert len(idx) == 20000 and idx.min() >= 0 and idx.max() < 20000
    continues = ((idx[1:] - idx[:-1]) % 20000 == 1).mean()      # P(next step continues the block)
    assert abs(continues - (1 - 1 / 20)) < 0.01


def test_holm_known_example():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_hac_variance_exceeds_naive_for_autocorrelated_differences():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    d = np.zeros(3000)
    for i in range(1, 3000):
        d[i] = 0.7 * d[i - 1] + e[i]
    assert long_run_variance(d, 4) > 1.5 * d.var()


def test_dm_test_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.3, 1500)
    null = dm_test(base, base + rng.normal(0, 0.01, 1500), horizon=5)
    assert null["p"] > 0.05
    real = dm_test(base - 0.05, base, horizon=5)               # A has lower loss
    assert real["mean_diff"] < 0 and real["p"] < 0.001


def _folds(rng, informative_a=True, n_folds=6, n=250):
    out = []
    for _ in range(n_folds):
        latent = rng.normal(size=n)
        y = (latent + rng.normal(0, 1, n) > 0.8).astype(int)
        good = 1 / (1 + np.exp(-(latent * 1.5)))
        noise = rng.random(n)
        out.append((y, good if informative_a else noise, noise))
    return out


def test_paired_bootstrap_separates_good_from_random_and_is_null_for_identical():
    rng = np.random.default_rng(3)
    better = paired_fold_bootstrap(_folds(rng, True), draws=400, mean_block=10, seed=1)
    assert better["obs_diff"] > 0.1 and better["ci_low"] > 0 and better["p"] < 0.05
    same = [(y, b, b) for y, _, b in _folds(rng, True)]
    null = paired_fold_bootstrap(same, draws=200, mean_block=10, seed=1)
    assert null["obs_diff"] == 0 and null["ci_low"] <= 0 <= null["ci_high"]


def _preds(rng, n_per_fold=250, folds=("wf_2020", "wf_2021", "wf_2022")):
    rows = []
    for f_i, f in enumerate(folds):
        dates = pd.bdate_range(f"{2020 + f_i}-01-01", periods=n_per_fold)
        latent = rng.normal(size=n_per_fold)
        y = (latent + rng.normal(0, 1, n_per_fold) > 0.8).astype(int)
        for model, p in {
            "har_logit": 1 / (1 + np.exp(-latent)), "vix_logit": rng.random(n_per_fold),
            "hist_gb": 1 / (1 + np.exp(-2 * latent)),
        }.items():
            rows.append(pd.DataFrame({"model": model, "fold": f, "as_of_date": dates, "prob": p, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_pairwise_tests_and_fold_mean_table_run_end_to_end():
    from riskplatform.validation.walkforward import evaluate
    preds = _preds(np.random.default_rng(4))
    cfg = {"bootstrap_draws": 150, "mean_block_length": 10, "dm_horizon": 5, "seed": 1}
    tests = pairwise_tests(preds, cfg)
    assert set(tests["vs"]) == {"har_logit", "vix_logit"} and set(tests["model"]) == {"hist_gb"}
    row = tests[tests["vs"] == "vix_logit"].iloc[0]
    assert row["pr_auc_diff"] > 0 and (tests["p_boot_holm"] >= tests["p_boot"]).all()
    table = fold_mean_table(evaluate(preds))
    assert table.index[0] in {"hist_gb", "har_logit"} and "folds_beat_har" in table.columns
