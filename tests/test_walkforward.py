import numpy as np
import pandas as pd

from riskplatform.features.compute import BASE_FEATURES
from riskplatform.features.labels import make_labels
from riskplatform.models.benchmarks import default_ladder
from riskplatform.validation.metrics import classification_metrics
from riskplatform.validation.walkforward import (
    eligible_rows, evaluate, make_folds, run_walk_forward, split_fold,
)

MODELING_START = "2006-01-01"
HOLDOUT = "2024-01-01"


def _frame(n=5300, seed=0):
    """Synthetic daily data with persistent (clustered) volatility, shaped like the real frame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.97 * lv[i - 1] + rng.normal(0, 0.12)
    sigma = 0.008 * np.exp(lv)
    r = pd.Series(sigma * rng.normal(size=n), index=idx)
    f = pd.DataFrame(rng.normal(size=(n, len(BASE_FEATURES))), index=idx, columns=BASE_FEATURES)
    for w in (5, 21, 63):
        f[f"rv_{w}d"] = np.sqrt(252 * r.pow(2).rolling(w).mean())
    f["vix_level"] = 100 * sigma * np.sqrt(252) * np.exp(rng.normal(0, 0.1, n))
    f["parkinson_21d"] = f["rv_21d"] * np.exp(rng.normal(0, 0.1, n))
    f["garman_klass_21d"] = f["rv_21d"] * np.exp(rng.normal(0, 0.1, n))
    labels = make_labels(r, 5, 0.80, 252)
    return f.join(labels[["y", "label_known_date", "threshold", "rv_fwd"]])


def _run(frame, folds=None):
    folds = folds or make_folds(2013, 2023)
    return run_walk_forward(frame, default_ladder(), folds, list(BASE_FEATURES), MODELING_START, HOLDOUT)


def test_folds_cover_the_declared_years():
    folds = make_folds(2013, 2023)
    assert len(folds) == 11 and folds[0].test_start == pd.Timestamp("2013-01-01")
    assert folds[-1].test_end == pd.Timestamp("2023-12-31") and folds[0].label == "wf_2013"


def test_training_labels_are_known_before_the_test_block_and_overlap_is_purged():
    rows = eligible_rows(_frame(), list(BASE_FEATURES), MODELING_START, HOLDOUT)
    fold = make_folds(2018, 2018)[0]
    train, test = split_fold(rows, fold)
    assert (train["label_known_date"] < fold.test_start).all()
    assert train.index.max() < fold.test_start
    last_five_before_test = rows[rows.index < fold.test_start].tail(5)
    assert not last_five_before_test.index.isin(train.index).any()   # their 5-day labels reach into 2018
    assert test.index.min() >= fold.test_start and test.index.max() <= fold.test_end


def test_final_holdout_cannot_influence_anything():
    frame = _frame()
    base, _ = _run(frame)
    assert base["as_of_date"].max() < pd.Timestamp(HOLDOUT)
    mutated = frame.copy()
    held = mutated.index >= pd.Timestamp(HOLDOUT)
    rng = np.random.default_rng(99)
    mutated.loc[held, BASE_FEATURES] = rng.normal(size=(held.sum(), len(BASE_FEATURES)))
    mutated.loc[held, "y"] = 1 - mutated.loc[held, "y"].astype("float").fillna(0).astype(int)
    again, _ = _run(mutated)
    pd.testing.assert_frame_equal(base, again)


def test_test_period_labels_never_leak_into_predictions():
    frame = _frame()
    fold = make_folds(2015, 2015)
    base, _ = _run(frame, fold)
    mutated = frame.copy()
    in_test = (mutated.index >= pd.Timestamp("2015-01-01")) & (mutated.index <= pd.Timestamp("2015-12-31"))
    mutated.loc[in_test, "y"] = (np.random.default_rng(5).random(in_test.sum()) < 0.5).astype(int)
    again, _ = _run(mutated, fold)
    pd.testing.assert_series_equal(base["prob"], again["prob"])


def test_every_model_is_scored_on_identical_rows():
    preds, _ = _run(_frame())
    keys = {m: set(zip(g["fold"], g["as_of_date"])) for m, g in preds.groupby("model")}
    first = next(iter(keys.values()))
    assert all(k == first for k in keys.values())
    assert preds["prob"].between(0, 1).all()


def test_ladder_behaves_sensibly_on_persistent_volatility():
    preds, _ = _run(_frame())
    metrics = evaluate(preds)
    # a constant-score model has no ranking skill inside any single fold. (Pooled across folds
    # it can drift off 0.5, because its constant differs from fold to fold.)
    per_fold = metrics[(metrics["model"] == "prevalence") & (metrics["scope"] != "pooled")]
    assert np.allclose(per_fold["roc_auc"], 0.5)
    pooled = metrics.query("scope == 'pooled'").set_index("model")
    assert pooled.loc["persistence_logit", "roc_auc"] > 0.65
    assert pooled.loc["har_logit", "pr_auc"] > pooled.loc["prevalence", "pr_auc"]


def test_metrics_known_values():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert np.isclose(m["brier"], 0.025)
    expected_ll = -np.mean(np.log([0.9, 0.8, 0.8, 0.9]))
    assert np.isclose(m["log_loss"], expected_ll)
    const = classification_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])       # constant scores must not crash
    assert np.isclose(const["roc_auc"], 0.5) and np.isclose(const["ece"], 0.0)
