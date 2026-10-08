import numpy as np
import pandas as pd

from riskplatform.features.compute import BASE_FEATURES
from riskplatform.models.learners import hist_gb, inner_splits, logit_tuned, random_forest
from riskplatform.validation.walkforward import eligible_rows, make_folds, run_walk_forward
from test_walkforward import HOLDOUT, MODELING_START, _frame


def _train_rows(year=2016):
    rows = eligible_rows(_frame(), list(BASE_FEATURES), MODELING_START, HOLDOUT)
    fold = make_folds(year, year)[0]
    return rows[rows["label_known_date"] < fold.test_start], rows, fold


def test_inner_splits_are_purged_and_chronological():
    train, _, _ = _train_rows()
    splits = list(inner_splits(train, 3))
    assert len(splits) == 3
    for tr, va in splits:
        val_start = pd.Timestamp(va.index.min().year, 1, 1)
        assert (tr["label_known_date"] < val_start).all()      # purged: no label reaches into validation
        assert tr.index.max() < val_start <= va.index.min()


def _run_one(spec, year=2016):
    _, rows, fold = _train_rows(year)
    res = run_walk_forward(_frame(), {spec.name: spec}, [fold], list(BASE_FEATURES), MODELING_START, HOLDOUT)
    return res


def test_tuned_models_log_every_configuration_and_pick_the_best():
    for spec, n_configs in ((logit_tuned(), 3), (hist_gb(), 4)):
        res = _run_one(spec)
        assert len(res.trials) == n_configs
        assert res.preds["prob"].between(0, 1).all()
        assert set(res.trials["metric"]) == {"log_loss"}


def test_trees_learn_persistent_volatility_without_touching_the_test_year():
    for spec in (hist_gb(), random_forest()):
        res = _run_one(spec)
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(res.preds["y"], res.preds["prob"]) > 0.65
        # test-year labels must not influence predictions
        frame = _frame()
        in_test = (frame.index >= pd.Timestamp("2016-01-01")) & (frame.index <= pd.Timestamp("2016-12-31"))
        frame.loc[in_test, "y"] = (np.random.default_rng(1).random(in_test.sum()) < 0.5).astype(int)
        fold = make_folds(2016, 2016)
        again = run_walk_forward(frame, {spec.name: spec}, fold, list(BASE_FEATURES),
                                 MODELING_START, HOLDOUT)
        np.testing.assert_allclose(res.preds["prob"].to_numpy(), again.preds["prob"].to_numpy())
