"""Reporting: fold-averaged metrics and the pre-declared pairwise comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import dm_test, holm, paired_fold_bootstrap

ML_MODELS = ["logit_tuned", "hist_gb", "random_forest"]
BENCHMARKS = ["har_logit", "vix_logit"]            # H1 and H2 in the spec


def fold_mean_table(metrics: pd.DataFrame, reference: str = "har_logit") -> pd.DataFrame:
    """PRIMARY summary: each metric averaged over folds, every year weighted equally."""
    per_fold = metrics[metrics["scope"] != "pooled"]
    cols = ["pr_auc", "roc_auc", "brier", "log_loss", "ece"]
    table = per_fold.groupby("model", sort=False)[cols].mean()
    if reference in per_fold["model"].values:
        ref = per_fold[per_fold["model"] == reference].set_index("scope")["pr_auc"]
        wins = {}
        for model, g in per_fold.groupby("model", sort=False):
            g = g.set_index("scope")["pr_auc"]
            wins[model] = int((g > ref.reindex(g.index)).sum())
        table[f"folds_beating_{reference}"] = pd.Series(wins)
    return table.sort_values("pr_auc", ascending=False)


def _logloss_per_day(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pairwise_tests(preds: pd.DataFrame, stats_cfg: dict, pairs=None) -> pd.DataFrame:
    """Each ML model vs each benchmark. Holm-adjusted within each test family."""
    models = set(preds["model"].unique())
    pairs = pairs or [(a, b) for a in ML_MODELS for b in BENCHMARKS if a in models and b in models]
    rows = []
    for a, b in pairs:
        pa = preds[preds["model"] == a].set_index(["fold", "as_of_date"]).sort_index()
        pb = preds[preds["model"] == b].set_index(["fold", "as_of_date"]).sort_index()
        pa, pb = pa.align(pb, join="inner", axis=0)
        folds = [
            (g["y"].to_numpy(int), g["prob"].to_numpy(),
             pb.loc[fold]["prob"].to_numpy())
            for fold, g in pa.groupby(level="fold", sort=True)
        ]
        boot = paired_fold_bootstrap(folds, stats_cfg["bootstrap_draws"],
                                     stats_cfg["mean_block_length"], stats_cfg["seed"])
        dm = dm_test(_logloss_per_day(pa["y"], pa["prob"]), _logloss_per_day(pb["y"], pb["prob"]),
                     stats_cfg["dm_horizon"])
        rows.append({"model": a, "vs": b, "pr_auc_diff": boot["obs_diff"],
                     "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "p_boot": boot["p"],
                     "logloss_diff": dm["mean_diff"], "p_dm": dm["p"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_boot_holm"] = holm(out["p_boot"])
        out["p_dm_holm"] = holm(out["p_dm"].fillna(1.0))
    return out
