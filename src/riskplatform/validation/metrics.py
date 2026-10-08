"""Classification metrics (spec section 8.3). Database-free."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins of predicted probability vs observed frequency."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop") if df["p"].nunique() > 1 else 0
    g = df.groupby("bin", observed=True)
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_pos": g["y"].mean(), "n": g.size()})


def classification_metrics(y, p, bins: int = 10) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {"n": len(y), "prevalence": float(y.mean())}
    both_classes = y.min() != y.max()
    out["roc_auc"] = float(roc_auc_score(y, p)) if both_classes else np.nan
    out["pr_auc"] = float(average_precision_score(y, p)) if both_classes else np.nan
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["log_loss"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    cal = calibration_table(y, p, bins)
    out["ece"] = float((cal["n"] * (cal["mean_pred"] - cal["frac_pos"]).abs()).sum() / cal["n"].sum())
    return out
