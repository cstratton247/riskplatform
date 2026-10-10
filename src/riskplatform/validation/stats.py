"""Statistical tests for forecast comparison (spec section 8.4). Database-free.

* Diebold-Mariano with a Bartlett HAC variance (lag = horizon - 1) and the
  Harvey-Leybourne-Newbold small-sample correction.
* Stationary (block) bootstrap, because overlapping h-day labels make observations dependent.
* Holm step-down adjustment for the family of comparisons.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length, wrapped circularly."""
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    pos = np.arange(n)
    block_start = np.maximum.accumulate(np.where(new, pos, 0))
    starts = rng.integers(n, size=n)
    return (starts[block_start] + (pos - block_start)) % n


def long_run_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    v = float(dc @ dc) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / n
    return v


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 5) -> dict:
    """H0: equal expected loss. mean_diff < 0 means model A has the lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    mean = float(d.mean())
    v = long_run_variance(d, horizon - 1)
    if v <= 0:
        return {"mean_diff": mean, "stat": np.nan, "p": 1.0 if mean == 0 else np.nan}
    stat = mean / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat *= hln
    return {"mean_diff": mean, "stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), df=n - 1))}


def _ap(y, p):
    if y.sum() == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return average_precision_score(y, p)


def paired_fold_bootstrap(folds: list[tuple], draws: int, mean_block: float, seed: int) -> dict:
    """Bootstrap the difference in fold-averaged PR-AUC between models A and B.

    `folds` is a list of (y, p_a, p_b), one per fold, in time order. Within each fold the
    SAME resampled days are used for both models (paired), preserving time dependence.
    """
    rng = np.random.default_rng(seed)
    obs = np.nanmean([_ap(y, a) - _ap(y, b) for y, a, b in folds])
    out = np.empty(draws)
    for i in range(draws):
        diffs = []
        for y, a, b in folds:
            idx = stationary_bootstrap_indices(len(y), mean_block, rng)
            diffs.append(_ap(y[idx], a[idx]) - _ap(y[idx], b[idx]))
        out[i] = np.nanmean(diffs)
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return {
        "obs_diff": float(obs), "ci_low": float(np.percentile(out, 2.5)),
        "ci_high": float(np.percentile(out, 97.5)), "p": float(min(1.0, max(p, 1.0 / (draws + 1)))),
    }


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def sharpe(excess: np.ndarray, periods: int = 252) -> float:
    sd = np.std(excess, ddof=1)
    return float(np.mean(excess) / sd * np.sqrt(periods)) if sd > 0 else np.nan


def paired_sharpe_bootstrap(
    excess_a: np.ndarray, excess_b: np.ndarray, draws: int, mean_block: float, seed: int
) -> dict:
    """Bootstrap Sharpe(A) - Sharpe(B) using the SAME resampled days for both strategies."""
    a, b = np.asarray(excess_a, float), np.asarray(excess_b, float)
    rng = np.random.default_rng(seed)
    obs = sharpe(a) - sharpe(b)
    out = np.empty(draws)
    for i in range(draws):
        idx = stationary_bootstrap_indices(len(a), mean_block, rng)
        out[i] = sharpe(a[idx]) - sharpe(b[idx])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return {"obs_diff": float(obs), "ci_low": float(np.nanpercentile(out, 2.5)),
            "ci_high": float(np.nanpercentile(out, 97.5)),
            "p": float(min(1.0, max(p, 1.0 / (draws + 1))))}
