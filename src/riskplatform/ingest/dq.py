"""Data-quality checks for price data (spec section 5.3). Database-free.

`severity` is the level a check carries when it FAILS. Only failed checks with
severity 'error' block a load; failed 'warn' checks are recorded for review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DQResult:
    check_name: str
    severity: str                       # level if failing: 'warn' or 'error'
    passed: bool
    entity: str
    details: dict = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return (not self.passed) and self.severity == "error"

    @property
    def stored_severity(self) -> str:
        return self.severity if not self.passed else "info"


def _sample_dates(index, n=10) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in list(index)[:n]]


def run_price_checks(
    df: pd.DataFrame,
    ticker: str,
    asset_type: str,
    trading_days: pd.DatetimeIndex,
    outlier_threshold: float = 0.15,
    stale_run_length: int = 5,
    max_missing_fraction: float = 0.005,
) -> list[DQResult]:
    """`df` must be normalized (see normalize_prices) with one ticker's rows."""
    res: list[DQResult] = []

    if df.empty:
        return [DQResult("no_rows", "error", False, ticker, {"row_count": 0})]

    dates = pd.DatetimeIndex(df["trade_date"])

    n_dup = int(df["trade_date"].duplicated().sum())
    res.append(DQResult("duplicate_dates", "error", n_dup == 0, ticker, {"count": n_dup}))

    n_null_close = int(df["close"].isna().sum())
    res.append(DQResult("null_close", "error", n_null_close == 0, ticker, {"count": n_null_close}))

    price_cols = ["open", "high", "low", "close", "adj_close"]
    n_nonpos = int((df[price_cols] <= 0).any(axis=1).sum())
    res.append(DQResult("non_positive_price", "error", n_nonpos == 0, ticker, {"count": n_nonpos}))

    n_hl = int((df["high"] < df["low"]).sum())
    res.append(DQResult("high_below_low", "error", n_hl == 0, ticker, {"count": n_hl}))

    tol = 1e-6
    outside = (df["close"] > df["high"] * (1 + tol)) | (df["close"] < df["low"] * (1 - tol))
    res.append(
        DQResult(
            "close_outside_range", "warn", int(outside.sum()) == 0, ticker,
            {"count": int(outside.sum()), "sample": _sample_dates(dates[outside.to_numpy()])},
        )
    )

    extra = dates.difference(trading_days)
    res.append(
        DQResult("non_trading_dates", "error", len(extra) == 0, ticker,
                 {"count": len(extra), "sample": _sample_dates(extra)})
    )

    expected = trading_days[(trading_days >= dates.min()) & (trading_days <= dates.max())]
    missing = expected.difference(dates)
    frac = len(missing) / max(len(expected), 1)
    res.append(
        DQResult(
            "missing_trading_days",
            "error" if frac > max_missing_fraction else "warn",
            len(missing) == 0,
            ticker,
            {"count": len(missing), "fraction": round(frac, 6), "sample": _sample_dates(missing)},
        )
    )

    if asset_type != "volatility_index":          # a volatility index is a level, not a price
        r = np.log(df["close"]).diff()
        big = r.abs() > outlier_threshold
        res.append(
            DQResult(
                "return_outliers", "warn", int(big.sum()) == 0, ticker,
                {"count": int(big.sum()), "threshold": outlier_threshold,
                 "sample": _sample_dates(dates[big.to_numpy()])},
            )
        )

    same = df["close"].diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    max_identical = int(run.max()) + 1
    res.append(
        DQResult("stale_prices", "warn", max_identical < stale_run_length, ticker,
                 {"max_identical_closes": max_identical})
    )
    return res


def drop_minor_non_trading_rows(
    df: pd.DataFrame, trading_days: pd.DatetimeIndex, ticker: str, max_fraction: float
) -> tuple[pd.DataFrame, DQResult | None]:
    """Drop a SMALL number of rows dated on non-trading days, and record them.

    Vendors occasionally publish filler rows on exchange holidays. If the off-calendar
    fraction is at most `max_fraction` the rows are removed and flagged (never silently);
    above it, the frame is returned untouched so the `non_trading_dates` check blocks
    the load, since a large number points to a systematic date problem.
    """
    if df.empty:
        return df, None
    extra = ~pd.DatetimeIndex(df["trade_date"]).isin(trading_days)
    n = int(extra.sum())
    if n == 0 or n / len(df) > max_fraction:
        return df, None
    dropped = df.loc[extra, ["trade_date", "close"]]
    result = DQResult(
        "non_trading_rows_dropped", "warn", False, ticker,
        {
            "count": n,
            "rows": [
                {"date": d.strftime("%Y-%m-%d"), "close": None if pd.isna(c) else float(c)}
                for d, c in zip(dropped["trade_date"], dropped["close"])
            ],
        },
    )
    return df.loc[~extra].reset_index(drop=True), result
