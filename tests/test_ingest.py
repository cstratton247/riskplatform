import numpy as np
import pandas as pd

from riskplatform.ingest.dq import drop_minor_non_trading_rows, run_price_checks
from riskplatform.ingest.normalize import normalize_prices, payload_and_hash, to_records

DAYS = pd.bdate_range("2020-01-01", periods=1000)


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return normalize_prices(
        pd.DataFrame(
            {
                "ticker": "X", "trade_date": DAYS[:n], "open": close, "high": close * 1.005,
                "low": close * 0.995, "close": close, "adj_close": close,
                "volume": rng.integers(1_000, 5_000, n),
            }
        )
    )


def _by_name(results):
    return {r.check_name: r for r in results}


def _run(df, asset_type="etf"):
    return _by_name(run_price_checks(df, "X", asset_type, DAYS))


def test_clean_data_passes_everything():
    results = run_price_checks(_frame(), "X", "etf", DAYS)
    assert all(r.passed for r in results)
    assert not any(r.blocking for r in results)


def test_duplicate_dates_block():
    df = _frame()
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    assert _run(df)["duplicate_dates"].blocking


def test_high_below_low_blocks():
    df = _frame()
    df.loc[5, "high"] = df.loc[5, "low"] - 1
    assert _run(df)["high_below_low"].blocking


def test_weekend_row_blocks():
    df = _frame()
    df.loc[3, "trade_date"] = pd.Timestamp("2020-01-04")      # a Saturday
    assert _run(df)["non_trading_dates"].blocking


def test_small_gap_warns_but_large_gap_blocks():
    small = _frame().drop(index=[100]).reset_index(drop=True)             # 0.1% missing
    r = _run(small)["missing_trading_days"]
    assert not r.passed and not r.blocking
    large = _frame().drop(index=range(100, 112)).reset_index(drop=True)   # 1.2% missing
    assert _run(large)["missing_trading_days"].blocking


def test_return_outlier_flagged_but_not_for_vol_index():
    df = _frame()
    df.loc[500:, ["open", "high", "low", "close", "adj_close"]] *= 0.75   # -25% jump
    assert not _run(df)["return_outliers"].passed
    assert "return_outliers" not in _run(df, asset_type="volatility_index")


def test_stale_prices_warn():
    df = _frame()
    df.loc[200:205, "close"] = df.loc[200, "close"]
    r = _run(df)["stale_prices"]
    assert not r.passed and not r.blocking


def test_empty_frame_blocks():
    empty = _frame().iloc[0:0]
    assert run_price_checks(empty, "X", "etf", DAYS)[0].blocking


def test_normalization_payload_and_records():
    df = _frame(5)
    df.loc[2, "volume"] = np.nan
    p1, h1 = payload_and_hash(df)
    p2, h2 = payload_and_hash(df.copy())
    assert h1 == h2 and "NaN" not in p1                      # deterministic, NaN -> null
    recs = to_records(df, asset_id=1, source_id=2, pull_id=3)
    assert recs[2]["volume"] is None
    assert isinstance(recs[0]["volume"], int)
    assert recs[0]["trade_date"] == DAYS[0].date()


def test_minor_holiday_rows_are_dropped_and_flagged():
    df = _frame()
    extra = df.iloc[[10, 20]].copy()
    extra["trade_date"] = [pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05")]  # weekend
    df = pd.concat([df, extra], ignore_index=True).sort_values("trade_date").reset_index(drop=True)
    cleaned, result = drop_minor_non_trading_rows(df, DAYS, "X", max_fraction=0.01)
    assert len(cleaned) == 1000 and result is not None and result.details["count"] == 2
    assert not result.blocking                      # recorded as a warning, not a failure
    assert all(r.passed for r in run_price_checks(cleaned, "X", "etf", DAYS))


def test_many_off_calendar_rows_still_block():
    df = _frame()
    extra = df.iloc[:50].copy()
    extra["trade_date"] = extra["trade_date"] + pd.Timedelta(days=1)   # shifted dates
    df = pd.concat([df, extra], ignore_index=True)
    cleaned, result = drop_minor_non_trading_rows(df, DAYS, "X", max_fraction=0.001)
    assert result is None and len(cleaned) == len(df)                   # nothing dropped
    assert _run(cleaned)["non_trading_dates"].blocking                  # the main check blocks
