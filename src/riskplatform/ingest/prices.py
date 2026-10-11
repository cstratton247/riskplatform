"""Price loading pipeline: fetch -> raw audit record -> DQ checks -> upsert.

Design decisions
  * FULL-HISTORY REFRESH each run. Adjusted close is restated by the vendor after every
    dividend; mixing rows from different pulls would put a break in the adjusted series
    (and in returns computed from it). Replacing the whole history keeps it consistent.
  * One transaction per ticker, so one bad ticker never rolls back the others.
  * The raw pull and every DQ result are stored even when the ticker is rejected.
  * Today's (possibly partial) bar is excluded: the default end date is today, exclusive.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from riskplatform.config import DEFAULT_CONFIG_PATH

from .base import DataSource
from .dq import DQResult, drop_minor_non_trading_rows, run_price_checks
from .normalize import normalize_prices, payload_and_hash, to_records

log = logging.getLogger(__name__)

UPSERT_SQL = text(
    """
    insert into mkt.price_daily
        (asset_id, trade_date, open, high, low, close, adj_close, volume, source_id, pull_id)
    values
        (:asset_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume,
         :source_id, :pull_id)
    on conflict (asset_id, trade_date) do update set
        open = excluded.open, high = excluded.high, low = excluded.low,
        close = excluded.close, adj_close = excluded.adj_close, volume = excluded.volume,
        source_id = excluded.source_id, pull_id = excluded.pull_id, ingested_at = now()
    """
)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _config_hash() -> str:
    return hashlib.sha256(Path(DEFAULT_CONFIG_PATH).read_bytes()).hexdigest()


def _store_dq(conn, run_id: int, results: list[DQResult]) -> None:
    if not results:
        return
    import json

    conn.execute(
        text(
            "insert into ops.dq_check_result (run_id, check_name, severity, passed, entity, details) "
            "values (:run_id, :check_name, :severity, :passed, :entity, cast(:details as jsonb))"
        ),
        [
            {
                "run_id": run_id,
                "check_name": r.check_name,
                "severity": r.stored_severity,
                "passed": r.passed,
                "entity": r.entity,
                "details": json.dumps(r.details),
            }
            for r in results
        ],
    )


def _revision_check(conn, asset_id: int, ticker: str, new: pd.DataFrame, tol: float) -> DQResult:
    old = pd.read_sql(
        text("select trade_date, close from mkt.price_daily where asset_id = :a"),
        conn,
        params={"a": asset_id},
    )
    if old.empty:
        return DQResult("vendor_revision_close", "warn", True, ticker, {"compared": 0})
    old["trade_date"] = pd.to_datetime(old["trade_date"])
    old["close"] = old["close"].astype(float)
    merged = old.merge(new[["trade_date", "close"]], on="trade_date", suffixes=("_old", "_new"))
    rel = (merged["close_new"] - merged["close_old"]).abs() / merged["close_old"]
    changed = merged[rel > tol]
    return DQResult(
        "vendor_revision_close", "warn", changed.empty, ticker,
        {"compared": len(merged), "changed": len(changed),
         "sample": changed["trade_date"].dt.strftime("%Y-%m-%d").head(10).tolist()},
    )


def load_prices(
    engine: Engine,
    source: DataSource,
    cfg: dict,
    tickers: list[str] | None,
    start: date,
    end: date,
) -> bool:
    """Returns True if every ticker loaded without a blocking failure."""
    from .calendar import load_calendar

    # Keep the trading calendar current; an out-of-date calendar would otherwise hide new days.
    load_calendar(engine, date.fromisoformat(cfg["data"]["start_date"]), end)

    configured = [t for group in cfg["data"]["tickers"].values() for t in group]
    wanted = tickers or configured
    dq_cfg = cfg["dq"]

    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                "insert into ops.pipeline_run (pipeline_name, git_sha, config_hash) "
                "values ('load_prices', :sha, :cfg) returning run_id"
            ),
            {"sha": _git_sha(), "cfg": _config_hash()},
        ).scalar_one()
        assets = {
            r.ticker: (r.asset_id, r.asset_type)
            for r in conn.execute(text("select ticker, asset_id, asset_type from ref.asset"))
        }
        source_id = conn.execute(
            text("select source_id from ref.data_source where name = :n"), {"n": source.name}
        ).scalar_one()
        td = pd.read_sql(text("select trade_date from ref.trading_day order by 1"), conn)

    if td.empty:
        raise RuntimeError("ref.trading_day is empty; run `riskplatform init-calendar` first")
    trading_days = pd.DatetimeIndex(pd.to_datetime(td["trade_date"]))

    rows_read = rows_written = 0
    failures: list[str] = []

    for ticker in wanted:
        if ticker not in assets:
            failures.append(f"{ticker}: not in ref.asset")
            log.error("%s not in ref.asset", ticker)
            continue
        asset_id, asset_type = assets[ticker]
        try:
            raw_df = normalize_prices(source.fetch_prices([ticker], start, end))
        except Exception as exc:                                  # vendor/network failure
            failures.append(f"{ticker}: fetch failed ({exc})")
            log.error("%s fetch failed: %s", ticker, exc)
            continue
        rows_read += len(raw_df)

        # The raw audit record keeps everything the vendor sent; cleaning happens after.
        payload, payload_hash = payload_and_hash(raw_df)
        df, drop_result = drop_minor_non_trading_rows(
            raw_df, trading_days, ticker, dq_cfg["max_non_trading_fraction"]
        )
        results = run_price_checks(
            df, ticker, asset_type, trading_days,
            outlier_threshold=dq_cfg["return_outlier_threshold"],
            stale_run_length=dq_cfg["stale_run_length"],
            max_missing_fraction=dq_cfg["max_missing_fraction"],
        )
        if drop_result is not None:
            results.append(drop_result)

        with engine.begin() as conn:
            pull_id = conn.execute(
                text(
                    "insert into raw.price_pull (run_id, source_id, asset_id, requested_start, "
                    "requested_end, payload_hash, row_count, payload) values "
                    "(:run, :src, :asset, :s, :e, :h, :n, cast(:p as jsonb)) returning pull_id"
                ),
                {"run": run_id, "src": source_id, "asset": asset_id, "s": start, "e": end,
                 "h": payload_hash, "n": len(raw_df), "p": payload},
            ).scalar_one()

            if not df.empty:
                results.append(
                    _revision_check(conn, asset_id, ticker, df, dq_cfg["revision_tolerance"])
                )
            _store_dq(conn, run_id, results)

            blocking = [r for r in results if r.blocking]
            if blocking:
                names = ", ".join(r.check_name for r in blocking)
                failures.append(f"{ticker}: blocked by {names}")
                log.error("%s rejected: %s", ticker, names)
                continue

            conn.execute(UPSERT_SQL, to_records(df, asset_id, source_id, pull_id))
            conn.execute(
                text("update ref.asset set first_trade_date = :d where asset_id = :a"),
                {"d": df["trade_date"].min().date(), "a": asset_id},
            )
            rows_written += len(df)
            warned = [r.check_name for r in results if not r.passed]
            log.info("%s loaded: %d rows%s", ticker, len(df),
                     f" (warnings: {', '.join(warned)})" if warned else "")

    with engine.begin() as conn:
        conn.execute(
            text(
                "update ops.pipeline_run set finished_at = now(), status = :st, "
                "rows_read = :rr, rows_written = :rw, error_message = :err where run_id = :id"
            ),
            {"st": "failed" if failures else "succeeded", "rr": rows_read, "rw": rows_written,
             "err": "; ".join(failures) or None, "id": run_id},
        )
    return not failures
