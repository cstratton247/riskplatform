"""Pure (database-free) helpers for shaping price data."""
from __future__ import annotations

import hashlib

import pandas as pd

NUMERIC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Typed, date-sorted copy: trade_date as midnight timestamps, numerics coerced."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("trade_date").reset_index(drop=True)


def payload_and_hash(df: pd.DataFrame) -> tuple[str, str]:
    """Canonical JSON of the pulled rows and its SHA-256, for the raw audit record."""
    payload = df.assign(trade_date=df["trade_date"].dt.strftime("%Y-%m-%d")).to_json(
        orient="records"
    )
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _num(x):
    return None if pd.isna(x) else float(x)


def to_records(df: pd.DataFrame, asset_id: int, source_id: int, pull_id: int) -> list[dict]:
    """Rows as parameter dicts for the upsert; NaN becomes NULL."""
    records = []
    for r in df.itertuples(index=False):
        records.append(
            {
                "asset_id": asset_id,
                "trade_date": r.trade_date.date(),
                "open": _num(r.open),
                "high": _num(r.high),
                "low": _num(r.low),
                "close": _num(r.close),
                "adj_close": _num(r.adj_close),
                "volume": None if pd.isna(r.volume) else int(r.volume),
                "source_id": source_id,
                "pull_id": pull_id,
            }
        )
    return records
