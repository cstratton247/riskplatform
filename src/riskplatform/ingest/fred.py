"""FRED / ALFRED client and macro data checks. Database-free.

Two availability methods (spec section 5.2):
  * alfred_vintage: request every historical vintage; each row says which value was
    publicly known between valid_from and valid_to (inclusive).
  * rule_lag: for daily, unrevised series; a value is assumed available one US-federal
    business day after its observation date.

SECURITY: request errors are re-raised WITHOUT the underlying message, because the
requests library embeds the full URL (including the API key) in its exceptions.
"""
from __future__ import annotations

import warnings
from datetime import date

import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .dq import DQResult

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
PAGE_LIMIT = 100_000
OPEN_ENDED = date(9999, 12, 31)


def parse_observations(payload: dict) -> list[dict]:
    """FRED JSON -> rows. Missing values (published as '.') are skipped."""
    rows = []
    for o in payload.get("observations", []):
        if o["value"] == ".":
            continue
        rows.append(
            {
                "observation_date": date.fromisoformat(o["date"]),
                "valid_from": date.fromisoformat(o["realtime_start"]),
                "valid_to": date.fromisoformat(o["realtime_end"]),
                "value": float(o["value"]),
            }
        )
    return rows


def fetch_observations(
    series_id: str, api_key: str, start: date, all_vintages: bool, timeout: int = 30
) -> list[dict]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "limit": PAGE_LIMIT,
    }
    if all_vintages:
        params["realtime_start"] = "1776-07-04"          # FRED's earliest allowed date
        params["realtime_end"] = "9999-12-31"
    rows: list[dict] = []
    offset = 0
    while True:
        try:
            resp = requests.get(FRED_URL, params={**params, "offset": offset}, timeout=timeout)
        except requests.RequestException as exc:         # message may contain the API key
            raise RuntimeError(f"FRED request failed for {series_id}: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise RuntimeError(
                f"FRED returned HTTP {resp.status_code} for {series_id}: {resp.text[:200]}"
            )
        payload = resp.json()
        rows.extend(parse_observations(payload))
        if len(payload.get("observations", [])) < PAGE_LIMIT:
            return rows
        offset += PAGE_LIMIT


def apply_rule_lag(rows: list[dict]) -> list[dict]:
    """Daily unrevised series: known from one US-federal business day after the observation."""
    if not rows:
        return rows
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    obs = pd.DatetimeIndex([r["observation_date"] for r in rows])
    with warnings.catch_warnings():            # pandas warns about non-vectorized offsets; fine here
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        valid_from = (obs + bday).date
    return [
        {**r, "valid_from": vf, "valid_to": OPEN_ENDED} for r, vf in zip(rows, valid_from)
    ]


def macro_checks(
    rows: list[dict], series_id: str, method: str, frequency: str, today: date,
    min_lag_days: int = 20, max_lag_days: int = 90,
) -> list[DQResult]:
    if not rows:
        return [DQResult("no_rows", "error", False, series_id, {"row_count": 0})]
    results: list[DQResult] = []

    if method == "alfred_vintage":
        first_known: dict[date, date] = {}
        for r in rows:
            d = r["observation_date"]
            if d not in first_known or r["valid_from"] < first_known[d]:
                first_known[d] = r["valid_from"]
        bad = {
            d: (vf - d).days for d, vf in first_known.items()
            if not (min_lag_days <= (vf - d).days <= max_lag_days)
        }
        results.append(
            DQResult(
                "first_vintage_lag", "warn", not bad, series_id,
                {"expected_days": [min_lag_days, max_lag_days], "count": len(bad),
                 "sample": [[d.isoformat(), lag] for d, lag in sorted(bad.items())[:10]]},
            )
        )

    latest = max(r["observation_date"] for r in rows)
    max_age = 10 if frequency == "daily" else 75
    age = (today - latest).days
    results.append(
        DQResult("staleness", "warn", age <= max_age, series_id,
                 {"latest_observation": latest.isoformat(), "age_days": age})
    )
    return results
