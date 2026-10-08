"""NYSE trading calendar loader (populates ref.trading_day)."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Engine, text


def build_trading_days(start: date, end: date) -> pd.DataFrame:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    close_ny = schedule["market_close"].dt.tz_convert("America/New_York")
    early = (close_ny.dt.hour * 60 + close_ny.dt.minute) < 16 * 60
    return pd.DataFrame({"trade_date": schedule.index.date, "is_early_close": early.to_numpy()})


def load_calendar(engine: Engine, start: date, end: date) -> int:
    days = build_trading_days(start, end)
    rows = [
        {"d": r.trade_date, "e": bool(r.is_early_close)} for r in days.itertuples(index=False)
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                "insert into ref.trading_day (trade_date, is_early_close) values (:d, :e) "
                "on conflict (trade_date) do update set is_early_close = excluded.is_early_close"
            ),
            rows,
        )
    return len(rows)
