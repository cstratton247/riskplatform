"""Yahoo Finance adapter (development use only; terms forbid redistribution).

Not yet exercised against the live service.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .base import PRICE_COLUMNS, DataSource


class YahooSource(DataSource):
    name = "yahoo_finance"

    def fetch_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so the package imports without it

        frames = []
        for ticker in tickers:
            raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if raw.empty:
                raise RuntimeError(f"No data returned for {ticker}")
            df = raw.rename(
                columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                }
            )
            df["trade_date"] = pd.to_datetime(df.index).tz_localize(None).normalize().date
            df["ticker"] = ticker
            frames.append(df[PRICE_COLUMNS])
        return pd.concat(frames, ignore_index=True)
