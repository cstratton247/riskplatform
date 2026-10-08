"""Data-source adapter interface.

Every price provider implements DataSource and returns the same normalized
frame, so swapping the development provider (Yahoo) for a licensed one never
touches downstream code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

PRICE_COLUMNS = ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]


class DataSource(ABC):
    name: str

    @abstractmethod
    def fetch_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Return one row per (ticker, trade_date) with exactly PRICE_COLUMNS."""
