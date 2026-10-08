"""Purging for walk-forward validation (spec section 8.1)."""
from __future__ import annotations

import pandas as pd


def train_mask(label_known_date: pd.Series, test_start: pd.Timestamp) -> pd.Series:
    """True for samples whose label was fully known strictly before the test block starts.

    Samples whose label window overlaps the test period are excluded, which removes
    the leakage caused by overlapping h-day labels.
    """
    return label_known_date.notna() & (label_known_date < pd.Timestamp(test_start))
