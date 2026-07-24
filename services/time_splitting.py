"""Date-based, purged cross-validation splits for panel market data."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import CV_GAP_SESSIONS, CV_N_SPLITS, CV_START_DATE


def sort_panel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic panel order shared by tuning and official CV."""
    required = {"trading_date", "symbol"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing panel sort columns: {sorted(missing)}")
    return frame.sort_values(["trading_date", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def iter_purged_date_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = CV_N_SPLITS,
    gap_sessions: int = CV_GAP_SESSIONS,
    start_date: str | None = CV_START_DATE,
    date_column: str = "trading_date",
    label_end_column: str = "label_end_date",
):
    """Yield positional row indices using whole market dates and purged labels."""
    missing = {date_column, label_end_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    dates = pd.to_datetime(frame[date_column], errors="raise").reset_index(drop=True)
    label_ends = pd.to_datetime(
        frame[label_end_column], errors="raise"
    ).reset_index(drop=True)
    market_dates = np.sort(dates.unique())
    if start_date is not None:
        market_dates = market_dates[market_dates >= pd.Timestamp(start_date)]
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_sessions)

    for train_date_idx, validation_date_idx in splitter.split(market_dates):
        train_dates = market_dates[train_date_idx]
        validation_dates = market_dates[validation_date_idx]
        validation_start = validation_dates[0]
        train_mask = dates.isin(train_dates) & (label_ends < validation_start)
        validation_mask = dates.isin(validation_dates)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        validation_idx = np.flatnonzero(validation_mask.to_numpy())
        if not len(train_idx) or not len(validation_idx):
            raise ValueError("Date-based CV produced an empty train or validation fold.")
        yield train_idx, validation_idx
