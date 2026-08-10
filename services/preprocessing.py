"""Kiểm tra, làm sạch OHLCV và xác định symbol đủ điều kiện train.

Module không tạo feature/model. Nó trả cả `cleaned_all` cho prediction và
`filtered_df` chỉ gồm symbol đủ MIN_TRADING_DAYS cho training.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    DATA_QUALITY_REPORT_PATH,
    ELIGIBLE_SYMBOLS_PATH,
    EXCLUDED_SYMBOLS_PATH,
    MIN_AVERAGE_VOLUME,
    MIN_TRADING_DAYS,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
)
from services.pipeline_utils import atomic_dataframe_to_csv


def dataset_check(raw_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Đọc raw CSV và thống kê lỗi; chưa xóa hay sửa dòng nào."""
    path = Path(raw_path or RAW_DATA_PATH)
    report = {
        "path": str(path),
        "file_exists": path.exists(),
        "read_success": False,
    }
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    report["read_success"] = True
    report["rows"] = int(len(df))
    report["columns"] = list(df.columns)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    report["missing_required_columns"] = missing_columns
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    dates = pd.to_datetime(df["trading_date"], errors="coerce")
    numeric = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid_ohlc = (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
    )
    negative_rows = (numeric[["open", "high", "low", "close", "volume"]] < 0).any(axis=1)

    report.update(
        {
            "symbols": int(df["symbol"].nunique(dropna=True)),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "missing_values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
            "missing_by_column": {
                k: int(v) for k, v in df[REQUIRED_COLUMNS].isna().sum().to_dict().items()
            },
            "duplicates_symbol_trading_date": int(df.duplicated(["symbol", "trading_date"]).sum()),
            "invalid_ohlc_rows": int(invalid_ohlc.sum()),
            "negative_price_volume_rows": int(negative_rows.sum()),
        }
    )
    return df, report


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Chuẩn hóa row, bỏ dữ liệu sai và lọc symbol đủ điều kiện train."""
    df = raw_df[REQUIRED_COLUMNS].copy()
    original_rows = len(df)

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    after_dropna = len(df)
    df = df.drop_duplicates(subset=["symbol", "trading_date"], keep="last")
    after_dedup = len(df)

    # Một nến hợp lệ phải có low <= open/close/high và high >= các giá còn lại.
    valid_prices = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = df["volume"] >= 0
    valid_ohlc = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    )
    df = df[valid_prices & valid_volume & valid_ohlc].copy()
    df["volume"] = df["volume"].round().astype("int64")
    df["trading_date"] = df["trading_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["symbol", "trading_date"]).reset_index(drop=True)

    symbol_stats = (
        df.groupby("symbol")
        .agg(
            rows=("trading_date", "size"),
            start_date=("trading_date", "min"),
            end_date=("trading_date", "max"),
            average_volume=("volume", "mean"),
        )
        .reset_index()
    )

    reasons = []
    eligible_flags = []
    for _, row in symbol_stats.iterrows():
        reason_parts = []
        if row["rows"] < MIN_TRADING_DAYS:
            reason_parts.append("fewer_than_250_trading_days")
        if MIN_AVERAGE_VOLUME > 0 and row["average_volume"] < MIN_AVERAGE_VOLUME:
            reason_parts.append("low_average_volume")
        reason = ";".join(reason_parts)
        reasons.append(reason)
        eligible_flags.append(len(reason_parts) == 0)

    symbol_stats["eligible_for_training"] = eligible_flags
    symbol_stats["exclusion_reason"] = reasons

    # cleaned_all vẫn giữ mọi mã hợp lệ để web dự báo; filtered_df mới dùng train.
    eligible_symbols = set(symbol_stats.loc[symbol_stats["eligible_for_training"], "symbol"])
    filtered_df = df[df["symbol"].isin(eligible_symbols)].copy()

    report = {
        "original_rows": int(original_rows),
        "rows_after_drop_missing": int(after_dropna),
        "rows_after_drop_duplicates": int(after_dedup),
        "rows_after_cleaning": int(len(df)),
        "rows_removed_by_cleaning": int(original_rows - len(df)),
        "symbols_after_cleaning": int(df["symbol"].nunique()),
        "eligible_symbols": int(len(eligible_symbols)),
        "excluded_symbols": int(len(symbol_stats) - len(eligible_symbols)),
        "rows_after_symbol_filter": int(len(filtered_df)),
        "min_trading_days": MIN_TRADING_DAYS,
        "min_average_volume": MIN_AVERAGE_VOLUME,
    }
    return df, filtered_df, symbol_stats, report


def write_clean_outputs(
    cleaned_all: pd.DataFrame,
    symbol_stats: pd.DataFrame,
) -> None:
    """Ghi 4 output của bước làm sạch, mỗi file một mục đích riêng.

    - ``cleaned_hose_stock.csv``: TOÀN BỘ dòng đã sạch (không lọc mã). Đây là
      nguồn duy nhất cho các bước sau; ``cleaned_for_training`` được tính LẠI từ
      file này ở ``scripts/build_features.py`` nên không ghi ra đĩa.
    - ``data_quality_report.csv``: thống kê từng mã, gồm cờ ``eligible_for_training``.
    - ``eligible_symbols.csv`` / ``excluded_symbols.csv``: tách đôi theo cờ đó.
      ``eligible_symbols.csv`` không chỉ để đọc cho vui — nó là bộ lọc mã ở bước
      build features, và là scope dự phòng của chatbot với artifact cũ chưa có
      ``training_symbols`` trong metadata.

    Toàn bộ ghi qua ``atomic_dataframe_to_csv`` (tmp rồi ``os.replace``) để web UI
    đang đọc song song không bao giờ thấy file ghi dở.
    """
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_to_csv(cleaned_all, CLEANED_DATA_PATH, index=False)
    atomic_dataframe_to_csv(symbol_stats, DATA_QUALITY_REPORT_PATH, index=False)

    eligible = symbol_stats[symbol_stats["eligible_for_training"]].copy()
    excluded = symbol_stats[~symbol_stats["eligible_for_training"]].copy()
    atomic_dataframe_to_csv(eligible, ELIGIBLE_SYMBOLS_PATH, index=False)
    atomic_dataframe_to_csv(excluded, EXCLUDED_SYMBOLS_PATH, index=False)
