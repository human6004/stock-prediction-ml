"""Đồng bộ CSV/report sang SQLite và ghi lịch sử prediction.

Raw/clean/features dùng `replace` nên được dựng lại toàn bảng. Tuning/evaluation
dùng `append` để giữ lịch sử; prediction được INSERT từng lần gọi web/CLI.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_DATA_PATH,
    MODEL_COMPARISON_PATH,
    RAW_DATA_PATH,
    TUNING_RESULTS_PATH,
)
from database.db_connection import get_connection, init_database


def sync_raw_data() -> int:
    """Thay toàn bộ raw_prices bằng snapshot raw CSV hiện tại."""
    if not _path_exists(RAW_DATA_PATH):
        return 0
    df = pd.read_csv(RAW_DATA_PATH)
    with get_connection() as conn:
        df.to_sql("raw_prices", conn, if_exists="replace", index=False)
        conn.commit()
    return len(df)


def sync_clean_data() -> int:
    """Thay toàn bộ clean_prices bằng snapshot clean CSV hiện tại."""
    if not _path_exists(CLEANED_DATA_PATH):
        return 0
    df = pd.read_csv(CLEANED_DATA_PATH)
    with get_connection() as conn:
        df.to_sql("clean_prices", conn, if_exists="replace", index=False)
        conn.commit()
    return len(df)


def sync_features() -> int:
    """Gói 20 feature thành JSON cho từng (symbol, trading_date), rồi replace."""
    if not _path_exists(FEATURE_DATA_PATH):
        return 0
    df = pd.read_csv(FEATURE_DATA_PATH)
    rows = []
    for _, row in df.iterrows():
        feature_payload = {col: row[col] for col in FEATURE_COLUMNS if col in row}
        rows.append(
            {
                "symbol": row["symbol"],
                "trading_date": row["trading_date"],
                "close": row.get("close"),
                "feature_json": json.dumps(feature_payload),
            }
        )
    feature_db = pd.DataFrame(rows)
    with get_connection() as conn:
        feature_db.to_sql("features", conn, if_exists="replace", index=False)
        conn.commit()
    return len(feature_db)


def sync_tuning_results() -> int:
    """Append ba CV summary của official run; không phải toàn tuning history."""
    if not _path_exists(TUNING_RESULTS_PATH):
        return 0
    df = pd.read_csv(TUNING_RESULTS_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "model_name": row["model_name"],
                "cv_f1_up": row.get("cv_f1_up"),
                "best_params_json": "{}",
                "created_at": now,
            }
        )
    out = pd.DataFrame(records)
    with get_connection() as conn:
        out.to_sql("tuning_results", conn, if_exists="append", index=False)
        conn.commit()
    return len(out)


def sync_evaluations() -> int:
    """Append TEST metrics của bốn artifact từ model_comparison.csv."""
    if not _path_exists(MODEL_COMPARISON_PATH):
        return 0
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "model_name": row["model_name"],
                "accuracy": row.get("accuracy"),
                "f1_up": row.get("f1_up"),
                "recall_up": row.get("recall_up"),
                "selected": int(str(row.get("selected", False)).lower() == "true"),
                "created_at": now,
            }
        )
    out = pd.DataFrame(records)
    with get_connection() as conn:
        out.to_sql("model_evaluations", conn, if_exists="append", index=False)
        conn.commit()
    return len(out)


def log_prediction(result: dict) -> None:
    """Ghi một lần dự báo; caller có thể chọn best-effort để UI không bị lỗi."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (symbol, reference_date, prediction, probability_up, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("symbol"),
                result.get("reference_date"),
                result.get("prediction"),
                result.get("probability_up"),
                result.get("model_name"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def sync_all() -> dict:
    """Khởi tạo DB rồi sync lần lượt toàn bộ snapshot và report hiện có."""
    init_database()
    return {
        "raw_rows": sync_raw_data(),
        "clean_rows": sync_clean_data(),
        "feature_rows": sync_features(),
        "tuning_rows": sync_tuning_results(),
        "evaluation_rows": sync_evaluations(),
    }


def _path_exists(path) -> bool:
    return Path(path).exists()
