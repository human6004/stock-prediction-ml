"""Dự báo một symbol bằng dữ liệu offline và final model đã train.

Luồng: load clean data -> tính feature mới nhất -> load artifact/metadata
-> predict_proba -> áp decision_threshold -> trả dict cho Flask/CLI.
"""

from pathlib import Path

import joblib
import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
)
from services.feature_engineering import build_features
from services.preprocessing import clean_data, dataset_check


def _load_clean_data() -> pd.DataFrame:
    """Ưu tiên processed CSV; chỉ clean raw tại chỗ khi processed file thiếu."""
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH)
    if not Path(RAW_DATA_PATH).exists():
        raise FileNotFoundError(
            "Khong tim thay du lieu clean hoac raw. Hay chay pipeline truoc."
        )
    raw_df, _ = dataset_check()
    cleaned_all, _, _, _ = clean_data(raw_df)
    return cleaned_all


def compute_latest_features(symbol: str) -> pd.Series:
    """Build features on-the-fly for the latest trading date of a symbol."""
    normalized = symbol.strip().upper()
    clean_df = _load_clean_data()
    symbol_rows = clean_df[clean_df["symbol"] == normalized].copy()
    if symbol_rows.empty:
        raise ValueError(f"Khong tim thay ma {normalized} trong du lieu da xu ly.")

    features, _ = build_features(symbol_rows)
    if features.empty:
        raise ValueError(f"Khong du du lieu de tinh feature cho ma {normalized}.")
    latest = features.sort_values("trading_date").iloc[-1]
    return latest


def load_model_artifact() -> dict:
    """Load bundle gồm sklearn model và metadata tối thiểu lúc train."""
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chua co models/final_model.pkl. Hay chay: python scripts/run_pipeline.py"
        )
    return joblib.load(FINAL_MODEL_PATH)


def load_metadata() -> dict:
    """Load metadata report; fallback về thông tin nhúng trong artifact."""
    if MODEL_METADATA_PATH.exists():
        import json

        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    artifact = load_model_artifact()
    return {
        "model_name": artifact.get("model_name"),
        "feature_order": artifact.get("feature_columns", FEATURE_COLUMNS),
        "prediction_horizon": artifact.get("prediction_horizon"),
        "up_threshold": artifact.get("up_threshold"),
        "decision_threshold": artifact.get("decision_threshold", 0.5),
    }


def predict_symbol(symbol: str) -> dict:
    """Trả UP/NOT_UP cho row mới nhất; hàm này không fit hoặc sửa model."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Vui long nhap ma co phieu.")

    latest = compute_latest_features(normalized)
    artifact = load_model_artifact()
    metadata = load_metadata()
    model = artifact["model"]
    feature_columns = metadata.get("feature_order", artifact.get("feature_columns", FEATURE_COLUMNS))
    x_latest = latest[feature_columns].to_frame().T

    decision_threshold = metadata.get(
        "decision_threshold", artifact.get("decision_threshold", 0.5)
    )
    probability_up = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_latest)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability_up = float(probabilities[classes.index(1)])

    # Nhãn phục vụ dùng threshold đã tune, không dùng predict() mặc định 0.5.
    if probability_up is not None:
        label = "UP" if probability_up >= decision_threshold else "NOT_UP"
    else:
        label = "UP" if int(model.predict(x_latest)[0]) == 1 else "NOT_UP"

    horizon = metadata.get("prediction_horizon", artifact.get("prediction_horizon", 5))
    threshold = metadata.get("up_threshold", artifact.get("up_threshold", 0.01))
    model_name = metadata.get("model_name", artifact.get("model_name", "Unknown"))

    if label == "UP":
        prediction_label_vi = "Tang (UP)"
        prediction_short_vi = (
            f"Kha nang gia tang hon {int(threshold * 100)}% trong {horizon} phien tiep theo."
        )
    else:
        prediction_label_vi = "Khong du dieu kien tang (NOT_UP)"
        prediction_short_vi = (
            f"Khong du dieu kien tang hon {int(threshold * 100)}% trong {horizon} phien tiep theo."
        )

    prob_text = (
        f"{probability_up * 100:.1f}%"
        if probability_up is not None
        else "khong ho tro"
    )
    summary_sentence = (
        f"Tu phien {latest['trading_date']}, he thong du bao xu huong "
        f"{horizon} phien giao dich ke tiep la {prediction_label_vi} "
        f"(xac suat lop UP: {prob_text})."
    )

    return {
        "symbol": normalized,
        "latest_date": latest["trading_date"],
        "reference_date": latest["trading_date"],
        "close_at_reference": float(latest["close"]),
        "prediction": label,
        "probability_up": probability_up,
        "model_name": model_name,
        "horizon": horizon,
        "horizon_sessions": horizon,
        "threshold_percent": int(threshold * 100),
        "summary_sentence": summary_sentence,
        "prediction_label_vi": prediction_label_vi,
        "prediction_short_vi": prediction_short_vi,
    }
