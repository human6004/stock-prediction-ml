"""Dự báo offline bằng final model đã train, cho một hoặc hai mã."""

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    DECISION_THRESHOLD,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    RAW_DATA_PATH,
)
from services.feature_engineering import build_features
from services.preprocessing import clean_data, dataset_check


def _load_clean_data() -> pd.DataFrame:
    """Ưu tiên processed CSV; chỉ clean raw tại chỗ khi processed file thiếu."""
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH)
    if not Path(RAW_DATA_PATH).exists():
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu clean hoặc raw. Hãy chạy pipeline trước."
        )
    raw_df, _ = dataset_check()
    cleaned_all, _, _, _ = clean_data(raw_df)
    return cleaned_all


def load_model_artifact() -> dict:
    """Load bundle gồm sklearn model và metadata tối thiểu lúc train."""
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có models/final_model.pkl. Hãy chạy: python scripts/run_pipeline.py"
        )
    return joblib.load(FINAL_MODEL_PATH)


def load_metadata(artifact: dict | None = None) -> dict:
    """Load metadata report; fallback về artifact đã có nếu được truyền vào."""
    if artifact is None:
        artifact = load_model_artifact()
    artifact_metadata = {
        "model_name": artifact.get("model_name"),
        "feature_order": artifact.get("feature_columns", FEATURE_COLUMNS),
        "prediction_horizon": artifact.get("prediction_horizon"),
        "up_threshold": artifact.get("up_threshold"),
        "decision_threshold": float(
            artifact.get("decision_threshold", DECISION_THRESHOLD)
        ),
        "policy_id": artifact.get("policy_id", "legacy"),
        "content_fingerprint": artifact.get("content_fingerprint"),
        "baseline_passed": artifact.get("baseline_passed"),
        "baseline_warning": artifact.get("baseline_warning"),
        "train_through_date": artifact.get("train_through_date"),
        "validation_selection_metrics": artifact.get(
            "validation_selection_metrics", {}
        ),
        "final_test_metrics": artifact.get("final_test_metrics", {}),
    }
    if not MODEL_METADATA_PATH.exists():
        return artifact_metadata
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    identity_keys = ("model_name", "policy_id", "content_fingerprint")
    if any(
        artifact_metadata.get(key) != metadata.get(key)
        for key in identity_keys
        if artifact_metadata.get(key) is not None
    ):
        return artifact_metadata
    metadata["decision_threshold"] = float(
        artifact.get(
            "decision_threshold", metadata.get("decision_threshold", DECISION_THRESHOLD)
        )
    )
    return metadata


def _normalize_symbols(symbols: list[str]) -> list[str]:
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise ValueError("Vui lòng nhập từ một đến hai mã cổ phiếu.")

    normalized = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Vui lòng nhập mã cổ phiếu.")
        value = symbol.strip().upper()
        if value not in normalized:
            normalized.append(value)

    if len(normalized) > 2:
        raise ValueError("Chỉ có thể dự báo tối đa hai mã cổ phiếu mỗi lần.")
    return normalized


def _forecast_sessions(reference_date: date, *, horizon: int) -> list[dict]:
    sessions = []
    candidate = reference_date
    while len(sessions) < horizon:
        candidate += timedelta(days=1)
        if candidate.weekday() < 5:
            sessions.append(
                {"step": len(sessions) + 1, "date": candidate.isoformat(), "estimated": True}
            )
    return sessions


def _as_date(value) -> date:
    return pd.to_datetime(value).date()


def predict_symbols(symbols: list[str]) -> list[dict]:
    """Dự báo 1-2 mã bằng một lượt đọc dữ liệu, build feature và inference."""
    normalized = _normalize_symbols(symbols)
    clean_df = _load_clean_data()
    clean_df = clean_df.copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str).str.strip().str.upper()

    known_symbols = set(clean_df["symbol"])
    unknown = [symbol for symbol in normalized if symbol not in known_symbols]
    if unknown:
        raise ValueError(f"Không tìm thấy mã {unknown[0]} trong dữ liệu đã xử lý.")

    symbol_rows = clean_df[clean_df["symbol"].isin(normalized)].copy()
    features, _ = build_features(symbol_rows)
    featured_symbols = set(features["symbol"]) if not features.empty else set()
    insufficient = [symbol for symbol in normalized if symbol not in featured_symbols]
    if insufficient:
        raise ValueError(f"Không đủ dữ liệu để tính feature cho mã {insufficient[0]}.")

    latest_by_symbol = {
        symbol: features[features["symbol"] == symbol]
        .sort_values("trading_date")
        .iloc[-1]
        for symbol in normalized
    }
    latest_rows = pd.DataFrame([latest_by_symbol[symbol] for symbol in normalized])

    artifact = load_model_artifact()
    metadata = load_metadata(artifact)
    model = artifact["model"]
    feature_columns = metadata.get(
        "feature_order", artifact.get("feature_columns", FEATURE_COLUMNS)
    )
    x_latest = latest_rows[feature_columns]
    decision_threshold = float(
        artifact.get(
            "decision_threshold", metadata.get("decision_threshold", DECISION_THRESHOLD)
        )
    )

    probability_up = [None] * len(normalized)
    labels = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_latest)
        classes = list(model.classes_)
        if 1 in classes:
            probability_up = [
                float(row[classes.index(1)]) for row in probabilities
            ]
            labels = [
                "UP" if probability >= decision_threshold else "NOT_UP"
                for probability in probability_up
            ]
    if labels is None:
        labels = [
            "UP" if int(value) == 1 else "NOT_UP"
            for value in model.predict(x_latest)
        ]

    horizon = int(
        metadata.get("prediction_horizon", artifact.get("prediction_horizon", 5))
    )
    threshold = float(metadata.get("up_threshold", artifact.get("up_threshold", 0.01)))
    model_name = metadata.get("model_name", artifact.get("model_name", "Unknown"))
    policy_id = metadata.get("policy_id", artifact.get("policy_id", "legacy"))
    baseline_passed = metadata.get(
        "baseline_passed", artifact.get("baseline_passed")
    )
    baseline_warning = metadata.get(
        "baseline_warning", artifact.get("baseline_warning")
    )
    if baseline_passed is False and not baseline_warning:
        baseline_warning = "Final Model chưa vượt baseline always-UP trên TEST."
    if policy_id != EXPERIMENT_POLICY_ID and not baseline_warning:
        baseline_warning = (
            "Model đang dùng không thuộc policy hiện hành."
        )
    global_latest_date = _as_date(clean_df["trading_date"].max())

    results = []
    for index, symbol in enumerate(normalized):
        latest = latest_by_symbol[symbol]
        reference_date = _as_date(latest["trading_date"])
        sessions = _forecast_sessions(reference_date, horizon=horizon)
        forecast_end_date = _as_date(sessions[-1]["date"])
        close = float(latest["close"])
        label = labels[index]
        probability = probability_up[index]

        if label == "UP":
            prediction_label_vi = "Tăng (UP)"
            prediction_short_vi = (
                f"Khả năng giá tăng hơn {threshold * 100:g}% trong {horizon} phiên tiếp theo."
            )
        else:
            prediction_label_vi = "Không đủ điều kiện tăng (NOT_UP)"
            prediction_short_vi = (
                f"Không đủ điều kiện tăng hơn {threshold * 100:g}% trong {horizon} phiên tiếp theo."
            )

        prob_text = f"{probability * 100:.1f}%" if probability is not None else "không hỗ trợ"
        summary_sentence = (
            f"Từ phiên {reference_date.isoformat()}, hệ thống dự báo xu hướng "
            f"{horizon} phiên giao dịch kế tiếp là {prediction_label_vi} "
            f"(điểm lớp UP của model: {prob_text})."
        )
        history = (
            symbol_rows[symbol_rows["symbol"] == symbol]
            .sort_values("trading_date")
            .tail(horizon + 1)
        )

        results.append(
            {
                "symbol": symbol,
                "latest_date": reference_date.isoformat(),
                "reference_date": reference_date.isoformat(),
                "close_at_reference": close,
                "prediction": label,
                "probability_up": probability,
                "decision_threshold": decision_threshold,
                "target_close": close * (1 + threshold),
                "price_history": [
                    {"date": _as_date(row.trading_date).isoformat(), "close": float(row.close)}
                    for row in history.itertuples(index=False)
                ],
                "forecast_sessions": sessions,
                "forecast_start_date": sessions[0]["date"],
                "forecast_end_date": sessions[-1]["date"],
                "is_stale": reference_date < global_latest_date,
                "forecast_window_elapsed": forecast_end_date < date.today(),
                "return_20d": float(latest["return_20d"]),
                "volatility_20d": float(latest["volatility_20d"]),
                "volume_ratio_20": float(latest["volume_ratio_20"]),
                "model_name": model_name,
                "policy_id": policy_id,
                "baseline_passed": baseline_passed,
                "baseline_warning": baseline_warning,
                "horizon": horizon,
                "horizon_sessions": horizon,
                "threshold_percent": int(threshold * 100),
                "summary_sentence": summary_sentence,
                "prediction_label_vi": prediction_label_vi,
                "prediction_short_vi": prediction_short_vi,
            }
        )
    return results


def predict_symbol(symbol: str) -> dict:
    """Wrapper tương thích cho Flask và CLI đơn mã hiện tại."""
    return predict_symbols([symbol])[0]


def _dataset_signature() -> tuple:
    """Chữ ký dữ liệu clean để invalidate cache khi refresh data."""
    if CLEANED_DATA_PATH.exists():
        stat = CLEANED_DATA_PATH.stat()
        return (str(CLEANED_DATA_PATH), stat.st_size, stat.st_mtime_ns)
    if Path(RAW_DATA_PATH).exists():
        stat = Path(RAW_DATA_PATH).stat()
        return (str(RAW_DATA_PATH), stat.st_size, stat.st_mtime_ns)
    return ("missing", 0, 0)


def _model_signature() -> tuple:
    """Chữ ký artifact để invalidate cache khi đổi model đang phục vụ."""
    if FINAL_MODEL_PATH.exists():
        stat = FINAL_MODEL_PATH.stat()
        return (stat.st_size, stat.st_mtime_ns)
    return (0, 0)


@lru_cache(maxsize=1)
def _predict_all_symbols_cached(_dataset_sig: tuple, _model_sig: tuple) -> tuple[dict, ...]:
    """Inference toàn sàn trong một lượt; kết quả cache theo chữ ký data + model.

    Trả tuple (immutable) để lru_cache an toàn; caller tự copy nếu cần đổi.
    """
    clean_df = _load_clean_data().copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str).str.strip().str.upper()

    features, _ = build_features(clean_df)
    if features.empty:
        return tuple()

    latest_rows = (
        features.sort_values("trading_date")
        .groupby("symbol", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )

    artifact = load_model_artifact()
    metadata = load_metadata(artifact)
    model = artifact["model"]
    feature_columns = metadata.get(
        "feature_order", artifact.get("feature_columns", FEATURE_COLUMNS)
    )
    decision_threshold = float(
        artifact.get(
            "decision_threshold", metadata.get("decision_threshold", DECISION_THRESHOLD)
        )
    )
    model_name = metadata.get("model_name", artifact.get("model_name", "Unknown"))
    policy_id = metadata.get("policy_id", artifact.get("policy_id", "legacy"))
    baseline_passed = metadata.get("baseline_passed", artifact.get("baseline_passed"))
    baseline_warning = metadata.get(
        "baseline_warning", artifact.get("baseline_warning")
    )
    if baseline_passed is False and not baseline_warning:
        baseline_warning = "Final Model chưa vượt baseline always-UP trên TEST."
    if policy_id != EXPERIMENT_POLICY_ID and not baseline_warning:
        baseline_warning = "Model đang dùng không thuộc policy hiện hành."

    x_latest = latest_rows[feature_columns]
    probability_up = [None] * len(latest_rows)
    labels = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_latest)
        classes = list(model.classes_)
        if 1 in classes:
            up_index = classes.index(1)
            probability_up = [float(row[up_index]) for row in probabilities]
            labels = [
                "UP" if prob >= decision_threshold else "NOT_UP"
                for prob in probability_up
            ]
    if labels is None:
        labels = [
            "UP" if int(value) == 1 else "NOT_UP"
            for value in model.predict(x_latest)
        ]

    global_latest_date = _as_date(clean_df["trading_date"].max())

    rows = []
    for index, record in enumerate(latest_rows.itertuples(index=False)):
        reference_date = _as_date(record.trading_date)
        rows.append(
            {
                "symbol": str(record.symbol),
                "prediction": labels[index],
                "probability_up": probability_up[index],
                "decision_threshold": decision_threshold,
                "close_at_reference": float(record.close),
                "reference_date": reference_date.isoformat(),
                "return_20d": float(record.return_20d),
                "volatility_20d": float(record.volatility_20d),
                "volume_ratio_20": float(record.volume_ratio_20),
                "is_stale": reference_date < global_latest_date,
                "model_name": model_name,
                "policy_id": policy_id,
                "baseline_passed": baseline_passed,
                "baseline_warning": baseline_warning,
            }
        )
    return tuple(rows)


def predict_all_symbols() -> list[dict]:
    """Inference cho toàn bộ mã trong dataset offline (có cache theo fingerprint)."""
    cached = _predict_all_symbols_cached(_dataset_signature(), _model_signature())
    return [dict(row) for row in cached]
