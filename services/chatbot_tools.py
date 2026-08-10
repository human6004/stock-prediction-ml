"""Fixed, read-only action handlers for the local stock chatbot."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    ELIGIBLE_SYMBOLS_PATH,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_DEFINITIONS,
    PIPELINE_SUMMARY_PATH,
    TUNING_SCORING,
)
from services import experiment_state, prediction_service


PROJECT_TOPICS = {
    "overview",
    "dataset",
    "features",
    "model",
    "evaluation",
    "inference",
    "limitations",
}


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value, digits: int = 10):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _percent(value, digits: int = 2):
    number = _number(value)
    return round(number * 100, digits) if number is not None else None


def _load_scope(metadata: dict) -> tuple[set[str], bool]:
    if "training_symbols" in metadata:
        raw = metadata["training_symbols"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("invalid training_symbols")
        scope = {
            symbol.strip().upper()
            for symbol in raw
            if isinstance(symbol, str) and symbol.strip()
        }
        if len(scope) != len(raw):
            raise ValueError("invalid or duplicate training_symbols")
        expected_count = metadata.get("training_symbol_count")
        if expected_count is not None and (
            type(expected_count) is not int or expected_count != len(scope)
        ):
            raise ValueError("training_symbol_count mismatch")
        return scope, False

    rows = pd.read_csv(ELIGIBLE_SYMBOLS_PATH, usecols=["symbol"])
    scope = {
        str(symbol).strip().upper()
        for symbol in rows["symbol"].dropna()
        if str(symbol).strip()
    }
    if not scope:
        raise ValueError("empty legacy scope")
    return scope, True


def _load_runtime_state(*, include_scope: bool = True) -> dict:
    def unavailable(message: str) -> dict:
        return {"error": {"code": "model_unavailable", "message": message}}

    try:
        if experiment_state.is_pipeline_running():
            return unavailable("Pipeline đang chạy; model tạm thời chưa sẵn sàng.")
        artifact = prediction_service.load_model_artifact()
        if not isinstance(artifact, dict) or artifact.get("model") is None:
            return unavailable("Model chưa sẵn sàng.")
        metadata = prediction_service.load_metadata(artifact)
        if not isinstance(metadata, dict):
            return unavailable("Metadata model chưa sẵn sàng.")
        scope, legacy_scope = _load_scope(metadata) if include_scope else (set(), False)
    except Exception:
        return unavailable("Model hoặc metadata chưa sẵn sàng.")

    warnings = []
    if legacy_scope:
        warnings.append(
            {
                "code": "symbol_scope_unverified",
                "message": "Scope mã dùng fallback từ eligible_symbols.csv của release cũ.",
            }
        )
    if metadata.get("policy_id") != EXPERIMENT_POLICY_ID:
        warnings.append(
            {
                "code": "legacy_policy",
                "message": "Model đang dùng không thuộc policy hiện hành.",
            }
        )
    if metadata.get("baseline_passed") is False:
        warnings.append(
            {
                "code": "baseline_failed",
                "message": str(
                    metadata.get("baseline_warning")
                    or "Final model chưa vượt baseline Always UP trên TEST."
                ),
            }
        )
    return {
        "metadata": metadata,
        "summary": _read_json(PIPELINE_SUMMARY_PATH) or {},
        "scope": scope,
        "warnings": warnings,
        "model_trained_through": metadata.get("train_through_date"),
    }


def _signal_payload(row: dict, metadata: dict) -> dict:
    score = _number(row.get("probability_up"))
    threshold = _number(row.get("decision_threshold"))
    relation = None
    if score is not None and threshold is not None:
        relation = "equal" if score == threshold else "above" if score > threshold else "below"
    return {
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "prediction": row.get("prediction"),
        "up_score_percent": _percent(score),
        "decision_threshold_percent": _percent(threshold),
        "threshold_relation": relation,
        "threshold_gap_percent_points": (
            round(abs(score - threshold) * 100, 2)
            if score is not None and threshold is not None
            else None
        ),
        "close_at_reference": _number(row.get("close_at_reference")),
        "reference_date": row.get("reference_date"),
        "return_20d_percent": _percent(row.get("return_20d")),
        "volatility_20d_percent": _percent(row.get("volatility_20d")),
        "volume_ratio_20": _number(row.get("volume_ratio_20"), 2),
        "is_stale": bool(row.get("is_stale", False)),
        "horizon_sessions": row.get("horizon") or metadata.get("prediction_horizon"),
    }


def _stock_signal(arguments: dict, state: dict) -> dict:
    symbols = arguments["symbols"]
    outside = [symbol for symbol in symbols if symbol not in state["scope"]]
    if outside:
        message = f"Mã {', '.join(outside)} nằm ngoài phạm vi model đang phục vụ."
        return {
            "data": {"unsupported_symbols": outside},
            "sources": [],
            "warnings": [*state["warnings"], {"code": "symbol_out_of_scope", "message": message}],
            "data_as_of": None,
            "model_trained_through": state["model_trained_through"],
            "error": None,
        }

    try:
        rows = prediction_service.predict_symbols(symbols)
    except Exception:
        return {
            "data": {},
            "sources": [],
            "warnings": list(state["warnings"]),
            "data_as_of": None,
            "model_trained_through": state["model_trained_through"],
            "error": {
                "code": "prediction_unavailable",
                "message": "Không thể tạo tín hiệu từ dữ liệu offline lúc này.",
            },
        }

    dates = [row.get("reference_date") for row in rows if row.get("reference_date")]
    same_date = len(dates) == len(rows) and len(set(dates)) == 1
    if len(rows) > 1 and same_date:
        rows.sort(
            key=lambda row: (
                _number(row.get("probability_up")) is None,
                -(_number(row.get("probability_up")) or 0),
                str(row.get("symbol") or "").strip().upper(),
            )
        )
    payload = [_signal_payload(row, state["metadata"]) for row in rows]
    data = {"signals": payload}
    warnings = list(state["warnings"])
    for row in payload:
        if row["is_stale"]:
            warnings.append(
                {
                    "code": "stale_symbol_data",
                    "message": f"Dữ liệu của {row['symbol']} đã cũ ({row.get('reference_date')}).",
                }
            )

    if len(payload) > 1 and not same_date:
        warnings.append(
            {
                "code": "different_reference_dates",
                "message": "Các mã khác ngày tham chiếu nên giữ thứ tự yêu cầu và không so sánh trực tiếp.",
            }
        )

    data_as_of = dates[0] if same_date and dates else None
    return {
        "data": data,
        "sources": [{"kind": "stock_signal", "as_of": data_as_of, "symbols": symbols}],
        "warnings": warnings,
        "data_as_of": data_as_of,
        "model_trained_through": state["model_trained_through"],
        "error": None,
    }


def _stock_ranking(arguments: dict, state: dict) -> dict:
    try:
        raw_rows = prediction_service.predict_all_symbols()
    except Exception:
        raw_rows = None
    if raw_rows is None:
        return {
            "data": {},
            "sources": [],
            "warnings": list(state["warnings"]),
            "data_as_of": None,
            "model_trained_through": state["model_trained_through"],
            "error": {
                "code": "prediction_unavailable",
                "message": "Không thể tạo bảng xếp hạng lúc này.",
            },
        }

    eligible = [
        row
        for row in raw_rows
        if str(row.get("symbol") or "").strip().upper() in state["scope"]
        and _number(row.get("probability_up")) is not None
    ]
    excluded_stale_count = sum(bool(row.get("is_stale", False)) for row in eligible)
    current = [row for row in eligible if not bool(row.get("is_stale", False))]
    if not current:
        return {
            "data": {"excluded_stale_count": excluded_stale_count},
            "sources": [],
            "warnings": list(state["warnings"]),
            "data_as_of": None,
            "model_trained_through": state["model_trained_through"],
            "error": {
                "code": "no_current_signals",
                "message": "Không còn tín hiệu hiện tại để xếp hạng.",
            },
        }

    reverse = arguments["order"] == "highest"
    if reverse:
        current.sort(key=lambda row: (-float(row["probability_up"]), str(row["symbol"])))
    else:
        current.sort(key=lambda row: (float(row["probability_up"]), str(row["symbol"])))
    payload = [
        _signal_payload(row, state["metadata"])
        for row in current[: arguments["top_n"]]
    ]
    dates = [row["reference_date"] for row in payload if row.get("reference_date")]
    data_as_of = max(dates) if dates else None
    warnings = list(state["warnings"])
    if excluded_stale_count:
        warnings.append(
            {
                "code": "stale_symbols_excluded",
                "message": f"Đã loại {excluded_stale_count} mã có dữ liệu cũ khỏi bảng xếp hạng.",
            }
        )
    return {
        "data": {
            "order": arguments["order"],
            "top_n": arguments["top_n"],
            "ranking": payload,
            "excluded_stale_count": excluded_stale_count,
        },
        "sources": [
            {
                "kind": "ranking",
                "as_of": data_as_of,
                "symbols": [row["symbol"] for row in payload],
            }
        ],
        "warnings": warnings,
        "data_as_of": data_as_of,
        "model_trained_through": state["model_trained_through"],
        "error": None,
    }


def _project_info(topic: str, state: dict) -> dict:
    metadata = state.get("metadata") or {}
    summary = state.get("summary") or {}
    warnings = list(state.get("warnings") or [])
    data_as_of = None

    if topic == "overview":
        data = {
            "topic": topic,
            "title": "Hệ thống dự đoán tín hiệu kỹ thuật cổ phiếu HOSE",
            "facts": {
                "serving_mode": "offline_published_artifacts",
                "pipeline_steps": ["fetch", "clean", "features", "train", "publish", "predict"],
                "capabilities": ["stock_signal", "ranking", "project_info"],
            },
        }
        sources = [{"kind": "project_contract", "topic": topic}]
    elif topic == "limitations":
        data = {
            "topic": topic,
            "title": "Giới hạn hệ thống",
            "facts": {
                "realtime_data": False,
                "news": False,
                "fundamentals": False,
                "investment_advice": False,
            },
        }
        sources = [{"kind": "project_contract", "topic": topic}]
    elif topic == "dataset":
        try:
            rows = pd.read_csv(CLEANED_DATA_PATH, usecols=["symbol", "trading_date"])
            rows["symbol"] = rows["symbol"].astype(str).str.strip().str.upper()
            dates = pd.to_datetime(rows["trading_date"], errors="coerce").dropna()
            if dates.empty:
                raise ValueError("empty dates")
        except (OSError, ValueError, pd.errors.ParserError):
            return {
                "data": {},
                "sources": [],
                "warnings": warnings,
                "data_as_of": None,
                "model_trained_through": None,
                "error": {
                    "code": "report_unavailable",
                    "message": "Thông tin dataset offline chưa sẵn sàng.",
                },
            }
        date_min = dates.min().date().isoformat()
        date_max = dates.max().date().isoformat()
        report_date = (summary.get("dataset_report") or {}).get("date_max")
        if report_date and report_date != date_max:
            warnings.append(
                {
                    "code": "data_ahead_of_report",
                    "message": "Ngày dữ liệu clean khác ngày trong pipeline report.",
                }
            )
        data_as_of = date_max
        data = {
            "topic": topic,
            "offline": True,
            "date_min": date_min,
            "date_max": date_max,
            "clean_row_count": int(len(rows)),
            "clean_symbol_count": int(rows["symbol"].nunique()),
            "training_symbol_count": (summary.get("clean_report") or {}).get("eligible_symbols"),
            "feature_count": (summary.get("feature_report") or {}).get("feature_count")
            or len(FEATURE_COLUMNS),
            "split_report": summary.get("split_report")
            if isinstance(summary.get("split_report"), dict)
            else {},
        }
        sources = [{"kind": "dataset_info", "as_of": date_max}]
    elif topic == "features":
        try:
            rows = pd.read_csv(FEATURE_IMPORTANCE_PATH, usecols=["feature", "importance"])
            rows["importance"] = pd.to_numeric(rows["importance"], errors="coerce")
            rows = rows[rows["importance"].map(lambda value: _number(value) is not None)]
            rows = rows.sort_values(["importance", "feature"], ascending=[False, True]).head(10)
        except (OSError, ValueError, pd.errors.ParserError):
            rows = pd.DataFrame()
        if rows.empty:
            return {
                "data": {},
                "sources": [],
                "warnings": warnings,
                "data_as_of": None,
                "model_trained_through": None,
                "error": {
                    "code": "report_unavailable",
                    "message": "Feature importance chưa sẵn sàng.",
                },
            }
        data = {
            "topic": topic,
            "scope": "global_model_importance",
            "feature_order": list(FEATURE_COLUMNS),
            "global_importance": [
                {"feature": str(row.feature), "importance": _number(row.importance, 4)}
                for row in rows.itertuples(index=False)
            ],
        }
        sources = [{"kind": "feature_importance"}]
    elif topic == "model":
        horizon = metadata.get("prediction_horizon")
        threshold_percent = _percent(metadata.get("up_threshold"))
        selection = metadata.get("selection") or summary.get("selection_report") or {}
        data = {
            "topic": topic,
            "model_name": metadata.get("model_name"),
            "target": (
                f"Giá đóng cửa tăng hơn {threshold_percent:g}% sau {horizon} phiên"
                if threshold_percent is not None and horizon is not None
                else None
            ),
            "prediction_horizon": horizon,
            "decision_threshold_percent": _percent(metadata.get("decision_threshold")),
            "train_through_date": metadata.get("train_through_date"),
            "training_symbol_count": metadata.get("training_symbol_count"),
            "best_params": metadata.get("best_params"),
            "candidate_models": [MODEL_DEFINITIONS[key] for key in sorted(MODEL_DEFINITIONS)],
            "selection_metric": TUNING_SCORING,
            "selection_split": selection.get("selection_split"),
        }
        sources = [{"kind": "model_metadata"}]
    elif topic == "evaluation":
        baselines = metadata.get("final_test_baselines")
        baseline_rows = []
        if isinstance(baselines, list):
            for row in baselines:
                if not isinstance(row, dict):
                    continue
                metrics = {
                    key: _percent(value)
                    for key, value in row.items()
                    if key in {"accuracy", "precision_up", "recall_up", "f1_up"}
                    and _number(value) is not None
                }
                baseline_rows.append(
                    {"model_name": row.get("model_name"), "metrics_percent": metrics}
                )
        data = {
            "topic": topic,
            "validation_selection_metrics_percent": {
                key: _percent(value)
                for key, value in (metadata.get("validation_selection_metrics") or {}).items()
                if _number(value) is not None
            },
            "final_test_metrics_percent": {
                key: _percent(value)
                for key, value in (metadata.get("final_test_metrics") or {}).items()
                if _number(value) is not None
            },
            "final_test_baselines": baseline_rows,
            "baseline_passed": metadata.get("baseline_passed"),
            "baseline_warning": metadata.get("baseline_warning"),
        }
        sources = [{"kind": "model_metadata", "section": topic}]
    else:
        data = {
            "topic": topic,
            "flow": ["predict_proba", "score", "decision_threshold", "UP / NOT_UP"],
            "facts": {
                "triggers_training": False,
                "prediction_horizon_sessions": metadata.get("prediction_horizon"),
                "decision_threshold_percent": _percent(metadata.get("decision_threshold")),
                "labels": ["UP", "NOT_UP"],
            },
        }
        sources = [{"kind": "model_metadata", "section": topic}]

    return {
        "data": data,
        "sources": sources,
        "warnings": warnings,
        "data_as_of": data_as_of,
        "model_trained_through": state.get("model_trained_through"),
        "error": None,
    }


def execute_action(action: str, arguments: dict) -> dict:
    if action not in {"STOCK_SIGNAL", "STOCK_RANKING", "PROJECT_INFO"}:
        raise ValueError(f"Unsupported data action: {action}")
    if action == "PROJECT_INFO" and arguments.get("topic") not in PROJECT_TOPICS:
        raise ValueError("Unsupported project topic")

    if action == "PROJECT_INFO":
        topic = arguments["topic"]
        if topic in {"overview", "limitations"}:
            return _project_info(topic, {})
        if topic in {"dataset", "features"}:
            if experiment_state.is_pipeline_running():
                return {
                    "data": {},
                    "sources": [],
                    "warnings": [],
                    "data_as_of": None,
                    "model_trained_through": None,
                    "error": {
                        "code": "report_unavailable",
                        "message": "Pipeline đang ghi report; thông tin tạm thời chưa sẵn sàng.",
                    },
                }
            return _project_info(
                topic,
                {"summary": _read_json(PIPELINE_SUMMARY_PATH) or {}},
            )

    state = _load_runtime_state(include_scope=action != "PROJECT_INFO")
    if state.get("error"):
        return {
            "data": {},
            "sources": [],
            "warnings": [],
            "data_as_of": None,
            "model_trained_through": None,
            "error": state["error"],
        }
    if action == "STOCK_SIGNAL":
        return _stock_signal(arguments, state)
    if action == "STOCK_RANKING":
        return _stock_ranking(arguments, state)
    return _project_info(arguments["topic"], state)
