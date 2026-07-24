"""Grounded, read-only tools exposed to the local stock chatbot."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from config.settings import (
    CLEANED_DATA_PATH,
    CV_GAP_SESSIONS,
    CV_N_SPLITS,
    DATA_PROTOCOL_ID,
    ELIGIBLE_SYMBOLS_PATH,
    EXCLUDED_SYMBOLS_PATH,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_DEFINITIONS,
    PIPELINE_SUMMARY_PATH,
    PREDICTION_HORIZON,
    TEST_END_DATE,
    TRAIN_END_DATE,
    TUNING_SCORING,
    UP_THRESHOLD,
    VALIDATION_END_DATE,
)
from services import experiment_state, prediction_service


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_signals",
            "description": (
                "Đọc tín hiệu offline cho 1-2 mã thuộc model release. "
                "Output là Điểm UP, nhãn, ngày tham chiếu và chỉ báo đã rút gọn; "
                "nếu có 2 mã, server trả sẵn mã cao/thấp và chênh lệch Điểm UP; "
                "error trả về khi input/scope/release không hợp lệ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 2,
                        "description": "Một hoặc hai mã HOSE, ví dụ FPT, VNM.",
                    }
                },
                "required": ["symbols"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ranking",
            "description": (
                "Xếp hạng mã trong đúng symbol scope của serving release theo Điểm UP. "
                "Server loại score lỗi và dữ liệu stale rồi cắt top_n trước khi trả; "
                "error nếu input/release lỗi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "string",
                        "enum": ["highest_up_score", "lowest_up_score"],
                    },
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["order", "top_n"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": (
                "Đọc model, target, policy, threshold, metrics và baseline cùng release; "
                "error hoặc metrics bị chặn nếu release không an toàn."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_info",
            "description": (
                "Đọc snapshot dataset offline, khoảng ngày, số mã và feature; "
                "vẫn dùng được khi model lỗi, error nếu report/dataset đều thiếu."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_info",
            "description": (
                "Đọc feature order và global model importance, không giải thích nguyên nhân "
                "riêng một mã; error nếu report khác release."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "required": ["top_n"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_info",
            "description": (
                "Giải thích contract tĩnh, được whitelist của project HOSE ML theo topic; "
                "không đọc file tùy ý hoặc mô tả model serving hiện tại; error nếu topic "
                "hoặc arguments không hợp lệ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "overview",
                            "target",
                            "features",
                            "data_split",
                            "training",
                            "metrics",
                            "inference",
                            "limitations",
                        ],
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
]


PROJECT_TOPICS = (
    "overview",
    "target",
    "features",
    "data_split",
    "training",
    "metrics",
    "inference",
    "limitations",
)

_METRIC_KEYS = (
    "accuracy",
    "precision_up",
    "recall_up",
    "f1_up",
    "precision_not_up",
    "recall_not_up",
    "f1_not_up",
)


class UnknownToolError(ValueError):
    """Raised when the provider requests a tool outside the allowlist."""


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _summary_model_name(summary: dict | None) -> str | None:
    if not summary:
        return None
    selection = summary.get("selection_report")
    if isinstance(selection, dict) and selection.get("selected_model_name"):
        return str(selection["selected_model_name"])
    rows = summary.get("final_test_evaluation")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("row_type") == "final":
                return row.get("model_name")
    return None


def _identity(document: dict | None, *, summary: bool = False) -> tuple:
    if not document:
        return (None, None, None)
    model_name = _summary_model_name(document) if summary else document.get("model_name")
    return (
        model_name,
        document.get("policy_id"),
        document.get("content_fingerprint"),
    )


def _rounded_number(value, digits: int):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _percent(value):
    number = _rounded_number(value, 10)
    return None if number is None else round(number * 100, 2)


def _percent_point_gap(first, second):
    left = _rounded_number(first, 10)
    right = _rounded_number(second, 10)
    return None if left is None or right is None else round(abs(left - right) * 100, 2)


def _metrics_percent(metrics) -> dict:
    if not isinstance(metrics, dict):
        return {}
    return {
        key: percent
        for key in _METRIC_KEYS
        if key in metrics and (percent := _percent(metrics.get(key))) is not None
    }


def _always_up_f1(metadata: dict, summary: dict | None = None):
    collections = [
        metadata.get("final_test_baselines"),
        (summary or {}).get("final_test_evaluation"),
    ]
    for rows in collections:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if (
                isinstance(row, dict)
                and str(row.get("model_name", "")).casefold().replace("-", " ")
                == "always up"
            ):
                return _rounded_number(row.get("f1_up"), 10)
    return None


def _format_baseline_warning(metadata: dict, summary: dict | None = None) -> str | None:
    if metadata.get("baseline_passed") is not False:
        return None
    final_f1 = _rounded_number(
        (metadata.get("final_test_metrics") or {}).get("f1_up"), 10
    )
    baseline_f1 = _always_up_f1(metadata, summary)
    if final_f1 is None or baseline_f1 is None:
        return "Final Model chưa vượt baseline Always UP trên TEST."
    final_text = f"{final_f1 * 100:.2f}".replace(".", ",")
    baseline_text = f"{baseline_f1 * 100:.2f}".replace(".", ",")
    return (
        "Final Model chưa vượt baseline Always UP trên TEST: "
        f"F1 UP {final_text}% so với {baseline_text}%."
    )


def _load_scope(metadata: dict) -> tuple[set[str], bool, bool]:
    """Return scope, verified flag and validity flag."""
    has_symbols = "training_symbols" in metadata
    has_count = "training_symbol_count" in metadata
    if has_symbols or has_count:
        raw = metadata.get("training_symbols")
        count = metadata.get("training_symbol_count")
        if not isinstance(raw, list) or isinstance(count, bool) or not isinstance(count, int):
            return set(), False, False
        if any(not isinstance(item, str) or not item.strip() for item in raw):
            return set(), False, False
        normalized = [item.strip().upper() for item in raw]
        if not normalized or len(normalized) != len(set(normalized)) or count != len(normalized):
            return set(), False, False
        return set(normalized), True, True

    try:
        rows = pd.read_csv(ELIGIBLE_SYMBOLS_PATH, usecols=["symbol"])
    except (OSError, ValueError, pd.errors.ParserError):
        return set(), False, True
    scope = {
        str(symbol).strip().upper()
        for symbol in rows["symbol"].dropna()
        if str(symbol).strip()
    }
    return scope, False, True


def _release_signature() -> tuple:
    signature = []
    for path in (
        FINAL_MODEL_PATH,
        MODEL_METADATA_PATH,
        PIPELINE_SUMMARY_PATH,
        ELIGIBLE_SYMBOLS_PATH,
    ):
        try:
            stat = path.stat()
            signature.append((stat.st_size, stat.st_mtime_ns))
        except OSError:
            signature.append(None)
    return tuple(signature)


def _release_changed(state: dict) -> bool:
    return (
        experiment_state.is_pipeline_running()
        or state.get("signature") != _release_signature()
    )


def get_release_state() -> dict:
    """Read raw release files without prediction_service's mismatch fallback."""
    signature_before = _release_signature()
    try:
        artifact = joblib.load(FINAL_MODEL_PATH)
        if not isinstance(artifact, dict):
            artifact = None
    except Exception:  # trusted local artifact; details never leave this module
        artifact = None
    metadata = _read_json(MODEL_METADATA_PATH)
    summary = _read_json(PIPELINE_SUMMARY_PATH)
    signature_after = _release_signature()
    updating = experiment_state.is_pipeline_running()

    status = "missing"
    scope: set[str] = set()
    scope_verified = False
    if artifact is not None and metadata is not None and summary is not None:
        identities = (
            _identity(artifact),
            _identity(metadata),
            _identity(summary, summary=True),
        )
        identity_complete = all(all(value not in (None, "") for value in item) for item in identities)
        if not identity_complete or len(set(identities)) != 1:
            status = "inconsistent"
        else:
            scope, scope_verified, scope_valid = _load_scope(metadata)
            if (
                not scope_valid
                or (
                    not scope_verified
                    and metadata.get("policy_id") == EXPERIMENT_POLICY_ID
                )
            ):
                status = "inconsistent"
            else:
                status = (
                    "current"
                    if metadata.get("policy_id") == EXPERIMENT_POLICY_ID
                    else "legacy"
                )
    if updating or signature_before != signature_after:
        status = "inconsistent"

    warnings = []
    if status == "legacy":
        warnings.append(
            {
                "code": "legacy_policy",
                "message": "Model release đang dùng policy cũ, không phải policy hiện hành.",
            }
        )
    if status in {"current", "legacy"} and metadata.get("baseline_passed") is False:
        warnings.append(
            {
                "code": "baseline_failed",
                "message": _format_baseline_warning(metadata, summary),
            }
        )
    if status in {"current", "legacy"} and not scope_verified:
        warnings.append(
            {
                "code": "symbol_scope_unverified",
                "message": (
                    "Artifact legacy chưa có training_symbols; symbol scope đang tạm lấy "
                    "từ eligible_symbols.csv."
                ),
            }
        )
    if status == "inconsistent" and updating:
        warnings.append(
            {
                "code": "release_updating",
                "message": "Official pipeline đang cập nhật release; dữ liệu model tạm bị chặn.",
            }
        )
    elif status == "inconsistent":
        warnings.append(
            {
                "code": "release_inconsistent",
                "message": "Model, metadata và report không cùng một release.",
            }
        )
    if status == "missing":
        warnings.append(
            {"code": "model_missing", "message": "Model release chưa sẵn sàng."}
        )

    return {
        "status": status,
        "artifact": artifact or {},
        "metadata": metadata or {},
        "summary": summary or {},
        "scope": scope,
        "scope_verified": scope_verified,
        "signature": signature_after,
        "warnings": warnings,
    }


def _public_release(state: dict) -> dict:
    metadata = state.get("metadata", {})
    artifact = state.get("artifact", {})
    primary = artifact if state.get("status") == "inconsistent" else metadata or artifact
    return {
        "status": state["status"],
        "policy_id": primary.get("policy_id"),
        "content_fingerprint": primary.get("content_fingerprint"),
    }


def _envelope(
    state: dict,
    *,
    ok: bool,
    data: dict | None = None,
    source: dict | None = None,
    as_of: str | None = None,
    error: dict | None = None,
) -> dict:
    return {
        "ok": ok,
        "data": data or {},
        "source": source,
        "as_of": as_of,
        "release": _public_release(state),
        "warnings": list(state.get("warnings", [])),
        "error": error,
    }


def _error(state: dict, code: str, message: str, *, data: dict | None = None) -> dict:
    return _envelope(
        state,
        ok=False,
        data=data,
        error={"code": code, "message": message},
    )


def _validate_keys(arguments, *, allowed: set[str], required: set[str]) -> str | None:
    if not isinstance(arguments, dict):
        return "Arguments phải là JSON object."
    keys = set(arguments)
    if keys - allowed:
        return "Arguments chứa field không được hỗ trợ."
    if required - keys:
        return "Arguments thiếu field bắt buộc."
    return None


def _blocked_error(state: dict) -> dict | None:
    if state["status"] == "inconsistent":
        return _error(
            state,
            "release_inconsistent",
            "Tín hiệu bị chặn vì model và report không cùng release.",
        )
    if state["status"] == "missing":
        return _error(state, "model_missing", "Model release chưa sẵn sàng.")
    if not state["scope"]:
        return _error(
            state,
            "model_missing",
            "Chưa xác định được symbol scope của model release.",
        )
    return None


def _signal_payload(row: dict, metadata: dict) -> dict:
    raw_up_score = _rounded_number(row.get("probability_up"), 10)
    raw_threshold = _rounded_number(row.get("decision_threshold"), 10)
    up_score = _percent(raw_up_score)
    threshold = _percent(raw_threshold)
    threshold_gap = _percent_point_gap(
        raw_up_score, raw_threshold
    )
    threshold_relation = None
    if raw_up_score is not None and raw_threshold is not None:
        if raw_up_score == raw_threshold:
            threshold_relation = "equal"
        elif raw_up_score > raw_threshold:
            threshold_relation = "above"
        else:
            threshold_relation = "below"
    return {
        "symbol": str(row.get("symbol", "")).upper(),
        "prediction": row.get("prediction"),
        "up_score_percent": up_score,
        "decision_threshold_percent": threshold,
        "threshold_relation": threshold_relation,
        "threshold_gap_percent_points": threshold_gap,
        "close_at_reference": row.get("close_at_reference"),
        "price_unit": "source_native",
        "reference_date": row.get("reference_date"),
        "return_20d_percent": _percent(row.get("return_20d")),
        "volatility_20d_percent": _percent(row.get("volatility_20d")),
        "volume_ratio_20": _rounded_number(row.get("volume_ratio_20"), 2),
        "is_stale": bool(row.get("is_stale", False)),
        "horizon_sessions": row.get("horizon")
        or metadata.get("prediction_horizon"),
    }


def get_stock_signals(arguments: dict) -> dict:
    state = get_release_state()
    blocked = _blocked_error(state)
    if blocked:
        return blocked
    invalid = _validate_keys(arguments, allowed={"symbols"}, required={"symbols"})
    symbols = arguments.get("symbols") if isinstance(arguments, dict) else None
    if invalid or not isinstance(symbols, list) or not 1 <= len(symbols) <= 2:
        return _error(state, "invalid_arguments", invalid or "symbols phải có 1-2 mã.")
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
        return _error(state, "invalid_arguments", "Mỗi symbol phải là chuỗi không rỗng.")
    normalized = [symbol.strip().upper() for symbol in symbols]
    if len(normalized) != len(set(normalized)):
        return _error(state, "invalid_arguments", "Các mã cổ phiếu không được trùng nhau.")
    outside = [symbol for symbol in normalized if symbol not in state["scope"]]
    if outside:
        return _error(
            state,
            "symbol_out_of_scope",
            f"Mã {outside[0]} không thuộc symbol scope của model release.",
        )
    try:
        rows = prediction_service.predict_symbols(normalized)
    except Exception:
        return _error(
            state,
            "prediction_unavailable",
            "Không thể tạo tín hiệu từ dữ liệu offline lúc này.",
        )
    if _release_changed(state):
        return _error(
            state,
            "release_inconsistent",
            "Model release thay đổi trong lúc tạo tín hiệu. Vui lòng hỏi lại.",
        )
    payload = [_signal_payload(row, state["metadata"]) for row in rows]
    dates = [row.get("reference_date") for row in payload if row.get("reference_date")]
    same_reference_date = (
        len(dates) == len(payload) and len(set(dates)) == 1
    )
    as_of = dates[0] if same_reference_date else None
    data = {"signal_count": len(payload), "signals": payload}
    extra_warnings = [
        {
            "code": "stale_symbol_data",
            "message": (
                f"Dữ liệu của {row['symbol']} đã cũ, ngày tham chiếu "
                f"{row.get('reference_date') or 'không xác định'}."
            ),
        }
        for row in payload
        if row["is_stale"]
    ]
    if len(payload) == 2 and not same_reference_date:
        data["comparison"] = {
            "available": False,
            "reason": "different_reference_dates",
        }
        extra_warnings.append(
            {
                "code": "different_reference_dates",
                "message": (
                    "Hai mã không cùng ngày tham chiếu nên hệ thống không tính "
                    "chênh lệch Điểm UP."
                ),
            }
        )
    elif len(payload) == 2 and all(
        row["up_score_percent"] is not None for row in payload
    ):
        first, second = payload
        first_score = _rounded_number(rows[0].get("probability_up"), 10)
        second_score = _rounded_number(rows[1].get("probability_up"), 10)
        gap = _percent_point_gap(first_score, second_score)
        same_score = first_score == second_score
        higher, lower = (
            (first, second) if first_score > second_score else (second, first)
        )
        data["comparison"] = {
            "higher_up_score_symbol": None if same_score else higher["symbol"],
            "lower_up_score_symbol": None if same_score else lower["symbol"],
            "same_up_score": same_score,
            "up_score_gap_percent_points": gap,
        }
    result = _envelope(
        state,
        ok=True,
        data=data,
        source={"kind": "stock_signal", "as_of": as_of, "symbols": normalized},
        as_of=as_of,
    )
    result["warnings"].extend(extra_warnings)
    return result


def get_ranking(arguments: dict) -> dict:
    state = get_release_state()
    blocked = _blocked_error(state)
    if blocked:
        return blocked
    invalid = _validate_keys(
        arguments, allowed={"order", "top_n"}, required={"order", "top_n"}
    )
    if invalid:
        return _error(state, "invalid_arguments", invalid)
    order = arguments.get("order")
    top_n = arguments.get("top_n")
    if order not in {"highest_up_score", "lowest_up_score"}:
        return _error(state, "invalid_arguments", "order không hợp lệ.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 10:
        return _error(state, "invalid_arguments", "top_n phải là số nguyên từ 1 đến 10.")
    try:
        raw_rows = prediction_service.predict_all_symbols()
    except Exception:
        return _error(
            state,
            "prediction_unavailable",
            "Không thể tạo bảng xếp hạng từ dữ liệu offline lúc này.",
        )
    if _release_changed(state):
        return _error(
            state,
            "release_inconsistent",
            "Model release thay đổi trong lúc xếp hạng. Vui lòng hỏi lại.",
        )
    eligible_rows = [
        row
        for row in raw_rows
        if str(row.get("symbol", "")).upper() in state["scope"]
        and _rounded_number(row.get("probability_up"), 10) is not None
    ]
    excluded_stale_count = sum(bool(row.get("is_stale", False)) for row in eligible_rows)
    rows = [row for row in eligible_rows if not bool(row.get("is_stale", False))]
    if not rows:
        return _error(
            state,
            "no_current_signals",
            "Không còn tín hiệu cùng snapshot dữ liệu hiện tại để xếp hạng.",
            data={"excluded_stale_count": excluded_stale_count, "data_as_of": None},
        )
    if order == "highest_up_score":
        rows.sort(key=lambda row: (-float(row["probability_up"]), str(row["symbol"])))
    else:
        rows.sort(key=lambda row: (float(row["probability_up"]), str(row["symbol"])))
    payload = [_signal_payload(row, state["metadata"]) for row in rows[:top_n]]
    dates = [row.get("reference_date") for row in payload if row.get("reference_date")]
    as_of = max(dates) if dates else None
    return _envelope(
        state,
        ok=True,
        data={
            "order": order,
            "top_n": top_n,
            "ranking": payload,
            "excluded_stale_count": excluded_stale_count,
            "data_as_of": as_of,
        },
        source={
            "kind": "ranking",
            "as_of": as_of,
            "symbols": [row["symbol"] for row in payload],
        },
        as_of=as_of,
    )


def get_model_info(arguments: dict) -> dict:
    state = get_release_state()
    invalid = _validate_keys(arguments, allowed=set(), required=set())
    if invalid:
        return _error(state, "invalid_arguments", invalid)
    if state["status"] == "missing":
        return _error(state, "model_missing", "Model release chưa sẵn sàng.")
    consistent = state["status"] in {"current", "legacy"}
    metadata = state["metadata"] if consistent else state["artifact"]
    summary = state["summary"] if consistent else {}
    split_report = summary.get("split_report", {})
    up_threshold = metadata.get("up_threshold")
    data = {
        "model_name": metadata.get("model_name"),
        "target": (
            f"Giá đóng cửa tăng hơn {_percent(up_threshold):g}% sau "
            f"{metadata.get('prediction_horizon')} phiên"
            if up_threshold is not None
            else None
        ),
        "prediction_horizon": metadata.get("prediction_horizon"),
        "up_threshold_percent": _percent(up_threshold),
        "decision_threshold_percent": _percent(metadata.get("decision_threshold")),
        "policy_id": metadata.get("policy_id"),
        "content_fingerprint": metadata.get("content_fingerprint"),
        "trained_at": metadata.get("trained_at"),
        "train_through_date": metadata.get("train_through_date"),
        "train_end_date": metadata.get("train_end_date")
        or split_report.get("train_end_date"),
        "validation_end_date": metadata.get("validation_end_date")
        or split_report.get("validation_end_date"),
        "test_end_date": metadata.get("test_end_date")
        or split_report.get("test_end_date"),
        "training_symbol_count": len(state["scope"]) or None if consistent else None,
    }
    if consistent:
        baseline_f1 = _always_up_f1(metadata, summary)
        data.update(
            baseline_passed=metadata.get("baseline_passed"),
            baseline_warning=_format_baseline_warning(metadata, summary),
            always_up_baseline_f1_up_percent=_percent(baseline_f1),
            validation_selection_metrics_percent=_metrics_percent(
                metadata.get("validation_selection_metrics", {})
            ),
            final_test_metrics_percent=_metrics_percent(
                metadata.get("final_test_metrics", {})
            ),
        )
    return _envelope(
        state,
        ok=True,
        data=data,
        source={"kind": "model_metadata"},
        as_of=metadata.get("train_through_date"),
    )


def get_dataset_info(arguments: dict) -> dict:
    state = get_release_state()
    invalid = _validate_keys(arguments, allowed=set(), required=set())
    if invalid:
        return _error(state, "invalid_arguments", invalid)
    summary = state["summary"] or _read_json(PIPELINE_SUMMARY_PATH) or {}
    dataset_report = summary.get("dataset_report", {})
    clean_report = summary.get("clean_report", {})
    feature_report = summary.get("feature_report", {})
    data_as_of = dataset_report.get("date_max")
    date_min = dataset_report.get("date_min")
    clean_count = clean_report.get("symbols_after_cleaning")
    if data_as_of is None or clean_count is None:
        try:
            clean = pd.read_csv(CLEANED_DATA_PATH, usecols=["symbol", "trading_date"])
            data_as_of = data_as_of or str(clean["trading_date"].max())
            date_min = date_min or str(clean["trading_date"].min())
            clean_count = clean_count or int(clean["symbol"].nunique())
        except (OSError, ValueError, pd.errors.ParserError):
            return _error(
                state,
                "report_unavailable",
                "Thông tin dataset offline chưa sẵn sàng.",
            )
    metadata = state["metadata"] if state["status"] in {"current", "legacy"} else {}
    excluded_count = clean_report.get("excluded_symbols")
    if excluded_count is None:
        try:
            excluded = pd.read_csv(EXCLUDED_SYMBOLS_PATH, usecols=["symbol"])
            excluded_count = int(excluded["symbol"].nunique())
        except (OSError, ValueError, pd.errors.ParserError):
            excluded_count = None
    data = {
        "offline": True,
        "date_min": date_min,
        "date_max": data_as_of,
        "data_as_of": data_as_of,
        "model_trained_through": metadata.get("train_through_date"),
        "clean_symbol_count": int(clean_count),
        "training_symbol_count": len(state["scope"]) or None,
        "excluded_symbol_count": excluded_count,
        "feature_count": feature_report.get("feature_count") or len(FEATURE_COLUMNS),
    }
    return _envelope(
        state,
        ok=True,
        data=data,
        source={"kind": "dataset_info", "as_of": data_as_of},
        as_of=data_as_of,
    )


def get_feature_info(arguments: dict) -> dict:
    state = get_release_state()
    invalid = _validate_keys(arguments, allowed={"top_n"}, required={"top_n"})
    top_n = arguments.get("top_n") if isinstance(arguments, dict) else None
    if invalid or isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        return _error(
            state,
            "invalid_arguments",
            invalid or "top_n phải là số nguyên từ 1 đến 20.",
        )
    if state["status"] == "inconsistent":
        return _error(
            state,
            "release_inconsistent",
            "Feature importance bị chặn vì report không cùng model release.",
        )
    if state["status"] == "missing":
        return _error(state, "model_missing", "Model release chưa sẵn sàng.")
    if state["summary"].get("feature_importance_written") is not True:
        return _error(
            state,
            "report_unavailable",
            "Feature importance chưa được publish cho release này.",
        )
    try:
        rows = pd.read_csv(FEATURE_IMPORTANCE_PATH, usecols=["feature", "importance"])
        rows["importance"] = pd.to_numeric(rows["importance"], errors="raise")
    except (OSError, ValueError, pd.errors.ParserError):
        return _error(
            state,
            "report_unavailable",
            "Feature importance chưa sẵn sàng.",
        )
    if _release_changed(state):
        return _error(
            state,
            "release_inconsistent",
            "Feature importance thay đổi trong lúc đọc release. Vui lòng hỏi lại.",
        )
    rows = rows.sort_values(["importance", "feature"], ascending=[False, True]).head(top_n)
    importance = [
        {
            "feature": str(row.feature),
            "importance": _rounded_number(row.importance, 4),
        }
        for row in rows.itertuples(index=False)
    ]
    metadata = state["metadata"]
    return _envelope(
        state,
        ok=True,
        data={
            "scope": "global_model_importance",
            "feature_order": metadata.get("feature_order", FEATURE_COLUMNS),
            "global_importance": importance,
        },
        source={"kind": "feature_importance"},
        as_of=metadata.get("train_through_date"),
    )


def _project_contract(topic: str) -> dict:
    contracts = {
        "overview": {
            "title": "Tổng quan hệ thống HOSE ML",
            "facts": {
                "pipeline_steps": [
                    "fetch",
                    "preprocess",
                    "feature_and_label",
                    "train_and_tune",
                    "test",
                    "publish",
                ],
                "data_protocol_id": DATA_PROTOCOL_ID,
                "serving_mode": "offline_published_artifacts",
            },
            "notes": [
                "Chatbot chỉ đọc artifact đã publish và không tự chạy pipeline."
            ],
        },
        "target": {
            "title": "Target dự báo",
            "facts": {
                "positive_label": "UP",
                "negative_label": "NOT_UP",
                "prediction_horizon_sessions": PREDICTION_HORIZON,
                "up_threshold_percent": _percent(UP_THRESHOLD),
                "target_calendar": "common_market_exact_session",
            },
            "notes": [
                "UP nghĩa là giá đóng cửa vượt ngưỡng sau đúng số phiên dự báo.",
                "NOT_UP không đồng nghĩa chắc chắn giá sẽ giảm.",
            ],
        },
        "features": {
            "title": "Đặc trưng đầu vào",
            "facts": {
                "feature_count": len(FEATURE_COLUMNS),
                "feature_order": list(FEATURE_COLUMNS),
                "return_windows_sessions": [1, 3, 5, 10, 20],
                "moving_average_windows_sessions": [5, 20, 50],
                "volatility_windows_sessions": [5, 20],
                "rsi_period_sessions": 14,
                "high_low_window_sessions": 20,
            },
            "notes": [
                "Feature importance là mức quan trọng toàn cục, không phải nguyên nhân riêng của một mã."
            ],
        },
        "data_split": {
            "title": "Chia dữ liệu TRAIN, VALIDATION và TEST",
            "facts": {
                "strategy": "rolling_walk_forward_with_gap",
                "fallback_train_end_date": TRAIN_END_DATE,
                "fallback_validation_end_date": VALIDATION_END_DATE,
                "fallback_test_end_date": TEST_END_DATE,
                "cv_gap_sessions": CV_GAP_SESSIONS,
            },
            "notes": [
                "Các mốc runtime được suy ra theo dữ liệu mới nhất; đây là mốc fallback trong cấu hình."
            ],
        },
        "training": {
            "title": "Huấn luyện và chọn model",
            "facts": {
                "candidate_models": [
                    MODEL_DEFINITIONS[key] for key in sorted(MODEL_DEFINITIONS)
                ],
                "cv_n_splits": CV_N_SPLITS,
                "cv_gap_sessions": CV_GAP_SESSIONS,
                "selection_metric": TUNING_SCORING,
                "final_refit_scope": "train_plus_validation",
            },
            "notes": [
                "TEST chỉ dùng để đánh giá final model sau khi chọn và refit."
            ],
        },
        "metrics": {
            "title": "Chỉ số đánh giá",
            "facts": {
                "reported_metrics": list(_METRIC_KEYS),
                "primary_selection_metric": TUNING_SCORING,
                "test_baselines": ["Always UP", "Always NOT_UP"],
            },
            "notes": [
                "Metric VALIDATION dùng để chọn model; TEST dùng để đánh giá cuối."
            ],
        },
        "inference": {
            "title": "Luồng dự báo offline",
            "facts": {
                "reads_published_artifacts": True,
                "triggers_training": False,
                "symbol_scope_source": "training_symbols_in_model_metadata",
                "output_labels": ["UP", "NOT_UP"],
            },
            "notes": [
                "Điểm UP là score của model, không phải xác suất chắc chắn."
            ],
        },
        "limitations": {
            "title": "Giới hạn hệ thống",
            "facts": {
                "realtime_data": False,
                "news": False,
                "fundamentals": False,
                "investment_advice": False,
            },
            "notes": [
                "Kết quả dựa trên dữ liệu offline và không phải khuyến nghị mua hoặc bán.",
                "Release cũ hoặc dữ liệu stale phải được đọc cùng cảnh báo tương ứng.",
            ],
        },
    }
    return contracts[topic]


def get_project_info(arguments: dict) -> dict:
    state = {"status": None, "metadata": {}, "artifact": {}, "warnings": []}
    invalid = _validate_keys(arguments, allowed={"topic"}, required={"topic"})
    topic = arguments.get("topic") if isinstance(arguments, dict) else None
    if invalid or topic not in PROJECT_TOPICS:
        return _error(
            state,
            "invalid_arguments",
            invalid or "topic không thuộc danh sách được hỗ trợ.",
        )
    contract = _project_contract(topic)
    return _envelope(
        state,
        ok=True,
        data={"topic": topic, **contract},
        source={"kind": "project_contract", "topic": topic},
    )


_HANDLERS = {
    "get_stock_signals": get_stock_signals,
    "get_ranking": get_ranking,
    "get_model_info": get_model_info,
    "get_dataset_info": get_dataset_info,
    "get_feature_info": get_feature_info,
    "get_project_info": get_project_info,
}


def execute_tool(name: str, arguments: dict) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        raise UnknownToolError(name)
    return handler(arguments)
