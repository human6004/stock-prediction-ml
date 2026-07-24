"""Flask web layer: nhận request, gọi service và render giao diện.

Thuật toán ML không nằm ở đây. Route prediction gọi prediction_service; route
tuning gọi tuning_lab; official pipeline được chạy qua scripts/run_pipeline.py.
"""

import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    CONFUSION_MATRIX_PNG_PATH,
    FINAL_MODEL_EVALUATION_PATH,
    FEATURE_DATA_PATH,
    FETCH_REPORT_PATH,
    LAST_FETCH_RUN_LOG,
    LAST_PIPELINE_RUN_LOG,
    MODEL_COMPARISON_PATH,
    MODEL_KEY,
    MODEL_METADATA_PATH,
    EXPERIMENT_POLICY_ID,
    PREDICTION_HORIZON,
    REPORTS_DIR,
    TUNING_RESULTS_PATH,
    UP_THRESHOLD,
)
from services.experiment_state import (  # noqa: E402
    compute_dataset_fingerprint,
    compute_experiment_fingerprint,
    ensure_experiments_dir,
    find_run,
    get_tuning_progress,
    is_config_complete,
    is_fetch_running,
    is_pipeline_running,
    has_evaluated_snapshot,
    mark_best,
    read_fetch_log_tail,
    read_history,
    read_manual_config,
    read_pipeline_log_tail,
    save_selected_model,
    write_fetch_lock,
    write_pipeline_log,
)
from services.prediction_service import (  # noqa: E402
    load_metadata,
    predict_all_symbols,
    predict_symbol,
    predict_symbols,
)
from services import chatbot_service  # noqa: E402
from services.tuning_lab import (  # noqa: E402
    get_param_defaults,
    get_param_schema,
    get_tuning_job_state,
    is_tuning_job_running,
    start_tuning_job,
    validate_params,
)

app = Flask(__name__)

REFRESH_STEPS = [
    {"id": 1, "label": "Fetch OHLCV", "desc": "Tải giá & volume từ vnstock"},
    {"id": 2, "label": "Preprocess", "desc": "Làm sạch hose_stock_raw.csv"},
    {"id": 3, "label": "Build features", "desc": "Tạo ml_dataset.csv + fingerprint mới"},
]


def _infer_refresh_progress(log: str, running: bool) -> dict:
    log = log or ""
    current = 0
    completed = "Refresh data completed." in log
    if completed:
        current = 3
    elif "[3/3]" in log:
        current = 3
    elif "[2/3]" in log:
        current = 2
    elif "[1/3]" in log:
        current = 1

    steps = []
    for step in REFRESH_STEPS:
        if completed or step["id"] < current:
            state = "done"
        elif step["id"] == current and running:
            state = "active"
        elif step["id"] == current and not running and current > 0:
            state = "error"
        else:
            state = "pending"
        steps.append({**step, "state": state})

    if running:
        status = "running"
    elif completed:
        status = "completed"
    elif current > 0:
        status = "failed"
    else:
        status = "idle"

    if completed:
        percent = 100
    elif current == 0:
        percent = 0
    elif current == 1:
        percent = 33
    elif current == 2:
        percent = 66
    else:
        percent = 90

    return {
        "steps": steps,
        "current": current,
        "completed": completed,
        "status": status,
        "percent": percent,
    }


HISTORY_MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

HISTORY_PAGE_SIZE = 50


def get_dataset_meta() -> dict:
    if not FEATURE_DATA_PATH.exists():
        return {
            "dataset_max_date": None,
            "symbol_count": 0,
            "symbols": tuple(),
        }
    stat = FEATURE_DATA_PATH.stat()
    return _get_dataset_meta((stat.st_size, stat.st_mtime_ns))


@lru_cache(maxsize=1)
def _get_dataset_meta(_signature: tuple[int, int]) -> dict:
    cols = pd.read_csv(FEATURE_DATA_PATH, usecols=["symbol", "trading_date"])
    return {
        "dataset_max_date": str(cols["trading_date"].max()),
        "symbol_count": int(cols["symbol"].nunique()),
        "symbols": tuple(sorted(cols["symbol"].astype(str).unique())),
    }

def load_selected_model_from_report() -> str:
    if MODEL_METADATA_PATH.exists():
        metadata = load_metadata()
        if metadata.get("model_name"):
            return str(metadata["model_name"])
    if not MODEL_COMPARISON_PATH.exists():
        return ""
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    selected = comparison[comparison["selected"].astype(str).str.lower() == "true"]
    if selected.empty:
        return ""
    return str(selected.iloc[0]["model_name"])


def _read_report_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = pd.read_csv(path)
    if "cv_f1_up" in rows.columns:
        rows["cv_f1_up"] = rows["cv_f1_up"].where(rows["cv_f1_up"].notna(), None)
    if "selected" in rows.columns:
        rows["status"] = (
            rows["selected"]
            .astype(str)
            .str.lower()
            .map({"true": "Đang dùng"})
            .fillna("")
        )
    return rows.to_dict("records")


def load_evaluation_sections() -> dict:
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        try:
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}
    policy_id = metadata.get("policy_id")
    is_current_policy = policy_id == EXPERIMENT_POLICY_ID
    return {
        "is_current_policy": is_current_policy,
        "policy_id": policy_id or "legacy",
        "validation": _read_report_rows(MODEL_COMPARISON_PATH) if is_current_policy else [],
        "test": _read_report_rows(FINAL_MODEL_EVALUATION_PATH) if is_current_policy else [],
        "legacy": [] if is_current_policy else _read_report_rows(MODEL_COMPARISON_PATH),
        "baseline_warning": metadata.get("baseline_warning") if is_current_policy else None,
        "legacy_warning": None
        if is_current_policy
        else (
            "Artifact/report hiện tại không thuộc policy hiện hành. Hãy chạy CV, tự chọn cấu hình "
            "cho mỗi model rồi chạy official pipeline."
        ),
    }


def _template_context(**extra):
    """Tạo dữ liệu chung cho trang dự báo, tránh lặp ở GET và POST."""
    meta = get_dataset_meta()
    base = {
        "dataset_max_date": meta["dataset_max_date"],
        "symbol_count": meta["symbol_count"],
        "symbols": meta["symbols"],
        "horizon": PREDICTION_HORIZON,
        "threshold_percent": int(UP_THRESHOLD * 100),
        # Mặc định trang dự báo; route khác override qua **extra.
        "active_page": "prediction",
    }
    base.update(extra)
    return base


@app.context_processor
def inject_asset_version():
    """Gắn version theo mtime của app.css để browser tự nạp lại khi CSS đổi."""
    try:
        asset_ver = int((Path(app.static_folder) / "app.css").stat().st_mtime)
    except OSError:
        asset_ver = 0
    return {"asset_ver": asset_ver}


@app.template_filter("date_vi")
def date_vi(value) -> str:
    """Hiển thị ngày ISO theo định dạng quen thuộc trên giao diện."""
    if value is None or value == "":
        return "—"
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return str(value)


@app.route("/", methods=["GET"])
def index():
    selected_model = load_selected_model_from_report()
    return render_template(
        "index.html",
        **_template_context(selected_model=selected_model, active_page="prediction"),
    )


def _chat_error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _validate_chat_payload(payload) -> tuple[str, list[dict]]:
    if (
        not isinstance(payload, dict)
        or "message" not in payload
        or set(payload) - {"message", "history"}
    ):
        raise ValueError
    message = payload["message"]
    if not isinstance(message, str):
        raise ValueError
    message = message.strip()
    if not 1 <= len(message) <= chatbot_service.MAX_MESSAGE_CHARS:
        raise ValueError

    history = payload.get("history", [])
    if (
        not isinstance(history, list)
        or len(history) > chatbot_service.MAX_HISTORY_MESSAGES
        or len(history) % 2
    ):
        raise ValueError
    normalized_history = []
    total_chars = 0
    for index, item in enumerate(history):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item.get("role") != expected_role
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError
        content = item["content"].strip()
        if not 1 <= len(content) <= chatbot_service.MAX_MESSAGE_CHARS:
            raise ValueError
        total_chars += len(content)
        normalized_history.append({"role": expected_role, "content": content})
    if total_chars > chatbot_service.MAX_HISTORY_CHARS:
        raise ValueError
    return message, normalized_history


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template(
        "chat.html",
        **_template_context(
            selected_model=load_selected_model_from_report(), active_page="chat"
        ),
    )


@app.route("/api/chat", methods=["POST"])
def chat_api():
    if request.mimetype != "application/json":
        return _chat_error(
            "invalid_request", "Request phải dùng Content-Type application/json.", 400
        )
    try:
        message, history = _validate_chat_payload(request.get_json(silent=True))
    except ValueError:
        return _chat_error(
            "invalid_request", "Message hoặc history không hợp lệ.", 400
        )
    try:
        return jsonify(chatbot_service.chat(message, history))
    except chatbot_service.ChatbotServiceError as exc:
        return _chat_error(exc.code, exc.message, exc.status)
    except Exception:
        app.logger.error("Unexpected chatbot error")
        return _chat_error(
            "internal_error", "Trợ lý gặp lỗi nội bộ. Vui lòng thử lại sau.", 500
        )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Dự báo một mã từ artifact đang phục vụ.

    POST: submit từ form. GET ?symbol=: mở từ link (vd bảng xếp hạng).
    """
    if request.method == "GET":
        symbol = request.args.get("symbol", "")
        if not symbol:
            return redirect(url_for("index"))
    else:
        symbol = request.form.get("symbol", "")
    try:
        result = predict_symbol(symbol)
        return render_template(
            "index.html",
            **_template_context(
                result=result,
                submitted_symbol=result["symbol"],
                selected_model=result["model_name"],
            ),
        )
    except Exception as exc:
        return (
            render_template(
                "index.html",
                **_template_context(
                    error=str(exc),
                    submitted_symbol=symbol,
                    selected_model=load_selected_model_from_report(),
                ),
            ),
            400,
        )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """So sánh tín hiệu của đúng hai mã bằng một batch inference."""
    symbol_a = request.form.get("symbol_a", "")
    symbol_b = request.form.get("symbol_b", "")
    context = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "active_page": "compare",
    }
    if request.method == "GET":
        context["selected_model"] = load_selected_model_from_report()
        return render_template("compare.html", **_template_context(**context))

    normalized_a = symbol_a.strip().upper()
    normalized_b = symbol_b.strip().upper()
    if not normalized_a or not normalized_b:
        context["error"] = "Vui lòng nhập đủ hai mã cổ phiếu."
        return render_template("compare.html", **_template_context(**context)), 400
    if normalized_a == normalized_b:
        context["error"] = "Hai mã so sánh phải khác nhau."
        return render_template("compare.html", **_template_context(**context)), 400

    try:
        results = predict_symbols([normalized_a, normalized_b])
        context.update(
            results=results,
            symbol_a=normalized_a,
            symbol_b=normalized_b,
            selected_model=results[0]["model_name"],
        )
        return render_template("compare.html", **_template_context(**context))
    except Exception as exc:
        context["error"] = str(exc)
        return render_template("compare.html", **_template_context(**context)), 400


@app.route("/screener", methods=["GET"])
def screener():
    """Bảng xếp hạng tín hiệu toàn sàn: chia nhóm UP / NOT_UP, sort điểm UP."""
    context = {
        "selected_model": load_selected_model_from_report(),
        "active_page": "screener",
    }
    try:
        rows = predict_all_symbols()
    except Exception as exc:  # noqa: BLE001
        context["error"] = str(exc)
        return render_template("screener.html", **_template_context(**context)), 400

    def sort_key(row):
        prob = row.get("probability_up")
        # Mã không có điểm (model không hỗ trợ predict_proba) đẩy xuống cuối.
        return (prob is not None, prob if prob is not None else 0.0)

    up_rows = sorted(
        (row for row in rows if row.get("prediction") == "UP"),
        key=sort_key,
        reverse=True,
    )
    not_up_rows = sorted(
        (row for row in rows if row.get("prediction") != "UP"),
        key=sort_key,
        reverse=True,
    )
    baseline_warning = rows[0]["baseline_warning"] if rows else None
    # Một bảng thống nhất: UP trước (đã sort điểm giảm dần), rồi NOT_UP.
    all_rows = up_rows + not_up_rows
    context.update(
        all_rows=all_rows,
        up_count=len(up_rows),
        not_up_count=len(not_up_rows),
        total_count=len(rows),
        baseline_warning=baseline_warning,
        selected_model=rows[0]["model_name"] if rows else context["selected_model"],
    )
    return render_template("screener.html", **_template_context(**context))


@app.route("/reports/confusion_matrix.png", methods=["GET"])
def confusion_matrix_image():
    if not CONFUSION_MATRIX_PNG_PATH.exists():
        return "Confusion matrix not found. Run pipeline first.", 404
    return send_from_directory(REPORTS_DIR, "confusion_matrix.png")


@app.route("/evaluation", methods=["GET"])
def evaluation():
    sections = load_evaluation_sections()
    selected_model = load_selected_model_from_report()
    return render_template(
        "evaluation.html",
        sections=sections,
        baseline_warning=sections["baseline_warning"],
        selected_model=selected_model,
        dataset_max_date=get_dataset_meta()["dataset_max_date"],
        active_page="evaluation",
    )


def _valid_history_model(model_key: str | None) -> str | None:
    if model_key in HISTORY_MODEL_LABELS:
        return model_key
    return None


def _default_active_history_model(cfg: dict) -> str:
    selected_models = cfg.get("selected_models") or {}
    selected_at_by_model: list[tuple[str, str]] = []
    for model_key, selected in selected_models.items():
        if model_key not in HISTORY_MODEL_LABELS or not isinstance(selected, dict):
            continue
        selected_at_by_model.append((str(selected.get("selected_at") or ""), model_key))

    if selected_at_by_model:
        return max(selected_at_by_model)[1]
    return "logistic_regression"


def _parse_history_query(args, model_key: str, schema: dict) -> dict:
    """Validate and normalize GET filters for one history model."""
    warnings: list[str] = []
    model_schema = schema.get(model_key, {})

    def parse_number(key: str, *, integer: bool = False):
        raw = args.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = float(raw)
            if not math.isfinite(value) or (integer and not value.is_integer()):
                raise ValueError
            return int(value) if integer else value
        except (TypeError, ValueError, OverflowError):
            warnings.append(f"Bỏ qua {key} không hợp lệ.")
            return None

    def normalize_range(name: str, minimum, maximum):
        if minimum is not None and maximum is not None and minimum > maximum:
            warnings.append(f"Bỏ qua khoảng {name}: giá trị Từ lớn hơn Đến.")
            return None, None
        return minimum, maximum

    dataset = args.get("dataset", "current")
    if dataset not in {"current", "all"}:
        warnings.append("Dataset không hợp lệ; dùng dataset hiện tại.")
        dataset = "current"

    status = args.get("status", "ok")
    if status not in {"ok", "error", "all"}:
        warnings.append("Trạng thái không hợp lệ; dùng trạng thái ok.")
        status = "ok"

    f1_min, f1_max = normalize_range(
        "CV F1_UP",
        parse_number("f1_min"),
        parse_number("f1_max"),
    )

    param_filters: dict[str, dict] = {}
    sortable_params: set[str] = set()
    numeric_types = {"float", "int", "int_or_none", "str_or_float"}
    for field, spec in model_schema.items():
        field_type = spec.get("type")
        if field_type == "choice":
            choice = str(args.get(f"p_{field}", "")).strip()
            if choice and choice not in spec.get("choices", []):
                warnings.append(f"Bỏ qua {field} không hợp lệ.")
                choice = ""
            param_filters[field] = {"choice": choice}
            continue
        if field_type not in numeric_types:
            continue

        sortable_params.add(field)
        integer = field_type in {"int", "int_or_none"}
        minimum, maximum = normalize_range(
            field,
            parse_number(f"p_{field}_min", integer=integer),
            parse_number(f"p_{field}_max", integer=integer),
        )
        item = {"min": minimum, "max": maximum}

        if field_type == "int_or_none":
            kind = args.get(f"p_{field}_kind", "all")
            if kind not in {"all", "numeric", "none"}:
                warnings.append(f"Kiểu {field} không hợp lệ; dùng tất cả.")
                kind = "all"
            if kind == "none" and (minimum is not None or maximum is not None):
                warnings.append(f"Bỏ qua khoảng {field} khi đang lọc None.")
                minimum = maximum = None
                item.update(min=None, max=None)
            elif kind == "all" and (minimum is not None or maximum is not None):
                kind = "numeric"
            item["kind"] = kind
        elif field_type == "str_or_float":
            allowed_kinds = {"all", "numeric", *spec.get("choices", [])}
            kind = args.get(f"p_{field}_kind", "all")
            if kind not in allowed_kinds:
                warnings.append(f"Kiểu {field} không hợp lệ; dùng tất cả.")
                kind = "all"
            if kind in set(spec.get("choices", [])) and (
                minimum is not None or maximum is not None
            ):
                warnings.append(f"Bỏ qua khoảng số khi {field} là {kind}.")
                minimum = maximum = None
                item.update(min=None, max=None)
            elif kind == "all" and (minimum is not None or maximum is not None):
                kind = "numeric"
            item["kind"] = kind

        param_filters[field] = item

    sort = args.get("sort", "time_desc")
    allowed_sorts = {"time_desc", "f1_asc", "f1_desc"}
    for field in sortable_params:
        allowed_sorts.update({f"param_{field}_asc", f"param_{field}_desc"})
    if sort not in allowed_sorts:
        warnings.append("Sắp xếp không hợp lệ; dùng thời gian mới nhất.")
        sort = "time_desc"

    try:
        page = int(args.get("page", 1))
        if page < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        warnings.append("Trang không hợp lệ; dùng trang 1.")
        page = 1

    filters = {
        "dataset": dataset,
        "status": status,
        "f1_min": f1_min,
        "f1_max": f1_max,
        "best": args.get("best") == "1",
        "selected": args.get("selected") == "1",
        "sort": sort,
        "page": page,
        "param_filters": param_filters,
        "warnings": warnings,
    }
    query_args = {
        "history_model": model_key,
        "dataset": dataset,
        "status": status,
        "sort": sort,
    }
    if f1_min is not None:
        query_args["f1_min"] = f1_min
    if f1_max is not None:
        query_args["f1_max"] = f1_max
    if filters["best"]:
        query_args["best"] = "1"
    if filters["selected"]:
        query_args["selected"] = "1"
    if page > 1:
        query_args["page"] = page
    for field, item in param_filters.items():
        if item.get("choice"):
            query_args[f"p_{field}"] = item["choice"]
        if item.get("min") is not None:
            query_args[f"p_{field}_min"] = item["min"]
        if item.get("max") is not None:
            query_args[f"p_{field}_max"] = item["max"]
        if item.get("kind") not in {None, "all"}:
            query_args[f"p_{field}_kind"] = item["kind"]
    filters["query_args"] = query_args
    return filters


def _build_history_page(
    rows: list[dict],
    model_key: str,
    fingerprint: str,
    selected_run_id: str | None,
    filters: dict,
) -> dict:
    """Filter, sort and paginate already-annotated history rows."""
    active_rows = []
    for source in rows:
        if source.get("model_key") != model_key:
            continue
        if source.get("policy_id") != EXPERIMENT_POLICY_ID:
            continue
        row = dict(source)
        row["is_selected"] = bool(selected_run_id) and row.get("run_id") == selected_run_id
        active_rows.append(row)

    if filters["dataset"] == "current":
        active_rows = [
            row for row in active_rows if row.get("dataset_fingerprint") == fingerprint
        ]
    if filters["status"] != "all":
        active_rows = [
            row for row in active_rows if row.get("status") == filters["status"]
        ]

    def numeric(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    if filters["f1_min"] is not None:
        active_rows = [
            row
            for row in active_rows
            if numeric(row.get("cv_f1_up_mean")) is not None
            and numeric(row.get("cv_f1_up_mean")) >= filters["f1_min"]
        ]
    if filters["f1_max"] is not None:
        active_rows = [
            row
            for row in active_rows
            if numeric(row.get("cv_f1_up_mean")) is not None
            and numeric(row.get("cv_f1_up_mean")) <= filters["f1_max"]
        ]
    if filters["best"]:
        active_rows = [row for row in active_rows if row.get("is_best")]
    if filters["selected"]:
        active_rows = [row for row in active_rows if row.get("is_selected")]

    for field, item in filters["param_filters"].items():
        choice = item.get("choice")
        if choice:
            active_rows = [
                row for row in active_rows if row.get("params", {}).get(field) == choice
            ]
            continue
        kind = item.get("kind")
        if kind == "none":
            active_rows = [
                row
                for row in active_rows
                if field in (row.get("params") or {})
                and row["params"].get(field) is None
            ]
            continue
        if kind in {"sqrt", "log2"}:
            active_rows = [
                row for row in active_rows if row.get("params", {}).get(field) == kind
            ]
            continue
        if kind == "numeric":
            active_rows = [
                row
                for row in active_rows
                if numeric(row.get("params", {}).get(field)) is not None
            ]
        minimum = item.get("min")
        maximum = item.get("max")
        if minimum is not None:
            active_rows = [
                row
                for row in active_rows
                if numeric(row.get("params", {}).get(field)) is not None
                and numeric(row.get("params", {}).get(field)) >= minimum
            ]
        if maximum is not None:
            active_rows = [
                row
                for row in active_rows
                if numeric(row.get("params", {}).get(field)) is not None
                and numeric(row.get("params", {}).get(field)) <= maximum
            ]

    sort = filters["sort"]
    if sort == "time_desc":
        active_rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    else:
        descending = sort.endswith("_desc")
        if sort.startswith("f1_"):
            value_for = lambda row: numeric(row.get("cv_f1_up_mean"))
        else:
            field = sort.removeprefix("param_").removesuffix("_asc").removesuffix("_desc")
            value_for = lambda row: numeric(row.get("params", {}).get(field))
        present = [row for row in active_rows if value_for(row) is not None]
        missing = [row for row in active_rows if value_for(row) is None]
        present.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
        present.sort(key=value_for, reverse=descending)
        missing.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
        active_rows = present + missing

    total = len(active_rows)
    total_pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    page = min(filters["page"], total_pages)
    offset = (page - 1) * HISTORY_PAGE_SIZE
    page_rows = active_rows[offset : offset + HISTORY_PAGE_SIZE]
    return {
        "rows": page_rows,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "start": offset + 1 if total else 0,
        "end": min(offset + HISTORY_PAGE_SIZE, total) if total else 0,
    }


def _tuning_context(**extra) -> dict:
    """Ghép lịch sử, config, fingerprint và lock thành trạng thái Tuning Lab."""
    fingerprint = compute_dataset_fingerprint()
    experiment_fingerprint = compute_experiment_fingerprint()
    cfg = read_manual_config()
    history = mark_best(read_history())
    preferred_history_model = (
        extra.get("active_history_model")
        or extra.get("active_model")
        or request.args.get("history_model")
    )
    active_history_model = (
        _valid_history_model(preferred_history_model) or _default_active_history_model(cfg)
    )
    param_schema = get_param_schema()
    history_filters = _parse_history_query(
        request.args, active_history_model, param_schema
    )
    selected = (cfg.get("selected_models") or {}).get(active_history_model) or {}
    if selected.get("selection_method") != "manual":
        selected = {}
    history_page = _build_history_page(
        history,
        active_history_model,
        fingerprint["hash"],
        selected.get("run_id"),
        history_filters,
    )
    history_filters["page"] = history_page["page"]
    if history_page["page"] > 1:
        history_filters["query_args"]["page"] = history_page["page"]
    else:
        history_filters["query_args"].pop("page", None)
    counted_history = [
        row
        for row in history
        if row.get("policy_id") == EXPERIMENT_POLICY_ID
        and row.get("dataset_fingerprint") == fingerprint["hash"]
    ]
    history_counts = Counter(row.get("model_key") for row in counted_history)

    common_args = {
        key: value
        for key, value in history_filters["query_args"].items()
        if key in {"dataset", "status", "f1_min", "f1_max", "best", "selected", "sort"}
    }
    if str(common_args.get("sort", "")).startswith("param_"):
        common_args["sort"] = "time_desc"
    history_tabs = []
    for model_key in MODEL_KEY.values():
        tab_args = {**common_args, "history_model": model_key}
        if model_key == active_history_model:
            tab_args = {
                key: value
                for key, value in history_filters["query_args"].items()
                if key != "page"
            }
        history_tabs.append(
            {
                "model_key": model_key,
                "label": HISTORY_MODEL_LABELS[model_key],
                "count": history_counts.get(model_key, 0),
                "url": url_for("tuning", **tab_args),
                "is_active": model_key == active_history_model,
            }
        )

    def page_url(page_number: int) -> str:
        args = {**history_filters["query_args"], "page": page_number}
        return url_for("tuning", **args)

    history_page["prev_url"] = page_url(history_page["page"] - 1) if history_page["page"] > 1 else None
    history_page["next_url"] = (
        page_url(history_page["page"] + 1)
        if history_page["page"] < history_page["total_pages"]
        else None
    )
    tuning_progress = get_tuning_progress(history, fingerprint["hash"])
    complete = is_config_complete(cfg, fingerprint["hash"], history)
    snapshot_evaluated = has_evaluated_snapshot(experiment_fingerprint["hash"])
    pipeline_running = is_pipeline_running()
    fetch_running = is_fetch_running()
    tuning_running = is_tuning_job_running()
    tuning_job = get_tuning_job_state()
    if (
        tuning_job.get("status") == "completed"
        and (tuning_job.get("result") or {}).get("dataset_fingerprint")
        != fingerprint["hash"]
    ):
        tuning_job = {"status": "idle"}
    fetch_progress = _infer_refresh_progress(
        read_fetch_log_tail() if fetch_running else "",
        fetch_running,
    )
    can_run = (
        complete
        and not snapshot_evaluated
        and not pipeline_running
        and not fetch_running
        and not tuning_running
    )

    config_fingerprint = cfg.get("dataset_fingerprint")
    config_stale = bool(cfg.get("selected_models")) and (
        config_fingerprint != fingerprint["hash"]
        or cfg.get("policy_id") != EXPERIMENT_POLICY_ID
    )

    base = {
        "param_schema": param_schema,
        "param_defaults": get_param_defaults(),
        "model_key_map": MODEL_KEY,
        "manual_config": cfg,
        "current_fingerprint": fingerprint["hash"],
        "fingerprint_parts": fingerprint["parts"],
        "history_rows": history_page["rows"],
        "history_tabs": history_tabs,
        "history_param_schema": param_schema[active_history_model],
        "history_filters": history_filters,
        "history_filter_warnings": history_filters["warnings"],
        "history_query_args": history_filters["query_args"],
        "history_pagination": history_page,
        "history_reset_url": url_for(
            "tuning", history_model=active_history_model
        ),
        "active_history_model": active_history_model,
        "history_model_labels": HISTORY_MODEL_LABELS,
        "experiment_policy_id": EXPERIMENT_POLICY_ID,
        "tuning_progress": tuning_progress,
        "tuning_job": tuning_job,
        "tuning_running": tuning_running,
        "config_complete": complete,
        "config_stale": config_stale,
        "snapshot_evaluated": snapshot_evaluated,
        "pipeline_running": pipeline_running,
        "fetch_running": fetch_running,
        "fetch_progress": fetch_progress,
        "can_run": can_run,
    }
    base.update(extra)
    return base


def _render_tuning_ctx(**extra):
    """Context cho tuning.html + luôn kèm active_page ở tầng route (không chôn
    trong _tuning_context để test có thể mock builder mà nav vẫn sáng đúng)."""
    ctx = _tuning_context(**extra)
    ctx.setdefault("active_page", "tuning")
    return ctx


@app.route("/tuning", methods=["GET"])
def tuning():
    return render_template("tuning.html", **_render_tuning_ctx())


@app.route("/tuning/evaluate", methods=["POST"])
def tuning_evaluate():
    """Khởi chạy một CV job nền trên TRAIN; route này không đọc TEST."""
    model_key = request.form.get("model_key", "")
    if model_key not in get_param_schema():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error=f"Model không hợp lệ: {model_key}"),
        ), 400
    if is_pipeline_running() or is_fetch_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(
                error="Không thể tuning khi pipeline hoặc job làm mới dữ liệu đang chạy.",
                active_model=model_key,
            ),
        ), 409
    try:
        params = validate_params(model_key, request.form.to_dict())
        job_id = start_tuning_job(model_key, params)
        if job_id is None:
            return render_template(
                "tuning.html",
                **_render_tuning_ctx(
                    error="Một cấu hình khác đang được đánh giá. Hãy chờ job hoàn tất.",
                    active_model=model_key,
                ),
            ), 409
        return redirect(url_for("tuning", history_model=model_key, job=job_id))
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error=str(exc), active_model=model_key),
        ), 400


@app.route("/tuning/use-config", methods=["POST"])
def tuning_use_config():
    """Chốt một run hợp lệ cho model và dataset fingerprint hiện tại."""
    run_id = request.form.get("run_id", "")
    run = find_run(run_id)
    if run is None:
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error=f"Không tìm thấy run_id: {run_id}"),
        ), 400

    fingerprint = compute_dataset_fingerprint()
    if run.get("dataset_fingerprint") != fingerprint["hash"]:
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(
                error="Cấu hình này thuộc phiên bản dataset khác với hiện tại. "
                "Hãy chạy CV và tự chọn lại cấu hình trên dataset hiện tại."
            ),
        ), 400
    if run.get("status") != "ok":
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Run này bị lỗi, không thể chốt làm cấu hình."),
        ), 400

    params = json.loads(run.get("params_json") or "{}")
    cv_mean = run.get("cv_f1_up_mean")
    try:
        save_selected_model(run["model_key"], run_id, params, cv_mean, fingerprint)
    except ValueError as exc:
        return render_template(
            "tuning.html", **_render_tuning_ctx(error=str(exc), active_model=run["model_key"])
        ), 400
    history_filters = _parse_history_query(
        request.form, run["model_key"], get_param_schema()
    )
    return redirect(url_for("tuning", **history_filters["query_args"]))


@app.route("/tuning/run-pipeline", methods=["POST"])
def tuning_run_pipeline():
    """Chạy official pipeline sau khi đủ ba config cho snapshot hiện tại."""
    fingerprint = compute_dataset_fingerprint()
    experiment_fingerprint = compute_experiment_fingerprint()
    cfg = read_manual_config()

    if not is_config_complete(cfg, fingerprint["hash"]):
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(
                error="Chưa tự chọn cấu hình hợp lệ cho cả 3 model trên dataset hiện tại. "
                "Hãy chạy CV và chốt đủ 3 model trước."
            ),
        ), 400
    if has_evaluated_snapshot(experiment_fingerprint["hash"]):
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(
                error="Snapshot dữ liệu hiện tại đã được đánh giá. Hãy làm mới dữ liệu "
                "để tạo fingerprint mới trước khi chạy lại."
            ),
        ), 400
    if is_pipeline_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Pipeline đang chạy. Vui lòng đợi hoàn tất."),
        ), 409
    if is_fetch_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Đang làm mới dữ liệu, vui lòng đợi."),
        ), 409
    if is_tuning_job_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Tuning CV đang chạy. Hãy chờ job hoàn tất."),
        ), 409

    try:
        proc = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        write_pipeline_log((proc.stdout or "") + "\n" + (proc.stderr or ""))
        return redirect(url_for("evaluation"))
    except subprocess.CalledProcessError as exc:
        write_pipeline_log((exc.stdout or "") + "\n" + (exc.stderr or ""))
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(
                error="Pipeline chạy thất bại. Xem log bên dưới.",
                pipeline_log=read_pipeline_log_tail(),
            ),
        ), 500


def _load_fetch_report() -> dict | None:
    if not FETCH_REPORT_PATH.exists():
        return None
    try:
        return json.loads(FETCH_REPORT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@app.route("/tuning/fetch-data", methods=["POST"])
def tuning_fetch_data():
    """Khởi chạy refresh_data.py nền; trang status đọc lock và logfile."""
    if is_fetch_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Đang làm mới dữ liệu, vui lòng đợi."),
        ), 409
    if is_pipeline_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Pipeline đang chạy. Vui lòng đợi hoàn tất."),
        ), 409
    if is_tuning_job_running():
        return render_template(
            "tuning.html",
            **_render_tuning_ctx(error="Tuning CV đang chạy. Hãy chờ job hoàn tất."),
        ), 409

    ensure_experiments_dir()
    with open(LAST_FETCH_RUN_LOG, "w", encoding="utf-8", errors="replace") as logfile:
        proc = subprocess.Popen(
            [sys.executable, "-u", "scripts/refresh_data.py"],
            cwd=str(ROOT_DIR),
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
    write_fetch_lock(proc.pid)
    return redirect(url_for("tuning_fetch_status"))


@app.route("/tuning/fetch-status", methods=["GET"])
def tuning_fetch_status():
    running = is_fetch_running()
    log = read_fetch_log_tail()
    fetch_report = _load_fetch_report()
    fetch_progress = _infer_refresh_progress(log, running)
    return render_template(
        "fetch_status.html",
        running=running,
        log=log,
        fetch_report=fetch_report,
        fetch_progress=fetch_progress,
        active_page="tuning",
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
