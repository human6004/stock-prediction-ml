"""Flask web layer: nhận request, gọi service và render giao diện.

Thuật toán ML không nằm ở đây. Route prediction gọi prediction_service; route
tuning gọi tuning_lab; official pipeline được chạy qua scripts/run_pipeline.py.
"""

import io
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from flask import (
    Flask,
    Response,
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
    """Suy trạng thái 3 bước refresh **chỉ từ text log** + cờ tiến trình còn sống.

    Vì `scripts/refresh_data.py` chạy ở process con và không báo tiến độ qua API
    nào, web chỉ có hai nguồn thông tin: nội dung file log và việc fetch.lock còn
    hay hết. Hàm này ghép hai nguồn đó thành trạng thái cho UI:

    - `current`: bước cao nhất đã *bắt đầu*, đọc ngược từ marker `[3/3]` → `[1/3]`
      mà script in ra. Đọc ngược vì log là append-only: marker lớn nhất xuất hiện
      là bước mới nhất.
    - `completed`: chỉ khi thấy đúng câu kết `Refresh data completed.`
    - Mấu chốt phân biệt lỗi: bước `== current` mà tiến trình **không còn chạy**
      và chưa completed ⇒ script đã chết giữa bước đó ⇒ state `error`. Nếu vẫn
      chạy ⇒ `active`. Đây là cách duy nhất phát hiện crash khi không có exit code.
    - `percent` cố tình dừng ở 90 khi đang ở bước 3: chỉ câu kết mới cho 100, để
      thanh tiến trình không "đầy" trước khi ghi file xong.
    """
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

# Cột xuất ra file Excel khi tải các mã dự báo đã chọn từ trang Xếp hạng.
# Mỗi phần tử là (nhãn tiếng Việt, khóa dữ liệu). Điểm UP, ngưỡng quyết định và
# return/volatility trong `predict_all_symbols` là tỉ lệ 0..1 nên route export
# đổi thành phần trăm trước khi đưa vào bảng, không xuất thẳng giá trị gốc khó đọc.
EXPORT_HEADERS = [
    ("Mã", "symbol"),
    ("Dự báo", "prediction"),
    ("Điểm UP (%)", "diem_up_pct"),
    ("Ngưỡng quyết định (%)", "decision_threshold_pct"),
    ("Giá tham chiếu", "close_at_reference"),
    ("Ngày dữ liệu", "reference_date"),
    ("Return 20 phiên (%)", "return_20d_pct"),
    ("Volatility 20 phiên (%)", "volatility_20d_pct"),
    ("Volume / AVG20", "volume_ratio_20"),
    ("Dữ liệu cũ", "is_stale"),
    ("Model", "model_name"),
]

# Bảng màu cố định cho file Excel (openpyxl không đọc biến CSS của giao diện).
_HEADER_FILL = PatternFill("solid", fgColor="1F2937")
_UP_FONT = Font(color="166534", bold=True)
_UP_FILL = PatternFill("solid", fgColor="DCFCE7")
_DOWN_FONT = Font(color="991B1B", bold=True)
_DOWN_FILL = PatternFill("solid", fgColor="FEE2E2")
_THIN = Side(style="thin", color="E5E7EB")
_CELL_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _fmt_date_vi(iso: str) -> str:
    """Đổi ngày ISO ``YYYY-MM-DD`` sang ``DD/MM/YYYY`` cho dễ đọc trên file."""
    if not iso:
        return ""
    text = str(iso)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return str(iso)


def _build_export_xlsx(rows: list[dict]) -> bytes:
    """Dựng workbook Excel có style từ các dòng dự báo đã chuẩn hóa.

    Style: header nền tối chữ trắng + freeze dòng đầu; cột Dự báo tô UP xanh /
    NOT_UP đỏ; cột số căn phải kèm format thập phân; độ rộng cột tự co theo nội dung.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Dự báo"

    for col_idx, (label, _key) in enumerate(EXPORT_HEADERS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=label)
        cell.font = Font(color="FFFFFF", bold=True, size=11)
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _CELL_BORDER
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"

    for row_idx, r in enumerate(rows, start=2):
        prediction = r.get("prediction")
        for col_idx, (_label, key) in enumerate(EXPORT_HEADERS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=r.get(key))
            cell.border = _CELL_BORDER
            if key in ("diem_up_pct", "decision_threshold_pct"):
                cell.number_format = "0.0"
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif key in ("return_20d_pct", "volatility_20d_pct", "volume_ratio_20"):
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif key == "close_at_reference":
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif key == "symbol":
                cell.font = Font(bold=True)
                cell.alignment = Alignment(vertical="center")
            elif key == "prediction":
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if prediction == "UP":
                    cell.font = _UP_FONT
                    cell.fill = _UP_FILL
                else:
                    cell.font = _DOWN_FONT
                    cell.fill = _DOWN_FILL
            elif key == "is_stale":
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.font = Font(color="B45309", bold=True) if r.get(key) else Font(color="9CA3AF")
            else:
                cell.alignment = Alignment(vertical="center")
        ws.row_dimensions[row_idx].height = 20

    for col_idx, (label, _key) in enumerate(EXPORT_HEADERS, start=1):
        width = len(str(label))
        for r in rows:
            value = r.get(_key)
            if value is not None:
                width = max(width, len(str(value)))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(width + 2, 26)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

HISTORY_PAGE_SIZE = 50

# Kiểu tham số sắp xếp được: đọc ra số. Kiểu choice (solver) không có thứ tự.
HISTORY_SORTABLE_PARAM_TYPES = {"float", "int", "int_or_none", "str_or_float"}

# Trạng thái sắp xếp trung tính: thứ tự vốn có của bảng (mới nhất trước). Cần một
# token riêng chứ không dùng lại time_desc, nếu không cột "Thời điểm" sẽ không phân
# biệt được "chưa sắp" với "đang giảm dần" và mất trạng thái thứ ba.
HISTORY_DEFAULT_SORT = "default"

# Cột cố định cho phép sắp xếp ở bảng lịch sử: khóa dùng trong ?sort= → hàm lấy
# giá trị. Cột tham số (param_<field>) sinh động theo schema từng model nên xử lý
# riêng trong _parse_history_query. Bảng này phân trang ở server nên phải sắp ở
# server, không sắp bằng JS trên một trang.
HISTORY_SORT_COLUMNS = {
    "time": lambda row: str(row.get("timestamp") or ""),
    "f1": lambda row: row.get("cv_f1_up_mean"),
    "std": lambda row: row.get("cv_f1_up_std"),
    "threshold": lambda row: row.get("decision_threshold"),
    "seconds": lambda row: row.get("train_seconds"),
}


def get_dataset_meta() -> dict:
    """Meta hiển thị ở header mọi trang (ngày mới nhất, số mã, danh sách mã).

    Mẫu cache-theo-chữ-ký: hàm này KHÔNG cache, nó chỉ ``stat()`` file (rẻ) rồi
    truyền ``(size, mtime_ns)`` vào ``_get_dataset_meta`` — hàm mới có
    ``lru_cache``. Vì chữ ký là tham số, khi file feature bị ghi lại (refresh
    data / chạy pipeline) chữ ký đổi → lru_cache miss → đọc lại CSV. Nếu đặt
    ``lru_cache`` trực tiếp lên hàm đọc CSV thì web sẽ dính meta cũ đến khi
    restart. ``mtime_ns`` (nanosecond) thay vì ``mtime`` giây để không bỏ sót hai
    lần ghi trong cùng một giây.
    """
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
    # ``usecols`` để chỉ nạp 2 cột thay vì cả 20+ feature — file này hàng trăm
    # nghìn dòng, đọc full mỗi lần render trang là quá tốn.
    cols = pd.read_csv(FEATURE_DATA_PATH, usecols=["symbol", "trading_date"])
    return {
        "dataset_max_date": str(cols["trading_date"].max()),
        "symbol_count": int(cols["symbol"].nunique()),
        "symbols": tuple(sorted(cols["symbol"].astype(str).unique())),
    }

def load_selected_model_from_report() -> str:
    """Lấy tên model đang phục vụ, ưu tiên metadata rồi mới tới report.

    Thứ tự nguồn là có chủ ý:
    1. ``model_metadata.json`` — đi kèm chính file ``final_model.pkl`` được publish
       (cùng một ``atomic_model_release``), nên luôn khớp model đang chạy.
    2. ``model_comparison.csv`` — chỉ là fallback cho artifact cũ chưa có metadata.
       Report có thể được ghi lại ở lần chạy khác nên có nguy cơ lệch với .pkl.

    Trả "" khi không tra được: đây chỉ là nhãn hiển thị trên UI, thiếu thì để
    trống chứ không raise làm sập trang.
    """
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
    """Đọc CSV báo cáo thành list dict cho Jinja render, xử lý 2 cạm bẫy của pandas.

    1. ``cv_f1_up`` có thể trống (baseline không có CV). Sau ``read_csv`` ô trống
       thành ``NaN`` — Jinja in ra chữ "nan" và ``tojson`` sinh JSON không hợp lệ.
       ``.where(notna(), None)`` đổi về ``None`` để template hiện ô rỗng.
    2. Cột ``selected`` trong CSV là chuỗi "True"/"False" chứ không phải bool
       (đi qua CSV thì mất kiểu). Vì vậy phải ``astype(str).str.lower()`` rồi
       ``map`` — so ``== True`` sẽ luôn sai. ``map`` chỉ khớp "true"; mọi giá trị
       khác thành ``NaN`` rồi ``fillna("")`` → cột ``status`` chỉ có nhãn
       "Đang dùng" ở đúng một dòng, các dòng còn lại rỗng.
    """
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


def _read_model_metadata() -> dict:
    """Đọc ``model_metadata.json``, fail an toàn về ``{}``.

    Thiếu file hoặc JSON vỡ đều trả ``{}`` thay vì raise: metadata chỉ dùng để
    HIỂN THỊ (tên model, ngày train, metric), thiếu thì để trống chứ không được
    làm sập trang.
    """
    if not MODEL_METADATA_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _model_health_summary() -> dict:
    """Sức khỏe model đang publish, dạng gọn cho cột phải trang Dự báo.

    Dùng ĐÚNG cổng chặn policy của ``load_evaluation_sections``: artifact không
    thuộc ``EXPERIMENT_POLICY_ID`` thì **không** in metric TEST. Số liệu đó sinh
    bằng giao thức cũ (mốc split khác, cách chọn threshold khác); đặt nó cạnh
    model đang chạy sẽ khiến người đọc tưởng đây là chất lượng của policy hiện
    hành. Legacy chỉ còn tên model + ngày dữ liệu train + một dòng dẫn sang trang
    Đánh giá.

    ``final_model_evaluation.csv`` có cả dòng baseline (``row_type`` là
    ``baseline``) nên phải lọc ``row_type == "final"``, không lấy dòng đầu.
    """
    metadata = _read_model_metadata()
    is_current_policy = metadata.get("policy_id") == EXPERIMENT_POLICY_ID
    health = {
        "model_name": metadata.get("model_name") or "",
        "train_through_date": metadata.get("train_through_date")
        or metadata.get("validation_end_date"),
        "decision_threshold": metadata.get("decision_threshold"),
        "is_current_policy": is_current_policy,
        "baseline_warning": metadata.get("baseline_warning"),
        "metrics": [],
    }
    if not is_current_policy:
        return health

    final_rows = [
        row
        for row in _read_report_rows(FINAL_MODEL_EVALUATION_PATH)
        if row.get("row_type") == "final"
    ]
    if not final_rows:
        return health
    row = final_rows[0]
    health["metrics"] = [
        {
            "label": "Accuracy",
            "value": row.get("accuracy"),
            "hint": "Tỉ lệ đoán đúng trên toàn bộ tập TEST, tính cả UP và NOT_UP.",
        },
        {
            "label": "F1 (UP)",
            "value": row.get("f1_up"),
            "hint": "Điểm cân bằng giữa precision và recall của riêng nhãn UP.",
        },
        {
            "label": "Precision (UP)",
            "value": row.get("precision_up"),
            "hint": "Trong các mã model gọi UP, bao nhiêu phần thực sự tăng vượt ngưỡng.",
        },
    ]
    return health


def _top_signals(limit: int = 5) -> list[dict]:
    """Các mã điểm UP cao nhất, dùng lại đúng cache inference của trang Xếp hạng.

    Không thêm pipeline: ``predict_all_symbols`` đã cache theo chữ ký data +
    model nên gọi ở đây trùng cache với ``/screener``.

    Bọc ``except`` rộng có chủ ý: khối này chỉ là phần bổ trợ ở cột phải, thiếu
    artifact hay dataset thì trả rỗng và trang vẫn dự báo được — không được để
    nó làm 500 cả trang chính.
    """
    try:
        rows = predict_all_symbols()
    except Exception:  # noqa: BLE001
        return []
    scored = [row for row in rows if row.get("probability_up") is not None]
    scored.sort(key=lambda row: row["probability_up"], reverse=True)
    return scored[:limit]


def load_evaluation_sections() -> dict:
    """Gom dữ liệu trang Đánh giá, tách rõ report "policy hiện hành" vs "legacy".

    Cổng chặn nằm ở ``policy_id`` trong ``model_metadata.json``: nếu artifact không
    thuộc ``EXPERIMENT_POLICY_ID`` hiện tại thì các report trên đĩa được sinh bằng
    giao thức CŨ (mốc split khác, cách chọn threshold khác), nên số liệu VALIDATION
    và TEST không so sánh được với hiện tại. Khi đó:

    - ``validation``/``test`` trả rỗng — cố ý không hiển thị, thà thiếu hơn cho
      người đọc tưởng đây là kết quả của policy hiện hành;
    - toàn bộ ``model_comparison.csv`` được đẩy sang khoá ``legacy`` kèm
      ``legacy_warning`` để UI hiện banner "hãy chạy lại pipeline";
    - ``baseline_warning`` cũng bị bỏ vì cảnh báo baseline của policy cũ vô nghĩa.

    Metadata đọc lỗi (thiếu file, JSON vỡ) → ``metadata = {}`` → ``policy_id`` là
    ``None`` → coi như legacy. Fail an toàn, không raise ra 500.
    """
    metadata = _read_model_metadata()
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
        **_template_context(
            selected_model=selected_model,
            active_page="prediction",
            # Cột phải trạng thái rỗng: tín hiệu nổi bật + sức khỏe model. Chỉ tính ở
            # GET "/" — POST /predict đã có kết quả nên cột phải không render.
            top_signals=_top_signals(),
            model_health=_model_health_summary(),
        ),
    )


def _chat_error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _validate_chat_payload(payload) -> tuple[str, list[dict]]:
    """Kiểm tra payload /api/chat rồi trả message và history đã chuẩn hóa.

    Nghiêm ngặt có chủ đích — history do client gửi lên, nếu tin thẳng thì user
    có thể bơm role/nội dung tùy ý vào prompt của LLM (prompt injection qua
    lịch sử giả). Các luật:

    - ``set(payload) - {"message","history"}``: chặn key lạ, không chỉ thiếu key.
      Whitelist thay vì blacklist nên field mới thêm ở client cũng bị chặn tới
      khi server công nhận.
    - ``len(history) % 2``: history phải chẵn, tức luôn là từng cặp user↔assistant
      đã hoàn tất. Số lẻ nghĩa là có lượt bị cắt → coi là hỏng.
    - ``expected_role`` theo index chẵn/lẻ: bắt buộc xen kẽ user, assistant, user…
      Client không được tự khai role, thứ tự quyết định role.
    - ``set(item) != {"role","content"}``: mỗi item đúng 2 key, không hơn không kém.
    - Giới hạn ba tầng: mỗi message ≤ MAX_MESSAGE_CHARS, số lượt ≤
      MAX_HISTORY_MESSAGES, và tổng ký tự ≤ MAX_HISTORY_CHARS. Tầng tổng là chốt
      chi phí token: 20 message ngắn hợp lệ, nhưng 20 message dài thì không.

    Mọi lỗi đều ``raise ValueError`` trần (không message) vì caller chỉ trả về
    đúng một câu lỗi chung — không tiết lộ luật kiểm tra nào đã chặn.
    """
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
    """Endpoint chat: 3 tầng lỗi, tuyệt đối không để chi tiết nội bộ lọt ra client.

    - Chặn ``mimetype`` trước khi parse: bắt buộc ``application/json`` nên form
      POST từ trang khác (CSRF đơn giản) bị loại ngay.
    - ``get_json(silent=True)`` trả ``None`` khi body không phải JSON thay vì tự
      raise 400 của Werkzeug, để mọi lỗi input đi qua cùng một ``_chat_error``
      với mã ``invalid_request``.
    - ``ChatbotServiceError`` mang sẵn ``code``/``message``/``status`` (400/502/
      503/504) đã được service soạn cho người dùng đọc → chuyển thẳng.
    - ``except Exception`` cuối là chốt an toàn: log phía server nhưng client chỉ
      nhận ``internal_error`` chung. Cố ý KHÔNG đưa ``str(exc)`` vào response vì
      exception của provider có thể chứa URL endpoint hoặc mảnh API key.
    """
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
        # exception() thay vì error(): giữ traceback trong log server, nếu không
        # lỗi nội bộ chỉ còn một dòng vô nghĩa vì response cố ý không mang str(exc).
        app.logger.exception("Unexpected chatbot error")
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
                    # Nhập sai mã vẫn là trạng thái "chưa có kết quả" → template
                    # render cột phải. Không truyền context thì cột đó rỗng ruột,
                    # chỉ còn khung — tệ hơn cả lúc chưa bấm gì.
                    top_signals=_top_signals(),
                    model_health=_model_health_summary(),
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
    # Một bảng thống nhất: UP trước (đã sort điểm giảm dần), rồi NOT_UP.
    all_rows = up_rows + not_up_rows
    context.update(
        all_rows=all_rows,
        up_count=len(up_rows),
        not_up_count=len(not_up_rows),
        total_count=len(rows),
        selected_model=rows[0]["model_name"] if rows else context["selected_model"],
    )
    return render_template("screener.html", **_template_context(**context))


@app.route("/screener/export", methods=["POST"])
def screener_export():
    """Tải file Excel (.xlsx) gồm kết quả dự báo của các mã đã chọn trên trang Xếp hạng.

    Không gọi lại model: lọc từ ``predict_all_symbols()`` (đã cache theo fingerprint
    data+model). Giữ nguyên thứ tự mã client gửi — chính là thứ tự hiển thị trên
    bảng. Bảng có style (header tô màu, UP/NOT_UP phân màu, số làm tròn) để người
    dùng mở bằng Excel dễ đọc hơn CSV thô.
    """
    raw = request.form.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        return "Chưa chọn mã nào.", 400

    by_symbol = {row["symbol"]: row for row in predict_all_symbols()}
    missing = [s for s in symbols if s not in by_symbol]
    if missing:
        return f"Mã không có trong dữ liệu: {', '.join(missing)}", 400

    rows = []
    for s in symbols:
        r = by_symbol[s]
        prob = r.get("probability_up")
        rows.append(
            {
                "symbol": s,
                "prediction": r.get("prediction"),
                "diem_up_pct": round(prob * 100, 1) if prob is not None else None,
                "decision_threshold_pct": round(
                    float(r.get("decision_threshold", 0)) * 100, 1
                ),
                "close_at_reference": round(
                    float(r.get("close_at_reference") or 0), 2
                ),
                "reference_date": _fmt_date_vi(r.get("reference_date")),
                "return_20d_pct": round(float(r.get("return_20d") or 0) * 100, 2),
                "volatility_20d_pct": round(
                    float(r.get("volatility_20d") or 0) * 100, 2
                ),
                "volume_ratio_20": round(float(r.get("volume_ratio_20") or 0), 2),
                "is_stale": "Có" if r.get("is_stale") else "",
                "model_name": r.get("model_name"),
            }
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Response(
        _build_export_xlsx(rows),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=hose_du_bao_{stamp}.xlsx"
        },
    )


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
        selected_model=selected_model,
        dataset_max_date=get_dataset_meta()["dataset_max_date"],
        active_page="evaluation",
    )


def _valid_history_model(model_key: str | None) -> str | None:
    if model_key in HISTORY_MODEL_LABELS:
        return model_key
    return None


def _default_active_history_model(cfg: dict) -> str:
    """Tab model nào mở sẵn khi vào /tuning mà URL không chỉ định ``history_model``.

    Chọn model mà người dùng chốt cấu hình GẦN NHẤT — nghĩa là chỗ họ đang làm
    việc. Thủ thuật: ghép tuple ``(selected_at, model_key)`` rồi ``max()``; vì
    ``selected_at`` là ISO-8601 (``2026-07-19T19:12:27``) nên so sánh chuỗi cho
    ra đúng thứ tự thời gian, không cần parse datetime. Model chưa chốt (không
    có ``selected_at``) thành chuỗi rỗng → luôn thua. ``model_key`` ở vị trí thứ
    hai chỉ để tie-break tất định khi hai model chốt cùng một giây.

    Chưa chốt gì cả → mở Logistic Regression (model đơn giản nhất, cũng là bước
    đầu trong quy trình tuning 3 model).
    """
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
    """Đọc query string của bảng lịch sử tuning và chuẩn hóa thành filter dict.

    VÌ SAO PHỨC TẠP: các cột tham số (C, max_depth, max_features...) không cố
    định — mỗi model có schema riêng (``TUNABLE_PARAM_SCHEMA``). Nên hàm này
    không thể hard-code danh sách filter; nó *sinh* filter từ schema của model
    đang xem. Thêm nữa, hai kiểu tham số có "giá trị lai" vừa số vừa chữ:

    - ``int_or_none`` (vd ``max_depth``): hoặc là số nguyên, hoặc là ``None``
      (không giới hạn độ sâu).
    - ``str_or_float`` (vd ``max_features``): hoặc là số thực, hoặc là chuỗi
      ``"sqrt"``/``"log2"``.

    Không thể lọc kiểu lai chỉ bằng min/max, nên mỗi field như vậy có thêm tham
    số ``p_<field>_kind`` chọn *nhánh giá trị*: ``all`` | ``numeric`` | ``none``
    (hoặc chính tên choice). Bảng chân lý kind × range mà đoạn dưới thực thi:

        kind=none/sqrt/log2 + có min/max  → bỏ min/max (mâu thuẫn: đang lọc
                                            nhánh không phải số)
        kind=all             + có min/max  → tự nâng thành numeric (người dùng
                                            gõ khoảng số thì rõ ràng muốn nhánh số)
        kind=all             + không range → giữ all (không lọc gì)

    NGUYÊN TẮC CHUNG: input sai không bao giờ làm request lỗi. Mọi giá trị lạ bị
    bỏ qua, thay bằng mặc định an toàn, và đẩy một câu tiếng Việt vào
    ``warnings`` để template hiện cho người dùng biết filter nào đã bị bỏ.

    Trả về dict gồm: các filter đã sạch, ``warnings``, và ``query_args`` — bộ
    tham số tối giản (chỉ chứa filter khác mặc định) dùng để dựng lại URL cho
    link phân trang và link sắp xếp, nên bấm sang trang 2 không mất filter.
    """
    warnings: list[str] = []
    model_schema = schema.get(model_key, {})

    def parse_number(key: str, *, integer: bool = False):
        """Đọc một tham số số từ query string, sai thì bỏ qua chứ không lỗi 400.

        Query string do người dùng gõ/sửa tay được nên phải chịu mọi rác. Quy tắc:
        - Trống/không có → ``None`` (nghĩa là "không filter"), không phải cảnh báo.
        - ``math.isfinite`` chặn ``nan``/``inf``: ``float("nan")`` parse được nhưng
          mọi so sánh với nó đều False, sẽ lọc mất hết dòng một cách âm thầm.
        - ``integer=True`` đòi ``value.is_integer()`` nên "5.0" hợp lệ còn "5.5"
          thì không, dùng cho các tham số kiểu số nguyên (trang, n_estimators...).
        - Giá trị xấu → ghi vào ``warnings`` (closure của hàm ngoài, hiện lên UI)
          rồi trả ``None``, tức bỏ đúng filter đó và vẫn dựng được trang.
        """
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

    dataset = args.get("dataset", "all")
    if dataset not in {"current", "all"}:
        warnings.append("Dataset không hợp lệ; dùng toàn bộ lịch sử.")
        dataset = "all"

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
    numeric_types = HISTORY_SORTABLE_PARAM_TYPES
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

    sort = args.get("sort", HISTORY_DEFAULT_SORT)
    # Whitelist sort được SINH RA, không hard-code: cột cố định lấy từ
    # HISTORY_SORT_COLUMNS, cột tham số lấy từ sortable_params (đã gom ở vòng lặp
    # schema phía trên, chỉ gồm kiểu số — "solver" là choice nên không sắp được).
    # Whitelist chứ không parse tự do vì token sort sẽ tra vào bảng lambda; token
    # lạ mà lọt qua sẽ thành KeyError khi sắp xếp.
    # time_desc giữ lại vì link cũ / bookmark còn dùng, dù mặc định giờ là "default".
    allowed_sorts = {HISTORY_DEFAULT_SORT, "time_desc", "time_asc"}
    for column in HISTORY_SORT_COLUMNS:
        allowed_sorts.update({f"{column}_asc", f"{column}_desc"})
    for field in sortable_params:
        allowed_sorts.update({f"param_{field}_asc", f"param_{field}_desc"})
    if sort not in allowed_sorts:
        warnings.append("Sắp xếp không hợp lệ; dùng thời gian mới nhất.")
        sort = HISTORY_DEFAULT_SORT

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
    # query_args = bộ tham số ĐÃ LÀM SẠCH, dùng để dựng lại URL trong template
    # (link phân trang, link đổi chiều sort, link đổi model). Nhờ nó mà bấm "trang
    # sau" không mất filter đang bật, và một URL bẩn sau khi vào trang sẽ trở thành
    # URL sạch ở mọi link tiếp theo.
    # Chỉ ghi vào query_args những giá trị KHÁC mặc định (xem các `if` bên dưới) để
    # URL ngắn: không nhồi `f1_min=&best=&page=1` vào mọi link.
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


def _numeric(value):
    """Đọc value thành float, trả None nếu không phải số dùng được để so sánh.

    Ba cạm bẫy được chặn ở đây:
    - ``bool`` bị loại tường minh: trong Python ``True`` LÀ ``int`` (``1``), nên
      nếu không chặn thì một param boolean sẽ lọt vào bộ lọc khoảng số và so
      sánh được với 0/1 — vô nghĩa.
    - Chuỗi (vd ``max_features="sqrt"``) trả None, không cố ép sang số.
    - NaN/inf trả None: NaN so sánh với bất cứ gì cũng False nên nếu để lọt,
      dòng đó vừa không thỏa min vừa không thỏa max, và với ``sort`` thì NaN
      làm thứ tự trở nên không tất định. Với ``history_trend`` còn thêm một lý
      do cứng: NaN/inf serialize ra JSON không hợp lệ (``NaN``), làm
      ``|tojson`` sinh attribute mà ``JSON.parse`` phía client không đọc được.
    None ở đây luôn nghĩa là "không có giá trị số" → dòng bị loại khỏi bộ lọc
    khoảng, bị đẩy xuống cuối khi sắp xếp, hoặc bị loại khỏi biểu đồ trend.

    Nằm ở tầng module (không lồng trong ``_build_history_page``) để bảng lịch sử
    và biểu đồ trend dùng CHUNG một định nghĩa "số dùng được" — hai bản copy sẽ
    trôi khỏi nhau ngay lần sửa đầu tiên.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_selected_run(row: dict, selected_run_id: str | None) -> bool:
    """Row này có phải run đang được chốt làm cấu hình chính thức không.

    Tách ra vì cả bảng lịch sử lẫn biểu đồ trend đều cần đánh dấu điểm "đang
    dùng"; nếu mỗi bên tự so sánh thì một ngày nào đó bảng nói "đang dùng" mà
    biểu đồ lại không tô điểm nào.
    """
    return bool(selected_run_id) and row.get("run_id") == selected_run_id


def _build_history_trend(
    rows: list[dict],
    model_key: str,
    fingerprint: str,
    selected_run_id: str | None,
) -> list[dict]:
    """Chuỗi F1 theo thời gian của model đang mở — dữ liệu cho biểu đồ "đã hội tụ chưa".

    Khác bảng lịch sử ở ba điểm, và cả ba đều là chủ ý:

    - KHÔNG chịu ảnh hưởng của ``history_filters``. Bảng là công cụ tra cứu nên
      người dùng lọc/sắp tùy ý; biểu đồ là câu trả lời cho "tuning đã hội tụ
      chưa", mà câu đó chỉ có nghĩa khi trục X là thời gian trên TOÀN bộ số lần
      thử. Sắp theo F1 sẽ vẽ ra một đường luôn đi lên — đẹp và vô nghĩa.
    - Luôn khóa vào dataset hiện tại: F1 của hai fingerprint khác nhau là hai
      thang đo, nối chúng thành một đường là sai.
    - Chỉ giữ run có ``cv_f1_up_mean`` là số hữu hạn (xem ``_numeric``); run lỗi
      không có F1 nên không có điểm để vẽ.

    Thứ tự: theo ``timestamp`` tăng dần. ``timestamp`` là chuỗi ISO nên so sánh
    chuỗi đúng bằng so sánh thời gian, và dùng lại ``HISTORY_SORT_COLUMNS["time"]``
    để không tự viết lại cách đọc cột này. ``list.sort`` là stable, nên timestamp
    trùng nhau (hoặc rỗng/không parse được — đều quy về chuỗi) giữ nguyên thứ tự
    trong file, tức thứ tự append. Không parse datetime ở đây chính vì thế: một
    dòng CSV bị sửa tay sẽ làm ``fromisoformat`` ném lỗi và mất cả biểu đồ.
    """
    matching = [
        row
        for row in rows
        if row.get("model_key") == model_key
        and row.get("policy_id") == EXPERIMENT_POLICY_ID
        and row.get("dataset_fingerprint") == fingerprint
        and _numeric(row.get("cv_f1_up_mean")) is not None
    ]
    matching.sort(key=HISTORY_SORT_COLUMNS["time"])
    return [
        {
            "n": index,
            "f1": _numeric(row.get("cv_f1_up_mean")),
            "timestamp": str(row.get("timestamp") or ""),
            "is_best": bool(row.get("is_best")),
            "is_selected": _is_selected_run(row, selected_run_id),
        }
        for index, row in enumerate(matching, start=1)
    ]


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
        row["is_selected"] = _is_selected_run(row, selected_run_id)
        active_rows.append(row)

    if filters["dataset"] == "current":
        active_rows = [
            row for row in active_rows if row.get("dataset_fingerprint") == fingerprint
        ]
    if filters["status"] != "all":
        active_rows = [
            row for row in active_rows if row.get("status") == filters["status"]
        ]

    # Alias cục bộ tới helper module-level: giữ nguyên tên `numeric` cho phần
    # lọc/sắp bên dưới, nhưng định nghĩa nằm ở một chỗ duy nhất (_numeric) mà
    # _build_history_trend cũng dùng.
    numeric = _numeric

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
    descending = sort.endswith("_desc")
    column = sort.removesuffix("_asc").removesuffix("_desc")
    if sort == HISTORY_DEFAULT_SORT:
        column, descending = "time", True
    if column == "time":
        # Timestamp ISO nên so sánh chuỗi là đúng thứ tự thời gian, và không bao
        # giờ rỗng — sắp trực tiếp, không cần tách nhóm thiếu giá trị.
        active_rows.sort(key=HISTORY_SORT_COLUMNS["time"], reverse=descending)
    else:
        if column in HISTORY_SORT_COLUMNS:
            getter = HISTORY_SORT_COLUMNS[column]
        else:
            field = column.removeprefix("param_")
            getter = lambda row: (row.get("params") or {}).get(field)
        value_for = lambda row: numeric(getter(row))
        # Tách dòng có giá trị và dòng thiếu giá trị (None/NaN/kiểu lạ). Dòng
        # thiếu KHÔNG được tham gia so sánh số: nếu để lẫn, sort sẽ so None với
        # float và ném TypeError, hoặc (nếu quy về 0) đẩy dòng rỗng lên đầu bảng
        # khi sort tăng dần. Nhóm missing luôn xếp cuối, bất kể asc/desc.
        present = [row for row in active_rows if value_for(row) is not None]
        missing = [row for row in active_rows if value_for(row) is None]
        # Sort HAI LẦN, tận dụng tính stable của list.sort: lần đầu theo
        # timestamp giảm dần (tie-break), lần sau theo cột người dùng chọn. Vì
        # stable sort giữ nguyên thứ tự tương đối của các phần tử bằng nhau, kết
        # quả là "sắp theo cột chính, các dòng cùng giá trị thì mới nhất trước".
        # Thứ tự hai lệnh này KHÔNG đảo được: khóa phụ phải sort trước khóa chính.
        present.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
        present.sort(key=value_for, reverse=descending)
        missing.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
        active_rows = present + missing

    # Phân trang sau khi đã lọc + sắp. total_pages tối thiểu là 1 để bảng rỗng
    # vẫn hiển thị "trang 1/1" thay vì 1/0.
    total = len(active_rows)
    total_pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    # Kẹp page vào [1, total_pages]: người dùng đổi filter khi đang ở trang 5 có
    # thể còn 2 trang, khi đó hiển thị trang cuối chứ không trả bảng trắng.
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
        and (
            history_filters["dataset"] == "all"
            or row.get("dataset_fingerprint") == fingerprint["hash"]
        )
    ]
    history_counts = Counter(row.get("model_key") for row in counted_history)
    # Dựng từ `history` gốc, KHÔNG từ history_page["rows"]: biểu đồ phải thấy đủ
    # mọi lần thử theo thời gian, còn history_page đã bị lọc + phân trang theo ý
    # người dùng.
    history_trend = _build_history_trend(
        history,
        active_history_model,
        fingerprint["hash"],
        selected.get("run_id"),
    )

    # Khi bấm sang tab model khác, chỉ mang theo các filter KHÔNG phụ thuộc schema
    # (dataset/status/F1/best/selected/sort). Các filter p_<field> bị bỏ vì mỗi
    # model có bộ tham số riêng: mang p_learning_rate sang tab Random Forest sẽ
    # lọc trên một field không tồn tại → bảng trắng không rõ nguyên nhân.
    common_args = {
        key: value
        for key, value in history_filters["query_args"].items()
        if key in {"dataset", "status", "f1_min", "f1_max", "best", "selected", "sort"}
    }
    # Cùng lý do: sort theo cột tham số cũng vô nghĩa ở model khác → hạ về mặc định.
    if str(common_args.get("sort", "")).startswith("param_"):
        common_args["sort"] = HISTORY_DEFAULT_SORT
    history_tabs = []
    for model_key in MODEL_KEY.values():
        tab_args = {**common_args, "history_model": model_key}
        if model_key == active_history_model:
            # Tab đang mở giữ nguyên toàn bộ filter hiện tại (kể cả p_<field>) để
            # bấm lại chính nó không làm mất bộ lọc; chỉ bỏ page cho về trang 1.
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

    # Trạng thái sắp xếp cho từng cột header: xoay vòng tăng → giảm → mặc định.
    # Mặc định là token HISTORY_DEFAULT_SORT, không phải time_desc, để cột "Thời
    # điểm" cũng có đủ ba trạng thái. Sort đổi thì về trang 1 vì trang cũ không
    # còn ứng với tập dòng mới.
    active_sort = history_filters["sort"]

    def sort_state(column: str) -> dict:
        """Trạng thái hiện tại + link "bấm tiếp" cho một cột header.

        ``current`` là giá trị cho attribute ``aria-sort`` của <th> (chuỗi rỗng =
        cột này không đang sắp), nên template không phải tự suy ra.

        Vòng xoay ba bước khi bấm liên tiếp vào cùng một cột:
        chưa sắp → tăng → giảm → về mặc định (bỏ sắp). Không có bước "về tăng"
        vì lần bấm thứ ba phải cho người dùng đường quay lại thứ tự gốc.

        ``page`` bị loại khỏi query khi đổi sort: tập dòng sắp lại nên trang 5 cũ
        không còn tương ứng với dữ liệu gì — luôn về trang 1.
        """
        current = ""
        if active_sort == f"{column}_asc":
            current = "ascending"
        elif active_sort == f"{column}_desc":
            current = "descending"
        next_sort = {
            "": f"{column}_asc",
            "ascending": f"{column}_desc",
            "descending": HISTORY_DEFAULT_SORT,
        }[current]
        args = {
            key: value
            for key, value in history_filters["query_args"].items()
            if key != "page"
        }
        args["sort"] = next_sort
        return {
            "current": current,
            "next": next_sort,
            "url": url_for("tuning", **args),
        }

    history_sort = {
        column: sort_state(column) for column in HISTORY_SORT_COLUMNS
    }
    # Chỉ cột tham số kiểu số mới sắp được; loại choice (solver) vì thứ tự chuỗi
    # tùy chọn không có nghĩa. Danh sách phải khớp allowed_sorts ở _parse_history_query.
    for field, spec in param_schema[active_history_model].items():
        if spec.get("type") in HISTORY_SORTABLE_PARAM_TYPES:
            history_sort[f"param_{field}"] = sort_state(f"param_{field}")

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
    # Job state nằm trong biến module (RAM), không gắn với dataset. Nếu người dùng
    # refresh dữ liệu sau khi job xong, kết quả cũ vẫn còn trong RAM nhưng đã thuộc
    # fingerprint khác — hiển thị tiếp sẽ khiến người đọc tưởng vừa CV trên dữ liệu
    # mới. Nên coi như "idle" thay vì xóa state (giữ nguyên để tránh race với
    # thread đang chạy).
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
    # Điều kiện bật nút "Chạy official pipeline". Năm điều kiện, mỗi cái chặn một
    # tình huống khác nhau:
    # - complete: đã chốt đủ cấu hình cho cả 3 model trên dataset hiện tại.
    # - not snapshot_evaluated: snapshot này CHƯA từng đánh giá TEST. Đây là khóa
    #   một-lần của giao thức: mỗi snapshot dữ liệu chỉ được chạm TEST đúng một
    #   lần, nếu không việc chọn model sẽ dần "học" TEST qua nhiều lần thử.
    # - ba cờ còn lại: không có job nào (pipeline / fetch / tuning CV) đang chạy,
    #   vì tất cả đều ghi vào cùng bộ file trong experiments/ và reports/.
    can_run = (
        complete
        and not snapshot_evaluated
        and not pipeline_running
        and not fetch_running
        and not tuning_running
    )

    # config_stale = "đã chốt cấu hình nhưng nó thuộc dữ liệu/policy khác".
    # Chỉ cảnh báo khi thực sự có cấu hình đã chốt (bool(selected_models)), chứ
    # người dùng mới vào lần đầu chưa chốt gì thì không phải là "cũ".
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
        "history_sort": history_sort,
        "history_pagination": history_page,
        "history_trend": history_trend,
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

    # Pipeline chạy ĐỒNG BỘ (subprocess.run + check=True): request HTTP bị treo
    # đến khi train xong. Khác hẳn fetch bên dưới (Popen nền). Lý do: pipeline
    # phải xong mới có report để redirect sang /evaluation, và pipeline.lock do
    # chính run_pipeline.py tự giữ nên không cần app quản lý PID.
    # `check=True` biến exit code != 0 thành CalledProcessError → nhánh except
    # dưới vẫn ghi được log stdout/stderr (log là thứ duy nhất user thấy khi lỗi).
    # encoding="utf-8" + errors="replace" vì log tiếng Việt trên Windows dễ vỡ
    # codec; replace đảm bảo không bao giờ crash chỉ vì một byte lạ.
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
    """Trang theo dõi job refresh dữ liệu — trạng thái suy từ ĐĨA, không từ session.

    Job refresh chạy bằng ``Popen`` nền nên tiến trình Flask xử lý request này có
    thể không phải tiến trình đã khởi động job (reload, nhiều worker). Vì vậy mọi
    thứ đọc lại từ đĩa mỗi lần F5:

    - ``is_fetch_running()``: đọc ``fetch.lock`` rồi kiểm PID còn sống thật, nên
      lock mồ côi (máy tắt giữa job) không làm trang treo ở "đang chạy" mãi.
    - ``read_fetch_log_tail()``: chỉ lấy phần cuối log, tránh nạp cả file vào RAM.
    - ``_infer_refresh_progress(log, running)``: parse marker ``[1/3]``…``[3/3]``
      trong log để ra phần trăm. Truyền cả ``running`` vào vì cùng một log "đã
      thấy [3/3]" có nghĩa khác nhau: còn chạy = đang ở bước 3, đã dừng = xong.
    """
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
