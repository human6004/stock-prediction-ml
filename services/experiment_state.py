"""Quản lý toàn bộ trạng thái thí nghiệm trong thư mục ``experiments/``.

Đọc nhanh:
- fingerprint: dấu vân tay nhận diện đúng phiên bản dữ liệu/config.
- tuning history: sổ mọi lần chạy CV thủ công, kể cả lần lỗi.
- manual config: cấu hình người dùng đã chốt cho official pipeline.
- evaluation registry: ghi snapshot đã mở TEST để chặn đánh giá lặp.
- pipeline/fetch lock: ngăn hai tiến trình cùng ghi dữ liệu hoặc artifact.
- log: kết quả lần chạy pipeline/fetch gần nhất.

Module này không train model; nó là sổ trạng thái và các cổng kiểm tra an toàn.
"""

import csv
import hashlib
import json
import math
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    CV_GAP_SESSIONS,
    CV_N_SPLITS,
    CV_START_DATE,
    DECISION_THRESHOLD,
    DATA_PROTOCOL_ID,
    EVALUATION_REGISTRY_PATH,
    EXPERIMENTS_DIR,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FETCH_LOCK_PATH,
    LAST_FETCH_RUN_LOG,
    LAST_PIPELINE_RUN_LOG,
    MANUAL_CONFIG_PATH,
    MANUAL_CONFIG_SCHEMA_VERSION,
    ML_DATA_PATH,
    PIPELINE_LOCK_PATH,
    PREDICTION_HORIZON,
    TUNING_HISTORY_PATH,
    THRESHOLD_MAX_PREDICTED_UP_RATIO,
    UP_THRESHOLD,
)
from services.protocol_dates import resolve_protocol_dates

HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "model_key",
    "model_name",
    "params_json",
    "cv_f1_up_mean",
    "cv_f1_up_std",
    "cv_f1_up_folds_json",
    "cv_precision_up_mean",
    "cv_precision_up_std",
    "cv_recall_up_mean",
    "cv_recall_up_std",
    "cv_precision_up_folds_json",
    "cv_recall_up_folds_json",
    "cv_fold_ranges_json",
    "decision_threshold",
    "oof_f1_up",
    "oof_precision_up",
    "oof_recall_up",
    "oof_up_rate",
    "oof_predicted_up_ratio",
    "threshold_constraint_passed",
    "train_seconds",
    "dataset_fingerprint",
    "content_fingerprint",
    "policy_id",
    "status",
    "note",
]

REQUIRED_MODEL_KEYS = ("logistic_regression", "random_forest", "gradient_boosting")

_HISTORY_LOCK = threading.Lock()
_FINGERPRINT_LOCK = threading.Lock()
_FINGERPRINT_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def ensure_experiments_dir() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Dataset fingerprint
# --------------------------------------------------------------------------- #
def _fingerprint_scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Lọc dataset xuống đúng phần dữ liệu mà một loại fingerprint được phép thấy.

    Có HAI scope vì hai fingerprint trả lời hai câu hỏi khác nhau:

    - ``"train"`` → dùng cho ``compute_dataset_fingerprint`` (tuning fingerprint).
      Chỉ giữ row có ``label_end_date <= train_end``. Mọi cấu hình CV chọn ở
      Tuning Lab chỉ được gắn với PHẦN TRAIN; nếu scope này bao gồm cả
      VALIDATION/TEST thì mỗi lần dữ liệu mới về là fingerprint đổi và toàn bộ
      lịch sử tuning bị vô hiệu dù phần train không hề khác.
    - ``"experiment"`` → dùng cho ``compute_experiment_fingerprint``, định danh
      snapshot đầy đủ TRAIN+VALIDATION+TEST. Đây là khóa của registry "TEST đã
      mở chưa", nên bắt buộc phải đổi khi bất kỳ split nào đổi.

    Ba mask cố ý dùng cột KHÁC nhau, không phải nhầm lẫn:
    - TRAIN lọc theo ``label_end_date`` (nhãn phải kết thúc trước mốc train, đây
      chính là purge chống rò rỉ).
    - VALIDATION lọc theo ``trading_date > train_end`` (row phải nằm sau train) VÀ
      ``label_end_date <= validation_end`` (nhãn không được vượt sang TEST).
    - TEST chỉ lọc theo ``trading_date`` trong khoảng, vì nhãn của TEST là phần
      tương lai xa nhất, không có split nào phía sau để rò rỉ vào.

    Frame thiếu ``trading_date``/``label_end_date`` được trả về nguyên vẹn: đó là
    dataset legacy chưa có cột thời gian, hash toàn bộ vẫn tất định.
    """
    if scope not in {"train", "experiment"}:
        raise ValueError(f"Unsupported fingerprint scope: {scope}")
    if not {"trading_date", "label_end_date"}.issubset(frame.columns):
        return frame

    trading_dates = pd.to_datetime(frame["trading_date"], errors="raise")
    label_end_dates = pd.to_datetime(frame["label_end_date"], errors="raise")
    # Mốc suy động từ chính frame (full dataset) qua cùng một nguồn với
    # protocol_time_split, nên fingerprint và split thật không lệch nhau.
    dates = resolve_protocol_dates(frame)
    train_end = pd.Timestamp(dates["train_end_date"])
    validation_end = pd.Timestamp(dates["validation_end_date"])
    test_end = pd.Timestamp(dates["test_end_date"])
    train_mask = label_end_dates <= train_end
    if scope == "train":
        return frame.loc[train_mask]

    validation_mask = (trading_dates > train_end) & (
        label_end_dates <= validation_end
    )
    test_mask = (trading_dates > validation_end) & (trading_dates <= test_end)
    return frame.loc[train_mask | validation_mask | test_mask]


def compute_dataset_fingerprint(
    ml_dataset_df: pd.DataFrame | None = None, *, scope: str = "train"
) -> dict:
    """Return an order-independent fingerprint for TRAIN or a rolling snapshot."""
    global _FINGERPRINT_CACHE
    if ml_dataset_df is None:
        path = Path(ML_DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(
                "Chua co data/processed/ml_dataset.csv. Hay chay buoc chuan bi du "
                "lieu (preprocess_data.py + build_features.py) hoac chay pipeline truoc."
            )
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        with _FINGERPRINT_LOCK:
            cached = _FINGERPRINT_CACHE.get(scope)
            if cached and cached[0] == signature:
                return cached[1]
        ml_dataset_df = pd.read_csv(path)

    source_rows = int(len(ml_dataset_df))
    scoped_df = _fingerprint_scope_frame(ml_dataset_df, scope)
    # "Order-independent": hash phải chỉ phụ thuộc NỘI DUNG, không phụ thuộc thứ
    # tự cột hay thứ tự dòng. Nếu không, chỉ cần sort lại dataset là fingerprint
    # đổi → manual_config bị coi là "thuộc dataset khác" và Tuning Lab bắt tuning
    # lại dù dữ liệu y nguyên. Cách chuẩn hóa gồm 3 bước bên dưới.
    # (1) Cột: sắp theo tên rồi reindex, nên thứ tự cột trong CSV không ảnh hưởng.
    columns = sorted(str(column) for column in scoped_df.columns)
    normalized = scoped_df.loc[:, columns].copy()
    schema = json.dumps(
        [(column, str(normalized[column].dtype)) for column in columns],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    # (2) Schema (tên cột + dtype) vào hash trước dữ liệu: đổi dtype của một cột
    # là thay đổi thật, phải làm fingerprint khác dù giá trị in ra giống nhau.
    content_hasher = hashlib.sha256(schema.encode("utf-8"))
    # (3) Dòng: băm từng dòng thành uint64 rồi SORT mảng hash. Sort hash (không
    # sort DataFrame) vừa rẻ vừa bỏ hoàn toàn thứ tự dòng khỏi kết quả.
    # index=False để index của pandas không lọt vào hash.
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=True
    ).to_numpy(copy=True)
    row_hashes.sort()
    content_hasher.update(row_hashes.tobytes())
    content_hash = content_hasher.hexdigest()

    if "trading_date" in scoped_df.columns and len(scoped_df):
        max_trading_date = str(scoped_df["trading_date"].max())
    else:
        max_trading_date = None

    # Mốc rolling suy từ full dataset (cùng nguồn với split thật). Nhúng vào
    # parts để identity đổi theo mốc mỗi khi dataset có phiên mới.
    resolved_dates = resolve_protocol_dates(ml_dataset_df)

    parts = {
        "fingerprint_scope": scope,
        "data_protocol_id": DATA_PROTOCOL_ID,
        "dataset_rows": int(len(scoped_df)),
        "max_trading_date": max_trading_date,
        "split_date": resolved_dates["train_end_date"],
        "validation_end_date": resolved_dates["validation_end_date"],
        "test_end_date": resolved_dates["test_end_date"],
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": list(FEATURE_COLUMNS),
        "content_sha256": content_hash,
    }
    identity = hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    result = {
        "hash": identity[:12],
        "content_hash": content_hash,
        "parts": parts,
        "source_dataset_rows": source_rows,
    }
    if "signature" in locals():
        with _FINGERPRINT_LOCK:
            _FINGERPRINT_CACHE[scope] = (signature, result)
    return result


def compute_experiment_fingerprint(
    ml_dataset_df: pd.DataFrame | None = None,
) -> dict:
    """Fingerprint the resolved rolling TRAIN+VALIDATION+TEST snapshot."""
    return compute_dataset_fingerprint(ml_dataset_df, scope="experiment")


# --------------------------------------------------------------------------- #
# Tuning history
# --------------------------------------------------------------------------- #
def canonical_params_json(params: dict) -> str:
    return json.dumps(params or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _rewrite_history_with_current_schema() -> None:
    """Nâng cấp header CSV lịch sử sang ``HISTORY_COLUMNS`` hiện hành.

    Vì sao cần: ``append_history`` ghi bằng ``csv.DictWriter`` với fieldnames là
    schema MỚI. Nếu file cũ có ít cột hơn (bản trước thêm cột mới), append thẳng
    sẽ tạo file lệch cột — dòng cũ đọc theo header cũ, dòng mới theo header mới,
    và ``read_history`` sẽ đọc sai giá trị sang cột khác.

    Cách xử lý: đọc toàn bộ dòng cũ, ghi lại đủ ``HISTORY_COLUMNS`` (thiếu thì
    để rỗng), gán ``policy_id="legacy"`` cho dòng chưa có để chúng bị loại ở
    ``_eligible_policy_rows`` thay vì lẫn vào ranking. Ghi ra ``.tmp`` rồi
    ``replace`` — atomic, không mất lịch sử nếu crash giữa đường.

    Không làm gì khi header đã đúng (đường nhanh, gọi trước mỗi lần append).
    """
    path = Path(TUNING_HISTORY_PATH)
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == HISTORY_COLUMNS:
            return
        prior_rows = list(reader)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for raw in prior_rows:
            raw["policy_id"] = raw.get("policy_id") or "legacy"
            writer.writerow({column: raw.get(column, "") for column in HISTORY_COLUMNS})
    temp_path.replace(path)


def append_history(record: dict) -> None:
    """Append one training run to the history CSV (creates header if needed)."""
    ensure_experiments_dir()
    with _HISTORY_LOCK:
        _rewrite_history_with_current_schema()
        file_exists = Path(TUNING_HISTORY_PATH).exists()
        enriched = {**record}
        enriched.setdefault("policy_id", EXPERIMENT_POLICY_ID)
        row = {col: enriched.get(col, "") for col in HISTORY_COLUMNS}
        with open(TUNING_HISTORY_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def read_history() -> list[dict]:
    """Đọc `tuning_history.csv` và dựng lại kiểu dữ liệu đã bị CSV làm phẳng.

    CSV chỉ lưu text, nên mọi thứ đọc ra đều là str. Hàm này đảo ngược quá trình
    ghi ở `append_history_row`:
    - Cột số → `_to_float` (trả None nếu rỗng/không parse được, KHÔNG phải 0.0;
      phân biệt "chưa đo" với "đo được 0" là bắt buộc cho `_ranking_key`).
    - `threshold_constraint_passed` → bool qua whitelist {"1","true","yes"};
      bất kỳ giá trị lạ nào đều thành False (fail an toàn: coi như chưa đạt
      ràng buộc threshold).
    - Các cột `*_json` → list/dict qua `json.loads`, bọc try/except để một dòng
      hỏng không làm sập cả bảng lịch sử: fallback về `{}`/`[]` rồi vẫn giữ dòng
      đó. Lịch sử tuning là dữ liệu chỉ-đọc dùng cho UI, mất một dòng còn tệ hơn
      hiển thị dòng thiếu chi tiết fold.
    - `policy_id` rỗng → "legacy": row cũ ghi trước khi có policy id, phải đánh
      dấu để `_eligible_policy_rows` loại chúng ra khỏi việc chọn config.
    - `is_best` luôn khởi tạo False; cờ này do `get_best_runs`/tầng UI gán sau
      chứ không nằm trong file.
    """
    if not Path(TUNING_HISTORY_PATH).exists():
        return []
    rows: list[dict] = []
    with open(TUNING_HISTORY_PATH, "r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["cv_f1_up_mean"] = _to_float(raw.get("cv_f1_up_mean"))
            row["cv_f1_up_std"] = _to_float(raw.get("cv_f1_up_std"))
            row["cv_precision_up_mean"] = _to_float(raw.get("cv_precision_up_mean"))
            row["cv_precision_up_std"] = _to_float(raw.get("cv_precision_up_std"))
            row["cv_recall_up_mean"] = _to_float(raw.get("cv_recall_up_mean"))
            row["cv_recall_up_std"] = _to_float(raw.get("cv_recall_up_std"))
            row["decision_threshold"] = _to_float(raw.get("decision_threshold"))
            for field in (
                "oof_f1_up",
                "oof_precision_up",
                "oof_recall_up",
                "oof_up_rate",
                "oof_predicted_up_ratio",
            ):
                row[field] = _to_float(raw.get(field))
            row["threshold_constraint_passed"] = str(
                raw.get("threshold_constraint_passed") or ""
            ).lower() in {"1", "true", "yes"}
            row["train_seconds"] = _to_float(raw.get("train_seconds"))
            try:
                row["params"] = json.loads(raw.get("params_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["params"] = {}
            try:
                row["folds"] = json.loads(raw.get("cv_f1_up_folds_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["folds"] = []
            for source, destination in (
                ("cv_precision_up_folds_json", "precision_folds"),
                ("cv_recall_up_folds_json", "recall_folds"),
                ("cv_fold_ranges_json", "fold_ranges"),
            ):
                try:
                    row[destination] = json.loads(raw.get(source) or "[]")
                except (json.JSONDecodeError, TypeError):
                    row[destination] = []
            row["policy_id"] = raw.get("policy_id") or "legacy"
            row["is_best"] = False
            rows.append(row)
    return rows


def _ranking_key(row: dict) -> tuple:
    """Khóa sắp xếp để chọn run tốt nhất — dùng với ``min()``, không phải ``max()``.

    Mọi thành phần đều theo hướng "nhỏ hơn = tốt hơn":
    - ``-cv_f1_up_mean``: đảo dấu F1 nên F1 cao thành số nhỏ → thắng.
    - ``cv_f1_up_std``: cùng F1 thì ưu tiên độ lệch giữa các fold nhỏ hơn (ổn định
      hơn). ``None`` → ``inf`` để run thiếu std luôn xếp sau.
    - ``run_id``: tie-break từ vựng, chỉ để kết quả tất định (cùng input → cùng
      run thắng), không mang ý nghĩa chất lượng.
    """
    std = row.get("cv_f1_up_std")
    return (
        -float(row["cv_f1_up_mean"]),
        float("inf") if std is None else float(std),
        str(row.get("run_id") or ""),
    )


def _eligible_policy_rows(
    rows: list[dict], fingerprint_hash: str | None = None
) -> list[dict]:
    """Lọc ra các run được phép đưa vào ranking / chọn làm config chính thức.

    Đây là cửa chất lượng duy nhất của lịch sử tuning: mọi hàm chọn "best" đều
    phải đi qua đây. Một run bị loại nếu vướng bất kỳ điều kiện nào sau đây:

    1. Không ``status == "ok"``, hoặc thiếu/NaN/inf mean-std → số liệu không dùng được.
    2. ``policy_id`` khác policy hiện hành → run của luật thí nghiệm cũ (đổi
       feature set, đổi horizon, ...) không so sánh được với run mới.
    3. ``dataset_fingerprint`` khác (khi caller truyền ``fingerprint_hash``) → run
       của phiên bản dữ liệu khác; F1 giữa hai dataset khác nhau là hai thang đo.
    4. Không có ``content_fingerprint`` → run cũ trước khi thêm trường này, không
       chứng minh được đã chạy trên nội dung dữ liệu nào.
    5. Threshold ngoài (0,1) hoặc cờ ``threshold_constraint_passed`` sai.
    6. Vi phạm lại chính ràng buộc threshold khi kiểm tra bằng số đã lưu:
       ``precision >= oof_up_rate`` và ``predicted_up_ratio <= max``. Tính lại ở
       đây để một dòng CSV bị sửa tay hoặc ghi lỗi vẫn không lọt qua.
       ``1e-12`` là biên sai số float, tránh loại oan khi hai số gần bằng nhau.
    7. Số fold trong các mảng JSON không đúng ``CV_N_SPLITS`` → provenance không
       đầy đủ; pipeline sau này cần đúng số fold để tái dựng CV (xem
       ``model_tuning._cv_from_selected``).
    """
    eligible = []
    for row in rows:
        # _to_float thay vì float(): read_history() đã ép số cho mọi dòng đọc từ
        # CSV, nhưng hàm này cũng nhận dict dựng trực tiếp (test, caller khác).
        # Một giá trị chuỗi như "n/a" từng làm float() nổ ValueError giữa vòng lặp
        # → cả trang /tuning 500 chỉ vì một dòng lịch sử bẩn. Giá trị không đọc
        # được nay bị loại như thiếu số, đúng ý điều kiện 1 ở docstring.
        mean = _to_float(row.get("cv_f1_up_mean"))
        std = _to_float(row.get("cv_f1_up_std"))
        if row.get("status") != "ok" or mean is None or std is None:
            continue
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            continue
        if row.get("policy_id") != EXPERIMENT_POLICY_ID:
            continue
        if fingerprint_hash and row.get("dataset_fingerprint") != fingerprint_hash:
            continue
        if not row.get("content_fingerprint"):
            continue
        threshold = _to_float(row.get("decision_threshold"))
        up_rate = _to_float(row.get("oof_up_rate"))
        precision = _to_float(row.get("oof_precision_up"))
        predicted_ratio = _to_float(row.get("oof_predicted_up_ratio"))
        if threshold is None or not 0 < threshold < 1:
            continue
        if not row.get("threshold_constraint_passed"):
            continue
        if None in (up_rate, precision, predicted_ratio):
            continue
        if precision + 1e-12 < up_rate:
            continue
        if predicted_ratio > THRESHOLD_MAX_PREDICTED_UP_RATIO + 1e-12:
            continue
        if any(
            len(row.get(field) or []) != CV_N_SPLITS
            for field in ("folds", "precision_folds", "recall_folds", "fold_ranges")
        ):
            continue
        eligible.append(row)
    return eligible


def mark_best(rows: list[dict]) -> list[dict]:
    """Gắn cờ ``is_best`` cho run tốt nhất của từng (model, dataset fingerprint).

    Nhóm theo cả fingerprint chứ không chỉ theo model: mỗi phiên bản dữ liệu có
    "nhà vô địch" riêng, nên bảng lịch sử vẫn thấy được best của các dataset cũ.
    Run legacy / không đủ điều kiện không bao giờ được gắn cờ (bị
    ``_eligible_policy_rows`` chặn trước).

    Hàm sửa ``rows`` tại chỗ rồi trả lại chính list đó — template dùng luôn.
    """
    for row in rows:
        row["is_best"] = False
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in _eligible_policy_rows(rows):
        key = (str(row.get("model_key")), str(row.get("dataset_fingerprint")))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        min(group, key=_ranking_key)["is_best"] = True
    return rows


def get_best_runs(rows: list[dict], fingerprint_hash: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in _eligible_policy_rows(rows, fingerprint_hash):
        grouped.setdefault(str(row.get("model_key")), []).append(row)
    return {model_key: min(group, key=_ranking_key) for model_key, group in grouped.items()}


def get_tuning_progress(rows: list[dict], fingerprint_hash: str) -> dict:
    """Tóm tắt "đã tune đủ chưa" cho từng model, dùng để mở/khoá nút chạy pipeline.

    Chỉ đếm run hợp lệ với dataset hiện tại (`_eligible_policy_rows` lọc theo
    `policy_id` + `dataset_fingerprint`), nên khi dữ liệu mới về thì fingerprint
    đổi và tiến độ tự reset về 0 — đúng ý: tham số tune trên dataset cũ không
    còn bảo đảm gì trên dataset mới.

    `valid_config_count` đếm số CẤU HÌNH THAM SỐ KHÁC NHAU, không phải số run:
    khoá dedupe là `canonical_params_json` (sắp key + chuẩn hoá số) nên chạy lại
    cùng một bộ tham số 5 lần vẫn tính là 1. Ngược lại `ready` chỉ cần ≥1 config,
    còn `complete` yêu cầu CẢ `REQUIRED_MODEL_KEYS` đều ready — pipeline so sánh
    LR/RF/GB nên thiếu một model là không chạy được.
    """
    eligible = _eligible_policy_rows(rows, fingerprint_hash)
    best = get_best_runs(eligible, fingerprint_hash)
    models = {}
    for model_key in REQUIRED_MODEL_KEYS:
        configs = {}
        for row in eligible:
            if row.get("model_key") == model_key:
                configs.setdefault(
                    canonical_params_json(row.get("params") or {}),
                    dict(row.get("params") or {}),
                )
        models[model_key] = {
            "valid_config_count": len(configs),
            "ready": bool(configs),
            "best_run": best.get(model_key),
        }
    return {
        "policy_id": EXPERIMENT_POLICY_ID,
        "models": models,
        "complete": all(item["ready"] for item in models.values()),
    }


def find_run(run_id: str) -> dict | None:
    if not run_id:
        return None
    for row in read_history():
        if row.get("run_id") == run_id:
            return row
    return None


# --------------------------------------------------------------------------- #
# Manual config
# --------------------------------------------------------------------------- #
def _default_manual_config() -> dict:
    """Khung cấu hình thủ công rỗng, đã chốt sẵn phần CV/threshold.

    ``selected_models`` để rỗng là đúng: pipeline KHÔNG tự chọn tham số, phải do
    người dùng chốt từng model ở Tuning Lab (xem ``is_config_complete``).

    Các giá trị còn lại chụp lại từ ``config/settings.py`` NGAY LÚC tạo config, để
    file config tự mô tả được điều kiện CV đã dùng. Nhờ vậy nếu sau này ai đổi
    ``CV_N_SPLITS``/``CV_GAP_SESSIONS``/``CV_START_DATE`` trong settings thì
    ``is_config_complete`` sẽ so ra khác và coi config cũ là hết hạn, thay vì âm
    thầm train với điều kiện khác lúc tuning.
    """
    return {
        "schema_version": MANUAL_CONFIG_SCHEMA_VERSION,
        "policy_id": EXPERIMENT_POLICY_ID,
        "dataset_fingerprint": "",
        "dataset_fingerprint_parts": {},
        "cv_settings": {
            "n_splits": CV_N_SPLITS,
            "gap_sessions": CV_GAP_SESSIONS,
            "start_date": CV_START_DATE,
            "metric": "f1_up",
            "threshold_policy": {
                "max_predicted_up_ratio": THRESHOLD_MAX_PREDICTED_UP_RATIO,
                "precision_floor": "oof_up_rate",
            },
        },
        "selected_models": {},
    }


def _empty_manual_config(fingerprint: dict) -> dict:
    config = _default_manual_config()
    config["dataset_fingerprint"] = fingerprint["hash"]
    config["content_fingerprint"] = fingerprint.get("content_hash", fingerprint["hash"])
    config["dataset_fingerprint_parts"] = fingerprint["parts"]
    return config


def read_manual_config() -> dict:
    """Always return a config dict (defaults when the file is missing/invalid)."""
    if not Path(MANUAL_CONFIG_PATH).exists():
        return _default_manual_config()
    try:
        data = json.loads(Path(MANUAL_CONFIG_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_manual_config()
    # Backfill any missing top-level keys for template safety.
    base = _default_manual_config()
    base.update(data)
    base.setdefault("selected_models", {})
    base.setdefault(
        "cv_settings",
        {
            "n_splits": CV_N_SPLITS,
            "gap_sessions": CV_GAP_SESSIONS,
            "start_date": CV_START_DATE,
            "metric": "f1_up",
            "threshold_policy": {
                "max_predicted_up_ratio": THRESHOLD_MAX_PREDICTED_UP_RATIO,
                "precision_floor": "oof_up_rate",
            },
        },
    )
    return base


def _write_manual_config(config: dict) -> dict:
    ensure_experiments_dir()
    path = Path(MANUAL_CONFIG_PATH)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return config


def save_selected_model(
    model_key: str,
    run_id: str,
    params: dict,
    cv_f1_up_mean: float | None,
    fingerprint: dict,
) -> dict:
    """Persist the valid CV run explicitly chosen by the user."""
    run = next(
        (
            row
            for row in _eligible_policy_rows(read_history(), fingerprint["hash"])
            if row.get("model_key") == model_key and row.get("run_id") == run_id
        ),
        None,
    )
    if run is None:
        raise ValueError("Run không hợp lệ cho model và dataset hiện tại.")
    if canonical_params_json(run.get("params") or {}) != canonical_params_json(params):
        raise ValueError("Tham số không khớp với run đã chọn.")

    config = read_manual_config()
    if (
        config.get("schema_version") != MANUAL_CONFIG_SCHEMA_VERSION
        or config.get("policy_id") != EXPERIMENT_POLICY_ID
        or config.get("dataset_fingerprint") != fingerprint["hash"]
    ):
        config = _empty_manual_config(fingerprint)
    else:
        config = {**config, "selected_models": dict(config.get("selected_models") or {})}
    config["selected_models"][model_key] = {
        "run_id": run_id,
        "params": dict(run.get("params") or {}),
        "cv_f1_up_mean": run.get("cv_f1_up_mean", cv_f1_up_mean),
        "cv_f1_up_std": run.get("cv_f1_up_std"),
        "cv_precision_up_mean": run.get("cv_precision_up_mean"),
        "cv_recall_up_mean": run.get("cv_recall_up_mean"),
        "f1_up_folds": list(run.get("folds") or []),
        "precision_up_folds": list(run.get("precision_folds") or []),
        "recall_up_folds": list(run.get("recall_folds") or []),
        "fold_date_ranges": list(run.get("fold_ranges") or []),
        "decision_threshold": run.get("decision_threshold"),
        "oof_f1_up": run.get("oof_f1_up"),
        "oof_precision_up": run.get("oof_precision_up"),
        "oof_recall_up": run.get("oof_recall_up"),
        "oof_up_rate": run.get("oof_up_rate"),
        "oof_predicted_up_ratio": run.get("oof_predicted_up_ratio"),
        "threshold_constraint_passed": run.get("threshold_constraint_passed"),
        "selected_at": _now_iso(),
        "selection_method": "manual",
    }
    return _write_manual_config(config)


def is_config_complete(
    config: dict | None,
    current_fingerprint_hash: str,
    rows: list[dict] | None = None,
) -> bool:
    """Kiểm tra manual config còn dùng được không — mọi cửa ải fail đều trả False.

    Đây là hàng rào chống chạy pipeline bằng cấu hình đã "mục". Thứ tự kiểm tra
    từ rẻ đến đắt (đọc file lịch sử để cuối cùng):
    1. `policy_id` + `schema_version`: config ghi bởi phiên bản luật khác thì các
       field bên trong có thể mang nghĩa khác → loại thẳng, không cố migrate.
    2. `cv_settings` phải khớp ĐÚNG hằng số CV đang hiệu lực (n_splits, gap,
       start_date). Nếu code đổi gap mà config vẫn giữ gap cũ thì điểm CV đã
       chọn tham số không còn so sánh được.
    3. `dataset_fingerprint` phải khớp dataset hiện tại — tham số tune trên dữ
       liệu khác coi như vô giá trị.
    4. Với TỪNG model bắt buộc: phải là `selection_method == "manual"` (config
       này là lựa chọn tay, không nhận run auto), `run_id` phải còn tồn tại
       trong lịch sử đủ điều kiện, và tham số phải khớp qua
       `canonical_params_json` (so chuỗi chuẩn hoá thay vì so dict thô, tránh
       khác biệt do thứ tự key hay 5 vs 5.0), cuối cùng `decision_threshold`
       phải trùng với threshold của chính run đó sau khi cùng đi qua `_to_float`
       (so số, không so text — "0.50" và "0.5" là một).

    Bất kỳ lệch nào → False, UI buộc người dùng chọn lại. Không tự sửa config vì
    đoán sai ở đây nghĩa là chạy pipeline bằng tham số không ai kiểm chứng.
    """
    if not config:
        return False
    if config.get("policy_id") != EXPERIMENT_POLICY_ID:
        return False
    if config.get("schema_version") != MANUAL_CONFIG_SCHEMA_VERSION:
        return False
    cv_settings = config.get("cv_settings") or {}
    if (
        cv_settings.get("n_splits") != CV_N_SPLITS
        or cv_settings.get("gap_sessions") != CV_GAP_SESSIONS
        or cv_settings.get("start_date") != CV_START_DATE
    ):
        return False
    if config.get("dataset_fingerprint") != current_fingerprint_hash:
        return False
    eligible = _eligible_policy_rows(
        read_history() if rows is None else rows, current_fingerprint_hash
    )
    selected = config.get("selected_models", {})
    for model_key in REQUIRED_MODEL_KEYS:
        choice = selected.get(model_key) or {}
        if choice.get("selection_method") != "manual":
            return False
        run = next(
            (
                row
                for row in eligible
                if row.get("model_key") == model_key
                and row.get("run_id") == choice.get("run_id")
            ),
            None,
        )
        if run is None or canonical_params_json(run.get("params") or {}) != canonical_params_json(
            choice.get("params") or {}
        ):
            return False
        if _to_float(choice.get("decision_threshold")) != _to_float(
            run.get("decision_threshold")
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Rolling evaluation registry
# --------------------------------------------------------------------------- #
def read_evaluation_registry(*, path: Path = EVALUATION_REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_evaluation_entry(
    entry: dict, *, path: Path = EVALUATION_REGISTRY_PATH
) -> dict:
    """Upsert một entry vào registry đánh giá, khoá theo `experiment_fingerprint`.

    Fingerprint là khoá chính và bắt buộc: không có nó thì không thể biết snapshot
    nào đã mở TEST, nên raise thay vì ghi một entry vô danh.

    Merge kiểu 3 lớp `{**cũ, **mới, ...}`: giữ field cũ chưa được entry mới nhắc
    tới (ví dụ `started_at` từ lượt trước vẫn còn khi lượt này chỉ báo
    `evaluated`), rồi ép lại `policy_id` và `updated_at`. Nhờ vậy `run_pipeline`
    gọi nhiều lần theo vòng đời started → evaluated → published mà không mất dữ
    liệu các bước trước.

    Ghi qua `.tmp` rồi `replace` để registry không bao giờ ở trạng thái JSON dở
    dang — nếu tiến trình chết giữa lúc ghi, file cũ còn nguyên. `indent=2` +
    `ensure_ascii=False` cho người đọc trực tiếp bằng mắt.
    """
    fingerprint = str(entry.get("experiment_fingerprint") or "").strip()
    if not fingerprint:
        raise ValueError("Evaluation entry requires experiment_fingerprint.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = read_evaluation_registry(path=path)
    registry[fingerprint] = {
        **registry.get(fingerprint, {}),
        **entry,
        "policy_id": entry.get("policy_id", EXPERIMENT_POLICY_ID),
        "updated_at": _now_iso(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return registry[fingerprint]


def has_evaluated_snapshot(
    fingerprint_hash: str, *, path: Path = EVALUATION_REGISTRY_PATH
) -> bool:
    entry = read_evaluation_registry(path=path).get(fingerprint_hash) or {}
    return entry.get("status") in {
        "started",
        "evaluated",
        "published",
        "release_failed",
    }


# --------------------------------------------------------------------------- #
# Pipeline lock (concurrency guard)
# --------------------------------------------------------------------------- #
def _pid_alive(pid) -> bool:
    """Kiểm tra tiến trình còn sống, để phát hiện lock mồ côi (stale lock).

    Cần thiết vì tiến trình pipeline có thể bị kill/crash mà không xoá được
    ``pipeline.lock``; nếu chỉ nhìn sự tồn tại của file thì lock đó chặn mọi lần
    chạy sau vĩnh viễn. Có PID còn sống hay không mới là câu trả lời thật.

    Hai nhánh nền tảng:
    - Windows: ``os.kill(pid, 0)`` không có nghĩa "thăm dò" như POSIX, nên phải
      gọi Win32 API. ``OpenProcess`` thất bại → coi là đã chết;
      ``GetExitCodeProcess`` trả ``STILL_ACTIVE`` (259) mới là đang chạy.
      Dùng ``PROCESS_QUERY_LIMITED_INFORMATION`` (quyền tối thiểu) để mở được
      handle cả khi tiến trình chạy ở mức quyền khác.
    - POSIX: ``os.kill(pid, 0)`` chỉ kiểm tra sự tồn tại, không gửi signal thật.

    ``except Exception: return False`` cố ý bao rộng: PID rác, kiểu dữ liệu sai,
    hay bị từ chối quyền đều dẫn tới kết luận an toàn hơn là "không còn chạy" —
    tức cho phép thu hồi lock, thay vì để hệ thống tự khoá chết.
    """
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_pipeline_lock() -> dict | None:
    if not Path(PIPELINE_LOCK_PATH).exists():
        return None
    try:
        return json.loads(Path(PIPELINE_LOCK_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _lock_is_stale(lock: dict) -> bool:
    return not _pid_alive(lock.get("pid"))


def _remove_pipeline_lock_if_matches(expected: dict) -> bool:
    path = Path(PIPELINE_LOCK_PATH)
    current = _read_pipeline_lock()
    if current != expected:
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def is_pipeline_running() -> bool:
    lock = _read_pipeline_lock()
    if not lock:
        return False
    if _lock_is_stale(lock):
        _remove_pipeline_lock_if_matches(lock)
        return False
    return True


def acquire_pipeline_lock() -> str | None:
    """Atomically create pipeline.lock and return its owner token."""
    ensure_experiments_dir()
    path = Path(PIPELINE_LOCK_PATH)
    owner_token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "owner_token": owner_token,
        "started_at": _now_iso(),
    }
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return owner_token
        except FileExistsError:
            current = _read_pipeline_lock()
            if current is None or not _lock_is_stale(current):
                return None
            if not _remove_pipeline_lock_if_matches(current):
                return None
    return None


def release_pipeline_lock(owner_token: str) -> bool:
    """Release only the lock owned by the caller."""
    lock = _read_pipeline_lock()
    if not lock or lock.get("owner_token") != owner_token:
        return False
    return _remove_pipeline_lock_if_matches(lock)


# --------------------------------------------------------------------------- #
# Pipeline run log
# --------------------------------------------------------------------------- #
def write_pipeline_log(text: str) -> None:
    ensure_experiments_dir()
    Path(LAST_PIPELINE_RUN_LOG).write_text(text or "", encoding="utf-8")


def read_pipeline_log_tail(max_lines: int = 200) -> str:
    if not Path(LAST_PIPELINE_RUN_LOG).exists():
        return ""
    try:
        text = Path(LAST_PIPELINE_RUN_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# --------------------------------------------------------------------------- #
# Fetch lock (background data refresh)
# --------------------------------------------------------------------------- #
def _read_fetch_lock() -> dict | None:
    if not Path(FETCH_LOCK_PATH).exists():
        return None
    try:
        return json.loads(Path(FETCH_LOCK_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_fetch_running() -> bool:
    lock = _read_fetch_lock()
    if not lock:
        return False
    if _lock_is_stale(lock):
        release_fetch_lock()
        return False
    return True


def write_fetch_lock(pid: int) -> None:
    ensure_experiments_dir()
    payload = {"pid": int(pid), "started_at": _now_iso()}
    Path(FETCH_LOCK_PATH).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def release_fetch_lock() -> None:
    try:
        Path(FETCH_LOCK_PATH).unlink()
    except FileNotFoundError:
        pass


def read_fetch_log_tail(max_lines: int = 200) -> str:
    if not Path(LAST_FETCH_RUN_LOG).exists():
        return ""
    try:
        text = Path(LAST_FETCH_RUN_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
