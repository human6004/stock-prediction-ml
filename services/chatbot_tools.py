"""Grounded, read-only data handlers for the local stock chatbot."""

from __future__ import annotations

import json
import math
from functools import lru_cache
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


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _summary_model_name(summary: dict | None) -> str | None:
    """Lấy tên model từ pipeline summary, chấp nhận cả hai định dạng report.

    Report mới ghi thẳng ``selection_report.selected_model_name``. Report cũ
    không có khóa đó, chỉ có bảng ``final_test_evaluation`` gồm nhiều dòng
    (final + các baseline) — phải lọc ``row_type == "final"`` mới ra model thật,
    nếu lấy dòng đầu tiên có thể nhặt đúng dòng baseline.
    """
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
    """Bộ ba định danh một release: (model_name, policy_id, content_fingerprint).

    Ba file (artifact .pkl, metadata .json, pipeline summary .json) được ghi ở ba
    thời điểm khác nhau trong pipeline. Chỉ khi CẢ BA cho ra cùng một tuple thì
    mới chắc chúng thuộc một lần chạy; lệch một khóa nghĩa là có file cũ sót lại
    hoặc pipeline chết giữa đường → release "inconsistent" và chatbot phải chặn.

    ``summary=True`` vì file summary lưu tên model ở chỗ khác (xem
    ``_summary_model_name``), không phải khóa ``model_name`` phẳng như hai file kia.
    """
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
    """Dò F1_UP của baseline "Always UP" trên TEST từ metadata, fallback sang summary.

    Cần cả hai nguồn vì artifact cũ có thể thiếu ``final_test_baselines``; khi đó
    ``pipeline_summary.json`` vẫn còn bảng ``final_test_evaluation``. Thứ tự trong
    ``collections`` là thứ tự ưu tiên: metadata thắng vì nó là artifact được publish.

    So tên bằng ``casefold().replace("-", " ")`` để khớp mọi biến thể viết
    ("Always UP", "always-up", "ALWAYS UP") — tên baseline từng đổi qua các phiên
    bản report nên không thể so chuỗi tuyệt đối.

    Trả ``None`` nếu không nguồn nào có row baseline; caller phải xử lý None chứ
    không được coi là 0 (0 sẽ khiến mọi model trông như đã vượt baseline).
    """
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
    """Sinh câu cảnh báo tiếng Việt khi model không thắng baseline Always UP.

    Điều kiện vào là ``is not False`` chứ không phải ``if not ...``: chỉ cảnh báo
    khi biết CHẮC đã fail. Artifact thiếu key (``None``) nghĩa là "không rõ" →
    im lặng, vì bịa cảnh báo cho release cũ cũng sai như bỏ sót cảnh báo thật.

    Nếu thiếu số để so thì vẫn trả cảnh báo dạng ngắn (không kèm số) — người đọc
    cần biết fact "chưa vượt baseline" kể cả khi không hiển thị được con số.

    ``replace(".", ",")`` đổi sang dấu phẩy thập phân kiểu VN. Chuỗi này đi vào
    ``warnings`` của envelope và có thể được LM nhắc lại, nên mọi con số ở đây
    phải nằm trong tập grounded number của phía service.
    """
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
    """Trả về (scope, verified, valid) — tập mã mà release được phép trả lời.

    Vì sao cần scope: model chỉ được train trên các mã đủ điều kiện. Nếu chatbot
    trả tín hiệu cho một mã ngoài tập đó, con số là ngoại suy chứ không phải kết
    quả của model → phải từ chối thay vì trả bừa.

    Hai nguồn, theo thứ tự tin cậy:
    1. ``metadata["training_symbols"]`` + ``training_symbol_count`` — nguồn chuẩn,
       ghi kèm lúc train. Kiểm tra rất chặt: phải là list str không rỗng, không
       trùng lặp, và ``count`` phải KHỚP đúng ``len(normalized)``. Count lệch nghĩa
       là metadata bị sửa tay hoặc ghi dở → trả ``valid=False`` để release thành
       "inconsistent" (KHÔNG âm thầm fallback, vì fallback ở đây sẽ che mất lỗi thật).
       ``isinstance(count, bool)`` bị loại riêng vì trong Python ``True`` cũng là ``int``.
    2. Artifact cũ (legacy) không có hai khóa đó → đọc tạm ``eligible_symbols.csv``,
       trả ``verified=False``. Caller sẽ gắn warning ``symbol_scope_unverified``, và
       nếu artifact khai policy hiện hành thì vẫn bị coi là inconsistent — policy mới
       bắt buộc phải có training_symbols.

    Lỗi đọc CSV trả ``valid=True`` (không phải lỗi định danh release), chỉ là scope
    rỗng → mọi mã sẽ bị từ chối một cách an toàn.
    """
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


@lru_cache(maxsize=1)
def _clean_data_max_date_cached(_signature) -> str | None:
    """Đọc và cache ngày trading_date lớn nhất của cleaned CSV, key là (size, mtime_ns).

    Không nạp cả file — chỉ đọc cột trading_date để tiết kiệm RAM với file 557k dòng.
    Được cache theo file signature: fetch mới → size/mtime đổi → cache tự vô hiệu.
    """
    try:
        df = pd.read_csv(CLEANED_DATA_PATH, usecols=["trading_date"])
        max_date = df["trading_date"].max()
        # CSV rỗng hoặc cột all-NaN → max() trả NaN (truthy string "nan" nếu ép str)
        return None if pd.isna(max_date) else str(max_date)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _clean_data_max_date() -> str | None:
    """Trả ngày trading_date lớn nhất của cleaned CSV (live, không phụ thuộc report).

    Tính (size, mtime_ns) của file trước mỗi lần gọi; nếu file đổi thì cache
    _clean_data_max_date_cached sẽ miss và tự đọc lại. stat() lỗi → trả None.
    """
    try:
        stat = CLEANED_DATA_PATH.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None
    return _clean_data_max_date_cached(signature)


def _release_signature() -> tuple:
    """Dấu vân tay nhẹ của 4 file release: (size, mtime_ns) mỗi file.

    Không hash nội dung vì hàm này được gọi hai lần quanh mỗi lần đọc release
    (trước và sau) để phát hiện pipeline ghi file GIỮA lúc đọc — hash sẽ quá đắt
    cho mục đích đó. File thiếu → ``None`` trong tuple, vẫn so sánh được.
    """
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
    """Read raw release files without prediction_service's mismatch fallback.

    Vì sao KHÔNG dùng ``prediction_service``: service đó có fallback (lệch metadata
    thì vẫn cố suy ra kết quả để UI không trắng trang). Chatbot cần điều ngược lại
    — thà chặn còn hơn nói số của một release không xác định. Nên ở đây đọc file
    thô và tự phân loại ``status``:

    - ``missing``: thiếu một trong artifact/metadata/summary.
    - ``inconsistent``: 3 nguồn không cùng identity (model_name, policy_id,
      content_fingerprint), hoặc identity thiếu field, hoặc scope không hợp lệ,
      hoặc pipeline đang chạy, hoặc signature trước/sau lần đọc khác nhau
      (file bị ghi giữa lúc đọc → dữ liệu vừa đọc là trộn hai release).
    - ``current``: khớp ``EXPERIMENT_POLICY_ID`` hiện hành.
    - ``legacy``: nhất quán nhưng policy cũ → vẫn trả lời được, kèm warning.

    Gate riêng cho policy hiện hành: artifact ``current`` mà không có
    ``training_symbols`` (scope chưa verify được) bị hạ xuống ``inconsistent``,
    trong khi artifact legacy chỉ bị warning ``symbol_scope_unverified``. Lý do:
    policy mới bắt buộc ghi scope, thiếu là dấu hiệu artifact hỏng.

    ``warnings`` trả về dạng list dict {code, message} để UI/LLM đều dùng được.
    """
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

    # Kiểm tra data_ahead_of_report: CSV cleaned mới hơn report snapshot
    clean_max = _clean_data_max_date()
    report_date_max = ((summary or {}).get("dataset_report") or {}).get("date_max")
    if clean_max is not None and report_date_max is not None and clean_max > report_date_max:
        warnings.append({
            "code": "data_ahead_of_report",
            "message": "Dữ liệu cleaned CSV mới hơn pipeline report lần chạy cuối; số đếm dataset trong báo cáo có thể chưa phản ánh đúng.",
        })

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
    """Phần release công khai đi kèm mọi kết quả handler.

    Khi release ``inconsistent`` thì metadata chính là thứ đang lệch, nên lấy
    identity từ ``artifact`` (file model thật) mới phản ánh đúng cái đang nằm trên
    đĩa; các trạng thái khác ưu tiên ``metadata`` vì đó là nguồn mô tả chuẩn.
    Identity này được ``chatbot_service.chat`` dùng để phát hiện release đổi giữa
    lượt trả lời.
    """
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
    """Đóng gói kết quả handler theo hợp đồng 7 khóa mà chatbot_service kiểm.

    Mọi handler trả về qua hàm này (kể cả lỗi, qua ``_error``) để phía
    service luôn thấy đủ ``ok/data/source/as_of/release/warnings/error``; thiếu
    khóa sẽ bị coi là lỗi giao thức.

    Hai trường lấy từ ``state`` chứ không từ caller:
    - ``release``: định danh release đã đọc được lúc chạy handler → service dùng để
      ghim (policy_id, content_fingerprint) và phát hiện release đổi giữa lượt.
    - ``warnings``: copy bằng ``list(...)`` để caller sửa list sau đó không làm
      thay đổi envelope đã trả (envelope phải là snapshot bất biến).
    """
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
    """Cổng chặn chung cho handler đọc số của model. None = được phép chạy tiếp.

    Ba điều kiện chặn, theo thứ tự: release không nhất quán, release thiếu file,
    và scope rỗng (không biết model được train trên mã nào). Trường hợp thứ ba
    cũng trả ``model_missing`` vì với người dùng thì hệ quả giống nhau: không có
    release dùng được.
    """
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
    """Rút gọn 1 dòng dự đoán thành payload an toàn cho LLM.

    Hai việc quan trọng ở đây:
    1. **Server tính sẵn phần so sánh** (``threshold_relation`` above/below/equal
       và ``threshold_gap_percent_points``) thay vì để LLM tự trừ hai số. LLM trừ
       số rất dễ sai, mà lớp kiểm duyệt ``_validate_grounded_answer`` chỉ cho phép
       những số CÓ TRONG context — nên nếu không tính sẵn, câu trả lời hợp lệ
       về nội dung vẫn bị chặn vì con số hiệu không có gốc.
    2. **Làm tròn 10 chữ số trước khi đổi sang %**: ``_rounded_number(..., 10)`` giữ
       giá trị thô để so sánh chính xác, còn ``_percent`` mới là số hiển thị (2 chữ
       số). So sánh ``raw_up_score > raw_threshold`` dùng bản thô để không bị đảo
       kết luận khi hai số chỉ khác nhau sau chữ số thập phân thứ hai.

    ``price_unit: "source_native"`` khai báo rõ giá giữ nguyên đơn vị của nguồn
    (không quy đổi nghìn đồng) để model không tự thêm đơn vị sai.
    """
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
        "indicator_window_sessions": 20,
        "is_stale": bool(row.get("is_stale", False)),
        "horizon_sessions": row.get("horizon")
        or metadata.get("prediction_horizon"),
    }


def get_stock_signals(arguments: dict) -> dict:
    """Handler tín hiệu offline cho 1-2 mã, kèm mọi gate an toàn.

    Thứ tự kiểm tra là có chủ ý — release trước, argument sau: nếu release đã hỏng
    thì báo lỗi release chính xác hơn là báo lỗi input.

    1. ``_blocked_error``: release missing/inconsistent hoặc scope rỗng → chặn.
    2. Validate arguments: đúng key, 1-2 mã, chuỗi không rỗng, không trùng nhau.
    3. ``symbol_out_of_scope``: mã ngoài tập train → từ chối. Model không học mã đó,
       trả số sẽ là ngoại suy.
    4. Gọi ``prediction_service.predict_symbols``, bọc ``except Exception`` để chi
       tiết lỗi nội bộ không rò ra ngoài (chỉ trả ``prediction_unavailable``).
    5. ``_release_changed(state)`` SAU khi dự đoán: pipeline có thể publish release
       mới trong lúc predict đang chạy → kết quả vừa tính thuộc release nào không rõ
       nữa, phải bỏ và yêu cầu hỏi lại.

    ``same_reference_date``: chỉ đặt ``as_of`` khi TẤT CẢ mã cùng ngày tham chiếu.
    Nếu hai mã lệch ngày thì so sánh trực tiếp là sai lệch, nên ``comparison`` chỉ
    được tính khi cùng ngày (nhánh dưới), còn lệch ngày thì trả warning để model
    nói rõ lý do không so sánh được.
    """
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
    """Handler xếp hạng Điểm UP trong đúng symbol scope của release.

    Hai lớp lọc trước khi sort, và thứ tự lọc quyết định con số báo cho user:
    - ``eligible_rows``: chỉ mã trong scope VÀ có ``probability_up`` hữu hạn
      (``_rounded_number`` trả None cho NaN/inf → loại).
    - ``excluded_stale_count`` đếm số dòng stale TRONG eligible_rows (đếm trước khi
      loại), rồi ``rows`` mới bỏ stale. Đếm trước là cố ý: cần cho user biết "đã bỏ
      bao nhiêu mã vì dữ liệu cũ", nếu đếm sau thì luôn bằng 0.

    Vì sao phải bỏ stale: xếp hạng là so sánh chéo nhiều mã, một mã có dữ liệu dừng
    từ tuần trước sẽ đứng chung bảng với mã cập nhật hôm nay → thứ tự vô nghĩa.
    Không còn dòng nào → ``no_current_signals`` chứ không trả bảng rỗng.

    Sort key gồm ``str(row["symbol"])`` làm tie-break để hai mã cùng score luôn ra
    cùng một thứ tự (deterministic, test reproduce được).

    ``as_of = max(dates)``: sau khi đã loại stale, các mã gần như cùng ngày; lấy max
    để hiển thị ngày tham chiếu mới nhất của bảng.
    """
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
    """Handler mô tả model đang serving (target, threshold, mốc split, metrics).

    Handler duy nhất vẫn trả dữ liệu khi release ``inconsistent`` — nhưng chỉ phần
    định danh, không có metrics:
    - ``consistent`` (current/legacy) → đọc ``metadata`` + ``summary`` đầy đủ.
    - ``inconsistent`` → chỉ đọc ``artifact`` (file model, nguồn đáng tin nhất về
      "model nào đang nằm trên đĩa"), ``summary`` bỏ trắng. Khối ``if consistent``
      phía dưới bị bỏ qua nên KHÔNG có metrics/baseline nào lọt ra. Lý do: nói
      "model là GradientBoosting" khi release lệch vẫn đúng, nhưng nói "F1 UP 61%"
      thì con số đó có thể thuộc release khác → sai nghiêm trọng.

    ``metadata.get(x) or split_report.get(x)``: artifact cũ không ghi mốc split vào
    metadata, phải lấy bù từ report của pipeline.

    ``training_symbol_count``: ``len(scope) or None`` để scope rỗng trả None thay vì
    số 0 gây hiểu nhầm "train 0 mã".
    """
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
    """Handler mô tả snapshot dataset offline (khoảng ngày, số mã, số feature).

    Handler này không phụ thuộc release: dataset tồn tại độc lập với model, nên
    câu "dữ liệu tới ngày nào" vẫn trả lời được dù model hỏng. Vì vậy không gọi
    ``_blocked_error``.

    Chuỗi fallback hai tầng:
    1. Đọc từ ``pipeline_summary`` (rẻ, đã tổng hợp sẵn).
    2. Thiếu ``date_max``/``clean_count`` → đọc trực tiếp ``hose_stock_clean.csv``
       (chỉ 2 cột qua ``usecols`` để không nạp cả file vào RAM). Cả hai đều thất bại
       mới trả ``report_unavailable``.
    ``excluded_count`` cũng fallback tương tự nhưng lỗi thì để ``None`` — số mã bị
    loại là thông tin phụ, không đáng làm cả handler thất bại.

    ``data_as_of`` (dữ liệu tới ngày nào) tách riêng khỏi ``model_trained_through``
    (model học tới ngày nào) để user thấy được độ trễ giữa hai mốc.
    """
    state = get_release_state()
    invalid = _validate_keys(arguments, allowed=set(), required=set())
    if invalid:
        return _error(state, "invalid_arguments", invalid)
    summary = state["summary"] or _read_json(PIPELINE_SUMMARY_PATH) or {}
    dataset_report = summary.get("dataset_report") or {}
    clean_report = summary.get("clean_report") or {}
    feature_report = summary.get("feature_report") or {}
    live_max = _clean_data_max_date()
    report_as_of = dataset_report.get("date_max")
    data_as_of = live_max or report_as_of
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
    # clean_count có thể là chuỗi phi số trong summary hỏng ("unknown") — int() nu
    # sẽ raise ValueError ngoài mọi handler và sập cả handler thành internal_error.
    if clean_count is not None:
        try:
            clean_count = int(clean_count)
        except (TypeError, ValueError):
            return _error(
                state,
                "report_unavailable",
                "Số đếm dataset trong pipeline report không phải số hợp lệ.",
            )
    metadata = state["metadata"] if state["status"] in {"current", "legacy"} else {}
    excluded_count = clean_report.get("excluded_symbols")
    if excluded_count is None:
        try:
            excluded = pd.read_csv(EXCLUDED_SYMBOLS_PATH, usecols=["symbol"])
            excluded_count = int(excluded["symbol"].nunique())
        except (OSError, ValueError, pd.errors.ParserError):
            excluded_count = None
    # report_as_of chỉ đưa vào data khi khác data_as_of (tức pipeline chạy chưa kịp cập nhật)
    stale_report_as_of = report_as_of if (report_as_of is not None and report_as_of != data_as_of) else None
    data = {
        "offline": True,
        "date_min": date_min,
        "date_max": data_as_of,
        "data_as_of": data_as_of,
        "report_as_of": stale_report_as_of,
        "model_trained_through": metadata.get("train_through_date"),
        "clean_symbol_count": clean_count,
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
    """Handler thứ tự feature + top importance toàn cục của model release.

    Khác ``get_dataset_info``: importance chỉ có nghĩa khi gắn với đúng model đã
    publish, nên release ``inconsistent``/``missing`` là chặn thẳng.

    ``feature_importance_written is not True`` (so sánh identity, không truthy) là
    cờ do pipeline ghi vào summary: file CSV có thể còn sót từ release TRƯỚC, nên
    không được tin sự tồn tại của file mà phải tin cờ của release hiện tại.

    ``pd.to_numeric(..., errors="raise")``: cột importance hỏng thì báo
    ``report_unavailable`` chứ không để ``NaN`` lọt vào sort rồi trả số vô nghĩa.

    ``_release_changed`` kiểm lại sau khi đọc CSV — cùng lý do như các handler khác:
    pipeline có thể ghi đè file giữa lúc đọc.

    Sort ``["importance", "feature"]`` với ``ascending=[False, True]``: tie-break
    theo tên feature để top-N ổn định (deterministic) giữa các lần gọi.
    """
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
    """Bảng "hợp đồng tĩnh" mô tả thiết kế project, theo topic đã whitelist.

    Vì sao viết cứng trong code chứ không để model tự kể: đây là câu trả lời cho
    các câu hỏi kiểu "target là gì", "chia dữ liệu thế nào". Nếu để model tự diễn
    giải, nó sẽ bịa. Ở đây mọi con số đều lấy từ ``config.settings`` nên tài liệu
    không bao giờ lệch với cấu hình thật.

    Chỉ nói về THIẾT KẾ, không nói về release đang serve (việc đó là
    ``get_model_info``). Vì vậy các mốc ngày ở topic ``data_split`` được gọi rõ là
    ``fallback_*``: mốc runtime do ``protocol_dates`` suy ra theo dữ liệu mới nhất,
    còn đây chỉ là giá trị dự phòng trong config.

    ``contracts[topic]`` truy cập trực tiếp (không ``.get``) là an toàn vì caller
    đã chặn topic lạ bằng ``PROJECT_TOPICS`` trước khi gọi.
    """
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
                # Cố ý KHÔNG viết "khuyến nghị mua hoặc bán" ở đây: LM hay nhắc
                # lại nguyên văn note, và cụm đó khớp _UNSAFE_ANSWER_RE ở tầng
                # safety (hard-fail, không soft-block được) nên chính câu
                # disclaimer lại làm sập lượt chat.
                "Kết quả dựa trên dữ liệu offline, chỉ mang tính tham khảo kỹ thuật.",
                "Hệ thống không đưa ra lời khuyên đầu tư dưới bất kỳ hình thức nào.",
                "Release cũ hoặc dữ liệu stale phải được đọc cùng cảnh báo tương ứng.",
            ],
        },
    }
    return contracts[topic]


def get_project_info(arguments: dict) -> dict:
    """Handler duy nhất không đọc release: contract tĩnh không cần gate release.

    ``state`` được dựng giả với ``status=None`` (thay vì gọi ``get_release_state``)
    vì câu trả lời không phụ thuộc artifact — model lỗi vẫn giải thích được thiết kế.
    Nhờ vậy ``_public_release`` trả policy_id/content_fingerprint = None, và
    ``chatbot_service`` sẽ không ghim release identity từ block này.
    """
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


def _domain_result(envelope: dict) -> dict:
    data = dict(envelope.get("data") or {})
    source = envelope.get("source")
    source_kind = source.get("kind") if isinstance(source, dict) else None
    as_of = envelope.get("as_of")
    data_as_of = data.get("data_as_of")
    model_trained_through = data.get("model_trained_through")
    if source_kind in {"stock_signal", "ranking", "dataset_info"}:
        data_as_of = data_as_of or as_of
    if source_kind in {"model_metadata", "feature_importance"}:
        model_trained_through = data.get("train_through_date") or as_of
    return {
        "data": data,
        "sources": [source] if source else [],
        "warnings": list(envelope.get("warnings") or []),
        "data_as_of": data_as_of,
        "model_trained_through": model_trained_through,
        "error": envelope.get("error"),
    }


def execute_action(action: str, arguments: dict) -> dict:
    """Dispatch một action dữ liệu đã được chatbot_service validate."""
    if action == "STOCK_SIGNAL":
        result = _domain_result(get_stock_signals({"symbols": arguments["symbols"]}))
        result["data"]["focus"] = arguments["focus"]
        return result

    if action == "STOCK_RANKING":
        result = _domain_result(
            get_ranking(
                {
                    "order": f"{arguments['order']}_up_score",
                    "top_n": arguments["top_n"],
                }
            )
        )
        result["data"]["order"] = arguments["order"]
        return result

    if action != "PROJECT_INFO":
        raise ValueError(f"Unsupported data action: {action}")

    topic = arguments["topic"]
    if topic == "model":
        envelope = get_model_info({})
    elif topic == "dataset":
        envelope = get_dataset_info({})
    elif topic == "features":
        envelope = get_feature_info({"top_n": 10})
    elif topic == "method":
        sections = {}
        sources = []
        warnings = []
        for section in ("target", "data_split", "training", "inference"):
            result = _domain_result(get_project_info({"topic": section}))
            if result["error"]:
                return result
            sections[section] = result["data"]
            for source in result["sources"]:
                if source not in sources:
                    sources.append(source)
            for warning in result["warnings"]:
                if warning not in warnings:
                    warnings.append(warning)
        return {
            "data": {"topic": "method", "sections": sections},
            "sources": sources,
            "warnings": warnings,
            "data_as_of": None,
            "model_trained_through": None,
            "error": None,
        }
    else:
        envelope = get_project_info({"topic": topic})

    result = _domain_result(envelope)
    result["data"]["topic"] = topic
    return result
