"""Nguồn mốc TRAIN/VALIDATION/TEST duy nhất cho toàn pipeline.

Trước đây 3 mốc là hằng số ghi cứng trong ``config.settings``; mọi lần fetch
dữ liệu mới không đổi được chúng nên fingerprint dataset giữ nguyên và khóa TEST
không bao giờ mở lại. Module này chuyển 3 mốc sang **rolling walk-forward**: neo
``TEST_END_DATE`` vào phiên giao dịch mới nhất của dataset rồi lùi lại theo đúng
độ dài cửa sổ cố định.

QUAN TRỌNG: cả ``feature_engineering.protocol_time_split`` (cắt split thật) lẫn
``experiment_state._fingerprint_scope_frame`` (mask để tính fingerprint) đều gọi
``resolve_protocol_dates`` ở đây. Đây là điểm duy nhất tính mốc, nên split thật
và fingerprint không thể lệch nhau.

Vì ``create_labels`` đã ``dropna`` target trước khi ghi ``ml_dataset.csv``, phiên
``trading_date`` lớn nhất trong dataset chính là phiên có nhãn t+5 hợp lệ mới
nhất — đúng điểm cần neo ``TEST_END_DATE``, không phải max trading_date thô.
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    TEST_END_DATE,
    TEST_WINDOW_DAYS,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_WINDOW_DAYS,
)

# Mốc fallback dùng khi dataset chưa đủ dài để lấp đầy một cửa sổ đầy đủ.
_FALLBACK_DATES = {
    "train_end_date": TRAIN_END_DATE,
    "validation_end_date": VALIDATION_END_DATE,
    "test_end_date": TEST_END_DATE,
    "source": "fallback_fixed",
}

# Dataset phải trải dài hơn tổng hai cửa sổ VALIDATION+TEST thì TRAIN mới còn
# dữ liệu (train_end lùi về trước min trading_date sẽ khiến TRAIN rỗng).
_MIN_SPAN_DAYS = VALIDATION_WINDOW_DAYS + TEST_WINDOW_DAYS


def resolve_protocol_dates(dataset: pd.DataFrame | None) -> dict:
    """Trả về 3 mốc TRAIN/VALIDATION/TEST cho ``dataset``.

    - Neo ``test_end_date`` = phiên ``trading_date`` mới nhất trong dataset.
    - ``validation_end_date`` = test_end - ``TEST_WINDOW_DAYS`` ngày.
    - ``train_end_date`` = validation_end - ``VALIDATION_WINDOW_DAYS`` ngày.
    - Nếu dataset là None/rỗng/thiếu cột ``trading_date`` hoặc không đủ dài để
      lấp đầy một cửa sổ đầy đủ, trả về mốc fallback cố định.

    Các mốc trả về là chuỗi ``YYYY-MM-DD`` để đồng nhất với so sánh chuỗi ở
    ``protocol_time_split`` và ``verify_protocol_splits``.
    """
    if dataset is None or "trading_date" not in getattr(dataset, "columns", []):
        return dict(_FALLBACK_DATES)

    trading_dates = pd.to_datetime(dataset["trading_date"], errors="coerce").dropna()
    if trading_dates.empty:
        return dict(_FALLBACK_DATES)

    max_date = trading_dates.max()
    min_date = trading_dates.min()
    if (max_date - min_date).days <= _MIN_SPAN_DAYS:
        return dict(_FALLBACK_DATES)

    test_end = max_date
    validation_end = test_end - pd.Timedelta(days=TEST_WINDOW_DAYS)
    train_end = validation_end - pd.Timedelta(days=VALIDATION_WINDOW_DAYS)

    # An toàn kép: nếu train_end lùi tới trước phiên sớm nhất thì TRAIN rỗng.
    if train_end <= min_date:
        return dict(_FALLBACK_DATES)

    return {
        "train_end_date": train_end.strftime("%Y-%m-%d"),
        "validation_end_date": validation_end.strftime("%Y-%m-%d"),
        "test_end_date": test_end.strftime("%Y-%m-%d"),
        "source": "rolling_max_session",
    }
