"""Date-based, purged cross-validation splits for panel market data."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import CV_GAP_SESSIONS, CV_N_SPLITS, CV_START_DATE


def sort_panel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic panel order shared by tuning and official CV."""
    required = {"trading_date", "symbol"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing panel sort columns: {sorted(missing)}")
    return frame.sort_values(["trading_date", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def iter_purged_date_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = CV_N_SPLITS,
    gap_sessions: int = CV_GAP_SESSIONS,
    start_date: str | None = CV_START_DATE,
    date_column: str = "trading_date",
    label_end_column: str = "label_end_date",
):
    """Yield positional row indices using whole market dates and purged labels."""
    missing = {date_column, label_end_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    dates = pd.to_datetime(frame[date_column], errors="raise").reset_index(drop=True)
    label_ends = pd.to_datetime(
        frame[label_end_column], errors="raise"
    ).reset_index(drop=True)
    # Điểm mấu chốt: TimeSeriesSplit được chạy trên DANH SÁCH NGÀY DUY NHẤT của
    # sàn, KHÔNG chạy trên các dòng của bảng. Dữ liệu là panel (mỗi ngày có ~vài
    # trăm mã), nên nếu cắt theo chỉ số dòng thì một ngày sẽ bị xé đôi: vài mã
    # của ngày 05/07 vào train, vài mã cùng ngày đó vào validation -> rò rỉ chéo
    # theo mã. Cắt theo ngày đảm bảo "cả phiên đi cùng nhau".
    market_dates = np.sort(dates.unique())
    if start_date is not None:
        # Bỏ giai đoạn quá cũ (chế độ thị trường khác) trước khi chia fold.
        market_dates = market_dates[market_dates >= pd.Timestamp(start_date)]
    # gap=gap_sessions: TimeSeriesSplit tự chừa gap PHIÊN giữa train và
    # validation của mỗi fold — lớp chống rò rỉ thứ nhất (theo thời gian).
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_sessions)

    for train_date_idx, validation_date_idx in splitter.split(market_dates):
        # splitter trả về chỉ số TRONG market_dates, phải map lại thành ngày thật.
        train_dates = market_dates[train_date_idx]
        validation_dates = market_dates[validation_date_idx]
        validation_start = validation_dates[0]
        # Lớp chống rò rỉ thứ hai (theo nhãn): dòng train chỉ hợp lệ nếu nhãn
        # t+5 của nó đã đóng TRƯỚC phiên đầu của validation. Nếu thiếu điều kiện
        # này, model được train trên dòng mà đáp án nằm bên trong khoảng
        # validation -> điểm CV bị thổi phồng.
        train_mask = dates.isin(train_dates) & (label_ends < validation_start)
        # Validation không cần purge: nó chỉ được dự đoán, không được học.
        validation_mask = dates.isin(validation_dates)
        # Đổi mask boolean -> chỉ số dòng (positional), vì sklearn cross-validate
        # nhận (train_idx, val_idx) dạng vị trí. `frame` đã reset_index nên vị trí
        # trùng khớp với thứ tự dòng của X/y truyền vào.
        train_idx = np.flatnonzero(train_mask.to_numpy())
        validation_idx = np.flatnonzero(validation_mask.to_numpy())
        if not len(train_idx) or not len(validation_idx):
            raise ValueError("Date-based CV produced an empty train or validation fold.")
        yield train_idx, validation_idx
