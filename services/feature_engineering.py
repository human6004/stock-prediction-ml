"""Biến OHLCV thành feature, exact-market labels và TRAIN/TEST dataset.

Feature rolling chạy riêng theo symbol. Target dùng phiên thứ năm trên lịch
thị trường chung và yêu cầu symbol có giá đúng tại phiên đích.
"""

import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_COLUMNS,
    FEATURE_DATA_PATH,
    ML_DATA_PATH,
    PREDICTION_HORIZON,
    SPLIT_SUMMARY_PATH,
    UP_THRESHOLD,
)
from services.protocol_dates import resolve_protocol_dates
from services.pipeline_utils import atomic_dataframe_to_csv


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI(14) theo trung bình cộng đơn giản (SMA) của lãi/lỗ trong cửa sổ.

    Ý nghĩa: RSI đo "trong 14 phiên gần nhất, phần tăng chiếm bao nhiêu so với
    tổng biến động". Gần 100 = gần như phiên nào cũng tăng (quá mua), gần 0 =
    phiên nào cũng giảm (quá bán), 50 = cân bằng.

    Cách chạy từng dòng:
      delta         : chênh lệch giá đóng cửa so với phiên trước (có thể âm).
      gain          : chỉ giữ phần tăng (âm -> 0) rồi lấy trung bình 14 phiên.
      loss          : chỉ giữ phần giảm, đổi dấu thành số dương (nên `-delta`
                      với `clip(upper=0)`), rồi trung bình 14 phiên.
      rs            : tỷ lệ lãi/lỗ. `loss.replace(0, np.nan)` để tránh chia 0 —
                      pandas sẽ cho NaN thay vì inf, ta xử lý NaN ngay dưới.
      rsi           : công thức chuẩn 100 - 100/(1+RS).

    Hai dòng `.where` cuối là xử lý hai ca biên mà công thức gốc không định
    nghĩa được (nếu bỏ đi thì cột rsi14 sẽ có NaN và các dòng đó bị dropna,
    tức là mất dữ liệu một cách âm thầm):
      1) loss == 0 (14 phiên không có phiên nào giảm) -> RS = vô cực -> RSI = 100.
      2) gain == 0 VÀ loss == 0 (giá đứng yên suốt 14 phiên, hay gặp ở mã thanh
         khoản kém) -> không có xu hướng nào -> quy ước RSI = 50 (trung tính).
    Thứ tự quan trọng: ca (2) là tập con của ca (1) nên phải gán sau, nếu không
    giá đứng yên sẽ bị gán 100 (quá mua) — sai hoàn toàn về ý nghĩa.

    `min_periods=window` bắt buộc đủ 14 phiên mới ra số; 13 phiên đầu của mỗi
    symbol là NaN và sẽ bị loại ở `build_features`.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50)
    return rsi


def build_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tính 20 feature chỉ từ dòng hiện tại và quá khứ của từng symbol.

    Vì sao phải `groupby("symbol")` rồi mới rolling: mọi phép `pct_change`,
    `rolling`, `diff` đều đọc các dòng liền trước trong Series. Nếu chạy trên
    cả bảng gộp thì dòng đầu của mã B sẽ lấy giá cuối của mã A làm "phiên
    trước" — feature bị trộn giữa hai mã, một dạng rò rỉ dữ liệu. Tách nhóm
    rồi `concat` lại đảm bảo mỗi cửa sổ chỉ nằm trong một mã.

    Toàn bộ feature ở đây chỉ dùng dòng hiện tại và quá khứ (không có `shift(-n)`
    hay `.max()` toàn cục), nên tại thời điểm dự báo phiên t ta thực sự có đủ
    mọi giá trị này — không look-ahead.
    """
    frames = []
    for _, group in clean_df.groupby("symbol", sort=False):
        g = group.sort_values("trading_date").copy()
        close = g["close"]
        volume = g["volume"]

        g["return_1d"] = close.pct_change(1)
        g["return_3d"] = close.pct_change(3)
        g["return_5d"] = close.pct_change(5)
        g["close_open_return"] = (g["close"] / g["open"]) - 1
        g["sma5"] = close.rolling(window=5, min_periods=5).mean()
        g["sma20"] = close.rolling(window=20, min_periods=20).mean()
        g["sma50"] = close.rolling(window=50, min_periods=50).mean()
        g["close_vs_sma20"] = (g["close"] / g["sma20"]) - 1
        g["sma20_vs_sma50"] = (g["sma20"] / g["sma50"]) - 1
        g["rsi14"] = compute_rsi(close, 14)
        g["volatility_5d"] = g["return_1d"].rolling(window=5, min_periods=5).std()
        g["volatility_20d"] = g["return_1d"].rolling(window=20, min_periods=20).std()
        g["price_range"] = (g["high"] - g["low"]) / g["close"]
        g["volume_change_1d"] = volume.pct_change(1)
        volume_ma20 = volume.rolling(window=20, min_periods=20).mean()
        g["volume_ratio_20"] = volume / volume_ma20
        g["return_10d"] = close.pct_change(10)
        g["return_20d"] = close.pct_change(20)
        roll_high20 = g["high"].rolling(window=20, min_periods=20).max()
        roll_low20 = g["low"].rolling(window=20, min_periods=20).min()
        g["dist_high20"] = (g["close"] / roll_high20) - 1
        g["dist_low20"] = (g["close"] / roll_low20) - 1
        g["month"] = pd.to_datetime(g["trading_date"]).dt.month
        frames.append(g)

    features = pd.concat(frames, ignore_index=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    # Rolling window dài nhất cần 50 row; các row đầu chưa đủ lịch sử bị loại.
    features = features.dropna(subset=FEATURE_COLUMNS).copy()
    features = features.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    report = {
        "rows_after_features": int(len(features)),
        "symbols_after_features": int(features["symbol"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
    }
    return features, report


def create_labels(
    clean_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    horizon: int = PREDICTION_HORIZON,
    up_threshold: float = UP_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Attach exact common-market t+horizon targets to feature-valid rows.

    The market calendar comes from all clean rows. A symbol must have a close on
    the exact future market session; missing prices are dropped, never skipped.
    """
    required_clean = {"symbol", "trading_date", "close"}
    required_features = {"symbol", "trading_date"}
    if missing := required_clean - set(clean_df.columns):
        raise ValueError(f"Clean data missing label columns: {sorted(missing)}")
    if missing := required_features - set(features.columns):
        raise ValueError(f"Features missing label keys: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least one market session.")

    prices = clean_df[["symbol", "trading_date", "close"]].copy()
    prices["symbol"] = prices["symbol"].astype(str).str.strip().str.upper()
    prices["trading_date"] = pd.to_datetime(
        prices["trading_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if prices.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("Clean data contains duplicate symbol/trading_date rows.")

    # --- Bước 1: dựng "lịch thị trường chung" và ánh xạ phiên t -> phiên t+5 ---
    # Không dùng `close.shift(-5)` theo từng mã, vì mã nghỉ giao dịch vài phiên
    # sẽ khiến shift(-5) nhảy xa hơn 5 phiên thị trường thật (mỗi mã có mốc t+5
    # khác nhau -> nhãn không so sánh được với nhau, và CV theo ngày sẽ sai).
    # Thay vào đó: lấy tập ngày giao dịch của TOÀN sàn, sắp tăng dần, rồi
    # shift(-horizon) trên chính danh sách ngày đó. Kết quả `future_by_date` là
    # từ điển {ngày t: ngày t+5 của sàn}, dùng chung cho mọi mã.
    market_dates = pd.Series(sorted(prices["trading_date"].unique()))
    future_by_date = dict(zip(market_dates, market_dates.shift(-horizon)))
    labels = prices.rename(columns={"close": "close_at_label_start"})
    # 5 phiên cuối dataset không có t+5 -> label_end_date = NaN (bị bỏ ở dropna).
    labels["label_end_date"] = labels["trading_date"].map(future_by_date)

    # --- Bước 2: self-join lấy giá đóng cửa TẠI ĐÚNG phiên t+5 của chính mã đó ---
    # Mẹo: đổi tên `trading_date` -> `label_end_date` để join chính bảng giá với
    # bản thân nó theo (symbol, ngày-đích). `how="left"` nên nếu mã đó không có
    # giao dịch đúng phiên t+5 thì future_close_5d = NaN, dòng bị loại ở dưới —
    # cố tình KHÔNG lấy giá gần nhất thay thế, vì đó sẽ là nhãn sai kỳ hạn.
    # `validate="many_to_one"` là chốt an toàn: bên phải phải duy nhất theo
    # (symbol, ngày) — nếu raw còn trùng dòng, pandas raise ngay thay vì nhân
    # đôi số dòng một cách âm thầm.
    future_prices = prices.rename(
        columns={"trading_date": "label_end_date", "close": "future_close_5d"}
    )
    labels = labels.merge(
        future_prices,
        on=["symbol", "label_end_date"],
        how="left",
        validate="many_to_one",
    )
    labels["future_return_5d"] = (
        labels["future_close_5d"] / labels["close_at_label_start"]
    ) - 1
    labels["target"] = (labels["future_return_5d"] > up_threshold).astype("int8")
    labels["target_label"] = np.where(labels["target"] == 1, "UP", "NOT_UP")

    rows_without_market_horizon = int(labels["label_end_date"].isna().sum())
    rows_missing_symbol_future_close = int(
        (labels["label_end_date"].notna() & labels["future_close_5d"].isna()).sum()
    )
    label_columns = [
        "symbol",
        "trading_date",
        "future_close_5d",
        "future_return_5d",
        "label_end_date",
        "target",
        "target_label",
    ]
    feature_rows = features.copy()
    feature_rows["symbol"] = feature_rows["symbol"].astype(str).str.strip().str.upper()
    feature_rows["trading_date"] = pd.to_datetime(
        feature_rows["trading_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    dataset = feature_rows.merge(
        labels[label_columns],
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    dataset = dataset.dropna(
        subset=["future_close_5d", "future_return_5d", "label_end_date"]
    ).copy()
    if dataset.empty:
        raise ValueError("No feature rows have an exact future market-session price.")
    dataset["target"] = (dataset["future_return_5d"] > up_threshold).astype(int)
    dataset["target_label"] = np.where(dataset["target"] == 1, "UP", "NOT_UP")
    dataset = dataset.sort_values(["trading_date", "symbol"]).reset_index(drop=True)

    up_count = int((dataset["target"] == 1).sum())
    not_up_count = int((dataset["target"] == 0).sum())
    report = {
        "horizon": horizon,
        "threshold": up_threshold,
        "target_calendar": "common_market_exact_session",
        "rows_without_market_horizon": rows_without_market_horizon,
        "rows_missing_symbol_future_close": rows_missing_symbol_future_close,
        "rows_after_labeling": int(len(dataset)),
        "up_count": up_count,
        "not_up_count": not_up_count,
        "up_ratio": round(up_count / len(dataset), 6),
        "not_up_ratio": round(not_up_count / len(dataset), 6),
    }
    return dataset, report


def protocol_time_split(
    dataset: pd.DataFrame,
    *,
    train_end_date: str | None = None,
    validation_end_date: str | None = None,
    test_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split rolling TRAIN/VALIDATION/TEST windows.

    Khi không truyền mốc tường minh, mốc được suy động từ chính ``dataset``
    (rolling walk-forward) qua ``resolve_protocol_dates`` — cùng nguồn với
    fingerprint ở ``experiment_state`` nên split và fingerprint không lệch nhau.
    """
    required = {"trading_date", "label_end_date", "target"}
    if missing := required - set(dataset.columns):
        raise ValueError(f"Dataset missing split columns: {sorted(missing)}")
    if train_end_date is None or validation_end_date is None or test_end_date is None:
        resolved = resolve_protocol_dates(dataset)
        train_end_date = train_end_date or resolved["train_end_date"]
        validation_end_date = validation_end_date or resolved["validation_end_date"]
        test_end_date = test_end_date or resolved["test_end_date"]
    if not train_end_date < validation_end_date < test_end_date:
        raise ValueError(
            "Expected TRAIN_END_DATE < VALIDATION_END_DATE < TEST_END_DATE."
        )

    frame = dataset.copy()
    trading_dates = pd.to_datetime(frame["trading_date"], errors="raise")
    label_end_dates = pd.to_datetime(frame["label_end_date"], errors="raise")
    train_end = pd.Timestamp(train_end_date)
    validation_end = pd.Timestamp(validation_end_date)
    test_end = pd.Timestamp(test_end_date)

    # --- Ba mask: điều kiện lọc dòng cho từng split ---
    # Điểm cốt lõi: TRAIN lọc theo `label_end_date` (ngày nhãn KẾT THÚC), không
    # theo `trading_date`. Lý do: một dòng ngày 28/06 có nhãn nhìn tới 05/07.
    # Nếu chỉ cắt theo trading_date <= 30/06 thì model đã "biết" giá tới 05/07
    # trong khi VALIDATION bắt đầu 01/07 -> rò rỉ tương lai. Ràng buộc
    # `label_end_date <= train_end` loại đúng những dòng biên đó (gọi là purge).
    train_mask = label_end_dates <= train_end
    # VALIDATION: feature phải nằm sau mốc TRAIN, và nhãn cũng phải đóng trước
    # mốc VALIDATION (purge lần hai, cùng lý do như trên với TEST).
    validation_mask = (trading_dates > train_end) & (
        label_end_dates <= validation_end
    )
    # TEST lọc theo trading_date thuần: đây là tập đánh giá cuối, ta muốn giữ
    # đủ mọi phiên trong cửa sổ TEST, kể cả dòng có nhãn vượt ra sau test_end
    # (không có split nào nằm sau TEST nên không thể rò rỉ sang đâu nữa).
    test_mask = (trading_dates > validation_end) & (trading_dates <= test_end)

    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Protocol split produced an empty set "
            f"(train={len(train)}, validation={len(validation)}, test={len(test)})."
        )

    # --- Đếm số dòng bị "purge" (hy sinh) ở mỗi vùng đệm, chỉ để báo cáo ---
    # Dòng có trading_date còn trong vùng TRAIN nhưng bị train_mask loại,
    # tức nhãn của nó lấn qua mốc TRAIN. Đây là "cái giá" của việc chống rò rỉ:
    # khoảng 5 phiên cuối mỗi vùng bị bỏ đi, không rơi vào split nào cả.
    train_validation_purge = (~train_mask) & (trading_dates <= train_end)
    # Tương tự ở biên VALIDATION -> TEST.
    validation_test_purge = (
        (trading_dates > train_end)
        & (trading_dates <= validation_end)
        & (label_end_dates > validation_end)
    )
    # Dòng sau mốc TEST: không dùng để train/đánh giá, nhưng vẫn hữu ích cho
    # suy luận thực tế (dự báo phiên mới nhất) nên chỉ đếm để biết còn bao nhiêu.
    post_test_rows = trading_dates > test_end
    report = {
        "split_method": "rolling_dates_with_label_purge",
        "train_end_date": train_end_date,
        "validation_end_date": validation_end_date,
        "test_end_date": test_end_date,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "purged_train_validation_rows": int(train_validation_purge.sum()),
        "purged_validation_test_rows": int(validation_test_purge.sum()),
        "post_test_inference_rows": int(post_test_rows.sum()),
        "train_date_min": str(train["trading_date"].min()),
        "train_date_max": str(train["trading_date"].max()),
        "train_label_end_max": str(train["label_end_date"].max()),
        "validation_date_min": str(validation["trading_date"].min()),
        "validation_date_max": str(validation["trading_date"].max()),
        "validation_label_end_max": str(validation["label_end_date"].max()),
        "test_date_min": str(test["trading_date"].min()),
        "test_date_max": str(test["trading_date"].max()),
    }
    return train, validation, test, report


def write_split_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    validation: pd.DataFrame | None = None,
) -> None:
    """Xuất `split_summary.csv` — bằng chứng kiểm tra rò rỉ dữ liệu giữa các split.

    Điểm quan trọng: mỗi split cố tình ghi cột ``label_end_date`` KHÁC nhau, không
    phải sơ suất.
    - TRAIN/VALIDATION ghi ``label_end_max``: mốc muộn nhất mà nhãn của split này
      "nhìn thấy" tương lai.
    - TEST ghi ``label_end_min``: mốc sớm nhất TEST cần biết.

    Đặt cạnh nhau, người đọc so trực tiếp ``label_end_max`` của TRAIN với
    ``date_min`` của TEST: nếu nhãn TRAIN kết thúc sau khi TEST bắt đầu thì đã rò
    rỉ. Ghi cả hai cột cho mọi split sẽ khiến bảng rộng mà mất đúng phép so này.

    Vì ``rows`` là list các dict khác key nhau, ``pd.DataFrame`` tự điền NaN vào ô
    thiếu — nên CSV có cả hai cột và mỗi hàng chỉ có ô của mình.

    ``validation=None`` là nhánh cho protocol cũ hai-split; khi đó bảng chỉ có
    train/test.
    """
    rows = [
        {
            "split": "train",
            "rows": len(train),
            "symbols": train["symbol"].nunique(),
            "date_min": train["trading_date"].min(),
            "date_max": train["trading_date"].max(),
            "label_end_max": train["label_end_date"].max(),
            "up_count": int((train["target"] == 1).sum()),
            "not_up_count": int((train["target"] == 0).sum()),
        },
    ]
    if validation is not None:
        rows.append(
            {
                "split": "validation",
                "rows": len(validation),
                "symbols": validation["symbol"].nunique(),
                "date_min": validation["trading_date"].min(),
                "date_max": validation["trading_date"].max(),
                "label_end_max": validation["label_end_date"].max(),
                "up_count": int((validation["target"] == 1).sum()),
                "not_up_count": int((validation["target"] == 0).sum()),
            }
        )
    rows.append(
        {
            "split": "test",
            "rows": len(test),
            "symbols": test["symbol"].nunique(),
            "date_min": test["trading_date"].min(),
            "date_max": test["trading_date"].max(),
            "label_end_min": test["label_end_date"].min(),
            "up_count": int((test["target"] == 1).sum()),
            "not_up_count": int((test["target"] == 0).sum()),
        }
    )
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_to_csv(pd.DataFrame(rows), SPLIT_SUMMARY_PATH, index=False)


def write_feature_outputs(features: pd.DataFrame, dataset: pd.DataFrame) -> None:
    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_to_csv(features, FEATURE_DATA_PATH, index=False)
    atomic_dataframe_to_csv(dataset, ML_DATA_PATH, index=False)


def verify_protocol_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dates: dict | None = None,
) -> None:
    """Reject any date/label leakage across rolling boundaries.

    ``dates`` là 3 mốc thực tế đã dùng để cắt split (lấy từ report của
    ``protocol_time_split``). Nếu None, suy lại mốc từ chính dữ liệu split để
    dùng chung một nguồn với bước cắt.
    """
    if dates is None:
        combined = pd.concat([train, validation, test], ignore_index=True)
        dates = resolve_protocol_dates(combined)
    train_end_date = dates["train_end_date"]
    validation_end_date = dates["validation_end_date"]
    test_end_date = dates["test_end_date"]
    if train["label_end_date"].max() > train_end_date:
        raise ValueError("TRAIN label window crosses TRAIN_END_DATE.")
    if validation["trading_date"].min() <= train_end_date:
        raise ValueError("VALIDATION starts inside TRAIN period.")
    if validation["label_end_date"].max() > validation_end_date:
        raise ValueError("VALIDATION label window crosses VALIDATION_END_DATE.")
    if test["trading_date"].min() <= validation_end_date:
        raise ValueError("TEST starts inside VALIDATION period.")
    if test["trading_date"].max() > test_end_date:
        raise ValueError("TEST extends past the resolved TEST boundary.")
    if train["label_end_date"].max() >= validation["trading_date"].min():
        raise ValueError("TRAIN labels overlap VALIDATION features.")
    if validation["label_end_date"].max() >= test["trading_date"].min():
        raise ValueError("VALIDATION labels overlap TEST features.")
    leaked = {
        "future_close_5d",
        "future_return_5d",
        "target",
        "label_end_date",
    } & set(FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Label columns leaked into FEATURE_COLUMNS: {leaked}")
