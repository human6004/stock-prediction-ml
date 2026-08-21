"""CLI tạo feature, label và bảng TRAIN/VALIDATION/TEST từ dữ liệu clean."""

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import CLEANED_DATA_PATH, ELIGIBLE_SYMBOLS_PATH  # noqa: E402
from services.feature_engineering import (  # noqa: E402
    build_features,
    create_labels,
    protocol_time_split,
    verify_protocol_splits,
    write_feature_outputs,
    write_split_summary,
)
from services.preprocessing import clean_data, dataset_check  # noqa: E402


def _load_clean_for_training() -> pd.DataFrame:
    """Ưu tiên CSV clean có sẵn và chỉ giữ symbol đủ điều kiện train.

    Ba nhánh, theo thứ tự rẻ → đắt:
    1. Có ``hose_stock_clean.csv`` + ``eligible_symbols.csv`` → đọc CSV rồi lọc theo
       danh sách mã đủ điều kiện. Đây là đường đi bình thường sau bước preprocess.
    2. Có clean CSV nhưng CHƯA có eligible list → dùng nguyên clean CSV. Chấp nhận
       được vì bản thân clean CSV đã qua kiểm tra OHLCV; chỉ thiếu bộ lọc thanh khoản
       (MIN_TRADING_DAYS / MIN_AVERAGE_VOLUME).
    3. Chưa có clean CSV → chạy lại ``dataset_check`` + ``clean_data`` từ raw. Đắt
       nhất (đọc toàn bộ raw), nên chỉ dùng khi thật sự chưa preprocess lần nào.

    Lưu ý: nhánh 3 trả về ``clean_for_training`` (đã lọc sẵn) nên ``return`` ngay,
    không đi tiếp xuống phần lọc eligible bên dưới.
    """
    if CLEANED_DATA_PATH.exists():
        clean_all = pd.read_csv(CLEANED_DATA_PATH)
    else:
        raw_df, _ = dataset_check()
        _, clean_for_training, _, _ = clean_data(raw_df)
        return clean_for_training

    if ELIGIBLE_SYMBOLS_PATH.exists():
        eligible = pd.read_csv(ELIGIBLE_SYMBOLS_PATH)["symbol"].astype(str)
        return clean_all[clean_all["symbol"].isin(eligible)].copy()
    return clean_all


def main() -> None:
    """Dựng feature + nhãn + split từ dữ liệu sạch, chỉ ghi file khi verify pass.

    Thứ tự các bước ở đây là quan trọng, không đổi được:
    # 1. build_features: tính chỉ báo trên chuỗi giá đầy đủ (cần đủ lịch sử cho
    #    sma50, return_20d... nên phải làm TRƯỚC khi cắt bớt dòng).
    # 2. create_labels: gán target theo phiên thị trường chung t+5. Truyền CẢ
    #    clean_for_training (làm lịch thị trường + giá tương lai) lẫn features, vì
    #    label phải tra giá của phiên t+5 kể cả khi dòng đó đã bị feature loại.
    # 3. protocol_time_split + verify_protocol_splits: chia TRAIN/VALIDATION/TEST
    #    theo mốc rolling rồi KIỂM TRA lại (không tin tưởng ngầm) — verify sẽ raise
    #    nếu có rò rỉ nhãn qua ranh giới.
    # 4. Chỉ ghi file sau khi verify pass, để không publish dataset bị lỗi split.
    """
    clean_for_training = _load_clean_for_training()
    features, feature_report = build_features(clean_for_training)
    ml_dataset, label_report = create_labels(clean_for_training, features)
    train, validation, test, split_report = protocol_time_split(ml_dataset)
    verify_protocol_splits(train, validation, test, dates=split_report)
    write_feature_outputs(features, ml_dataset)
    write_split_summary(train, test, validation)
    print("Build features done.")
    print(f"Feature rows: {feature_report['rows_after_features']}")
    print(f"ML dataset rows: {label_report['rows_after_labeling']}")
    print(
        f"Train rows: {split_report['train_rows']}, "
        f"Validation rows: {split_report['validation_rows']}, "
        f"Test rows: {split_report['test_rows']}"
    )


if __name__ == "__main__":
    main()
