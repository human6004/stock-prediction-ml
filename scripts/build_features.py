"""CLI tạo feature, label và bảng TRAIN/TEST từ dữ liệu đã clean."""

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
    time_based_split,
    verify_data_pipeline,
    write_feature_outputs,
    write_train_test_summary,
)
from services.preprocessing import clean_data, dataset_check  # noqa: E402


def _load_clean_for_training() -> pd.DataFrame:
    """Ưu tiên CSV clean có sẵn và chỉ giữ symbol đủ điều kiện train."""
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
    # Thứ tự bắt buộc: feature quá khứ -> label tương lai -> split -> verify.
    clean_for_training = _load_clean_for_training()
    features, feature_report = build_features(clean_for_training)
    ml_dataset, label_report = create_labels(features)
    train, test, split_report = time_based_split(ml_dataset)
    verify_data_pipeline(train, test)
    write_feature_outputs(features, ml_dataset)
    write_train_test_summary(train, test)
    print("Build features done.")
    print(f"Feature rows: {feature_report['rows_after_features']}")
    print(f"ML dataset rows: {label_report['rows_after_labeling']}")
    print(f"Train rows: {split_report['train_rows']}, Test rows: {split_report['test_rows']}")


if __name__ == "__main__":
    main()
