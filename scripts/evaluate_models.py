"""Debug evaluator: đọc TEST và ghi model_comparison.csv.

Script độc lập này không kiểm tra test_evaluation_lock.json. Không chạy lặp để
chọn/tune model; official flow an toàn hơn nằm trong scripts/run_pipeline.py.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import COMPARISON_COLUMNS, ML_DATA_PATH, MODEL_COMPARISON_PATH  # noqa: E402
from services.feature_engineering import time_based_split  # noqa: E402
from services.model_evaluation import evaluate_tuned_models  # noqa: E402


def load_split():
    if not ML_DATA_PATH.exists():
        raise FileNotFoundError("ML dataset not found. Run build_features.py first.")
    dataset = pd.read_csv(ML_DATA_PATH)
    return time_based_split(dataset)


def main() -> None:
    # TEST metrics được tạo ở đây; train chỉ giữ để cùng interface split/report.
    train, test, split_report = load_split()
    comparison, _ = evaluate_tuned_models(train, test)
    cols = [c for c in COMPARISON_COLUMNS if c in comparison.columns]
    comparison[cols].to_csv(MODEL_COMPARISON_PATH, index=False)
    print("Evaluate models done.")
    print(f"Test rows: {split_report['test_rows']}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
