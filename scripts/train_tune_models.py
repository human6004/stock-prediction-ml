import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import ML_DATA_PATH  # noqa: E402
from services.feature_engineering import time_based_split  # noqa: E402
from services.model_tuning import tune_models  # noqa: E402


def load_train() -> pd.DataFrame:
    if not ML_DATA_PATH.exists():
        raise FileNotFoundError("ML dataset not found. Run build_features.py first.")
    dataset = pd.read_csv(ML_DATA_PATH)
    train, _, _ = time_based_split(dataset)
    return train


def main() -> None:
    train = load_train()
    fitted, tuning_df, report = tune_models(train)
    print("Train/tune done.")
    print(f"Tuned models: {report['tuned_models']}")
    print(f"CV: n_splits={report['cv_n_splits']}, gap={report['cv_gap']}")
    print(tuning_df.to_string(index=False))
    print(f"Artifacts saved for model ids: {sorted(fitted.keys())}")


if __name__ == "__main__":
    main()
