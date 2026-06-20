import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import ML_DATA_PATH, MODEL_COMPARISON_PATH  # noqa: E402
from services.feature_engineering import time_based_split  # noqa: E402
from services.model_evaluation import (  # noqa: E402
    evaluate_tuned_models,
    select_final_model,
    verify_model_selection,
    write_reports,
)


def main() -> None:
    if not ML_DATA_PATH.exists():
        raise FileNotFoundError("ML dataset not found. Run build_features.py first.")
    dataset = pd.read_csv(ML_DATA_PATH)
    train, test, split_report = time_based_split(dataset)

    comparison, fitted = evaluate_tuned_models(train, test)

    comparison, selected_artifact, selection_report = select_final_model(comparison, fitted)
    verify_model_selection(comparison, selected_artifact)

    summary = {"split_report": split_report, "selection_report": selection_report}
    write_reports(
        test,
        comparison,
        selected_artifact,
        summary,
        train=train,
        selection_report=selection_report,
    )
    print("Final model selected:", selection_report["selected_model_name"])
    print(f"F1_UP: {selection_report['selected_f1_up']:.4f}")


if __name__ == "__main__":
    main()
