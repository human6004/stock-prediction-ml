from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ROADMAP_PATH = r"D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md"
RAW_DATA_PATH = r"D:\study\niên luận\shared_dataset\hose_stock_raw.csv"

RANDOM_STATE = 42
SPLIT_DATE = "2025-12-31"
PREDICTION_HORIZON = 5
UP_THRESHOLD = 0.01
MIN_TRADING_DAYS = 250
MIN_AVERAGE_VOLUME = 0

FETCH_SOURCE = "KBS"
FETCH_SLEEP_SECONDS = 3.5
FETCH_MAX_RETRIES = 3
FETCH_END_DATE = None

CV_N_SPLITS = 5
CV_GAP = 5
TUNING_N_ITER = 12
TUNING_SCORING = "f1"

REQUIRED_COLUMNS = [
    "symbol",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "close_open_return",
    "sma5",
    "sma20",
    "sma50",
    "close_vs_sma20",
    "sma20_vs_sma50",
    "rsi14",
    "volatility_5d",
    "volatility_20d",
    "price_range",
    "volume_change_1d",
    "volume_ratio_20",
]

LABEL_COLUMNS = [
    "future_close_5d",
    "future_return_5d",
    "label_end_date",
    "target",
    "target_label",
]

MODEL_DEFINITIONS = {
    1: "Dummy Classifier",
    2: "Logistic Regression",
    3: "Random Forest",
    4: "Gradient Boosting",
}

SIMPLICITY_RANK = {2: 1, 3: 2, 4: 3}

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "stock_prediction.db"

CLEANED_DATA_PATH = PROCESSED_DIR / "hose_stock_clean.csv"
FEATURE_DATA_PATH = PROCESSED_DIR / "hose_stock_features.csv"
ML_DATA_PATH = PROCESSED_DIR / "ml_dataset.csv"

MODEL_PATHS = {
    1: MODELS_DIR / "dummy.pkl",
    2: MODELS_DIR / "logistic_regression_tuned.pkl",
    3: MODELS_DIR / "random_forest_tuned.pkl",
    4: MODELS_DIR / "gradient_boosting_tuned.pkl",
}
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.csv"
ELIGIBLE_SYMBOLS_PATH = REPORTS_DIR / "eligible_symbols.csv"
EXCLUDED_SYMBOLS_PATH = REPORTS_DIR / "excluded_symbols.csv"
TRAIN_TEST_SUMMARY_PATH = REPORTS_DIR / "train_test_summary.csv"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.csv"
BEST_PARAMS_PATH = REPORTS_DIR / "best_params.json"
CV_FOLD_RESULTS_PATH = REPORTS_DIR / "cv_fold_results.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FINAL_MODEL_EVALUATION_PATH = REPORTS_DIR / "final_model_evaluation.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
MODEL_SELECTION_REPORT_PATH = REPORTS_DIR / "model_selection_report.txt"
PIPELINE_SUMMARY_PATH = REPORTS_DIR / "pipeline_summary.json"
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
FETCH_REPORT_PATH = REPORTS_DIR / "fetch_report.json"
HYPERPARAMETER_EXPLANATION_PATH = REPORTS_DIR / "hyperparameter_explanation.md"

PRESERVED_REPORT_JSON = {
    FETCH_REPORT_PATH.name,
    BEST_PARAMS_PATH.name,
    "model_metadata.json",
}

COMPARISON_COLUMNS = [
    "model_id",
    "model_name",
    "cv_f1_up",
    "accuracy",
    "precision_up",
    "recall_up",
    "f1_up",
    "precision_not_up",
    "recall_not_up",
    "f1_not_up",
    "selected",
]
