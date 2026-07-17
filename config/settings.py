from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ROADMAP_PATH = r"D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md"
RAW_DATA_PATH = r"D:\study\niên luận\shared_dataset\hose_stock_raw.csv"

RANDOM_STATE = 42
SPLIT_DATE = "2025-06-30"
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
    "return_10d",
    "return_20d",
    "dist_high20",
    "dist_low20",
    "month",
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

MANUAL_CONFIG_SCHEMA_VERSION = 1

MODEL_KEY = {
    2: "logistic_regression",
    3: "random_forest",
    4: "gradient_boosting",
}
MODEL_KEY_TO_ID = {key: model_id for model_id, key in MODEL_KEY.items()}

TUNABLE_PARAM_SCHEMA = {
    "logistic_regression": {
        "C": {"type": "float", "min": 0.0, "inclusive_min": False, "label": "C (nghịch đảo cường độ regularization)"},
        "solver": {"type": "choice", "choices": ["lbfgs", "liblinear"], "label": "solver"},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "min": 1, "label": "n_estimators (số cây)"},
        "max_depth": {"type": "int_or_none", "min": 1, "label": "max_depth (để trống = None)"},
        "min_samples_leaf": {"type": "int", "min": 1, "label": "min_samples_leaf"},
        "max_features": {
            "type": "str_or_float",
            "choices": ["sqrt", "log2"],
            "min": 0.0,
            "max": 1.0,
            "inclusive_min": False,
            "label": "max_features (sqrt/log2 hoặc số thực trong (0,1])",
        },
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "min": 1, "label": "n_estimators (số boosting stage)"},
        "learning_rate": {"type": "float", "min": 0.0, "max": 1.0, "inclusive_min": False, "label": "learning_rate (0,1]"},
        "max_depth": {"type": "int", "min": 1, "label": "max_depth"},
        "subsample": {"type": "float", "min": 0.0, "max": 1.0, "inclusive_min": False, "label": "subsample (0,1]"},
    },
}

MANUAL_BASELINE_PARAMS = {
    "logistic_regression": {"C": 1.0, "solver": "lbfgs"},
    "random_forest": {
        "n_estimators": 120,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.85,
    },
}

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "stock_prediction.db"

# Tuning Lab experiment state (kept outside cleanup_outputs() targets).
EXPERIMENTS_DIR = BASE_DIR / "experiments"
TUNING_HISTORY_PATH = EXPERIMENTS_DIR / "tuning_history.csv"
MANUAL_CONFIG_PATH = EXPERIMENTS_DIR / "manual_config.json"
TEST_EVAL_LOCK_PATH = EXPERIMENTS_DIR / "test_evaluation_lock.json"
PIPELINE_LOCK_PATH = EXPERIMENTS_DIR / "pipeline.lock"
LAST_PIPELINE_RUN_LOG = EXPERIMENTS_DIR / "last_pipeline_run.log"
FETCH_LOCK_PATH = EXPERIMENTS_DIR / "fetch.lock"
LAST_FETCH_RUN_LOG = EXPERIMENTS_DIR / "last_fetch_run.log"

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
