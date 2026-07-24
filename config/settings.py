"""Cấu hình trung tâm của toàn project.

Các module khác import giá trị từ đây thay vì tự ghi cứng đường dẫn, feature,
ranh giới TRAIN/TEST hoặc vị trí output. Đổi bài toán dự báo cần bắt đầu từ file này.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Input nằm ngoài repo; mọi output còn lại được tạo tương đối từ BASE_DIR.
RAW_DATA_PATH = r"D:\study\niên luận\shared_dataset\hose_stock_raw.csv"

# Định nghĩa bài toán: phiên thị trường chung thứ 5 tăng hơn 1% được gán nhãn UP.
RANDOM_STATE = 42
# Stable identities; schema revisions remain separate numeric fields.
DATA_PROTOCOL_ID = "exact_market_t5"
# Mốc TRAIN/VALIDATION/TEST giờ dịch động theo phiên mới nhất của dataset
# (rolling walk-forward). 3 hằng số dưới đây là FALLBACK khi dataset chưa đủ
# dài để lấp đầy một cửa sổ đầy đủ; services/protocol_dates.py suy ra mốc thực.
TRAIN_END_DATE = "2025-06-30"
VALIDATION_END_DATE = "2026-03-31"
# Snapshot TEST fallback; khi có đủ data, mốc này = phiên nhãn hợp lệ mới nhất.
TEST_END_DATE = "2026-07-03"
# Độ dài cố định của mỗi cửa sổ (số ngày lịch), giữ nguyên như thiết kế gốc:
# TEST = TEST_END - VALIDATION_END, VALIDATION = VALIDATION_END - TRAIN_END.
TEST_WINDOW_DAYS = 94
VALIDATION_WINDOW_DAYS = 274
PREDICTION_HORIZON = 5
UP_THRESHOLD = 0.01
# Default used while constructing a new artifact.
DECISION_THRESHOLD = 0.5
EXPERIMENT_POLICY_ID = "rolling_recent_cv_oof_threshold"
MIN_TRADING_DAYS = 250
MIN_AVERAGE_VOLUME = 0

FETCH_SOURCE = "KBS"
FETCH_SLEEP_SECONDS = 3.5
FETCH_MAX_RETRIES = 3
FETCH_END_DATE = None

# CV chỉ chấm regime có panel đủ dày; final fit vẫn dùng toàn bộ TRAIN.
CV_START_DATE = "2021-01-01"
CV_N_SPLITS = 4
CV_GAP_SESSIONS = 5
TUNING_SCORING = "f1"
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
THRESHOLD_MAX_PREDICTED_UP_RATIO = 0.50

REQUIRED_COLUMNS = [
    "symbol",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Thứ tự này là contract giữa feature engineering, model artifact và prediction.
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

MODEL_DEFINITIONS = {
    2: "Logistic Regression",
    3: "Random Forest",
    4: "Gradient Boosting",
}

# Tie-break cuối: số nhỏ hơn được xem là model đơn giản hơn.
SIMPLICITY_RANK = {2: 1, 3: 2, 4: 3}

MANUAL_CONFIG_SCHEMA_VERSION = 4

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

# Giá trị hiển thị ban đầu trên form Tuning Lab, chưa phải config đã chốt.
MANUAL_BASELINE_PARAMS = {
    "logistic_regression": {"C": 4.12316e-7, "solver": "liblinear"},
    "random_forest": {
        "n_estimators": 130,
        "max_depth": 6,
        "min_samples_leaf": 40,
        "max_features": 0.25,
    },
    "gradient_boosting": {
        "n_estimators": 120,
        "learning_rate": 0.2,
        "max_depth": 2,
        "subsample": 0.9,
    },
}

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
# Tuning Lab experiment state, separate from published model/report outputs.
EXPERIMENTS_DIR = BASE_DIR / "experiments"
TUNING_HISTORY_PATH = EXPERIMENTS_DIR / "tuning_history.csv"
MANUAL_CONFIG_PATH = EXPERIMENTS_DIR / "manual_config.json"
EVALUATION_REGISTRY_PATH = EXPERIMENTS_DIR / "evaluation_registry.json"
PIPELINE_LOCK_PATH = EXPERIMENTS_DIR / "pipeline.lock"
LAST_PIPELINE_RUN_LOG = EXPERIMENTS_DIR / "last_pipeline_run.log"
FETCH_LOCK_PATH = EXPERIMENTS_DIR / "fetch.lock"
LAST_FETCH_RUN_LOG = EXPERIMENTS_DIR / "last_fetch_run.log"

CLEANED_DATA_PATH = PROCESSED_DIR / "hose_stock_clean.csv"
FEATURE_DATA_PATH = PROCESSED_DIR / "hose_stock_features.csv"
ML_DATA_PATH = PROCESSED_DIR / "ml_dataset.csv"

FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.csv"
ELIGIBLE_SYMBOLS_PATH = REPORTS_DIR / "eligible_symbols.csv"
EXCLUDED_SYMBOLS_PATH = REPORTS_DIR / "excluded_symbols.csv"
SPLIT_SUMMARY_PATH = REPORTS_DIR / "split_summary.csv"
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

COMPARISON_COLUMNS = [
    "model_id",
    "model_name",
    "cv_f1_up",
    "decision_threshold",
    "accuracy",
    "precision_up",
    "recall_up",
    "f1_up",
    "precision_not_up",
    "recall_not_up",
    "f1_not_up",
    "selected",
]
