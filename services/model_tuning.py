from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import (
    BEST_PARAMS_PATH,
    CV_GAP,
    CV_N_SPLITS,
    FEATURE_COLUMNS,
    MANUAL_CONFIG_SCHEMA_VERSION,
    MODEL_DEFINITIONS,
    MODEL_KEY,
    MODEL_PATHS,
    PREDICTION_HORIZON,
    RANDOM_STATE,
    TUNING_RESULTS_PATH,
    TUNING_SCORING,
    UP_THRESHOLD,
)
from services.experiment_state import compute_dataset_fingerprint, read_manual_config
from services.pipeline_utils import write_json

TUNABLE_MODEL_IDS = [2, 3, 4]
REQUIRED_MODEL_KEYS = ["logistic_regression", "random_forest", "gradient_boosting"]


def _f1_up_scorer():
    return make_scorer(f1_score, pos_label=1)


def _time_series_cv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)


def build_estimator(model_id: int, params: dict):
    """Construct a fresh estimator for a given model id from manual params.

    Shared by the Tuning Lab (CV evaluation) and the official pipeline so both
    use identical model construction. RandomForest keeps n_jobs=-1; callers must
    run cross-validation with n_jobs=1 to avoid nested parallelism.
    """
    params = params or {}
    if model_id == 2:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        solver=params.get("solver", "lbfgs"),
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_id == 3:
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_id == 4:
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 1.0)),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model_id for build_estimator: {model_id}")


def run_cv_metrics(estimator, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run TimeSeriesSplit CV on TRAIN and return F1/precision/recall for UP.

    Uses n_jobs=1 in cross_validate; RandomForest parallelizes internally
    (n_jobs=-1) so we avoid oversubscribing CPU cores.
    """
    cv = _time_series_cv()
    scoring = {
        "f1_up": _f1_up_scorer(),
        "precision_up": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_up": make_scorer(recall_score, pos_label=1, zero_division=0),
    }
    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    f1_folds = results["test_f1_up"]
    return {
        "f1_up_mean": float(np.mean(f1_folds)),
        "f1_up_std": float(np.std(f1_folds)),
        "f1_up_folds": [float(v) for v in f1_folds],
        "precision_up_mean": float(np.mean(results["test_precision_up"])),
        "recall_up_mean": float(np.mean(results["test_recall_up"])),
    }


def tune_models(train: pd.DataFrame) -> tuple[dict[int, dict], pd.DataFrame, dict]:
    """Train models from manual_config.json, re-running CV on the current TRAIN.

    The CV score in manual_config is only provenance; here we always recompute
    CV against the current data so the reported numbers stay correct even after
    the dataset changes.
    """
    train_sorted = train.sort_values("trading_date").reset_index(drop=True)
    X_train = train_sorted[FEATURE_COLUMNS]
    y_train = train_sorted["target"]

    config = read_manual_config()
    selected = config.get("selected_models", {}) if config else {}
    if not selected:
        raise ValueError(
            "Chua co cau hinh trong experiments/manual_config.json. Hay vao Tuning "
            "Lab (/tuning) de chon cau hinh cho 3 model truoc khi chay pipeline."
        )
    if config.get("schema_version") != MANUAL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "manual_config.json sai schema_version. Hay chon lai cau hinh trong Tuning Lab."
        )

    missing = [key for key in REQUIRED_MODEL_KEYS if key not in selected]
    if missing:
        raise ValueError(
            f"Thieu cau hinh cho: {', '.join(missing)}. Hay chot du 3 model trong Tuning Lab."
        )

    current = compute_dataset_fingerprint()
    if config.get("dataset_fingerprint") != current["hash"]:
        raise ValueError(
            "manual_config.json thuoc dataset khac (fingerprint khong khop). "
            "Hay tuning lai trong Tuning Lab voi du lieu hien tai."
        )

    tuning_rows = []
    best_params = {}
    fitted_artifacts: dict[int, dict] = {}

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_artifact = _make_artifact(1, dummy, cv_f1_up=None, best_params={})
    joblib.dump(dummy_artifact, MODEL_PATHS[1])
    fitted_artifacts[1] = dummy_artifact

    for model_id in TUNABLE_MODEL_IDS:
        model_key = MODEL_KEY[model_id]
        params = selected[model_key].get("params", {})

        estimator = build_estimator(model_id, params)
        cv = run_cv_metrics(estimator, X_train, y_train)
        estimator.fit(X_train, y_train)

        artifact = _make_artifact(
            model_id, estimator, cv_f1_up=cv["f1_up_mean"], best_params=params
        )
        artifact["cv_metrics"] = cv
        joblib.dump(artifact, MODEL_PATHS[model_id])
        fitted_artifacts[model_id] = artifact

        tuning_rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_DEFINITIONS[model_id],
                "cv_f1_up": cv["f1_up_mean"],
                "cv_f1_up_std": cv["f1_up_std"],
                "cv_precision_up": cv["precision_up_mean"],
                "cv_recall_up": cv["recall_up_mean"],
                "best_score": cv["f1_up_mean"],
                "cv_n_splits": CV_N_SPLITS,
                "cv_gap": CV_GAP,
            }
        )
        best_params[MODEL_DEFINITIONS[model_id]] = params

    tuning_df = pd.DataFrame(tuning_rows)
    TUNING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False)
    write_json(
        BEST_PARAMS_PATH,
        {
            "cv_n_splits": CV_N_SPLITS,
            "cv_gap": CV_GAP,
            "scoring": TUNING_SCORING,
            "source": "manual_config",
            "dataset_fingerprint": current["hash"],
            "best_params": best_params,
        },
    )

    return fitted_artifacts, tuning_df, {
        "tuned_models": len(TUNABLE_MODEL_IDS),
        "cv_n_splits": CV_N_SPLITS,
        "cv_gap": CV_GAP,
        "source": "manual_config",
        "dataset_fingerprint": current["hash"],
    }


def _make_artifact(
    model_id: int,
    model,
    cv_f1_up: float | None,
    best_params: dict,
) -> dict:
    return {
        "model": model,
        "model_id": model_id,
        "model_name": MODEL_DEFINITIONS[model_id],
        "feature_columns": FEATURE_COLUMNS,
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cv_f1_up": cv_f1_up,
        "best_params": best_params,
        "cv_config": {"n_splits": CV_N_SPLITS, "gap": CV_GAP},
    }
