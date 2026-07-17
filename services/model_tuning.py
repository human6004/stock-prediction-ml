from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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


def tune_threshold(y_true, proba) -> float:
    """Return the P(UP) cutoff that maximizes F1 for the UP class.

    Target is imbalanced (~39% UP) so the default 0.5 cutoff under-predicts UP.
    Clamped to [0.05, 0.95] to reject degenerate all-positive/all-negative rules.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba, pos_label=1)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    return float(np.clip(thresholds[best], 0.05, 0.95))


def _proba_up(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)]


def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """Classify UP=1 when P(UP) >= threshold instead of sklearn's 0.5 default."""
    return (_proba_up(model, X) >= threshold).astype(int)


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


def run_cv_metrics(estimator, X: pd.DataFrame, y: pd.Series, model_id: int | None = None) -> dict:
    """Run TimeSeriesSplit CV on TRAIN and return F1/precision/recall for UP.

    The decision threshold is tuned on each TRAIN fold and applied to the held-out
    VAL fold (never tuned on VAL), so the reported F1 is honest. The target is
    imbalanced (~39% UP), so scoring at sklearn's default 0.5 cutoff systematically
    under-predicts UP; a fold-tuned cutoff fixes that without leakage.

    GradientBoostingClassifier has no class_weight param (unlike RandomForest's
    balanced_subsample and LogisticRegression's balanced), so when model_id == 4
    we pass a "balanced" sample_weight into fit() to reach the same effect.
    """
    cv = _time_series_cv()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_arr = np.asarray(y)

    f1_folds, precision_folds, recall_folds, threshold_folds = [], [], [], []
    for train_idx, val_idx in cv.split(X):
        model = clone(estimator)
        X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
        X_val, y_val = X.iloc[val_idx], y_arr[val_idx]
        if model_id == 4:
            model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        else:
            model.fit(X_tr, y_tr)

        threshold = tune_threshold(y_tr, _proba_up(model, X_tr))
        y_pred = (_proba_up(model, X_val) >= threshold).astype(int)
        f1_folds.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        precision_folds.append(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
        recall_folds.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        threshold_folds.append(threshold)

    return {
        "f1_up_mean": float(np.mean(f1_folds)),
        "f1_up_std": float(np.std(f1_folds)),
        "f1_up_folds": [float(v) for v in f1_folds],
        "precision_up_mean": float(np.mean(precision_folds)),
        "recall_up_mean": float(np.mean(recall_folds)),
        "decision_threshold": float(np.mean(threshold_folds)),
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
    dummy_artifact = _make_artifact(
        1, dummy, cv_f1_up=None, best_params={}, decision_threshold=0.5
    )
    joblib.dump(dummy_artifact, MODEL_PATHS[1])
    fitted_artifacts[1] = dummy_artifact

    for model_id in TUNABLE_MODEL_IDS:
        model_key = MODEL_KEY[model_id]
        params = selected[model_key].get("params", {})

        estimator = build_estimator(model_id, params)
        cv = run_cv_metrics(estimator, X_train, y_train, model_id=model_id)
        if model_id == 4:
            estimator.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            estimator.fit(X_train, y_train)

        decision_threshold = tune_threshold(y_train, _proba_up(estimator, X_train))
        artifact = _make_artifact(
            model_id,
            estimator,
            cv_f1_up=cv["f1_up_mean"],
            best_params=params,
            decision_threshold=decision_threshold,
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
    decision_threshold: float,
) -> dict:
    return {
        "model": model,
        "model_id": model_id,
        "model_name": MODEL_DEFINITIONS[model_id],
        "feature_columns": FEATURE_COLUMNS,
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": decision_threshold,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cv_f1_up": cv_f1_up,
        "best_params": best_params,
        "cv_config": {"n_splits": CV_N_SPLITS, "gap": CV_GAP},
    }
