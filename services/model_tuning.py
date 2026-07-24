"""Xây estimator, chạy CV trên TRAIN và fit model artifact.

Tuning Lab và official pipeline dùng chung các hàm ở đây để cấu hình model giống
nhau. Module không đánh giá TEST; TEST metrics nằm ở model_evaluation.py.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import (
    BEST_PARAMS_PATH,
    CV_GAP_SESSIONS,
    CV_FOLD_RESULTS_PATH,
    CV_N_SPLITS,
    DECISION_THRESHOLD,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    MANUAL_CONFIG_SCHEMA_VERSION,
    MODEL_DEFINITIONS,
    MODEL_KEY,
    PREDICTION_HORIZON,
    RANDOM_STATE,
    TUNING_RESULTS_PATH,
    TUNING_SCORING,
    THRESHOLD_MAX,
    THRESHOLD_MAX_PREDICTED_UP_RATIO,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    UP_THRESHOLD,
)
from services.experiment_state import (
    compute_dataset_fingerprint,
    is_config_complete,
    read_manual_config,
)
from services.pipeline_utils import atomic_dataframe_to_csv, write_json
from services.time_splitting import iter_purged_date_splits, sort_panel_frame

TUNABLE_MODEL_IDS = [2, 3, 4]
REQUIRED_MODEL_KEYS = ["logistic_regression", "random_forest", "gradient_boosting"]


def _proba_up(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)]


def predict_with_threshold(model, X, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    """Classify UP=1 from the project's explicit decision policy."""
    return (_proba_up(model, X) >= threshold).astype(int)


def select_oof_threshold(y_true, probabilities) -> dict:
    """Choose one deterministic F1 threshold without collapsing to always-UP."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("OOF labels and probabilities must be non-empty and aligned.")

    up_rate = float(np.mean(y_true))
    candidates = []
    for threshold in np.arange(
        THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP
    ):
        threshold = round(float(threshold), 2)
        predicted = (probabilities >= threshold).astype(int)
        predicted_up_ratio = float(np.mean(predicted))
        precision = float(
            precision_score(y_true, predicted, pos_label=1, zero_division=0)
        )
        if (
            predicted_up_ratio <= THRESHOLD_MAX_PREDICTED_UP_RATIO + 1e-12
            and precision + 1e-12 >= up_rate
        ):
            candidates.append(
                (
                    float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
                    precision,
                    threshold,
                    predicted_up_ratio,
                    float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
                )
            )
    if not candidates:
        raise ValueError("No OOF threshold satisfies precision and UP-ratio constraints.")

    f1_up, precision_up, threshold, predicted_up_ratio, recall_up = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return {
        "decision_threshold": threshold,
        "oof_f1_up": f1_up,
        "oof_precision_up": precision_up,
        "oof_recall_up": recall_up,
        "oof_up_rate": up_rate,
        "oof_predicted_up_ratio": predicted_up_ratio,
        "threshold_constraint_passed": True,
    }


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


def run_cv_metrics(
    estimator, train: pd.DataFrame, model_id: int | None = None
) -> dict:
    """Run recent purged CV, choose one threshold from pooled OOF probabilities."""
    required = {*FEATURE_COLUMNS, "symbol", "trading_date", "label_end_date", "target"}
    if missing := required - set(train.columns):
        raise ValueError(f"TRAIN missing CV columns: {sorted(missing)}")

    frame = sort_panel_frame(train)
    X = frame[FEATURE_COLUMNS]
    y_arr = frame["target"].to_numpy()
    fold_outputs, fold_ranges = [], []
    for fold_number, (train_idx, val_idx) in enumerate(
        iter_purged_date_splits(frame), start=1
    ):
        model = clone(estimator)
        X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
        X_val, y_val = X.iloc[val_idx], y_arr[val_idx]
        if model_id == 4:
            model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        else:
            model.fit(X_tr, y_tr)

        fold_outputs.append((y_val, _proba_up(model, X_val)))

        train_fold = frame.iloc[train_idx]
        validation_fold = frame.iloc[val_idx]
        fold_ranges.append(
            {
                "fold": fold_number,
                "train_start": str(train_fold["trading_date"].min()),
                "train_end": str(train_fold["trading_date"].max()),
                "train_label_end_max": str(train_fold["label_end_date"].max()),
                "validation_start": str(validation_fold["trading_date"].min()),
                "validation_end": str(validation_fold["trading_date"].max()),
                "train_rows": int(len(train_fold)),
                "validation_rows": int(len(validation_fold)),
            }
        )

    threshold_metrics = select_oof_threshold(
        np.concatenate([item[0] for item in fold_outputs]),
        np.concatenate([item[1] for item in fold_outputs]),
    )
    threshold = threshold_metrics["decision_threshold"]
    f1_folds, precision_folds, recall_folds = [], [], []
    for y_val, probabilities in fold_outputs:
        y_pred = (probabilities >= threshold).astype(int)
        f1_folds.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        precision_folds.append(
            precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        )
        recall_folds.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))

    return {
        "f1_up_mean": float(np.mean(f1_folds)),
        "f1_up_std": float(np.std(f1_folds)),
        "f1_up_folds": [float(v) for v in f1_folds],
        "precision_up_folds": [float(v) for v in precision_folds],
        "recall_up_folds": [float(v) for v in recall_folds],
        "precision_up_mean": float(np.mean(precision_folds)),
        "recall_up_mean": float(np.mean(recall_folds)),
        "fold_date_ranges": fold_ranges,
        **threshold_metrics,
    }


def _cv_from_selected(choice: dict) -> dict:
    """Restore the exact CV result already paid for in Tuning Lab."""
    cv = {
        "f1_up_mean": choice.get("cv_f1_up_mean"),
        "f1_up_std": choice.get("cv_f1_up_std"),
        "f1_up_folds": list(choice.get("f1_up_folds") or []),
        "precision_up_mean": choice.get("cv_precision_up_mean"),
        "precision_up_folds": list(choice.get("precision_up_folds") or []),
        "recall_up_mean": choice.get("cv_recall_up_mean"),
        "recall_up_folds": list(choice.get("recall_up_folds") or []),
        "fold_date_ranges": list(choice.get("fold_date_ranges") or []),
        "decision_threshold": choice.get("decision_threshold"),
        "oof_f1_up": choice.get("oof_f1_up"),
        "oof_precision_up": choice.get("oof_precision_up"),
        "oof_recall_up": choice.get("oof_recall_up"),
        "oof_up_rate": choice.get("oof_up_rate"),
        "oof_predicted_up_ratio": choice.get("oof_predicted_up_ratio"),
        "threshold_constraint_passed": choice.get("threshold_constraint_passed"),
    }
    required = (
        "f1_up_mean",
        "f1_up_std",
        "precision_up_mean",
        "recall_up_mean",
        "decision_threshold",
        "oof_up_rate",
        "oof_predicted_up_ratio",
    )
    if any(cv[field] is None for field in required) or any(
        len(cv[field]) != CV_N_SPLITS
        for field in (
            "f1_up_folds",
            "precision_up_folds",
            "recall_up_folds",
            "fold_date_ranges",
        )
    ):
        raise ValueError("Selected tuning run lacks complete CV provenance.")
    return cv


def tune_models(train: pd.DataFrame) -> tuple[dict[int, dict], pd.DataFrame, dict]:
    """Fit selected configs once on TRAIN, reusing their fingerprinted CV runs."""
    train_sorted = sort_panel_frame(train)
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
    if config.get("policy_id") != EXPERIMENT_POLICY_ID:
        raise ValueError("manual_config.json không thuộc experiment policy hiện hành.")
    if config.get("dataset_fingerprint") != current["hash"]:
        raise ValueError(
            "manual_config.json thuoc dataset khac (fingerprint khong khop). "
            "Hay tuning lai trong Tuning Lab voi du lieu hien tai."
        )
    if config.get("content_fingerprint") != current.get("content_hash"):
        raise ValueError("manual_config.json không khớp content fingerprint hiện tại.")
    if not is_config_complete(config, current["hash"]):
        raise ValueError(
            "Cần cấu hình CV do người dùng tự chọn cho cả LR, RF, GB."
        )

    tuning_rows = []
    fold_rows = []
    best_params = {}
    fitted_artifacts: dict[int, dict] = {}

    for model_id in TUNABLE_MODEL_IDS:
        model_key = MODEL_KEY[model_id]
        choice = selected[model_key]
        params = choice.get("params", {})

        # CV đã chạy một lần trong Tuning Lab; pipeline chỉ fit full TRAIN.
        estimator = build_estimator(model_id, params)
        cv = _cv_from_selected(choice)
        if model_id == 4:
            estimator.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            estimator.fit(X_train, y_train)

        artifact = _make_artifact(
            model_id,
            estimator,
            cv_f1_up=cv["f1_up_mean"],
            best_params=params,
            decision_threshold=float(cv["decision_threshold"]),
        )
        artifact["cv_metrics"] = cv
        artifact["policy_id"] = EXPERIMENT_POLICY_ID
        artifact["content_fingerprint"] = current.get("content_hash", current["hash"])
        artifact["train_through_date"] = str(train_sorted["label_end_date"].max())
        fitted_artifacts[model_id] = artifact

        tuning_rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_DEFINITIONS[model_id],
                "cv_f1_up": cv["f1_up_mean"],
                "cv_f1_up_std": cv["f1_up_std"],
                "cv_precision_up": cv["precision_up_mean"],
                "cv_recall_up": cv["recall_up_mean"],
                "decision_threshold": cv["decision_threshold"],
                "oof_predicted_up_ratio": cv["oof_predicted_up_ratio"],
                "threshold_constraint_passed": cv["threshold_constraint_passed"],
                "best_score": cv["f1_up_mean"],
                "cv_n_splits": CV_N_SPLITS,
                "cv_gap": CV_GAP_SESSIONS,
            }
        )
        best_params[MODEL_DEFINITIONS[model_id]] = params
        for index, fold_range in enumerate(cv["fold_date_ranges"]):
            fold_rows.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_DEFINITIONS[model_id],
                    "policy_id": EXPERIMENT_POLICY_ID,
                    "content_fingerprint": current.get("content_hash", current["hash"]),
                    "params_json": json.dumps(params, sort_keys=True, ensure_ascii=False),
                    **fold_range,
                    "f1_up": cv["f1_up_folds"][index],
                    "precision_up": cv["precision_up_folds"][index],
                    "recall_up": cv["recall_up_folds"][index],
                    "decision_threshold": cv["decision_threshold"],
                }
            )

    tuning_df = pd.DataFrame(tuning_rows)
    TUNING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_to_csv(tuning_df, TUNING_RESULTS_PATH, index=False)
    atomic_dataframe_to_csv(
        pd.DataFrame(fold_rows), CV_FOLD_RESULTS_PATH, index=False
    )
    write_json(
        BEST_PARAMS_PATH,
        {
            "cv_n_splits": CV_N_SPLITS,
            "cv_gap": CV_GAP_SESSIONS,
            "scoring": TUNING_SCORING,
            "source": "selected_tuning_run",
            "dataset_fingerprint": current["hash"],
            "best_params": best_params,
        },
    )

    return fitted_artifacts, tuning_df, {
        "tuned_models": len(TUNABLE_MODEL_IDS),
        "cv_n_splits": CV_N_SPLITS,
        "cv_gap": CV_GAP_SESSIONS,
        "source": "selected_tuning_run",
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
        "feature_order": FEATURE_COLUMNS,
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": decision_threshold,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cv_f1_up": cv_f1_up,
        "best_params": best_params,
        "cv_config": {"n_splits": CV_N_SPLITS, "gap_sessions": CV_GAP_SESSIONS},
    }
