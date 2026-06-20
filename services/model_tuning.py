from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import (
    BEST_PARAMS_PATH,
    CV_FOLD_RESULTS_PATH,
    CV_GAP,
    CV_N_SPLITS,
    FEATURE_COLUMNS,
    MODEL_DEFINITIONS,
    MODEL_PATHS,
    PREDICTION_HORIZON,
    RANDOM_STATE,
    TUNING_N_ITER,
    TUNING_RESULTS_PATH,
    TUNING_SCORING,
    UP_THRESHOLD,
)
from services.pipeline_utils import write_json

TUNABLE_MODEL_IDS = [2, 3, 4]


def _f1_up_scorer():
    return make_scorer(f1_score, pos_label=1)


def _time_series_cv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)


def _build_search_spaces() -> dict:
    return {
        2: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        3: (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": [80, 120, 160, 200],
                "max_depth": [6, 8, 10, 12, None],
                "min_samples_leaf": [10, 20, 30, 50],
                "max_features": ["sqrt", "log2", 0.5],
            },
        ),
        4: (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [60, 80, 100, 120],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
            },
        ),
    }


def _extract_cv_fold_results(
    search: RandomizedSearchCV, model_id: int, X: pd.DataFrame
) -> list[dict]:
    rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(search.cv.split(X)):
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_DEFINITIONS[model_id],
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
    return rows


def tune_models(train: pd.DataFrame) -> tuple[dict[int, dict], pd.DataFrame, dict]:
    """Tune LogReg, RF, GB with TimeSeriesSplit(gap=5); fit Dummy baseline."""
    train_sorted = train.sort_values("trading_date").reset_index(drop=True)
    X_train = train_sorted[FEATURE_COLUMNS]
    y_train = train_sorted["target"]

    cv = _time_series_cv()
    scorer = _f1_up_scorer() if TUNING_SCORING == "f1" else "accuracy"
    search_spaces = _build_search_spaces()

    tuning_rows = []
    cv_fold_rows = []
    best_params = {}
    fitted_artifacts: dict[int, dict] = {}

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_artifact = _make_artifact(1, dummy, cv_f1_up=None, best_params={})
    joblib.dump(dummy_artifact, MODEL_PATHS[1])
    fitted_artifacts[1] = dummy_artifact

    for model_id in TUNABLE_MODEL_IDS:
        estimator, param_dist = search_spaces[model_id]
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=TUNING_N_ITER,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        cv_f1_up = float(search.best_score_)
        best_model = search.best_estimator_
        params = {k: (None if v is None else v) for k, v in search.best_params_.items()}

        artifact = _make_artifact(model_id, best_model, cv_f1_up=cv_f1_up, best_params=params)
        joblib.dump(artifact, MODEL_PATHS[model_id])
        fitted_artifacts[model_id] = artifact

        tuning_rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_DEFINITIONS[model_id],
                "cv_f1_up": cv_f1_up,
                "best_score": cv_f1_up,
                "n_iter": TUNING_N_ITER,
                "cv_n_splits": CV_N_SPLITS,
                "cv_gap": CV_GAP,
            }
        )
        best_params[MODEL_DEFINITIONS[model_id]] = params
        cv_fold_rows.extend(_extract_cv_fold_results(search, model_id, X_train))

    tuning_df = pd.DataFrame(tuning_rows)
    cv_fold_df = pd.DataFrame(cv_fold_rows)

    TUNING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False)
    cv_fold_df.to_csv(CV_FOLD_RESULTS_PATH, index=False)
    write_json(
        BEST_PARAMS_PATH,
        {
            "cv_n_splits": CV_N_SPLITS,
            "cv_gap": CV_GAP,
            "tuning_n_iter": TUNING_N_ITER,
            "scoring": TUNING_SCORING,
            "best_params": best_params,
        },
    )

    return fitted_artifacts, tuning_df, {
        "tuned_models": len(TUNABLE_MODEL_IDS),
        "cv_n_splits": CV_N_SPLITS,
        "cv_gap": CV_GAP,
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
