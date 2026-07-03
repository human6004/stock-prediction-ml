"""Tuning Lab logic: parameter schema, server-side validation, and CV evaluation.

Powers the manual tuning workflow at /tuning. It evaluates a single user config
with TimeSeriesSplit CV on the TRAIN split only and records each run into the
experiment history. It never touches the TEST split.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    FEATURE_COLUMNS,
    MANUAL_BASELINE_PARAMS,
    ML_DATA_PATH,
    MODEL_DEFINITIONS,
    MODEL_KEY_TO_ID,
    TUNABLE_PARAM_SCHEMA,
)
from services.experiment_state import append_history, compute_dataset_fingerprint
from services.feature_engineering import time_based_split
from services.model_tuning import build_estimator, run_cv_metrics

MODEL_RUN_PREFIX = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "gradient_boosting": "GB",
}


def get_param_schema() -> dict:
    return TUNABLE_PARAM_SCHEMA


def get_param_defaults() -> dict:
    return MANUAL_BASELINE_PARAMS


def load_train() -> pd.DataFrame:
    """Return the TRAIN split (label_end_date <= SPLIT_DATE) for CV."""
    if not Path(ML_DATA_PATH).exists():
        raise FileNotFoundError(
            "Chua co data/processed/ml_dataset.csv. Hay chay buoc chuan bi du lieu "
            "(python scripts/preprocess_data.py va python scripts/build_features.py) "
            "hoac chay pipeline mot lan truoc khi dung Tuning Lab."
        )
    dataset = pd.read_csv(ML_DATA_PATH)
    train, _, _ = time_based_split(dataset)
    return train


# --------------------------------------------------------------------------- #
# Server-side validation
# --------------------------------------------------------------------------- #
def _check_bounds(name: str, value: float, spec: dict) -> None:
    minimum = spec.get("min")
    if minimum is not None:
        if spec.get("inclusive_min", True):
            if value < minimum:
                raise ValueError(f"{name} phai >= {minimum}.")
        elif value <= minimum:
            raise ValueError(f"{name} phai > {minimum}.")
    maximum = spec.get("max")
    if maximum is not None:
        if spec.get("inclusive_max", True):
            if value > maximum:
                raise ValueError(f"{name} phai <= {maximum}.")
        elif value >= maximum:
            raise ValueError(f"{name} phai < {maximum}.")


def _parse_int(name: str, raw, spec: dict) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name} phai la so nguyen.")
    _check_bounds(name, value, spec)
    return value


def _parse_float(name: str, raw, spec: dict) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name} phai la so thuc.")
    _check_bounds(name, value, spec)
    return value


def validate_params(model_key: str, raw_form: dict) -> dict:
    """Validate and coerce raw form values for one model.

    Raises ValueError with a clear Vietnamese message on any invalid input.
    """
    if model_key not in TUNABLE_PARAM_SCHEMA:
        raise ValueError(f"Model khong hop le: {model_key}")

    schema = TUNABLE_PARAM_SCHEMA[model_key]
    params: dict = {}
    for name, spec in schema.items():
        raw = raw_form.get(name)
        field_type = spec["type"]

        if field_type == "int":
            params[name] = _parse_int(name, raw, spec)
        elif field_type == "float":
            params[name] = _parse_float(name, raw, spec)
        elif field_type == "int_or_none":
            text = "" if raw is None else str(raw).strip()
            if text == "" or text.lower() == "none":
                params[name] = None
            else:
                params[name] = _parse_int(name, raw, spec)
        elif field_type == "choice":
            text = "" if raw is None else str(raw).strip()
            if text not in spec["choices"]:
                raise ValueError(f"{name} chi nhan: {', '.join(spec['choices'])}.")
            params[name] = text
        elif field_type == "str_or_float":
            text = "" if raw is None else str(raw).strip()
            if text in spec.get("choices", []):
                params[name] = text
            else:
                params[name] = _parse_float(name, raw, spec)
        else:
            raise ValueError(f"Kieu tham so khong ho tro: {field_type}")
    return params


# --------------------------------------------------------------------------- #
# Single-config CV evaluation
# --------------------------------------------------------------------------- #
def _make_run_id(model_key: str) -> str:
    prefix = MODEL_RUN_PREFIX.get(model_key, "MM")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def evaluate_single_config(
    model_key: str,
    params: dict,
    train: pd.DataFrame,
    note: str = "",
) -> dict:
    """Run CV for one config on TRAIN, record history, and return flat results.

    On CV failure, an error row is recorded in history and the exception is
    re-raised so the route can show a clear message (the app never crashes).
    """
    model_id = MODEL_KEY_TO_ID[model_key]
    fingerprint = compute_dataset_fingerprint()

    X = train[FEATURE_COLUMNS]
    y = train["target"]

    estimator = build_estimator(model_id, params)
    run_id = _make_run_id(model_key)
    started = time.perf_counter()
    try:
        cv = run_cv_metrics(estimator, X, y)
    except Exception as exc:  # noqa: BLE001 - record failure then re-raise
        append_history(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_key": model_key,
                "model_name": MODEL_DEFINITIONS[model_id],
                "params_json": json.dumps(params, ensure_ascii=False),
                "cv_f1_up_mean": "",
                "cv_f1_up_std": "",
                "cv_f1_up_folds_json": "[]",
                "train_seconds": round(time.perf_counter() - started, 3),
                "dataset_fingerprint": fingerprint["hash"],
                "status": "error",
                "note": str(exc),
            }
        )
        raise

    train_seconds = round(time.perf_counter() - started, 3)
    append_history(
        {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "model_name": MODEL_DEFINITIONS[model_id],
            "params_json": json.dumps(params, ensure_ascii=False),
            "cv_f1_up_mean": cv["f1_up_mean"],
            "cv_f1_up_std": cv["f1_up_std"],
            "cv_f1_up_folds_json": json.dumps(cv["f1_up_folds"]),
            "train_seconds": train_seconds,
            "dataset_fingerprint": fingerprint["hash"],
            "status": "ok",
            "note": note,
        }
    )

    return {
        "run_id": run_id,
        "model_key": model_key,
        "model_name": MODEL_DEFINITIONS[model_id],
        "params": params,
        "cv_f1_up_mean": cv["f1_up_mean"],
        "cv_f1_up_std": cv["f1_up_std"],
        "cv_f1_up_folds": cv["f1_up_folds"],
        "precision_up_mean": cv["precision_up_mean"],
        "recall_up_mean": cv["recall_up_mean"],
        "train_seconds": train_seconds,
        "dataset_fingerprint": fingerprint["hash"],
    }
