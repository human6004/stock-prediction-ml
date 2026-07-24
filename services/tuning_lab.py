"""Tuning Lab logic: parameter schema, server-side validation, and CV evaluation.

Powers the manual tuning workflow at /tuning. It evaluates one user-entered
config with purged date-based CV on TRAIN and records each run. It never touches
VALIDATION or TEST.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    CV_N_SPLITS,
    EXPERIMENT_POLICY_ID,
    MANUAL_BASELINE_PARAMS,
    ML_DATA_PATH,
    MODEL_DEFINITIONS,
    MODEL_KEY_TO_ID,
    TUNABLE_PARAM_SCHEMA,
    THRESHOLD_MAX_PREDICTED_UP_RATIO,
)
from services.experiment_state import (
    append_history,
    compute_dataset_fingerprint,
)
from services.feature_engineering import protocol_time_split
from services.model_tuning import build_estimator, run_cv_metrics

MODEL_RUN_PREFIX = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "gradient_boosting": "GB",
}

# ponytail: process-local lock is enough for this single-process Flask app; use a
# file/DB lock before deploying with multiple worker processes.
_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None
_JOB_STATE = {"status": "idle"}


def _validate_cv_provenance(cv: dict) -> None:
    required_fold_fields = (
        "f1_up_folds",
        "precision_up_folds",
        "recall_up_folds",
        "fold_date_ranges",
    )
    for field in required_fold_fields:
        if len(cv.get(field) or []) != CV_N_SPLITS:
            raise ValueError(f"CV thiếu provenance {field} cho {CV_N_SPLITS} fold.")
    threshold = float(cv.get("decision_threshold", -1))
    if not 0 < threshold < 1:
        raise ValueError("CV không có decision threshold hợp lệ.")
    if not cv.get("threshold_constraint_passed"):
        raise ValueError("OOF threshold không vượt qua ràng buộc.")
    if float(cv.get("oof_predicted_up_ratio", 1)) > THRESHOLD_MAX_PREDICTED_UP_RATIO:
        raise ValueError("OOF threshold dự báo UP quá giới hạn.")
    if float(cv.get("oof_precision_up", 0)) < float(cv.get("oof_up_rate", 1)):
        raise ValueError("OOF threshold không đạt precision floor.")


def get_param_schema() -> dict:
    return TUNABLE_PARAM_SCHEMA


def get_param_defaults() -> dict:
    return {model_key: dict(params) for model_key, params in MANUAL_BASELINE_PARAMS.items()}


def load_train() -> pd.DataFrame:
    """Return TRAIN for current-policy date-purged CV."""
    if not Path(ML_DATA_PATH).exists():
        raise FileNotFoundError(
            "Chua co data/processed/ml_dataset.csv. Hay chay buoc chuan bi du lieu "
            "(python scripts/preprocess_data.py va python scripts/build_features.py) "
            "hoac chay pipeline mot lan truoc khi dung Tuning Lab."
        )
    dataset = pd.read_csv(ML_DATA_PATH)
    return protocol_time_split(dataset)[0]


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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def evaluate_single_config(
    model_key: str,
    params: dict,
    train: pd.DataFrame,
    note: str = "",
    fingerprint: dict | None = None,
) -> dict:
    """Run CV for one config on TRAIN, record history, and return flat results.

    On CV failure, an error row is recorded in history and the exception is
    re-raised so the route can show a clear message (the app never crashes).
    """
    model_id = MODEL_KEY_TO_ID[model_key]
    fingerprint = fingerprint or compute_dataset_fingerprint()

    estimator = build_estimator(model_id, params)
    run_id = _make_run_id(model_key)
    started = time.perf_counter()
    try:
        cv = run_cv_metrics(estimator, train, model_id=model_id)
        _validate_cv_provenance(cv)
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
                "cv_precision_up_folds_json": "[]",
                "cv_recall_up_folds_json": "[]",
                "cv_fold_ranges_json": "[]",
                "decision_threshold": "",
                "train_seconds": round(time.perf_counter() - started, 3),
                "dataset_fingerprint": fingerprint["hash"],
                "content_fingerprint": fingerprint.get("content_hash", fingerprint["hash"]),
                "policy_id": EXPERIMENT_POLICY_ID,
                "status": "error",
                "note": str(exc),
            }
        )
        raise

    train_seconds = round(time.perf_counter() - started, 3)
    precision_folds = cv.get("precision_up_folds", [])
    recall_folds = cv.get("recall_up_folds", [])
    fold_ranges = cv.get("fold_date_ranges", cv.get("fold_ranges", []))
    precision_std = float(pd.Series(precision_folds).std(ddof=0))
    recall_std = float(pd.Series(recall_folds).std(ddof=0))
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
            "cv_precision_up_mean": cv.get("precision_up_mean"),
            "cv_precision_up_std": precision_std,
            "cv_recall_up_mean": cv.get("recall_up_mean"),
            "cv_recall_up_std": recall_std,
            "cv_precision_up_folds_json": json.dumps(precision_folds),
            "cv_recall_up_folds_json": json.dumps(recall_folds),
            "cv_fold_ranges_json": json.dumps(fold_ranges),
            "decision_threshold": cv["decision_threshold"],
            "oof_f1_up": cv["oof_f1_up"],
            "oof_precision_up": cv["oof_precision_up"],
            "oof_recall_up": cv["oof_recall_up"],
            "oof_up_rate": cv["oof_up_rate"],
            "oof_predicted_up_ratio": cv["oof_predicted_up_ratio"],
            "threshold_constraint_passed": cv["threshold_constraint_passed"],
            "train_seconds": train_seconds,
            "dataset_fingerprint": fingerprint["hash"],
            "content_fingerprint": fingerprint.get("content_hash", fingerprint["hash"]),
            "policy_id": EXPERIMENT_POLICY_ID,
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
        "decision_threshold": cv["decision_threshold"],
        "oof_f1_up": cv["oof_f1_up"],
        "oof_precision_up": cv["oof_precision_up"],
        "oof_recall_up": cv["oof_recall_up"],
        "oof_up_rate": cv["oof_up_rate"],
        "oof_predicted_up_ratio": cv["oof_predicted_up_ratio"],
        "threshold_constraint_passed": cv["threshold_constraint_passed"],
        "precision_up_mean": cv["precision_up_mean"],
        "precision_up_std": precision_std,
        "recall_up_mean": cv["recall_up_mean"],
        "recall_up_std": recall_std,
        "train_seconds": train_seconds,
        "dataset_fingerprint": fingerprint["hash"],
    }


def _run_tuning_job(job_id: str, model_key: str, params: dict, note: str) -> None:
    global _JOB_STATE
    try:
        train = load_train()
        result = evaluate_single_config(model_key, params, train, note=note)
        state = {"status": "completed", "job_id": job_id, "result": result}
    except Exception as exc:  # noqa: BLE001 - surfaced through the job status panel.
        state = {"status": "error", "job_id": job_id, "error": str(exc)}
    with _JOB_LOCK:
        _JOB_STATE = state


def start_tuning_job(model_key: str, params: dict, note: str = "") -> str | None:
    """Start one CV job in-process; return None while another job is active."""
    global _JOB_STATE, _JOB_THREAD
    params = validate_params(model_key, params)
    with _JOB_LOCK:
        if _JOB_THREAD is not None and _JOB_THREAD.is_alive():
            return None
        job_id = f"TUNE_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        _JOB_STATE = {
            "status": "running",
            "job_id": job_id,
            "model_key": model_key,
            "params": dict(params),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        _JOB_THREAD = threading.Thread(
            target=_run_tuning_job,
            args=(job_id, model_key, dict(params), note),
            name="tuning-lab-job",
            daemon=True,
        )
        _JOB_THREAD.start()
        return job_id


def get_tuning_job_state() -> dict:
    with _JOB_LOCK:
        return dict(_JOB_STATE)


def is_tuning_job_running() -> bool:
    return get_tuning_job_state().get("status") == "running"


def wait_for_tuning_job(timeout: float | None = None) -> None:
    with _JOB_LOCK:
        thread = _JOB_THREAD
    if thread is not None:
        thread.join(timeout=timeout)


def _reset_tuning_job_for_tests() -> None:
    global _JOB_STATE, _JOB_THREAD
    with _JOB_LOCK:
        if _JOB_THREAD is not None and _JOB_THREAD.is_alive():
            raise RuntimeError("Cannot reset a running tuning job.")
        _JOB_THREAD = None
        _JOB_STATE = {"status": "idle"}
