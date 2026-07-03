"""State management for the Tuning Lab experiments folder.

Centralizes all IO under ``experiments/``:
- dataset fingerprint (data version identity)
- tuning history CSV (one row per manual training run)
- manual_config.json (the chosen config used by the official pipeline)
- test_evaluation_lock.json (test set used once per dataset fingerprint)
- pipeline.lock (prevents concurrent pipeline runs)
- last_pipeline_run.log (stdout/stderr of the most recent pipeline run)
"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import (
    CV_GAP,
    CV_N_SPLITS,
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    LAST_PIPELINE_RUN_LOG,
    MANUAL_CONFIG_PATH,
    MANUAL_CONFIG_SCHEMA_VERSION,
    ML_DATA_PATH,
    PIPELINE_LOCK_PATH,
    PREDICTION_HORIZON,
    SPLIT_DATE,
    TEST_EVAL_LOCK_PATH,
    TUNING_HISTORY_PATH,
    UP_THRESHOLD,
)

HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "model_key",
    "model_name",
    "params_json",
    "cv_f1_up_mean",
    "cv_f1_up_std",
    "cv_f1_up_folds_json",
    "train_seconds",
    "dataset_fingerprint",
    "status",
    "note",
]

REQUIRED_MODEL_KEYS = ("logistic_regression", "random_forest", "gradient_boosting")

# A pipeline.lock older than this (and/or with a dead PID) is treated as stale.
PIPELINE_LOCK_STALE_SECONDS = 3 * 60 * 60


def ensure_experiments_dir() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Dataset fingerprint
# --------------------------------------------------------------------------- #
def compute_dataset_fingerprint(ml_dataset_df: pd.DataFrame | None = None) -> dict:
    """Return a stable fingerprint of the current ML dataset version.

    Derived from data size/recency plus the labeling/feature config so results
    from different dataset versions are never mixed up.
    """
    if ml_dataset_df is None:
        if not Path(ML_DATA_PATH).exists():
            raise FileNotFoundError(
                "Chua co data/processed/ml_dataset.csv. Hay chay buoc chuan bi du "
                "lieu (preprocess_data.py + build_features.py) hoac chay pipeline truoc."
            )
        ml_dataset_df = pd.read_csv(ML_DATA_PATH, usecols=["trading_date"])

    if "trading_date" in ml_dataset_df.columns and len(ml_dataset_df):
        max_trading_date = str(ml_dataset_df["trading_date"].max())
    else:
        max_trading_date = None

    parts = {
        "dataset_rows": int(len(ml_dataset_df)),
        "max_trading_date": max_trading_date,
        "split_date": SPLIT_DATE,
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return {"hash": digest, "parts": parts}


# --------------------------------------------------------------------------- #
# Tuning history
# --------------------------------------------------------------------------- #
def append_history(record: dict) -> None:
    """Append one training run to the history CSV (creates header if needed)."""
    ensure_experiments_dir()
    file_exists = Path(TUNING_HISTORY_PATH).exists()
    row = {col: record.get(col, "") for col in HISTORY_COLUMNS}
    with open(TUNING_HISTORY_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_history() -> list[dict]:
    if not Path(TUNING_HISTORY_PATH).exists():
        return []
    rows: list[dict] = []
    with open(TUNING_HISTORY_PATH, "r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["cv_f1_up_mean"] = _to_float(raw.get("cv_f1_up_mean"))
            row["cv_f1_up_std"] = _to_float(raw.get("cv_f1_up_std"))
            row["train_seconds"] = _to_float(raw.get("train_seconds"))
            try:
                row["params"] = json.loads(raw.get("params_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["params"] = {}
            try:
                row["folds"] = json.loads(raw.get("cv_f1_up_folds_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["folds"] = []
            row["is_best"] = False
            rows.append(row)
    return rows


def mark_best(rows: list[dict]) -> list[dict]:
    """Flag the best run per (model_key, dataset_fingerprint).

    Best = highest cv_f1_up_mean, tie-break lowest cv_f1_up_std. Only rows with
    status == 'ok' and a numeric score are considered.
    """
    best_index: dict[tuple, int] = {}
    for idx, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        score = row.get("cv_f1_up_mean")
        if score is None:
            continue
        key = (row.get("model_key"), row.get("dataset_fingerprint"))
        current = best_index.get(key)
        if current is None:
            best_index[key] = idx
            continue
        best_row = rows[current]
        best_score = best_row.get("cv_f1_up_mean")
        std = row.get("cv_f1_up_std")
        best_std = best_row.get("cv_f1_up_std")
        if score > best_score or (
            score == best_score
            and std is not None
            and best_std is not None
            and std < best_std
        ):
            best_index[key] = idx

    for idx in best_index.values():
        rows[idx]["is_best"] = True
    return rows


def find_run(run_id: str) -> dict | None:
    if not run_id:
        return None
    for row in read_history():
        if row.get("run_id") == run_id:
            return row
    return None


# --------------------------------------------------------------------------- #
# Manual config
# --------------------------------------------------------------------------- #
def _default_manual_config() -> dict:
    return {
        "schema_version": MANUAL_CONFIG_SCHEMA_VERSION,
        "dataset_fingerprint": "",
        "dataset_fingerprint_parts": {},
        "cv_settings": {"n_splits": CV_N_SPLITS, "gap": CV_GAP, "metric": "f1_up"},
        "selected_models": {},
    }


def _empty_manual_config(fingerprint: dict) -> dict:
    config = _default_manual_config()
    config["dataset_fingerprint"] = fingerprint["hash"]
    config["dataset_fingerprint_parts"] = fingerprint["parts"]
    return config


def read_manual_config() -> dict:
    """Always return a config dict (defaults when the file is missing/invalid)."""
    if not Path(MANUAL_CONFIG_PATH).exists():
        return _default_manual_config()
    try:
        data = json.loads(Path(MANUAL_CONFIG_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_manual_config()
    # Backfill any missing top-level keys for template safety.
    base = _default_manual_config()
    base.update(data)
    base.setdefault("selected_models", {})
    base.setdefault("cv_settings", {"n_splits": CV_N_SPLITS, "gap": CV_GAP, "metric": "f1_up"})
    return base


def save_selected_model(
    model_key: str,
    run_id: str,
    params: dict,
    cv_f1_up_mean: float | None,
    fingerprint: dict,
) -> dict:
    """Persist a chosen config for one model into manual_config.json.

    If the current dataset fingerprint differs from the stored one, the
    selection is reset so all 3 models always share the same fingerprint.
    """
    ensure_experiments_dir()
    config = read_manual_config()
    if config.get("dataset_fingerprint") != fingerprint["hash"]:
        config = _empty_manual_config(fingerprint)

    config["selected_models"][model_key] = {
        "run_id": run_id,
        "params": params,
        "cv_f1_up_mean_when_selected": cv_f1_up_mean,
        "selected_at": _now_iso(),
    }
    Path(MANUAL_CONFIG_PATH).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def is_config_complete(config: dict | None, current_fingerprint_hash: str) -> bool:
    if not config:
        return False
    if config.get("dataset_fingerprint") != current_fingerprint_hash:
        return False
    selected = config.get("selected_models", {})
    return set(REQUIRED_MODEL_KEYS).issubset(set(selected.keys()))


# --------------------------------------------------------------------------- #
# Test evaluation lock
# --------------------------------------------------------------------------- #
def read_test_lock() -> dict | None:
    if not Path(TEST_EVAL_LOCK_PATH).exists():
        return None
    try:
        return json.loads(Path(TEST_EVAL_LOCK_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_test_lock(fingerprint_hash: str, final_model: str) -> None:
    ensure_experiments_dir()
    payload = {
        "dataset_fingerprint": fingerprint_hash,
        "evaluated_at": _now_iso(),
        "final_model": final_model,
        "status": "completed",
    }
    Path(TEST_EVAL_LOCK_PATH).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def is_test_locked(fingerprint_hash: str) -> bool:
    lock = read_test_lock()
    if not lock:
        return False
    return (
        lock.get("dataset_fingerprint") == fingerprint_hash
        and lock.get("status") == "completed"
    )


# --------------------------------------------------------------------------- #
# Pipeline lock (concurrency guard)
# --------------------------------------------------------------------------- #
def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_pipeline_lock() -> dict | None:
    if not Path(PIPELINE_LOCK_PATH).exists():
        return None
    try:
        return json.loads(Path(PIPELINE_LOCK_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _lock_is_stale(lock: dict) -> bool:
    pid = lock.get("pid")
    started_at = lock.get("started_at")
    if not _pid_alive(pid):
        return True
    if started_at:
        try:
            started = datetime.fromisoformat(started_at)
            now = datetime.now(tz=started.tzinfo) if started.tzinfo else datetime.now()
            if (now - started).total_seconds() > PIPELINE_LOCK_STALE_SECONDS:
                return True
        except ValueError:
            return True
    return False


def is_pipeline_running() -> bool:
    lock = _read_pipeline_lock()
    if not lock:
        return False
    if _lock_is_stale(lock):
        release_pipeline_lock()
        return False
    return True


def acquire_pipeline_lock() -> bool:
    """Create pipeline.lock; return False if one is already active."""
    ensure_experiments_dir()
    if is_pipeline_running():
        return False
    payload = {"pid": os.getpid(), "started_at": _now_iso()}
    Path(PIPELINE_LOCK_PATH).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def release_pipeline_lock() -> None:
    try:
        Path(PIPELINE_LOCK_PATH).unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Pipeline run log
# --------------------------------------------------------------------------- #
def write_pipeline_log(text: str) -> None:
    ensure_experiments_dir()
    Path(LAST_PIPELINE_RUN_LOG).write_text(text or "", encoding="utf-8")


def read_pipeline_log_tail(max_lines: int = 200) -> str:
    if not Path(LAST_PIPELINE_RUN_LOG).exists():
        return ""
    try:
        text = Path(LAST_PIPELINE_RUN_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
