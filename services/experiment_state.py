"""State management for the Tuning Lab experiments folder.

Centralizes all IO under ``experiments/``:
- dataset fingerprint (data version identity)
- tuning history CSV (one row per manual training run)
- manual_config.json (the chosen config used by the official pipeline)
- evaluation_registry.json (rolling snapshots already evaluated)
- pipeline.lock (prevents concurrent pipeline runs)
- fetch.lock (prevents concurrent background data refresh)
- last_pipeline_run.log (stdout/stderr of the most recent pipeline run)
- last_fetch_run.log (stdout/stderr of the most recent fetch/refresh run)
"""

import csv
import hashlib
import json
import math
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    CV_GAP_SESSIONS,
    CV_N_SPLITS,
    CV_START_DATE,
    DECISION_THRESHOLD,
    DATA_PROTOCOL_ID,
    EVALUATION_REGISTRY_PATH,
    EXPERIMENTS_DIR,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FETCH_LOCK_PATH,
    LAST_FETCH_RUN_LOG,
    LAST_PIPELINE_RUN_LOG,
    MANUAL_CONFIG_PATH,
    MANUAL_CONFIG_SCHEMA_VERSION,
    ML_DATA_PATH,
    PIPELINE_LOCK_PATH,
    PREDICTION_HORIZON,
    TUNING_HISTORY_PATH,
    THRESHOLD_MAX_PREDICTED_UP_RATIO,
    UP_THRESHOLD,
)
from services.protocol_dates import resolve_protocol_dates

HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "model_key",
    "model_name",
    "params_json",
    "cv_f1_up_mean",
    "cv_f1_up_std",
    "cv_f1_up_folds_json",
    "cv_precision_up_mean",
    "cv_precision_up_std",
    "cv_recall_up_mean",
    "cv_recall_up_std",
    "cv_precision_up_folds_json",
    "cv_recall_up_folds_json",
    "cv_fold_ranges_json",
    "decision_threshold",
    "oof_f1_up",
    "oof_precision_up",
    "oof_recall_up",
    "oof_up_rate",
    "oof_predicted_up_ratio",
    "threshold_constraint_passed",
    "train_seconds",
    "dataset_fingerprint",
    "content_fingerprint",
    "policy_id",
    "status",
    "note",
]

REQUIRED_MODEL_KEYS = ("logistic_regression", "random_forest", "gradient_boosting")

_HISTORY_LOCK = threading.Lock()
_FINGERPRINT_LOCK = threading.Lock()
_FINGERPRINT_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}

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
def _fingerprint_scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in {"train", "experiment"}:
        raise ValueError(f"Unsupported fingerprint scope: {scope}")
    if not {"trading_date", "label_end_date"}.issubset(frame.columns):
        return frame

    trading_dates = pd.to_datetime(frame["trading_date"], errors="raise")
    label_end_dates = pd.to_datetime(frame["label_end_date"], errors="raise")
    # Mốc suy động từ chính frame (full dataset) qua cùng một nguồn với
    # protocol_time_split, nên fingerprint và split thật không lệch nhau.
    dates = resolve_protocol_dates(frame)
    train_end = pd.Timestamp(dates["train_end_date"])
    validation_end = pd.Timestamp(dates["validation_end_date"])
    test_end = pd.Timestamp(dates["test_end_date"])
    train_mask = label_end_dates <= train_end
    if scope == "train":
        return frame.loc[train_mask]

    validation_mask = (trading_dates > train_end) & (
        label_end_dates <= validation_end
    )
    test_mask = (trading_dates > validation_end) & (trading_dates <= test_end)
    return frame.loc[train_mask | validation_mask | test_mask]


def compute_dataset_fingerprint(
    ml_dataset_df: pd.DataFrame | None = None, *, scope: str = "train"
) -> dict:
    """Return an order-independent fingerprint for TRAIN or a rolling snapshot."""
    global _FINGERPRINT_CACHE
    if ml_dataset_df is None:
        path = Path(ML_DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(
                "Chua co data/processed/ml_dataset.csv. Hay chay buoc chuan bi du "
                "lieu (preprocess_data.py + build_features.py) hoac chay pipeline truoc."
            )
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        with _FINGERPRINT_LOCK:
            cached = _FINGERPRINT_CACHE.get(scope)
            if cached and cached[0] == signature:
                return cached[1]
        ml_dataset_df = pd.read_csv(path)

    source_rows = int(len(ml_dataset_df))
    scoped_df = _fingerprint_scope_frame(ml_dataset_df, scope)
    columns = sorted(str(column) for column in scoped_df.columns)
    normalized = scoped_df.loc[:, columns].copy()
    schema = json.dumps(
        [(column, str(normalized[column].dtype)) for column in columns],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    content_hasher = hashlib.sha256(schema.encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=True
    ).to_numpy(copy=True)
    row_hashes.sort()
    content_hasher.update(row_hashes.tobytes())
    content_hash = content_hasher.hexdigest()

    if "trading_date" in scoped_df.columns and len(scoped_df):
        max_trading_date = str(scoped_df["trading_date"].max())
    else:
        max_trading_date = None

    # Mốc rolling suy từ full dataset (cùng nguồn với split thật). Nhúng vào
    # parts để identity đổi theo mốc mỗi khi dataset có phiên mới.
    resolved_dates = resolve_protocol_dates(ml_dataset_df)

    parts = {
        "fingerprint_scope": scope,
        "data_protocol_id": DATA_PROTOCOL_ID,
        "dataset_rows": int(len(scoped_df)),
        "max_trading_date": max_trading_date,
        "split_date": resolved_dates["train_end_date"],
        "validation_end_date": resolved_dates["validation_end_date"],
        "test_end_date": resolved_dates["test_end_date"],
        "prediction_horizon": PREDICTION_HORIZON,
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": list(FEATURE_COLUMNS),
        "content_sha256": content_hash,
    }
    identity = hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    result = {
        "hash": identity[:12],
        "content_hash": content_hash,
        "parts": parts,
        "source_dataset_rows": source_rows,
    }
    if "signature" in locals():
        with _FINGERPRINT_LOCK:
            _FINGERPRINT_CACHE[scope] = (signature, result)
    return result


def compute_experiment_fingerprint(
    ml_dataset_df: pd.DataFrame | None = None,
) -> dict:
    """Fingerprint the resolved rolling TRAIN+VALIDATION+TEST snapshot."""
    return compute_dataset_fingerprint(ml_dataset_df, scope="experiment")


# --------------------------------------------------------------------------- #
# Tuning history
# --------------------------------------------------------------------------- #
def canonical_params_json(params: dict) -> str:
    return json.dumps(params or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _rewrite_history_with_current_schema() -> None:
    path = Path(TUNING_HISTORY_PATH)
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == HISTORY_COLUMNS:
            return
        prior_rows = list(reader)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for raw in prior_rows:
            raw["policy_id"] = raw.get("policy_id") or "legacy"
            writer.writerow({column: raw.get(column, "") for column in HISTORY_COLUMNS})
    temp_path.replace(path)


def append_history(record: dict) -> None:
    """Append one training run to the history CSV (creates header if needed)."""
    ensure_experiments_dir()
    with _HISTORY_LOCK:
        _rewrite_history_with_current_schema()
        file_exists = Path(TUNING_HISTORY_PATH).exists()
        enriched = {**record}
        enriched.setdefault("policy_id", EXPERIMENT_POLICY_ID)
        row = {col: enriched.get(col, "") for col in HISTORY_COLUMNS}
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
            row["cv_precision_up_mean"] = _to_float(raw.get("cv_precision_up_mean"))
            row["cv_precision_up_std"] = _to_float(raw.get("cv_precision_up_std"))
            row["cv_recall_up_mean"] = _to_float(raw.get("cv_recall_up_mean"))
            row["cv_recall_up_std"] = _to_float(raw.get("cv_recall_up_std"))
            row["decision_threshold"] = _to_float(raw.get("decision_threshold"))
            for field in (
                "oof_f1_up",
                "oof_precision_up",
                "oof_recall_up",
                "oof_up_rate",
                "oof_predicted_up_ratio",
            ):
                row[field] = _to_float(raw.get(field))
            row["threshold_constraint_passed"] = str(
                raw.get("threshold_constraint_passed") or ""
            ).lower() in {"1", "true", "yes"}
            row["train_seconds"] = _to_float(raw.get("train_seconds"))
            try:
                row["params"] = json.loads(raw.get("params_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["params"] = {}
            try:
                row["folds"] = json.loads(raw.get("cv_f1_up_folds_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["folds"] = []
            for source, destination in (
                ("cv_precision_up_folds_json", "precision_folds"),
                ("cv_recall_up_folds_json", "recall_folds"),
                ("cv_fold_ranges_json", "fold_ranges"),
            ):
                try:
                    row[destination] = json.loads(raw.get(source) or "[]")
                except (json.JSONDecodeError, TypeError):
                    row[destination] = []
            row["policy_id"] = raw.get("policy_id") or "legacy"
            row["is_best"] = False
            rows.append(row)
    return rows


def _ranking_key(row: dict) -> tuple:
    std = row.get("cv_f1_up_std")
    return (
        -float(row["cv_f1_up_mean"]),
        float("inf") if std is None else float(std),
        str(row.get("run_id") or ""),
    )


def _eligible_policy_rows(
    rows: list[dict], fingerprint_hash: str | None = None
) -> list[dict]:
    eligible = []
    for row in rows:
        mean = row.get("cv_f1_up_mean")
        std = row.get("cv_f1_up_std")
        if row.get("status") != "ok" or mean is None or std is None:
            continue
        if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or float(std) < 0:
            continue
        if row.get("policy_id") != EXPERIMENT_POLICY_ID:
            continue
        if fingerprint_hash and row.get("dataset_fingerprint") != fingerprint_hash:
            continue
        if not row.get("content_fingerprint"):
            continue
        threshold = _to_float(row.get("decision_threshold"))
        up_rate = _to_float(row.get("oof_up_rate"))
        precision = _to_float(row.get("oof_precision_up"))
        predicted_ratio = _to_float(row.get("oof_predicted_up_ratio"))
        if threshold is None or not 0 < threshold < 1:
            continue
        if not row.get("threshold_constraint_passed"):
            continue
        if None in (up_rate, precision, predicted_ratio):
            continue
        if precision + 1e-12 < up_rate:
            continue
        if predicted_ratio > THRESHOLD_MAX_PREDICTED_UP_RATIO + 1e-12:
            continue
        if any(
            len(row.get(field) or []) != CV_N_SPLITS
            for field in ("folds", "precision_folds", "recall_folds", "fold_ranges")
        ):
            continue
        eligible.append(row)
    return eligible


def mark_best(rows: list[dict]) -> list[dict]:
    """Flag current-policy best per model/content; legacy rows stay excluded."""
    for row in rows:
        row["is_best"] = False
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in _eligible_policy_rows(rows):
        key = (str(row.get("model_key")), str(row.get("dataset_fingerprint")))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        min(group, key=_ranking_key)["is_best"] = True
    return rows


def get_best_runs(rows: list[dict], fingerprint_hash: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in _eligible_policy_rows(rows, fingerprint_hash):
        grouped.setdefault(str(row.get("model_key")), []).append(row)
    return {model_key: min(group, key=_ranking_key) for model_key, group in grouped.items()}


def get_tuning_progress(rows: list[dict], fingerprint_hash: str) -> dict:
    eligible = _eligible_policy_rows(rows, fingerprint_hash)
    best = get_best_runs(eligible, fingerprint_hash)
    models = {}
    for model_key in REQUIRED_MODEL_KEYS:
        configs = {}
        for row in eligible:
            if row.get("model_key") == model_key:
                configs.setdefault(
                    canonical_params_json(row.get("params") or {}),
                    dict(row.get("params") or {}),
                )
        models[model_key] = {
            "valid_config_count": len(configs),
            "ready": bool(configs),
            "best_run": best.get(model_key),
        }
    return {
        "policy_id": EXPERIMENT_POLICY_ID,
        "models": models,
        "complete": all(item["ready"] for item in models.values()),
    }


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
        "policy_id": EXPERIMENT_POLICY_ID,
        "dataset_fingerprint": "",
        "dataset_fingerprint_parts": {},
        "cv_settings": {
            "n_splits": CV_N_SPLITS,
            "gap_sessions": CV_GAP_SESSIONS,
            "start_date": CV_START_DATE,
            "metric": "f1_up",
            "threshold_policy": {
                "max_predicted_up_ratio": THRESHOLD_MAX_PREDICTED_UP_RATIO,
                "precision_floor": "oof_up_rate",
            },
        },
        "selected_models": {},
    }


def _empty_manual_config(fingerprint: dict) -> dict:
    config = _default_manual_config()
    config["dataset_fingerprint"] = fingerprint["hash"]
    config["content_fingerprint"] = fingerprint.get("content_hash", fingerprint["hash"])
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
    base.setdefault(
        "cv_settings",
        {
            "n_splits": CV_N_SPLITS,
            "gap_sessions": CV_GAP_SESSIONS,
            "start_date": CV_START_DATE,
            "metric": "f1_up",
            "threshold_policy": {
                "max_predicted_up_ratio": THRESHOLD_MAX_PREDICTED_UP_RATIO,
                "precision_floor": "oof_up_rate",
            },
        },
    )
    return base


def _write_manual_config(config: dict) -> dict:
    ensure_experiments_dir()
    path = Path(MANUAL_CONFIG_PATH)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return config


def save_selected_model(
    model_key: str,
    run_id: str,
    params: dict,
    cv_f1_up_mean: float | None,
    fingerprint: dict,
) -> dict:
    """Persist the valid CV run explicitly chosen by the user."""
    run = next(
        (
            row
            for row in _eligible_policy_rows(read_history(), fingerprint["hash"])
            if row.get("model_key") == model_key and row.get("run_id") == run_id
        ),
        None,
    )
    if run is None:
        raise ValueError("Run không hợp lệ cho model và dataset hiện tại.")
    if canonical_params_json(run.get("params") or {}) != canonical_params_json(params):
        raise ValueError("Tham số không khớp với run đã chọn.")

    config = read_manual_config()
    if (
        config.get("schema_version") != MANUAL_CONFIG_SCHEMA_VERSION
        or config.get("policy_id") != EXPERIMENT_POLICY_ID
        or config.get("dataset_fingerprint") != fingerprint["hash"]
    ):
        config = _empty_manual_config(fingerprint)
    else:
        config = {**config, "selected_models": dict(config.get("selected_models") or {})}
    config["selected_models"][model_key] = {
        "run_id": run_id,
        "params": dict(run.get("params") or {}),
        "cv_f1_up_mean": run.get("cv_f1_up_mean", cv_f1_up_mean),
        "cv_f1_up_std": run.get("cv_f1_up_std"),
        "cv_precision_up_mean": run.get("cv_precision_up_mean"),
        "cv_recall_up_mean": run.get("cv_recall_up_mean"),
        "f1_up_folds": list(run.get("folds") or []),
        "precision_up_folds": list(run.get("precision_folds") or []),
        "recall_up_folds": list(run.get("recall_folds") or []),
        "fold_date_ranges": list(run.get("fold_ranges") or []),
        "decision_threshold": run.get("decision_threshold"),
        "oof_f1_up": run.get("oof_f1_up"),
        "oof_precision_up": run.get("oof_precision_up"),
        "oof_recall_up": run.get("oof_recall_up"),
        "oof_up_rate": run.get("oof_up_rate"),
        "oof_predicted_up_ratio": run.get("oof_predicted_up_ratio"),
        "threshold_constraint_passed": run.get("threshold_constraint_passed"),
        "selected_at": _now_iso(),
        "selection_method": "manual",
    }
    return _write_manual_config(config)


def is_config_complete(
    config: dict | None,
    current_fingerprint_hash: str,
    rows: list[dict] | None = None,
) -> bool:
    if not config:
        return False
    if config.get("policy_id") != EXPERIMENT_POLICY_ID:
        return False
    if config.get("schema_version") != MANUAL_CONFIG_SCHEMA_VERSION:
        return False
    cv_settings = config.get("cv_settings") or {}
    if (
        cv_settings.get("n_splits") != CV_N_SPLITS
        or cv_settings.get("gap_sessions") != CV_GAP_SESSIONS
        or cv_settings.get("start_date") != CV_START_DATE
    ):
        return False
    if config.get("dataset_fingerprint") != current_fingerprint_hash:
        return False
    eligible = _eligible_policy_rows(
        read_history() if rows is None else rows, current_fingerprint_hash
    )
    selected = config.get("selected_models", {})
    for model_key in REQUIRED_MODEL_KEYS:
        choice = selected.get(model_key) or {}
        if choice.get("selection_method") != "manual":
            return False
        run = next(
            (
                row
                for row in eligible
                if row.get("model_key") == model_key
                and row.get("run_id") == choice.get("run_id")
            ),
            None,
        )
        if run is None or canonical_params_json(run.get("params") or {}) != canonical_params_json(
            choice.get("params") or {}
        ):
            return False
        if _to_float(choice.get("decision_threshold")) != _to_float(
            run.get("decision_threshold")
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Rolling evaluation registry
# --------------------------------------------------------------------------- #
def read_evaluation_registry(*, path: Path = EVALUATION_REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_evaluation_entry(
    entry: dict, *, path: Path = EVALUATION_REGISTRY_PATH
) -> dict:
    fingerprint = str(entry.get("experiment_fingerprint") or "").strip()
    if not fingerprint:
        raise ValueError("Evaluation entry requires experiment_fingerprint.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = read_evaluation_registry(path=path)
    registry[fingerprint] = {
        **registry.get(fingerprint, {}),
        **entry,
        "policy_id": entry.get("policy_id", EXPERIMENT_POLICY_ID),
        "updated_at": _now_iso(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return registry[fingerprint]


def has_evaluated_snapshot(
    fingerprint_hash: str, *, path: Path = EVALUATION_REGISTRY_PATH
) -> bool:
    entry = read_evaluation_registry(path=path).get(fingerprint_hash) or {}
    return entry.get("status") in {
        "started",
        "evaluated",
        "published",
        "release_failed",
    }


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
    return not _pid_alive(lock.get("pid"))


def _remove_pipeline_lock_if_matches(expected: dict) -> bool:
    path = Path(PIPELINE_LOCK_PATH)
    current = _read_pipeline_lock()
    if current != expected:
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def is_pipeline_running() -> bool:
    lock = _read_pipeline_lock()
    if not lock:
        return False
    if _lock_is_stale(lock):
        _remove_pipeline_lock_if_matches(lock)
        return False
    return True


def acquire_pipeline_lock() -> str | None:
    """Atomically create pipeline.lock and return its owner token."""
    ensure_experiments_dir()
    path = Path(PIPELINE_LOCK_PATH)
    owner_token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "owner_token": owner_token,
        "started_at": _now_iso(),
    }
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return owner_token
        except FileExistsError:
            current = _read_pipeline_lock()
            if current is None or not _lock_is_stale(current):
                return None
            if not _remove_pipeline_lock_if_matches(current):
                return None
    return None


def release_pipeline_lock(owner_token: str) -> bool:
    """Release only the lock owned by the caller."""
    lock = _read_pipeline_lock()
    if not lock or lock.get("owner_token") != owner_token:
        return False
    return _remove_pipeline_lock_if_matches(lock)


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


# --------------------------------------------------------------------------- #
# Fetch lock (background data refresh)
# --------------------------------------------------------------------------- #
def _read_fetch_lock() -> dict | None:
    if not Path(FETCH_LOCK_PATH).exists():
        return None
    try:
        return json.loads(Path(FETCH_LOCK_PATH).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_fetch_running() -> bool:
    lock = _read_fetch_lock()
    if not lock:
        return False
    if _lock_is_stale(lock):
        release_fetch_lock()
        return False
    return True


def write_fetch_lock(pid: int) -> None:
    ensure_experiments_dir()
    payload = {"pid": int(pid), "started_at": _now_iso()}
    Path(FETCH_LOCK_PATH).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def release_fetch_lock() -> None:
    try:
        Path(FETCH_LOCK_PATH).unlink()
    except FileNotFoundError:
        pass


def read_fetch_log_tail(max_lines: int = 200) -> str:
    if not Path(LAST_FETCH_RUN_LOG).exists():
        return ""
    try:
        text = Path(LAST_FETCH_RUN_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
