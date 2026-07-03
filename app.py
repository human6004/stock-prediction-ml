import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    CONFUSION_MATRIX_PNG_PATH,
    FEATURE_DATA_PATH,
    LAST_PIPELINE_RUN_LOG,
    MODEL_COMPARISON_PATH,
    MODEL_KEY,
    MODEL_METADATA_PATH,
    PREDICTION_HORIZON,
    REPORTS_DIR,
    SPLIT_DATE,
    UP_THRESHOLD,
)
from services.database_service import log_prediction  # noqa: E402
from services.experiment_state import (  # noqa: E402
    compute_dataset_fingerprint,
    find_run,
    is_config_complete,
    is_pipeline_running,
    is_test_locked,
    mark_best,
    read_history,
    read_manual_config,
    read_pipeline_log_tail,
    save_selected_model,
    write_pipeline_log,
)
from services.prediction_service import load_metadata, predict_symbol  # noqa: E402
from services.tuning_lab import (  # noqa: E402
    evaluate_single_config,
    get_param_defaults,
    get_param_schema,
    load_train,
    validate_params,
)

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_dataset_meta() -> dict:
    if not FEATURE_DATA_PATH.exists():
        return {
            "dataset_max_date": None,
            "symbol_count": 0,
            "symbols": tuple(),
        }
    cols = pd.read_csv(FEATURE_DATA_PATH, usecols=["symbol", "trading_date"])
    return {
        "dataset_max_date": str(cols["trading_date"].max()),
        "symbol_count": int(cols["symbol"].nunique()),
        "symbols": tuple(sorted(cols["symbol"].astype(str).unique())),
    }


def load_selected_model_from_report() -> str:
    if MODEL_METADATA_PATH.exists():
        metadata = load_metadata()
        if metadata.get("model_name"):
            return str(metadata["model_name"])
    if not MODEL_COMPARISON_PATH.exists():
        return ""
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    selected = comparison[comparison["selected"].astype(str).str.lower() == "true"]
    if selected.empty:
        return ""
    return str(selected.iloc[0]["model_name"])


def load_evaluation_rows() -> list[dict]:
    if not MODEL_COMPARISON_PATH.exists():
        return []
    rows = pd.read_csv(MODEL_COMPARISON_PATH)
    if "cv_f1_up" in rows.columns:
        rows["cv_f1_up"] = rows["cv_f1_up"].where(rows["cv_f1_up"].notna(), None)
    rows["status"] = (
        rows["selected"]
        .astype(str)
        .str.lower()
        .map({"true": "Đang dùng"})
        .fillna("")
    )
    return rows.to_dict("records")


def _template_context(**extra):
    meta = get_dataset_meta()
    base = {
        "dataset_max_date": meta["dataset_max_date"],
        "symbol_count": meta["symbol_count"],
        "symbols": meta["symbols"],
        "horizon": PREDICTION_HORIZON,
        "threshold_percent": int(UP_THRESHOLD * 100),
        "split_date": SPLIT_DATE,
    }
    base.update(extra)
    return base


@app.route("/", methods=["GET"])
def index():
    selected_model = load_selected_model_from_report()
    return render_template(
        "index.html",
        **_template_context(selected_model=selected_model),
    )


@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.form.get("symbol", "")
    try:
        result = predict_symbol(symbol)
        try:
            log_prediction(result)
        except Exception:
            pass
        return render_template(
            "index.html",
            **_template_context(
                result=result,
                selected_model=result["model_name"],
            ),
        )
    except Exception as exc:
        return (
            render_template(
                "index.html",
                **_template_context(
                    error=str(exc),
                    selected_model=load_selected_model_from_report(),
                ),
            ),
            400,
        )


@app.route("/reports/confusion_matrix.png", methods=["GET"])
def confusion_matrix_image():
    if not CONFUSION_MATRIX_PNG_PATH.exists():
        return "Confusion matrix not found. Run pipeline first.", 404
    return send_from_directory(REPORTS_DIR, "confusion_matrix.png")


@app.route("/evaluation", methods=["GET"])
def evaluation():
    rows = load_evaluation_rows()
    selected_model = load_selected_model_from_report()
    return render_template(
        "evaluation.html",
        rows=rows,
        selected_model=selected_model,
        dataset_max_date=get_dataset_meta()["dataset_max_date"],
        split_date=SPLIT_DATE,
    )


def _tuning_context(**extra) -> dict:
    fingerprint = compute_dataset_fingerprint()
    cfg = read_manual_config()
    history = mark_best(read_history())
    history = sorted(history, key=lambda r: r.get("timestamp", ""), reverse=True)
    complete = is_config_complete(cfg, fingerprint["hash"])
    test_locked = is_test_locked(fingerprint["hash"])
    pipeline_running = is_pipeline_running()
    can_run = complete and not test_locked and not pipeline_running

    config_fingerprint = cfg.get("dataset_fingerprint")
    config_stale = bool(cfg.get("selected_models")) and config_fingerprint != fingerprint["hash"]

    base = {
        "param_schema": get_param_schema(),
        "param_defaults": get_param_defaults(),
        "model_key_map": MODEL_KEY,
        "manual_config": cfg,
        "current_fingerprint": fingerprint["hash"],
        "fingerprint_parts": fingerprint["parts"],
        "history": history,
        "config_complete": complete,
        "config_stale": config_stale,
        "test_locked": test_locked,
        "pipeline_running": pipeline_running,
        "can_run": can_run,
    }
    base.update(extra)
    return base


@app.route("/tuning", methods=["GET"])
def tuning():
    return render_template("tuning.html", **_tuning_context())


@app.route("/tuning/evaluate", methods=["POST"])
def tuning_evaluate():
    model_key = request.form.get("model_key", "")
    if model_key not in get_param_schema():
        return render_template(
            "tuning.html",
            **_tuning_context(error=f"Model không hợp lệ: {model_key}"),
        ), 400
    try:
        params = validate_params(model_key, request.form.to_dict())
        train = load_train()
        result = evaluate_single_config(model_key, params, train)
        return render_template(
            "tuning.html",
            **_tuning_context(eval_result=result, active_model=model_key),
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "tuning.html",
            **_tuning_context(error=str(exc), active_model=model_key),
        ), 400


@app.route("/tuning/use-config", methods=["POST"])
def tuning_use_config():
    run_id = request.form.get("run_id", "")
    run = find_run(run_id)
    if run is None:
        return render_template(
            "tuning.html",
            **_tuning_context(error=f"Không tìm thấy run_id: {run_id}"),
        ), 400

    fingerprint = compute_dataset_fingerprint()
    if run.get("dataset_fingerprint") != fingerprint["hash"]:
        return render_template(
            "tuning.html",
            **_tuning_context(
                error="Cấu hình này thuộc phiên bản dataset khác với hiện tại. "
                "Hãy train lại trên dataset hiện tại trước khi chốt."
            ),
        ), 400
    if run.get("status") != "ok":
        return render_template(
            "tuning.html",
            **_tuning_context(error="Run này bị lỗi, không thể chốt làm cấu hình."),
        ), 400

    import json

    params = json.loads(run.get("params_json") or "{}")
    cv_mean = run.get("cv_f1_up_mean")
    save_selected_model(
        run["model_key"], run_id, params, cv_mean, fingerprint
    )
    return redirect(url_for("tuning"))


@app.route("/tuning/run-pipeline", methods=["POST"])
def tuning_run_pipeline():
    fingerprint = compute_dataset_fingerprint()
    cfg = read_manual_config()

    if not is_config_complete(cfg, fingerprint["hash"]):
        return render_template(
            "tuning.html",
            **_tuning_context(
                error="Chưa đủ cấu hình cho cả 3 model trên dataset hiện tại, "
                "hoặc cấu hình thuộc dataset khác. Hãy chốt đủ 3 model trước."
            ),
        ), 400
    if is_test_locked(fingerprint["hash"]):
        return render_template(
            "tuning.html",
            **_tuning_context(
                error="TEST của dataset hiện tại đã được dùng. Không chạy lại để "
                "tránh leakage. Hãy cập nhật dữ liệu để có fingerprint mới."
            ),
        ), 400
    if is_pipeline_running():
        return render_template(
            "tuning.html",
            **_tuning_context(error="Pipeline đang chạy. Vui lòng đợi hoàn tất."),
        ), 409

    try:
        proc = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        write_pipeline_log((proc.stdout or "") + "\n" + (proc.stderr or ""))
        return redirect(url_for("evaluation"))
    except subprocess.CalledProcessError as exc:
        write_pipeline_log((exc.stdout or "") + "\n" + (exc.stderr or ""))
        return render_template(
            "tuning.html",
            **_tuning_context(
                error="Pipeline chạy thất bại. Xem log bên dưới.",
                pipeline_log=read_pipeline_log_tail(),
            ),
        ), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
