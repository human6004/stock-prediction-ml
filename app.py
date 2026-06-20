import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    CONFUSION_MATRIX_PNG_PATH,
    FEATURE_DATA_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    PREDICTION_HORIZON,
    REPORTS_DIR,
    SPLIT_DATE,
    UP_THRESHOLD,
)
from services.database_service import log_prediction  # noqa: E402
from services.prediction_service import load_metadata, predict_symbol  # noqa: E402

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


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
