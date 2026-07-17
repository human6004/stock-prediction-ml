import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.settings import (
    CLASSIFICATION_REPORT_PATH,
    COMPARISON_COLUMNS,
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    CV_GAP,
    CV_N_SPLITS,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    FINAL_MODEL_EVALUATION_PATH,
    FINAL_MODEL_PATH,
    HYPERPARAMETER_EXPLANATION_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DEFINITIONS,
    MODEL_METADATA_PATH,
    MODEL_PATHS,
    MODEL_SELECTION_REPORT_PATH,
    PIPELINE_SUMMARY_PATH,
    SIMPLICITY_RANK,
    SPLIT_DATE,
    UP_THRESHOLD,
)
from services.model_tuning import predict_with_threshold
from services.pipeline_utils import write_json


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1, 0],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_up": float(precision[0]),
        "recall_up": float(recall[0]),
        "f1_up": float(f1[0]),
        "precision_not_up": float(precision[1]),
        "recall_not_up": float(recall[1]),
        "f1_not_up": float(f1[1]),
    }


def evaluate_tuned_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fitted_artifacts: dict[int, dict] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    if fitted_artifacts is None:
        fitted_artifacts = {}
        for model_id, path in MODEL_PATHS.items():
            if path.exists():
                fitted_artifacts[model_id] = joblib.load(path)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    rows = []

    for model_id, artifact in sorted(fitted_artifacts.items()):
        model = artifact["model"]
        y_pred = predict_with_threshold(model, X_test, artifact["decision_threshold"])
        metrics = evaluate_predictions(y_test, y_pred)
        artifact["metrics"] = metrics
        artifact["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        joblib.dump(artifact, MODEL_PATHS[model_id])
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_DEFINITIONS[model_id],
                "cv_f1_up": artifact.get("cv_f1_up"),
                **metrics,
                "selected": False,
            }
        )

    return pd.DataFrame(rows), fitted_artifacts


def select_final_model(
    comparison: pd.DataFrame, fitted_artifacts: dict[int, dict]
) -> tuple[pd.DataFrame, dict, dict]:
    non_dummy = comparison[comparison["model_id"] != 1].copy()
    non_dummy["simplicity_rank"] = non_dummy["model_id"].map(SIMPLICITY_RANK)
    selected_id = int(
        non_dummy.sort_values(
            ["f1_up", "recall_up", "simplicity_rank"],
            ascending=[False, False, True],
        ).iloc[0]["model_id"]
    )

    comparison = comparison.copy()
    comparison["selected"] = comparison["model_id"] == selected_id
    selected_artifact = fitted_artifacts[selected_id]
    joblib.dump(selected_artifact, FINAL_MODEL_PATH)

    selection_report = {
        "selected_model_id": selected_id,
        "selected_model_name": MODEL_DEFINITIONS[selected_id],
        "selected_f1_up": float(
            comparison.loc[comparison["model_id"] == selected_id, "f1_up"].iloc[0]
        ),
        "selected_recall_up": float(
            comparison.loc[comparison["model_id"] == selected_id, "recall_up"].iloc[0]
        ),
        "selected_cv_f1_up": selected_artifact.get("cv_f1_up"),
    }
    return comparison, selected_artifact, selection_report


def write_model_selection_report(comparison: pd.DataFrame, selected_artifact: dict) -> None:
    ranked = (
        comparison[comparison["model_id"] != 1]
        .copy()
        .assign(simplicity_rank=lambda df: df["model_id"].map(SIMPLICITY_RANK))
        .sort_values(["f1_up", "recall_up", "simplicity_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    lines = [
        "MODEL SELECTION REPORT:",
        "",
        "1. Tieu chi chinh: F1_UP cao nhat tren tap test.",
        "2. Tieu chi phu: Recall_UP cao hon neu F1_UP gan bang nhau.",
        "3. Tieu chi don gian: Logistic Regression < Random Forest < Gradient Boosting.",
        "",
        "4. Xep hang theo F1_UP (giam dan), tie-break Recall_UP, roi do don gian:",
    ]
    for index, row in ranked.iterrows():
        cv_text = (
            f", CV_F1_UP: {row['cv_f1_up']:.6f}"
            if pd.notna(row.get("cv_f1_up"))
            else ""
        )
        lines.append(
            f"   {index + 1}. Model: {row['model_name']}, "
            f"F1_UP: {row['f1_up']:.6f}, "
            f"Recall_UP: {row['recall_up']:.6f}, "
            f"Accuracy: {row['accuracy']:.6f}{cv_text}"
        )
    lines.extend(
        [
            "",
            "6. Model duoc chon:",
            f"   Model: {selected_artifact['model_name']}",
            f"   F1_UP: {selected_artifact['metrics']['f1_up']:.6f}",
            f"   Recall_UP: {selected_artifact['metrics']['recall_up']:.6f}",
            "",
            "7. File model cuoi cung: models/final_model.pkl",
        ]
    )
    MODEL_SELECTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_SELECTION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_png(cm: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix - Final Model")
        ax.set_xticks([0, 1], labels=["Pred NOT_UP", "Pred UP"])
        ax.set_yticks([0, 1], labels=["Actual NOT_UP", "Actual UP"])
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=160)
        plt.close(fig)
    except Exception:
        pass


def write_feature_importance(selected_artifact: dict) -> bool:
    model = selected_artifact["model"]
    values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if inner is not None and hasattr(inner, "coef_"):
            values = np.abs(np.ravel(inner.coef_))

    if values is None or len(values) != len(FEATURE_COLUMNS):
        return False

    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": values}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return True


def write_classification_report(test: pd.DataFrame, selected_artifact: dict) -> None:
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    y_pred = predict_with_threshold(
        selected_artifact["model"], X_test, selected_artifact["decision_threshold"]
    )
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["NOT_UP", "UP"], output_dict=True
    )
    rows = []
    for label in ["NOT_UP", "UP"]:
        rows.append(
            {
                "class": label,
                "precision": report_dict[label]["precision"],
                "recall": report_dict[label]["recall"],
                "f1_score": report_dict[label]["f1-score"],
                "support": int(report_dict[label]["support"]),
            }
        )
    rows.append(
        {
            "class": "accuracy",
            "precision": report_dict["accuracy"],
            "recall": "",
            "f1_score": "",
            "support": int(report_dict["macro avg"]["support"]),
        }
    )
    pd.DataFrame(rows).to_csv(CLASSIFICATION_REPORT_PATH, index=False)


def write_hyperparameter_explanation() -> None:
    content = """# Giai thich sieu tham so

## TimeSeriesSplit
- `n_splits=5`: chia train thanh 5 fold theo thoi gian.
- `gap=5`: bo qua 5 mau giua train/validation de tranh leakage tu nhan 5 phien.

## Logistic Regression
- `C`: do manh regularization (C nho = regularization manh hon).
- `solver`: thuat toan toi uu (`lbfgs`, `liblinear`).

## Random Forest
- `n_estimators`: so cay trong rung.
- `max_depth`: do sau toi da moi cay.
- `min_samples_leaf`: so mau toi thieu o nut la.
- `max_features`: so dac trung xet moi lan split.

## Gradient Boosting
- `n_estimators`: so boosting stages.
- `learning_rate`: buoc hoc moi stage.
- `max_depth`: do sau cay co so.
- `subsample`: ty le mau dung moi stage.

## Tieu chi chon model
1. F1_UP tren test cao nhat.
2. Neu gan bang: Recall_UP cao hon.
3. Neu van gan bang: model don gian hon (LogReg > RF > GB).
"""
    HYPERPARAMETER_EXPLANATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMETER_EXPLANATION_PATH.write_text(content, encoding="utf-8")


def write_model_metadata(
    selected_artifact: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selection_report: dict,
) -> None:
    metadata = {
        "model_name": selected_artifact["model_name"],
        "model_id": selected_artifact["model_id"],
        "feature_order": FEATURE_COLUMNS,
        "prediction_horizon": selected_artifact["prediction_horizon"],
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": selected_artifact.get("decision_threshold", 0.5),
        "split_date": SPLIT_DATE,
        "cv_config": selected_artifact.get("cv_config", {"n_splits": CV_N_SPLITS, "gap": CV_GAP}),
        "best_params": selected_artifact.get("best_params", {}),
        "cv_f1_up": selected_artifact.get("cv_f1_up"),
        "test_metrics": selected_artifact.get("metrics", {}),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "trained_at": selected_artifact.get("trained_at"),
        "selection": selection_report,
    }
    write_json(MODEL_METADATA_PATH, metadata)


def write_reports(
    test: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_artifact: dict,
    summary: dict,
    train: pd.DataFrame | None = None,
    selection_report: dict | None = None,
) -> dict:
    cols = [c for c in COMPARISON_COLUMNS if c in comparison.columns]
    comparison_out = comparison[cols].copy()
    comparison_out.to_csv(MODEL_COMPARISON_PATH, index=False)
    comparison_out[comparison_out["selected"]].to_csv(FINAL_MODEL_EVALUATION_PATH, index=False)
    write_model_selection_report(comparison, selected_artifact)
    write_classification_report(test, selected_artifact)
    write_hyperparameter_explanation()

    if train is not None and selection_report is not None:
        write_model_metadata(selected_artifact, train, test, selection_report)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    y_pred = predict_with_threshold(
        selected_artifact["model"], X_test, selected_artifact["decision_threshold"]
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_NOT_UP", "actual_UP"],
        columns=["pred_NOT_UP", "pred_UP"],
    )
    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH)
    save_confusion_matrix_png(cm)

    feature_importance_written = write_feature_importance(selected_artifact)

    summary["confusion_matrix"] = {
        "pred_NOT_UP_actual_NOT_UP": int(cm[0, 0]),
        "pred_UP_actual_NOT_UP": int(cm[0, 1]),
        "pred_NOT_UP_actual_UP": int(cm[1, 0]),
        "pred_UP_actual_UP": int(cm[1, 1]),
        "total": int(cm.sum()),
        "test_rows": int(len(test)),
    }
    summary["feature_importance_written"] = feature_importance_written
    summary["report_files"] = [
        str(MODEL_COMPARISON_PATH.name),
        str(FINAL_MODEL_EVALUATION_PATH.name),
        str(CLASSIFICATION_REPORT_PATH.name),
        str(CONFUSION_MATRIX_CSV_PATH.name),
        str(CONFUSION_MATRIX_PNG_PATH.name),
        str(MODEL_SELECTION_REPORT_PATH.name),
        str(HYPERPARAMETER_EXPLANATION_PATH.name),
        str(MODEL_METADATA_PATH.name),
    ]
    if feature_importance_written:
        summary["report_files"].append(str(FEATURE_IMPORTANCE_PATH.name))

    PIPELINE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "confusion_matrix_sum": int(cm.sum()),
        "test_rows": int(len(test)),
        "feature_importance_written": feature_importance_written,
    }


def verify_model_selection(comparison: pd.DataFrame, selected_artifact: dict) -> None:
    selected_count = int(comparison["selected"].sum())
    if selected_count != 1:
        raise ValueError(f"Expected exactly 1 selected model, got {selected_count}")
    selected_row = comparison[comparison["selected"]].iloc[0]
    if selected_row["model_id"] == 1:
        raise ValueError("Dummy Classifier cannot be selected as final model.")
    if selected_artifact["model_name"] != selected_row["model_name"]:
        raise ValueError("Selected artifact model_name does not match comparison row.")
