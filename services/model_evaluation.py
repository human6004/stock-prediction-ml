"""Chọn model trên VALIDATION, refit winner và đánh giá TEST đúng một lần."""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import (
    CLASSIFICATION_REPORT_PATH,
    COMPARISON_COLUMNS,
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    CV_GAP_SESSIONS,
    CV_N_SPLITS,
    DECISION_THRESHOLD,
    EXPERIMENT_POLICY_ID,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    FINAL_MODEL_EVALUATION_PATH,
    HYPERPARAMETER_EXPLANATION_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DEFINITIONS,
    MODEL_METADATA_PATH,
    MODEL_SELECTION_REPORT_PATH,
    SIMPLICITY_RANK,
    UP_THRESHOLD,
)
from services.protocol_dates import resolve_protocol_dates
from services.model_tuning import predict_with_threshold
from services.pipeline_utils import (
    atomic_dataframe_to_csv,
    atomic_output_path,
    atomic_write_text,
    write_json,
)
from services.time_splitting import sort_panel_frame

CANDIDATE_MODEL_IDS = (2, 3, 4)
METRIC_COLUMNS = [
    "accuracy",
    "precision_up",
    "recall_up",
    "f1_up",
    "precision_not_up",
    "recall_not_up",
    "f1_not_up",
]


def _artifact_threshold(artifact: dict) -> float:
    return float(artifact.get("decision_threshold", DECISION_THRESHOLD))


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Tính metric hai lớp, đặt lớp UP=1 ở vị trí đầu để lấy đúng F1_UP."""
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


def _comparison_row(
    model_id: int,
    model_name: str,
    metrics: dict,
    *,
    split: str,
    row_type: str,
    cv_f1_up=None,
    decision_threshold=None,
) -> dict:
    return {
        "model_id": model_id,
        "model_name": model_name,
        "split": split,
        "row_type": row_type,
        "cv_f1_up": cv_f1_up,
        "decision_threshold": decision_threshold,
        **metrics,
        "selected": False,
    }


def evaluate_validation_candidates(
    validation: pd.DataFrame, fitted_artifacts: dict[int, dict]
) -> pd.DataFrame:
    """Score only LR/RF/GB candidates on VALIDATION."""
    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["target"]
    rows = []
    for model_id in CANDIDATE_MODEL_IDS:
        if model_id not in fitted_artifacts:
            raise ValueError(f"Missing fitted candidate model_id={model_id}")
        artifact = fitted_artifacts[model_id]
        threshold = _artifact_threshold(artifact)
        y_pred = predict_with_threshold(artifact["model"], X_validation, threshold)
        metrics = evaluate_predictions(y_validation, y_pred)
        artifact["validation_selection_metrics"] = metrics
        rows.append(
            _comparison_row(
                model_id,
                MODEL_DEFINITIONS[model_id],
                metrics,
                split="validation",
                row_type="candidate",
                cv_f1_up=artifact.get("cv_f1_up"),
                decision_threshold=threshold,
            )
        )
    return pd.DataFrame(rows)


def evaluate_baselines(y_true: pd.Series, *, split: str) -> pd.DataFrame:
    """Return explicit constant baselines for the requested split."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    rows = []
    for model_id, name, prediction in (
        (-1, "Always UP", 1),
        (-2, "Always NOT_UP", 0),
    ):
        metrics = evaluate_predictions(y_true, np.full(len(y_true), prediction))
        rows.append(
            _comparison_row(
                model_id,
                name,
                metrics,
                split=split,
                row_type="baseline",
            )
        )
    return pd.DataFrame(rows)


def select_final_model(
    comparison: pd.DataFrame, fitted_artifacts: dict[int, dict]
) -> tuple[pd.DataFrame, dict, dict]:
    """Select one candidate on VALIDATION; baselines are comparison-only."""
    candidates = comparison[comparison["model_id"].isin(CANDIDATE_MODEL_IDS)].copy()
    if set(candidates["model_id"]) != set(CANDIDATE_MODEL_IDS):
        raise ValueError("VALIDATION comparison must contain LR, RF and GB")
    candidates["simplicity_rank"] = candidates["model_id"].map(SIMPLICITY_RANK)
    selected_id = int(
        candidates.sort_values(
            ["f1_up", "recall_up", "simplicity_rank"],
            ascending=[False, False, True],
        ).iloc[0]["model_id"]
    )

    comparison = comparison.copy()
    comparison["selected"] = comparison["model_id"] == selected_id
    selected_artifact = fitted_artifacts[selected_id]
    selected_row = comparison.loc[comparison["model_id"] == selected_id].iloc[0]
    selected_f1_up = float(selected_row["f1_up"])

    baselines = (
        comparison[comparison["row_type"] == "baseline"]
        if "row_type" in comparison.columns
        else comparison.iloc[0:0]
    )
    best_baseline = None
    if not baselines.empty:
        baseline_row = baselines.sort_values("f1_up", ascending=False).iloc[0]
        best_baseline = {
            "model_name": baseline_row["model_name"],
            "f1_up": float(baseline_row["f1_up"]),
            "recall_up": float(baseline_row["recall_up"]),
        }
    validation_baseline_passed = (
        best_baseline is None
        or selected_f1_up > best_baseline["f1_up"]
    )
    validation_baseline_warning = None
    if not validation_baseline_passed:
        validation_baseline_warning = (
            f"Selected candidate VALIDATION F1_UP={selected_f1_up:.6f} "
            f"is below baseline {best_baseline['model_name']}="
            f"{best_baseline['f1_up']:.6f}."
        )
        raise RuntimeError(validation_baseline_warning)
    validation_metrics = {key: float(selected_row[key]) for key in METRIC_COLUMNS}
    selected_artifact["validation_selection_metrics"] = validation_metrics
    selected_artifact["validation_baseline_passed"] = validation_baseline_passed
    selected_artifact["validation_baseline_warning"] = validation_baseline_warning

    selection_report = {
        "policy_id": EXPERIMENT_POLICY_ID,
        "selection_split": "validation",
        "selected_model_id": selected_id,
        "selected_model_name": MODEL_DEFINITIONS[selected_id],
        "selected_f1_up": validation_metrics["f1_up"],
        "selected_recall_up": validation_metrics["recall_up"],
        "selected_cv_f1_up": selected_artifact.get("cv_f1_up"),
        "validation_selection_metrics": validation_metrics,
        "best_validation_baseline": best_baseline,
        "validation_baseline_passed": validation_baseline_passed,
        "validation_baseline_warning": validation_baseline_warning,
    }
    return comparison, selected_artifact, selection_report


def refit_and_evaluate_final_model(
    selected_artifact: dict,
    train_validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Fresh-fit the winner on TRAIN+VALIDATION and evaluate TEST once."""
    fitted_data = sort_panel_frame(train_validation)
    X_fit = fitted_data[FEATURE_COLUMNS]
    y_fit = fitted_data["target"]
    model = clone(selected_artifact["model"])
    if selected_artifact["model_id"] == 4:
        model.fit(X_fit, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
    else:
        model.fit(X_fit, y_fit)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    threshold = _artifact_threshold(selected_artifact)
    y_pred = predict_with_threshold(model, X_test, threshold)
    final_test_metrics = evaluate_predictions(y_test, y_pred)
    final_test_row = _comparison_row(
        selected_artifact["model_id"],
        selected_artifact["model_name"],
        final_test_metrics,
        split="test",
        row_type="final",
        cv_f1_up=selected_artifact.get("cv_f1_up"),
        decision_threshold=threshold,
    )
    final_test_row["selected"] = True
    baseline_rows = evaluate_baselines(y_test, split="test")
    test_evaluation = pd.DataFrame(
        [final_test_row, *baseline_rows.to_dict(orient="records")]
    )
    always_up = test_evaluation.loc[
        test_evaluation["model_name"] == "Always UP"
    ].iloc[0]
    baseline_passed = final_test_metrics["f1_up"] > float(always_up["f1_up"])
    baseline_warning = None
    if not baseline_passed:
        baseline_warning = (
            "Final Model chưa vượt baseline always-UP trên TEST "
            f"(F1_UP {final_test_metrics['f1_up']:.6f} <= {float(always_up['f1_up']):.6f})."
        )
    final_artifact = {
        **selected_artifact,
        "model": model,
        "policy_id": EXPERIMENT_POLICY_ID,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": final_test_metrics,
        "final_test_metrics": final_test_metrics,
        "baseline_passed": baseline_passed,
        "baseline_warning": baseline_warning,
        "final_test_baselines": baseline_rows.to_dict(orient="records"),
        "train_through_date": str(fitted_data["label_end_date"].max()),
    }
    return final_artifact, test_evaluation


def write_model_selection_report(comparison: pd.DataFrame, selected_artifact: dict) -> None:
    final_metrics = selected_artifact.get(
        "final_test_metrics", selected_artifact.get("metrics", {})
    )
    ranked = (
        comparison[comparison["model_id"].isin(CANDIDATE_MODEL_IDS)]
        .copy()
        .assign(simplicity_rank=lambda df: df["model_id"].map(SIMPLICITY_RANK))
        .sort_values(["f1_up", "recall_up", "simplicity_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    lines = [
        "MODEL SELECTION REPORT:",
        "",
        "1. Chon model chi tren VALIDATION; TEST khong tham gia xep hang.",
        "2. Tieu chi phu: Recall_UP cao hon neu F1_UP bang nhau.",
        "3. Tieu chi don gian: Logistic Regression < Random Forest < Gradient Boosting.",
        "",
        "4. Xep hang VALIDATION theo F1_UP, Recall_UP, roi do don gian:",
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
    baselines = (
        comparison[comparison["row_type"] == "baseline"]
        if "row_type" in comparison.columns
        else comparison.iloc[0:0]
    )
    if not baselines.empty:
        lines.append("")
        lines.append("   Baselines tren VALIDATION:")
        for _, row in baselines.iterrows():
            lines.append(
                f"   - {row['model_name']}: F1_UP={row['f1_up']:.6f}, "
                f"Recall_UP={row['recall_up']:.6f}"
            )
    lines.extend(
        [
            "",
            "5. Model duoc chon tren VALIDATION:",
            f"   Model: {selected_artifact['model_name']}",
            f"   F1_UP: {selected_artifact['validation_selection_metrics']['f1_up']:.6f}",
            f"   Recall_UP: {selected_artifact['validation_selection_metrics']['recall_up']:.6f}",
            f"   Validation baseline passed: {selected_artifact.get('validation_baseline_passed')}",
            f"   Validation baseline warning: {selected_artifact.get('validation_baseline_warning') or ''}",
            "",
            "6. Danh gia FINAL TEST mot lan:",
            f"   F1_UP: {final_metrics.get('f1_up', float('nan')):.6f}",
            f"   Recall_UP: {final_metrics.get('recall_up', float('nan')):.6f}",
            f"   TEST always-UP passed: {selected_artifact.get('baseline_passed')}",
            f"   TEST baseline warning: {selected_artifact.get('baseline_warning') or ''}",
            "",
            "7. File model cuoi cung: models/final_model.pkl",
        ]
    )
    MODEL_SELECTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(MODEL_SELECTION_REPORT_PATH, "\n".join(lines))


def save_confusion_matrix_png(cm: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix - Final Model")
        ax.set_xticks([0, 1], labels=["Pred NOT_UP", "Pred UP"])
        ax.set_yticks([0, 1], labels=["Actual NOT_UP", "Actual UP"])
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        with atomic_output_path(CONFUSION_MATRIX_PNG_PATH) as temp_path:
            fig.savefig(temp_path, dpi=160)
    finally:
        plt.close(fig)


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
    atomic_dataframe_to_csv(importance_df, FEATURE_IMPORTANCE_PATH, index=False)
    return True


def write_classification_report(test: pd.DataFrame, selected_artifact: dict) -> None:
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    y_pred = predict_with_threshold(
        selected_artifact["model"], X_test, _artifact_threshold(selected_artifact)
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
    atomic_dataframe_to_csv(
        pd.DataFrame(rows), CLASSIFICATION_REPORT_PATH, index=False
    )


def write_hyperparameter_explanation() -> None:
    content = """# Giai thich sieu tham so

## TimeSeriesSplit
- `n_splits=4`: chia train thanh 4 fold theo thoi gian tu 2021.
- `gap=5`: bo dung 5 ngay giao dich chung truoc validation; khong dem row.
- Purge: loai train row co `label_end_date >= validation_start`.
- Threshold: chon rieng tung model tu OOF TRAIN, gioi han ty le du bao UP va precision.

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
1. F1_UP tren VALIDATION cao nhat.
2. Neu bang nhau: Recall_UP cao hon.
3. Neu van bang nhau: model don gian hon (LogReg > RF > GB).
4. Refit winner tren TRAIN+VALIDATION, sau do danh gia FINAL TEST mot lan.
"""
    HYPERPARAMETER_EXPLANATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(HYPERPARAMETER_EXPLANATION_PATH, content)


def build_model_metadata(
    selected_artifact: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selection_report: dict,
    *,
    content_fingerprint: str,
) -> dict:
    train_validation = pd.concat([train, validation], ignore_index=True)
    train_through_date = str(train_validation["label_end_date"].max())
    training_symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in train_validation["symbol"].dropna()
            if str(symbol).strip()
        }
    )
    # Mốc thực tế đã dùng để cắt split, suy từ chính dữ liệu (rolling).
    resolved_dates = resolve_protocol_dates(
        pd.concat([train, validation, test], ignore_index=True)
    )
    metadata = {
        "policy_id": EXPERIMENT_POLICY_ID,
        "content_fingerprint": content_fingerprint,
        "training_content_fingerprint": selected_artifact.get(
            "training_content_fingerprint"
        ),
        "tuning_fingerprint": selected_artifact.get("tuning_fingerprint"),
        "experiment_fingerprint": selected_artifact.get(
            "experiment_fingerprint"
        ),
        "model_name": selected_artifact["model_name"],
        "model_id": selected_artifact["model_id"],
        "feature_order": FEATURE_COLUMNS,
        "prediction_horizon": selected_artifact["prediction_horizon"],
        "up_threshold": UP_THRESHOLD,
        "decision_threshold": _artifact_threshold(selected_artifact),
        "split_date": resolved_dates["train_end_date"],
        "train_end_date": resolved_dates["train_end_date"],
        "validation_end_date": resolved_dates["validation_end_date"],
        "test_end_date": resolved_dates["test_end_date"],
        "cv_config": selected_artifact.get(
            "cv_config", {"n_splits": CV_N_SPLITS, "gap_sessions": CV_GAP_SESSIONS}
        ),
        "best_params": selected_artifact.get("best_params", {}),
        "cv_f1_up": selected_artifact.get("cv_f1_up"),
        "baseline_passed": bool(selected_artifact.get("baseline_passed")),
        "baseline_warning": selected_artifact.get("baseline_warning"),
        "train_through_date": train_through_date,
        "training_symbols": training_symbols,
        "training_symbol_count": len(training_symbols),
        "validation_selection_metrics": selection_report.get(
            "validation_selection_metrics", {}
        ),
        "final_test_metrics": selected_artifact.get("final_test_metrics", {}),
        "final_test_baselines": selected_artifact.get("final_test_baselines", []),
        # Kept for serving/report compatibility; authoritative key is final_test_metrics.
        "test_metrics": selected_artifact.get("final_test_metrics", {}),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_validation_rows": int(len(train_validation)),
        "test_rows": int(len(test)),
        "trained_at": selected_artifact.get("trained_at"),
        "selection": selection_report,
    }
    return metadata


def write_model_metadata(
    selected_artifact: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selection_report: dict,
    *,
    content_fingerprint: str,
) -> None:
    metadata = build_model_metadata(
        selected_artifact,
        train,
        validation,
        test,
        selection_report,
        content_fingerprint=content_fingerprint,
    )
    write_json(MODEL_METADATA_PATH, metadata)


def write_reports(
    test: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_artifact: dict,
    summary: dict,
    selection_report: dict | None = None,
    final_test_evaluation: pd.DataFrame | None = None,
    content_fingerprint: str | None = None,
) -> dict:
    cols = [
        c
        for c in ["model_id", "model_name", "split", "row_type", *COMPARISON_COLUMNS]
        if c in comparison.columns
    ]
    cols = list(dict.fromkeys(cols))
    comparison_out = comparison[cols].copy()
    atomic_dataframe_to_csv(comparison_out, MODEL_COMPARISON_PATH, index=False)
    if final_test_evaluation is None:
        final_test_row = _comparison_row(
            selected_artifact["model_id"],
            selected_artifact["model_name"],
            selected_artifact.get("final_test_metrics", selected_artifact.get("metrics", {})),
            split="test",
            row_type="final",
            cv_f1_up=selected_artifact.get("cv_f1_up"),
            decision_threshold=_artifact_threshold(selected_artifact),
        )
        final_test_row["selected"] = True
        final_test_evaluation = pd.DataFrame(
            [
                final_test_row,
                *evaluate_baselines(test["target"], split="test").to_dict(
                    orient="records"
                ),
            ]
        )
    atomic_dataframe_to_csv(
        final_test_evaluation, FINAL_MODEL_EVALUATION_PATH, index=False
    )
    write_model_selection_report(comparison, selected_artifact)
    write_classification_report(test, selected_artifact)
    write_hyperparameter_explanation()

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    y_pred = predict_with_threshold(
        selected_artifact["model"], X_test, _artifact_threshold(selected_artifact)
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_NOT_UP", "actual_UP"],
        columns=["pred_NOT_UP", "pred_UP"],
    )
    atomic_dataframe_to_csv(cm_df, CONFUSION_MATRIX_CSV_PATH)
    save_confusion_matrix_png(cm)

    feature_importance_written = write_feature_importance(selected_artifact)

    summary["policy_id"] = EXPERIMENT_POLICY_ID
    summary["content_fingerprint"] = content_fingerprint
    summary["validation_selection"] = {
        "rows": comparison_out.to_dict(orient="records"),
        "baseline_passed": selection_report.get("validation_baseline_passed")
        if selection_report
        else None,
        "baseline_warning": selection_report.get("validation_baseline_warning")
        if selection_report
        else None,
    }
    summary["final_test_evaluation"] = final_test_evaluation.to_dict(orient="records")
    summary["baseline_passed"] = selected_artifact.get("baseline_passed")
    summary["baseline_warning"] = selected_artifact.get("baseline_warning")
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
    if selected_row["model_id"] not in CANDIDATE_MODEL_IDS:
        raise ValueError("Only LR, RF or GB can be selected as final model.")
    if selected_artifact["model_name"] != selected_row["model_name"]:
        raise ValueError("Selected artifact model_name does not match comparison row.")
