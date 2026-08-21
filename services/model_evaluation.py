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
    """Dựng 1 dòng cho bảng so sánh model — mọi dòng phải cùng bộ cột.

    Hàm này tồn tại để `model_comparison.csv` không bị lệch schema: dòng candidate,
    dòng baseline và dòng final TEST đều đi qua đây nên luôn có đủ
    `model_id/model_name/split/row_type/cv_f1_up/decision_threshold/selected`.
    Nếu mỗi chỗ tự dựng dict riêng thì `pd.DataFrame([...])` sẽ sinh cột NaN rải rác.

    Hai cột phân loại dòng, đọc report phải dựa vào chúng:
    - `split`: dữ liệu nào đã tính metric ("validation" hay "test").
    - `row_type`: vai trò của dòng ("candidate" = model dự tuyển, "baseline" =
      hằng số để so, "final" = model đã chọn, refit và đánh giá TEST).

    `**metrics` bung sau các khóa cố định nên nếu `metrics` chứa cùng tên khóa thì
    nó sẽ GHI ĐÈ — chủ ý, vì `metrics` là số vừa đo, đáng tin hơn giá trị mặc định.
    `selected=False` luôn là mặc định; `select_final_model` mới bật True cho đúng
    một dòng, và `verify_model_selection` hậu kiểm rằng tổng `selected` bằng 1.
    """
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
    """Select one candidate on VALIDATION; baselines are comparison-only.

    Vì sao chọn trên VALIDATION mà không phải TEST: TEST chỉ được "mở" đúng một
    lần sau khi đã chốt model. Nếu chọn model theo điểm TEST thì TEST trở thành
    tập tuning và metric báo cáo sẽ lạc quan giả.

    Thứ tự sort là tie-break 3 tầng, có chủ ý:
    1. ``f1_up`` giảm — chỉ số chính (cân bằng precision/recall cho lớp UP).
    2. ``recall_up`` giảm — bằng F1 thì ưu tiên model bắt được nhiều phiên UP hơn.
    3. ``simplicity_rank`` tăng — vẫn bằng nhau thì lấy model ĐƠN GIẢN hơn
       (LR < RF < GB theo ``SIMPLICITY_RANK``). Tránh chọn model phức tạp chỉ vì
       trùng điểm do nhiễu.

    ``baselines`` (Always UP / Always NOT_UP) chỉ để SO SÁNH, không được vào danh
    sách ứng viên. Nếu ứng viên tốt nhất không hơn baseline tốt nhất thì vẫn chọn
    ứng viên đó, nhưng ghi ``validation_baseline_passed=False`` và cảnh báo. Điều
    kiện là ``>`` (nghiêm ngặt): bằng baseline cũng bị coi là không đạt.
    """
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
    """Fresh-fit the winner on TRAIN+VALIDATION and evaluate TEST once.

    Ba chi tiết dễ bỏ sót:
    - ``clone(selected_artifact["model"])``: tạo estimator MỚI cùng hyperparameter
      nhưng CHƯA fit. Nếu fit lại trực tiếp object cũ (đã học trên TRAIN) thì tùy
      model có thể mang theo trạng thái cũ; clone đảm bảo fit sạch trên
      TRAIN+VALIDATION — nhiều dữ liệu hơn, sát thời điểm serving hơn.
    - ``model_id == 4`` (Gradient Boosting) được truyền
      ``compute_sample_weight("balanced")`` vì GB của sklearn không có
      ``class_weight``; LR/RF đã khai báo class_weight lúc dựng nên không cần.
      Thiếu dòng này GB sẽ lệch hẳn về lớp đa số NOT_UP.
    - ``predict_with_threshold`` dùng ĐÚNG threshold đã chọn từ OOF lúc tuning
      (``_artifact_threshold``), không phải 0.5. Dùng 0.5 ở đây sẽ làm metric TEST
      không khớp với threshold mà model thật sự serve.

    ``baseline_passed``: so F1_UP của final model với baseline "Always UP" trên
    TEST. Không vượt thì vẫn publish kèm ``baseline_warning`` để UI và chatbot
    cảnh báo, vì tới bước này TEST đã
    được mở, chạy lại pipeline để "tìm số đẹp hơn" chính là data leakage.
    """
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
    """Xuất báo cáo text giải trình VÌ SAO model này được chọn.

    Mục đích là để người đọc (giảng viên/phản biện) kiểm tra lại quy trình mà
    không cần chạy code:
    - `ranked` được sắp lại đúng bằng bộ khoá của `select_final_model`
      (f1_up desc, recall_up desc, simplicity_rank asc) và chỉ lấy các row
      thuộc `CANDIDATE_MODEL_IDS`, nên thứ tự in ra chính là thứ tự cạnh tranh
      thật; hạng 1 phải trùng model đã chọn.
    - Baseline (always-UP / always-NOT_UP) in riêng ở khối dưới vì chúng là mốc
      so sánh, không phải ứng viên; lọc bằng `row_type == "baseline"` và có
      nhánh phòng khi `comparison` cũ không có cột `row_type`.
    - Mục 5 lấy số từ `validation_selection_metrics` (điểm dùng để CHỌN), mục 6
      lấy từ `final_test_metrics` (điểm chỉ để BÁO CÁO). Tách hai nguồn số này
      để không ai đọc nhầm rằng TEST đã tham gia xếp hạng.
    - `cv_f1_up` chỉ in khi không NaN vì baseline và một số model không tune
      thì không có điểm CV.
    Ghi bằng `atomic_write_text` để báo cáo không bao giờ ở trạng thái nửa vời.
    """
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
    """Vẽ confusion matrix ra PNG.

    Ba chi tiết cố ý:
    - import matplotlib NGAY TRONG hàm (không ở đầu file) để pipeline không phải
      tải backend đồ hoạ khi chỉ chạy train/predict.
    - `matplotlib.use("Agg")` phải gọi TRƯỚC khi import `pyplot`: Agg là backend
      không cần màn hình, nếu không server/CI sẽ lỗi vì thiếu display.
    - `finally: plt.close(fig)` để giải phóng figure kể cả khi savefig lỗi,
      tránh rò rỉ bộ nhớ khi pipeline chạy nhiều lần trong cùng tiến trình.
    Số trong ô được ghi thủ công bằng `ax.text` vì imshow chỉ tô màu, không in giá trị.
    """
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
    """Xuất độ quan trọng đặc trưng; trả False nếu model không hỗ trợ.

    Ba loại model cho ra ba dạng khác nhau nên phải tách nhánh:
    - RF/GB có `feature_importances_` trực tiếp (mảng >= 0, tổng 1).
    - Logistic Regression nằm trong Pipeline nên phải lấy `named_steps["model"]`
      rồi dùng `abs(coef_)`: dấu của coef chỉ nói chiều tác động, còn "quan trọng"
      là độ lớn. `np.ravel` vì `coef_` của bài toán 2 lớp có shape (1, n_features).
    - Model nào không rơi vào hai nhánh trên thì `values is None`.

    Chốt an toàn `len(values) != len(FEATURE_COLUMNS)`: nếu số hệ số không khớp
    số cột đặc trưng thì việc zip tên với giá trị sẽ gán SAI nhãn (ví dụ khi
    pipeline có bước tạo thêm cột). Trường hợp đó thà bỏ file còn hơn xuất báo
    cáo lệch, nên trả False để caller ghi `feature_importance_written = False`.
    """
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
    """Xuất precision/recall/f1 từng lớp trên TEST ra CSV.

    Điểm cần lưu ý:
    - `predict_with_threshold(..., _artifact_threshold(...))`: dùng ngưỡng đã chọn
      từ OOF, KHÔNG phải 0.5 mặc định của `model.predict`, nên số ở đây khớp với
      số mà web app thực sự phục vụ.
    - `labels=[0, 1]` là bắt buộc: nếu TEST (hoặc dự báo) tình cờ chỉ có 1 lớp,
      sklearn sẽ trả bảng thiếu dòng và code bên dưới sẽ KeyError. Truyền labels
      cố định để bảng luôn có đủ NOT_UP và UP (giá trị 0 thay vì mất dòng).
    - Dòng "accuracy" được nhồi vào cột `precision` (còn recall/f1 để rỗng) vì
      accuracy là số vô hướng, không có phiên bản theo lớp; support lấy từ
      `macro avg` = tổng số mẫu TEST.
    """
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
    """Ghi file markdown giải thích siêu tham số — nội dung TĨNH, không đọc từ model.

    Đây là tài liệu cho người đọc báo cáo, không phải dump cấu hình thực tế: chuỗi
    được hardcode nên nếu ai đó đổi `CV_N_SPLITS`/`CV_GAP_SESSIONS` trong
    `config/settings.py` mà quên sửa file này thì văn bản sẽ lệch với thực tế.
    Số liệu cấu hình đáng tin là `cv_config` trong `model_metadata.json`.

    Toàn bộ text viết KHÔNG DẤU (tieng viet khong dau) để tránh lệ thuộc encoding
    khi file được mở bằng editor/console dùng codepage Windows mặc định.
    """
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
    """Dựng `model_metadata.json` — hợp đồng định danh giữa training và serving.

    Metadata này là thứ duy nhất tầng inference (`prediction_service`) và chatbot
    dùng để biết artifact đang phục vụ là bản nào, nên mọi field ở đây đều có
    người tiêu thụ:
    - `content_fingerprint` / `training_content_fingerprint`: hash nội dung dataset.
      Nếu hai giá trị lệch nhau nghĩa là model được huấn luyện trên dataset khác
      với dataset hiện tại → serving phải cảnh báo stale.
    - `feature_order`: thứ tự cột lúc fit. Inference PHẢI xếp cột theo đúng list
      này, vì sklearn chỉ nhớ vị trí chứ không nhớ tên.
    - `decision_threshold`: ngưỡng chọn từ OOF; thiếu nó là serving rơi về 0.5 và
      cho ra tỷ lệ UP hoàn toàn khác báo cáo.
    - `training_symbols` + `training_symbol_count`: phạm vi mã đã học, lấy từ
      TRAIN+VALIDATION (chính là tập cuối dùng để refit) và chuẩn hoá
      `strip().upper()` để so khớp không lệ thuộc cách người dùng gõ. Chatbot
      dùng cặp field này để từ chối mã ngoài phạm vi; `count` được ghi riêng để
      bên đọc phát hiện file bị sửa tay (count != len(list) → inconsistent).
    - `train_through_date` = max `label_end_date` của TRAIN+VALIDATION, tức ngày
      cuối cùng mà nhãn t+5 đã "nhìn thấy" — dùng để đo độ trễ dữ liệu.
    - Các mốc split lấy từ `resolve_protocol_dates` trên toàn bộ dữ liệu, KHÔNG
      lấy hằng số trong settings, vì protocol là rolling theo ngày mới nhất.
    - `test_metrics` là bản sao của `final_test_metrics`, giữ lại chỉ để tương
      thích các report/UI cũ; nguồn chuẩn là `final_test_metrics`.
    """
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
    """Ghi `model_metadata.json` — bọc mỏng quanh `build_model_metadata`.

    Tách dựng-dict và ghi-file để test có thể kiểm nội dung metadata mà không
    chạm đĩa. `write_json` ghi nguyên tử (tmp + `os.replace`) và dùng
    `allow_nan=False`, nên nếu một metric nào lỡ mang NaN thì `_json_safe` phải
    đổi thành `None` trước — chi tiết ở `services/pipeline_utils.py`.
    """
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
    """Ghi toàn bộ bộ báo cáo của một lần chạy pipeline ra `reports/`.

    Thứ tự và ý nghĩa từng file:
    - `model_comparison.csv`: bảng xếp hạng VALIDATION (kèm baseline) — bằng chứng
      cho việc chọn model.
    - `final_model_evaluation.csv`: dòng TEST của model thắng + baseline TEST. Nếu
      caller không truyền `final_test_evaluation` (đường chạy cũ) thì hàm tự dựng
      lại từ `selected_artifact["final_test_metrics"]` để file không bị thiếu.
    - `model_selection_report.txt`, `classification_report.csv`,
      `confusion_matrix.csv/.png`, `hyperparameter_explanation.md`,
      `feature_importance.csv` (chỉ khi model hỗ trợ).

    Mọi CSV đều ghi qua `atomic_dataframe_to_csv` (tmp + os.replace) nên người đọc
    web/report không bao giờ nhìn thấy file nửa vời.

    Confusion matrix được tính lại tại đây bằng ĐÚNG ngưỡng của artifact
    (`_artifact_threshold`), không phải 0.5, để khớp con số trong
    `classification_report.csv`.

    `summary` bị mutate tại chỗ (thêm `policy_id`, `content_fingerprint`,
    `validation_selection`, các ô confusion matrix, `report_files`) vì caller
    dùng chính dict đó làm `pipeline_summary.json`. Giá trị trả về chỉ là vài số
    để pipeline in log/kiểm tra nhanh.
    """
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
    """Hậu kiểm tính nhất quán của kết quả chọn model; sai thì raise, không sửa.

    Ba bất biến, đều là dấu hiệu logic chọn model đã hỏng nếu vi phạm:
    1. Đúng 1 dòng `selected` — 0 nghĩa là không chọn được ai, >1 nghĩa là tie-break
       thất bại và artifact xuất ra sẽ mơ hồ.
    2. Model được chọn phải nằm trong `CANDIDATE_MODEL_IDS` — chặn trường hợp một
       dòng baseline (always-UP / always-NOT_UP) lọt vào vị trí model cuối cùng.
    3. `model_name` của artifact phải trùng dòng comparison — nếu lệch thì file
       .pkl đang lưu một model khác với model mà báo cáo đang mô tả.
    """
    selected_count = int(comparison["selected"].sum())
    if selected_count != 1:
        raise ValueError(f"Expected exactly 1 selected model, got {selected_count}")
    selected_row = comparison[comparison["selected"]].iloc[0]
    if selected_row["model_id"] not in CANDIDATE_MODEL_IDS:
        raise ValueError("Only LR, RF or GB can be selected as final model.")
    if selected_artifact["model_name"] != selected_row["model_name"]:
        raise ValueError("Selected artifact model_name does not match comparison row.")
