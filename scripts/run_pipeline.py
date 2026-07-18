"""Entrypoint chạy official pipeline từ raw CSV đến model/report/SQLite.

Luồng: kiểm tra raw -> clean -> feature/label -> split -> CV + fit TRAIN
-> evaluate TEST -> chọn final model -> ghi report -> sync database.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import (  # noqa: E402
    CLEANED_DATA_PATH,
    FEATURE_DATA_PATH,
    FINAL_MODEL_PATH,
    ML_DATA_PATH,
    PIPELINE_SUMMARY_PATH,
    ROADMAP_PATH,
)
from services.database_service import sync_all  # noqa: E402
from services.experiment_state import (  # noqa: E402
    acquire_pipeline_lock,
    compute_dataset_fingerprint,
    is_test_locked,
    release_pipeline_lock,
    write_test_lock,
)
from services.feature_engineering import (  # noqa: E402
    build_features,
    create_labels,
    time_based_split,
    verify_data_pipeline,
    write_feature_outputs,
    write_train_test_summary,
)
from services.model_evaluation import (  # noqa: E402
    evaluate_tuned_models,
    select_final_model,
    verify_model_selection,
    write_reports,
)
from services.model_tuning import tune_models  # noqa: E402
from services.pipeline_utils import cleanup_outputs, read_roadmap  # noqa: E402
from services.preprocessing import clean_data, dataset_check, write_clean_outputs  # noqa: E402


def print_report(title: str, items: list[tuple[str, object]]) -> None:
    print(f"\n{title}:")
    for index, (label, value) in enumerate(items, start=1):
        print(f"{index}. {label}: {value}")


def main() -> None:
    if not acquire_pipeline_lock():
        print("Pipeline đang chạy ở tiến trình khác (experiments/pipeline.lock). Thoát.")
        sys.exit(1)
    try:
        _run_pipeline()
    finally:
        release_pipeline_lock()


def _run_pipeline() -> None:
    print("Starting HOSE stock prediction pipeline (roadmap-aligned)...")
    roadmap_report = read_roadmap()
    if not roadmap_report["read_success"]:
        raise FileNotFoundError(f"Cannot read roadmap: {ROADMAP_PATH}")

    # Giai đoạn 1: dựng lại toàn bộ dataset học từ raw OHLCV.
    raw_df, dataset_report = dataset_check()
    cleanup_report = cleanup_outputs()

    cleaned_all, cleaned_for_training, symbol_stats, clean_report = clean_data(raw_df)
    write_clean_outputs(cleaned_all, symbol_stats)

    features, feature_report = build_features(cleaned_for_training)
    ml_dataset, label_report = create_labels(features)
    train, test, split_report = time_based_split(ml_dataset)
    verify_data_pipeline(train, test)
    write_feature_outputs(features, ml_dataset)
    write_train_test_summary(train, test)

    # Giai đoạn 2: chặn việc xem lại cùng TEST sau khi đã biết kết quả.
    # Lưu ý cleanup hiện chạy trước guard này; xem cảnh báo trong tài liệu project.
    fingerprint = compute_dataset_fingerprint()
    allow_reeval = os.environ.get("STOCK_ALLOW_TEST_REEVAL") == "1"
    if is_test_locked(fingerprint["hash"]) and not allow_reeval:
        raise RuntimeError(
            "TEST của dataset hiện tại đã được dùng "
            f"(fingerprint={fingerprint['hash']}). Không đánh giá lại trên cùng "
            "tập test để tránh leakage. Đổi/cập nhật dữ liệu để có fingerprint mới, "
            "hoặc đặt STOCK_ALLOW_TEST_REEVAL=1 (chỉ dành cho debug)."
        )

    # Giai đoạn 3: CV/fit chỉ trên TRAIN, sau đó mới chạm TEST để chọn final.
    fitted_artifacts, tuning_df, tuning_report = tune_models(train)
    comparison, fitted_artifacts = evaluate_tuned_models(train, test, fitted_artifacts)
    comparison, selected_artifact, selection_report = select_final_model(
        comparison, fitted_artifacts
    )
    verify_model_selection(comparison, selected_artifact)

    # Giai đoạn 4: đóng gói provenance và mọi output phục vụ demo/bảo vệ.
    summary = {
        "roadmap_report": roadmap_report,
        "dataset_report": dataset_report,
        "cleanup_report": cleanup_report,
        "clean_report": clean_report,
        "feature_report": feature_report,
        "label_report": label_report,
        "split_report": split_report,
        "tuning_report": tuning_report,
        "tuning_results": tuning_df.to_dict(orient="records"),
        "selection_report": selection_report,
        "class_distribution_test": {
            "NOT_UP": int((test["target"] == 0).sum()),
            "UP": int((test["target"] == 1).sum()),
        },
        "output_files": [
            str(CLEANED_DATA_PATH.relative_to(ROOT_DIR)),
            str(FEATURE_DATA_PATH.relative_to(ROOT_DIR)),
            str(ML_DATA_PATH.relative_to(ROOT_DIR)),
            str(FINAL_MODEL_PATH.relative_to(ROOT_DIR)),
        ],
    }
    report_meta = write_reports(
        test,
        comparison,
        selected_artifact,
        summary,
        train=train,
        selection_report=selection_report,
    )

    # Database là bản sync phụ; lỗi DB không được làm mất model/report đã tạo.
    try:
        db_report = sync_all()
        summary["database_sync"] = db_report
    except Exception as exc:
        summary["database_sync_error"] = str(exc)

    PIPELINE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary["dataset_fingerprint"] = fingerprint["hash"]
    write_test_lock(fingerprint["hash"], selection_report["selected_model_name"])

    print_report(
        "ROADMAP CHECK",
        [
            ("Roadmap path", ROADMAP_PATH),
            ("Read success", roadmap_report["read_success"]),
        ],
    )
    print_report(
        "DATASET CHECK",
        [
            ("Rows", dataset_report["rows"]),
            ("Symbols", dataset_report["symbols"]),
            ("Date range", f"{dataset_report['date_min']} -> {dataset_report['date_max']}"),
        ],
    )
    print_report(
        "SPLIT (label_end_date)",
        [
            ("Split date", split_report["split_date"]),
            ("Train rows", split_report["train_rows"]),
            ("Test rows", split_report["test_rows"]),
            ("Train label_end max", split_report["train_label_end_max"]),
            ("Test label_end min", split_report["test_label_end_min"]),
        ],
    )
    print_report(
        "MODEL SELECTION",
        [
            ("Selected model", selection_report["selected_model_name"]),
            ("F1_UP", f"{selection_report['selected_f1_up']:.4f}"),
            ("Recall_UP", f"{selection_report['selected_recall_up']:.4f}"),
            ("CV F1_UP", selection_report.get("selected_cv_f1_up")),
        ],
    )
    print_report(
        "REPORTS",
        [
            ("Confusion matrix rows", report_meta["test_rows"]),
            ("Feature importance written", report_meta["feature_importance_written"]),
        ],
    )
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
