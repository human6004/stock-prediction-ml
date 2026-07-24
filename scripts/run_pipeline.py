"""Run the official rolling TRAIN/VALIDATION/TEST pipeline.

Flow: raw -> clean/feature/label -> rolling split -> manual CV config -> choose on
VALIDATION -> evaluate one refit winner on TEST -> publish artifact and reports.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import (  # noqa: E402
    CLEANED_DATA_PATH,
    EXPERIMENT_POLICY_ID,
    FEATURE_DATA_PATH,
    FINAL_MODEL_PATH,
    ML_DATA_PATH,
    MODEL_METADATA_PATH,
    PIPELINE_SUMMARY_PATH,
)
from services.experiment_state import (  # noqa: E402
    acquire_pipeline_lock,
    compute_dataset_fingerprint,
    compute_experiment_fingerprint,
    has_evaluated_snapshot,
    is_config_complete,
    read_manual_config,
    release_pipeline_lock,
    write_evaluation_entry,
)
from services.feature_engineering import (  # noqa: E402
    build_features,
    create_labels,
    protocol_time_split,
    verify_protocol_splits,
    write_feature_outputs,
    write_split_summary,
)
from services.model_evaluation import (  # noqa: E402
    evaluate_baselines,
    evaluate_validation_candidates,
    build_model_metadata,
    refit_and_evaluate_final_model,
    select_final_model,
    verify_model_selection,
    write_reports,
)
from services.model_tuning import tune_models  # noqa: E402
from services.pipeline_utils import atomic_model_release, write_json  # noqa: E402
from services.preprocessing import clean_data, dataset_check, write_clean_outputs  # noqa: E402


def print_report(title: str, items: list[tuple[str, object]]) -> None:
    print(f"\n{title}:")
    for index, (label, value) in enumerate(items, start=1):
        print(f"{index}. {label}: {value}")


def _normalize_fingerprint_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Mirror numeric parsing after ml_dataset.csv is written and read."""
    normalized = dataset.copy()
    numeric_columns = normalized.select_dtypes(include="number").columns
    normalized[numeric_columns] = normalized[numeric_columns].apply(
        lambda series: pd.to_numeric(series.astype(str), errors="coerce")
    )
    return normalized


def _guard_unevaluated_snapshot(fingerprint_hash: str | None) -> None:
    if fingerprint_hash and has_evaluated_snapshot(fingerprint_hash):
        raise RuntimeError(
            "Snapshot dữ liệu này đã được đánh giá trên TEST "
            f"(fingerprint={fingerprint_hash})."
        )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args:
        raise ValueError("Official pipeline không nhận tham số dòng lệnh.")
    owner_token = acquire_pipeline_lock()
    if owner_token is None:
        print("Pipeline đang chạy ở tiến trình khác (experiments/pipeline.lock). Thoát.")
        sys.exit(1)
    try:
        _run_pipeline()
    finally:
        release_pipeline_lock(owner_token)


def _run_pipeline() -> None:
    print("Starting HOSE stock prediction pipeline...")
    evaluation_fingerprint: str | None = None
    evaluation_started = False
    try:
        raw_df, dataset_report = dataset_check()
        cleaned_all, cleaned_for_training, symbol_stats, clean_report = clean_data(raw_df)
        features, feature_report = build_features(cleaned_for_training)
        ml_dataset, label_report = create_labels(cleaned_for_training, features)
        train, validation, test, split_report = protocol_time_split(ml_dataset)
        train_validation = pd.concat([train, validation], ignore_index=True)
        verify_protocol_splits(train, validation, test, dates=split_report)

        fingerprint_dataset = _normalize_fingerprint_dataset(ml_dataset)
        tuning_fingerprint = compute_dataset_fingerprint(fingerprint_dataset)
        experiment_fingerprint = compute_experiment_fingerprint(fingerprint_dataset)
        del fingerprint_dataset
        evaluation_fingerprint = experiment_fingerprint["hash"]
        _guard_unevaluated_snapshot(evaluation_fingerprint)

        config = read_manual_config()
        if not is_config_complete(config, tuning_fingerprint["hash"]):
            raise RuntimeError(
                "Pipeline cần cấu hình CV do người dùng chọn cho LR, RF và GB "
                "trên đúng content fingerprint hiện tại."
            )

        write_clean_outputs(cleaned_all, symbol_stats)
        write_feature_outputs(features, ml_dataset)
        write_split_summary(train, test, validation)

        written_tuning_fingerprint = compute_dataset_fingerprint()
        written_experiment_fingerprint = compute_experiment_fingerprint()
        if (
            written_tuning_fingerprint["content_hash"]
            != tuning_fingerprint["content_hash"]
            or written_experiment_fingerprint["content_hash"]
            != experiment_fingerprint["content_hash"]
        ):
            raise RuntimeError("Fingerprint thay đổi sau khi ghi ml_dataset.csv.")
        _guard_unevaluated_snapshot(written_experiment_fingerprint["hash"])

        fitted_artifacts, tuning_df, tuning_report = tune_models(train)
        candidate_comparison = evaluate_validation_candidates(
            validation, fitted_artifacts
        )
        baseline_comparison = evaluate_baselines(
            validation["target"], split="validation"
        )
        comparison = pd.DataFrame(
            [
                *candidate_comparison.to_dict(orient="records"),
                *baseline_comparison.to_dict(orient="records"),
            ]
        )
        comparison, selected_artifact, selection_report = select_final_model(
            comparison, fitted_artifacts
        )
        verify_model_selection(comparison, selected_artifact)

        write_evaluation_entry(
            {
                "experiment_fingerprint": evaluation_fingerprint,
                "content_fingerprint": experiment_fingerprint["content_hash"],
                "policy_id": EXPERIMENT_POLICY_ID,
                "status": "started",
            }
        )
        evaluation_started = True

        selected_artifact, final_test_evaluation = refit_and_evaluate_final_model(
            selected_artifact,
            train_validation,
            test,
        )
        final_test_row = final_test_evaluation.loc[
            final_test_evaluation["row_type"] == "final"
        ].iloc[0]
        selected_artifact["training_content_fingerprint"] = tuning_fingerprint[
            "content_hash"
        ]
        selected_artifact["tuning_fingerprint"] = tuning_fingerprint["hash"]
        selected_artifact["content_fingerprint"] = experiment_fingerprint[
            "content_hash"
        ]
        selected_artifact["experiment_fingerprint"] = evaluation_fingerprint

        write_evaluation_entry(
            {
                "experiment_fingerprint": evaluation_fingerprint,
                "content_fingerprint": experiment_fingerprint["content_hash"],
                "policy_id": EXPERIMENT_POLICY_ID,
                "status": "evaluated",
                "model_name": selection_report["selected_model_name"],
                "test_metrics": selected_artifact.get("final_test_metrics", {}),
            }
        )

        summary = {
            "dataset_report": dataset_report,
            "clean_report": clean_report,
            "feature_report": feature_report,
            "label_report": label_report,
            "split_report": split_report,
            "validation_report": {
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "train_label_end_max": str(train["label_end_date"].max()),
                "validation_date_min": str(validation["trading_date"].min()),
                "validation_label_end_max": str(
                    validation["label_end_date"].max()
                ),
                "test_date_min": str(test["trading_date"].min()),
            },
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
            "dataset_fingerprint": tuning_fingerprint["hash"],
            "tuning_fingerprint": tuning_fingerprint["hash"],
            "experiment_fingerprint": evaluation_fingerprint,
        }
        report_meta = write_reports(
            test,
            comparison,
            selected_artifact,
            summary,
            selection_report=selection_report,
            final_test_evaluation=final_test_evaluation,
            content_fingerprint=experiment_fingerprint["content_hash"],
        )

        metadata = build_model_metadata(
            selected_artifact,
            train,
            validation,
            test,
            selection_report,
            content_fingerprint=experiment_fingerprint["content_hash"],
        )
        atomic_model_release(
            selected_artifact,
            metadata,
            model_path=FINAL_MODEL_PATH,
            metadata_path=MODEL_METADATA_PATH,
        )
        write_json(PIPELINE_SUMMARY_PATH, summary)
        write_evaluation_entry(
            {
                "experiment_fingerprint": evaluation_fingerprint,
                "content_fingerprint": experiment_fingerprint["content_hash"],
                "policy_id": EXPERIMENT_POLICY_ID,
                "status": "published",
                "model_name": selection_report["selected_model_name"],
                "test_metrics": selected_artifact.get("final_test_metrics", {}),
            }
        )

        print_report(
            "DATASET CHECK",
            [
                ("Rows", dataset_report["rows"]),
                ("Symbols", dataset_report["symbols"]),
                (
                    "Date range",
                    f"{dataset_report['date_min']} -> {dataset_report['date_max']}",
                ),
            ],
        )
        print_report(
            "ROLLING TRAIN / VALIDATION / TEST",
            [
                ("Train through", split_report["train_end_date"]),
                ("Validation through", split_report["validation_end_date"]),
                ("TEST through", split_report["test_end_date"]),
                ("Train rows", len(train)),
                ("Validation rows", len(validation)),
                ("Final test rows", len(test)),
            ],
        )
        print_report(
            "MODEL SELECTION",
            [
                ("Selected model", selection_report["selected_model_name"]),
                ("VALIDATION F1_UP", f"{selection_report['selected_f1_up']:.4f}"),
                (
                    "VALIDATION Recall_UP",
                    f"{selection_report['selected_recall_up']:.4f}",
                ),
                ("TEST baseline passed", selected_artifact["baseline_passed"]),
                ("FINAL TEST F1_UP", f"{final_test_row['f1_up']:.4f}"),
                ("CV F1_UP", selection_report.get("selected_cv_f1_up")),
            ],
        )
        print_report(
            "REPORTS",
            [
                ("Confusion matrix rows", report_meta["test_rows"]),
                (
                    "Feature importance written",
                    report_meta["feature_importance_written"],
                ),
            ],
        )
        print("\nPipeline completed successfully.")
    except Exception as exc:
        if evaluation_started and evaluation_fingerprint:
            write_evaluation_entry(
                {
                    "experiment_fingerprint": evaluation_fingerprint,
                    "policy_id": EXPERIMENT_POLICY_ID,
                    "status": "release_failed",
                    "error": str(exc),
                }
            )
        raise


if __name__ == "__main__":
    main()
