"""Focused checks for validation-first final-model selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import scripts.run_pipeline as pipeline_script
from config.settings import CV_N_SPLITS, FEATURE_COLUMNS
from services import model_evaluation
from services.model_evaluation import (
    evaluate_baselines,
    evaluate_validation_candidates,
    refit_and_evaluate_final_model,
    select_final_model,
    write_model_metadata,
)
from services.pipeline_utils import atomic_joblib_dump
from config.settings import EXPERIMENT_POLICY_ID


class ScoreClassifier(ClassifierMixin, BaseEstimator):
    """Small cloneable estimator whose first feature is the UP score."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.fit_rows_ = len(X)
        return self

    def predict_proba(self, X):
        scores = np.asarray(X.iloc[:, 0], dtype=float)
        return np.column_stack([1 - scores, scores])


def frame(dates, targets, scores=None):
    scores = scores or [0.5] * len(dates)
    rows = []
    for date, target, score in zip(dates, targets, scores):
        row = {column: 0.0 for column in FEATURE_COLUMNS}
        row[FEATURE_COLUMNS[0]] = score
        row.update(
            {
                "symbol": "FPT",
                "trading_date": date,
                "label_end_date": date,
                "target": target,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class ProtocolSelectionTests(unittest.TestCase):
    def test_only_three_candidates_are_scored_on_validation(self):
        validation = frame(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            [0, 1, 0, 1],
            [0.1, 0.9, 0.2, 0.8],
        )
        artifacts = {
            1: {"model": Mock(side_effect=AssertionError("Dummy must not run"))},
            **{
                model_id: {
                    "model": ScoreClassifier().fit(validation[FEATURE_COLUMNS], validation["target"]),
                    "model_id": model_id,
                    "model_name": name,
                    "cv_f1_up": 0.4,
                }
                for model_id, name in (
                    (2, "Logistic Regression"),
                    (3, "Random Forest"),
                    (4, "Gradient Boosting"),
                )
            },
        }

        comparison = evaluate_validation_candidates(validation, artifacts)

        self.assertEqual(comparison["model_id"].tolist(), [2, 3, 4])
        self.assertEqual(set(comparison["split"]), {"validation"})
        self.assertEqual(set(comparison["row_type"]), {"candidate"})

    def test_baselines_are_explicit_and_block_weaker_candidate(self):
        baselines = evaluate_baselines(pd.Series([0, 1, 1]), split="validation")
        candidates = pd.DataFrame(
            [
                {"model_id": 2, "model_name": "Logistic Regression", "f1_up": 0.5, "recall_up": 0.7},
                {"model_id": 3, "model_name": "Random Forest", "f1_up": 0.5, "recall_up": 0.7},
                {"model_id": 4, "model_name": "Gradient Boosting", "f1_up": 0.49, "recall_up": 0.9},
            ]
        )
        comparison = pd.concat([candidates, baselines], ignore_index=True, sort=False)
        artifacts = {model_id: {"model_id": model_id} for model_id in (2, 3, 4)}

        with self.assertRaisesRegex(RuntimeError, "baseline"):
            select_final_model(comparison, artifacts)

        self.assertEqual(set(baselines["model_name"]), {"Always UP", "Always NOT_UP"})

    def test_recall_breaks_equal_f1_before_simplicity(self):
        comparison = pd.DataFrame(
            [
                {"model_id": 2, "f1_up": 0.5, "recall_up": 0.6},
                {"model_id": 3, "f1_up": 0.5, "recall_up": 0.7},
                {"model_id": 4, "f1_up": 0.49, "recall_up": 0.9},
            ]
        )
        for metric in model_evaluation.METRIC_COLUMNS:
            if metric not in comparison:
                comparison[metric] = 0.0
        artifacts = {
            model_id: {"model_id": model_id, "cv_f1_up": 0.4}
            for model_id in (2, 3, 4)
        }

        _, selected, _ = select_final_model(comparison, artifacts)

        self.assertEqual(selected["model_id"], 3)


class FinalRefitTests(unittest.TestCase):
    def test_fresh_winner_is_refit_then_the_exact_tested_artifact_is_saved(self):
        train = frame(["2025-01-01", "2025-01-02"], [0, 1], [0.1, 0.9])
        validation = frame(["2025-01-03", "2025-01-04"], [0, 1], [0.2, 0.8])
        test = frame(["2025-01-05", "2025-01-06"], [0, 1], [0.6, 0.9])
        original = ScoreClassifier().fit(train[FEATURE_COLUMNS], train["target"])
        selected = {
            "model": original,
            "model_id": 2,
            "model_name": "Logistic Regression",
            "feature_columns": FEATURE_COLUMNS,
            "prediction_horizon": 5,
            "up_threshold": 0.01,
            "decision_threshold": 0.8,
            "best_params": {},
            "cv_config": {"n_splits": 4, "gap": 5},
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_model.pkl"
            final_artifact, test_evaluation = refit_and_evaluate_final_model(
                selected, pd.concat([train, validation]), test
            )
            atomic_joblib_dump(final_artifact, path)
            saved = joblib.load(path)

        self.assertIsNot(final_artifact["model"], original)
        self.assertEqual(final_artifact["model"].fit_rows_, 4)
        self.assertEqual(set(test_evaluation["split"]), {"test"})
        self.assertEqual(
            set(test_evaluation["model_name"]),
            {"Logistic Regression", "Always UP", "Always NOT_UP"},
        )
        self.assertEqual(len(test_evaluation), 3)
        self.assertEqual(saved["final_test_metrics"], final_artifact["final_test_metrics"])
        self.assertTrue(final_artifact["baseline_passed"])
        self.assertEqual(saved["baseline_passed"], final_artifact["baseline_passed"])
        self.assertEqual(saved["model"].fit_rows_, 4)
        self.assertEqual(final_artifact["final_test_metrics"]["accuracy"], 1.0)


class MetadataAndAtomicWriteTests(unittest.TestCase):
    def test_metadata_separates_validation_selection_from_final_test(self):
        train = frame(["2025-01-01", "2025-01-02"], [0, 1])
        validation = frame(["2025-01-03"], [1])
        test = frame(["2025-01-04"], [0])
        train.loc[1, "symbol"] = "VNM"
        validation.loc[:, "symbol"] = "ACB"
        test.loc[:, "symbol"] = "HPG"
        artifact = {
            "model_name": "Logistic Regression",
            "model_id": 2,
            "prediction_horizon": 5,
            "best_params": {"C": 1.0},
            "cv_f1_up": 0.4,
            "cv_config": {"n_splits": CV_N_SPLITS, "gap": 5},
            "trained_at": "2026-07-19T00:00:00",
            "validation_selection_metrics": {"f1_up": 0.5},
            "final_test_metrics": {"f1_up": 0.45},
            "baseline_passed": False,
            "baseline_warning": "below always-UP on TEST",
        }
        selection = {
            "validation_baseline_passed": False,
            "validation_baseline_warning": "below baseline on validation",
            "validation_selection_metrics": {"f1_up": 0.5},
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            with patch.object(model_evaluation, "MODEL_METADATA_PATH", path):
                write_model_metadata(
                    artifact,
                    train,
                    validation,
                    test,
                    selection,
                    content_fingerprint="abc123",
                )
            metadata = pd.read_json(path, typ="series")

        self.assertEqual(metadata["policy_id"], EXPERIMENT_POLICY_ID)
        self.assertEqual(metadata["content_fingerprint"], "abc123")
        self.assertFalse(metadata["baseline_passed"])
        self.assertEqual(metadata["train_through_date"], "2025-01-03")
        self.assertEqual(metadata["validation_selection_metrics"]["f1_up"], 0.5)
        self.assertEqual(metadata["final_test_metrics"]["f1_up"], 0.45)
        self.assertEqual(metadata["training_symbols"], ["ACB", "FPT", "VNM"])
        self.assertEqual(metadata["training_symbol_count"], 3)

    def test_atomic_dump_leaves_no_partial_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.pkl"
            atomic_joblib_dump({"version": 1}, path)
            atomic_joblib_dump({"version": 2}, path)

            self.assertEqual(joblib.load(path), {"version": 2})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_atomic_dump_failure_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.pkl"
            atomic_joblib_dump({"version": 1}, path)
            with patch(
                "services.pipeline_utils.joblib.dump", side_effect=OSError("disk full")
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_joblib_dump({"version": 2}, path)

            self.assertEqual(joblib.load(path), {"version": 1})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_reports_keep_validation_comparison_separate_from_final_test(self):
        test = frame(["2025-01-05", "2025-01-06"], [0, 1], [0.1, 0.9])
        metrics = {
            "accuracy": 1.0,
            "precision_up": 1.0,
            "recall_up": 1.0,
            "f1_up": 1.0,
            "precision_not_up": 1.0,
            "recall_not_up": 1.0,
            "f1_not_up": 1.0,
        }
        comparison = pd.DataFrame(
            [
                {
                    "model_id": 2,
                    "model_name": "Logistic Regression",
                    "split": "validation",
                    "row_type": "candidate",
                    "cv_f1_up": 0.8,
                    **metrics,
                    "selected": True,
                }
            ]
        )
        final_row = {
            "model_id": 2,
            "model_name": "Logistic Regression",
            "split": "test",
            "row_type": "final",
            "cv_f1_up": 0.8,
            **metrics,
            "selected": True,
        }
        final_evaluation = pd.DataFrame(
            [
                final_row,
                *evaluate_baselines(test["target"], split="test").to_dict(
                    orient="records"
                ),
            ]
        )
        artifact = {
            "model": ScoreClassifier().fit(test[FEATURE_COLUMNS], test["target"]),
            "model_id": 2,
            "model_name": "Logistic Regression",
            "validation_selection_metrics": metrics,
            "final_test_metrics": metrics,
            "metrics": metrics,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = (
                patch.object(model_evaluation, "MODEL_COMPARISON_PATH", root / "comparison.csv"),
                patch.object(model_evaluation, "FINAL_MODEL_EVALUATION_PATH", root / "final.csv"),
                patch.object(model_evaluation, "CONFUSION_MATRIX_CSV_PATH", root / "cm.csv"),
                patch.object(model_evaluation, "write_model_selection_report"),
                patch.object(model_evaluation, "write_classification_report"),
                patch.object(model_evaluation, "write_hyperparameter_explanation"),
                patch.object(model_evaluation, "save_confusion_matrix_png"),
                patch.object(model_evaluation, "write_feature_importance", return_value=False),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                model_evaluation.write_reports(
                    test,
                    comparison,
                    artifact,
                    {},
                    final_test_evaluation=final_evaluation,
                )
            validation_report = pd.read_csv(root / "comparison.csv")
            final_report = pd.read_csv(root / "final.csv")

        self.assertEqual(set(validation_report["split"]), {"validation"})
        self.assertEqual(final_report["split"].tolist(), ["test", "test", "test"])
        self.assertEqual(
            set(final_report["model_name"]),
            {"Logistic Regression", "Always UP", "Always NOT_UP"},
        )


class PipelineGuardTests(unittest.TestCase):
    def test_preflight_waits_for_prospective_dataset_fingerprint(self):
        with patch.object(
            pipeline_script, "has_evaluated_snapshot", return_value=False
        ) as has_evaluated_snapshot:
            pipeline_script._guard_unevaluated_snapshot(None)
            has_evaluated_snapshot.assert_not_called()

            pipeline_script._guard_unevaluated_snapshot("new-fingerprint")
            has_evaluated_snapshot.assert_called_once_with("new-fingerprint")

    def test_evaluated_snapshot_cannot_be_reused(self):
        with patch.object(
            pipeline_script, "has_evaluated_snapshot", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "đã được đánh giá"):
                pipeline_script._guard_unevaluated_snapshot("fingerprint")

    def test_guard_runs_before_dataset_outputs(self):
        write_clean = Mock()
        with (
            patch.object(pipeline_script, "dataset_check", return_value=(pd.DataFrame(), {})),
            patch.object(pipeline_script, "clean_data", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})),
            patch.object(pipeline_script, "build_features", return_value=(pd.DataFrame(), {})),
            patch.object(pipeline_script, "create_labels", return_value=(pd.DataFrame(), {})),
            patch.object(pipeline_script, "protocol_time_split", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})),
            patch.object(pipeline_script, "verify_protocol_splits"),
            patch.object(pipeline_script, "compute_dataset_fingerprint", return_value={"hash": "train", "content_hash": "content"}),
            patch.object(pipeline_script, "compute_experiment_fingerprint", return_value={"hash": "snapshot", "content_hash": "content"}),
            patch.object(pipeline_script, "_guard_unevaluated_snapshot", side_effect=RuntimeError("evaluated")),
            patch.object(pipeline_script, "write_clean_outputs", write_clean),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluated"):
                pipeline_script._run_pipeline()

        write_clean.assert_not_called()


if __name__ == "__main__":
    unittest.main()
