"""Contracts for recent purged CV and constrained OOF thresholds."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from services import model_tuning
from services.model_tuning import select_oof_threshold
from services.time_splitting import iter_purged_date_splits


class RecentCvTests(unittest.TestCase):
    def test_default_cv_has_four_recent_purged_folds(self):
        dates = pd.date_range("2020-01-01", periods=900, freq="D")
        frame = pd.DataFrame(
            {
                "symbol": ["AAA"] * len(dates),
                "trading_date": dates.strftime("%Y-%m-%d"),
                "label_end_date": [
                    dates[min(index + 5, len(dates) - 1)].strftime("%Y-%m-%d")
                    for index in range(len(dates))
                ],
            }
        )

        folds = list(iter_purged_date_splits(frame))

        self.assertEqual(len(folds), 4)
        market_dates = list(frame["trading_date"].unique())
        for train_idx, validation_idx in folds:
            train = frame.iloc[train_idx]
            validation = frame.iloc[validation_idx]
            self.assertGreaterEqual(train["trading_date"].min(), "2021-01-01")
            train_end = market_dates.index(train["trading_date"].max())
            validation_start = market_dates.index(validation["trading_date"].min())
            self.assertEqual(validation_start - train_end - 1, 5)
            self.assertLess(
                train["label_end_date"].max(), validation["trading_date"].min()
            )


class OofThresholdTests(unittest.TestCase):
    def test_threshold_maximizes_f1_without_predicting_up_too_often(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.3, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.05])

        result = select_oof_threshold(y_true, probabilities)

        self.assertAlmostEqual(result["decision_threshold"], 0.9)
        self.assertLessEqual(result["oof_predicted_up_ratio"], 0.5)
        self.assertGreaterEqual(result["oof_precision_up"], result["oof_up_rate"])
        self.assertTrue(result["threshold_constraint_passed"])


class SelectedRunReuseTests(unittest.TestCase):
    @staticmethod
    def _choice(threshold):
        return {
            "params": {},
            "cv_f1_up_mean": 0.5,
            "cv_f1_up_std": 0.01,
            "cv_precision_up_mean": 0.6,
            "cv_recall_up_mean": 0.4,
            "f1_up_folds": [0.5] * 4,
            "precision_up_folds": [0.6] * 4,
            "recall_up_folds": [0.4] * 4,
            "fold_date_ranges": [{"fold": index} for index in range(1, 5)],
            "decision_threshold": threshold,
            "oof_f1_up": 0.5,
            "oof_precision_up": 0.6,
            "oof_recall_up": 0.4,
            "oof_up_rate": 0.4,
            "oof_predicted_up_ratio": 0.5,
            "threshold_constraint_passed": True,
        }

    def test_pipeline_reuses_selected_cv_runs_without_cv_again(self):
        rows = 8
        train = pd.DataFrame(
            {
                **{column: np.arange(rows, dtype=float) for column in model_tuning.FEATURE_COLUMNS},
                "symbol": ["AAA"] * rows,
                "trading_date": pd.date_range("2025-01-01", periods=rows).strftime("%Y-%m-%d"),
                "label_end_date": pd.date_range("2025-01-02", periods=rows).strftime("%Y-%m-%d"),
                "target": [0, 1] * 4,
            }
        )
        thresholds = {
            "logistic_regression": 0.51,
            "random_forest": 0.57,
            "gradient_boosting": 0.63,
        }
        config = {
            "schema_version": model_tuning.MANUAL_CONFIG_SCHEMA_VERSION,
            "policy_id": model_tuning.EXPERIMENT_POLICY_ID,
            "dataset_fingerprint": "train",
            "content_fingerprint": "content",
            "selected_models": {
                key: self._choice(threshold) for key, threshold in thresholds.items()
            },
        }
        fingerprint = {"hash": "train", "content_hash": "content"}

        with (
            patch.object(model_tuning, "read_manual_config", return_value=config),
            patch.object(model_tuning, "compute_dataset_fingerprint", return_value=fingerprint),
            patch.object(model_tuning, "is_config_complete", return_value=True),
            patch.object(model_tuning, "run_cv_metrics", side_effect=AssertionError("CV reran")),
            patch.object(model_tuning, "write_json"),
            patch.object(model_tuning, "atomic_dataframe_to_csv"),
        ):
            artifacts, _, report = model_tuning.tune_models(train)

        self.assertEqual(report["source"], "selected_tuning_run")
        self.assertEqual(
            {artifacts[model_id]["model_name"]: artifacts[model_id]["decision_threshold"] for model_id in (2, 3, 4)},
            {
                "Logistic Regression": 0.51,
                "Random Forest": 0.57,
                "Gradient Boosting": 0.63,
            },
        )


if __name__ == "__main__":
    unittest.main()
