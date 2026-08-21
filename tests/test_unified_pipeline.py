"""Contracts for the rolling, version-neutral pipeline."""

import json
import math
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config import settings
from scripts import fetch_hose_data
from services import experiment_state, model_evaluation, pipeline_utils, prediction_service


class NeutralContractTests(unittest.TestCase):
    def test_policy_and_protocol_use_stable_ids(self):
        self.assertEqual(settings.EXPERIMENT_POLICY_ID, "rolling_recent_cv_oof_threshold")
        self.assertEqual(settings.DATA_PROTOCOL_ID, "exact_market_t5")
        self.assertFalse(hasattr(settings, "EXPERIMENT_POLICY_VERSION"))
        self.assertFalse(hasattr(settings, "DATA_PROTOCOL_VERSION"))


class RollingEvaluationRegistryTests(unittest.TestCase):
    def test_registry_keeps_multiple_snapshots_and_blocks_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_registry.json"
            first = {"experiment_fingerprint": "snapshot-a", "status": "published"}
            second = {"experiment_fingerprint": "snapshot-b", "status": "evaluated"}

            experiment_state.write_evaluation_entry(first, path=path)
            experiment_state.write_evaluation_entry(second, path=path)

            registry = experiment_state.read_evaluation_registry(path=path)
            self.assertEqual(set(registry), {"snapshot-a", "snapshot-b"})
            self.assertTrue(experiment_state.has_evaluated_snapshot("snapshot-a", path=path))
            self.assertTrue(experiment_state.has_evaluated_snapshot("snapshot-b", path=path))
            self.assertFalse(experiment_state.has_evaluated_snapshot("snapshot-c", path=path))

    def test_pipeline_lock_can_only_be_released_by_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.lock"
            with patch.object(experiment_state, "PIPELINE_LOCK_PATH", path):
                owner = experiment_state.acquire_pipeline_lock()
                self.assertIsInstance(owner, str)
                self.assertIsNone(experiment_state.acquire_pipeline_lock())
                self.assertFalse(experiment_state.release_pipeline_lock("not-owner"))
                self.assertTrue(path.exists())
                self.assertTrue(experiment_state.release_pipeline_lock(owner))
                self.assertFalse(path.exists())


class ValidationBaselineTests(unittest.TestCase):
    def test_candidate_below_baseline_is_selected_with_warning(self):
        comparison = pd.DataFrame(
            [
                {"model_id": 2, "model_name": "Logistic Regression", "row_type": "candidate", "f1_up": 0.40, "recall_up": 0.50},
                {"model_id": 3, "model_name": "Random Forest", "row_type": "candidate", "f1_up": 0.39, "recall_up": 0.60},
                {"model_id": 4, "model_name": "Gradient Boosting", "row_type": "candidate", "f1_up": 0.38, "recall_up": 0.70},
                {"model_id": -1, "model_name": "Always UP", "row_type": "baseline", "f1_up": 0.41, "recall_up": 1.0},
            ]
        )
        artifacts = {
            model_id: {"model_id": model_id, "model_name": name, "cv_f1_up": 0.3}
            for model_id, name in settings.MODEL_DEFINITIONS.items()
            if model_id in (2, 3, 4)
        }
        for metric in model_evaluation.METRIC_COLUMNS:
            if metric not in comparison:
                comparison[metric] = 0.0

        _, selected, report = model_evaluation.select_final_model(comparison, artifacts)

        self.assertEqual(selected["model_id"], 2)
        self.assertFalse(report["validation_baseline_passed"])
        self.assertIn("Always UP", report["validation_baseline_warning"])


class FetchWindowTests(unittest.TestCase):
    def test_empty_provider_data_is_not_a_failed_symbol(self):
        wrapped_error = RuntimeError("RetryError")
        wrapped_error.__cause__ = ValueError(
            "Dữ liệu trống cho mã BCG với interval 1D."
        )

        with patch("vnstock.api.quote.Quote") as quote:
            quote.return_value.history.side_effect = wrapped_error
            result = fetch_hose_data.fetch_symbol_history(
                "BCG", "2025-10-09", "2026-07-31"
            )

        self.assertTrue(result.empty)

    def test_real_provider_error_still_fails_after_retries(self):
        with (
            patch("vnstock.api.quote.Quote") as quote,
            patch.object(fetch_hose_data, "FETCH_MAX_RETRIES", 2),
            patch.object(fetch_hose_data.time, "sleep"),
        ):
            quote.return_value.history.side_effect = ValueError("Sai schema")
            with self.assertRaisesRegex(RuntimeError, "Sai schema"):
                fetch_hose_data.fetch_symbol_history(
                    "AAA", "2026-07-31", "2026-07-31"
                )

    def test_rate_limit_system_exit_waits_then_retries(self):
        row = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "time": ["2026-07-31"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
                "volume": [100],
            }
        )
        with (
            patch("vnstock.api.quote.Quote") as quote,
            patch.object(fetch_hose_data, "FETCH_MAX_RETRIES", 2),
            patch.object(fetch_hose_data.time, "sleep") as sleep,
        ):
            quote.return_value.history.side_effect = [
                SystemExit("Rate limit exceeded. 20/20. Process terminated."),
                row,
            ]
            result = fetch_hose_data.fetch_symbol_history(
                "AAA", "2026-07-31", "2026-07-31"
            )

        self.assertEqual(len(result), 1)
        sleep.assert_called_once_with(65)

    def test_unrelated_system_exit_is_not_swallowed(self):
        with patch("vnstock.api.quote.Quote") as quote:
            quote.return_value.history.side_effect = SystemExit("manual stop")
            with self.assertRaisesRegex(SystemExit, "manual stop"):
                fetch_hose_data.fetch_symbol_history(
                    "AAA", "2026-07-31", "2026-07-31"
                )

    def test_fetch_window_is_resolved_per_symbol(self):
        existing = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB"],
                "trading_date": ["2026-07-19", "2026-07-20", "2026-07-10"],
            }
        )

        windows = fetch_hose_data.resolve_fetch_windows(existing, end="2026-07-21")

        self.assertEqual(windows["AAA"], ("2026-07-21", "2026-07-21", True))
        self.assertEqual(windows["BBB"], ("2026-07-11", "2026-07-21", True))


class OutputContractTests(unittest.TestCase):
    def test_json_writer_normalizes_non_finite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            pipeline_utils.write_json(path, {"nan": math.nan, "nested": [math.inf]})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload, {"nan": None, "nested": [None]})

    def test_forecast_uses_requested_horizon(self):
        sessions = prediction_service._forecast_sessions(date(2026, 7, 17), horizon=3)
        self.assertEqual(len(sessions), 3)
        self.assertEqual(sessions[-1]["step"], 3)

    def test_atomic_csv_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            path.write_text("old\n", encoding="utf-8")
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    pipeline_utils.atomic_dataframe_to_csv(
                        pd.DataFrame({"value": [1]}), path
                    )

            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_model_release_rolls_back_if_metadata_promotion_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "final_model.pkl"
            metadata_path = Path(directory) / "model_metadata.json"
            pipeline_utils.atomic_joblib_dump({"model": "old"}, model_path)
            pipeline_utils.write_json(metadata_path, {"model": "old"})
            real_replace = os.replace
            failed = False

            def fail_metadata_once(source, target):
                nonlocal failed
                if Path(target) == metadata_path and not failed:
                    failed = True
                    raise OSError("metadata promotion failed")
                return real_replace(source, target)

            with patch.object(
                pipeline_utils.os, "replace", side_effect=fail_metadata_once
            ):
                with self.assertRaisesRegex(OSError, "metadata promotion failed"):
                    pipeline_utils.atomic_model_release(
                        {"model": "new"},
                        {"model": "new"},
                        model_path=model_path,
                        metadata_path=metadata_path,
                    )

            import joblib

            self.assertEqual(joblib.load(model_path), {"model": "old"})
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                {"model": "old"},
            )


if __name__ == "__main__":
    unittest.main()
