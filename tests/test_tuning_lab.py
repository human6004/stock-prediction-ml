"""Contracts for Tuning Lab history, selection and job control."""

from __future__ import annotations

import json
import math
import threading
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config.settings import (
    CV_GAP_SESSIONS,
    CV_N_SPLITS,
    CV_START_DATE,
    EXPERIMENT_POLICY_ID,
)
import app as web_app
from scripts import run_pipeline as pipeline_script
from services import tuning_lab
from services import experiment_state
from services.experiment_state import (
    REQUIRED_MODEL_KEYS,
    get_best_runs,
    get_tuning_progress,
    is_config_complete,
    mark_best,
    save_selected_model,
)


FINGERPRINT = "dataset-current"


def row(
    run_id: str,
    model_key: str = "logistic_regression",
    *,
    mean: float = 0.5,
    std: float = 0.02,
    params: dict | None = None,
    fingerprint: str = FINGERPRINT,
    policy: str = EXPERIMENT_POLICY_ID,
    status: str = "ok",
) -> dict:
    params = params or {"C": 0.1, "solver": "lbfgs"}
    return {
        "run_id": run_id,
        "model_key": model_key,
        "params": params,
        "params_json": experiment_state.canonical_params_json(params),
        "cv_f1_up_mean": mean,
        "cv_f1_up_std": std,
        "folds": [mean] * CV_N_SPLITS,
        "precision_folds": [mean] * CV_N_SPLITS,
        "recall_folds": [mean] * CV_N_SPLITS,
        "fold_ranges": [{"fold": index} for index in range(1, CV_N_SPLITS + 1)],
        "decision_threshold": 0.6,
        "oof_f1_up": mean,
        "oof_precision_up": 0.5,
        "oof_recall_up": mean,
        "oof_up_rate": 0.4,
        "oof_predicted_up_ratio": 0.5,
        "threshold_constraint_passed": True,
        "dataset_fingerprint": fingerprint,
        "content_fingerprint": "full",
        "policy_id": policy,
        "status": status,
        "is_best": False,
    }


class ManualParamTests(unittest.TestCase):
    def test_validation_accepts_in_bounds_config_outside_catalog(self):
        params = tuning_lab.validate_params(
            "logistic_regression", {"C": "9", "solver": "lbfgs"}
        )

        self.assertEqual(params, {"C": 9.0, "solver": "lbfgs"})

    def test_validation_still_rejects_out_of_bounds_config(self):
        with self.assertRaises(ValueError):
            tuning_lab.validate_params(
                "logistic_regression", {"C": "0", "solver": "lbfgs"}
            )


class BestRunTests(unittest.TestCase):
    def test_best_is_deterministic_by_mean_std_then_run_id_ascending(self):
        rows = [
            row("run-c", mean=0.7, std=0.03),
            row("run-b", mean=0.7, std=0.01),
            row("run-a", mean=0.7, std=0.01),
        ]

        marked = mark_best(rows)

        self.assertEqual([item["run_id"] for item in marked if item["is_best"]], ["run-a"])
        self.assertEqual(
            get_best_runs(marked, FINGERPRINT)["logistic_regression"]["run_id"],
            "run-a",
        )

    def test_legacy_wrong_policy_and_wrong_fingerprint_never_enter_ranking(self):
        rows = [
            row("valid", mean=0.4),
            row("legacy", mean=0.99, policy="legacy"),
            row("wrong-data", mean=0.98, fingerprint="other"),
        ]

        best = get_best_runs(rows, FINGERPRINT)

        self.assertEqual(best["logistic_regression"]["run_id"], "valid")

    def test_user_can_select_a_valid_run_even_when_it_is_not_best(self):
        rows = [
            row("chosen", params={"C": 9.0, "solver": "lbfgs"}, mean=0.4),
            row("best", params={"C": 10.0, "solver": "lbfgs"}, mean=0.8),
        ]
        fingerprint = {"hash": FINGERPRINT, "parts": {}, "content_hash": "full"}

        with (
            patch.object(experiment_state, "read_history", return_value=rows),
            patch.object(experiment_state, "read_manual_config", return_value={}),
            patch.object(
                experiment_state, "_write_manual_config", side_effect=lambda config: config
            ),
        ):
            config = save_selected_model(
                "logistic_regression",
                "chosen",
                {"C": 9.0, "solver": "lbfgs"},
                0.4,
                fingerprint,
            )

        selected = config["selected_models"]["logistic_regression"]
        self.assertEqual(selected["run_id"], "chosen")
        self.assertEqual(selected["selection_method"], "manual")


class FingerprintTests(unittest.TestCase):
    def test_csv_round_trip_preserves_fingerprint(self):
        frame = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "trading_date": ["2025-06-20"],
                "label_end_date": ["2025-06-27"],
                "month": pd.Series([6], dtype="int32"),
                "return_1d": [math.nextafter(0.1, 1.0)],
                "target": [1],
            }
        )
        round_tripped = pd.read_csv(StringIO(frame.to_csv(index=False)))

        normalized = pipeline_script._normalize_fingerprint_dataset(frame)
        before = experiment_state.compute_dataset_fingerprint(normalized)
        after = experiment_state.compute_dataset_fingerprint(round_tripped)

        self.assertEqual(before["content_hash"], after["content_hash"])
        self.assertEqual(before["hash"], after["hash"])

    def test_content_change_changes_hash_but_row_order_does_not(self):
        frame = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "trading_date": ["2026-01-02", "2026-01-02"],
                "target": [0, 1],
            }
        )
        reordered = frame.iloc[::-1].reset_index(drop=True)
        changed = frame.copy()
        changed.loc[0, "target"] = 1

        first = experiment_state.compute_dataset_fingerprint(frame)
        second = experiment_state.compute_dataset_fingerprint(reordered)
        third = experiment_state.compute_dataset_fingerprint(changed)

        self.assertEqual(first["hash"], second["hash"])
        self.assertNotEqual(first["content_hash"], third["content_hash"])
        self.assertNotEqual(first["hash"], third["hash"])

    def test_new_trading_session_shifts_rolling_window_and_reopens(self):
        """Rolling walk-forward: thêm phiên giao dịch mới hơn làm mốc dịch lên,
        nên fingerprint đổi và khóa TEST cũ tự mở lại — đây là hành vi mong muốn
        (thay cho ngữ nghĩa cũ 'inference row không được mở khóa').

        Đồng thời kiểm tra biên: fetch lại đúng dữ liệu cũ (không có phiên mới)
        giữ nguyên fingerprint nên khóa vẫn còn — không mở vô điều kiện.
        """
        # Dataset đủ dài để vào chế độ rolling (span > VALIDATION+TEST window).
        base = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "trading_date": "2024-06-03",
                    "label_end_date": "2024-06-10",
                    "target": 0,
                },
                {
                    "symbol": "AAA",
                    "trading_date": "2025-08-01",
                    "label_end_date": "2025-08-08",
                    "target": 1,
                },
                {
                    "symbol": "AAA",
                    "trading_date": "2026-03-02",
                    "label_end_date": "2026-03-09",
                    "target": 0,
                },
                {
                    "symbol": "AAA",
                    "trading_date": "2026-05-04",
                    "label_end_date": "2026-05-11",
                    "target": 1,
                },
            ]
        )
        newer_session = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "trading_date": "2026-07-06",
                    "label_end_date": "2026-07-13",
                    "target": 1,
                }
            ]
        )
        extended = pd.concat([base, newer_session], ignore_index=True)

        experiment_before = experiment_state.compute_experiment_fingerprint(base)
        experiment_after = experiment_state.compute_experiment_fingerprint(extended)
        experiment_refetch = experiment_state.compute_experiment_fingerprint(
            base.copy()
        )

        # Phiên mới hơn → mốc rolling dịch → fingerprint đổi → khóa cũ mở lại.
        self.assertNotEqual(experiment_before["hash"], experiment_after["hash"])
        # Không có phiên mới → fingerprint ổn định → khóa vẫn giữ.
        self.assertEqual(experiment_before["hash"], experiment_refetch["hash"])


class LegacyHistoryTests(unittest.TestCase):
    def test_rows_without_policy_are_loaded_as_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text(
                "run_id,model_key,params_json,cv_f1_up_mean,status\n"
                'old,logistic_regression,"{}",0.9,ok\n',
                encoding="utf-8",
            )
            with patch.object(experiment_state, "TUNING_HISTORY_PATH", path):
                rows = experiment_state.read_history()

        self.assertEqual(rows[0]["policy_id"], "legacy")


class ProgressTests(unittest.TestCase):
    def test_duplicate_params_count_once_without_catalog_gate(self):
        params_by_model = {
            "logistic_regression": {"C": 9.0, "solver": "lbfgs"},
            "random_forest": {
                "n_estimators": 77,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "max_features": 0.4,
            },
            "gradient_boosting": {
                "n_estimators": 88,
                "learning_rate": 0.07,
                "max_depth": 2,
                "subsample": 0.9,
            },
        }
        rows = [
            row(model_key, model_key, params=params)
            for model_key, params in params_by_model.items()
        ]
        rows.append(row("duplicate", params=params_by_model["logistic_regression"]))

        progress = get_tuning_progress(rows, FINGERPRINT)

        self.assertTrue(progress["complete"])
        for model_key in REQUIRED_MODEL_KEYS:
            self.assertEqual(progress["models"][model_key]["valid_config_count"], 1)

    def test_error_and_non_current_rows_do_not_count(self):
        wrong_threshold = row(
            "wrong-threshold", params={"C": 0.01, "solver": "lbfgs"}
        )
        wrong_threshold["threshold_constraint_passed"] = False
        missing_folds = row(
            "missing-folds", params={"C": 0.001, "solver": "lbfgs"}
        )
        missing_folds["fold_ranges"] = []
        rows = [
            row("valid"),
            row("error", params={"C": 0.01, "solver": "lbfgs"}, status="error"),
            row("legacy", params={"C": 0.001, "solver": "lbfgs"}, policy="legacy"),
            wrong_threshold,
            missing_folds,
        ]

        progress = get_tuning_progress(rows, FINGERPRINT)

        self.assertEqual(progress["models"]["logistic_regression"]["valid_config_count"], 1)
        self.assertFalse(progress["complete"])

    def test_pipeline_gate_requires_explicit_selection_for_all_models(self):
        params_by_model = {
            "logistic_regression": {"C": 9.0, "solver": "lbfgs"},
            "random_forest": {
                "n_estimators": 77,
                "max_depth": 6,
                "min_samples_leaf": 3,
                "max_features": 0.4,
            },
            "gradient_boosting": {
                "n_estimators": 88,
                "learning_rate": 0.07,
                "max_depth": 2,
                "subsample": 0.9,
            },
        }
        rows = [
            row(model_key, model_key, params=params)
            for model_key, params in params_by_model.items()
        ]
        config = {
            "schema_version": experiment_state.MANUAL_CONFIG_SCHEMA_VERSION,
            "policy_id": EXPERIMENT_POLICY_ID,
            "dataset_fingerprint": FINGERPRINT,
            "cv_settings": {
                "n_splits": CV_N_SPLITS,
                "gap_sessions": CV_GAP_SESSIONS,
                "start_date": CV_START_DATE,
            },
            "selected_models": {
                model_key: {
                    "run_id": model_key,
                    "params": params,
                    "decision_threshold": 0.6,
                    "selection_method": "manual",
                }
                for model_key, params in params_by_model.items()
            },
        }

        self.assertTrue(is_config_complete(config, FINGERPRINT, rows))
        del config["selected_models"]["gradient_boosting"]
        self.assertFalse(is_config_complete(config, FINGERPRINT, rows))


class BackgroundJobTests(unittest.TestCase):
    def tearDown(self):
        tuning_lab._reset_tuning_job_for_tests()

    def test_only_one_background_job_can_run(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_evaluate(*_args, **_kwargs):
            entered.set()
            release.wait(timeout=2)
            return {"run_id": "run-a"}

        params = {"C": 0.00001, "solver": "lbfgs"}
        with (
            patch.object(tuning_lab, "load_train", return_value=object()),
            patch.object(tuning_lab, "evaluate_single_config", side_effect=blocking_evaluate),
        ):
            first = tuning_lab.start_tuning_job("logistic_regression", params)
            self.assertTrue(entered.wait(timeout=1))
            second = tuning_lab.start_tuning_job("logistic_regression", params)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            release.set()
            tuning_lab.wait_for_tuning_job(timeout=2)

        self.assertEqual(tuning_lab.get_tuning_job_state()["status"], "completed")


class EvaluateHistoryTests(unittest.TestCase):
    def test_success_records_all_fold_metrics_ranges_policy_and_fingerprint(self):
        train = pd.DataFrame({"sentinel": [1]})
        cv = {
            "f1_up_mean": 0.5,
            "f1_up_std": 0.01,
            "f1_up_folds": [0.5] * CV_N_SPLITS,
            "precision_up_mean": 0.6,
            "recall_up_mean": 0.4,
            "precision_up_folds": [0.6] * CV_N_SPLITS,
            "recall_up_folds": [0.4] * CV_N_SPLITS,
            "fold_date_ranges": [
                {"fold": index} for index in range(1, CV_N_SPLITS + 1)
            ],
            "decision_threshold": 0.6,
            "oof_f1_up": 0.5,
            "oof_precision_up": 0.6,
            "oof_recall_up": 0.4,
            "oof_up_rate": 0.4,
            "oof_predicted_up_ratio": 0.5,
            "threshold_constraint_passed": True,
        }
        fingerprint = {"hash": FINGERPRINT, "parts": {}, "content_hash": "full"}
        params = {"C": 0.00001, "solver": "lbfgs"}
        with (
            patch.object(tuning_lab, "build_estimator", return_value=object()),
            patch.object(tuning_lab, "run_cv_metrics", return_value=cv) as run_cv,
            patch.object(tuning_lab, "append_history") as append,
            patch.object(experiment_state, "_write_manual_config") as write_config,
        ):
            tuning_lab.evaluate_single_config(
                "logistic_regression", params, train, fingerprint=fingerprint
            )

        run_cv.assert_called_once()
        self.assertIs(run_cv.call_args.args[1], train)
        record = append.call_args.args[0]
        self.assertEqual(record["policy_id"], EXPERIMENT_POLICY_ID)
        self.assertEqual(record["dataset_fingerprint"], FINGERPRINT)
        self.assertEqual(record["content_fingerprint"], "full")
        self.assertEqual(record["cv_precision_up_std"], 0.0)
        self.assertEqual(record["cv_recall_up_std"], 0.0)
        self.assertEqual(len(json.loads(record["cv_fold_ranges_json"])), CV_N_SPLITS)
        self.assertEqual(len(json.loads(record["cv_precision_up_folds_json"])), CV_N_SPLITS)
        self.assertEqual(len(json.loads(record["cv_recall_up_folds_json"])), CV_N_SPLITS)
        write_config.assert_not_called()


class BackgroundRouteTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_evaluate_route_starts_job_and_redirects_immediately(self):
        params = {"C": 0.00001, "solver": "lbfgs"}
        with (
            patch.object(web_app, "validate_params", return_value=params),
            patch.object(web_app, "start_tuning_job", return_value="job-1") as start,
        ):
            response = self.client.post(
                "/tuning/evaluate",
                data={"model_key": "logistic_regression", **params},
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("job-1", response.headers["Location"])
        start.assert_called_once_with("logistic_regression", params)

    def test_evaluate_route_rejects_second_running_job(self):
        params = {"C": 0.00001, "solver": "lbfgs"}
        with (
            patch.object(web_app, "validate_params", return_value=params),
            patch.object(web_app, "start_tuning_job", return_value=None),
            patch.object(web_app, "_tuning_context", return_value={}),
            patch.object(web_app, "render_template", return_value="busy"),
        ):
            response = self.client.post(
                "/tuning/evaluate",
                data={"model_key": "logistic_regression", **params},
            )

        self.assertEqual(response.status_code, 409)

    def test_evaluate_route_does_not_start_during_data_refresh(self):
        with (
            patch.object(web_app, "is_pipeline_running", return_value=False),
            patch.object(web_app, "is_fetch_running", return_value=True),
            patch.object(web_app, "_tuning_context", return_value={}),
            patch.object(web_app, "render_template", return_value="busy"),
            patch.object(web_app, "start_tuning_job") as start,
        ):
            response = self.client.post(
                "/tuning/evaluate",
                data={"model_key": "logistic_regression", "C": "0.00001", "solver": "lbfgs"},
            )

        self.assertEqual(response.status_code, 409)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
