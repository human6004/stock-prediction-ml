"""Contract tests for batch prediction, forecast charts, and symbol comparison."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import app as web_app
from config.settings import EXPERIMENT_POLICY_ID, FEATURE_COLUMNS
from services import prediction_service


ROOT_DIR = Path(__file__).resolve().parents[1]


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 7, 18)


class FakeProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probabilities=(0.8, 0.4)):
        self.probabilities = probabilities
        self.predict_proba = Mock(side_effect=self._predict_proba)

    def _predict_proba(self, rows):
        return [[1 - value, value] for value in self.probabilities[: len(rows)]]


def make_clean_data() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-10", periods=250).strftime("%Y-%m-%d").tolist()
    rows = []
    for symbol, final_close in (("FPT", 70.6), ("VNM", 63.0)):
        for offset, trading_date in enumerate(dates):
            close = final_close - (len(dates) - 1 - offset) * 0.1
            rows.append(
                {
                    "symbol": symbol,
                    "trading_date": trading_date,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000 + offset,
                }
            )
    for offset, trading_date in enumerate(dates[-6:]):
        close = 20.0 + offset
        rows.append(
            {
                "symbol": "CRV",
                "trading_date": trading_date,
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000 + offset,
            }
        )
    return pd.DataFrame(rows)


def make_features(symbols=("FPT", "VNM")) -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(symbols):
        row = {column: float(index + 1) for column in FEATURE_COLUMNS}
        row.update(
            {
                "symbol": symbol,
                "trading_date": "2026-07-10",
                "close": 70.6 if symbol == "FPT" else 63.0,
                "return_20d": 0.12 if symbol == "FPT" else -0.03,
                "volatility_20d": 0.02 if symbol == "FPT" else 0.04,
                "volume_ratio_20": 1.25 if symbol == "FPT" else 0.85,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_artifact(model) -> dict:
    return {
        "model": model,
        "model_name": "Logistic Regression",
        "feature_columns": FEATURE_COLUMNS,
        "prediction_horizon": 5,
        "up_threshold": 0.01,
        "decision_threshold": 0.6,
    }


def make_metadata() -> dict:
    return {
        "model_name": "Logistic Regression",
        "feature_order": FEATURE_COLUMNS,
        "prediction_horizon": 5,
        "up_threshold": 0.01,
        "decision_threshold": 0.6,
    }


def make_result(
    symbol="FPT",
    probability=0.8,
    reference_date="2026-07-10",
    stale=False,
    elapsed=True,
):
    history_dates = [
        "2026-07-03",
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        reference_date,
    ]
    forecast_dates = [
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
    ]
    prediction = "UP" if probability is not None and probability >= 0.6 else "NOT_UP"
    return {
        "symbol": symbol,
        "latest_date": reference_date,
        "reference_date": reference_date,
        "close_at_reference": 70.6,
        "prediction": prediction,
        "probability_up": probability,
        "decision_threshold": 0.6,
        "target_close": 70.6 * 1.01,
        "price_history": [
            {"date": trading_date, "close": 65.6 + index}
            for index, trading_date in enumerate(history_dates)
        ],
        "forecast_sessions": [
            {"step": index, "date": trading_date, "estimated": True}
            for index, trading_date in enumerate(forecast_dates, start=1)
        ],
        "forecast_start_date": forecast_dates[0],
        "forecast_end_date": forecast_dates[-1],
        "is_stale": stale,
        "forecast_window_elapsed": elapsed,
        "return_20d": 0.12,
        "volatility_20d": 0.02,
        "volume_ratio_20": 1.25,
        "model_name": "Logistic Regression",
        "horizon": 5,
        "horizon_sessions": 5,
        "threshold_percent": 1,
        "summary_sentence": "Du bao thu nghiem.",
        "prediction_label_vi": "Tăng (UP)" if prediction == "UP" else "Không đủ điều kiện tăng (NOT_UP)",
        "prediction_short_vi": "Không đủ điều kiện tăng hơn 1% trong 5 phiên tiếp theo."
        if prediction == "NOT_UP"
        else "Khả năng giá tăng hơn 1% trong 5 phiên tiếp theo.",
    }


class PredictionServiceTests(unittest.TestCase):
    def test_metadata_mismatch_falls_back_to_the_loaded_artifact(self):
        artifact = {
            **make_artifact(FakeProbabilityModel()),
            "policy_id": EXPERIMENT_POLICY_ID,
            "content_fingerprint": "new-content",
            "baseline_passed": False,
            "baseline_warning": "new warning",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(
                json.dumps(
                    {
                        "model_name": "Old Model",
                        "policy_id": "legacy",
                        "content_fingerprint": "old-content",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(prediction_service, "MODEL_METADATA_PATH", path):
                metadata = prediction_service.load_metadata(artifact)

        self.assertEqual(metadata["model_name"], "Logistic Regression")
        self.assertEqual(metadata["content_fingerprint"], "new-content")
        self.assertFalse(metadata["baseline_passed"])

    def _run_batch(self, symbols=(" fpt ", "vnm", "FPT")):
        clean = make_clean_data()
        features = make_features()
        model = FakeProbabilityModel()
        artifact = make_artifact(model)
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=clean) as load_data,
            patch.object(prediction_service, "build_features", return_value=(features, {})) as build,
            patch.object(prediction_service, "load_model_artifact", return_value=artifact) as load_model,
            patch.object(prediction_service, "load_metadata", return_value=make_metadata()) as load_metadata,
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            results = prediction_service.predict_symbols(list(symbols))
        return results, model, artifact, load_data, build, load_model, load_metadata

    def test_batch_normalizes_deduplicates_and_loads_each_resource_once(self):
        results, model, artifact, load_data, build, load_model, load_metadata = self._run_batch()

        self.assertEqual([result["symbol"] for result in results], ["FPT", "VNM"])
        load_data.assert_called_once_with()
        load_model.assert_called_once_with()
        load_metadata.assert_called_once_with(artifact)
        build.assert_called_once()
        self.assertEqual(set(build.call_args.args[0]["symbol"]), {"FPT", "VNM"})
        model.predict_proba.assert_called_once()
        self.assertEqual(len(model.predict_proba.call_args.args[0]), 2)

    def test_batch_accepts_five_unique_symbols_after_deduplication(self):
        symbols = ("FPT", "VNM", "HPG", "MWG", "SSI")
        clean = make_clean_data()
        template = clean[clean["symbol"] == "FPT"]
        clean = pd.concat(
            [
                clean,
                *[
                    template.assign(symbol=symbol)
                    for symbol in symbols
                    if symbol not in {"FPT", "VNM"}
                ],
            ],
            ignore_index=True,
        )
        model = FakeProbabilityModel((0.8, 0.7, 0.6, 0.5, 0.4))
        artifact = make_artifact(model)
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=clean),
            patch.object(
                prediction_service,
                "build_features",
                return_value=(make_features(symbols), {}),
            ),
            patch.object(prediction_service, "load_model_artifact", return_value=artifact),
            patch.object(prediction_service, "load_metadata", return_value=make_metadata()),
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            results = prediction_service.predict_symbols(
                ["FPT", "fpt", "VNM", "HPG", "MWG", "SSI"]
            )

        self.assertEqual([result["symbol"] for result in results], list(symbols))
        model.predict_proba.assert_called_once()
        self.assertEqual(len(model.predict_proba.call_args.args[0]), 5)

    def test_inference_uses_threshold_from_artifact_metadata(self):
        features = make_features()
        model = FakeProbabilityModel((0.5, 0.4999))
        artifact = make_artifact(model)
        self.assertEqual(artifact["decision_threshold"], 0.6)

        with (
            patch.object(prediction_service, "_load_clean_data", return_value=make_clean_data()),
            patch.object(prediction_service, "build_features", return_value=(features, {})),
            patch.object(prediction_service, "load_model_artifact", return_value=artifact),
            patch.object(prediction_service, "load_metadata", return_value=make_metadata()),
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            results = prediction_service.predict_symbols(["FPT", "VNM"])

        self.assertEqual([row["prediction"] for row in results], ["NOT_UP", "NOT_UP"])
        self.assertEqual([row["decision_threshold"] for row in results], [0.6, 0.6])

    def test_inference_exposes_baseline_warning_from_metadata(self):
        metadata = {
            **make_metadata(),
            "policy_id": EXPERIMENT_POLICY_ID,
            "baseline_passed": False,
            "baseline_warning": "Final Model chưa vượt baseline always-UP trên TEST.",
        }
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=make_clean_data()),
            patch.object(prediction_service, "build_features", return_value=(make_features(), {})),
            patch.object(
                prediction_service,
                "load_model_artifact",
                return_value=make_artifact(FakeProbabilityModel()),
            ),
            patch.object(prediction_service, "load_metadata", return_value=metadata),
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            result = prediction_service.predict_symbols(["FPT"])[0]

        self.assertEqual(result["policy_id"], EXPERIMENT_POLICY_ID)
        self.assertIs(result["baseline_passed"], False)
        self.assertIn("always-UP", result["baseline_warning"])

    def test_result_schema_contains_real_history_and_estimated_sessions_only(self):
        results, *_ = self._run_batch(("FPT",))
        result = results[0]
        required = {
            "symbol",
            "reference_date",
            "close_at_reference",
            "prediction",
            "probability_up",
            "decision_threshold",
            "target_close",
            "price_history",
            "forecast_sessions",
            "forecast_start_date",
            "forecast_end_date",
            "is_stale",
            "forecast_window_elapsed",
            "return_20d",
            "volatility_20d",
            "volume_ratio_20",
            "latest_date",
            "model_name",
            "horizon",
            "horizon_sessions",
            "threshold_percent",
            "summary_sentence",
            "prediction_label_vi",
            "prediction_short_vi",
        }
        self.assertTrue(required.issubset(result))
        self.assertEqual(len(result["price_history"]), 6)
        self.assertEqual(
            [point["date"] for point in result["price_history"]],
            sorted(point["date"] for point in result["price_history"]),
        )
        self.assertEqual(len(result["forecast_sessions"]), 5)
        for session in result["forecast_sessions"]:
            self.assertEqual(set(session), {"step", "date", "estimated"})
            self.assertIs(session["estimated"], True)
        self.assertAlmostEqual(result["target_close"], result["close_at_reference"] * 1.01)
        self.assertEqual(result["return_20d"], 0.12)
        self.assertEqual(result["volatility_20d"], 0.02)
        self.assertEqual(result["volume_ratio_20"], 1.25)

    def test_forecast_skips_weekend_and_sets_elapsed_flag(self):
        result = self._run_batch(("FPT",))[0][0]
        self.assertEqual(
            [session["date"] for session in result["forecast_sessions"]],
            [
                "2026-07-13",
                "2026-07-14",
                "2026-07-15",
                "2026-07-16",
                "2026-07-17",
            ],
        )
        self.assertEqual(result["forecast_start_date"], "2026-07-13")
        self.assertEqual(result["forecast_end_date"], "2026-07-17")
        self.assertIs(result["forecast_window_elapsed"], True)

    def test_symbol_older_than_global_data_max_is_stale(self):
        features = make_features(("VNM",))
        features.loc[0, "trading_date"] = "2026-07-09"
        features.loc[0, "close"] = 62.9
        model = FakeProbabilityModel((0.4,))
        artifact = make_artifact(model)
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=make_clean_data()),
            patch.object(prediction_service, "build_features", return_value=(features, {})),
            patch.object(prediction_service, "load_model_artifact", return_value=artifact),
            patch.object(prediction_service, "load_metadata", return_value=make_metadata()),
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            result = prediction_service.predict_symbols(["VNM"])[0]

        self.assertEqual(result["reference_date"], "2026-07-09")
        self.assertIs(result["is_stale"], True)

    def test_validation_rejects_empty_items_and_more_than_five_symbols(self):
        invalid_inputs = (
            [],
            [""],
            ["FPT", " "],
            ["FPT", "VNM", "HPG", "MWG", "SSI", "VCB"],
        )
        for symbols in invalid_inputs:
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                prediction_service.predict_symbols(symbols)

    def test_unknown_and_insufficient_history_are_distinct_errors(self):
        clean = make_clean_data()
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=clean),
            patch.object(prediction_service, "build_features") as build,
        ):
            with self.assertRaises(ValueError) as unknown:
                prediction_service.predict_symbols(["AAA"])
            build.assert_not_called()

        with (
            patch.object(prediction_service, "_load_clean_data", return_value=clean),
            patch.object(
                prediction_service,
                "build_features",
                return_value=(pd.DataFrame(columns=["symbol", "trading_date"]), {}),
            ),
        ):
            with self.assertRaises(ValueError) as insufficient:
                prediction_service.predict_symbols(["CRV"])

        self.assertIn("AAA", str(unknown.exception))
        self.assertIn("CRV", str(insufficient.exception))
        self.assertNotEqual(str(unknown.exception), str(insufficient.exception))

    def test_predict_symbol_is_a_batch_wrapper(self):
        expected = make_result()
        with patch.object(prediction_service, "predict_symbols", return_value=[expected]) as batch:
            actual = prediction_service.predict_symbol(" fpt ")
        self.assertEqual(actual, expected)
        batch.assert_called_once_with([" fpt "])

    def test_fallback_predict_runs_once_for_the_whole_batch(self):
        model = Mock(spec=["predict"])
        model.predict.return_value = [1, 0]
        artifact = make_artifact(model)
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=make_clean_data()),
            patch.object(
                prediction_service,
                "build_features",
                return_value=(make_features(), {}),
            ),
            patch.object(prediction_service, "load_model_artifact", return_value=artifact),
            patch.object(prediction_service, "load_metadata", return_value=make_metadata()),
            patch.object(prediction_service, "date", FrozenDate, create=True),
        ):
            results = prediction_service.predict_symbols(["FPT", "VNM"])

        model.predict.assert_called_once()
        self.assertEqual([result["prediction"] for result in results], ["UP", "NOT_UP"])
        self.assertEqual([result["probability_up"] for result in results], [None, None])
        self.assertNotIn("giảm", results[1]["prediction_short_vi"].lower())

    def test_cached_all_symbol_inference_batches_and_returns_copies(self):
        model = FakeProbabilityModel((0.8, 0.4))
        artifact = make_artifact(model)
        metadata = {
            **make_metadata(),
            "policy_id": "legacy",
            "baseline_passed": False,
            "baseline_warning": None,
        }
        prediction_service._predict_all_symbols_cached.cache_clear()
        with (
            patch.object(prediction_service, "_load_clean_data", return_value=make_clean_data()),
            patch.object(
                prediction_service,
                "build_features",
                return_value=(make_features(), {}),
            ),
            patch.object(prediction_service, "load_model_artifact", return_value=artifact),
            patch.object(prediction_service, "load_metadata", return_value=metadata),
            patch.object(prediction_service, "_dataset_signature", return_value=("data",)),
            patch.object(prediction_service, "_model_signature", return_value=("model",)),
        ):
            first = prediction_service.predict_all_symbols()
            first[0]["symbol"] = "CHANGED"
            second = prediction_service.predict_all_symbols()

        self.assertEqual([row["symbol"] for row in second], ["FPT", "VNM"])
        self.assertEqual([row["prediction"] for row in second], ["UP", "NOT_UP"])
        self.assertTrue(all(row["baseline_warning"] for row in second))
        model.predict_proba.assert_called_once()
        prediction_service._predict_all_symbols_cached.cache_clear()

    def test_data_and_model_signatures_cover_clean_raw_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.csv"
            raw = root / "raw.csv"
            model = root / "model.pkl"
            clean.write_text("clean", encoding="utf-8")
            raw.write_text("raw", encoding="utf-8")
            model.write_text("model", encoding="utf-8")
            with (
                patch.object(prediction_service, "CLEANED_DATA_PATH", clean),
                patch.object(prediction_service, "RAW_DATA_PATH", raw),
                patch.object(prediction_service, "FINAL_MODEL_PATH", model),
            ):
                self.assertEqual(prediction_service._dataset_signature()[0], str(clean))
                self.assertNotEqual(prediction_service._model_signature(), (0, 0))
                clean.unlink()
                self.assertEqual(prediction_service._dataset_signature()[0], str(raw))
                raw.unlink()
                model.unlink()
                self.assertEqual(prediction_service._dataset_signature(), ("missing", 0, 0))
                self.assertEqual(prediction_service._model_signature(), (0, 0))


class PredictionRouteTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()
        self.meta_patcher = patch.object(
            web_app,
            "get_dataset_meta",
            return_value={
                "dataset_max_date": "2026-07-10",
                "symbol_count": 3,
                "symbols": ("FPT", "VNM", "TCD"),
            },
        )
        self.selected_patcher = patch.object(
            web_app,
            "load_selected_model_from_report",
            return_value="Logistic Regression",
        )
        self.meta_patcher.start()
        self.selected_patcher.start()
        self.addCleanup(self.meta_patcher.stop)
        self.addCleanup(self.selected_patcher.stop)

    def test_legacy_artifact_warning_uses_current_policy_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "model_metadata.json"
            metadata_path.write_text(
                json.dumps({"policy_id": "legacy"}), encoding="utf-8"
            )
            with patch.object(web_app, "MODEL_METADATA_PATH", metadata_path):
                sections = web_app.load_evaluation_sections()

        warning = sections["legacy_warning"]
        self.assertIn("policy hiện hành", warning)
        self.assertNotIn("version", warning.lower())

    def test_failed_final_refresh_step_never_reports_full_completion(self):
        progress = web_app._infer_refresh_progress("[3/3] Build features\nerror", False)

        self.assertEqual(progress["status"], "failed")
        self.assertLess(progress["percent"], 100)

    def test_single_prediction_renders_chart_payload(self):
        result = make_result()
        with (
            patch.object(web_app, "predict_symbol", return_value=result) as predict,
        ):
            response = self.client.post("/predict", data={"symbol": "FPT"})

        self.assertEqual(response.status_code, 200)
        predict.assert_called_once_with("FPT")
        html = response.get_data(as_text=True)
        self.assertIn("FPT", html)
        self.assertIn("10/07/2026", html)
        self.assertIn("13/07/2026", html)
        self.assertIn("Ngưỡng UP, không phải giá dự báo", html)
        self.assertIn("Điểm UP của model", html)
        self.assertIn("Ngưỡng quyết định", html)
        self.assertIn("Khoảng dự báo", html)
        self.assertIn("Mốc đánh giá", html)
        self.assertIn('id="prediction-chart"', html)
        self.assertIn("aria-label=", html)
        self.assertIn("chart.umd.min.js", html)
        self.assertIn('"estimated": true', html)
        self.assertIn('"close": 70.6', html)

    def test_prediction_page_hides_baseline_warning(self):
        result = make_result()
        result.update(
            policy_id=EXPERIMENT_POLICY_ID,
            baseline_passed=False,
            baseline_warning="Final Model chưa vượt baseline always-UP trên TEST.",
        )
        with (
            patch.object(web_app, "predict_symbol", return_value=result),
        ):
            response = self.client.post("/predict", data={"symbol": "FPT"})

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("chưa vượt baseline always-UP", response.get_data(as_text=True))

    def test_screener_hides_baseline_warning(self):
        result = make_result()
        result["baseline_warning"] = "Không hiển thị cảnh báo baseline."
        with patch.object(web_app, "predict_all_symbols", return_value=[result]):
            response = self.client.get("/screener")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Không hiển thị cảnh báo baseline", response.get_data(as_text=True))

    def test_chart_template_leaves_future_prices_null_and_does_not_span_gaps(self):
        source = (ROOT_DIR / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertRegex(source, r"fill\s*\(\s*null\s*\)")
        self.assertRegex(source, r"spanGaps\s*:\s*false")
        self.assertIn("target_close", source)
        self.assertIn("tojson", source)

    def test_evaluation_renders_one_candidate_table_and_test_summary(self):
        def row(model_name, *, row_type="candidate", selected=False):
            return {
                "model_name": model_name,
                "row_type": row_type,
                "accuracy": 0.59,
                "precision_up": 0.41,
                "recall_up": 0.47,
                "f1_up": 0.44,
                "precision_not_up": 0.70,
                "recall_not_up": 0.65,
                "f1_not_up": 0.67,
                "cv_f1_up": 0.46,
                "decision_threshold": 0.6,
                "status": "Đang dùng" if selected else "",
            }

        sections = {
            "is_current_policy": True,
            "legacy_warning": None,
            "baseline_warning": "Không hiển thị cảnh báo baseline.",
            "validation": [
                row("Logistic Regression"),
                row("Random Forest"),
                row("Gradient Boosting", selected=True),
                row("Always UP", row_type="baseline"),
            ],
            "test": [
                row("Gradient Boosting", row_type="final", selected=True),
                row("Always UP", row_type="baseline"),
            ],
            "legacy": [],
        }
        with patch.object(web_app, "load_evaluation_sections", return_value=sections):
            response = self.client.get("/evaluation")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        # Đếm "<table" chứ không phải "<table>": thẻ nay mang thêm thuộc tính
        # data-ui="sortable" cho phần sắp xếp cột. Ý vẫn là đúng một bảng.
        self.assertEqual(html.count("<table"), 1)
        self.assertIn("Bảng so sánh model", html)
        self.assertIn("CV F1_UP", html)
        self.assertIn("Kết quả Final Model trên TEST", html)
        for model_name in ("Logistic Regression", "Random Forest", "Gradient Boosting"):
            self.assertIn(model_name, html)
        self.assertNotIn("Always UP", html)
        self.assertNotIn("Không hiển thị cảnh báo baseline", html)

    def test_evaluation_legacy_table_hides_baselines(self):
        def row(model_name, row_type):
            return {
                "model_name": model_name,
                "row_type": row_type,
                "accuracy": 0.5,
                "precision_up": 0.4,
                "recall_up": 0.4,
                "f1_up": 0.4,
                "precision_not_up": 0.6,
                "recall_not_up": 0.6,
                "f1_not_up": 0.6,
                "status": "",
            }

        sections = {
            "is_current_policy": False,
            "legacy_warning": None,
            "baseline_warning": None,
            "validation": [],
            "test": [],
            "legacy": [
                row("Random Forest", "candidate"),
                row("Always UP", "baseline"),
                row("Always NOT_UP", "baseline"),
            ],
        }
        with patch.object(web_app, "load_evaluation_sections", return_value=sections):
            response = self.client.get("/evaluation")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Random Forest", html)
        self.assertNotIn("Always UP", html)
        self.assertNotIn("Always NOT_UP", html)

    def test_compare_renders_exact_metrics_highlight_and_warnings(self):
        first = make_result("FPT", 0.8, stale=True)
        second = make_result("VNM", 0.4, reference_date="2026-07-09")
        first["baseline_warning"] = "Không hiển thị cảnh báo baseline."
        second["forecast_start_date"] = "2026-07-10"
        second["forecast_end_date"] = "2026-07-16"
        with (
            patch.object(web_app, "predict_symbols", return_value=[first, second], create=True) as predict,
        ):
            response = self.client.post(
                "/compare", data={"symbol_a": "FPT", "symbol_b": "VNM"}
            )

        self.assertEqual(response.status_code, 200)
        predict.assert_called_once_with(["FPT", "VNM"])
        html = response.get_data(as_text=True)
        for label in (
            "Ngày tham chiếu",
            "Giá đóng cửa",
            "Điểm UP",
            "Kết quả",
            "Return 20 phiên",
            "Volatility 20 phiên",
            "Volume / AVG20",
        ):
            self.assertIn(label, html)
        body_match = re.search(r"<tbody>(.*?)</tbody>", html, flags=re.DOTALL)
        self.assertIsNotNone(body_match)
        self.assertEqual(len(re.findall(r"<tr(?:\s|>)", body_match.group(1))), 7)
        self.assertEqual(html.count("Tín hiệu UP cao hơn theo mô hình"), 1)
        self.assertIn("chưa thể so sánh trực tiếp", html)
        self.assertIn("Dữ liệu đã cũ; hãy làm mới trước khi dùng kết quả.", html)
        self.assertNotIn("Không hiển thị cảnh báo baseline", html)
        for forbidden in ("Nên mua", "Mã tốt nhất", "điểm tổng hợp"):
            self.assertNotIn(forbidden, html)

    def test_equal_probabilities_do_not_highlight_either_symbol(self):
        results = [make_result("FPT", 0.5), make_result("VNM", 0.5)]
        with (
            patch.object(web_app, "predict_symbols", return_value=results, create=True),
        ):
            response = self.client.post(
                "/compare", data={"symbol_a": "FPT", "symbol_b": "VNM"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Tín hiệu UP cao hơn theo mô hình", response.get_data(as_text=True))

    def test_compare_rejects_duplicate_symbols_without_calling_service(self):
        with patch.object(web_app, "predict_symbols", create=True) as predict:
            response = self.client.post(
                "/compare", data={"symbol_a": " fpt ", "symbol_b": "FPT"}
            )
        self.assertEqual(response.status_code, 400)
        predict.assert_not_called()
        html = response.get_data(as_text=True)
        self.assertIn('value=" fpt "', html)
        self.assertIn('value="FPT"', html)

    def test_compare_service_error_is_inline_and_preserves_inputs(self):
        with patch.object(
            web_app,
            "predict_symbols",
            side_effect=ValueError("Khong tim thay ma AAA."),
            create=True,
        ):
            response = self.client.post(
                "/compare", data={"symbol_a": "AAA", "symbol_b": "FPT"}
            )
        self.assertEqual(response.status_code, 400)
        html = response.get_data(as_text=True)
        self.assertIn("AAA", html)
        self.assertIn('value="AAA"', html)
        self.assertIn('value="FPT"', html)

    def test_page_routes_remain_available(self):
        with (
            patch.object(web_app, "render_template", return_value="ok"),
            patch.object(
                web_app,
                "load_evaluation_sections",
                return_value={"baseline_warning": None},
            ),
            patch.object(web_app, "_tuning_context", return_value={}),
        ):
            statuses = {
                path: self.client.get(path).status_code
                for path in ("/", "/evaluation", "/tuning", "/compare")
            }
        self.assertEqual(statuses, {path: 200 for path in statuses})

    def test_local_chart_asset_keeps_version_and_license(self):
        asset = ROOT_DIR / "static" / "vendor" / "chart.umd.min.js"
        self.assertTrue(asset.is_file())
        header = asset.read_text(encoding="utf-8")[:500]
        self.assertIn("Chart.js v4.4.9", header)
        self.assertIn("MIT", header)


if __name__ == "__main__":
    unittest.main()
