import json
import unittest
from unittest.mock import patch

import pandas as pd

from config import settings
from services import chatbot_tools


def release_state(*, metadata=None, summary=None, warnings=None):
    metadata = metadata or {
        "model_name": "Random Forest",
        "policy_id": settings.EXPERIMENT_POLICY_ID,
        "content_fingerprint": "release-123",
        "prediction_horizon": 5,
        "up_threshold": 0.01,
        "decision_threshold": 0.49996,
        "feature_order": settings.FEATURE_COLUMNS,
        "training_symbols": ["FPT", "VNM"],
        "training_symbol_count": 2,
    }
    return {
        "status": "current",
        "artifact": dict(metadata),
        "metadata": metadata,
        "summary": summary or {"feature_importance_written": True},
        "scope": {"FPT", "VNM"},
        "scope_verified": True,
        "signature": (),
        "warnings": warnings or [],
    }


def signal(symbol, score, *, date="2026-07-20", stale=False):
    return {
        "symbol": symbol,
        "prediction": "UP" if score >= 0.49996 else "NOT_UP",
        "probability_up": score,
        "decision_threshold": 0.49996,
        "close_at_reference": 70.6789,
        "reference_date": date,
        "return_20d": 0.123456,
        "volatility_20d": 0.020049,
        "volume_ratio_20": 1.256,
        "is_stale": stale,
        "horizon": 5,
    }


class ChatbotToolUpgradeTests(unittest.TestCase):
    def test_six_closed_tools_include_project_topic_enum(self):
        functions = {
            item["function"]["name"]: item["function"]
            for item in chatbot_tools.TOOL_DEFINITIONS
        }

        self.assertEqual(len(functions), 6)
        project = functions["get_project_info"]
        self.assertFalse(project["parameters"]["additionalProperties"])
        self.assertEqual(
            project["parameters"]["properties"]["topic"]["enum"],
            [
                "overview",
                "target",
                "features",
                "data_split",
                "training",
                "metrics",
                "inference",
                "limitations",
            ],
        )

    def test_signal_precision_and_gap_use_raw_scores(self):
        state = release_state()
        rows = [signal("FPT", 0.50004), signal("VNM", 0.49996)]
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service, "predict_symbols", return_value=rows
            ),
        ):
            result = chatbot_tools.get_stock_signals({"symbols": ["FPT", "VNM"]})

        first = result["data"]["signals"][0]
        self.assertEqual(first["up_score_percent"], 50.0)
        self.assertEqual(first["decision_threshold_percent"], 50.0)
        self.assertEqual(first["threshold_gap_percent_points"], 0.01)
        self.assertEqual(first["return_20d_percent"], 12.35)
        self.assertEqual(first["volatility_20d_percent"], 2.0)
        self.assertEqual(first["volume_ratio_20"], 1.26)
        self.assertEqual(
            result["data"]["comparison"]["up_score_gap_percent_points"], 0.01
        )

    def test_different_dates_disable_comparison_and_warn_about_stale_symbol(self):
        state = release_state()
        rows = [
            signal("FPT", 0.7, date="2026-07-20"),
            signal("VNM", 0.6, date="2025-10-08", stale=True),
        ]
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service, "predict_symbols", return_value=rows
            ),
        ):
            result = chatbot_tools.get_stock_signals({"symbols": ["FPT", "VNM"]})

        self.assertIsNone(result["as_of"])
        self.assertIsNone(result["source"]["as_of"])
        self.assertEqual(
            result["data"]["comparison"],
            {"available": False, "reason": "different_reference_dates"},
        )
        self.assertEqual(
            {warning["code"] for warning in result["warnings"]},
            {"stale_symbol_data", "different_reference_dates"},
        )

    def test_ranking_excludes_stale_before_sort_and_limit(self):
        state = release_state()
        rows = [
            signal("FPT", 0.99, date="2025-10-08", stale=True),
            signal("VNM", 0.61),
        ]
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_all_symbols",
                return_value=rows,
            ),
        ):
            result = chatbot_tools.get_ranking(
                {"order": "highest_up_score", "top_n": 1}
            )

        self.assertEqual(
            [row["symbol"] for row in result["data"]["ranking"]], ["VNM"]
        )
        self.assertEqual(result["data"]["excluded_stale_count"], 1)
        self.assertEqual(result["data"]["data_as_of"], "2026-07-20")
        self.assertEqual(result["as_of"], "2026-07-20")

    def test_ranking_returns_domain_error_when_no_fresh_signal_remains(self):
        state = release_state()
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_all_symbols",
                return_value=[signal("FPT", 0.9, stale=True)],
            ),
        ):
            result = chatbot_tools.get_ranking(
                {"order": "highest_up_score", "top_n": 5}
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "no_current_signals")
        self.assertEqual(result["data"]["excluded_stale_count"], 1)

    def test_ranking_discards_nan_and_infinite_scores(self):
        state = release_state()
        rows = [
            signal("FPT", float("nan")),
            signal("VNM", float("inf")),
            {**signal("VNM", 0.61), "symbol": "VNM"},
        ]
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_all_symbols",
                return_value=rows,
            ),
        ):
            result = chatbot_tools.get_ranking(
                {"order": "highest_up_score", "top_n": 5}
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            [row["symbol"] for row in result["data"]["ranking"]], ["VNM"]
        )

    def test_relations_compare_raw_values_before_rounding_for_display(self):
        state = release_state()
        rows = [signal("FPT", 0.500001), signal("VNM", 0.500000)]
        rows[0]["decision_threshold"] = 0.5
        rows[1]["decision_threshold"] = 0.5
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(
                chatbot_tools.prediction_service, "predict_symbols", return_value=rows
            ),
        ):
            result = chatbot_tools.get_stock_signals({"symbols": ["FPT", "VNM"]})

        first = result["data"]["signals"][0]
        comparison = result["data"]["comparison"]
        self.assertEqual(first["threshold_gap_percent_points"], 0.0)
        self.assertEqual(first["threshold_relation"], "above")
        self.assertFalse(comparison["same_up_score"])
        self.assertEqual(comparison["higher_up_score_symbol"], "FPT")

    def test_model_metrics_are_percent_fields_with_split_dates(self):
        metadata = release_state()["metadata"] | {
            "train_end_date": "2025-07-10",
            "validation_end_date": "2026-04-10",
            "test_end_date": "2026-07-13",
            "baseline_passed": False,
            "validation_selection_metrics": {"accuracy": 0.56789, "f1_up": 0.47734},
            "final_test_metrics": {"accuracy": 0.57661, "f1_up": 0.375381},
            "final_test_baselines": [
                {"model_name": "Always UP", "f1_up": 0.383895}
            ],
        }
        state = release_state(metadata=metadata)
        with patch.object(chatbot_tools, "get_release_state", return_value=state):
            result = chatbot_tools.get_model_info({})

        data = result["data"]
        self.assertEqual(
            data["validation_selection_metrics_percent"],
            {"accuracy": 56.79, "f1_up": 47.73},
        )
        self.assertEqual(
            data["final_test_metrics_percent"],
            {"accuracy": 57.66, "f1_up": 37.54},
        )
        self.assertNotIn("validation_selection_metrics", data)
        self.assertNotIn("final_test_metrics", data)
        self.assertEqual(data["train_end_date"], "2025-07-10")
        self.assertEqual(data["validation_end_date"], "2026-04-10")
        self.assertEqual(data["test_end_date"], "2026-07-13")
        self.assertEqual(
            data["baseline_warning"],
            "Final Model chưa vượt baseline Always UP trên TEST: "
            "F1 UP 37,54% so với 38,39%.",
        )

    def test_release_baseline_warning_is_readable(self):
        metadata = release_state()["metadata"] | {
            "baseline_passed": False,
            "final_test_metrics": {"f1_up": 0.375381},
            "final_test_baselines": [
                {"model_name": "Always UP", "f1_up": 0.383895}
            ],
        }
        artifact = dict(metadata)
        summary = {
            "policy_id": settings.EXPERIMENT_POLICY_ID,
            "content_fingerprint": "release-123",
            "selection_report": {"selected_model_name": "Random Forest"},
        }
        with (
            patch.object(chatbot_tools.joblib, "load", return_value=artifact),
            patch.object(chatbot_tools, "_read_json", side_effect=[metadata, summary]),
            patch.object(chatbot_tools, "_release_signature", return_value=()),
            patch.object(
                chatbot_tools.experiment_state,
                "is_pipeline_running",
                return_value=False,
            ),
        ):
            state = chatbot_tools.get_release_state()

        warning = next(
            item for item in state["warnings"] if item["code"] == "baseline_failed"
        )
        self.assertEqual(
            warning["message"],
            "Final Model chưa vượt baseline Always UP trên TEST: "
            "F1 UP 37,54% so với 38,39%.",
        )

    def test_feature_importance_is_rounded_to_four_decimals(self):
        state = release_state()
        frame = pd.DataFrame(
            {"feature": ["return_1d"], "importance": [0.123456789]}
        )
        with (
            patch.object(chatbot_tools, "get_release_state", return_value=state),
            patch.object(chatbot_tools, "_release_changed", return_value=False),
            patch.object(chatbot_tools.pd, "read_csv", return_value=frame),
        ):
            result = chatbot_tools.get_feature_info({"top_n": 1})

        self.assertEqual(
            result["data"]["global_importance"],
            [{"feature": "return_1d", "importance": 0.1235}],
        )

    def test_project_topics_are_whitelisted_static_contracts(self):
        state = release_state()
        topics = [
            "overview",
            "target",
            "features",
            "data_split",
            "training",
            "metrics",
            "inference",
            "limitations",
        ]
        with patch.object(chatbot_tools, "get_release_state", return_value=state):
            results = [chatbot_tools.get_project_info({"topic": topic}) for topic in topics]

        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual(
            [result["data"]["topic"] for result in results], topics
        )
        target = results[1]
        self.assertEqual(
            target["data"]["facts"]["prediction_horizon_sessions"],
            settings.PREDICTION_HORIZON,
        )
        self.assertEqual(
            target["data"]["facts"]["up_threshold_percent"],
            round(settings.UP_THRESHOLD * 100, 2),
        )
        self.assertEqual(
            target["source"], {"kind": "project_contract", "topic": "target"}
        )
        serialized = json.dumps(results, ensure_ascii=False).lower()
        self.assertNotIn("model_name", serialized)
        self.assertNotIn(".md", serialized)
        self.assertNotIn("raw_data_path", serialized)
        feature_facts = results[2]["data"]["facts"]
        self.assertEqual(feature_facts["moving_average_windows_sessions"], [5, 20, 50])
        self.assertEqual(feature_facts["return_windows_sessions"], [1, 3, 5, 10, 20])
        self.assertEqual(feature_facts["rsi_period_sessions"], 14)

    def test_project_info_is_independent_from_serving_release(self):
        with patch.object(chatbot_tools, "get_release_state") as get_release:
            result = chatbot_tools.get_project_info({"topic": "overview"})

        get_release.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["release"],
            {"status": None, "policy_id": None, "content_fingerprint": None},
        )
        self.assertEqual(result["warnings"], [])

    def test_project_tool_rejects_unknown_topic_and_extra_fields(self):
        state = release_state()
        with patch.object(chatbot_tools, "get_release_state", return_value=state):
            unknown = chatbot_tools.get_project_info({"topic": "../../secret"})
            extra = chatbot_tools.get_project_info(
                {"topic": "overview", "path": "models/final_model.pkl"}
            )

        self.assertEqual(unknown["error"]["code"], "invalid_arguments")
        self.assertEqual(extra["error"]["code"], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
