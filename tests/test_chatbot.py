"""Behavior tests for the action-decision chatbot."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import app as web_app
from services import chatbot_service, chatbot_tools


ROOT = Path(__file__).resolve().parents[1]


def provider_response(content, *, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def fake_client(*responses, side_effect=None):
    create = Mock(side_effect=side_effect or list(responses))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def decision_json(action, arguments):
    return json.dumps(
        {"action": action, "arguments": arguments},
        ensure_ascii=False,
    )


def domain_result(data=None, **extra):
    return {
        "data": data or {},
        "sources": extra.get("sources", []),
        "warnings": extra.get("warnings", []),
        "data_as_of": extra.get("data_as_of"),
        "model_trained_through": extra.get("model_trained_through"),
        "error": extra.get("error"),
    }


class DecisionTests(unittest.TestCase):
    def test_parses_and_normalizes_all_five_actions(self):
        cases = (
            (
                "STOCK_SIGNAL",
                {"symbols": [" fpt ", "VNM", " hpg ", "MWG", "SSI"]},
                {"symbols": ["FPT", "VNM", "HPG", "MWG", "SSI"]},
            ),
            (
                "STOCK_RANKING",
                {"order": "lowest", "top_n": 3},
                {"order": "lowest", "top_n": 3},
            ),
            ("PROJECT_INFO", {"topic": "evaluation"}, {"topic": "evaluation"}),
            ("OUT_OF_SCOPE", {"reason": "news"}, {"reason": "news"}),
            ("GENERAL_CHAT", {"kind": "clarify_symbol"}, {"kind": "clarify_symbol"}),
        )
        for action, arguments, normalized in cases:
            with self.subTest(action=action):
                client = fake_client(provider_response(decision_json(action, arguments)))
                result = chatbot_service._decide(
                    "message", [], client, monotonic=Mock(side_effect=[0, 0, 0])
                )
                self.assertEqual(set(result), {"action", "arguments"})
                self.assertEqual(result["arguments"], normalized)
                client.chat.completions.create.assert_called_once()

    def test_accepts_fixed_general_kinds_and_seven_project_topics(self):
        for kind in (
            "greeting",
            "thanks",
            "capabilities",
            "clarify_symbol",
            "clarify_request",
        ):
            with self.subTest(kind=kind):
                decision = chatbot_service._validate_decision(
                    {"action": "GENERAL_CHAT", "arguments": {"kind": kind}}
                )
                self.assertEqual(decision["arguments"], {"kind": kind})

        for topic in (
            "overview",
            "dataset",
            "features",
            "model",
            "evaluation",
            "inference",
            "limitations",
        ):
            with self.subTest(topic=topic):
                decision = chatbot_service._validate_decision(
                    {"action": "PROJECT_INFO", "arguments": {"topic": topic}}
                )
                self.assertEqual(decision["arguments"], {"topic": topic})

        decision = chatbot_service._validate_decision(
            {
                "action": "STOCK_SIGNAL",
                "arguments": {"symbols": [" fpt ", "VNM", "FPT"]},
            }
        )
        self.assertEqual(decision["arguments"], {"symbols": ["FPT", "VNM"]})

    def test_rejects_malformed_extra_unknown_wrong_types_and_removed_values(self):
        invalid = (
            "not json",
            "```json\n{}\n```",
            decision_json("UNKNOWN", {}),
            json.dumps(
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {"kind": "greeting"},
                    "extra": True,
                }
            ),
            json.dumps(
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {"kind": "greeting"},
                    "direct_answer": "schema cũ",
                }
            ),
            decision_json("GENERAL_CHAT", {"kind": "unknown"}),
            decision_json("STOCK_RANKING", {"order": "highest", "top_n": True}),
            decision_json(
                "STOCK_SIGNAL",
                {"symbols": ["FPT", "VNM", "HPG", "MWG", "SSI", "VCB"]},
            ),
            decision_json(
                "STOCK_SIGNAL", {"symbols": ["FPT"], "focus": "comparison"}
            ),
            decision_json("PROJECT_INFO", {"topic": "method"}),
            decision_json("OUT_OF_SCOPE", {"reason": "unsupported_symbol"}),
        )
        for content in invalid:
            with self.subTest(content=content):
                client = fake_client(provider_response(content))
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._decide(
                        "message", [], client, monotonic=Mock(side_effect=[0, 0, 0])
                    )
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("provider_protocol_error", 502),
                )
                client.chat.completions.create.assert_called_once()

    def test_rejects_tool_call_protocol(self):
        client = fake_client(
            provider_response(
                decision_json("GENERAL_CHAT", {"kind": "greeting"}),
                tool_calls=[{"id": "unexpected"}],
            )
        )
        with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
            chatbot_service._decide(
                "message", [], client, monotonic=Mock(side_effect=[0, 0, 0])
            )
        self.assertEqual(
            (raised.exception.code, raised.exception.status),
            ("provider_protocol_error", 502),
        )

    def test_provider_request_has_one_call_six_history_messages_and_no_tools(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
            for index in range(8)
        ]
        client = fake_client(
            provider_response(decision_json("GENERAL_CHAT", {"kind": "greeting"}))
        )
        chatbot_service.chat("Chào", history, client=client)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][1:-1], history[-6:])
        for key in ("tools", "tool_choice", "response_format"):
            self.assertNotIn(key, request)
        self.assertNotIn("CONTEXT_JSON", request["messages"][0]["content"])

    def test_monotonic_deadline_blocks_before_and_after_provider(self):
        response = provider_response(
            decision_json("GENERAL_CHAT", {"kind": "greeting"})
        )
        cases = (
            (
                [
                    0,
                    chatbot_service.TOTAL_DEADLINE_SECONDS,
                    chatbot_service.TOTAL_DEADLINE_SECONDS,
                ],
                False,
            ),
            ([0, 0, chatbot_service.TOTAL_DEADLINE_SECONDS], True),
        )
        for clock_values, provider_called in cases:
            with self.subTest(provider_called=provider_called):
                client = fake_client(response)
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._decide(
                        "message", [], client, monotonic=Mock(side_effect=clock_values)
                    )
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("provider_timeout", 504),
                )
                self.assertEqual(
                    client.chat.completions.create.call_count,
                    int(provider_called),
                )


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "metadata": {
                "model_name": "Random Forest",
                "prediction_horizon": 5,
                "up_threshold": 0.01,
                "decision_threshold": 0.49,
                "train_through_date": "2026-04-10",
                "final_test_metrics": {"f1_up": 0.4812},
            },
            "summary": {},
            "scope": {"FPT", "VNM", "MWG", "HPG", "SSI"},
            "warnings": [],
            "model_trained_through": "2026-04-10",
        }

    def test_symbol_scope_is_all_or_nothing_before_inference(self):
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict,
        ):
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL", {"symbols": ["FPT", "AAA", "VNM", "BBB"]}
            )
        predict.assert_not_called()
        self.assertEqual(result["data"]["unsupported_symbols"], ["AAA", "BBB"])
        self.assertNotIn("signals", result["data"])
        warning = next(
            item for item in result["warnings"] if item["code"] == "symbol_out_of_scope"
        )
        self.assertIn("AAA", warning["message"])
        self.assertIn("BBB", warning["message"])

    def test_multi_signal_same_date_sorts_score_desc_then_symbol(self):
        rows = [
            {
                "symbol": "VNM",
                "probability_up": 0.5,
                "decision_threshold": 0.49,
                "reference_date": "2026-07-20",
                "prediction": "UP",
                "is_stale": False,
            },
            {
                "symbol": "FPT",
                "probability_up": 0.5,
                "decision_threshold": 0.49,
                "reference_date": "2026-07-20",
                "prediction": "UP",
                "is_stale": False,
            },
            {
                "symbol": "MWG",
                "probability_up": 0.7,
                "decision_threshold": 0.49,
                "reference_date": "2026-07-20",
                "prediction": "UP",
                "is_stale": False,
            },
        ]
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service, "predict_symbols", return_value=rows
            ) as predict,
        ):
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL",
                {"symbols": ["VNM", "FPT", "MWG"]},
            )
        predict.assert_called_once_with(["VNM", "FPT", "MWG"])
        self.assertEqual(
            [row["symbol"] for row in result["data"]["signals"]],
            ["MWG", "FPT", "VNM"],
        )
        self.assertEqual(result["data_as_of"], "2026-07-20")

    def test_multi_signal_different_dates_preserves_input_order_and_warns(self):
        rows = [
            {"symbol": "VNM", "probability_up": 0.3, "reference_date": "2026-07-19"},
            {"symbol": "FPT", "probability_up": 0.7, "reference_date": "2026-07-20"},
            {"symbol": "MWG", "probability_up": 0.9, "reference_date": "2026-07-18"},
        ]
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(chatbot_tools.prediction_service, "predict_symbols", return_value=rows),
        ):
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL",
                {"symbols": ["VNM", "FPT", "MWG"]},
            )
        self.assertEqual(
            [row["symbol"] for row in result["data"]["signals"]],
            ["VNM", "FPT", "MWG"],
        )
        self.assertNotIn("comparison", result["data"])
        self.assertIsNone(result["data_as_of"])
        self.assertTrue(any(w["code"] == "different_reference_dates" for w in result["warnings"]))

    def test_ranking_filters_scope_nonfinite_stale_then_sorts_and_limits(self):
        rows = [
            {"symbol": "FPT", "probability_up": 0.60, "reference_date": "2026-07-20", "prediction": "UP"},
            {"symbol": "VNM", "probability_up": 0.20, "reference_date": "2026-07-19", "is_stale": True},
            {"symbol": "ABC", "probability_up": 0.99, "reference_date": "2026-07-20"},
            {"symbol": "MWG", "probability_up": float("nan"), "reference_date": "2026-07-20"},
        ]
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service, "predict_all_symbols", return_value=rows
            ) as predict,
        ):
            result = chatbot_tools.execute_action(
                "STOCK_RANKING", {"order": "highest", "top_n": 2}
            )
        predict.assert_called_once_with()
        self.assertEqual([row["symbol"] for row in result["data"]["ranking"]], ["FPT"])
        self.assertEqual(result["data"]["excluded_stale_count"], 1)

    def test_ranking_lowest_sorts_ascending_then_symbol(self):
        rows = [
            {"symbol": "VNM", "probability_up": 0.2, "reference_date": "2026-07-20"},
            {"symbol": "FPT", "probability_up": 0.2, "reference_date": "2026-07-20"},
            {"symbol": "MWG", "probability_up": 0.7, "reference_date": "2026-07-20"},
        ]
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service, "predict_all_symbols", return_value=rows
            ),
        ):
            result = chatbot_tools.execute_action(
                "STOCK_RANKING", {"order": "lowest", "top_n": 2}
            )
        self.assertEqual(
            [row["symbol"] for row in result["data"]["ranking"]],
            ["FPT", "VNM"],
        )

    def test_runtime_readiness_legacy_scope_fallback_is_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eligible_path = root / "eligible.csv"
            pd.DataFrame({"symbol": ["fpt", "VNM"]}).to_csv(eligible_path, index=False)
            with (
                patch.object(chatbot_tools, "ELIGIBLE_SYMBOLS_PATH", eligible_path),
                patch.object(chatbot_tools.experiment_state, "is_pipeline_running", return_value=False),
                patch.object(
                    chatbot_tools.prediction_service,
                    "load_model_artifact",
                    return_value={"model": object()},
                ),
                patch.object(
                    chatbot_tools.prediction_service,
                    "load_metadata",
                    return_value={"model_name": "Random Forest", "policy_id": "legacy"},
                ),
            ):
                state = chatbot_tools._load_runtime_state()
        self.assertEqual(state["scope"], {"FPT", "VNM"})
        self.assertTrue(any(w["code"] == "symbol_scope_unverified" for w in state["warnings"]))

    def test_runtime_state_prefers_verified_metadata_scope_and_reports_warnings(self):
        metadata = {
            "model_name": "Random Forest",
            "training_symbols": [" fpt ", "VNM"],
            "training_symbol_count": 2,
            "policy_id": "legacy",
            "baseline_passed": False,
            "baseline_warning": None,
            "train_through_date": "2026-04-10",
        }
        with (
            patch.object(chatbot_tools.experiment_state, "is_pipeline_running", return_value=False),
            patch.object(
                chatbot_tools.prediction_service,
                "load_model_artifact",
                return_value={"model": object()},
            ),
            patch.object(
                chatbot_tools.prediction_service,
                "load_metadata",
                return_value=metadata,
            ),
            patch.object(chatbot_tools, "_read_json", return_value={}),
        ):
            state = chatbot_tools._load_runtime_state()

        self.assertEqual(state["scope"], {"FPT", "VNM"})
        self.assertEqual(
            {warning["code"] for warning in state["warnings"]},
            {"legacy_policy", "baseline_failed"},
        )
        for invalid in (
            {"training_symbols": []},
            {"training_symbols": ["FPT", "fpt"]},
            {"training_symbols": ["FPT"], "training_symbol_count": 2},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                chatbot_tools._load_scope(invalid)

    def test_data_action_failures_and_pipeline_write_guard_are_stable(self):
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_symbols",
                side_effect=RuntimeError("private detail"),
            ),
        ):
            signal = chatbot_tools.execute_action(
                "STOCK_SIGNAL", {"symbols": ["FPT"]}
            )
        self.assertEqual(signal["error"]["code"], "prediction_unavailable")
        self.assertNotIn("private detail", signal["error"]["message"])

        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_all_symbols",
                side_effect=RuntimeError("private detail"),
            ),
        ):
            ranking = chatbot_tools.execute_action(
                "STOCK_RANKING", {"order": "highest", "top_n": 5}
            )
        self.assertEqual(ranking["error"]["code"], "prediction_unavailable")

        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(
                chatbot_tools.prediction_service,
                "predict_all_symbols",
                return_value=[
                    {
                        "symbol": "FPT",
                        "probability_up": 0.7,
                        "reference_date": "2026-07-19",
                        "is_stale": True,
                    }
                ],
            ),
        ):
            no_current = chatbot_tools.execute_action(
                "STOCK_RANKING", {"order": "highest", "top_n": 5}
            )
        self.assertEqual(no_current["error"]["code"], "no_current_signals")

        with patch.object(
            chatbot_tools.experiment_state, "is_pipeline_running", return_value=True
        ):
            for topic in ("dataset", "features"):
                result = chatbot_tools.execute_action(
                    "PROJECT_INFO", {"topic": topic}
                )
                self.assertEqual(result["error"]["code"], "report_unavailable")

    def test_static_project_topics_do_not_require_model_readiness(self):
        with (
            patch.object(
                chatbot_tools.experiment_state, "is_pipeline_running", return_value=True
            ),
            patch.object(
                chatbot_tools.prediction_service,
                "load_model_artifact",
                side_effect=AssertionError("static topic must not load model"),
            ) as load_model,
        ):
            results = {
                topic: chatbot_tools.execute_action("PROJECT_INFO", {"topic": topic})
                for topic in ("overview", "limitations")
            }
        load_model.assert_not_called()
        self.assertTrue(all(result["error"] is None for result in results.values()))
        self.assertTrue(
            all(result["data"]["topic"] == topic for topic, result in results.items())
        )

    def test_model_project_topics_require_verified_metadata_but_not_symbol_scope(self):
        metadata = {
            **self.state["metadata"],
            "policy_id": chatbot_tools.EXPERIMENT_POLICY_ID,
        }
        with (
            patch.object(
                chatbot_tools.experiment_state, "is_pipeline_running", return_value=False
            ),
            patch.object(
                chatbot_tools.prediction_service,
                "load_model_artifact",
                return_value={"model": object()},
            ),
            patch.object(
                chatbot_tools.prediction_service,
                "load_metadata",
                return_value=metadata,
            ),
            patch.object(
                chatbot_tools, "_load_scope", side_effect=AssertionError("scope not needed")
            ) as load_scope,
            patch.object(chatbot_tools, "_read_json", return_value={}),
        ):
            results = {
                topic: chatbot_tools.execute_action("PROJECT_INFO", {"topic": topic})
                for topic in ("model", "evaluation", "inference")
            }

        load_scope.assert_not_called()
        self.assertTrue(all(result["error"] is None for result in results.values()))

    def test_seven_project_topics_keep_distinct_meanings_and_real_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.csv"
            importance = root / "importance.csv"
            pd.DataFrame(
                {
                    "symbol": ["FPT", "VNM"],
                    "trading_date": ["2020-01-01", "2026-07-20"],
                }
            ).to_csv(clean, index=False)
            pd.DataFrame(
                {"feature": ["return_1d", "rsi_14"], "importance": [0.3, 0.2]}
            ).to_csv(importance, index=False)
            with (
                patch.object(chatbot_tools, "CLEANED_DATA_PATH", clean),
                patch.object(chatbot_tools, "FEATURE_IMPORTANCE_PATH", importance),
                patch.object(
                    chatbot_tools,
                    "_load_runtime_state",
                    return_value={
                        **self.state,
                        "metadata": {
                            **self.state["metadata"],
                            "final_test_metrics": {
                                "f1_up": 0.4812,
                                "missing_metric": None,
                            },
                        },
                    },
                ),
            ):
                results = {
                    topic: chatbot_tools.execute_action("PROJECT_INFO", {"topic": topic})
                    for topic in (
                        "overview",
                        "dataset",
                        "features",
                        "model",
                        "evaluation",
                        "inference",
                        "limitations",
                    )
                }
        self.assertTrue(all(result["error"] is None for result in results.values()))
        self.assertTrue(
            all(result["data"]["topic"] == topic for topic, result in results.items())
        )
        self.assertEqual(results["dataset"]["data_as_of"], "2026-07-20")
        self.assertIsNone(results["dataset"]["model_trained_through"])
        for topic in ("model", "evaluation", "inference"):
            self.assertEqual(results[topic]["model_trained_through"], "2026-04-10")
        self.assertEqual(results["features"]["data"]["scope"], "global_model_importance")
        evaluation = json.dumps(results["evaluation"]["data"], ensure_ascii=False)
        self.assertIn("48.12", evaluation)
        self.assertNotIn("missing_metric", evaluation)
        inference = json.dumps(results["inference"]["data"], ensure_ascii=False)
        for term in ("predict_proba", "decision_threshold", "UP", "NOT_UP"):
            self.assertIn(term, inference)

        with self.assertRaises(ValueError):
            chatbot_tools.execute_action("PROJECT_INFO", {"topic": "method"})

    def test_unknown_action_never_loads_runtime_or_calls_ml(self):
        with (
            patch.object(chatbot_tools, "_load_runtime_state") as load,
            patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict,
            self.assertRaises(ValueError),
        ):
            chatbot_tools.execute_action("GENERAL_CHAT", {})
        load.assert_not_called()
        predict.assert_not_called()


class FormatterAndFlowTests(unittest.TestCase):
    def test_project_formatter_uses_only_available_contract_fields(self):
        overview = chatbot_service._format_project_info(
            {
                "title": "HOSE offline",
                "facts": {
                    "serving_mode": "offline_published_artifacts",
                    "pipeline_steps": ["clean", "train", "predict"],
                },
            },
            "overview",
        )
        model = chatbot_service._format_project_info(
            {
                "model_name": "Random Forest",
                "target": "UP sau 5 phiên",
                "prediction_horizon": 5,
                "train_through_date": "2026-04-10",
                "selection_metric": "f1",
                "candidate_models": ["Logistic Regression", "Random Forest"],
                "best_params": {"n_estimators": 200},
            },
            "model",
        )
        dataset = chatbot_service._format_project_info(
            {
                "date_min": "2020-01-01",
                "date_max": "2026-07-20",
                "clean_row_count": 100,
                "clean_symbol_count": 5,
                "training_symbol_count": 4,
                "feature_count": 20,
                "split_report": {
                    "train_date_min": "2020-01-01",
                    "train_date_max": "2024-12-31",
                    "train_rows": 70,
                    "validation_date_min": "2025-01-01",
                    "validation_date_max": "2025-12-31",
                    "validation_rows": 20,
                    "test_date_min": "2026-01-01",
                    "test_date_max": "2026-07-20",
                    "test_rows": 10,
                },
            },
            "dataset",
        )
        features = chatbot_service._format_project_info(
            {
                "feature_order": ["return_1d", "rsi_14"],
                "global_importance": [
                    {"feature": "return_1d", "importance": 0.3},
                    {"invalid": True},
                ],
            },
            "features",
        )
        evaluation = chatbot_service._format_project_info(
            {
                "validation_selection_metrics_percent": {"f1_up": 48.12},
                "final_test_metrics_percent": {"accuracy": 57.66, "missing": None},
                "final_test_baselines": [
                    {
                        "model_name": "Always UP",
                        "metrics_percent": {"f1_up": 38.39},
                    }
                ],
                "baseline_warning": "Chưa vượt baseline.",
            },
            "evaluation",
        )
        inference = chatbot_service._format_project_info(
            {
                "facts": {
                    "decision_threshold_percent": 49,
                    "prediction_horizon_sessions": 5,
                }
            },
            "inference",
        )

        self.assertIn("clean → train → predict", overview)
        self.assertIn("n_estimators=200", model)
        self.assertIn("TRAIN 2020-01-01 → 2024-12-31, 70 dòng", dataset)
        self.assertIn("return_1d: 0.3", features)
        self.assertIn("Always UP", evaluation)
        self.assertNotIn("missing", evaluation)
        self.assertIn("decision_threshold 49%", inference)

    def test_single_signal_is_full_and_multi_signal_is_compact(self):
        signal = {
            "symbol": "FPT",
            "reference_date": "2026-07-20",
            "close_at_reference": 123.456,
            "return_20d_percent": 4.567,
            "volatility_20d_percent": 1.234,
            "volume_ratio_20": 1.876,
            "prediction": "UP",
            "up_score_percent": 62.345,
            "decision_threshold_percent": 49.0,
            "threshold_relation": "above",
            "threshold_gap_percent_points": 13.345,
        }
        single = {"action": "STOCK_SIGNAL", "arguments": {"symbols": ["FPT"]}}
        multi = {
            "action": "STOCK_SIGNAL",
            "arguments": {"symbols": ["FPT", "VNM"]},
        }
        second = {**signal, "symbol": "VNM", "up_score_percent": 51.0}
        with patch.object(chatbot_service, "_provider_call") as provider:
            single_response = chatbot_service._format_response(
                single, domain_result({"signals": [signal]})
            )
            multi_response = chatbot_service._format_response(
                multi, domain_result({"signals": [signal, second]})
            )
        provider.assert_not_called()
        for label in (
            "Giá đóng cửa tham chiếu",
            "Lợi suất 20 phiên",
            "Biến động 20 phiên",
            "Tỷ lệ khối lượng 20 phiên",
        ):
            self.assertIn(label, single_response["answer"])
            self.assertNotIn(label, multi_response["answer"])
        self.assertIn("FPT", single_response["answer"])
        self.assertIn("VNM", multi_response["answer"])
        self.assertTrue(single_response["answer"].endswith(chatbot_service.STOCK_DISCLAIMER))
        self.assertTrue(multi_response["answer"].endswith(chatbot_service.STOCK_DISCLAIMER))

    def test_general_out_of_scope_and_symbol_warning_are_deterministic(self):
        answers = {}
        for kind in (
            "greeting",
            "thanks",
            "capabilities",
            "clarify_symbol",
            "clarify_request",
        ):
            decision = {"action": "GENERAL_CHAT", "arguments": {"kind": kind}}
            first = chatbot_service._format_response(decision)["answer"]
            second = chatbot_service._format_response(decision)["answer"]
            self.assertEqual(first, second)
            self.assertTrue(first.strip())
            answers[kind] = first
        self.assertEqual(len(set(answers.values())), len(answers))

        news = {
            "action": "OUT_OF_SCOPE",
            "arguments": {"reason": "news"},
        }
        self.assertEqual(
            chatbot_service._format_response(news)["answer"],
            chatbot_service.OUT_OF_SCOPE_MESSAGES["news"],
        )
        stock = {
            "action": "STOCK_SIGNAL",
            "arguments": {"symbols": ["ABC"]},
        }
        response = chatbot_service._format_response(
            stock,
            domain_result(
                {"unsupported_symbols": ["ABC"]},
                warnings=[
                    {"code": "symbol_out_of_scope", "message": "Mã ABC nằm ngoài phạm vi model đang phục vụ."}
                ],
            ),
        )
        self.assertIn("ABC", response["answer"])
        self.assertEqual(len(response["warnings"]), 1)

    def test_five_actions_end_to_end_use_exactly_one_provider_call(self):
        cases = (
            ("GENERAL_CHAT", {"kind": "greeting"}, None),
            ("STOCK_SIGNAL", {"symbols": ["FPT"]}, domain_result({"signals": []})),
            (
                "STOCK_RANKING",
                {"order": "highest", "top_n": 1},
                domain_result({"ranking": []}),
            ),
            ("PROJECT_INFO", {"topic": "overview"}, domain_result({"topic": "overview"})),
            ("OUT_OF_SCOPE", {"reason": "news"}, None),
        )
        for action, arguments, action_result in cases:
            with self.subTest(action=action):
                client = fake_client(provider_response(decision_json(action, arguments)))
                with patch.object(
                    chatbot_service.chatbot_tools,
                    "execute_action",
                    return_value=action_result,
                ) as execute:
                    response = chatbot_service.chat("message", [], client=client)
                client.chat.completions.create.assert_called_once()
                if action in {"STOCK_SIGNAL", "STOCK_RANKING", "PROJECT_INFO"}:
                    execute.assert_called_once_with(action, arguments)
                else:
                    execute.assert_not_called()
                self.assertEqual(
                    set(response),
                    {"answer", "sources", "warnings", "data_as_of", "model_trained_through"},
                )

    def test_invalid_provider_decision_never_calls_dispatcher(self):
        client = fake_client(provider_response("not json"))
        with (
            patch.object(chatbot_service.chatbot_tools, "execute_action") as execute,
            self.assertRaises(chatbot_service.ChatbotServiceError),
        ):
            chatbot_service.chat("FPT", [], client=client)
        execute.assert_not_called()

    def test_followups_are_resolved_by_llm_from_history(self):
        cases = (
            (
                "Còn HPG?",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "STOCK_SIGNAL",
                {"symbols": ["HPG"]},
            ),
            (
                "Còn VNM?",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "STOCK_SIGNAL",
                {"symbols": ["VNM"]},
            ),
            (
                "So với MWG?",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "STOCK_SIGNAL",
                {"symbols": ["FPT", "MWG"]},
            ),
            (
                "Vậy train thế nào?",
                [{"role": "user", "content": "Dataset dùng gì?"}, {"role": "assistant", "content": "Dữ liệu offline."}],
                "PROJECT_INFO",
                {"topic": "model"},
            ),
            (
                "Mã thứ hai thì sao?",
                [{"role": "user", "content": "Top 3 mã"}, {"role": "assistant", "content": "1. FPT 2. VNM 3. MWG"}],
                "STOCK_SIGNAL",
                {"symbols": ["VNM"]},
            ),
            (
                "Cảm ơn",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "GENERAL_CHAT",
                {"kind": "thanks"},
            ),
        )
        for message, history, action, arguments in cases:
            client = fake_client(provider_response(decision_json(action, arguments)))
            result = domain_result(
                {"signals": []} if action == "STOCK_SIGNAL" else {"topic": "model"}
            )
            with patch.object(
                chatbot_service.chatbot_tools, "execute_action", return_value=result
            ) as execute:
                chatbot_service.chat(message, history, client=client)
            if action in {"STOCK_SIGNAL", "STOCK_RANKING", "PROJECT_INFO"}:
                execute.assert_called_once_with(action, arguments)
            else:
                execute.assert_not_called()
            request_messages = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual(request_messages[1:-1], history)

    def test_provider_errors_config_and_url_validation_have_stable_public_status(self):
        for error, code, status in (
            (RuntimeError("secret"), "provider_error", 502),
            (TimeoutError("slow"), "provider_timeout", 504),
        ):
            with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                chatbot_service.chat("FPT", [], client=fake_client(side_effect=error))
            self.assertEqual((raised.exception.code, raised.exception.status), (code, status))
            self.assertNotIn("secret", raised.exception.message)

        with (
            patch.multiple(chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""),
            self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
        ):
            chatbot_service.chat("FPT", [])
        self.assertEqual((raised.exception.code, raised.exception.status), ("llm_not_configured", 503))

        for invalid in (
            "http://provider.example/v1",
            "https://user:pass@provider.example/v1",
            "https://provider.example/v1?x=1",
        ):
            with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                chatbot_service._validate_base_url(invalid)
            self.assertEqual(raised.exception.code, "llm_invalid_config")


class ApiTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_rejects_invalid_payload_history_and_removed_state(self):
        eight_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "x" * 1_000,
            }
            for index in range(8)
        ]
        cases = (
            {"data": "{}", "content_type": "text/plain"},
            {"data": "{bad", "content_type": "application/json"},
            {"json": {"message": "FPT", "conversation_state": {}}},
            {"json": {"message": "   "}},
            {"json": {"message": "x" * 1001}},
            {"json": {"message": "FPT", "history": eight_messages}},
            {"json": {"message": "FPT", "history": [{"role": "user", "content": "odd"}]}},
            {"json": {"message": "FPT", "history": [{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}]}},
            {
                "json": {
                    "message": "FPT",
                    "history": [
                        {"role": "user", "content": "x" * 1001},
                        {"role": "assistant", "content": "y"},
                    ],
                }
            },
            {
                "json": {
                    "message": "FPT",
                    "history": [
                        {"role": "user", "content": "x", "extra": True},
                        {"role": "assistant", "content": "y"},
                    ],
                }
            },
        )
        with patch.object(web_app.chatbot_service, "chat") as chat:
            for payload in cases:
                response = self.client.post("/api/chat", **payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["error"]["code"], "invalid_request")
        chat.assert_not_called()

    def test_accepts_exact_message_history_count_and_character_boundaries(self):
        expected = {
            "answer": "ok",
            "sources": [],
            "warnings": [],
            "data_as_of": None,
            "model_trained_through": None,
        }
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": str(index) * 1_000,
            }
            for index in range(6)
        ]
        with patch.object(
            web_app.chatbot_service, "chat", return_value=expected
        ) as chat:
            response = self.client.post(
                "/api/chat", json={"message": "m" * 1_000, "history": history}
            )
        self.assertEqual(response.status_code, 200)
        chat.assert_called_once_with("m" * 1_000, history)

    def test_trims_payload_and_returns_five_field_contract(self):
        expected = {
            "answer": "Kết quả",
            "sources": [],
            "warnings": [],
            "data_as_of": "2026-07-20",
            "model_trained_through": "2026-04-10",
        }
        history = [
            {"role": "user", "content": " FPT "},
            {"role": "assistant", "content": " UP "},
        ]
        with patch.object(web_app.chatbot_service, "chat", return_value=expected) as chat:
            response = self.client.post(
                "/api/chat", json={"message": " Còn mã đó? ", "history": history}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, expected)
        chat.assert_called_once_with(
            "Còn mã đó?",
            [{"role": "user", "content": "FPT"}, {"role": "assistant", "content": "UP"}],
        )

    def test_maps_service_errors_and_hides_unexpected_details(self):
        for code, status in (
            ("provider_protocol_error", 502),
            ("llm_not_configured", 503),
            ("provider_timeout", 504),
        ):
            with patch.object(
                web_app.chatbot_service,
                "chat",
                side_effect=chatbot_service.ChatbotServiceError(code, "Public", status),
            ):
                response = self.client.post("/api/chat", json={"message": "FPT"})
            self.assertEqual((response.status_code, response.json["error"]["code"]), (status, code))

        with patch.object(
            web_app.chatbot_service,
            "chat",
            side_effect=RuntimeError(r"secret C:\private\artifact.pkl"),
        ):
            response = self.client.post("/api/chat", json={"message": "FPT"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private", response.get_data(as_text=True))


class FrontendTests(unittest.TestCase):
    def test_chat_page_and_floating_dock_load_assets_locally_once(self):
        base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        page = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        # Trang /chat tự nạp bộ asset riêng của nó, mỗi thứ đúng một lần.
        self.assertEqual(page.count("chat-ui.css"), 1)
        self.assertEqual(page.count("chat-client.js"), 1)
        for contract in ('id="chat-form"', 'id="chat-transcript"', 'aria-live="polite"'):
            self.assertIn(contract, page)
        # Khung nổi nạp transport dùng chung + script/CSS riêng, không kéo theo
        # layout của trang /chat.
        self.assertEqual(base.count("chat-client.js"), 1)
        self.assertEqual(base.count("chat-dock.js"), 1)
        self.assertEqual(base.count("chat-dock.css"), 1)
        self.assertNotIn("chat-ui.css", base)
        # CSS và markup của dock đều nằm sau cùng một guard nên /chat không bị
        # trùng id với trang chat đầy đủ.
        self.assertEqual(base.count("{% if active_page != 'chat' %}"), 2)
        for contract in (
            'id="chat-dock"',
            'id="chat-dock-panel"',
            'id="chat-dock-toggle"',
            'id="chat-dock-form"',
            'id="chat-dock-transcript"',
        ):
            self.assertIn(contract, base)

    def test_client_uses_safe_dom_short_history_and_no_removed_protocol(self):
        page = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        client = (ROOT / "static" / "chat-client.js").read_text(encoding="utf-8")
        dock = (ROOT / "static" / "chat-dock.js").read_text(encoding="utf-8")
        combined = page + client + dock
        self.assertIn("session.transcript.slice(-6)", client)
        self.assertIn("window.sessionStorage", client)
        self.assertIn("textContent", combined)
        self.assertNotIn("innerHTML", combined)
        for removed in (
            "conversation_state",
            "release_status",
            "renderRichText",
            "buildSourceDetails",
            "buildCopyButton",
        ):
            self.assertNotIn(removed, combined)

    @unittest.skipUnless(shutil.which("node"), "cần Node.js để kiểm session runtime")
    def test_session_runtime_is_bounded_recovers_and_skips_failed_exchange(self):
        script = r"""
        const assert = require("node:assert/strict");
        const values = new Map();
        global.window = {
            sessionStorage: {
                getItem(k) { return values.has(k) ? values.get(k) : null; },
                setItem(k, v) { values.set(k, v); },
                removeItem(k) { values.delete(k); }
            },
            setTimeout() { return 1; }, clearTimeout() {}
        };
        require("./static/chat-client.js");
        const kit = window.ChatClientKit;
        (async () => {
            values.set(kit.SESSION_KEY, JSON.stringify({transcript: [{role:"user",content:"odd"}]}));
            assert.deepEqual(kit.loadSession(), {transcript: []});
            const requests = [];
            global.fetch = async (_url, options) => {
                requests.push(JSON.parse(options.body));
                return {ok:true,json:async()=>({answer:"ok",sources:[],warnings:[]})};
            };
            for (let i = 0; i < 21; i += 1) await kit.sendMessage("m" + i);
            const saved = JSON.parse(values.get(kit.SESSION_KEY));
            assert.equal(saved.transcript.length, 40);
            assert.equal(requests.at(-1).history.length, 6);
            const before = values.get(kit.SESSION_KEY);
            global.fetch = async () => ({ok:false,json:async()=>({error:{message:"fail"}})});
            await assert.rejects(kit.sendMessage("bad"));
            assert.equal(values.get(kit.SESSION_KEY), before);
        })().catch((error) => process.nextTick(() => { throw error; }));
        """
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
