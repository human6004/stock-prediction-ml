"""Contract tests for the local grounded chatbot."""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import pandas as pd

import app as web_app
from config.settings import EXPERIMENT_POLICY_ID, FEATURE_COLUMNS
from services import chatbot_service, chatbot_tools

ROOT_DIR = Path(__file__).resolve().parents[1]


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


def context_bundle(*, context=None, state=None):
    warning = {"code": "legacy_policy", "message": "Release policy cũ."}
    return {
        "context": context
        or {
            "project_snapshot": {
                "model_name": "Random Forest",
                "release_status": "legacy",
                "data_as_of": "2026-07-20",
            }
        },
        "sources": [{"kind": "project_snapshot", "as_of": "2026-07-20"}],
        "warnings": [warning],
        "release_status": "legacy",
        "data_as_of": "2026-07-20",
        "grounded_numbers": [2026.0, -7.0, -20.0],
        "stock_facts": {},
        "conversation_state": state
        or {
            "active_symbols": [],
            "topic": None,
            "ranking_order": None,
            "last_result_symbols": [],
        },
    }


class ChatbotToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model_path = root / "final_model.pkl"
        self.metadata_path = root / "model_metadata.json"
        self.summary_path = root / "pipeline_summary.json"
        self.eligible_path = root / "eligible_symbols.csv"
        self.excluded_path = root / "excluded_symbols.csv"
        self.feature_importance_path = root / "feature_importance.csv"
        self.cleaned_path = root / "hose_stock_clean.csv"

        self.identity = {
            "model_name": "Random Forest",
            "policy_id": EXPERIMENT_POLICY_ID,
            "content_fingerprint": "release-123",
        }
        self.artifact = {
            **self.identity,
            "prediction_horizon": 5,
            "up_threshold": 0.01,
            "decision_threshold": 0.49,
            "baseline_passed": True,
            "trained_at": "2026-07-20T12:00:00",
            "train_through_date": "2026-04-10",
        }
        self.metadata = {
            **self.identity,
            "feature_order": FEATURE_COLUMNS,
            "prediction_horizon": 5,
            "up_threshold": 0.01,
            "decision_threshold": 0.49,
            "baseline_passed": True,
            "baseline_warning": None,
            "trained_at": "2026-07-20T12:00:00",
            "train_through_date": "2026-04-10",
            "training_symbols": ["FPT", "VNM"],
            "training_symbol_count": 2,
            "validation_selection_metrics": {"f1_up": 0.51},
            "final_test_metrics": {"f1_up": 0.48},
        }
        self.summary = {
            "policy_id": EXPERIMENT_POLICY_ID,
            "content_fingerprint": "release-123",
            "selection_report": {"selected_model_name": "Random Forest"},
            "dataset_report": {
                "date_min": "2019-08-14",
                "date_max": "2026-07-20",
            },
            "clean_report": {
                "symbols_after_cleaning": 400,
                "excluded_symbols": 4,
            },
            "feature_report": {"feature_count": 20},
            "feature_importance_written": True,
        }
        self._write_release()
        pd.DataFrame({"symbol": ["FPT", "VNM", "HPG"]}).to_csv(
            self.eligible_path, index=False
        )
        pd.DataFrame({"symbol": ["CRV"]}).to_csv(self.excluded_path, index=False)
        pd.DataFrame(
            {
                "feature": ["return_1d", "volatility_20d"],
                "importance": [0.3, 0.2],
            }
        ).to_csv(self.feature_importance_path, index=False)
        pd.DataFrame(
            {
                "symbol": ["FPT", "VNM"],
                "trading_date": ["2026-07-20", "2026-07-20"],
            }
        ).to_csv(self.cleaned_path, index=False)

        replacements = {
            "FINAL_MODEL_PATH": self.model_path,
            "MODEL_METADATA_PATH": self.metadata_path,
            "PIPELINE_SUMMARY_PATH": self.summary_path,
            "ELIGIBLE_SYMBOLS_PATH": self.eligible_path,
            "EXCLUDED_SYMBOLS_PATH": self.excluded_path,
            "FEATURE_IMPORTANCE_PATH": self.feature_importance_path,
            "CLEANED_DATA_PATH": self.cleaned_path,
        }
        self.patchers = [
            patch.object(chatbot_tools, name, value)
            for name, value in replacements.items()
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tmp.cleanup()

    def _write_release(self):
        joblib.dump(self.artifact, self.model_path)
        self.metadata_path.write_text(
            json.dumps(self.metadata), encoding="utf-8"
        )
        self.summary_path.write_text(json.dumps(self.summary), encoding="utf-8")

    @staticmethod
    def _signal(symbol, score, *, date="2026-07-20"):
        return {
            "symbol": symbol,
            "prediction": "UP" if score >= 0.49 else "NOT_UP",
            "probability_up": score,
            "decision_threshold": 0.49,
            "close_at_reference": 70.6,
            "reference_date": date,
            "return_20d": 0.12,
            "volatility_20d": 0.02,
            "volume_ratio_20": 1.25,
            "is_stale": False,
            "horizon": 5,
        }

    def test_server_context_handlers_are_direct(self):
        for name in (
            "get_stock_signals",
            "get_ranking",
            "get_model_info",
            "get_dataset_info",
            "get_feature_info",
            "get_project_info",
        ):
            self.assertTrue(callable(getattr(chatbot_tools, name)))
        self.assertFalse(hasattr(chatbot_tools, "TOOL_DEFINITIONS"))

    def test_stock_signals_normalize_scope_call_once_and_convert_percentages(self):
        rows = [self._signal("FPT", 0.812), self._signal("VNM", 0.42)]
        with patch.object(
            chatbot_tools.prediction_service, "predict_symbols", return_value=rows
        ) as predict:
            result = chatbot_tools.get_stock_signals(
                {"symbols": [" fpt ", "vnm"]}
            )

        predict.assert_called_once_with(["FPT", "VNM"])
        self.assertTrue(result["ok"])
        first = result["data"]["signals"][0]
        self.assertEqual(first["up_score_percent"], 81.2)
        self.assertEqual(first["decision_threshold_percent"], 49.0)
        self.assertEqual(first["threshold_relation"], "above")
        self.assertEqual(first["threshold_gap_percent_points"], 32.2)
        self.assertEqual(first["return_20d_percent"], 12.0)
        self.assertEqual(first["volatility_20d_percent"], 2.0)
        self.assertEqual(first["price_unit"], "source_native")
        self.assertNotIn("probability_up", first)
        self.assertEqual(result["data"]["signal_count"], 2)
        self.assertEqual(
            result["data"]["comparison"],
            {
                "higher_up_score_symbol": "FPT",
                "lower_up_score_symbol": "VNM",
                "same_up_score": False,
                "up_score_gap_percent_points": 39.2,
            },
        )
        self.assertEqual(result["source"]["symbols"], ["FPT", "VNM"])

    def test_stock_signals_reject_duplicates_and_out_of_scope_before_inference(self):
        with patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict:
            duplicate = chatbot_tools.get_stock_signals(
                {"symbols": ["FPT", " fpt "]}
            )
            outside = chatbot_tools.get_stock_signals({"symbols": ["HPG"]})

        self.assertFalse(duplicate["ok"])
        self.assertEqual(duplicate["error"]["code"], "invalid_arguments")
        self.assertEqual(outside["error"]["code"], "symbol_out_of_scope")
        predict.assert_not_called()

    def test_legacy_fallback_has_policy_baseline_and_scope_warnings(self):
        legacy = "legacy_pre_validation_baseline_gate"
        self.artifact.update(policy_id=legacy, baseline_passed=False)
        self.metadata.pop("training_symbols")
        self.metadata.pop("training_symbol_count")
        self.metadata.update(
            policy_id=legacy,
            baseline_passed=False,
            baseline_warning="Không vượt baseline TEST.",
        )
        self.summary["policy_id"] = legacy
        self._write_release()

        with patch.object(
            chatbot_tools.prediction_service,
            "predict_symbols",
            return_value=[self._signal("HPG", 0.6)],
        ):
            result = chatbot_tools.get_stock_signals({"symbols": ["HPG"]})

        self.assertTrue(result["ok"])
        self.assertEqual(result["release"]["status"], "legacy")
        self.assertEqual(
            {warning["code"] for warning in result["warnings"]},
            {"legacy_policy", "baseline_failed", "symbol_scope_unverified"},
        )

    def test_ranking_filters_scope_none_scores_sorts_and_limits(self):
        rows = [
            self._signal("FPT", 0.8),
            self._signal("VNM", 0.4),
            self._signal("HPG", 0.99),
            {**self._signal("FPT", 0.8), "symbol": "FPT2", "probability_up": None},
        ]
        with patch.object(
            chatbot_tools.prediction_service, "predict_all_symbols", return_value=rows
        ) as predict:
            highest = chatbot_tools.get_ranking(
                {"order": "highest_up_score", "top_n": 1}
            )
            lowest = chatbot_tools.get_ranking(
                {"order": "lowest_up_score", "top_n": 2}
            )

        self.assertEqual(predict.call_count, 2)
        self.assertEqual([row["symbol"] for row in highest["data"]["ranking"]], ["FPT"])
        self.assertEqual(
            [row["symbol"] for row in lowest["data"]["ranking"]],
            ["VNM", "FPT"],
        )

    def test_model_dataset_and_feature_tools_return_release_safe_fields(self):
        model = chatbot_tools.get_model_info({})
        dataset = chatbot_tools.get_dataset_info({})
        feature = chatbot_tools.get_feature_info({"top_n": 1})

        self.assertEqual(model["data"]["model_name"], "Random Forest")
        self.assertEqual(model["data"]["decision_threshold_percent"], 49.0)
        self.assertEqual(model["data"]["training_symbol_count"], 2)
        self.assertEqual(
            model["data"]["final_test_metrics_percent"], {"f1_up": 48.0}
        )
        self.assertTrue(dataset["data"]["offline"])
        self.assertEqual(dataset["data"]["data_as_of"], "2026-07-20")
        self.assertEqual(dataset["data"]["clean_symbol_count"], 400)
        self.assertEqual(dataset["data"]["excluded_symbol_count"], 4)
        self.assertEqual(dataset["data"]["feature_count"], 20)
        self.assertEqual(feature["data"]["scope"], "global_model_importance")
        self.assertEqual(len(feature["data"]["global_importance"]), 1)

    def test_inconsistent_release_blocks_predictions_metrics_and_importance(self):
        self.summary["content_fingerprint"] = "other-release"
        self._write_release()

        with (
            patch.object(chatbot_tools.prediction_service, "predict_symbols") as signals,
            patch.object(chatbot_tools.prediction_service, "predict_all_symbols") as ranking,
        ):
            signal_result = chatbot_tools.get_stock_signals({"symbols": ["FPT"]})
            ranking_result = chatbot_tools.get_ranking(
                {"order": "highest_up_score", "top_n": 5}
            )
        model_result = chatbot_tools.get_model_info({})
        feature_result = chatbot_tools.get_feature_info({"top_n": 5})
        dataset_result = chatbot_tools.get_dataset_info({})

        self.assertEqual(signal_result["error"]["code"], "release_inconsistent")
        self.assertEqual(ranking_result["error"]["code"], "release_inconsistent")
        self.assertNotIn("final_test_metrics_percent", model_result["data"])
        self.assertEqual(feature_result["error"]["code"], "release_inconsistent")
        self.assertTrue(dataset_result["ok"])
        signals.assert_not_called()
        ranking.assert_not_called()

    def test_invalid_metadata_symbol_snapshot_is_inconsistent(self):
        self.metadata["training_symbol_count"] = 99
        self._write_release()

        release = chatbot_tools.get_release_state()

        self.assertEqual(release["status"], "inconsistent")

    def test_current_release_requires_exact_training_symbol_snapshot(self):
        self.metadata.pop("training_symbols")
        self.metadata.pop("training_symbol_count")
        self._write_release()

        release = chatbot_tools.get_release_state()

        self.assertEqual(release["status"], "inconsistent")

    def test_missing_release_blocks_signal_but_dataset_fallback_still_works(self):
        self.model_path.unlink()
        self.summary_path.unlink()

        with patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict:
            signal = chatbot_tools.get_stock_signals({"symbols": ["FPT"]})
        dataset = chatbot_tools.get_dataset_info({})

        self.assertEqual(signal["release"]["status"], "missing")
        self.assertEqual(signal["error"]["code"], "model_missing")
        self.assertTrue(dataset["ok"])
        self.assertEqual(dataset["data"]["excluded_symbol_count"], 1)
        predict.assert_not_called()

    def test_pipeline_running_blocks_release_dependent_tools(self):
        with (
            patch.object(chatbot_tools.experiment_state, "is_pipeline_running", return_value=True),
            patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict,
        ):
            signal = chatbot_tools.get_stock_signals({"symbols": ["FPT"]})
            feature = chatbot_tools.get_feature_info({"top_n": 1})

        self.assertEqual(signal["error"]["code"], "release_inconsistent")
        self.assertEqual(feature["error"]["code"], "release_inconsistent")
        self.assertIn(
            "release_updating", {item["code"] for item in signal["warnings"]}
        )
        predict.assert_not_called()

    def test_inconsistent_model_info_uses_serving_artifact_and_omits_baseline(self):
        self.metadata["model_name"] = "Other Model"
        self._write_release()

        result = chatbot_tools.get_model_info({})

        self.assertEqual(result["release"]["status"], "inconsistent")
        self.assertEqual(result["data"]["model_name"], "Random Forest")
        self.assertNotIn("baseline_passed", result["data"])
        self.assertNotIn("baseline_warning", result["data"])

    def test_prediction_error_is_sanitized(self):
        with patch.object(
            chatbot_tools.prediction_service,
            "predict_symbols",
            side_effect=RuntimeError(r"secret at C:\private\model.pkl"),
        ):
            result = chatbot_tools.get_stock_signals({"symbols": ["FPT"]})

        serialized = json.dumps(result)
        self.assertEqual(result["error"]["code"], "prediction_unavailable")
        self.assertNotIn("private", serialized)
        self.assertNotIn("model.pkl", serialized)


class ChatbotDispatcherTests(unittest.TestCase):
    @staticmethod
    def _envelope(data=None, *, source=None, as_of=None, warnings=None, error=None):
        return {
            "ok": error is None,
            "data": data or {},
            "source": source,
            "as_of": as_of,
            "release": {"status": "legacy"},
            "warnings": warnings or [],
            "error": error,
        }

    def test_dispatches_stock_signal_and_ranking_with_domain_metadata(self):
        signal_envelope = self._envelope(
            {"signals": [{"symbol": "FPT"}]},
            source={"kind": "stock_signal", "symbols": ["FPT"]},
            as_of="2026-07-20",
            warnings=[{"code": "legacy", "message": "Legacy."}],
        )
        with patch.object(
            chatbot_tools, "get_stock_signals", return_value=signal_envelope
        ) as get_signal:
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL", {"symbols": ["FPT"], "focus": "prediction"}
            )

        get_signal.assert_called_once_with({"symbols": ["FPT"]})
        self.assertEqual(result["data"]["focus"], "prediction")
        self.assertEqual(result["data_as_of"], "2026-07-20")
        self.assertEqual(result["sources"], [signal_envelope["source"]])
        self.assertEqual(result["warnings"], signal_envelope["warnings"])

        ranking_envelope = self._envelope(
            {"ranking": []}, source={"kind": "ranking"}, as_of="2026-07-20"
        )
        with patch.object(
            chatbot_tools, "get_ranking", return_value=ranking_envelope
        ) as get_ranking:
            result = chatbot_tools.execute_action(
                "STOCK_RANKING", {"order": "lowest", "top_n": 3}
            )

        get_ranking.assert_called_once_with(
            {"order": "lowest_up_score", "top_n": 3}
        )
        self.assertEqual(result["data"]["order"], "lowest")

    def test_dispatches_six_project_topics(self):
        project_envelope = self._envelope(
            {"title": "Project"}, source={"kind": "project_contract"}
        )
        model_envelope = self._envelope(
            {"model_name": "Random Forest", "train_through_date": "2026-04-10"},
            source={"kind": "model_metadata"},
            as_of="2026-04-10",
        )
        dataset_envelope = self._envelope(
            {"data_as_of": "2026-07-20"},
            source={"kind": "dataset_info"},
            as_of="2026-07-20",
        )
        feature_envelope = self._envelope(
            {"global_importance": []}, source={"kind": "feature_importance"}
        )

        with (
            patch.object(
                chatbot_tools, "get_project_info", return_value=project_envelope
            ) as get_project,
            patch.object(
                chatbot_tools, "get_model_info", return_value=model_envelope
            ) as get_model,
            patch.object(
                chatbot_tools, "get_dataset_info", return_value=dataset_envelope
            ) as get_dataset,
            patch.object(
                chatbot_tools, "get_feature_info", return_value=feature_envelope
            ) as get_features,
        ):
            overview = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "overview"}
            )
            model = chatbot_tools.execute_action("PROJECT_INFO", {"topic": "model"})
            dataset = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "dataset"}
            )
            features = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "features"}
            )
            method = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "method"}
            )
            limitations = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "limitations"}
            )

        self.assertEqual(overview["data"]["topic"], "overview")
        self.assertEqual(model["model_trained_through"], "2026-04-10")
        self.assertEqual(dataset["data_as_of"], "2026-07-20")
        self.assertEqual(features["data"]["topic"], "features")
        self.assertEqual(set(method["data"]["sections"]), {
            "target", "data_split", "training", "inference"
        })
        self.assertEqual(limitations["data"]["topic"], "limitations")
        get_model.assert_called_once_with({})
        get_dataset.assert_called_once_with({})
        get_features.assert_called_once_with({"top_n": 10})
        self.assertEqual(get_project.call_count, 6)

    def test_unknown_action_never_calls_a_handler(self):
        handlers = (
            "get_stock_signals",
            "get_ranking",
            "get_model_info",
            "get_dataset_info",
            "get_feature_info",
            "get_project_info",
        )
        patches = [patch.object(chatbot_tools, name) for name in handlers]
        mocks = [item.start() for item in patches]
        try:
            with self.assertRaises(ValueError):
                chatbot_tools.execute_action("GENERAL_CHAT", {})
            for mock in mocks:
                mock.assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()


class ChatbotDecisionTests(unittest.TestCase):
    def test_decides_each_supported_action_with_one_plain_json_call(self):
        cases = (
            (
                "Dự đoán FPT",
                {
                    "action": "STOCK_SIGNAL",
                    "arguments": {"symbols": ["fpt"], "focus": "prediction"},
                    "direct_answer": None,
                },
                {
                    "action": "STOCK_SIGNAL",
                    "arguments": {"symbols": ["FPT"], "focus": "prediction"},
                    "direct_answer": None,
                },
            ),
            (
                "So sánh FPT với VNM",
                {
                    "action": "STOCK_SIGNAL",
                    "arguments": {
                        "symbols": ["FPT", "vnm"],
                        "focus": "comparison",
                    },
                    "direct_answer": None,
                },
                {
                    "action": "STOCK_SIGNAL",
                    "arguments": {
                        "symbols": ["FPT", "VNM"],
                        "focus": "comparison",
                    },
                    "direct_answer": None,
                },
            ),
            (
                "Top 3 mã thấp nhất",
                {
                    "action": "STOCK_RANKING",
                    "arguments": {"order": "lowest", "top_n": 3},
                    "direct_answer": None,
                },
                {
                    "action": "STOCK_RANKING",
                    "arguments": {"order": "lowest", "top_n": 3},
                    "direct_answer": None,
                },
            ),
            (
                "Model dùng gì?",
                {
                    "action": "PROJECT_INFO",
                    "arguments": {"topic": "model"},
                    "direct_answer": None,
                },
                {
                    "action": "PROJECT_INFO",
                    "arguments": {"topic": "model"},
                    "direct_answer": None,
                },
            ),
            (
                "Tin hôm nay?",
                {
                    "action": "OUT_OF_SCOPE",
                    "arguments": {"reason": "news"},
                    "direct_answer": None,
                },
                {
                    "action": "OUT_OF_SCOPE",
                    "arguments": {"reason": "news"},
                    "direct_answer": None,
                },
            ),
            (
                "Phân tích cổ phiếu",
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {},
                    "direct_answer": "Bạn muốn phân tích mã cổ phiếu nào?",
                },
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {},
                    "direct_answer": "Bạn muốn phân tích mã cổ phiếu nào?",
                },
            ),
        )

        for message, provider_value, expected in cases:
            with self.subTest(message=message):
                client = fake_client(
                    provider_response(json.dumps(provider_value, ensure_ascii=False))
                )
                actual = chatbot_service._decide(
                    message, [], client, monotonic=Mock(side_effect=[0, 0, 0])
                )

                self.assertEqual(actual, expected)
                client.chat.completions.create.assert_called_once()
                request = client.chat.completions.create.call_args.kwargs
                self.assertNotIn("tools", request)
                self.assertNotIn("tool_choice", request)
                self.assertNotIn("response_format", request)

    def test_decision_rejects_invalid_protocol_without_retry(self):
        invalid_values = (
            "```json\n{}\n```",
            '{"action":"GENERAL_CHAT","arguments":{},"direct_answer":"Chào"} prose',
            json.dumps(
                {
                    "action": "UNKNOWN",
                    "arguments": {},
                    "direct_answer": None,
                }
            ),
            json.dumps(
                {
                    "action": "STOCK_RANKING",
                    "arguments": {"order": "highest", "top_n": True},
                    "direct_answer": None,
                }
            ),
            json.dumps(
                {
                    "action": "STOCK_SIGNAL",
                    "arguments": {"symbols": ["FPT"], "focus": []},
                    "direct_answer": None,
                }
            ),
            json.dumps(
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {},
                    "direct_answer": "Chào",
                    "extra": True,
                }
            ),
        )

        for content in invalid_values:
            with self.subTest(content=content):
                client = fake_client(provider_response(content))
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._decide(
                        "test", [], client, monotonic=Mock(side_effect=[0, 0, 0])
                    )
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("provider_protocol_error", 502),
                )
                client.chat.completions.create.assert_called_once()


class ChatbotServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.multiple(
            chatbot_service,
            LLM_BASE_URL="http://provider.local/v1",
            LLM_API_KEY="test-secret",
            LLM_MODEL="tool-model",
        )
        self.config.start()

    def tearDown(self):
        self.config.stop()

    def test_every_turn_calls_provider_once_without_tool_protocol(self):
        bundle = context_bundle()
        client = fake_client(provider_response("Chào bạn, mình có thể giúp gì?"))
        with patch.object(
            chatbot_service, "build_context", return_value=bundle
        ) as build_context:
            result = chatbot_service.chat("Chào bạn", [], client=client)

        build_context.assert_called_once()
        self.assertEqual(build_context.call_args.args, ("Chào bạn", [], None))
        self.assertTrue(callable(build_context.call_args.kwargs["deadline_check"]))
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertIn("CONTEXT_JSON", request["messages"][0]["content"])
        self.assertIn("project_snapshot", request["messages"][0]["content"])
        self.assertEqual(result["sources"], bundle["sources"])
        self.assertEqual(result["warnings"], bundle["warnings"])
        self.assertEqual(result["release_status"], "legacy")
        self.assertEqual(result["data_as_of"], "2026-07-20")

    def test_history_is_bounded_and_state_only_guides_current_context(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
            for index in range(8)
        ]
        state = {
            "active_symbols": ["fpt"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": [],
        }
        canonical = {**state, "active_symbols": ["FPT"]}
        client = fake_client(provider_response("Mình đã đọc lại dữ liệu hiện tại."))
        with patch.object(
            chatbot_service,
            "build_context",
            return_value=context_bundle(state=canonical),
        ) as build_context:
            result = chatbot_service.chat(
                "Còn nó?", history, state, client=client
            )

        self.assertEqual(build_context.call_args.args, ("Còn nó?", history, state))
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[1:-1], history[-chatbot_service.MAX_HISTORY_MESSAGES :])
        self.assertEqual(messages[-1], {"role": "user", "content": "Còn nó?"})
        self.assertEqual(result["conversation_state"], canonical)

    def test_tool_call_or_empty_content_is_provider_protocol_error(self):
        responses = (
            provider_response(
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_model_info", "arguments": "{}"},
                    }
                ],
            ),
            provider_response("   "),
            {"choices": []},
        )
        for response in responses:
            with self.subTest(response=response):
                client = fake_client(response)
                with (
                    patch.object(
                        chatbot_service,
                        "build_context",
                        return_value=context_bundle(),
                    ),
                    patch.object(
                        chatbot_service.chatbot_tools, "get_model_info"
                    ) as get_model_info,
                    self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
                ):
                    chatbot_service.chat("Model nào?", [], client=client)
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("provider_protocol_error", 502),
                )
                client.chat.completions.create.assert_called_once()
                get_model_info.assert_not_called()

    def test_prompt_requires_server_context_and_blocks_trading_advice(self):
        prompt = chatbot_service.SYSTEM_PROMPT.lower()

        self.assertIn("văn phong tự nhiên", prompt)
        self.assertIn("context_json do server cung cấp", prompt)
        self.assertIn("không gọi tool", prompt)
        self.assertIn("không quyết định mua/bán", prompt)
        self.assertIn("history", prompt)

    def test_provider_error_timeout_and_missing_config_have_stable_statuses(self):
        cases = (
            (RuntimeError("secret auth details"), "provider_error", 502),
            (TimeoutError("slow"), "provider_timeout", 504),
        )
        for error, code, status in cases:
            with self.subTest(code=code):
                client = fake_client(side_effect=error)
                with (
                    patch.object(
                        chatbot_service,
                        "build_context",
                        return_value=context_bundle(),
                    ),
                    self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
                ):
                    chatbot_service.chat("FPT", [], client=client)
                self.assertEqual((raised.exception.code, raised.exception.status), (code, status))
                self.assertNotIn("secret", raised.exception.message)

        with (
            patch.multiple(
                chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""
            ),
            patch.object(
                chatbot_service, "build_context", return_value=context_bundle()
            ),
            self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
        ):
            chatbot_service.chat("FPT", [])
        self.assertEqual((raised.exception.code, raised.exception.status), ("llm_not_configured", 503))

    def test_context_budget_and_deadline_fail_before_provider(self):
        oversized = context_bundle(
            context={"project_snapshot": {"blob": "x" * chatbot_service.MAX_CONTEXT_CHARS}}
        )
        client = fake_client(provider_response("không được gọi"))
        with (
            patch.object(chatbot_service, "build_context", return_value=oversized),
            self.assertRaises(chatbot_service.ChatbotServiceError) as over_budget,
        ):
            chatbot_service.chat("Chào", [], client=client)
        self.assertEqual(over_budget.exception.code, "context_budget_exceeded")
        client.chat.completions.create.assert_not_called()

        def slow_context(_message, _history, _state, *, deadline_check):
            deadline_check()
            return context_bundle()

        timed_client = fake_client(provider_response("không được gọi"))
        with (
            patch.object(chatbot_service, "build_context", side_effect=slow_context),
            self.assertRaises(chatbot_service.ChatbotServiceError) as timed_out,
        ):
            chatbot_service.chat(
                "Model?",
                [],
                client=timed_client,
                monotonic=Mock(
                    side_effect=[0, chatbot_service.TOTAL_DEADLINE_SECONDS + 1]
                ),
            )
        self.assertEqual(
            (timed_out.exception.code, timed_out.exception.status),
            ("provider_timeout", 504),
        )
        timed_client.chat.completions.create.assert_not_called()

    def test_ungrounded_numbers_and_direct_advice_are_rejected(self):
        answers = ("FPT chắc chắn tăng 99%.", "Bạn nên mua FPT ngay.")
        for answer in answers:
            with self.subTest(answer=answer):
                client = fake_client(provider_response(answer))
                with (
                    patch.object(
                        chatbot_service,
                        "build_context",
                        return_value=context_bundle(),
                    ),
                    self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
                ):
                    chatbot_service.chat("Model?", [], client=client)
                self.assertEqual(raised.exception.code, "ungrounded_response")
                self.assertEqual(raised.exception.status, 502)

    def test_directional_answer_must_match_server_stock_facts(self):
        bundle = context_bundle()
        bundle["grounded_numbers"] = [62.5, 50.0]
        bundle["stock_facts"] = {
            "FPT": {
                "prediction": "UP",
                "up_score_percent": 62.5,
                "decision_threshold_percent": 50.0,
            }
        }
        client = fake_client(
            provider_response("FPT có Điểm UP 62,5% và nghiêng tích cực.")
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("FPT thế nào?", [], client=client)
        self.assertIn("nghiêng tích cực", result["answer"])

    def test_baseline_warning_numbers_pass_without_model_facts(self):
        # Warning baseline_failed nằm trong MỌI response nên LLM hay nhắc lại
        # "F1 UP 37,54% so với 38,39%". Lượt hỏi không kéo get_model_info thì
        # model_facts rỗng → validator không có expected để so, phải bỏ qua
        # thay vì raise 502 (bug cũ: 502 không xác định, phụ thuộc LLM có
        # nhắc số hay không).
        bundle = context_bundle()
        bundle["grounded_numbers"] = [37.54, 38.39]
        bundle["model_facts"] = {}
        client = fake_client(
            provider_response(
                "Trên TEST, F1 UP đạt 37,54% so với baseline Always UP 38,39%."
            )
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("Hạn chế của model?", [], client=client)
        self.assertIn("37,54%", result["answer"])

    def test_model_metric_still_blocked_when_split_fact_exists(self):
        # Có ground truth thì cổng metric vẫn phải chặn: fix trên chỉ nới đúng
        # trường hợp expected is None, không tắt validator.
        bundle = context_bundle()
        bundle["grounded_numbers"] = [37.54, 99.99]
        bundle["model_facts"] = {"test": {"f1_up": 37.54}}
        client = fake_client(provider_response("F1 UP trên TEST là 99,99%."))
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("F1 UP test bao nhiêu?", [], client=client)
        self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
        self.assertIn(
            "answer_blocked_ungrounded",
            [w["code"] for w in result["warnings"]],
        )

    def test_threshold_distance_phrasing_is_not_read_as_threshold_value(self):
        # Câu thật provider từng trả: 0,60 là KHOẢNG CÁCH tới ngưỡng, không phải
        # giá trị ngưỡng (49,0) → không được raise.
        bundle = context_bundle()
        bundle["grounded_numbers"] = [48.4, 49.0, 0.6]
        bundle["stock_facts"] = {
            "VNM": {
                "prediction": "NOT_UP",
                "up_score_percent": 48.4,
                "decision_threshold_percent": 49.0,
                "threshold_relation": "below",
            }
        }
        for answer in (
            "VNM: 48,40%, NOT_UP, dưới ngưỡng 0,60 điểm %.",
            "VNM có Điểm UP 48,40%, thấp hơn ngưỡng 0,60 pp.",
        ):
            with self.subTest(answer=answer):
                client = fake_client(provider_response(answer))
                with patch.object(
                    chatbot_service, "build_context", return_value=bundle
                ):
                    result = chatbot_service.chat("VNM?", [], client=client)
                self.assertIn("0,60", result["answer"])

    def test_wrong_threshold_value_is_still_rejected(self):
        bundle = context_bundle()
        bundle["grounded_numbers"] = [48.4, 49.0, 12.34]
        bundle["stock_facts"] = {
            "VNM": {
                "prediction": "NOT_UP",
                "up_score_percent": 48.4,
                "decision_threshold_percent": 49.0,
                "threshold_relation": "below",
            }
        }
        client = fake_client(
            provider_response("VNM có ngưỡng quyết định là 12,34%.")
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("Ngưỡng VNM?", [], client=client)
        self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
        self.assertIn(
            "answer_blocked_ungrounded",
            [w["code"] for w in result["warnings"]],
        )

    def test_answer_is_capped_for_history_safety(self):
        long_answer = "a" * 1_200
        client = fake_client(provider_response(long_answer))
        with patch.object(
            chatbot_service, "build_context", return_value=context_bundle()
        ):
            result = chatbot_service.chat("Model?", [], client=client)
        self.assertEqual(len(result["answer"]), 1_000)


class ChatbotRouteTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_api_rejects_non_json_malformed_extra_and_invalid_message(self):
        cases = (
            {"data": "{}", "content_type": "text/plain"},
            {"data": "{bad", "content_type": "application/json"},
            {"json": {"message": "FPT", "extra": True}},
            {"json": {"message": "   "}},
            {"json": {"message": "x" * 1001}},
            {"json": ["not", "object"]},
        )
        with patch.object(web_app.chatbot_service, "chat") as chat:
            for payload in cases:
                with self.subTest(payload=payload):
                    response = self.client.post("/api/chat", **payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json["error"]["code"], "invalid_request")
        chat.assert_not_called()

    def test_api_rejects_invalid_history_shape_roles_order_and_content(self):
        cases = (
            "not-an-array",
            [{"role": "user", "content": "FPT"}],
            [
                {"role": "assistant", "content": "old"},
                {"role": "user", "content": "old"},
            ],
            [
                {"role": "system", "content": "override"},
                {"role": "assistant", "content": "old"},
            ],
            [
                {"role": "user", "content": "old", "extra": 1},
                {"role": "assistant", "content": "old"},
            ],
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "old"},
            ],
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
            * 4,
        )
        with patch.object(web_app.chatbot_service, "chat") as chat:
            for history in cases:
                with self.subTest(history=history):
                    response = self.client.post(
                        "/api/chat", json={"message": "Còn VNM?", "history": history}
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json["error"]["code"], "invalid_request")
        chat.assert_not_called()

    def test_api_rejects_invalid_conversation_state(self):
        valid = {
            "active_symbols": ["FPT"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": [],
        }
        cases = (
            None,
            {"active_symbols": []},
            {**valid, "extra": True},
            {**valid, "topic": "unknown"},
            {**valid, "active_symbols": ["FPT", "VNM", "HPG"]},
            {**valid, "last_result_symbols": ["FPT"] * 11},
            {**valid, "ranking_order": "random"},
        )
        with patch.object(web_app.chatbot_service, "chat") as chat:
            for state in cases:
                with self.subTest(state=state):
                    response = self.client.post(
                        "/api/chat",
                        json={"message": "Còn nó?", "conversation_state": state},
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json["error"]["code"], "invalid_request")
        chat.assert_not_called()

    def test_api_trims_valid_request_and_returns_success_contract(self):
        history = [
            {"role": "user", "content": " Phân tích FPT "},
            {"role": "assistant", "content": " Kết quả cũ "},
        ]
        expected = {
            "answer": "Kết quả mới",
            "sources": [{"kind": "stock_signal", "symbols": ["VNM"]}],
            "warnings": [],
            "release_status": "current",
            "data_as_of": "2026-07-20",
            "conversation_state": {
                "active_symbols": ["VNM"],
                "topic": "signal",
                "ranking_order": None,
                "last_result_symbols": ["VNM"],
            },
        }
        state = {
            "active_symbols": [" vnm "],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": ["vnm"],
        }
        with patch.object(
            web_app.chatbot_service, "chat", return_value=expected
        ) as chat:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": " Còn VNM? ",
                    "history": history,
                    "conversation_state": state,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, expected)
        chat.assert_called_once_with(
            "Còn VNM?",
            [
                {"role": "user", "content": "Phân tích FPT"},
                {"role": "assistant", "content": "Kết quả cũ"},
            ],
            {
                "active_symbols": ["VNM"],
                "topic": "signal",
                "ranking_order": None,
                "last_result_symbols": ["VNM"],
            },
        )

    def test_api_without_state_remains_backward_compatible(self):
        with patch.object(
            web_app.chatbot_service, "chat", return_value={"answer": "Chào bạn."}
        ) as chat:
            response = self.client.post("/api/chat", json={"message": " Chào "})

        self.assertEqual(response.status_code, 200)
        chat.assert_called_once_with("Chào", [], None)

    def test_api_maps_service_statuses_and_hides_internal_errors(self):
        cases = (
            ("provider_error", 502),
            ("llm_not_configured", 503),
            ("provider_timeout", 504),
        )
        for code, status in cases:
            with self.subTest(code=code), patch.object(
                web_app.chatbot_service,
                "chat",
                side_effect=chatbot_service.ChatbotServiceError(
                    code, "Không thể kết nối trợ lý lúc này.", status
                ),
            ):
                response = self.client.post("/api/chat", json={"message": "FPT"})
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json["error"]["code"], code)

        with patch.object(
            web_app.chatbot_service,
            "chat",
            side_effect=RuntimeError(r"secret C:\private\artifact.pkl"),
        ):
            response = self.client.post("/api/chat", json={"message": "FPT"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["error"]["code"], "internal_error")
        self.assertNotIn("private", response.get_data(as_text=True))

    def test_missing_llm_config_returns_503_without_breaking_flask(self):
        with (
            patch.multiple(
                web_app.chatbot_service,
                LLM_BASE_URL="",
                LLM_API_KEY="",
                LLM_MODEL="",
            ),
            patch.object(
                web_app.chatbot_service,
                "build_context",
                return_value=context_bundle(),
            ),
        ):
            response = self.client.post("/api/chat", json={"message": "FPT"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"]["code"], "llm_not_configured")


class ChatbotTemplateTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_chat_page_contains_accessible_local_only_contract(self):
        with patch.object(
            web_app, "load_selected_model_from_report", return_value="Random Forest"
        ):
            response = self.client.get("/chat")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Trợ lý dữ liệu và mô hình HOSE", html)
        self.assertIn('id="chat-transcript"', html)
        self.assertIn('<label for="chat-message">', html)
        self.assertIn('maxlength="1000"', html)
        self.assertIn('id="send-chat"', html)
        self.assertIn('id="clear-chat"', html)
        self.assertEqual(html.count("data-prompt="), 3)
        self.assertIn("Dữ liệu offline", html)
        self.assertIn("Random Forest", html)
        self.assertIn("không phải khuyến nghị đầu tư", html)
        self.assertIn("theme.js", html)

    def test_shared_client_uses_tab_session_bounded_history_and_safe_dom(self):
        base = (ROOT_DIR / "templates" / "base.html").read_text(encoding="utf-8")
        page = (ROOT_DIR / "templates" / "chat.html").read_text(encoding="utf-8")
        client = (ROOT_DIR / "static" / "chat-client.js").read_text(encoding="utf-8")

        self.assertEqual(base.count("chat-client.js"), 1)
        self.assertNotIn("chat-client.js", page)
        self.assertIn("window.ChatClientKit", page)
        self.assertIn('SESSION_KEY = "hose-chat-session-v1"', client)
        self.assertIn("MAX_TRANSCRIPT = 40", client)
        self.assertIn("window.sessionStorage.getItem(SESSION_KEY)", client)
        self.assertIn("window.sessionStorage.setItem(SESSION_KEY", client)
        self.assertIn("window.sessionStorage.removeItem(SESSION_KEY)", client)
        self.assertIn("session.transcript.slice(-6)", client)
        self.assertIn("payload.conversation_state = session.conversation_state", client)
        self.assertIn("AbortController", client)
        self.assertIn("70000", client)
        self.assertIn("if (busy)", page)
        self.assertIn('aria-busy', page)
        self.assertIn("result.focus()", page)
        self.assertIn("clear.disabled = value", page)
        combined = "\n".join((base, page, client))
        self.assertIn("document.createElement", combined)
        self.assertIn("textContent", combined)
        self.assertNotIn("innerHTML", combined)

    def test_sources_release_warnings_are_separate_and_llm_html_stays_text(self):
        source = (ROOT_DIR / "templates" / "chat.html").read_text(encoding="utf-8")
        client = (ROOT_DIR / "static" / "chat-client.js").read_text(encoding="utf-8")
        for element_id in ("chat-sources", "chat-release", "chat-warnings"):
            self.assertIn(f'id="{element_id}"', source)
        self.assertIn("kit.buildSourceDetails", source)
        self.assertIn("release_status", client)
        self.assertIn("warnings", client)

        payload = '<img src=x onerror="alert(1)">'
        with patch.object(
            web_app.chatbot_service,
            "chat",
            return_value={
                "answer": payload,
                "sources": [],
                "warnings": [],
                "release_status": "current",
            },
        ):
            response = self.client.post("/api/chat", json={"message": "FPT"})
        self.assertEqual(response.json["answer"], payload)
        self.assertNotIn("innerHTML", source + client)

    def test_five_primary_pages_link_to_chat_without_new_asset_files(self):
        # Sau redesign, nav (gồm link Trợ lý) sống trong base.html dùng chung,
        # không lặp trong từng template con. Kiểm bằng HTML render thật: mọi
        # trang chính đều dẫn tới /chat qua shell chung, và không thêm asset.
        meta = {
            "dataset_max_date": "2026-07-10",
            "symbol_count": 3,
            "symbols": ("FPT", "VNM", "TCD"),
        }
        routes = {
            "/": {},
            "/compare": {},
            "/screener": {"predict_all_symbols": {"return_value": []}},
            "/evaluation": {
                "load_evaluation_sections": {"return_value": {"baseline_warning": None}}
            },
            "/tuning": {
                "compute_dataset_fingerprint": {
                    "return_value": {"hash": "fp", "parts": {}}
                },
                "read_manual_config": {
                    "return_value": {
                        "dataset_fingerprint": "fp",
                        "selected_models": {},
                        "cv_settings": {"n_splits": 4, "gap": 5},
                    }
                },
                "read_history": {"return_value": []},
                "mark_best": {"side_effect": lambda rows: rows},
                "is_config_complete": {"return_value": False},
                "has_evaluated_snapshot": {"return_value": False},
                "is_pipeline_running": {"return_value": False},
                "is_fetch_running": {"return_value": False},
            },
        }
        chat_href = "/chat"
        for path, extra_patches in routes.items():
            with self.subTest(path=path):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.object(web_app, "get_dataset_meta", return_value=meta)
                    )
                    stack.enter_context(
                        patch.object(
                            web_app,
                            "load_selected_model_from_report",
                            return_value="Random Forest",
                        )
                    )
                    for target, kwargs in extra_patches.items():
                        stack.enter_context(patch.object(web_app, target, **kwargs))
                    html = self.client.get(path).get_data(as_text=True)
                self.assertIn('data-nav="chat"', html)
                self.assertIn(f'href="{chat_href}"', html)
                self.assertIn("Trợ lý", html)
        self.assertFalse((ROOT_DIR / "static" / "chat.css").exists())
        self.assertFalse((ROOT_DIR / "static" / "chat.js").exists())

    def test_shared_css_has_chat_mobile_layout(self):
        css = (ROOT_DIR / "static" / "app.css").read_text(encoding="utf-8")
        self.assertIn(".chat-transcript", css)
        self.assertIn(".chat-compose", css)
        self.assertRegex(css, r"@media\s*\(max-width:\s*760px\)")


if __name__ == "__main__":
    unittest.main()
