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


def decision_json(action, arguments, direct_answer=None):
    return json.dumps(
        {
            "action": action,
            "arguments": arguments,
            "direct_answer": direct_answer,
        },
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
                {"symbols": [" fpt "], "focus": "prediction"},
                None,
                {"symbols": ["FPT"], "focus": "prediction"},
            ),
            (
                "STOCK_RANKING",
                {"order": "lowest", "top_n": 3},
                None,
                {"order": "lowest", "top_n": 3},
            ),
            ("PROJECT_INFO", {"topic": "model"}, None, {"topic": "model"}),
            ("OUT_OF_SCOPE", {"reason": "news"}, None, {"reason": "news"}),
            ("GENERAL_CHAT", {}, "Bạn muốn xem mã nào?", {}),
        )
        for action, arguments, answer, normalized in cases:
            with self.subTest(action=action):
                client = fake_client(
                    provider_response(decision_json(action, arguments, answer))
                )
                result = chatbot_service._decide(
                    "message", [], client, monotonic=Mock(side_effect=[0, 0, 0])
                )
                self.assertEqual(result["arguments"], normalized)
                self.assertEqual(result["direct_answer"], answer)
                client.chat.completions.create.assert_called_once()

    def test_rejects_malformed_extra_unknown_wrong_types_and_invalid_symbols(self):
        invalid = (
            "not json",
            "```json\n{}\n```",
            decision_json("UNKNOWN", {}),
            json.dumps(
                {
                    "action": "GENERAL_CHAT",
                    "arguments": {},
                    "direct_answer": "Chào",
                    "extra": True,
                }
            ),
            decision_json("STOCK_RANKING", {"order": "highest", "top_n": True}),
            decision_json(
                "STOCK_SIGNAL", {"symbols": ["FPT", "fpt"], "focus": "analysis"}
            ),
            decision_json(
                "STOCK_SIGNAL", {"symbols": ["FPT"], "focus": "comparison"}
            ),
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

    def test_provider_request_has_one_call_six_history_messages_and_no_tools(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
            for index in range(8)
        ]
        client = fake_client(
            provider_response(decision_json("GENERAL_CHAT", {}, "Chào bạn."))
        )
        chatbot_service.chat("Chào", history, client=client)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][1:-1], history[-6:])
        for key in ("tools", "tool_choice", "response_format"):
            self.assertNotIn(key, request)
        self.assertNotIn("CONTEXT_JSON", request["messages"][0]["content"])


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
            "scope": {"FPT", "VNM", "MWG"},
            "warnings": [],
            "model_trained_through": "2026-04-10",
        }

    def test_symbol_scope_is_checked_before_inference(self):
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(chatbot_tools.prediction_service, "predict_symbols") as predict,
        ):
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL", {"symbols": ["ABC"], "focus": "prediction"}
            )
        predict.assert_not_called()
        self.assertEqual(result["data"]["unsupported_symbols"], ["ABC"])
        self.assertIsNone(result["error"])

    def test_stock_focus_and_comparison_use_one_batch_and_raw_score(self):
        rows = [
            {
                "symbol": "FPT",
                "probability_up": 0.50004,
                "decision_threshold": 0.49,
                "reference_date": "2026-07-20",
                "prediction": "UP",
                "is_stale": False,
            },
            {
                "symbol": "VNM",
                "probability_up": 0.50001,
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
                {"symbols": ["FPT", "VNM"], "focus": "comparison"},
            )
        predict.assert_called_once_with(["FPT", "VNM"])
        self.assertEqual(result["data"]["focus"], "comparison")
        self.assertEqual(result["data"]["comparison"]["higher_up_score_symbol"], "FPT")

    def test_different_reference_dates_do_not_produce_comparison_gap(self):
        rows = [
            {"symbol": "FPT", "probability_up": 0.7, "reference_date": "2026-07-20"},
            {"symbol": "VNM", "probability_up": 0.3, "reference_date": "2026-07-19"},
        ]
        with (
            patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            patch.object(chatbot_tools.prediction_service, "predict_symbols", return_value=rows),
        ):
            result = chatbot_tools.execute_action(
                "STOCK_SIGNAL",
                {"symbols": ["FPT", "VNM"], "focus": "comparison"},
            )
        self.assertFalse(result["data"]["comparison"]["available"])
        self.assertNotIn("up_score_gap_percent_points", result["data"]["comparison"])
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

    def test_runtime_readiness_and_legacy_scope_fallback_are_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            eligible_path = root / "eligible.csv"
            metadata_path.write_text('{"model_name":"Random Forest"}', encoding="utf-8")
            pd.DataFrame({"symbol": ["fpt", "VNM"]}).to_csv(eligible_path, index=False)
            with (
                patch.object(chatbot_tools, "MODEL_METADATA_PATH", metadata_path),
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

        with patch.object(
            chatbot_tools.experiment_state, "is_pipeline_running", return_value=True
        ):
            unavailable = chatbot_tools.execute_action(
                "PROJECT_INFO", {"topic": "overview"}
            )
        self.assertEqual(unavailable["error"]["code"], "model_unavailable")
        self.assertNotIn("path", json.dumps(unavailable))

    def test_six_project_topics_and_dates_keep_distinct_meanings(self):
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
                patch.object(chatbot_tools, "_load_runtime_state", return_value=self.state),
            ):
                results = {
                    topic: chatbot_tools.execute_action("PROJECT_INFO", {"topic": topic})
                    for topic in (
                        "overview", "model", "dataset", "features", "method", "limitations"
                    )
                }
        self.assertTrue(all(result["error"] is None for result in results.values()))
        self.assertEqual(results["dataset"]["data_as_of"], "2026-07-20")
        self.assertEqual(results["dataset"]["model_trained_through"], "2026-04-10")
        self.assertEqual(results["features"]["data"]["scope"], "global_model_importance")
        self.assertIn("sections", results["method"]["data"])

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
    def test_formatter_handles_four_stock_focuses_without_provider(self):
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
        for focus in ("info", "prediction", "analysis", "comparison"):
            decision = {
                "action": "STOCK_SIGNAL",
                "arguments": {"symbols": ["FPT"], "focus": focus},
                "direct_answer": None,
            }
            with patch.object(chatbot_service, "_provider_call") as provider:
                response = chatbot_service._format_response(
                    decision, domain_result({"focus": focus, "signals": [signal]})
                )
            provider.assert_not_called()
            self.assertIn("FPT", response["answer"])
            self.assertNotIn("123.456", response["answer"])
            self.assertTrue(response["answer"].endswith(chatbot_service.STOCK_DISCLAIMER))

    def test_general_advice_out_of_scope_and_symbol_warning_are_deterministic(self):
        general = {"action": "GENERAL_CHAT", "arguments": {}, "direct_answer": "Chào bạn."}
        self.assertEqual(chatbot_service._format_response(general)["answer"], "Chào bạn.")
        advice = {**general, "direct_answer": "Bạn nên mua FPT ngay."}
        self.assertEqual(
            chatbot_service._format_response(advice)["answer"],
            chatbot_service.OUT_OF_SCOPE_MESSAGES["trading_advice"],
        )
        news = {
            "action": "OUT_OF_SCOPE",
            "arguments": {"reason": "news"},
            "direct_answer": None,
        }
        self.assertEqual(
            chatbot_service._format_response(news)["answer"],
            chatbot_service.OUT_OF_SCOPE_MESSAGES["news"],
        )
        stock = {
            "action": "STOCK_SIGNAL",
            "arguments": {"symbols": ["ABC"], "focus": "prediction"},
            "direct_answer": None,
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
            ("GENERAL_CHAT", {}, "Chào bạn.", None),
            ("STOCK_SIGNAL", {"symbols": ["FPT"], "focus": "prediction"}, None, domain_result({"signals": []})),
            ("STOCK_RANKING", {"order": "highest", "top_n": 1}, None, domain_result({"ranking": []})),
            ("PROJECT_INFO", {"topic": "overview"}, None, domain_result({"topic": "overview"})),
            ("OUT_OF_SCOPE", {"reason": "news"}, None, None),
        )
        for action, arguments, answer, action_result in cases:
            with self.subTest(action=action):
                client = fake_client(
                    provider_response(decision_json(action, arguments, answer))
                )
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
                "Còn mã đó thì sao?",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "STOCK_SIGNAL",
                {"symbols": ["FPT"], "focus": "analysis"},
            ),
            (
                "So với MWG?",
                [{"role": "user", "content": "Dự đoán FPT"}, {"role": "assistant", "content": "FPT: UP"}],
                "STOCK_SIGNAL",
                {"symbols": ["FPT", "MWG"], "focus": "comparison"},
            ),
            (
                "Còn model thì sao?",
                [{"role": "user", "content": "Dataset dùng gì?"}, {"role": "assistant", "content": "Dữ liệu offline."}],
                "PROJECT_INFO",
                {"topic": "model"},
            ),
        )
        for message, history, action, arguments in cases:
            client = fake_client(
                provider_response(decision_json(action, arguments))
            )
            result = domain_result({"signals": []} if action == "STOCK_SIGNAL" else {"topic": "model"})
            with patch.object(
                chatbot_service.chatbot_tools, "execute_action", return_value=result
            ) as execute:
                chatbot_service.chat(message, history, client=client)
            execute.assert_called_once_with(action, arguments)
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
        cases = (
            {"data": "{}", "content_type": "text/plain"},
            {"data": "{bad", "content_type": "application/json"},
            {"json": {"message": "FPT", "conversation_state": {}}},
            {"json": {"message": "   "}},
            {"json": {"message": "x" * 1001}},
            {"json": {"message": "FPT", "history": [{"role": "user", "content": "odd"}]}},
            {"json": {"message": "FPT", "history": [{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}]}},
        )
        with patch.object(web_app.chatbot_service, "chat") as chat:
            for payload in cases:
                response = self.client.post("/api/chat", **payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["error"]["code"], "invalid_request")
        chat.assert_not_called()

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
    def test_single_accessible_page_loads_chat_assets_locally(self):
        base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        page = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertNotIn("chat-client.js", base)
        self.assertNotIn("chat-ui.css", base)
        self.assertNotIn("chat-dock", base)
        self.assertEqual(page.count("chat-client.js"), 1)
        self.assertEqual(page.count("chat-ui.css"), 1)
        for contract in ('id="chat-form"', 'id="chat-transcript"', 'aria-live="polite"'):
            self.assertIn(contract, page)

    def test_client_uses_safe_dom_short_history_and_no_removed_protocol(self):
        page = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        client = (ROOT / "static" / "chat-client.js").read_text(encoding="utf-8")
        combined = page + client
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
