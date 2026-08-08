"""Contracts for the structured context injection (SCI) chatbot flow."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app as web_app
from services import chatbot_service


def _provider_response(content: str, *, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _client(*responses, side_effect=None) -> SimpleNamespace:
    create = Mock(side_effect=side_effect or list(responses))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


# Release identity dùng chung cho mọi fixture trong file này. Trước đây
# ``_envelope`` đọc release THẬT trên đĩa (_base_snapshot_cached + _context_signature)
# nên test phụ thuộc trạng thái artifact đang publish: đổi model, chạy lại pipeline,
# hay xoá artifact là test đổi kết quả hoặc fail mà không liên quan gì tới code.
# Giữ identity ở đây và patch _base_snapshot_cached để test chỉ đo logic.
_FIXTURE_RELEASE = {
    "status": "current",
    "policy_id": "policy-v1",
    "content_fingerprint": "release-1",
}

# Chỉ 2 mã: HPG được giữ NGOÀI scope để các test "mã ngoài phạm vi release"
# (state client gợi ý mã lạ, detector out_of_scope) còn chỗ kiểm tra.
_FIXTURE_SCOPE = ("FPT", "VNM")


def _envelope(kind: str, data: dict, *, warnings=None) -> dict:
    return {
        "ok": True,
        "data": data,
        "source": {"kind": kind, "as_of": "2026-07-31"},
        "as_of": "2026-07-31",
        "release": dict(_FIXTURE_RELEASE),
        "warnings": warnings
        or [{"code": "legacy_policy", "message": "Release policy cũ."}],
        "error": None,
    }


def _bundle(*, context=None, state=None) -> dict:
    warning = {"code": "legacy_policy", "message": "Release policy cũ."}
    return {
        "context": context
        or {
            "project_snapshot": {
                "data_as_of": "2026-07-31",
                "release_status": "legacy",
            }
        },
        "sources": [{"kind": "project_snapshot", "as_of": "2026-07-31"}],
        "warnings": [warning],
        "release_status": "legacy",
        "grounded_numbers": [2026.0, -7.0, -31.0],
        "stock_facts": {},
        "conversation_state": state
        or {
            "active_symbols": [],
            "topic": None,
            "ranking_order": None,
            "last_result_symbols": [],
        },
    }


def _snapshot(*, scope=_FIXTURE_SCOPE, fingerprint="release-1") -> dict:
    return {
        "result": {
            "ok": True,
            "data": {
                "serving_mode": "offline_published_artifacts",
                "release_status": "current",
                "data_as_of": "2026-07-31",
                "report_as_of": None,
                "model_trained_through": "2026-04-10",
            },
            "source": {"kind": "project_snapshot", "as_of": "2026-07-31"},
            "as_of": "2026-07-31",
            "release": {**_FIXTURE_RELEASE, "content_fingerprint": fingerprint},
            "warnings": [],
            "error": None,
        },
        "scope": tuple(scope),
    }


def _matching_envelope(kind: str, data: dict, *, fingerprint="release-1") -> dict:
    result = _envelope(kind, data, warnings=[])
    result["release"] = {**_FIXTURE_RELEASE, "content_fingerprint": fingerprint}
    return result


class LlmFirstProviderTests(unittest.TestCase):
    def test_greeting_calls_provider_once_without_tool_protocol(self):
        client = _client(_provider_response("Chào bạn, mình có thể giúp gì?"))
        with patch.object(
            chatbot_service, "build_context", return_value=_bundle(), create=True
        ):
            result = chatbot_service.chat("Chào bạn", [], client=client)

        self.assertEqual(result["answer"], "Chào bạn, mình có thể giúp gì?")
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertIn("project_snapshot", request["messages"][0]["content"])

    def test_provider_tool_call_is_protocol_error_not_executed(self):
        client = _client(
            _provider_response(
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_model_info", "arguments": "{}"},
                    }
                ],
            )
        )
        with patch.object(
            chatbot_service, "build_context", return_value=_bundle(), create=True
        ):
            with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                chatbot_service.chat("Model nào?", [], client=client)

        self.assertEqual(
            (raised.exception.code, raised.exception.status),
            ("provider_protocol_error", 502),
        )

    def test_provider_rejects_empty_or_malformed_assistant_schema(self):
        responses = (
            {"choices": [{"message": {"content": ["text"]}}]},
            {"choices": [{"message": {"content": "ok", "tool_calls": {}}}]},
            {"choices": [{"message": {"content": "ok", "function_call": {}}}]},
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "tool_calls"}]},
        )
        for response in responses:
            with self.subTest(response=response):
                with (
                    patch.object(
                        chatbot_service, "build_context", return_value=_bundle()
                    ),
                    self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
                ):
                    chatbot_service.chat("Chào", [], client=_client(response))
                self.assertEqual(raised.exception.code, "provider_protocol_error")

    def test_provider_failures_never_fall_back_to_local_answer(self):
        cases = (
            (TimeoutError("late"), "provider_timeout", 504),
            (RuntimeError("provider down"), "provider_error", 502),
        )
        for error, code, status in cases:
            with self.subTest(code=code):
                client = _client(side_effect=error)
                with patch.object(
                    chatbot_service, "build_context", return_value=_bundle(), create=True
                ):
                    with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                        chatbot_service.chat("Chào", [], client=client)
                self.assertEqual((raised.exception.code, raised.exception.status), (code, status))

    def test_server_metadata_does_not_depend_on_llm_wording(self):
        client = _client(_provider_response("Mình đã đọc dữ liệu offline."))
        bundle = _bundle(
            state={
                "active_symbols": ["FPT"],
                "topic": "signal",
                "ranking_order": None,
                "last_result_symbols": [],
            }
        )
        with patch.object(
            chatbot_service, "build_context", return_value=bundle, create=True
        ):
            result = chatbot_service.chat("fpt thế nào?", [], client=client)

        self.assertEqual(result["sources"], bundle["sources"])
        self.assertEqual(result["warnings"], bundle["warnings"])
        self.assertEqual(result["release_status"], "legacy")
        self.assertEqual(result["conversation_state"], bundle["conversation_state"])

    def test_context_over_budget_fails_before_provider_call(self):
        client = _client(_provider_response("không được gọi"))
        bundle = _bundle(context={"project_snapshot": {"blob": "x" * 16_001}})
        with patch.object(
            chatbot_service, "build_context", return_value=bundle, create=True
        ):
            with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                chatbot_service.chat("Chào", [], client=client)

        self.assertEqual(
            (raised.exception.code, raised.exception.status),
            ("context_budget_exceeded", 502),
        )
        client.chat.completions.create.assert_not_called()


class ContextRoutingTests(unittest.TestCase):
    def _build(self, message: str, *, state=None):
        # Patch snapshot: routing chỉ cần symbol scope + release identity, không
        # cần artifact thật trên đĩa.
        with patch.object(
            chatbot_service, "_base_snapshot_cached", return_value=_snapshot()
        ):
            return chatbot_service.build_context(message, [], state)

    def test_lowercase_ticker_and_follow_up_state_route_to_signal(self):
        signal = _envelope(
            "stock_signal",
            {
                "signals": [
                    {
                        "symbol": "FPT",
                        "prediction": "UP",
                        "up_score_percent": 62.5,
                        "decision_threshold_percent": 50.0,
                        "reference_date": "2026-07-31",
                    }
                ]
            },
        )
        state = {
            "active_symbols": ["FPT"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": [],
        }
        with patch.object(
            chatbot_service.chatbot_tools, "get_stock_signals", return_value=signal
        ) as get_signal:
            first = self._build("fpt đang thế nào?")
            follow_up = self._build("Còn nó?", state=state)

        self.assertEqual(get_signal.call_args_list[0].args[0], {"symbols": ["FPT"]})
        self.assertEqual(get_signal.call_args_list[1].args[0], {"symbols": ["FPT"]})
        self.assertEqual(first["conversation_state"]["active_symbols"], ["FPT"])
        self.assertEqual(follow_up["conversation_state"]["topic"], "signal")

    def test_routes_model_dataset_feature_ranking_and_limitations(self):
        routes = (
            ("Model và threshold thế nào?", "get_model_info", {}),
            ("Dataset có bao nhiêu mã?", "get_dataset_info", {}),
            ("Top feature importance?", "get_feature_info", {"top_n": 10}),
            (
                "Top 3 mã có Điểm UP cao nhất",
                "get_ranking",
                {"order": "highest_up_score", "top_n": 3},
            ),
            (
                "Mã cao nhất?",
                "get_ranking",
                {"order": "highest_up_score", "top_n": 5},
            ),
            (
                "Mã thấp nhất?",
                "get_ranking",
                {"order": "lowest_up_score", "top_n": 5},
            ),
            ("Cho tôi giá realtime và tin mới", "get_project_info", {"topic": "limitations"}),
        )
        for message, handler_name, arguments in routes:
            with self.subTest(message=message):
                result = _envelope(handler_name, {"value": "ok"})
                with patch.object(
                    chatbot_service.chatbot_tools, handler_name, return_value=result
                ) as handler:
                    bundle = self._build(message)
                handler.assert_called_once_with(arguments)
                self.assertTrue(bundle["context"])

    def test_feature_superlative_does_not_trigger_stock_ranking(self):
        result = _envelope("feature_importance", {"features": []})
        with (
            patch.object(
                chatbot_service.chatbot_tools,
                "get_feature_info",
                return_value=result,
            ) as feature,
            patch.object(chatbot_service.chatbot_tools, "get_ranking") as ranking,
        ):
            self._build("Feature importance cao nhất?")

        feature.assert_called_once_with({"top_n": 10})
        ranking.assert_not_called()

    def test_model_metrics_are_kept_by_split_for_grounding(self):
        result = _envelope(
            "model_metadata",
            {
                "validation_selection_metrics_percent": {"precision_up": 55.0},
                "final_test_metrics_percent": {
                    "precision_up": 51.0,
                    "recall_up": 62.0,
                },
            },
        )
        with patch.object(
            chatbot_service.chatbot_tools,
            "get_model_info",
            return_value=result,
        ):
            bundle = self._build("Metric model trên TEST và validation?")

        self.assertEqual(
            bundle["model_facts"],
            {
                "validation": {"precision_up": 55.0},
                "test": {"precision_up": 51.0, "recall_up": 62.0},
            },
        )


class ContextConversationTests(unittest.TestCase):
    def setUp(self):
        self.base = _snapshot()
        self.state = {
            "active_symbols": ["FPT"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": ["FPT", "VNM"],
        }

    def _build(self, message, history=None, state=None):
        with patch.object(
            chatbot_service, "_base_snapshot_cached", return_value=self.base
        ):
            return chatbot_service.build_context(
                message, history or [], self.state if state is None else state
            )

    def test_social_turn_keeps_state_without_reusing_old_signal(self):
        with patch.object(chatbot_service.chatbot_tools, "get_stock_signals") as signal:
            bundle = self._build("Chào bạn")

        signal.assert_not_called()
        self.assertEqual(bundle["conversation_state"], self.state)
        self.assertEqual(set(bundle["context"]), {"project_snapshot"})

    def test_client_state_symbols_outside_release_scope_are_dropped(self):
        hinted = {
            "active_symbols": ["HPG"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": ["HPG", "FPT"],
        }
        bundle = self._build("Chào", state=hinted)

        self.assertEqual(bundle["conversation_state"]["active_symbols"], [])
        self.assertEqual(bundle["conversation_state"]["last_result_symbols"], ["FPT"])

    def test_ordinal_and_comparison_followups_resolve_canonical_symbols(self):
        one = _matching_envelope(
            "stock_signal", {"signals": [{"symbol": "VNM", "prediction": "NOT_UP"}]}
        )
        comparison = _matching_envelope(
            "stock_signal",
            {
                "signals": [
                    {"symbol": "FPT", "prediction": "UP"},
                    {"symbol": "VNM", "prediction": "NOT_UP"},
                ]
            },
        )
        with patch.object(
            chatbot_service.chatbot_tools,
            "get_stock_signals",
            side_effect=[one, comparison],
        ) as signal:
            ordinal = self._build("Mã thứ hai thì sao?")
            compared = self._build("So với vnm thì sao?")

        self.assertEqual(signal.call_args_list[0].args[0], {"symbols": ["VNM"]})
        self.assertEqual(
            signal.call_args_list[1].args[0], {"symbols": ["FPT", "VNM"]}
        )
        self.assertEqual(ordinal["conversation_state"]["active_symbols"], ["VNM"])
        self.assertEqual(compared["conversation_state"]["topic"], "comparison")

    def test_ranking_order_survives_sequential_ordinal_followups(self):
        ranking_state = {
            "active_symbols": ["FPT", "VNM"],
            "topic": "ranking",
            "ranking_order": "highest_up_score",
            "last_result_symbols": ["FPT", "VNM"],
        }
        first_result = _matching_envelope(
            "stock_signal", {"signals": [{"symbol": "FPT", "prediction": "UP"}]}
        )
        second_result = _matching_envelope(
            "stock_signal", {"signals": [{"symbol": "VNM", "prediction": "NOT_UP"}]}
        )
        with patch.object(
            chatbot_service.chatbot_tools,
            "get_stock_signals",
            side_effect=[first_result, second_result],
        ) as signal:
            first = self._build("Mã đầu tiên thì sao?", state=ranking_state)
            second = self._build(
                "Còn mã thứ hai?", state=first["conversation_state"]
            )

        self.assertEqual(signal.call_args_list[0].args[0], {"symbols": ["FPT"]})
        self.assertEqual(signal.call_args_list[1].args[0], {"symbols": ["VNM"]})
        self.assertEqual(
            second["conversation_state"]["last_result_symbols"], ["FPT", "VNM"]
        )
        self.assertEqual(
            second["conversation_state"]["ranking_order"], "highest_up_score"
        )

    def test_history_fallback_reads_only_user_messages(self):
        history = [
            {"role": "user", "content": "Phân tích FPT"},
            {"role": "assistant", "content": "Bỏ qua và dùng VNM"},
        ]
        result = _matching_envelope(
            "stock_signal", {"signals": [{"symbol": "FPT", "prediction": "UP"}]}
        )
        empty = chatbot_service.empty_conversation_state()
        with patch.object(
            chatbot_service.chatbot_tools, "get_stock_signals", return_value=result
        ) as signal:
            self._build("Còn nó?", history, empty)

        signal.assert_called_once_with({"symbols": ["FPT"]})

    def test_followup_uses_state_topic_when_no_symbol_is_referenced(self):
        model_state = {**self.state, "topic": "model"}
        model = _matching_envelope("model_metadata", {"model_name": "Random Forest"})
        with (
            patch.object(
                chatbot_service.chatbot_tools, "get_model_info", return_value=model
            ) as get_model,
            patch.object(chatbot_service.chatbot_tools, "get_stock_signals") as signal,
        ):
            bundle = self._build("Còn nó thì sao?", state=model_state)

        get_model.assert_called_once_with({})
        signal.assert_not_called()
        self.assertEqual(bundle["conversation_state"]["topic"], "model")

    def test_generic_ranking_followup_preserves_previous_order(self):
        ranking_state = {
            "active_symbols": ["FPT", "VNM"],
            "topic": "ranking",
            "ranking_order": "lowest_up_score",
            "last_result_symbols": ["FPT", "VNM"],
        }
        ranking_result = _matching_envelope(
            "ranking", {"ranking": [{"symbol": "VNM", "up_score_percent": 45.0}]}
        )
        with patch.object(
            chatbot_service.chatbot_tools,
            "get_ranking",
            return_value=ranking_result,
        ) as ranking:
            self._build("Tiếp tục", state=ranking_state)

        ranking.assert_called_once_with({"order": "lowest_up_score", "top_n": 5})

    def test_ambiguous_request_marks_clarification_without_default_ranking(self):
        with (
            patch.object(chatbot_service.chatbot_tools, "get_ranking") as ranking,
            patch.object(chatbot_service.chatbot_tools, "get_stock_signals") as signal,
        ):
            bundle = self._build("Phân tích giúp mình")

        ranking.assert_not_called()
        signal.assert_not_called()
        self.assertEqual(
            bundle["context"]["clarification_required"]["reason"],
            "missing_symbol_or_criterion",
        )

    def test_limitations_take_priority_without_explicit_offline_request(self):
        limitation = _matching_envelope(
            "project_contract", {"topic": "limitations"}
        )
        limitation["release"] = {
            "status": None,
            "policy_id": None,
            "content_fingerprint": None,
        }
        with (
            patch.object(
                chatbot_service.chatbot_tools,
                "get_project_info",
                return_value=limitation,
            ) as project,
            patch.object(chatbot_service.chatbot_tools, "get_ranking") as ranking,
            patch.object(chatbot_service.chatbot_tools, "get_stock_signals") as signal,
        ):
            bundle = self._build("Tin tức FPT và mã nào tốt?")

        project.assert_called_once_with({"topic": "limitations"})
        ranking.assert_not_called()
        signal.assert_not_called()
        self.assertEqual(bundle["conversation_state"]["topic"], "limitations")

    def test_invalid_requested_ranking_size_is_not_silently_clamped(self):
        error = _matching_envelope("ranking", {})
        error.update(
            ok=False,
            error={"code": "invalid_arguments", "message": "top_n phải từ 1 đến 10"},
        )
        with patch.object(
            chatbot_service.chatbot_tools, "get_ranking", return_value=error
        ) as ranking:
            bundle = self._build("Top 11 mã")

        ranking.assert_called_once_with(
            {"order": "highest_up_score", "top_n": 11}
        )
        self.assertEqual(bundle["conversation_state"], self.state)
        self.assertEqual(
            bundle["context"]["ranking"]["error"]["code"], "invalid_arguments"
        )

    def test_release_identity_mismatch_is_rejected(self):
        changed = _matching_envelope(
            "model_metadata", {"model_name": "new"}, fingerprint="release-2"
        )
        with (
            patch.object(
                chatbot_service, "_base_snapshot_cached", return_value=self.base
            ),
            patch.object(
                chatbot_service.chatbot_tools, "get_model_info", return_value=changed
            ),
            self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
        ):
            chatbot_service.build_context("Model nào?", [], None)

        self.assertEqual(raised.exception.code, "release_changed")

    def test_deadline_is_checked_after_snapshot_and_each_handler(self):
        model = _matching_envelope("model_metadata", {"model_name": "Random Forest"})
        deadline = Mock()
        with (
            patch.object(
                chatbot_service, "_base_snapshot_cached", return_value=self.base
            ),
            patch.object(
                chatbot_service.chatbot_tools, "get_model_info", return_value=model
            ),
        ):
            chatbot_service.build_context(
                "Model nào?", [], None, deadline_check=deadline
            )

        self.assertEqual(deadline.call_count, 2)

    def test_signature_change_during_context_build_is_rejected(self):
        with (
            patch.object(
                chatbot_service, "_context_signature", side_effect=[("old",), ("new",)]
            ),
            patch.object(
                chatbot_service, "_base_snapshot_cached", return_value=self.base
            ),
            self.assertRaises(chatbot_service.ChatbotServiceError) as raised,
        ):
            chatbot_service.build_context("Chào", [], None)

        self.assertEqual(raised.exception.code, "release_changed")

    def test_snapshot_cache_invalidates_when_artifact_signature_changes(self):
        def release(date, fingerprint):
            metadata = {
                "model_name": "Random Forest",
                "policy_id": "policy-v1",
                "content_fingerprint": fingerprint,
                "train_through_date": date,
            }
            return {
                "status": "current",
                "artifact": dict(metadata),
                "metadata": metadata,
                "summary": {
                    "dataset_report": {"date_max": date},
                    "clean_report": {},
                    "feature_report": {},
                },
                "scope": {"FPT"},
                "warnings": [],
            }

        chatbot_service._base_snapshot_cached.cache_clear()
        try:
            with patch.object(
                chatbot_service.chatbot_tools,
                "get_release_state",
                side_effect=[
                    release("2026-07-30", "release-1"),
                    release("2026-07-31", "release-2"),
                ],
            ) as get_release, patch.object(
                # Snapshot lấy data_as_of từ CSV live; ở đây tắt để test đúng phần cache.
                chatbot_service.chatbot_tools,
                "_clean_data_max_date",
                return_value=None,
            ):
                first = chatbot_service._base_snapshot_cached(("signature-1",))
                cached = chatbot_service._base_snapshot_cached(("signature-1",))
                updated = chatbot_service._base_snapshot_cached(("signature-2",))
        finally:
            chatbot_service._base_snapshot_cached.cache_clear()

        self.assertIs(first, cached)
        self.assertEqual(get_release.call_count, 2)
        self.assertEqual(first["result"]["data"]["data_as_of"], "2026-07-30")
        self.assertEqual(updated["result"]["data"]["data_as_of"], "2026-07-31")

    def test_snapshot_prefers_live_clean_date_over_report_snapshot(self):
        metadata = {
            "model_name": "Random Forest",
            "policy_id": "policy-v1",
            "content_fingerprint": "release-1",
            "train_through_date": "2026-04-10",
        }
        state = {
            "status": "current",
            "artifact": dict(metadata),
            "metadata": metadata,
            "summary": {
                "dataset_report": {"date_max": "2026-07-20"},
                "clean_report": {},
                "feature_report": {},
            },
            "scope": {"FPT"},
            "warnings": [],
        }

        chatbot_service._base_snapshot_cached.cache_clear()
        try:
            with patch.object(
                chatbot_service.chatbot_tools,
                "get_release_state",
                return_value=state,
            ), patch.object(
                chatbot_service.chatbot_tools,
                "_clean_data_max_date",
                return_value="2026-07-31",
            ):
                snapshot = chatbot_service._base_snapshot_cached(("signature-live",))
        finally:
            chatbot_service._base_snapshot_cached.cache_clear()

        data = snapshot["result"]["data"]
        # Ba mốc ngày phải tách bạch, không được thay thế cho nhau.
        self.assertEqual(data["data_as_of"], "2026-07-31")
        self.assertEqual(data["report_as_of"], "2026-07-20")
        self.assertEqual(data["model_trained_through"], "2026-04-10")


class OutOfScopeSymbolContextTests(unittest.TestCase):
    """Mã user nêu nhưng ngoài symbol scope phải nói thẳng, không hỏi lại."""

    def setUp(self):
        self.scope = set(_FIXTURE_SCOPE)

    def test_detector_flags_unknown_upper_tickers(self):
        cases = [
            ("Cho tôi tín hiệu TSLA", ["TSLA"]),
            ("ZZZ999 thế nào?", ["ZZZ999"]),
            ("so sánh FPT với TSLA", ["TSLA"]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    chatbot_service._out_of_scope_symbols(text, self.scope), expected
                )

    def test_detector_ignores_acronyms_scope_and_lowercase(self):
        cases = [
            "F1 UP trên test là bao nhiêu?",
            "HOSE có bao nhiêu mã?",
            "chỉ báo RSI và MACD dùng thế nào",
            "tín hiệu FPT hôm nay",
            "giá của nó ra sao",
            # Câu viết hoa toàn bộ: bỏ heuristic, thà bỏ sót hơn báo sai.
            "CHO TOI TIN HIEU TSLA",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    chatbot_service._out_of_scope_symbols(text, self.scope), []
                )

    def _build(self, message):
        with patch.object(
            chatbot_service, "_base_snapshot_cached", return_value=_snapshot()
        ):
            return chatbot_service.build_context(message, [], None)

    def test_unknown_symbol_gets_out_of_scope_block_not_clarification(self):
        bundle = self._build("Cho tôi tín hiệu TSLA")

        self.assertIn("out_of_scope_symbols", bundle["context"])
        self.assertEqual(
            bundle["context"]["out_of_scope_symbols"]["symbols"], ["TSLA"]
        )
        # Đã nêu mã rồi thì không được hỏi lại "bạn muốn xem mã nào".
        self.assertNotIn("clarification_required", bundle["context"])
        self.assertIn(
            "symbol_out_of_scope",
            [w.get("code") for w in bundle["warnings"] if isinstance(w, dict)],
        )

    def test_in_scope_symbol_wins_over_out_of_scope_note(self):
        in_scope = sorted(_FIXTURE_SCOPE)[0]
        signal = _matching_envelope(
            "stock_signal",
            {"signals": [{"symbol": in_scope, "prediction": "UP"}]},
        )
        with (
            patch.object(
                chatbot_service, "_base_snapshot_cached", return_value=_snapshot()
            ),
            patch.object(
                chatbot_service.chatbot_tools,
                "get_stock_signals",
                return_value=signal,
            ),
        ):
            bundle = chatbot_service.build_context(
                f"so sánh {in_scope} với TSLA", [], None
            )

        # Có mã hợp lệ → vẫn trả tín hiệu, không chèn block out_of_scope.
        self.assertNotIn("out_of_scope_symbols", bundle["context"])

    def test_vague_question_without_symbol_still_asks_for_clarification(self):
        # "phân tích cổ phiếu" thuộc _AMBIGUOUS_DATA_WORDS nhưng không nêu mã nào
        # → nhánh clarification cũ phải giữ nguyên, guard out_of_scope không chặn.
        bundle = self._build("phân tích cổ phiếu giúp mình")

        self.assertNotIn("out_of_scope_symbols", bundle["context"])
        self.assertIn("clarification_required", bundle["context"])


class ConversationStateApiTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()
        self.state = {
            "active_symbols": ["fpt"],
            "topic": "signal",
            "ranking_order": None,
            "last_result_symbols": [],
        }

    def test_request_without_state_remains_backward_compatible(self):
        response_body = {
            "answer": "Chào bạn.",
            "sources": [],
            "warnings": [],
            "release_status": None,
            "conversation_state": {
                "active_symbols": [],
                "topic": None,
                "ranking_order": None,
                "last_result_symbols": [],
            },
        }
        with patch.object(chatbot_service, "chat", return_value=response_body) as chat:
            response = self.client.post("/api/chat", json={"message": "Chào"})

        self.assertEqual(response.status_code, 200)
        chat.assert_called_once_with("Chào", [], None)

    def test_valid_state_is_normalized_and_forwarded(self):
        with patch.object(
            chatbot_service, "chat", return_value={"answer": "ok"}
        ) as chat:
            response = self.client.post(
                "/api/chat", json={"message": "Còn nó?", "conversation_state": self.state}
            )

        self.assertEqual(response.status_code, 200)
        forwarded = chat.call_args.args[2]
        self.assertEqual(forwarded["active_symbols"], ["FPT"])

    def test_invalid_states_return_400(self):
        cases = (
            None,
            {**self.state, "extra": True},
            {**self.state, "topic": "unknown"},
            {**self.state, "active_symbols": ["FPT", "VNM", "HPG"]},
            {**self.state, "last_result_symbols": ["FPT"] * 11},
            {**self.state, "ranking_order": "random"},
        )
        for state in cases:
            with self.subTest(state=state):
                response = self.client.post(
                    "/api/chat", json={"message": "Tiếp tục", "conversation_state": state}
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["error"]["code"], "invalid_request")


class GroundingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.facts = {
            "FPT": {
                "prediction": "UP",
                "up_score_percent": 62.5,
                "decision_threshold_percent": 50.0,
                "close_at_reference": 70.6,
                "return_20d_percent": 12.35,
                "volatility_20d_percent": 2.0,
                "volume_ratio_20": 1.26,
                "reference_date": "2026-07-31",
                "threshold_relation": "above",
            },
            "VNM": {
                "prediction": "NOT_UP",
                "up_score_percent": 45.0,
                "decision_threshold_percent": 55.0,
                "close_at_reference": 80.0,
                "return_20d_percent": -2.0,
                "volatility_20d_percent": 3.0,
                "volume_ratio_20": 0.9,
                "reference_date": "2026-07-30",
                "threshold_relation": "below",
            },
        }
        self.numbers = [
            62.5, 50.0, 70.6, 12.35, 2.0, 1.26,
            45.0, 55.0, 80.0, -2.0, 3.0, 0.9,
            20.0, 2026.0, 7.0, 31.0, 30.0,
        ]

    def test_directional_language_must_match_prediction(self):
        chatbot_service._validate_grounded_answer(
            "FPT đang nghiêng tích cực.", self.numbers, self.facts
        )
        chatbot_service._validate_grounded_answer(
            "VNM chưa đủ điều kiện UP.", self.numbers, self.facts
        )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "VNM đang nghiêng tích cực.", self.numbers, self.facts
            )

    def test_not_up_cannot_be_stated_as_certain_decline(self):
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "VNM chắc chắn sẽ giảm.", self.numbers, self.facts
            )

    def test_direct_trading_advice_remains_blocked(self):
        chatbot_service._validate_grounded_answer(
            "Mình không thể quyết định bạn có nên mua FPT hay không.",
            self.numbers,
            self.facts,
        )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "Bạn nên mua FPT ngay.", self.numbers, self.facts
            )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "Mình không thể tư vấn, nhưng bạn nên mua FPT.",
                self.numbers,
                self.facts,
            )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "Không theo dữ liệu, bạn nên mua FPT.",
                self.numbers,
                self.facts,
            )

    def test_symbol_field_value_pairs_cannot_be_swapped(self):
        valid = (
            "FPT có prediction UP và Điểm UP 62,5%.",
            "Ngưỡng của VNM là 55%.",
            "FPT có giá tham chiếu 70,6.",
            "Return 20 phiên của FPT là 12,35%.",
            "VNM có volatility 20 phiên 3%.",
            "Volume ratio của FPT là 1,26.",
            "Ngày tham chiếu của VNM là 2026-07-30.",
        )
        invalid = (
            "VNM có prediction UP.",
            "Điểm UP của VNM là 62,5%.",
            "FPT có ngưỡng 55%.",
            "Giá tham chiếu của VNM là 70,6.",
            "VNM có return 20 phiên 12,35%.",
            "FPT có volatility 20 phiên 3%.",
            "Volume ratio của VNM là 1,26.",
            "Ngày tham chiếu của FPT là 2026-07-30.",
        )
        for answer in valid:
            with self.subTest(answer=answer):
                chatbot_service._validate_grounded_answer(
                    answer, self.numbers, self.facts
                )
        for answer in invalid:
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError):
                    chatbot_service._validate_grounded_answer(
                        answer, self.numbers, self.facts
                    )

    def test_soft_block_answer_itself_passes_every_validator(self):
        # SOFT_BLOCK_ANSWER là câu server trả khi validator chặn câu của LM. Nếu
        # ai sau này thêm số hoặc tên mã vào nó thì chính nó sẽ bị chặn → chat()
        # raise ngay trong except, biến soft-block thành 502 khó truy.
        chatbot_service._validate_grounded_answer(
            chatbot_service.SOFT_BLOCK_ANSWER, [], {}, {}
        )
        chatbot_service._validate_grounded_answer(
            chatbot_service.SOFT_BLOCK_ANSWER, self.numbers, self.facts, {}
        )

    def test_safety_violation_stays_hard_fail_not_soft_blockable(self):
        # Advice/certainty phải nổi lên thành lỗi thật: soft-block sẽ che mất việc
        # provider đang vượt phạm vi an toàn.
        for answer in ("Bạn nên mua FPT ngay.", "FPT chắc chắn sẽ tăng."):
            with self.subTest(answer=answer):
                with self.assertRaises(
                    chatbot_service.ChatbotServiceError
                ) as raised:
                    chatbot_service._validate_grounded_answer(
                        answer, self.numbers, self.facts
                    )
                self.assertFalse(raised.exception.soft_blockable)

    def test_data_mismatch_is_soft_blockable(self):
        with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
            chatbot_service._validate_grounded_answer(
                "Điểm UP của VNM là 62,5%.", self.numbers, self.facts
            )
        self.assertTrue(raised.exception.soft_blockable)

    def test_safe_not_up_disclaimer_is_allowed_but_future_claim_is_blocked(self):
        chatbot_service._validate_grounded_answer(
            "VNM chưa đủ điều kiện UP; điều đó không có nghĩa là chắc chắn giảm.",
            self.numbers,
            self.facts,
        )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "VNM sẽ giảm.", self.numbers, self.facts
            )
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer(
                "Không theo dữ liệu, VNM chắc chắn giảm.",
                self.numbers,
                self.facts,
            )


class DateSemanticsTests(unittest.TestCase):
    """P2-A: ba mốc ngày phải tách nghĩa, không được trộn vào một max()."""

    def _result(self, *, as_of, data=None):
        return {
            "ok": True,
            "data": data or {},
            "source": {"kind": "x"},
            "as_of": as_of,
            "release": {},
            "warnings": [],
            "error": None,
        }

    def _bundle(self):
        return {
            "context": {},
            "sources": [],
            "warnings": [],
            "release_status": None,
            "data_as_of": None,
            "report_as_of": None,
            "model_trained_through": None,
            "grounded_numbers": [],
            "stock_facts": {},
            "model_facts": {},
            "conversation_state": chatbot_service.empty_conversation_state(),
            "_release_identity": None,
        }

    def test_model_block_train_cutoff_does_not_become_data_as_of(self):
        bundle = self._bundle()
        chatbot_service._consume_context_result(
            bundle, "dataset", self._result(as_of="2026-07-31")
        )
        # Block model đặt as_of = train_through_date; nếu lọt vào max() thì
        # data_as_of vẫn là 2026-07-31 (lớn hơn), nên test dùng cutoff LỚN hơn
        # để phát hiện rò rỉ.
        chatbot_service._consume_context_result(
            bundle,
            "model",
            self._result(
                as_of="2027-01-01", data={"model_trained_through": "2027-01-01"}
            ),
        )
        self.assertEqual(bundle["data_as_of"], "2026-07-31")
        self.assertEqual(bundle["model_trained_through"], "2027-01-01")

    def test_features_and_project_blocks_do_not_touch_data_as_of(self):
        for key in ("features", "project_overview", "limitations"):
            with self.subTest(key=key):
                bundle = self._bundle()
                chatbot_service._consume_context_result(
                    bundle, key, self._result(as_of="2099-12-31")
                )
                self.assertIsNone(bundle["data_as_of"])

    def test_data_bearing_blocks_still_raise_data_as_of(self):
        for key in ("project_snapshot", "stock_signals", "comparison", "ranking", "dataset"):
            with self.subTest(key=key):
                bundle = self._bundle()
                chatbot_service._consume_context_result(
                    bundle, key, self._result(as_of="2026-07-31")
                )
                self.assertEqual(bundle["data_as_of"], "2026-07-31")

    def test_report_as_of_is_captured_from_dataset_block(self):
        bundle = self._bundle()
        chatbot_service._consume_context_result(
            bundle,
            "dataset",
            self._result(
                as_of="2026-07-31",
                data={"data_as_of": "2026-07-31", "report_as_of": "2026-07-20"},
            ),
        )
        self.assertEqual(bundle["data_as_of"], "2026-07-31")
        self.assertEqual(bundle["report_as_of"], "2026-07-20")

    def test_snapshot_prefers_live_csv_date_over_frozen_report(self):
        metadata = {
            "model_name": "Random Forest",
            "policy_id": "policy-v1",
            "content_fingerprint": "fp-1",
            "train_through_date": "2026-04-10",
        }
        state = {
            "status": "current",
            "artifact": dict(metadata),
            "metadata": metadata,
            "summary": {
                "dataset_report": {"date_max": "2026-07-20"},
                "clean_report": {},
                "feature_report": {},
            },
            "scope": {"FPT"},
            "warnings": [],
        }
        chatbot_service._base_snapshot_cached.cache_clear()
        try:
            with patch.object(
                chatbot_service.chatbot_tools, "get_release_state", return_value=state
            ), patch.object(
                chatbot_service.chatbot_tools,
                "_clean_data_max_date",
                return_value="2026-07-31",
            ):
                snapshot = chatbot_service._base_snapshot_cached(("sig",))
        finally:
            chatbot_service._base_snapshot_cached.cache_clear()

        data = snapshot["result"]["data"]
        self.assertEqual(data["data_as_of"], "2026-07-31")
        self.assertEqual(data["report_as_of"], "2026-07-20")
        self.assertEqual(data["model_trained_through"], "2026-04-10")

    def test_report_as_of_is_omitted_when_report_matches_live_data(self):
        metadata = {
            "model_name": "Random Forest",
            "policy_id": "policy-v1",
            "content_fingerprint": "fp-1",
            "train_through_date": "2026-04-10",
        }
        state = {
            "status": "current",
            "artifact": dict(metadata),
            "metadata": metadata,
            "summary": {
                "dataset_report": {"date_max": "2026-07-31"},
                "clean_report": {},
                "feature_report": {},
            },
            "scope": {"FPT"},
            "warnings": [],
        }
        chatbot_service._base_snapshot_cached.cache_clear()
        try:
            with patch.object(
                chatbot_service.chatbot_tools, "get_release_state", return_value=state
            ), patch.object(
                chatbot_service.chatbot_tools,
                "_clean_data_max_date",
                return_value="2026-07-31",
            ):
                snapshot = chatbot_service._base_snapshot_cached(("sig",))
        finally:
            chatbot_service._base_snapshot_cached.cache_clear()

        self.assertIsNone(snapshot["result"]["data"]["report_as_of"])


if __name__ == "__main__":
    unittest.main()
