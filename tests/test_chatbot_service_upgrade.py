"""Focused regression tests for natural, safe chatbot orchestration."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services import chatbot_service


def _final_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _tool_response(
    name: str = "get_stock_signals",
    arguments: dict | None = None,
) -> dict:
    arguments = arguments or {"symbols": ["FPT"]}
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _tool_result() -> dict:
    return {
        "ok": True,
        "data": {
            "signals": [
                {
                    "symbol": "FPT",
                    "up_score_percent": 50.64,
                    "horizon_sessions": 20,
                }
            ]
        },
        "source": {
            "kind": "stock_signal",
            "symbols": ["FPT"],
            "as_of": "2026-07-20",
        },
        "as_of": "2026-07-20",
        "release": {
            "status": "current",
            "policy_id": "policy",
            "content_fingerprint": "fingerprint",
        },
        "warnings": [],
        "error": None,
    }


def _client(*responses) -> SimpleNamespace:
    create = Mock(side_effect=list(responses))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


class NaturalDialogueTests(unittest.TestCase):
    def test_social_and_advice_work_without_provider_configuration(self):
        cases = (
            ("Chào!", "Chào bạn!"),
            ("Cảm ơn bạn.", "Không có gì"),
            ("Nên mua cổ phiếu nào?", "Mình không thể quyết định mua hoặc bán thay bạn."),
            ("Chọn giúp tôi một mã để mua", "Mình không thể quyết định mua hoặc bán thay bạn."),
            ("Gợi ý mã đầu tư", "Mình không thể quyết định mua hoặc bán thay bạn."),
        )
        with patch.multiple(
            chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""
        ):
            for message, expected in cases:
                with self.subTest(message=message):
                    result = chatbot_service.chat(message, [])
                    self.assertIn(expected, result["answer"])
                    self.assertEqual(result["sources"], [])
                    self.assertEqual(result["warnings"], [])
                    self.assertIsNone(result["release_status"])

    def test_social_and_advice_matching_is_conservative(self):
        messages = (
            "Chào, phân tích FPT giúp mình",
            "So sánh FPT và VNM để tôi tham khảo",
        )
        for message in messages:
            with self.subTest(message=message):
                client = _client(_final_response("Không dùng tool."))
                result = chatbot_service.chat(message, [], client=client)
                client.chat.completions.create.assert_called_once()
                self.assertEqual(result["answer"], chatbot_service.SCOPE_MESSAGE)

    def test_common_transaction_advice_phrasings_are_handled_locally(self):
        messages = (
            "Mua FPT được không?",
            "Theo bạn mã nào đáng mua?",
            "Bạn khuyên tôi mua FPT không?",
        )
        with patch.multiple(
            chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""
        ):
            for message in messages:
                with self.subTest(message=message):
                    result = chatbot_service.chat(message, [])
                    self.assertEqual(result["answer"], chatbot_service.ADVICE_MESSAGE)

    def test_safe_no_tool_clarification_is_allowed(self):
        clarification = "Bạn muốn mình phân tích mã nào?"
        client = _client(_final_response(clarification))

        result = chatbot_service.chat("Phân tích giúp mình", [], client=client)

        self.assertEqual(result["answer"], clarification)
        self.assertEqual(result["sources"], [])

    def test_unsafe_no_tool_answers_still_fall_back_to_scope(self):
        cases = (
            "Bạn muốn xem top 5 mã?",
            "Bạn muốn xem FPT?\nHay VNM?",
            "Bạn muốn mua mã nào?",
            "Mình có thể giúp bạn chọn mã?",
            "Bạn muốn xem mã nào??",
            "Bạn muốn " + ("x" * 231) + "?",
        )
        for answer in cases:
            with self.subTest(answer=answer[:40]):
                client = _client(_final_response(answer))
                result = chatbot_service.chat("Câu hỏi thiếu thông tin", [], client=client)
                self.assertEqual(result["answer"], chatbot_service.SCOPE_MESSAGE)

    def test_no_tool_clarification_cannot_embed_a_model_claim(self):
        client = _client(
            _final_response("Bạn muốn xem model hiện tại đang dùng XGBoost?")
        )

        result = chatbot_service.chat("Model nào?", [], client=client)

        self.assertEqual(result["answer"], chatbot_service.SCOPE_MESSAGE)

    def test_advice_spelled_percent_and_wrong_symbol_score_are_rejected(self):
        answers = (
            "Bạn có thể mua FPT.",
            "Mình khuyên bạn mua FPT.",
            "FPT phù hợp để mua.",
            "Hãy bán VNM.",
            "Điểm UP của FPT là chín mươi chín phần trăm.",
            "Điểm UP của FPT là 20%.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                client = _client(_tool_response(), _final_response(answer))
                with patch.object(
                    chatbot_service.chatbot_tools,
                    "execute_tool",
                    return_value=_tool_result(),
                ):
                    with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                        chatbot_service.chat("Phân tích FPT", [], client=client)
                self.assertEqual(raised.exception.code, "ungrounded_response")

    def test_correct_symbol_score_claim_remains_allowed(self):
        client = _client(
            _tool_response(), _final_response("FPT có Điểm UP 50,64%.")
        )
        with patch.object(
            chatbot_service.chatbot_tools,
            "execute_tool",
            return_value=_tool_result(),
        ):
            result = chatbot_service.chat("Phân tích FPT", [], client=client)

        self.assertEqual(result["answer"], "FPT có Điểm UP 50,64%.")

    def test_project_question_uses_whitelisted_project_tool(self):
        client = _client(
            _tool_response("get_project_info", {"topic": "target"}),
            _final_response("Target dùng đúng 5 phiên thị trường."),
        )
        tool_result = {
            "ok": True,
            "data": {
                "topic": "target",
                "title": "Target dự báo",
                "facts": {"prediction_horizon_sessions": 5},
                "notes": [],
            },
            "source": {"kind": "project_contract", "topic": "target"},
            "as_of": None,
            "release": {
                "status": None,
                "policy_id": None,
                "content_fingerprint": None,
            },
            "warnings": [],
            "error": None,
        }
        with patch.object(
            chatbot_service.chatbot_tools,
            "execute_tool",
            return_value=tool_result,
        ) as execute:
            result = chatbot_service.chat("Target được tạo thế nào?", [], client=client)

        execute.assert_called_once_with("get_project_info", {"topic": "target"})
        self.assertEqual(result["sources"], [tool_result["source"]])

    def test_ranking_runs_only_after_user_explicitly_asks_for_it(self):
        first = chatbot_service.chat("Hôm nay nên mua cổ phiếu nào?", [])
        history = [
            {"role": "user", "content": "Hôm nay nên mua cổ phiếu nào?"},
            {"role": "assistant", "content": first["answer"]},
        ]
        client = _client(
            _tool_response(
                "get_ranking", {"order": "highest_up_score", "top_n": 5}
            ),
            _final_response("Mình đã đọc bảng xếp hạng offline."),
        )
        ranking_result = {
            **_tool_result(),
            "data": {"ranking": [], "top_n": 5, "excluded_stale_count": 0},
            "source": {"kind": "ranking", "symbols": [], "as_of": "2026-07-20"},
        }
        with patch.object(
            chatbot_service.chatbot_tools,
            "execute_tool",
            return_value=ranking_result,
        ) as execute:
            chatbot_service.chat(
                "Cho mình xem bảng xếp hạng Điểm UP", history, client=client
            )

        execute.assert_called_once_with(
            "get_ranking", {"order": "highest_up_score", "top_n": 5}
        )

    def test_prompt_locks_natural_tone_and_two_decimal_display(self):
        prompt = chatbot_service.SYSTEM_PROMPT.casefold()

        self.assertIn("xưng “mình”", prompt)
        self.assertIn("chỉ hỏi lại một câu", prompt)
        self.assertIn("tối đa hai chữ số thập phân", prompt)
        self.assertIn("không tự tính số mới", prompt)


class ProviderUrlSecurityTests(unittest.TestCase):
    def test_allowed_provider_urls(self):
        urls = (
            "https://provider.example/v1",
            "http://localhost/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(chatbot_service._validate_base_url(url), url)

    def test_invalid_provider_url_returns_stable_503(self):
        urls = (
            "http://provider.example/v1",
            "https://user:pass@provider.example/v1",
            "https://provider.example/v1?token=secret",
            "https://provider.example/v1#fragment",
            "ftp://provider.example/v1",
            "https:///v1",
            " https://provider.example/v1",
        )
        for url in urls:
            with self.subTest(url=url), patch.multiple(
                chatbot_service,
                LLM_BASE_URL=url,
                LLM_API_KEY="test-secret",
                LLM_MODEL="tool-model",
            ):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._create_client()
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("llm_invalid_config", 503),
                )
                self.assertNotIn(url, raised.exception.message)


if __name__ == "__main__":
    unittest.main()
