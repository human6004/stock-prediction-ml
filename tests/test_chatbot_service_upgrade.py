"""Các regression test bổ sung cho chatbot structured context injection (SCI)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services import chatbot_service


def _response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _client(content: str) -> SimpleNamespace:
    create = Mock(return_value=_response(content))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _bundle(*, facts=None, model_facts=None, numbers=None, clarification=False) -> dict:
    context = {
        "project_snapshot": {
            "ok": True,
            "data": {"serving_mode": "offline_published_artifacts"},
            "as_of": "2026-07-31",
            "warnings": [],
            "error": None,
        }
    }
    if clarification:
        context["clarification_required"] = {
            "reason": "missing_symbol_or_criterion"
        }
    return {
        "context": context,
        "sources": [{"kind": "project_snapshot", "as_of": "2026-07-31"}],
        "warnings": [],
        "release_status": "current",
        "data_as_of": "2026-07-31",
        "grounded_numbers": numbers or [2026.0, 7.0, 31.0],
        "stock_facts": facts or {},
        "model_facts": model_facts or {},
        "conversation_state": chatbot_service.empty_conversation_state(),
    }


class LlmFirstDialogueTests(unittest.TestCase):
    def test_social_and_advice_messages_all_use_one_provider_call(self):
        for message in ("Chào!", "Cảm ơn bạn.", "Có nên mua FPT không?"):
            with self.subTest(message=message):
                client = _client("Mình không quyết định giao dịch thay bạn.")
                with patch.object(
                    chatbot_service, "build_context", return_value=_bundle()
                ):
                    chatbot_service.chat(message, [], client=client)
                client.chat.completions.create.assert_called_once()

    def test_clarification_is_generated_in_the_same_llm_call(self):
        client = _client("Bạn muốn mình phân tích mã nào?")
        with patch.object(
            chatbot_service,
            "build_context",
            return_value=_bundle(clarification=True),
        ):
            result = chatbot_service.chat("Phân tích giúp mình", [], client=client)

        self.assertEqual(result["answer"], "Bạn muốn mình phân tích mã nào?")
        client.chat.completions.create.assert_called_once()

    def test_clarification_must_be_exactly_one_short_question(self):
        for answer in (
            "Mình chưa rõ. Bạn muốn mã nào?",
            "Bạn muốn mã nào? Hay muốn xem ranking?",
            "Bạn muốn xem top 5 mã?",
            "Mình cần thêm thông tin.",
        ):
            with self.subTest(answer=answer):
                with patch.object(
                    chatbot_service,
                    "build_context",
                    return_value=_bundle(clarification=True),
                ):
                    result = chatbot_service.chat(
                        "Phân tích giúp mình", [], client=_client(answer)
                    )
                self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
                self.assertIn(
                    "answer_blocked_ungrounded",
                    [w["code"] for w in result["warnings"]],
                )

    def test_correct_symbol_score_is_allowed_and_swapped_score_is_rejected(self):
        facts = {
            "FPT": {
                "prediction": "UP",
                "up_score_percent": 50.64,
                "decision_threshold_percent": 49.0,
                "threshold_relation": "above",
            },
            "VNM": {
                "prediction": "NOT_UP",
                "up_score_percent": 40.0,
                "decision_threshold_percent": 49.0,
                "threshold_relation": "below",
            },
        }
        bundle = _bundle(facts=facts, numbers=[50.64, 49.0, 40.0])
        client = _client("FPT có Điểm UP 50,64%.")
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("FPT thế nào?", [], client=client)
        self.assertEqual(result["answer"], "FPT có Điểm UP 50,64%.")

        client = _client("VNM có Điểm UP 50,64%.")
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("VNM thế nào?", [], client=client)
        self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
        self.assertIn(
            "answer_blocked_ungrounded",
            [w["code"] for w in result["warnings"]],
        )

    def test_colon_prediction_must_match_symbol_fact(self):
        bundle = _bundle(
            facts={"FPT": {"prediction": "NOT_UP"}},
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            result = chatbot_service.chat("FPT thế nào?", [], client=_client("FPT: UP"))
        self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
        self.assertIn(
            "answer_blocked_ungrounded",
            [w["code"] for w in result["warnings"]],
        )

        generic = "Nhãn: UP chỉ là output phân loại."
        with patch.object(
            chatbot_service,
            "build_context",
            return_value=_bundle(),
        ):
            result = chatbot_service.chat("UP là gì?", [], client=_client(generic))
        self.assertEqual(result["answer"], generic)

    def test_metric_value_is_bound_to_split_and_metric_name(self):
        bundle = _bundle(
            model_facts={
                "test": {"precision_up": 51.0, "recall_up": 62.0},
            },
            numbers=[51.0, 62.0],
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            valid = chatbot_service.chat(
                "Metric TEST?", [], client=_client("Precision UP TEST là 51%.")
            )
        self.assertEqual(valid["answer"], "Precision UP TEST là 51%.")

        for answer in (
            "Precision UP TEST là 62%.",
            "Precision UP đạt 62% trên TEST.",
        ):
            with (
                self.subTest(answer=answer),
                patch.object(chatbot_service, "build_context", return_value=bundle),
            ):
                result = chatbot_service.chat("Metric TEST?", [], client=_client(answer))
            self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
            self.assertIn(
                "answer_blocked_ungrounded",
                [w["code"] for w in result["warnings"]],
            )

    def test_up_score_comparison_relation_must_match_symbol_facts(self):
        bundle = _bundle(
            facts={
                "FPT": {"up_score_percent": 40.0},
                "VNM": {"up_score_percent": 60.0},
            },
            numbers=[40.0, 60.0],
        )
        with patch.object(chatbot_service, "build_context", return_value=bundle):
            valid = chatbot_service.chat(
                "So sánh FPT VNM",
                [],
                client=_client("FPT có Điểm UP thấp hơn VNM."),
            )
        self.assertEqual(valid["answer"], "FPT có Điểm UP thấp hơn VNM.")

        for answer in (
            "FPT có Điểm UP cao hơn VNM.",
            "FPT cao hơn VNM về Điểm UP.",
        ):
            with (
                self.subTest(answer=answer),
                patch.object(chatbot_service, "build_context", return_value=bundle),
            ):
                result = chatbot_service.chat(
                    "So sánh FPT VNM", [], client=_client(answer)
                )
            self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
            self.assertIn(
                "answer_blocked_ungrounded",
                [w["code"] for w in result["warnings"]],
            )

    def test_negated_certainty_statement_is_allowed(self):
        answer = "Điểm UP không đảm bảo giá sẽ tăng."
        with patch.object(
            chatbot_service,
            "build_context",
            return_value=_bundle(),
        ):
            result = chatbot_service.chat("Điểm UP nghĩa là gì?", [], client=_client(answer))

        self.assertEqual(result["answer"], answer)

    def test_prompt_locks_llm_first_rules(self):
        prompt = chatbot_service.SYSTEM_PROMPT.casefold()
        for phrase in (
            "không gọi tool",
            "context_json",
            "history do client cung cấp",
            "chỉ hỏi lại đúng một câu",
            "tối đa hai chữ số thập phân",
            "không tự tính số mới",
        ):
            self.assertIn(phrase, prompt)


class ProviderUrlSecurityTests(unittest.TestCase):
    def test_allowed_provider_urls(self):
        for url in (
            "https://provider.example/v1",
            "http://localhost/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        ):
            with self.subTest(url=url):
                self.assertEqual(chatbot_service._validate_base_url(url), url)

    def test_invalid_provider_url_returns_stable_503(self):
        for url in (
            "http://provider.example/v1",
            "https://user:pass@provider.example/v1",
            "https://provider.example/v1?token=secret",
            "https://provider.example/v1#fragment",
            "ftp://provider.example/v1",
            "https:///v1",
            " https://provider.example/v1",
        ):
            with self.subTest(url=url), patch.multiple(
                chatbot_service,
                LLM_BASE_URL=url,
                LLM_API_KEY="test-secret",
                LLM_MODEL="llm-first-model",
            ):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._create_client()
                self.assertEqual(
                    (raised.exception.code, raised.exception.status),
                    ("llm_invalid_config", 503),
                )
                self.assertNotIn(url, raised.exception.message)

    def test_whitespace_only_key_or_model_is_invalid_configuration(self):
        for key, model in ((" ", "model"), ("secret", " ")):
            with self.subTest(key=key, model=model), patch.multiple(
                chatbot_service,
                LLM_BASE_URL="https://provider.example/v1",
                LLM_API_KEY=key,
                LLM_MODEL=model,
            ):
                self.assertFalse(chatbot_service.is_configured())
                with self.assertRaises(chatbot_service.ChatbotServiceError) as raised:
                    chatbot_service._create_client()
                self.assertEqual(raised.exception.status, 503)


if __name__ == "__main__":
    unittest.main()
