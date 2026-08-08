"""Integration contracts for the natural, local-first chatbot flow."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app as web_app
from services import chatbot_service


class ChatbotUpgradeRouteTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_every_message_requires_provider_configuration(self):
        messages = (
            "chào",
            "chọn giúp tôi một mã để mua",
        )
        with patch.multiple(
            chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""
        ):
            for message in messages:
                with self.subTest(message=message):
                    response = self.client.post("/api/chat", json={"message": message})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(
                        response.json["error"]["code"], "llm_not_configured"
                    )

    def test_greeting_with_factual_request_still_requires_provider(self):
        with patch.multiple(
            chatbot_service, LLM_BASE_URL="", LLM_API_KEY="", LLM_MODEL=""
        ):
            response = self.client.post(
                "/api/chat", json={"message": "Chào, phân tích FPT giúp mình"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"]["code"], "llm_not_configured")

    def test_invalid_remote_http_provider_maps_to_stable_503(self):
        with patch.multiple(
            chatbot_service,
            LLM_BASE_URL="http://provider.example/v1",
            LLM_API_KEY="test-key",
            LLM_MODEL="tool-model",
        ):
            response = self.client.post("/api/chat", json={"message": "Phân tích FPT"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"]["code"], "llm_invalid_config")


if __name__ == "__main__":
    unittest.main()
