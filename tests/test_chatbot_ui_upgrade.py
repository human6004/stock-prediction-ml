"""UI and documentation contracts for the conversational chatbot upgrade."""

from pathlib import Path
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]


class ChatbotUiUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT_DIR / "templates" / "chat.html").read_text(
            encoding="utf-8"
        )
        self.client = (ROOT_DIR / "static" / "chat-client.js").read_text(
            encoding="utf-8"
        )

    def test_prompts_are_natural_and_project_focused(self):
        prompts = (
            "FPT đang thế nào?",
            "So sánh FPT với VNM",
            "Model học từ dữ liệu gì?",
        )
        for prompt in prompts:
            self.assertIn(f'data-prompt="{prompt}"', self.source)
            self.assertIn(f">{prompt}</button>", self.source)

    def test_release_and_source_kinds_have_vietnamese_labels(self):
        expected_pairs = (
            ('current: "Hiện hành"',),
            ('legacy: "Bản cũ"',),
            ('inconsistent: "Không đồng nhất"',),
            ('missing: "Chưa sẵn sàng"',),
            ('stock_signal: "Tín hiệu cổ phiếu"',),
            ('ranking: "Bảng xếp hạng Điểm UP"',),
            ('model_metadata: "Thông tin model"',),
            ('dataset_info: "Thông tin dataset"',),
            ('feature_info: "Thông tin feature"',),
            ('project_contract: "Kiến thức project"',),
        )
        for (expected,) in expected_pairs:
            self.assertIn(expected, self.source)

    def test_warnings_render_as_safe_list(self):
        combined = self.source + self.client
        self.assertIn('document.createElement("ul")', combined)
        self.assertIn('document.createElement("li")', combined)
        self.assertRegex(self.source, r'setMeta\(\s*warnings,\s*"Cảnh báo"')
        self.assertNotIn("innerHTML", combined)
        self.assertNotIn('values.join(" | ")', self.source)

    def test_each_request_clears_previous_metadata_before_fetch(self):
        self.assertIn("function clearMetadata()", self.source)
        submit = self.source.index('form.addEventListener("submit"')
        clear_metadata = self.source.index("clearMetadata();", submit)
        send = self.source.index("kit.sendMessage(message)", submit)
        self.assertLess(clear_metadata, send)
        self.assertIn('fetch("/api/chat"', self.client)

    def test_transcript_announces_and_focuses_the_new_assistant_entry(self):
        transcript_tag = self.source.split('id="chat-transcript"', 1)[1].split(
            ">", 1
        )[0]
        self.assertIn('aria-live="polite"', transcript_tag)
        self.assertIn("item.tabIndex = -1", self.source)
        self.assertIn("return item", self.source)
        self.assertIn('const result = appendMessage("assistant", data.answer)', self.source)
        self.assertIn("result.focus()", self.source)
        self.assertNotIn('document.getElementById("chat-result")', self.source)

    def test_transport_and_json_errors_never_render_raw_exception_messages(self):
        self.assertIn('typeof data.error.message === "string"', self.client)
        self.assertIn("error.publicMessage", self.client)
        self.assertIn("Không thể kết nối trợ lý lúc này. Vui lòng thử lại.", self.source)
        self.assertNotIn(": error.message;", self.source)


class ChatbotDocumentationUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.document = (ROOT_DIR / "docs" / "CHATBOT_RAG_MUC_B.md").read_text(
            encoding="utf-8"
        )

    def test_document_describes_server_built_context_sources(self):
        # Neo vào định danh code (hàm, module, field provider payload) thay vì
        # văn phong tài liệu: chỉ đổi khi code đổi, không đổi khi sửa câu chữ.
        required = (
            "build_context",
            "chatbot_tools.py",
            "get_stock_signals",
            "get_ranking",
            "get_model_info",
            "get_dataset_info",
            "get_feature_info",
            "get_project_info",
            "tool_choice",
        )
        for identifier in required:
            self.assertIn(identifier, self.document)

    def test_document_covers_new_runtime_safety_contracts(self):
        required = (
            "max_retries=0",
            "context_budget_exceeded",
            "conversation_state",
            "up_score_percent",
            "decision_threshold_percent",
            "excluded_stale_count",
            "no_current_signals",
            "llm_invalid_config",
            "LLM_API_KEY",
            "LLM_BASE_URL",
        )
        for identifier in required:
            self.assertIn(identifier, self.document)

    def test_503_documents_missing_or_invalid_provider_configuration(self):
        self.assertRegex(self.document, r"`503`\s*\|")
        self.assertRegex(self.document, r"503\s+llm_not_configured")
        self.assertIn("llm_invalid_config", self.document)


if __name__ == "__main__":
    unittest.main()
