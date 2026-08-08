"""UI contracts for the simplified /chat page."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ChatbotUiTests(unittest.TestCase):
    def setUp(self):
        self.page = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.client = (ROOT / "static" / "chat-client.js").read_text(encoding="utf-8")

    def test_prompts_remain_natural_and_project_focused(self):
        for prompt in (
            "FPT đang thế nào?",
            "So sánh FPT với VNM",
            "Model học từ dữ liệu gì?",
        ):
            self.assertIn(f'data-prompt="{prompt}"', self.page)

    def test_metadata_is_limited_to_sources_warnings_and_two_dates(self):
        for field in ("sources", "warnings", "data_as_of", "model_trained_through"):
            self.assertIn(field, self.page + self.client)
        self.assertNotIn("release_status", self.page + self.client)
        self.assertNotIn("conversation_state", self.page + self.client)
        self.assertNotIn('id="chat-release"', self.page)

    def test_warning_list_and_answers_use_safe_dom(self):
        combined = self.page + self.client
        self.assertIn('document.createElement("ul")', combined)
        self.assertIn('document.createElement("li")', combined)
        self.assertIn("textContent", combined)
        self.assertNotIn("innerHTML", combined)

    def test_accessibility_loading_timeout_and_public_error_remain(self):
        self.assertIn('aria-live="polite"', self.page)
        self.assertIn('aria-busy="false"', self.page)
        self.assertIn("Đang xử lý…", self.page)
        self.assertIn("AbortError", self.page)
        self.assertIn("error.publicMessage", self.client)
        self.assertIn("result.focus()", self.page)

    def test_rich_renderer_copy_and_dock_are_removed(self):
        combined = self.page + self.client
        for removed in (
            "renderRichText",
            "buildCopyButton",
            "buildSourceDetails",
            "createStaggerOnce",
            "chat-dock",
        ):
            self.assertNotIn(removed, combined)


if __name__ == "__main__":
    unittest.main()
