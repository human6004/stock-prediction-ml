# Trạng thái chatbot workflow

`chatbot.workflow.json` và `chatbot-workflow.html` đã được render lại theo kiến trúc Action-Decision hiện hành (một LLM call trả `{action, arguments}`, validator exact schema, fixed dispatcher, deterministic formatter). Không còn LEGACY/STALE.

Canonical:

- [`docs/CHATBOT_ARCHITECTURE.md`](../../CHATBOT_ARCHITECTURE.md) — mô tả kiến trúc, là source of truth khi có xung đột
- [Action-Decision sequence](../luuDo/06-sequence-chatbot-action-decision.html) và [JSON nguồn](../luuDo/06-sequence-chatbot-action-decision.sequence.json) — góc nhìn sequence
- `chatbot.workflow.json` / `chatbot-workflow.html` (thư mục này) — góc nhìn workflow theo lane

Khi luồng chatbot đổi: sửa JSON nguồn rồi render lại bằng skill `archify`, đừng sửa tay HTML.
