# Trạng thái chatbot workflow

**Đã cập nhật (12/08):** `chatbot.workflow.json` + `chatbot-workflow.html` (thư mục này) và `docs/diagrams/luuDo/01-kien-truc-runtime-local.*`, `06-sequence-chatbot-action-decision.*` đều đã vẽ đúng kiến trúc hai LLM call (decision + grounded compose, fallback formatter — xem `docs/CHATBOT_ARCHITECTURE.md` mục 1 và 7). HTML render lại từ JSON nguồn, card và lane nhất quán "một lượt, hai call".

Canonical:

- [`docs/CHATBOT_ARCHITECTURE.md`](../../CHATBOT_ARCHITECTURE.md) — mô tả kiến trúc, là source of truth khi có xung đột
- [Action-Decision sequence](../luuDo/06-sequence-chatbot-action-decision.html) và [JSON nguồn](../luuDo/06-sequence-chatbot-action-decision.sequence.json) — góc nhìn sequence
- `chatbot.workflow.json` / `chatbot-workflow.html` (thư mục này) — góc nhìn workflow theo lane

Khi luồng chatbot đổi: sửa JSON nguồn rồi render lại bằng skill `archify`, đừng sửa tay HTML.
