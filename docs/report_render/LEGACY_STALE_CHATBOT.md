# Đã cập nhật — nội dung chatbot không còn LEGACY/STALE

Nội dung chatbot trong report generator (`content_ch3a.py`, `content_ch3b.py`, `content_ch3c.py`, `content_front.py`, `make_flow_figures.py`) đã được viết lại theo kiến trúc Action-Decision hiện hành, nay đã cập nhật thêm LLM call thứ hai (grounded compose, có fallback formatter). DOCX và 6 hình PNG flow/sequence đã được build lại từ nguồn mới (12/08). Khi sửa nội dung, chạy lại `make_flow_figures.py` rồi `build_report.py`. Tên file này giữ nguyên làm dấu vết lịch sử.

Canonical:

- [`docs/CHATBOT_ARCHITECTURE.md`](../CHATBOT_ARCHITECTURE.md)
- [Action-Decision sequence](../diagrams/luuDo/06-sequence-chatbot-action-decision.html) và [JSON nguồn](../diagrams/luuDo/06-sequence-chatbot-action-decision.sequence.json)
