# Đã cập nhật — nội dung chatbot không còn LEGACY/STALE

Nội dung chatbot trong slide, script và chart (`outline.md`, `script_thuyet_trinh.md`, `build_pptx.py` slide 10–11, `make_chatbot_chart.py`) đã viết lại theo kiến trúc Action-Decision (decision → validator → dispatcher → grounded compose có fallback) và PPTX đã regenerate (12/08, selfcheck 0 lỗi). Lưu ý: `img/chatbot_demo.png` vẫn là screenshot trước rework — muốn khớp hoàn toàn slide 11 thì chạy `shoot_chat.py` với Flask đang chạy. Tên file này giữ nguyên làm dấu vết lịch sử; source of truth vẫn là tài liệu canonical bên dưới.

Canonical:

- [`docs/CHATBOT_ARCHITECTURE.md`](../CHATBOT_ARCHITECTURE.md)
- [Action-Decision sequence](../diagrams/luuDo/06-sequence-chatbot-action-decision.html) và [JSON nguồn](../diagrams/luuDo/06-sequence-chatbot-action-decision.sequence.json)
