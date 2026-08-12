# Kiến trúc chatbot LLM Action-Decision

> **CANONICAL:** Tài liệu này và sơ đồ [06-sequence-chatbot-action-decision.html](diagrams/luuDo/06-sequence-chatbot-action-decision.html) mô tả runtime chatbot hiện hành. Nguồn sơ đồ là `diagrams/luuDo/06-sequence-chatbot-action-decision.sequence.json`.
>
> Phạm vi: đồ án Niên luận cơ sở, dữ liệu cổ phiếu HOSE offline. LLM hiểu ngôn ngữ tự nhiên; backend sở hữu dữ liệu, ML và mọi câu trả lời có số liệu.

## 1. Luồng runtime

Hai giao diện dùng chung một API và một contract:

```text
User tại /chat hoặc floating dock
  → POST /api/chat
  → validate message + tối đa 3 cặp history gần nhất
  → đúng 1 LLM call trả {action, arguments}
  → validate exact schema + chuẩn hóa arguments
  → fixed dispatcher
  → scope check + data handler / prediction_service
  → deterministic formatter
  → JSON response 5 field
```

Không có keyword/regex router, retry decision, LLM call thứ hai hoặc conversation state riêng.

## 2. Vai trò của LLM

LLM chỉ:

1. hiểu câu hiện tại trong ngữ cảnh history gần nhất;
2. chọn đúng một action trong whitelist;
3. trích xuất arguments theo schema cố định.

LLM không đọc file, gọi ML, chọn model/threshold, xác nhận symbol scope hoặc tạo câu trả lời cuối. Probability, prediction, ranking, metric và số liệu dataset luôn đến từ backend.

## 3. Decision contract

Top-level phải có đúng hai khóa:

```json
{
  "action": "STOCK_SIGNAL",
  "arguments": {
    "symbols": ["FPT", "HPG"]
  }
}
```

Năm action cố định:

| Action | Arguments exact | Ý nghĩa |
|---|---|---|
| `GENERAL_CHAT` | `kind: greeting \| thanks \| capabilities \| clarify_symbol \| clarify_request` | Hội thoại ngắn hoặc hỏi lại bằng câu cố định |
| `STOCK_SIGNAL` | `symbols: list[str]` | Tín hiệu cho 1–5 mã; một action cho xem, phân tích và so sánh |
| `STOCK_RANKING` | `order: highest \| lowest`, `top_n: 1–10` | Xếp hạng Điểm UP |
| `PROJECT_INFO` | `topic: overview \| dataset \| features \| model \| evaluation \| inference \| limitations` | Giải thích đồ án từ nguồn thật hoặc contract cố định |
| `OUT_OF_SCOPE` | `reason: realtime \| news \| fundamentals \| trading_advice \| other` | Yêu cầu ngoài phạm vi |

Không có `direct_answer`, `focus`, `unsupported_symbol`, topic `method` hoặc action riêng cho comparison/best/worst.

### Quy tắc chuẩn hóa

- Symbol được trim, uppercase, bỏ trùng và giữ thứ tự; sau chuẩn hóa phải còn 1–5 mã.
- Ranking không nêu số lượng dùng `top_n=5`; “tốt nhất/xấu nhất” dùng `1`; số lớn hơn 10 được giới hạn thành `10` ngay ở decision.
- `top_n` phải là integer thật, không nhận boolean.
- Thiếu symbol dùng `GENERAL_CHAT {"kind":"clarify_symbol"}`; yêu cầu chưa rõ khác dùng `clarify_request`.
- Extra key, sai type/range/action hoặc JSON/tool call sai protocol trả `502`; không fallback sang router cục bộ.

## 4. Input và timeout

Provider nhận `DECISION_PROMPT`, tối đa 6 history entry đã validate và message hiện tại. Provider không nhận CSV, report, artifact, source code, domain-result JSON hoặc OpenAI tool schema.

Protection tại HTTP boundary:

- message: 1–1000 ký tự;
- history: tối đa 6 entry, tương đương 3 cặp gần nhất;
- mỗi history content: 1–1000 ký tự; tổng tối đa 6000;
- mỗi entry có đúng `role`, `content`; role `user`/`assistant` xen kẽ.

`TOTAL_DEADLINE_SECONDS` bao trùm provider call. `time.monotonic()` tính remaining time trước call và kiểm lại sau call; hết hạn trả `504`. SDK không retry.

## 5. Dispatcher, scope và ML

`execute_action()` chỉ chạy ba data action:

```text
STOCK_SIGNAL  → centralized symbol validation → predict_symbols()
STOCK_RANKING → predict_all_symbols() → filter/sort/limit
PROJECT_INFO  → static contract hoặc metadata/report/config
```

`GENERAL_CHAT` và `OUT_OF_SCOPE` dùng formatter cố định, không vào ML.

Symbol scope thuộc backend, không thuộc LLM. Với `STOCK_SIGNAL`, backend kiểm toàn bộ danh sách trước inference. Nếu có bất kỳ mã ngoài scope:

1. không gọi `predict_symbols()` cho mã nào;
2. trả một response deterministic liệt kê toàn bộ mã lỗi;
3. kèm warning `symbol_out_of_scope`.

ML vẫn chỉ nằm trong `services/prediction_service.py`. Chatbot không thay đổi feature, model, threshold hoặc phép tính `predict_proba()`.

## 6. Project topic và readiness

| Topic | Nguồn | Readiness |
|---|---|---|
| `overview` | Project contract cố định | Không cần model |
| `limitations` | Project contract cố định | Không cần model |
| `dataset` | Dataset report/config hiện hành | Pipeline đang ghi hoặc nguồn thiếu → `503` |
| `features` | Feature report/config hiện hành | Pipeline đang ghi hoặc nguồn thiếu → `503` |
| `model` | Metadata đã đối chiếu artifact | Cần model + metadata hợp lệ |
| `evaluation` | Metric thật trong metadata/report | Cần artifact hợp lệ; field thiếu thì bỏ, không tự tạo metric |
| `inference` | Metadata + contract prediction | Giải thích `predict_proba → score → decision_threshold → UP/NOT_UP` |

Static topic vẫn trả lời khi model chưa sẵn sàng. Data action khác chỉ chạy khi nguồn nó cần đã sẵn sàng.

## 7. Deterministic formatter

- Một symbol: hiển thị đầy đủ ngày tham chiếu, signal, Điểm UP, threshold và indicator có sẵn.
- Nhiều symbol cùng ngày: hiển thị gọn, sắp Điểm UP giảm dần; bằng điểm thì sort theo symbol.
- Nhiều symbol khác ngày: giữ thứ tự input, cảnh báo không so sánh trực tiếp.
- Ranking: sort theo `order`, tie-break bằng symbol, loại signal stale và giới hạn `top_n`.
- Project info: chỉ hiển thị field thật đang tồn tại.
- Stock/ranking hợp lệ luôn có disclaimer dữ liệu offline, không phải khuyến nghị đầu tư.

Formatter không gọi provider. `GENERAL_CHAT` không chứa prose do LLM tạo nên không cần regex hậu kiểm lời khuyên mua/bán.

## 8. API và frontend

Request:

```json
{
  "message": "Còn HPG?",
  "history": [
    {"role": "user", "content": "FPT thế nào?"},
    {"role": "assistant", "content": "FPT — dữ liệu ngày ..."}
  ]
}
```

Response công khai giữ nguyên năm field:

```json
{
  "answer": "...",
  "sources": [],
  "warnings": [],
  "data_as_of": null,
  "model_trained_through": null
}
```

| Status | Trường hợp |
|---|---|
| `400` | Payload/message/history sai |
| `502` | Provider lỗi hoặc decision sai protocol |
| `503` | Config hoặc nguồn cần cho action chưa sẵn sàng |
| `504` | Provider vượt deadline |
| `200` | Thành công, out-of-scope hoặc symbol ngoài scope |

Trang `/chat` và floating dock dùng chung `/api/chat`, transcript `sessionStorage` và history gần nhất. Cả hai render answer bằng `textContent`, không render HTML từ provider. Dock không xuất hiện tại `/chat` để tránh hai UI trùng nhau.

## 9. Scenario matrix

| # | User input | Decision mong đợi | Backend/response mong đợi |
|---:|---|---|---|
| 01 | “Xin chào” | `GENERAL_CHAT {kind:greeting}` | Câu chào cố định |
| 02 | “Cảm ơn nhé” | `GENERAL_CHAT {kind:thanks}` | Câu đáp cố định |
| 03 | “Bạn làm được gì?” | `GENERAL_CHAT {kind:capabilities}` | Liệt kê phạm vi cố định |
| 04 | “Dự đoán giúp tôi” | `GENERAL_CHAT {kind:clarify_symbol}` | Hỏi mã cần xem |
| 05 | “Phân tích đi” | `GENERAL_CHAT {kind:clarify_request}` | Hỏi lại yêu cầu ngắn |
| 06 | “FPT thế nào?” | `STOCK_SIGNAL {symbols:[FPT]}` | Một signal đầy đủ |
| 07 | “FPT với HPG thế nào?” | `STOCK_SIGNAL {symbols:[FPT,HPG]}` | Hai signal gọn |
| 08 | “So sánh FPT, HPG, VNM” | `STOCK_SIGNAL {symbols:[FPT,HPG,VNM]}` | So sánh từ dữ liệu thật, không action riêng |
| 09 | “FPT, fpt và HPG” | `STOCK_SIGNAL {symbols:[FPT,HPG]}` | Uppercase + bỏ trùng |
| 10 | “FPT và XYZ thế nào?” | `STOCK_SIGNAL {symbols:[FPT,XYZ]}` | Fail toàn request, liệt kê `XYZ`, không inference |
| 11 | “Còn HPG?” sau FPT | `STOCK_SIGNAL {symbols:[HPG]}` | History giải quyết follow-up |
| 12 | “So với MWG?” sau FPT | `STOCK_SIGNAL {symbols:[FPT,MWG]}` | Lấy mã cũ từ history |
| 13 | “Top cổ phiếu” | `STOCK_RANKING {order:highest,top_n:5}` | Mặc định top 5 |
| 14 | “Mã nào tốt nhất?” | `STOCK_RANKING {order:highest,top_n:1}` | Một mã cao nhất |
| 15 | “5 mã thấp nhất” | `STOCK_RANKING {order:lowest,top_n:5}` | Sort tăng dần |
| 16 | “Top 20” | `STOCK_RANKING {order:highest,top_n:10}` | Cap 10 |
| 17 | “Project làm gì?” | `PROJECT_INFO {topic:overview}` | Contract tĩnh |
| 18 | “Dataset từ đâu?” | `PROJECT_INFO {topic:dataset}` | Report/config thật |
| 19 | “Model dùng feature gì?” | `PROJECT_INFO {topic:features}` | Feature report thật |
| 20 | “Dùng model gì?” | `PROJECT_INFO {topic:model}` | Metadata đã đối chiếu artifact |
| 21 | “Kết quả evaluation?” | `PROJECT_INFO {topic:evaluation}` | Chỉ metric tồn tại |
| 22 | “Score và threshold dùng sao?” | `PROJECT_INFO {topic:inference}` | Giải thích luồng suy luận thật |
| 23 | “Hạn chế là gì?” | `PROJECT_INFO {topic:limitations}` | Contract tĩnh |
| 24 | “Vậy train thế nào?” sau model | `PROJECT_INFO {topic:model}` | History giữ topic |
| 25 | “Tin FPT hôm nay?” | `OUT_OF_SCOPE {reason:news}` | Từ chối cố định, không data handler |
| 26 | “Có nên mua FPT?” | `OUT_OF_SCOPE {reason:trading_advice}` | Không đưa lời khuyên |

Bảng trên là bản đại diện; bộ evaluator đầy đủ trong `scripts/evaluate_chatbot_decisions.py` có 33 scenario, dùng provider thật và chạy ngoài CI. `33/33` là target chất lượng prompt, không phải hard gate và không được ép bằng keyword router/hard-code.

## 10. Phạm vi cố ý không làm

Không RAG, embedding, Vector DB, LangChain/LangGraph, multi-agent, planner/executor, tool loop, chat database hoặc clarification state machine. History ngắn và fixed dispatcher đủ cho phạm vi niên luận.

## 11. File responsibility

```text
app.py
  validate HTTP payload → chatbot_service.chat()

services/chatbot_service.py
  prompt → decision validation → orchestration → formatter

services/chatbot_tools.py
  readiness → scope validation → fixed data handlers

services/prediction_service.py
  data/features/model inference

static/chat-client.js + templates/chat.html + static/chat-dock.js (floating dock)
  transcript, last-6 history, transport, safe DOM
```

## 12. Tài liệu LEGACY/STALE

Nội dung chatbot trong các đường dẫn sau là **LEGACY/STALE**, chỉ dùng tham khảo lịch sử; không dùng làm source of truth:

- `docs/slides/` (slide, outline và script thuyết trình còn mô tả SCI cũ);
- `docs/diagrams/soDoKienTruc/huong2-chatbot-rag.*` (sơ đồ SCI/RAG cũ).

`docs/GIAI_THICH_PROJECT.md`, `docs/SO_DO_KIEN_TRUC_HE_THONG.md`, bộ generator `docs/report_render/` cùng DOCX build từ nó đã được đồng bộ theo kiến trúc action-decision hiện hành, không còn LEGACY.

Canonical hiện hành: `docs/CHATBOT_ARCHITECTURE.md`, `docs/diagrams/luuDo/06-sequence-chatbot-action-decision.sequence.json`, HTML render cùng tên và index `docs/diagrams/luuDo/README.md`.

`docs/diagrams/pipeline-worklow/chatbot.workflow.json` và `chatbot-workflow.html` đã được render lại theo kiến trúc action-decision hiện hành, nên không còn LEGACY: workflow diagram này bổ sung góc nhìn swimlane cho sequence diagram canonical.

Notice tại nguồn cũ: `docs/slides/LEGACY_STALE_CHATBOT.md` và `docs/diagrams/soDoKienTruc/LEGACY_STALE_CHATBOT.md`. Nội dung legacy bên dưới hai nguồn đó không bị viết lại.

## 13. Test và nghiệm thu

```powershell
python -m pytest tests/test_chatbot.py
python -m pytest tests/test_prediction_flow.py
python -m pytest tests/
```

Hard gate: decision exact schema, one-call/no-tools, deadline, input/history bounds, centralized scope validation, stock 1–5 mã, ranking, 7 project topic, follow-up, API errors và safe DOM. Live-provider evaluator không nằm trong CI.

# GIẢI THÍCH ĐỂ BẢO VỆ NIÊN LUẬN

### 1. Chatbot giải quyết vấn đề gì?

Chatbot biến câu hỏi tự nhiên thành thao tác có sẵn của hệ thống dự đoán cổ phiếu, giúp người dùng không cần biết route hay cấu trúc dữ liệu.

### 2. Vì sao dùng LLM?

Cùng một ý có nhiều cách hỏi. LLM hiểu cách diễn đạt và follow-up tốt hơn danh sách keyword thủ công.

### 3. LLM được phép làm gì?

LLM chỉ chọn một trong năm action và trích xuất arguments đúng schema.

### 4. LLM không được phép làm gì?

LLM không tạo prediction, score, ranking, metric, số liệu dataset hoặc lời khuyên đầu tư.

### 5. Vì sao năm action hỗ trợ nhiều câu hỏi?

Mỗi action đại diện một nhóm nghiệp vụ; arguments chứa mã, thứ tự, số lượng hoặc topic. Vì vậy không cần action riêng cho từng cách hỏi.

### 6. Một request đi qua hệ thống thế nào?

API validate input, LLM tạo decision, backend validate lần nữa, dispatcher gọi handler cố định, rồi formatter viết response từ dữ liệu thật.

### 7. “FPT thế nào?” chạy ra sao?

LLM chọn `STOCK_SIGNAL` với `FPT`; backend kiểm scope, chạy inference và trình bày signal FPT.

### 8. “So sánh FPT và HPG” chạy ra sao?

LLM vẫn chọn `STOCK_SIGNAL`, lần này có hai symbol. Backend lấy hai signal và so sánh score thật nếu ngày dữ liệu tương thích.

### 9. “Top 5 cổ phiếu” chạy ra sao?

LLM chọn `STOCK_RANKING {order:highest, top_n:5}`; backend suy luận các mã hợp lệ, loại stale, sort và lấy năm dòng.

### 10. “Model train như thế nào?” chạy ra sao?

LLM chọn `PROJECT_INFO {topic:model}`; backend trả facts từ metadata/artifact hiện hành, không để LLM tự kể quy trình.

### 11. Nếu LLM chọn action sai thì sao?

Validator chặn decision sai schema. Decision đúng schema nhưng sai ý có thể cho kết quả không phù hợp; scenario evaluator dùng để cải thiện prompt, không thêm router thứ hai.

### 12. Vì sao không dùng RAG?

Phạm vi dữ liệu và action nhỏ, nguồn backend đã xác định. RAG thêm retrieval, dependency và nhiều luồng khó giải thích nhưng không giải quyết nhu cầu chính.

### 13. Vì sao không cho LLM viết số dự đoán?

LLM có thể bịa hoặc làm tròn sai. Backend lấy trực tiếp probability và threshold từ model nên kết quả kiểm thử và truy vết được.

### 14. Hạn chế hiện tại là gì?

Hệ thống dùng dữ liệu offline, không có tin tức/realtime/phân tích cơ bản; chất lượng hiểu câu phụ thuộc provider và prediction không phải bảo đảm giá tương lai.
