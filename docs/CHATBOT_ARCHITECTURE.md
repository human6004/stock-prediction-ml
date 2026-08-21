# Kiến trúc chatbot LLM Action-Decision

> **CANONICAL:** Tài liệu này và sơ đồ [chatbot_flow.png](diagrams/chatbot_flow.png) mô tả runtime chatbot hiện hành.
>
> Phạm vi: đồ án Niên luận cơ sở, dữ liệu cổ phiếu HOSE offline. LLM hiểu ngôn ngữ tự nhiên; backend sở hữu dữ liệu, ML và mọi câu trả lời có số liệu.

## 1. Luồng runtime

Hai giao diện dùng chung một API và một contract:

```text
User tại /chat hoặc floating dock
  → POST /api/chat
  → validate message + tối đa 3 cặp history gần nhất
  → LLM call 1 (decision) trả {action, arguments}
  → validate exact schema + chuẩn hóa arguments
  → fixed dispatcher
  → scope check + data handler / prediction_service
  → deterministic formatter (nguồn số liệu + fallback)
  → LLM call 2 (compose) diễn đạt lại answer từ JSON backend; lỗi thì giữ answer formatter
  → JSON response 5 field
```

Không có keyword/regex router, retry decision, tool loop hoặc conversation state riêng. Compose là LLM call thứ hai cố định (không loop) và phủ cả năm action: ba action dữ liệu chỉ compose khi có kết quả thật; `GENERAL_CHAT`/`OUT_OF_SCOPE` compose từ `kind`/`reason` kèm danh sách năng lực, không có số liệu nào. Luôn có fallback deterministic, tắt được bằng `CHATBOT_COMPOSE=0`.

## 2. Vai trò của LLM

LLM chỉ:

1. hiểu câu hiện tại trong ngữ cảnh history gần nhất;
2. chọn đúng một action trong whitelist;
3. trích xuất arguments theo schema cố định;
4. (compose) diễn đạt câu trả lời: action dữ liệu từ JSON số liệu backend đã tính; `GENERAL_CHAT`/`OUT_OF_SCOPE` từ `kind`/`reason` và danh sách năng lực, không có số liệu.

LLM không đọc file, gọi ML, chọn model/threshold, xác nhận symbol scope hoặc tự tạo số liệu. Probability, prediction, ranking, metric và số liệu dataset luôn đến từ backend; compose chỉ được dùng lại đúng các con số đó và disclaimer do backend gắn.

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
| `GENERAL_CHAT` | `kind: greeting \| thanks \| capabilities \| clarify_symbol \| clarify_request` | Hội thoại ngắn hoặc hỏi lại; compose diễn đạt, câu cố định làm fallback |
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

Call decision nhận `DECISION_PROMPT`, tối đa 6 history entry đã validate và message hiện tại — không nhận CSV, report, artifact, source code hoặc OpenAI tool schema. Call compose nhận `COMPOSE_PROMPT` và JSON `{question, action, data}` do backend tính — không nhận history; với `GENERAL_CHAT`/`OUT_OF_SCOPE`, `data` chỉ gồm `kind`/`reason` và `capabilities`, không có mã hay con số nào.

Protection tại HTTP boundary:

- message: 1–1000 ký tự;
- history: tối đa 6 entry, tương đương 3 cặp gần nhất;
- mỗi history content: 1–1000 ký tự; tổng tối đa 6000;
- mỗi entry có đúng `role`, `content`; role `user`/`assistant` xen kẽ.

`TOTAL_DEADLINE_SECONDS` bao trùm cả hai provider call. `time.monotonic()` tính remaining time trước mỗi call; decision hết hạn trả `504`, compose hết hạn thì bỏ qua và giữ answer formatter. SDK không retry.

## 5. Dispatcher, scope và ML

`execute_action()` chỉ chạy ba data action:

```text
STOCK_SIGNAL  → centralized symbol validation → predict_symbols()
STOCK_RANKING → predict_all_symbols() → filter/sort/limit
PROJECT_INFO  → static contract hoặc metadata/report/config
```

`GENERAL_CHAT` và `OUT_OF_SCOPE` không qua dispatcher và không vào ML: câu cố định trong `GENERAL_CHAT_MESSAGES`/`OUT_OF_SCOPE_MESSAGES` giờ là fallback, còn compose diễn đạt lời chào/từ chối từ `kind`/`reason` và capabilities.

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

## 7. Deterministic formatter và grounded compose

- Một symbol: hiển thị đầy đủ ngày tham chiếu, signal, Điểm UP, threshold và indicator có sẵn.
- Nhiều symbol cùng ngày: hiển thị gọn, sắp Điểm UP giảm dần; bằng điểm thì sort theo symbol.
- Nhiều symbol khác ngày: giữ thứ tự input, cảnh báo không so sánh trực tiếp.
- Ranking: sort theo `order`, tie-break bằng symbol, loại signal stale và giới hạn `top_n`.
- Project info: chỉ hiển thị field thật đang tồn tại.
- Stock/ranking hợp lệ luôn có disclaimer dữ liệu offline, không phải khuyến nghị đầu tư.

Formatter không gọi provider và là fallback khi compose thất bại. Sau formatter, cả năm action có thể đi qua compose: LLM nhận `{question, action, data}`, viết plain text tối đa 900 ký tự; backend gắn disclaimer (không gắn trùng). Action dữ liệu chỉ compose khi có kết quả thật; mã ngoài scope (`unsupported_symbols`) giữ câu deterministic, không compose. Compose lỗi, timeout, sai protocol hoặc quá dài → giữ nguyên answer formatter. Với `GENERAL_CHAT`/`OUT_OF_SCOPE`, an toàn dựa trên ba lớp thay vì regex hậu kiểm: (i) compose không nhận history và `data` không chứa mã hay con số nào, nên về cấu trúc không có gì để khuyến nghị; (ii) `COMPOSE_PROMPT` cấm khuyến nghị mua/bán và cấm nhận xét mã nào đáng mua; (iii) mọi lỗi đều rơi về câu từ chối cố định deterministic. Kiểm chứng bằng provider thật trên 10 câu thử tay (trong đó 4 câu dụ xin lời khuyên mua/bán, kể cả jailbreak “bỏ qua mọi quy tắc”): không câu nào đưa khuyến nghị, không câu nào nêu mã hay con số.

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

Trang `/chat` và floating dock dùng chung `/api/chat`, transcript `sessionStorage` và history gần nhất. Cả hai render answer bằng `textContent`, không render HTML từ provider, và chỉ hiển thị nội dung `answer`: `sources`, `warnings`, `data_as_of`, `model_trained_through` vẫn nằm trong JSON response nhưng không còn được render dưới câu trả lời. Dock không xuất hiện tại `/chat` để tránh hai UI trùng nhau.

## 9. Scenario matrix

| # | User input | Decision mong đợi | Backend/response mong đợi |
|---:|---|---|---|
| 01 | “Xin chào” | `GENERAL_CHAT {kind:greeting}` | Câu chào (compose, fallback cố định) |
| 02 | “Cảm ơn nhé” | `GENERAL_CHAT {kind:thanks}` | Câu đáp (compose, fallback cố định) |
| 03 | “Bạn làm được gì?” | `GENERAL_CHAT {kind:capabilities}` | Liệt kê năng lực (compose, fallback cố định) |
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
| 25 | “Tin FPT hôm nay?” | `OUT_OF_SCOPE {reason:news}` | Từ chối (compose, fallback cố định), không data handler |
| 26 | “Có nên mua FPT?” | `OUT_OF_SCOPE {reason:trading_advice}` | Từ chối, mời sang việc trong capabilities (compose, fallback cố định) |
| 27 | “Gợi ý mã đáng quan tâm” | `STOCK_RANKING {order:highest,top_n:5}` | Xếp hạng Điểm UP kèm disclaimer |
| 28 | “Bạn tự chọn giúp tôi” sau khi bot hỏi mã | `STOCK_RANKING {order:highest,top_n:5}` | Ranking thay vì lặp clarify |
| 29 | “Tại sao bạn chọn các mã này?” sau ranking | `PROJECT_INFO {topic:inference}` | Giải thích luồng suy luận, không clarify |

Bảng trên là bản đại diện; bộ evaluator đầy đủ trong `scripts/evaluate_chatbot_decisions.py` có 37 scenario (gồm hai case hỏi “tại sao chọn mã này” → `PROJECT_INFO {topic:inference}`), dùng provider thật và chạy ngoài CI. `37/37` là target chất lượng prompt, không phải hard gate và không được ép bằng keyword router/hard-code.

## 10. Phạm vi cố ý không làm

Không RAG, embedding, Vector DB, LangChain/LangGraph, multi-agent, planner/executor, tool loop, chat database hoặc clarification state machine. Compose là một call thẳng có fallback, không phải tool loop. History ngắn và fixed dispatcher đủ cho phạm vi niên luận.

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

## 12. Tài liệu hiện hành

Ba tài liệu dùng để đối chiếu kiến trúc hiện tại là `docs/CHATBOT_ARCHITECTURE.md`, `docs/GIAI_THICH_PROJECT.md` và `docs/SO_DO_KIEN_TRUC_HE_THONG.md`. Hai ảnh tổng quan tại `docs/diagrams/pipeline_flow.png` và `docs/diagrams/chatbot_flow.png` minh họa luồng pipeline và chatbot.

## 13. Test và nghiệm thu

```powershell
python -m pytest tests/test_chatbot.py
python -m pytest tests/test_prediction_flow.py
python -m pytest tests/
```

Hard gate: decision exact schema, decision+compose cố định (không tool loop, compose luôn có fallback deterministic), deadline chung hai call, input/history bounds, centralized scope validation, stock 1–5 mã, ranking, 7 project topic, follow-up, API errors và safe DOM. Live-provider evaluator không nằm trong CI.

# GIẢI THÍCH ĐỂ BẢO VỆ NIÊN LUẬN

### 1. Chatbot giải quyết vấn đề gì?

Chatbot biến câu hỏi tự nhiên thành thao tác có sẵn của hệ thống dự đoán cổ phiếu, giúp người dùng không cần biết route hay cấu trúc dữ liệu.

### 2. Vì sao dùng LLM?

Cùng một ý có nhiều cách hỏi. LLM hiểu cách diễn đạt và follow-up tốt hơn danh sách keyword thủ công.

### 3. LLM được phép làm gì?

LLM chọn một trong năm action, trích xuất arguments đúng schema, và diễn đạt câu trả lời (compose): với action dữ liệu là từ JSON số liệu backend đưa, với lời chào/từ chối là từ `kind`/`reason` và danh sách năng lực.

### 4. LLM không được phép làm gì?

LLM không tạo prediction, score, ranking, metric, số liệu dataset hoặc lời khuyên đầu tư. Compose chỉ được diễn đạt lại các số backend đã tính; ở nhánh chào/từ chối, compose không nhận history và `data` không chứa mã hay con số nào nên không có gì để khuyến nghị, prompt cũng cấm khuyến nghị mua/bán; lỗi thì hệ thống trả bản formatter cố định.

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

LLM có thể bịa hoặc làm tròn sai nếu tự tính. Mọi con số do backend lấy trực tiếp từ model; compose bị ràng buộc chỉ dùng số trong JSON, giới hạn độ dài, và mọi lỗi đều rơi về formatter cố định nên kết quả vẫn kiểm thử, truy vết được. Nhánh chào/từ chối cũng qua compose nhưng bị bỏ đói dữ liệu: không history, không mã, không số. Văn phong mỗi lần có thể khác nhau nhưng số liệu và disclaimer luôn giống nhau.

### 14. Hạn chế hiện tại là gì?

Hệ thống dùng dữ liệu offline, không có tin tức/realtime/phân tích cơ bản; chất lượng hiểu câu phụ thuộc provider và prediction không phải bảo đảm giá tương lai.
