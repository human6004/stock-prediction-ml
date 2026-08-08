# Kiến trúc chatbot Action Decision

> Phạm vi: đồ án Niên luận cơ sở, dữ liệu cổ phiếu HOSE offline.
> Mục tiêu: LLM hiểu câu tự nhiên và chọn action; backend kiểm soát toàn bộ dữ liệu, ML và câu trả lời có số liệu.

## 1. Kết luận ngắn

Chatbot dùng đúng một LLM call để trả JSON decision. Backend validate decision, gọi một nhánh cố định, rồi format kết quả bằng code. Không gọi LLM lần hai.

```text
User /chat
  → POST /api/chat
  → validate message + history
  → LLM trả JSON decision
  → validate action + arguments
  → fixed dispatcher
  → data handler / prediction_service
  → deterministic formatter
  → JSON response
```

Sơ đồ sequence canonical:

- [06-sequence-chatbot-action-decision.html](diagrams/luuDo/06-sequence-chatbot-action-decision.html)
- Nguồn: `diagrams/luuDo/06-sequence-chatbot-action-decision.sequence.json`

## 2. Vì sao cần LLM

Người dùng có thể hỏi cùng một ý theo nhiều cách: “Dự đoán FPT”, “FPT 5 phiên tới thế nào?”, hoặc “Còn mã đó thì sao?”. Viết keyword/if-else cho mọi cách diễn đạt khó bảo trì.

LLM chỉ làm ba việc trong phạm vi hẹp:

1. hiểu câu hiện tại cùng tối đa 6 history message;
2. chọn đúng một action và arguments trong whitelist;
3. chỉ với `GENERAL_CHAT`, viết câu chào hoặc câu hỏi làm rõ ngắn trong `direct_answer`.

LLM không đọc file, không chạy ML, không chọn model/threshold, không tự lấy dữ liệu và không viết lại câu trả lời dữ liệu.

## 3. Năm action cố định

| Action | Arguments | Ý nghĩa |
|---|---|---|
| `GENERAL_CHAT` | `{}` | Chào hỏi hoặc hỏi lại một thông tin còn thiếu |
| `STOCK_SIGNAL` | `symbols`, `focus` | Thông tin, prediction, analysis hoặc comparison cho 1–2 mã |
| `STOCK_RANKING` | `order`, `top_n` | Xếp hạng Điểm UP cao/thấp nhất |
| `PROJECT_INFO` | `topic` | Giải thích project, model, dataset, features, method, limitations |
| `OUT_OF_SCOPE` | `reason` | Realtime, news, fundamentals, advice hoặc yêu cầu ngoài phạm vi |

Không tách `GET_STOCK_INFO`, `PREDICT_STOCK`, `ANALYZE_STOCK`: cả ba cùng dùng `prediction_service.predict_symbols()`, chỉ khác `focus` và phần formatter.

## 4. LLM input

Provider nhận:

- một `DECISION_PROMPT` ngắn;
- whitelist 5 action và schema arguments;
- tối đa 6 history message đã validate;
- user message hiện tại.

Provider không nhận:

- CSV/raw data;
- model artifact hoặc report;
- source code;
- context JSON dữ liệu;
- OpenAI tool schema.

Request không có `tools`, `tool_choice` hoặc `response_format`. SDK đặt `max_retries=0`; JSON sai trả lỗi ngay, không retry/fallback router.

## 5. Decision contract

Ví dụ:

```json
{
  "action": "STOCK_SIGNAL",
  "arguments": {
    "symbols": ["FPT"],
    "focus": "prediction"
  },
  "direct_answer": null
}
```

Top-level phải đúng ba khóa. Validation chính:

- `symbols`: 1–2 chuỗi, trim, uppercase, không trùng;
- `focus`: `info`, `prediction`, `analysis`, `comparison`;
- `comparison`: đúng 2 mã;
- `order`: `highest` hoặc `lowest`;
- `top_n`: integer thật, không nhận bool, từ 1 đến 10;
- `topic`: `overview`, `model`, `dataset`, `features`, `method`, `limitations`;
- `reason`: `realtime`, `news`, `fundamentals`, `trading_advice`, `unsupported_symbol`, `other`;
- `direct_answer`: chỉ dùng cho `GENERAL_CHAT`, dài 1–1000 ký tự.

Thiếu mã/tiêu chí: LLM dùng `GENERAL_CHAT` và hỏi lại đúng một câu ngắn. Không có action `CLARIFY`.

## 6. Dispatcher và ML

`services/chatbot_tools.py::execute_action()` chỉ nhận ba data action:

```text
STOCK_SIGNAL  → _stock_signal()
STOCK_RANKING → _stock_ranking()
PROJECT_INFO  → _project_info()
```

`GENERAL_CHAT` và `OUT_OF_SCOPE` được format ngay, không gọi data handler.

ML chỉ nằm trong `services/prediction_service.py`:

- `predict_symbols()` cho 1–2 mã;
- `predict_all_symbols()` cho ranking;
- `predict_proba()` + threshold artifact tạo `UP`/`NOT_UP`.

LLM không trực tiếp dự đoán giá hoặc Điểm UP.

## 7. Readiness và symbol scope

Trước data action, `_load_runtime_state()` kiểm:

1. pipeline không chạy;
2. model artifact load được và có model;
3. metadata JSON parse được;
4. metadata khớp artifact theo logic của `prediction_service.load_metadata()`;
5. symbol scope không rỗng.

Release hiện tại là legacy và metadata chưa có `training_symbols`, nên scope fallback sang `reports/eligible_symbols.csv`. Response có warning `symbol_scope_unverified`; đây không phải lỗi inference.

Mã ngoài scope bị chặn trước ML, trả HTTP 200 cùng warning deterministic.

## 8. Domain result

Handler trả dict đơn giản:

```text
data
sources
warnings
data_as_of
model_trained_through
error
```

Không có envelope 7-field, fingerprint cross-handler hoặc release state machine.

Hai mốc ngày khác nhau:

- `data_as_of`: ngày dữ liệu offline dùng cho signal, ranking hoặc dataset;
- `model_trained_through`: ngày cutoff huấn luyện của model.

Không dùng hai mốc thay thế cho nhau.

## 9. Deterministic formatter

Response dữ liệu được viết bằng code:

- stock info: ngày, giá tham chiếu, return/volatility/volume ratio;
- prediction: label, Điểm UP, threshold;
- analysis: prediction + quan hệ score/threshold + indicators;
- comparison: hai signal + gap do handler tính từ score thô;
- ranking: symbol, Điểm UP, label, ngày;
- project info: facts từ metadata/report/config.

Formatter không gọi provider. Số hiển thị lấy trực tiếp từ domain result; thiếu field phụ thì bỏ dòng. Kết quả stock/ranking hợp lệ luôn kèm disclaimer offline.

`GENERAL_CHAT` có một gate nhỏ: nếu câu LLM chứa lời mua/bán trực tiếp, backend thay bằng câu từ chối cố định.

## 10. API contract

Request:

```json
{
  "message": "So với MWG?",
  "history": [
    {"role": "user", "content": "Dự đoán FPT"},
    {"role": "assistant", "content": "FPT: UP"}
  ]
}
```

Response:

```json
{
  "answer": "...",
  "sources": [],
  "warnings": [],
  "data_as_of": null,
  "model_trained_through": null
}
```

HTTP status:

| Status | Trường hợp |
|---|---|
| `400` | Payload/message/history sai |
| `502` | Provider lỗi hoặc decision sai protocol |
| `503` | LLM config, model, metadata hoặc report chưa sẵn sàng |
| `504` | Provider timeout |
| `200` | Out-of-scope hoặc symbol ngoài scope, kèm câu trả lời deterministic |

## 11. Frontend

Chỉ có một UI tại `/chat`:

- transcript lưu trong `sessionStorage` của tab, tối đa 40 entry;
- request chỉ gửi 6 entry cuối;
- không có custom conversation state;
- không có floating dock;
- answer dùng `textContent`, không render HTML từ LLM;
- giữ loading, timeout, public error, keyboard và focus cơ bản.

## 12. Ba ví dụ

### “Dự đoán FPT”

```text
LLM → STOCK_SIGNAL {symbols:[FPT], focus:prediction}
→ scope check
→ predict_symbols([FPT])
→ prediction formatter
```

### “Top 3 mã thấp nhất”

```text
LLM → STOCK_RANKING {order:lowest, top_n:3}
→ predict_all_symbols()
→ filter scope/nonfinite/stale
→ sort + limit
→ ranking formatter
```

### “Tin tức FPT hôm nay?”

```text
LLM → OUT_OF_SCOPE {reason:news}
→ fixed response
→ không gọi data service hoặc ML
```

## 13. Phạm vi cố ý không làm

Project không dùng RAG, embedding, Vector DB, LangChain/LangGraph, multi-agent, planner/executor, tool loop, chat database hoặc memory phức tạp. Những thành phần đó không cần cho mục tiêu niên luận và làm luồng khó giải thích hơn.

## 14. File responsibility

```text
app.py
  validate HTTP payload → chatbot_service.chat()

services/chatbot_service.py
  LLM decision → validation → orchestration → formatter

services/chatbot_tools.py
  readiness → fixed dispatcher → read-only handlers

services/prediction_service.py
  data/features/model inference

static/chat-client.js + templates/chat.html
  transcript, last-6 history, transport, safe DOM
```

## 15. Test

```powershell
python -m pytest tests/test_chatbot.py
python -m pytest tests/test_prediction_flow.py
python -m pytest tests/
```

Chatbot tests dùng fake provider; không cần network. Behavior chính được khóa: 5 action, malformed JSON, one-call/no-tools, scope trước inference, ranking filter/sort, 6 project topic, follow-up qua history, API errors và safe DOM.
