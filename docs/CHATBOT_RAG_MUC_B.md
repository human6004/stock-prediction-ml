# Chatbot Hỏi–Đáp Dữ liệu và Mô hình HOSE — Phương án Mức B

> Tài liệu thiết kế cho **Tool-grounded / Structured RAG**. Chưa phải code triển khai.
> Phạm vi: trợ lý hỏi–đáp về dữ liệu offline, tín hiệu ML và báo cáo của project.

---

## 1. Quyết định kiến trúc

Project dùng **structured retrieval qua tool/function calling**, không dùng vector database.

Lý do:

- Nguồn chính là hàm Python, JSON và CSV có schema rõ ràng.
- Câu trả lời cần lấy đúng mã, đúng ngày, đúng metric và đúng release model.
- Vector search phù hợp hơn với kho văn bản dài, không cấu trúc; project hiện chưa có kho đó.
- Gọi trực tiếp service hiện có ngắn hơn, dễ kiểm thử hơn và không cần thêm hạ tầng embedding.

Tên gọi nên dùng trong báo cáo:

> Hệ thống áp dụng **tool-grounded question answering với structured retrieval**
> (một dạng retrieval-augmented generation), trong đó LLM chọn tool, còn server kiểm soát
> việc truy xuất dữ liệu có cấu trúc và thực thi hàm.

Không nên viết rằng Tool RAG “đúng 100%”. Tool lấy đúng giá trị từ nguồn đã chọn, nhưng nguồn
có thể cũ, artifact có thể lệch policy và model vẫn có sai số dự báo.

---

## 2. Mục tiêu và phạm vi

### 2.1. Mục tiêu

- Trả lời tiếng Việt dựa trên dữ liệu thật của project.
- Diễn giải tín hiệu `UP` / `NOT_UP`, Điểm UP, model, metric, dataset và feature.
- Nêu rõ ngày tham chiếu, trạng thái dữ liệu offline, policy và cảnh báo baseline.
- Không để LLM tự đọc file tùy ý hoặc tự tạo số liệu ngoài tool result.

### 2.2. Trong phạm vi MVP

- Tín hiệu của một hoặc hai mã HOSE đủ điều kiện.
- So sánh Điểm UP của hai mã khi điều kiện so sánh hợp lệ.
- Xếp hạng tín hiệu theo Điểm UP.
- Model đang phục vụ, decision threshold, target, metric và baseline.
- Số lượng mã, khoảng ngày, ngày dữ liệu mới nhất và train cutoff.
- Danh sách 20 technical feature và feature importance toàn cục.

### 2.3. Ngoài phạm vi MVP

- Giá realtime, dữ liệu intraday hoặc lịch giao dịch realtime.
- Tin tức, báo cáo tài chính, P/E, EPS, cổ tức, ngành và dữ liệu cơ bản doanh nghiệp.
- Dự báo giá cụ thể hoặc mức lợi nhuận tương lai.
- Giải thích cục bộ kiểu “FPT được dự báo UP vì RSI”; project chưa có local explanation/SHAP.
- Khuyến nghị mua, bán, phân bổ vốn hoặc tư vấn đầu tư cá nhân.
- Mã ngoài allowlist của project và truy cập file tùy ý.

Tên hiển thị phù hợp hơn **“Trợ lý dữ liệu và mô hình HOSE”**. Tên “chatbot thông tin chứng
khoán” quá rộng so với dữ liệu thật hiện có.

---

## 3. Hiện trạng project cần bám theo

### 3.1. Thành phần có thể tái sử dụng

| Nhu cầu | Thành phần hiện có |
|---|---|
| Dự báo 1–2 mã | `services.prediction_service.predict_symbols()` |
| Wrapper một mã | `services.prediction_service.predict_symbol()` |
| Xếp hạng toàn sàn | `services.prediction_service.predict_all_symbols()` |
| Metadata model | `models/model_metadata.json` + `load_metadata()` |
| Thống kê pipeline | `reports/pipeline_summary.json` |
| So sánh model | `reports/model_comparison.csv` |
| TEST cuối | `reports/final_model_evaluation.csv` |
| Feature importance | `reports/feature_importance.csv` |
| Danh sách mã đủ điều kiện | `reports/eligible_symbols.csv` |
| OHLCV sạch | `data/processed/hose_stock_clean.csv` |

`predict_symbols()` đã batch tối đa hai mã. Chatbot phải tái sử dụng hàm này cho câu hỏi so
sánh, không gọi `predict_symbol()` hai lần rồi lặp việc đọc dữ liệu/build feature.

### 3.2. Snapshot tại lúc rà soát tài liệu

Tại thời điểm rà soát `2026-07-22`:

- Policy trong code: `rolling_recent_cv_oof_threshold`.
- Artifact/report đang phục vụ: `legacy_pre_validation_baseline_gate`.
- Artifact là Random Forest, decision threshold `0.49`.
- Artifact có `baseline_passed = false`.
- TEST F1_UP của artifact `0.375381`, Always-UP `0.383895`.
- Dữ liệu clean có 400 mã; 396 mã đủ điều kiện train trong snapshot hiện tại.

Các số trên chỉ mô tả snapshot, **không được hardcode vào prompt hoặc câu trả lời**. Tool phải
đọc trạng thái động mỗi request. Trước khi gọi chatbot là thành phần “hiện hành”, cần đồng bộ
policy, artifact, metadata, report và README thành cùng một release.

### 3.3. Quy tắc allowlist mã

MVP chỉ trả lời tín hiệu cho mã nằm trong `reports/eligible_symbols.csv` của snapshot hiện tại.

Lý do: inference service có thể tính feature cho một số mã thuộc `cleaned_all` nhưng từng bị
loại khỏi dữ liệu train. Nếu muốn hỗ trợ nhóm đó sau này, giao diện phải gọi đúng là “mã có đủ
dữ liệu inference”, không gọi là “mã đã train”.

Không fuzzy-match mã gần giống. Mã lạ hoặc gõ sai phải được báo lỗi rõ để tránh trả nhầm mã.

---

## 4. Mức A, Mức B và phương án chọn

| | Mức A — Context injection | Mức B — Tool calling |
|---|---|---|
| Chọn nguồn dữ liệu | Router/code | LLM đề nghị tool |
| Vòng gọi LLM | Thường 1 | Thường 2 trở lên |
| Provider | Chỉ cần chat | Phải hỗ trợ tool calling đúng model |
| Câu hỏi phức tạp | Cần thêm rule | Linh hoạt hơn |
| Chi phí và latency | Thấp hơn | Cao hơn |
| Rủi ro | Router thiếu case | Tool loop, schema, hallucination |

Khác biệt không chỉ là “ai chọn dữ liệu”. Mức B còn có vòng điều phối, validation tool call,
timeout, giới hạn vòng và nhiều failure mode hơn.

### Phương án chốt: Mức B-lite / hybrid

- Vẫn dùng function calling để thể hiện Mức B.
- Server giữ allowlist, schema, scope, release gate và warning cố định.
- Dùng batch tool cho so sánh.
- Câu trả lời định lượng chỉ được tạo sau ít nhất một tool call thành công.
- Nếu provider không hỗ trợ tool calling, fallback về Mức A; vẫn không cần Vector RAG.

`base_url + api_key` chỉ cho biết endpoint có thể tương thích chat. Nó chưa chứng minh model
đó hỗ trợ `tools`, `tool_call_id`, nhiều tool call hoặc đúng schema. Phải kiểm tra provider thật
trước khi triển khai.

---

## 5. Kiến trúc tổng thể và ranh giới tin cậy

```text
User / Browser
      │
      ▼
Flask GET /chat + POST /api/chat
      │ validate request, giới hạn độ dài
      ▼
chatbot_service
      ├── gửi prompt + tool schemas ───────────────► LLM provider bên ngoài
      │                                               │
      │◄──────────── tool calls / final text ──────────┘
      │
      ├── validate tên tool + arguments
      ▼
chatbot_tools (allowlist)
      ├── prediction_service
      ├── serving artifact + metadata
      ├── reports có cấu trúc
      └── eligible symbols / dataset metadata
```

Ranh giới tin cậy:

- LLM không được thực thi Python, đọc path hoặc gọi hàm tùy ý.
- Server chỉ chạy tool có trong allowlist.
- Mọi argument phải qua schema và kiểm tra business rule.
- Chỉ field cần thiết của tool result được gửi ra provider.
- API key, model pickle và toàn bộ CSV không được gửi tới provider.
- Câu hỏi user và tool result **có rời máy** để tới provider; phải mô tả đúng điều này trong
  phần privacy của báo cáo.

---

## 6. Cấu trúc file dự kiến

```text
config/settings.py
  thêm LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, timeout và giới hạn

requirements.txt
  thêm đúng một client OpenAI-compatible nếu provider cần

.gitignore
  ignore .env nếu chọn dùng file .env

services/
  chatbot_service.py   điều phối LLM, tool loop, grounding và lỗi provider
  chatbot_tools.py     tool schemas, validation business rule, dữ liệu trả về rút gọn

app.py
  GET /chat            render trang
  POST /api/chat       API JSON

templates/chat.html
  UI chat tối giản; render text an toàn

static/app.css
  bổ sung style cần thiết, tái sử dụng design system hiện tại

tests/
  test_chatbot_tools.py
  test_chatbot_service.py
  test_chatbot_routes.py
```

Project chưa dùng base template; link “Trợ lý AI” phải được thêm vào nav của các template cần
thiết, không chỉ `chat.html`.

Không thêm DB chat, vector store, framework agent, SPA hoặc streaming trong MVP.

### Cấu hình secret

Hai lựa chọn hợp lệ:

1. Dùng biến môi trường của hệ điều hành/hosting: không cần `.env` loader.
2. Dùng `.env` local: phải ignore `.env` và có cơ chế load rõ ràng.

Thiếu API key chỉ làm `/api/chat` trả lỗi cấu hình; không được làm toàn bộ Flask app không
khởi động được.

---

## 7. Tool contract

### 7.1. `get_stock_signals`

| Thuộc tính | Contract |
|---|---|
| Mục đích | Dự báo một mã hoặc batch so sánh hai mã |
| Bọc quanh | `prediction_service.predict_symbols()` |
| Input | `symbols`: mảng 1–2 mã, uppercase sau normalize |
| Validation | Không rỗng, không trùng, thuộc eligible allowlist |
| Output chính | symbol, reference_date, close, label, up_score, decision_threshold, target, technical context |
| Warning | stale, forecast elapsed, baseline, policy/release |

`probability_up` nội bộ phải được đổi tên thành `up_score` trong payload chatbot để LLM không
diễn giải nhầm thành xác suất đã calibration.

### 7.2. `get_ranking`

| Thuộc tính | Contract |
|---|---|
| Mục đích | Lấy vài mã có Điểm UP cao nhất hoặc thấp nhất |
| Bọc quanh | `prediction_service.predict_all_symbols()` |
| Input | `top_n`: 1–10; `order`: `highest_up_score` hoặc `lowest_up_score` |
| Filter | Chỉ eligible symbols; mặc định loại score `None` |
| Output | Tối đa `top_n` dòng đã sort tại server |

Không trả toàn bộ danh sách hàng trăm mã cho LLM. Không dùng `direction=NOT_UP` mơ hồ vì
“NOT_UP mạnh nhất” có thể hiểu là score thấp nhất hoặc gần threshold nhất.

### 7.3. `get_model_info`

Trả về:

- model đang phục vụ;
- target: tăng hơn 1% tại đúng phiên thị trường `t+5`;
- decision threshold;
- feature order;
- train-through date và trained-at;
- policy ID, content fingerprint và release status;
- validation/TEST metrics nếu cùng release;
- baseline status và warning động.

Tool không được coi riêng `model_selection_report.txt` là nguồn tuyệt đối. Phải ưu tiên serving
artifact + `load_metadata()`, rồi chỉ ghép report khi identity/policy/fingerprint khớp.

### 7.4. `get_dataset_info`

Trả về:

- dữ liệu offline hay realtime;
- ngày dữ liệu mới nhất;
- số mã clean, eligible và excluded;
- khoảng ngày dữ liệu;
- số feature;
- train cutoff của artifact;
- thời điểm report/pipeline nếu có.

Phải tách `data_as_of` khỏi `model_trained_through`. Dữ liệu inference có thể mới hơn dữ liệu
đã dùng fit model.

### 7.5. `get_feature_info`

| Thuộc tính | Contract |
|---|---|
| Input | `top_n`: 1–20 |
| Output | feature order, mô tả ngắn và importance toàn cục nếu có |
| Điều kiện | Feature importance phải thuộc cùng serving release |

Tool phải nói rõ importance là **toàn cục**, không phải nguyên nhân riêng cho một mã và không
chứng minh quan hệ nhân quả.

### 7.6. Envelope chung

Mỗi tool trả một object chuẩn gồm:

| Field | Ý nghĩa |
|---|---|
| `ok` | Thành công hay thất bại |
| `data` | Payload đã rút gọn |
| `source` | Loại nguồn nội bộ, không phải path tùy ý |
| `as_of` | Ngày/thời điểm dữ liệu |
| `release` | policy, fingerprint, status |
| `warnings` | stale, baseline, mismatch hoặc giới hạn diễn giải |
| `error` | Mã lỗi + thông báo an toàn khi `ok=false` |

Tool error không được làm sập route và không trả stack trace cho user.

---

## 8. HTTP và UI contract

### 8.1. Route

- `GET /chat`: trang Trợ lý AI.
- `POST /api/chat`: nhận/trả JSON.

Không dùng cùng `POST /chat` cho cả HTML và JSON; tách route giúp contract rõ hơn.

### 8.2. Request MVP

```json
{
  "message": "So sánh FPT với VNM"
}
```

MVP single-turn, chưa có `conversation_id`. Giới hạn message khoảng 1.000 ký tự.

### 8.3. Response

```json
{
  "answer": "Câu trả lời tiếng Việt đã grounding",
  "sources": [
    {"kind": "stock_signal", "as_of": "YYYY-MM-DD"}
  ],
  "warnings": [],
  "release_status": "current"
}
```

`answer` là plain text hoặc Markdown bị giới hạn. UI phải escape output; không đưa text LLM
thẳng vào `innerHTML`.

### 8.4. HTTP status tối thiểu

| Status | Khi dùng |
|---|---|
| `200` | Trả lời hoặc từ chối ngoài phạm vi hợp lệ |
| `400` | Request sai schema / message rỗng / quá dài |
| `502` | Provider trả lỗi |
| `503` | Chưa cấu hình LLM |
| `504` | Hết timeout tổng |
| `500` | Lỗi nội bộ; response không lộ chi tiết |

---

## 9. Vòng điều phối và grounding enforcement

Luồng chuẩn:

1. Validate JSON, độ dài câu hỏi và content type.
2. Gửi system prompt, user message và tool schemas tới provider.
3. Nếu LLM yêu cầu tool, validate tên tool và arguments.
4. Thực thi tool, nhận envelope và thêm kết quả rút gọn vào messages.
5. Lặp đến khi LLM trả final text hoặc chạm budget.
6. Server ghép warnings/provenance cố định vào response.

Budget MVP đề xuất:

- tối đa 3 vòng LLM;
- tối đa 5 tool call cho một request;
- tổng timeout khoảng 30 giây;
- `top_n <= 10`, tối đa 2 mã cho signal comparison.

### Quy tắc bắt buộc

- Tên tool ngoài allowlist: từ chối thực thi.
- Arguments thừa/sai type/sai enum: tool error có cấu trúc.
- Không có tool call thành công: không chấp nhận câu trả lời định lượng của LLM; trả thông báo
  phạm vi cố định hoặc lỗi phù hợp.
- Không hardcode `baseline_passed=false`; luôn lấy trạng thái động.
- Warning baseline, stale, policy mismatch và disclaimer được server/UI gắn cố định.
- Không cho LLM tự biến `NOT_UP` thành “giảm”.
- Không cho LLM tự gọi Điểm UP là xác suất chắc chắn.

System prompt vẫn cần, nhưng chỉ là một lớp. Grounding chính nằm ở allowlist, validation, source
identity, response provenance và test.

---

## 10. Ví dụ: “So sánh FPT với VNM”

Luồng đúng:

1. LLM yêu cầu một tool call `get_stock_signals` với `symbols=["FPT", "VNM"]`.
2. Server normalize, kiểm eligible allowlist rồi gọi `predict_symbols(["FPT", "VNM"])` một lần.
3. Tool trả hai Điểm UP, threshold, label, reference date và warnings.
4. Nếu reference date khác nhau, câu trả lời phải cảnh báo chưa thể so sánh trực tiếp hoàn toàn.
5. Nếu cùng ngày, LLM được nói mã nào có **Điểm UP cao hơn theo cùng model**.
6. Không được suy ra mã đó “nên mua”, “chắc tăng” hoặc có lợi nhuận cao hơn.

Mẫu cách diễn đạt:

> Theo dữ liệu offline tại ngày tham chiếu của từng mã, FPT có Điểm UP cao hơn VNM theo model
> đang phục vụ. Đây là điểm phân loại so với decision threshold, không phải xác suất chắc chắn
> và không phải khuyến nghị đầu tư.

Các con số cụ thể phải đến từ tool result tại thời điểm request, không ghi ví dụ số giả vào
prompt.

---

## 11. Release consistency và nguồn sự thật

### 11.1. Thứ tự ưu tiên

1. `final_model.pkl` đang phục vụ và metadata lấy qua `load_metadata()`.
2. `model_metadata.json` nếu identity khớp artifact.
3. Report CSV/JSON nếu policy/content fingerprint khớp release.
4. Dataset metadata hiện tại, tách riêng khỏi model release.

### 11.2. Trạng thái release

| Status | Điều kiện | Hành vi chatbot |
|---|---|---|
| `current` | Artifact, metadata, report và policy khớp | Trả lời bình thường |
| `legacy` | Artifact nhất quán nhưng policy cũ | Cho phép trả lời, hiện cảnh báo nổi bật |
| `inconsistent` | Artifact/metadata/report không khớp identity | Chặn số liệu model/report bị ảnh hưởng |
| `missing` | Thiếu artifact hoặc nguồn bắt buộc | Báo chưa sẵn sàng, không bịa fallback |

Không ghép metric từ report cũ với tín hiệu từ artifact mới.

---

## 12. Quy tắc diễn đạt tài chính

- Dùng **Điểm UP**, không dùng “xác suất tăng” nếu model chưa được calibration.
- `UP`: model dự báo đạt điều kiện tăng hơn 1% tại horizon 5 phiên.
- `NOT_UP`: không đạt điều kiện `UP`; không đồng nghĩa chắc chắn giảm.
- Giá trả về là “giá đóng cửa theo dữ liệu local tại ngày tham chiếu”, không gọi là giá hiện tại.
- Forecast dates trong service chỉ là mốc ước tính; không khẳng định đã xử lý đầy đủ ngày nghỉ HOSE.
- Feature importance là toàn cục, không phải nguyên nhân hay giải thích riêng từng mã.
- Luôn nêu dữ liệu offline và ngày tham chiếu khi trả tín hiệu.
- Câu hỏi “nên mua/bán không” phải từ chối khuyến nghị, sau đó có thể cung cấp dữ liệu model nếu
  user nêu mã cụ thể.
- Kết quả phục vụ học tập/nghiên cứu, không phải tư vấn đầu tư.

---

## 13. Bảo mật, privacy và độ tin cậy

### Bắt buộc cho local MVP

- API key chỉ ở server qua environment; không nhúng HTML/JS và không log.
- `.env` phải nằm trong `.gitignore` nếu dùng.
- Bind `127.0.0.1`; không expose Internet khi chưa có auth và rate limit.
- Validate input, symbol, `top_n`, enum và JSON schema.
- Tool map cố định; không hỗ trợ path/function do LLM truyền vào.
- Escape output LLM để tránh XSS.
- Lỗi trả message chung cho user; chi tiết chỉ ở server log đã redact.
- Timeout tổng và giới hạn vòng/call.
- Chỉ gửi các field cần thiết tới provider.

### Nếu triển khai ra ngoài local

Cần bổ sung authentication, HTTPS, rate limiting theo user/IP, CORS rõ ràng, CSRF phù hợp cơ
chế session, logging/monitoring, secret manager và chính sách lưu dữ liệu của provider.

---

## 14. Trạng thái hội thoại

MVP dùng **single-turn stateless**:

- mỗi request độc lập;
- không lưu lịch sử vào DB;
- không cần server session;
- đơn giản để kiểm thử và viết báo cáo.

Nếu cần follow-up như “còn VNM thì sao?”, phase sau cho browser gửi một history ngắn đã giới
hạn số lượt và tổng ký tự. Không lưu history trong cookie Flask lớn và không thêm DB trước khi
có yêu cầu thật.

---

## 15. Error và fallback

| Tình huống | Hành vi |
|---|---|
| Mã không tồn tại / không eligible | Báo rõ, không tự đoán mã gần giống |
| Thiếu model/report | Tool `ok=false`; UI báo chưa sẵn sàng |
| Policy legacy | Trả dữ liệu kèm warning động |
| Identity inconsistent | Chặn dữ liệu bị ảnh hưởng |
| Provider timeout/lỗi | Trả `502/504`, không tạo câu trả lời giả |
| Vượt số vòng | Dừng và báo không hoàn tất |
| Tool trả score `None` | Nêu model không cung cấp Điểm UP cho kết quả đó |
| LLM không gọi tool | Không cho trả số liệu in-scope |

Fallback bằng `summary_sentence` nội bộ chỉ hợp câu hỏi dự báo đơn đã có tool result. Không dùng
summary cũ hoặc trí nhớ LLM để thay dữ liệu bị thiếu.

---

## 16. Tiêu chí nghiệm thu

### 16.1. Đúng dữ liệu

- Mọi số trong câu trả lời khớp tool result.
- Không có metric/giá/Điểm UP tự sinh ngoài nguồn.
- Prediction luôn kèm reference date và trạng thái offline.
- Baseline, policy và stale warning khớp trạng thái động.
- Không trộn artifact và report khác release.

### 16.2. Đúng ngữ nghĩa

- Không gọi Điểm UP là xác suất chắc chắn.
- Không gọi `NOT_UP` là giảm.
- Không gọi giá local là realtime.
- Không biến global feature importance thành local explanation.
- Không đưa khuyến nghị mua/bán.

### 16.3. Đúng tool

Bộ câu hỏi chuẩn phải phủ:

- một mã;
- hai mã;
- ranking cao/thấp;
- model và baseline;
- dataset;
- feature;
- mã lạ/gõ sai;
- ngoài phạm vi;
- prompt injection yêu cầu bỏ qua rule;
- provider timeout;
- tool error;
- policy mismatch.

### 16.4. Chỉ số đánh giá cho báo cáo

- Tool-selection accuracy.
- Grounded numeric accuracy.
- Tỷ lệ câu trả lời có nguồn/ngày tham chiếu.
- Hallucinated-number rate.
- Scope-refusal accuracy.
- Latency p50/p95.
- Số vòng LLM, token và chi phí trung bình mỗi câu.

Trước khi merge chatbot, toàn bộ test hiện có và test chatbot phải xanh.

---

## 17. Checklist trước khi viết code

- [ ] Đồng bộ code policy, serving artifact, metadata, report và README.
- [ ] Chốt provider/model có tool calling thật.
- [ ] Chốt dùng environment thuần hay `.env` + loader.
- [ ] Chốt eligible-symbol allowlist.
- [ ] Chốt năm tool và JSON schema, gồm `additionalProperties: false`.
- [ ] Chốt request/response/status code.
- [ ] Chốt single-turn cho MVP.
- [ ] Chốt system prompt và warning do server gắn.
- [ ] Chốt timeout, vòng LLM, tool-call budget và `top_n`.
- [ ] Tạo bộ câu hỏi nghiệm thu và prompt-injection cases.
- [ ] Sửa test hiện tại đang fail trước khi tích hợp.

---

## 18. Chốt

- Kiến trúc phù hợp: **Mức B-lite / Tool-grounded Structured RAG**.
- Không cần Vector DB ở phạm vi hiện tại.
- Tái sử dụng `predict_symbols()`, `predict_all_symbols()` và report có cấu trúc.
- LLM chỉ chọn tool và diễn giải; server giữ toàn quyền validation, grounding, release gate,
  warning và security.
- Chatbot là trợ lý dữ liệu/model offline, không phải trợ lý đầu tư hay nguồn tin chứng khoán
  tổng quát.
- Nếu provider không hỗ trợ tool calling, dùng Mức A với cùng tool wrappers và cùng quy tắc
  grounding.

Việc ưu tiên trước implementation: **đồng bộ release hiện hành**, sau đó mới kiểm provider và
viết route/service/UI.
