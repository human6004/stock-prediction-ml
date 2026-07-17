# Giải thích 5 sơ đồ kiến trúc (v3)

Tài liệu này giải thích chi tiết 5 sơ đồ trong thư mục `docs/diagrams/v3/`, luồng hoạt
động của hệ thống và các thuật ngữ, để bạn hiểu và trình bày lại trong báo cáo niên luận.

Mỗi sơ đồ là 1 file `.html` tự chứa (mở bằng trình duyệt, có nút đổi nền sáng/tối và xuất
ảnh). File `.json` đi kèm là dữ liệu nguồn dùng để dựng sơ đồ.

| File | Loại | Trả lời câu hỏi |
|------|------|-----------------|
| `stock-architecture.html` | Kiến trúc hệ thống | Hệ thống gồm những khối nào, nối với nhau ra sao? |
| `stock-dataflow.html` | Luồng dữ liệu | Dữ liệu biến đổi thế nào từ thô đến dự đoán? |
| `stock-workflow.html` | Quy trình xử lý | Các bước chạy theo thứ tự nào, ai canh gác? |
| `stock-sequence.html` | Trình tự tương tác | Khi bấm "dự đoán", các thành phần gọi nhau ra sao? |
| `stock-lifecycle.html` | Vòng đời trạng thái | Mô hình đi qua những trạng thái nào đến khi phục vụ? |

---

## Bối cảnh hệ thống (đọc trước)

Đây là hệ thống **dự đoán xu hướng giá cổ phiếu HOSE** trong 5 phiên tới. Bài toán là
**phân loại nhị phân**: mỗi dòng dữ liệu (một mã cổ phiếu tại một ngày) được gán nhãn
`UP` nếu giá đóng cửa sau 5 phiên tăng hơn 1%, ngược lại là `NOT_UP`.

Hệ thống gồm 3 phần lớn:
1. **Pipeline offline** — chạy 1 lần để làm sạch dữ liệu, tạo đặc trưng, huấn luyện và
   chọn mô hình tốt nhất, lưu ra file.
2. **Tuning Lab** — giao diện web cho phép thử nghiệm siêu tham số từng mô hình thủ công
   trước khi chốt cấu hình cho pipeline chính thức.
3. **Serving** — web/CLI nạp mô hình đã lưu để dự đoán 1 mã cổ phiếu, không huấn luyện lại.

Công nghệ: **Flask** (web), **scikit-learn** (ML), **pandas** (xử lý bảng), **SQLite**
(lưu trữ phụ), lưu trữ chính là **file CSV/JSON/PKL**.

---

## 1. Sơ đồ Kiến trúc hệ thống — `stock-architecture.html`

**Mục đích:** cho thấy toàn cảnh các khối phần mềm và cách chúng kết nối. Đây là bức tranh
"tĩnh" — ai lưu gì, ai gọi ai.

### Luồng đọc sơ đồ
- **Góc trên trái → phải (thu thập dữ liệu):** `vnstock KBS` là nguồn dữ liệu ngoài; script
  `fetch_hose_data.py` kéo dữ liệu OHLCV về, ghi nối tiếp vào `Raw HOSE CSV` (nằm ngoài
  repo, trong `shared_dataset`, hiện 549084 dòng).
- **Khối giữa (xử lý):** `Pipeline` (`scripts/run_pipeline.py`) đọc raw + `Roadmap` +
  `Settings`, rồi điều phối `ML Services` (làm sạch, tạo đặc trưng, huấn luyện, đánh giá).
- **Tuning Lab & Experiment State:** `Tuning Lab` (`/tuning`) cho thử siêu tham số trên TRAIN,
  lưu kết quả qua `Experiment State` — nơi giữ lịch sử chạy, cấu hình đã chốt, và các khóa
  (lock). Experiment State cấp "cấu hình + khóa" cho Pipeline (mũi tên ngược lên).
- **Khối lưu trữ (phải):** `Processed CSVs` (dữ liệu sạch + đặc trưng + ml_dataset),
  `Model Store` (`final_model.pkl`), `Reports` (chỉ số + biểu đồ), và `SQLite DB` (bản
  sao đồng bộ). Nhóm 3 khối `processed/models/reports` nằm trong khung "regenerated offline
  artifacts" — nghĩa là bị xóa và tạo lại mỗi lần chạy pipeline.
- **Khối phục vụ (dưới trái):** `User` → `Flask Web` (3 trang) hoặc `CLI` → `Prediction
  Service`. Prediction Service đọc `Processed CSVs` (lấy lịch sử mã) + nạp `Model Store`,
  và ghi log dự đoán vào `SQLite` (best-effort).

### Ý nghĩa màu/kiểu khối
- Xanh dương nhạt = giao diện (frontend); xanh lá = dịch vụ xử lý (backend); tím/xanh cyan
  đậm = kho lưu trữ (database/file); hồng = bảo mật/canh gác (settings, experiment state);
  xám = thành phần ngoài hệ thống (vnstock, user, CLI).
- Mũi tên liền đậm = luồng chính; mũi tên đứt = luồng phụ/bất đồng bộ (đồng bộ DB, ghi log);
  mũi tên hồng = liên quan canh gác/cấu hình.

### Điểm cần nhấn khi trình bày
- Lưu trữ **chính là file**, SQLite chỉ là **bản sao (mirror)** — không phải nguồn sự thật
  để chọn mô hình.
- Raw data và vnstock **nằm ngoài repo** → hệ thống tách biệt phần dữ liệu dùng chung.

---

## 2. Sơ đồ Luồng dữ liệu — `stock-dataflow.html`

**Mục đích:** theo dõi dữ liệu **biến đổi qua từng giai đoạn**, kèm số dòng thực tế. Đọc từ
trái sang phải theo 5 giai đoạn (stage): Source → Clean → Feature+Label → Train+Evaluate →
Serve+Store.

### Luồng dữ liệu
1. **Raw CSV (549084 dòng)** → làm sạch → **Clean CSV (548858 dòng)**. `clean_data` loại
   dòng thiếu, trùng, giá không hợp lệ (ví dụ high < low), volume âm.
2. **Quality CSVs:** thống kê theo mã, giữ **396 mã đủ điều kiện** (lọc ≥250 phiên giao dịch),
   loại 4 mã (CRV, TCX, VCK, VPX).
3. **Feature CSV (511191 dòng, 15 tín hiệu):** tạo 15 đặc trưng kỹ thuật.
4. **ML Dataset (509211 dòng):** gắn thêm cột nhãn `target` (UP/NOT_UP).
5. **Time Split:** chia theo `label_end_date` so với ngày `2025-12-31` → TRAIN 466973 dòng,
   TEST 42238 dòng. Không xáo trộn (no shuffle).
6. **Tune Models:** huấn luyện 3 mô hình + 1 dummy trên TRAIN với CV.
7. **Final Model (Random Forest, F1_UP 0.4816):** mô hình thắng, lưu ra file; sinh
   **Reports** (chỉ số + biểu đồ). Cả hai đồng bộ vào **SQLite**.
8. **Serve:** Flask/CLI dùng `final_model.pkl` + lịch sử mã sạch để dự đoán.

### Thuật ngữ quan trọng (thẻ chú thích bên sơ đồ)
- **Kiểm soát rò rỉ (leakage control):** `future_close_5d`, `future_return_5d` chỉ dùng để
  tạo nhãn, **không bao giờ** vào danh sách đặc trưng — tránh mô hình "nhìn trộm" tương lai.
- **Class imbalance (mất cân bằng lớp):** lớp UP ít hơn NOT_UP. Xử lý khác nhau từng mô hình:
  LR `class_weight=balanced`, RF `balanced_subsample`, GB truyền `sample_weight=balanced`
  vào `fit()` (vì GB không có tham số class_weight).
- **Decision threshold (ngưỡng quyết định):** tối ưu điểm F1 trên mỗi TRAIN fold rồi áp lên
  VAL fold (trung thực, không rò rỉ), sau đó lưu vào artifact. Dự đoán = `P(UP) ≥ threshold`
  chứ không dùng mặc định 0.5.
- **Quy tắc chọn mô hình:** `select_final_model` bỏ dummy, xếp hạng theo f1_up, rồi recall_up,
  rồi độ đơn giản (LR < RF < GB).

---

## 3. Sơ đồ Quy trình xử lý — `stock-workflow.html`

**Mục đích:** thể hiện **thứ tự các bước** và **các cửa canh gác (guard)** theo 5 làn
(lane, tức nhóm trách nhiệm): Tuning Lab → Fetch+Prepare → Train+Evaluate+Select →
Artifacts+Database → Serving.

### Luồng quy trình
- **Làn Tuning Lab (thủ công, trước khi chạy chính thức):** `Set Params` (3 mô hình) →
  `CV on TRAIN` (kiểm định chéo trên TRAIN) → `Lock Config` (chốt cấu hình, gắn với
  fingerprint dataset). Đây là điều kiện để pipeline chính thức được phép chạy.
- **Làn Prepare:** `Fetch` (vnstock) → `Check Raw` (kiểm cột bắt buộc) → `Clean` (396 mã)
  → `Feature+Label` (15 tín hiệu).
- **Làn Train:** `Time Split` → `Re-CV+Fit` (chạy lại CV rồi huấn luyện, có cân bằng lớp)
  → `Test Metrics` (đo trên 42238 dòng TEST) → `Select RF` (chọn mô hình cuối).
- **Làn Artifacts:** ghi `CSVs`, `Model`, `Reports`, rồi đồng bộ vào `SQLite`.
- **Làn Serving:** `User` → `Web/CLI` → `Predict` (nạp mô hình RF, lấy dòng đặc trưng mới
  nhất) → `Result` (UP/NOT_UP), và ghi log vào SQLite.

### Thuật ngữ quan trọng
- **Fingerprint (dấu vân dữ liệu):** chuỗi băm (hash) đại diện phiên bản dataset. Cấu hình
  đã chốt phải khớp fingerprint hiện tại thì pipeline mới chạy — tránh dùng cấu hình cho
  bộ dữ liệu khác.
- **Guard (cửa canh gác):** 3 điều kiện chặn trước khi chạy pipeline chính thức: (1) đủ cấu
  hình 3 mô hình khớp fingerprint; (2) TEST chưa bị khóa (test lock — chống đánh giá lại
  cùng một tập TEST gây rò rỉ); (3) không có pipeline nào đang chạy (khóa `pipeline.lock`).
- **Dummy baseline:** mô hình ngây thơ (đoán theo lớp phổ biến nhất) chỉ để so sánh, không
  bao giờ được chọn làm mô hình cuối.

### Điểm cần nhấn khi trình bày
- Tuning Lab (thủ công) và Pipeline (tự động) là **hai giai đoạn nối tiếp**: phải chốt cấu
  hình thủ công trước, rồi pipeline mới chạy đồng loạt trên cấu hình đó.

---

## 4. Sơ đồ Trình tự tương tác — `stock-sequence.html`

**Mục đích:** phóng to đúng **một hành động: người dùng bấm dự đoán 1 mã**. Đọc từ trên
xuống theo trục thời gian; mỗi cột dọc là một thành phần, mũi tên ngang là một lời gọi.

### Trình tự
1. `User` gửi mã cổ phiếu → `Flask/CLI` gọi `predict_symbol()`.
2. `Prediction Service` đọc **Clean CSV** → nhận lịch sử của mã đó.
3. Gọi `build_features` → tạo đặc trưng, lấy **dòng mới nhất** (latest row).
4. Nạp **Model Store** (`final_model.pkl` + metadata) → nhận mô hình RF + siêu dữ liệu.
5. `predict_proba` → tính `P(UP)`, so với `decision_threshold` (0.4049 lấy từ metadata)
   → gán nhãn UP/NOT_UP.
6. Trả `result dict` về Flask → ghi log vào SQLite (best-effort) → render trang kết quả.

### Thuật ngữ quan trọng
- **Activation bar (thanh kích hoạt):** vạch dọc dày trên mỗi cột, cho thấy thành phần đó
  đang "bận xử lý" trong khoảng thời gian nào.
- **Return message (mũi tên trả về):** vẽ nhạt hơn mũi tên gọi đi, thể hiện kết quả trả lại.
- **predict_proba:** hàm scikit-learn trả về xác suất từng lớp; ta lấy xác suất lớp `1` (UP).
- **best-effort logging:** ghi log DB được bọc trong try/except — nếu DB lỗi, lời gọi bị
  "nuốt" (bỏ qua) để không làm hỏng việc trả kết quả cho người dùng.

### Điểm cần nhấn khi trình bày
- Dự đoán **tính lại đặc trưng từ lịch sử sạch**, không đọc cột nhãn — nhất quán với lúc
  huấn luyện.
- Thứ tự lớp (`classes_`) và ngưỡng đến từ **metadata đã lưu**, đảm bảo suy luận khớp
  quá trình huấn luyện.

---

## 5. Sơ đồ Vòng đời trạng thái — `stock-lifecycle.html`

**Mục đích:** thể hiện mô hình đi qua các **trạng thái (state)** nào, từ dữ liệu thô đến khi
phục vụ, kèm các nhánh canh gác và kết cục. Bố cục 3 dải (band): pha chính (trên), canh
gác + tác dụng phụ (giữa), kết cục cuối (dưới).

### Vòng đời
- **Dải pha chính (01→05):** `Raw Ready` → `Clean+Feature` → `Train/Tune` → `Evaluate` →
  `Serve`. Đây là "đường ray" chính chạy ngang.
- **Dải giữa (canh gác + phụ):** `Config Guard` (đủ 3 mô hình + khớp fingerprint),
  `Split Guard` (kiểm rò rỉ theo `label_end_date`), `DB Sync` (đồng bộ SQLite).
- **Dải kết cục (dưới):** `Pipeline Blocked` (bị chặn do thiếu cấu hình / test bị khóa),
  `Prediction Log` (đã lưu lịch sử dự đoán), `Model Ready` (Random Forest, F1 0.4816).

### Thuật ngữ quan trọng
- **State machine (máy trạng thái):** cách mô tả hệ thống bằng tập trạng thái và các chuyển
  tiếp giữa chúng.
- **Terminal state (trạng thái cuối):** trạng thái kết thúc, không quay lại pha đang hoạt
  động (ở đây: Blocked / Log / Ready).
- **Quality gate (cổng chất lượng):** `Evaluate` là điểm quyết định — chọn mô hình theo
  `f1_up → recall_up → simplicity`.
- **CV TimeSeriesSplit n=5, gap=5:** kiểm định chéo theo thời gian, 5 lần chia, chừa
  khoảng trống 5 phiên giữa TRAIN và VAL để khớp horizon dự đoán 5 phiên (chống rò rỉ).

---

## Bảng thuật ngữ tổng hợp (glossary)

| Thuật ngữ | Giải thích ngắn |
|-----------|-----------------|
| OHLCV | Open/High/Low/Close/Volume — giá mở, cao, thấp, đóng và khối lượng của một phiên. |
| Feature (đặc trưng) | Biến đầu vào cho mô hình, tính từ giá/khối lượng (SMA, RSI, volatility...). |
| Label / target (nhãn) | Kết quả cần dự đoán: UP nếu giá sau 5 phiên tăng >1%, ngược lại NOT_UP. |
| Horizon (tầm dự đoán) | Số phiên nhìn về tương lai để gán nhãn — ở đây là 5 phiên. |
| SMA5/20/50 | Simple Moving Average — trung bình giá đóng cửa 5/20/50 phiên. |
| RSI14 | Relative Strength Index 14 phiên — chỉ báo động lượng, thang 0–100. |
| Volatility | Độ dao động — độ lệch chuẩn của lợi suất ngày trong cửa sổ 5/20 phiên. |
| Train / Test split | Chia dữ liệu: TRAIN để huấn luyện, TEST để đánh giá cuối (chỉ đo 1 lần). |
| Cross-validation (CV) | Kiểm định chéo — chia TRAIN nhiều lần để ước lượng hiệu năng ổn định. |
| TimeSeriesSplit | CV cho dữ liệu thời gian: VAL luôn nằm sau TRAIN theo thời gian. |
| gap | Khoảng trống (số phiên) chừa giữa TRAIN và VAL trong CV để chống rò rỉ. |
| Class imbalance | Mất cân bằng lớp — số mẫu UP ít hơn NOT_UP. |
| class_weight / sample_weight | Cách tăng "trọng số" cho lớp thiểu số khi huấn luyện. |
| Decision threshold | Ngưỡng xác suất để quyết UP (P(UP) ≥ ngưỡng), tinh chỉnh thay cho 0.5. |
| F1_UP | Điểm F1 cho lớp UP — trung bình điều hòa của precision và recall. |
| Precision / Recall | Độ chính xác dự đoán UP / tỉ lệ bắt đúng các trường hợp UP thật. |
| Leakage (rò rỉ) | Vô tình cho mô hình thấy thông tin tương lai → chỉ số ảo cao. |
| Fingerprint | Chuỗi băm đại diện phiên bản dataset, dùng để khớp cấu hình. |
| Lock (khóa) | Cơ chế chặn: pipeline lock (chống chạy trùng), test lock (chống rò rỉ TEST). |
| Artifact | Sản phẩm lưu ra file: mô hình `.pkl`, metadata `.json`, báo cáo CSV. |
| Dummy classifier | Mô hình cơ sở đoán theo lớp phổ biến nhất, chỉ để so sánh. |

---

## Cách mở & xuất sơ đồ

Mở trực tiếp file `.html` bằng trình duyệt. Trong sơ đồ có sẵn:
- Nút đổi nền **sáng/tối** (góc trên).
- Menu **xuất ảnh**: sao chép PNG, tải PNG/JPEG/WebP (tối đa 4× độ phân giải), hoặc tải
  SVG hai chế độ nền (hợp để chèn vào báo cáo/README).

Muốn chỉnh nội dung: sửa file `.json` tương ứng rồi dựng lại bằng renderer của skill archify:
```
node renderers/<loại>/render-<loại>.mjs <file>.<loại>.json <file>.html
```
với `<loại>` ∈ architecture / dataflow / workflow / sequence / lifecycle.
