# Giải thích 3 sơ đồ workflow theo từng mô hình

Thư mục này chứa 3 sơ đồ `workflow` riêng biệt cho 3 mô hình có thể tinh chỉnh trong
project: **Logistic Regression**, **Random Forest**, **Gradient Boosting**. Mỗi mô hình
đi qua đúng một quy trình chung (Tuning Lab → chuẩn bị dữ liệu → huấn luyện/đánh giá →
so sánh), khác nhau ở tham số, cách xử lý mất cân bằng lớp, và kết quả cuối.

| File | Mô hình | Kết quả |
|------|---------|---------|
| `logistic-regression.html` | Logistic Regression | Không được chọn — F1_UP TEST 0.4701, rank 3/3 |
| `random-forest.html` | Random Forest | **Được chọn làm final_model** — F1_UP TEST 0.4816, rank 1/3 |
| `gradient-boosting.html` | Gradient Boosting | Không được chọn — F1_UP TEST 0.4795, rank 2/3 |

File `.json` cùng tên là dữ liệu nguồn để dựng lại sơ đồ bằng renderer của skill archify:
```
node renderers/workflow/render-workflow.mjs <file>.workflow.json <file>.html
```

---

## Cấu trúc chung của cả 3 sơ đồ

Mỗi sơ đồ có 4 làn (lane), đọc từ trên xuống, các bước trong 1 làn đọc từ trái sang phải:

1. **Tuning Lab** — `Set Params` (đặt siêu tham số) → `CV on TRAIN` (kiểm định chéo trên
   TRAIN, ra điểm F1_UP trung bình theo fold) → `Lock Config` (chốt cấu hình, gắn với
   fingerprint dataset để tránh dùng nhầm cấu hình cho dataset khác).
2. **Fetch + Prepare Data** — `Fetch` (kéo dữ liệu vnstock) → `Check Raw` (kiểm cột bắt
   buộc OHLCV) → `Clean` (loại dòng lỗi, còn 396 mã đạt chuẩn) → `Feature+Label` (tính 15
   đặc trưng kỹ thuật + gán nhãn UP/NOT_UP).
3. **Fit + Evaluate** — `Time Split` (chia TRAIN/TEST theo `label_end_date`, không xáo
   trộn) → bước huấn luyện riêng của từng mô hình (tên và tag khác nhau tuỳ mô hình) →
   `Tune Threshold` (tinh chỉnh ngưỡng quyết định, chi tiết bên dưới) → `Test Metrics`
   (đo precision/recall/F1 trên 42238 dòng TEST, chỉ đo một lần).
4. **Compare + Outcome** — `Compare` (xếp hạng cả 3 mô hình theo tiêu chí
   `f1_up → recall_up → simplicity`) → đối chiếu `vs Dummy` (mốc baseline phải vượt qua)
   → kết quả `Selected`/`Not Selected` → `Save Artifact` (file `.pkl` lưu lại; mô hình
   thắng được nhân bản thành `final_model.pkl`).

Riêng **Random Forest** (mô hình được chọn) có thêm **lane thứ 5 — Serving**: `User` →
`Web/CLI` → `Predict` → `Result` (xem mục 2). Hai mô hình còn lại kết thúc ở
`Save Artifact` với nhãn cạnh "kept, not served" — được giữ làm tham chiếu nhưng không
dùng để dự đoán.

### Hai node kỹ thuật quan trọng (có ở cả 3 sơ đồ)

- **`Tune Threshold` (node hồng, lane Fit+Evaluate):** Nhãn UP/NOT_UP không dùng ngưỡng
  mặc định 0.5. Vì lớp UP là thiểu số (~39% TRAIN), ngưỡng 0.5 sẽ báo UP quá ít. Hệ thống
  tối ưu ngưỡng cho F1_UP cao nhất **trên mỗi TRAIN fold rồi áp lên VAL fold** (trung
  thực, không rò rỉ — tag "per-fold, no leak"). Ngưỡng cuối lưu vào artifact khác nhau
  từng mô hình: LR **0.4722**, RF **0.4049**, GB **0.4009**.
- **`vs Dummy` (node xám, lane Compare+Outcome):** `Dummy Classifier` luôn đoán lớp phổ
  biến nhất (NOT_UP) → accuracy 0.69 nhưng F1_UP = 0. Nó là mốc "sàn": mọi mô hình thật
  phải vượt qua mới có ý nghĩa. Dummy được huấn luyện trong pipeline nhưng bị loại khỏi
  quá trình chọn (`select_final_model` bỏ `model_id == 1`).

Mũi tên hồng (`security` variant) đánh dấu các bước canh gác/khoá cấu hình/ngưỡng; mũi
tên đậm (`emphasis`) là luồng chính; mũi tên đứt (`dashed`) là nhánh phụ (đối chiếu
dummy, hoặc mô hình bị loại vẫn giữ artifact làm tham chiếu, không xoá).

---

## 1. Logistic Regression — `logistic-regression.html`

**Bước huấn luyện riêng:** `Scale + Fit` — pipeline `StandardScaler` rồi
`LogisticRegression(class_weight="balanced")`.

- Tham số đã chốt: `C = 2.11e-06` (rất nhỏ → regularization L2 rất mạnh, mô hình bị kéo
  về gần tuyến tính đơn giản), `solver = liblinear`.
- CV F1_UP trên TRAIN: **0.5419**. Test F1_UP: **0.4701** (giảm ~0.07 do lệch phân phối
  UP giữa TRAIN 38.7% và TEST 30.7%).
- Điểm mạnh: recall_up cao nhất trong 3 mô hình (**0.958**) — bắt được gần hết các
  trường hợp UP thật.
- Điểm yếu: precision_up thấp nhất (**0.311**) — báo UP sai rất nhiều, vì `C` quá nhỏ
  khiến mô hình gần như đoán thiên về lớp UP.
- Kết quả: xếp hạng 3/3 theo tiêu chí chọn mô hình (F1_UP → recall_up → độ đơn giản),
  không được chọn. Vẫn là mô hình đơn giản nhất (simplicity rank 1/3), giữ vai trò
  baseline tuyến tính để so sánh.

## 2. Random Forest — `random-forest.html`

**Bước huấn luyện riêng:** `Fit Forest` — `RandomForestClassifier(class_weight=
"balanced_subsample")`.

- Tham số đã chốt: `n_estimators=80`, `max_depth=8`, `min_samples_leaf=100`,
  `max_features=0.4`. `min_samples_leaf=100` bắt buộc mỗi lá phải có ≥100 mẫu — cơ chế
  chính giúp mô hình không học vẹt (overfit) trên ~467 nghìn dòng TRAIN.
- CV F1_UP trên TRAIN: **0.5427**. Test F1_UP: **0.4816** — cao nhất trong 3 mô hình và
  có khoảng cách CV→TEST nhỏ nhất (ổn định nhất khi chuyển từ CV sang dữ liệu thật).
- Cân bằng tốt giữa precision_up (0.328) và recall_up (0.905) — không nghiêng cực đoan
  như Logistic Regression.
- Decision threshold đã tối ưu: **0.4049** (áp dụng cả lúc đánh giá và lúc phục vụ dự
  đoán thật).
- Kết quả: xếp hạng 1/3, **được chọn làm mô hình cuối cùng**. Artifact được nhân bản
  thành `models/final_model.pkl` và là mô hình duy nhất được `prediction_service.py`
  nạp để dự đoán mã cổ phiếu.

**Lane Serving (chỉ Random Forest có):** vì chỉ mô hình thắng mới thành `final_model.pkl`,
sơ đồ RF có thêm dải thứ 5 mô tả lúc phục vụ thật:
`User` (nhập mã) → `Web/CLI` (route Flask `/predict`) → `Predict` → `Result` (UP/NOT_UP).
Bước `Predict` nạp `final_model.pkl`, tính lại dòng đặc trưng mới nhất của mã đó (không
đọc cột nhãn), rồi áp đúng ngưỡng **0.4049** đã lưu: `P(UP) ≥ 0.4049` thì gán UP. Đây là
lý do lane này chỉ xuất hiện ở RF — hai sơ đồ LR/GB dừng ở `Save Artifact` với nhãn
"kept, not served".

## 3. Gradient Boosting — `gradient-boosting.html`

**Bước huấn luyện riêng:** `Fit Boosting` — `GradientBoostingClassifier`. Mô hình này
**không có tham số `class_weight`** sẵn có (khác Logistic Regression và Random Forest),
nên phải xử lý mất cân bằng lớp bằng cách truyền `sample_weight` (tính theo kiểu
`"balanced"`) trực tiếp vào `fit()`.

- Tham số đã chốt: `n_estimators=40`, `learning_rate=0.2`, `max_depth=2`,
  `subsample=0.6`.
- CV F1_UP trên TRAIN: **0.5441** — cao nhất trong cả 3 mô hình ở giai đoạn CV. Nhưng
  Test F1_UP chỉ **0.4795** — khoảng cách CV→TEST lớn nhất trong 3 mô hình, đúng với đặc
  tính dễ overfit của boosting khi CV không phản ánh hết được sự dịch chuyển phân phối
  dữ liệu theo thời gian.
- Kết quả: xếp hạng 2/3, không được chọn dù CV cao nhất — minh chứng rõ cho việc điểm CV
  không phải lúc nào cũng dự đoán đúng thứ hạng trên TEST thật. Là mô hình phức tạp nhất
  (simplicity rank 3/3), giữ vai trò tham chiếu.

---

## Bài học rút ra khi so sánh 3 sơ đồ

- **CV không phải TEST:** Gradient Boosting có CV F1_UP cao nhất (0.5441) nhưng thua
  Random Forest trên TEST (0.4795 vs 0.4816) — thứ hạng đảo ngược giữa 2 giai đoạn.
- **Regularization cực mạnh không có nghĩa là tốt:** `C` cực nhỏ của Logistic Regression
  đẩy recall lên rất cao nhưng phá hỏng precision, kéo F1_UP xuống thấp nhất.
- **Cả 3 mô hình xử lý mất cân bằng lớp khác nhau** vì API scikit-learn khác nhau:
  `class_weight="balanced"` (LR), `class_weight="balanced_subsample"` (RF),
  `sample_weight` thủ công trong `fit()` (GB, vì không có tham số `class_weight`).
- **Tiêu chí chọn mô hình cuối** (trong `services/model_evaluation.py`,
  `select_final_model`) luôn ưu tiên F1_UP trên TEST trước, sau đó mới đến recall_up,
  cuối cùng mới đến độ đơn giản — nên dù Logistic Regression đơn giản nhất và có
  recall cao nhất, nó vẫn không được chọn vì F1_UP thấp nhất.

## Cách mở & xuất sơ đồ

Mở trực tiếp file `.html` bằng trình duyệt. Mỗi sơ đồ có sẵn:
- Nút đổi nền **sáng/tối** (góc trên, lưu lựa chọn vào `localStorage`).
- Menu **xuất ảnh**: sao chép PNG, tải PNG/JPEG/WebP (tối đa 4× độ phân giải), hoặc tải
  SVG hai chế độ nền (tự đổi theo `prefers-color-scheme` của nơi chèn vào, hợp để đưa
  vào báo cáo/README).

Muốn chỉnh nội dung: sửa file `.json` tương ứng rồi dựng lại bằng renderer workflow của
skill archify (xem lệnh ở đầu file này).
