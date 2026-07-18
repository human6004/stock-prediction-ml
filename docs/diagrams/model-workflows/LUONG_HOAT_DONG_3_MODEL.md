# Luồng hoạt động của 3 model — giải thích cho người mới

File này giải thích **cách 3 model chạy từ đầu đến lúc ra kết quả và lưu vào lịch sử**, đúng
với 3 sơ đồ trong thư mục này:

- `logistic-regression.html`
- `random-forest.html`
- `gradient-boosting.html`

Đọc file này trước, rồi mở 3 sơ đồ HTML để nhìn cho dễ hình dung.

> **Đọc 1 câu để nắm ý chính:** cả 3 model đi **chung một con đường** — chuẩn bị dữ liệu
> giống hệt nhau, chỉ khác ở chỗ *đặt tham số* và *thuật toán học*. Cuối con đường, mỗi
> model cho ra **một điểm số F1_UP** và điểm đó được **ghi vào một file lịch sử** để so sánh.

---

## 0. Vài từ cần hiểu trước (giải thích ngắn gọn)

Đọc lướt qua bảng này, gặp lại từ nào ở dưới thì quay lên tra:

| Từ | Hiểu nôm na là gì |
|---|---|
| **Model (mô hình)** | Một "cỗ máy đoán". Cho nó xem dữ liệu quá khứ, nó học ra quy luật rồi đoán tương lai. |
| **Feature (đặc trưng)** | Các con số mô tả cổ phiếu (giá tăng mấy %, biến động ra sao...). Đây là "đầu vào" model nhìn để đoán. |
| **Label / Target (nhãn)** | "Đáp án đúng" trong quá khứ: sau 5 phiên mã đó có tăng hơn 1% không (UP / NOT_UP). |
| **TRAIN** | Phần dữ liệu cũ, dùng để model **học**. |
| **TEST** | Phần dữ liệu mới hơn, dùng để **chấm điểm** model (ở sơ đồ này chưa dùng tới — xem mục 6). |
| **Tham số (hyperparameter)** | Các "nút vặn" của model mà người làm chỉnh tay trước khi train, ví dụ số cây, độ sâu cây. |
| **CV (Cross Validation)** | Cách chấm điểm model ngay trên TRAIN bằng cách chia nhỏ ra thử nhiều lần cho chắc ăn. |
| **F1_UP** | Điểm số đo model bắt lớp UP tốt tới đâu. Càng cao càng tốt (0 đến 1). |
| **Mất cân bằng lớp** | Trong dữ liệu, số ca UP ít hơn NOT_UP nhiều. Không xử lý thì model lười, đoán toàn NOT_UP. |

---

## 1. Bức tranh tổng thể

Cả 3 model chạy qua **2 chặng**:

```text
CHẶNG 1 — CHUẨN BỊ DỮ LIỆU  (giống hệt nhau ở cả 3 model)
   Fetch → Clean → Feature+Label → Time Split
                                        │
                                        ▼
CHẶNG 2 — HỌC & CHẤM ĐIỂM  (mỗi model làm KHÁC nhau)
   Set Params → Fit → Tune Threshold → Log CV F1_UP → tuning_history.csv
                                                          ▲
                                                   DỪNG Ở ĐÂY
```

Điểm mấu chốt cần nhớ:

- **Chặng 1 chỉ chạy 1 lần, dùng chung cho cả 3 model.** Dữ liệu ra giống nhau, không phụ
  thuộc bạn chọn model nào. (Trong sơ đồ, lane này ghi rõ *"shared — identical for all 3 models"*.)
- **Chặng 2 là chỗ 3 model khác nhau.** Khác ở 2 bước: `Set Params` và `Fit`.

---

## 2. Chặng 1 — Chuẩn bị dữ liệu (dùng chung)

Mục tiêu chặng này: biến dữ liệu giá thô, lộn xộn thành một **bảng số sạch, có sẵn đáp án**,
chia làm 2 phần TRAIN/TEST.

### Bước 1 — Fetch (kéo dữ liệu về)
Lấy dữ liệu giá cổ phiếu từ nguồn `vnstock` (một thư viện lấy dữ liệu chứng khoán Việt Nam).
Dữ liệu gồm giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng của từng mã theo từng ngày.

### Bước 2 — Clean (làm sạch)
Dữ liệu thô hay có lỗi: dòng trùng, giá âm, ngày cao nhất lại thấp hơn ngày thấp nhất (vô lý)...
Bước này loại các dòng lỗi đó. Nó cũng **loại các mã quá ít dữ liệu** (dưới 250 phiên giao dịch)
vì không đủ để model học. Sau bước này còn **396 mã** đạt chuẩn.

> *Phiên giao dịch = một ngày thị trường mở cửa. 250 phiên ≈ khoảng 1 năm giao dịch.*

### Bước 3 — Feature + Label (tạo đặc trưng và đáp án)
Đây là bước "dịch" giá cổ phiếu sang ngôn ngữ model hiểu được.

- **Feature (20 đặc trưng):** tính ra các con số như "giá 5 phiên qua tăng mấy %", "độ biến động
  20 phiên", "giá đang cách đỉnh gần nhất bao xa"... Đây là những gì model **nhìn vào để đoán**.
- **Label (đáp án):** với mỗi ngày, nhìn tới **5 phiên sau** xem giá có tăng hơn 1% không. Nếu có
  → gán `UP`, không → gán `NOT_UP`. Đây là "đáp án đúng" để model học.

> Vì sao gọi là "20 features"? Đây là phiên bản dữ liệu mới nhất, có 20 đặc trưng (bản cũ chỉ 15).

### Bước 4 — Time Split (chia theo thời gian)
Chia bảng dữ liệu làm 2:

- **TRAIN** (dữ liệu cũ hơn, mốc `<= 2025-06-30`): **417,806 dòng** — dùng để model học.
- **TEST** (dữ liệu mới hơn): để dành chấm điểm sau này.

> **Vì sao chia theo thời gian mà không chia ngẫu nhiên?**
> Vì đây là dữ liệu chứng khoán — quá khứ phải đi trước tương lai. Nếu trộn ngẫu nhiên, model có
> thể vô tình "thấy trước tương lai" trong lúc học (gọi là *data leakage* — rò rỉ dữ liệu), làm
> điểm số đẹp giả tạo. Chia theo thời gian tránh được lỗi này.

Hết chặng 1, ta có **tập TRAIN** để đưa sang chặng 2.

---

## 3. Chặng 2 — Học và chấm điểm (mỗi model khác nhau)

Chặng này chỉ chạy trên **TRAIN**. Nó là phần "Tuning Lab" — nơi thử tham số cho từng model.

### Bước 5 — Set Params (đặt tham số) ← **CHỖ 3 MODEL KHÁC NHAU (1)**
Trước khi train, ta vặn các "nút" của model. Mỗi loại model có bộ nút riêng:

| Model | Các nút chính (tham số đã chốt) |
|---|---|
| **Logistic Regression** | `C = 6.59e-05` (mức "ghìm" model cho đơn giản, số càng nhỏ ghìm càng mạnh), `solver = liblinear` |
| **Random Forest** | `n_estimators = 130` (số cây), `max_depth = 4` (độ sâu mỗi cây), `min_samples_leaf = 25`, `max_features = 0.35` |
| **Gradient Boosting** | `n_estimators = 120` (số vòng học), `learning_rate = 0.05` (học nhanh/chậm), `max_depth = 2`, `subsample = 1.0` |

Không cần thuộc các con số này. Chỉ cần hiểu: **mỗi model có kiểu điều chỉnh riêng, nên bước này
ở 3 sơ đồ ghi khác nhau.**

### Bước 6 — Fit (huấn luyện) ← **CHỖ 3 MODEL KHÁC NHAU (2)**
"Fit" nghĩa là **cho model học** trên TRAIN. Đây là lúc thuật toán thật sự khác nhau:

| Model | Cách học (rất ngắn gọn) | Cách xử lý mất cân bằng lớp |
|---|---|---|
| **Logistic Regression** | Model tuyến tính đơn giản — vẽ một "đường ranh giới" chia UP / NOT_UP. | `class_weight = "balanced"` (tăng trọng số cho lớp UP ít ỏi) |
| **Random Forest** | Trồng **nhiều cây quyết định**, mỗi cây bỏ một phiếu, lấy đa số. | `class_weight = "balanced_subsample"` |
| **Gradient Boosting** | Trồng cây **nối tiếp nhau**, cây sau chuyên sửa lỗi của cây trước. | Không có nút `class_weight` sẵn, nên phải **truyền tay** `sample_weight` vào lúc học |

> **Vì sao phải "xử lý mất cân bằng lớp"?**
> Trong TRAIN, số ca UP ít hơn NOT_UP khá nhiều (UP chỉ ~39%). Nếu để mặc kệ, model sẽ "lười":
> cứ đoán NOT_UP là đã đúng phần lớn thời gian → không bao giờ bắt được UP. Việc tăng trọng số
> lớp UP buộc model phải quan tâm tới nó. Điểm thú vị: **cả 3 model làm việc này theo 3 cách khác
> nhau**, vì thư viện scikit-learn cung cấp API khác nhau cho từng loại.

### Bước 7 — Tune Threshold (chỉnh ngưỡng quyết định)
Sau khi học xong, model không nói thẳng "UP" hay "NOT_UP" mà đưa ra một **xác suất** (ví dụ
"70% khả năng UP"). Ta cần một **ngưỡng** để chốt: xác suất bao nhiêu thì gọi là UP?

- Mặc định thường lấy 0.5 (50%). Nhưng vì lớp UP hiếm, lấy 0.5 sẽ **bỏ sót nhiều UP**.
- Nên hệ thống tự dò **ngưỡng nào cho F1_UP cao nhất** và dùng ngưỡng đó.

> **Điểm quan trọng để không bị "ăn gian" điểm số:** việc dò ngưỡng chỉ làm **trên phần dữ liệu
> model đã học (train-part của mỗi lần chia)**, rồi mới đem chấm trên phần để riêng ra
> (val-part). Nhờ vậy model không "nhìn trộm đáp án" khi tự chấm — điểm số ra mới trung thực.
> Trong sơ đồ, bước này gắn nhãn *"no leak"* (không rò rỉ).

### Bước 8 — Log CV F1_UP (ghi điểm vào lịch sử) ← **ĐIỂM DỪNG**
Đây là đích đến của 3 sơ đồ.

- Bước 6–7 không chỉ chạy 1 lần, mà chạy **5 lần** trên 5 cách chia TRAIN khác nhau (đây chính là
  **Cross Validation** — chia nhỏ TRAIN ra thử nhiều lần cho chắc). Trong sơ đồ ghi *"x5 folds"*.
- Lấy **trung bình 5 lần** đó ra được **một con số F1_UP**:

  | Model | CV F1_UP |
  |---|---|
  | Logistic Regression | **0.5490** |
  | Random Forest | **0.5488** |
  | Gradient Boosting | **0.5453** |

- Con số này được **ghi thêm một dòng vào file `experiments/tuning_history.csv`** — cuốn "nhật ký"
  lưu lại mọi lần thử tham số. **Đến đây là dừng.**

> **CV (Cross Validation) là gì, nói lại cho rõ:** thay vì học một lần rồi tự chấm (dễ ảo tưởng),
> ta chia TRAIN thành 5 phần theo thời gian, mỗi lần học trên vài phần và chấm trên phần còn lại,
> làm 5 lượt rồi lấy trung bình. Cách này cho điểm số **đáng tin hơn**, không may rủi.

---

## 4. Nhìn lại: 3 model giống và khác chỗ nào

```text
        LOGISTIC REGRESSION        RANDOM FOREST           GRADIENT BOOSTING
        ───────────────────        ─────────────           ─────────────────
CHẶNG 1   Fetch→Clean→Feature→Split   (GIỐNG HỆT — chạy chung 1 lần cho cả 3)
────────────────────────────────────────────────────────────────────────────
Set Params  C, solver               n_est, depth,          n_est, learning_rate,
                                     leaf, max_features     depth, subsample
Fit         đường tuyến tính         nhiều cây bỏ phiếu     cây nối tiếp sửa lỗi
cân bằng    class_weight=balanced    balanced_subsample     sample_weight (thủ công)
Threshold   dò ngưỡng F1_UP cao nhất (giống cơ chế ở cả 3)
CV F1_UP    0.5490                   0.5488                 0.5453
Dừng ở      ghi vào tuning_history.csv  (giống ở cả 3)
```

Tóm lại: **chung con đường, khác cái xe.** Đường (dữ liệu) như nhau; cái xe (thuật toán + tham số)
mỗi model một kiểu, nên về đích với điểm số hơi khác nhau.

---

## 5. Vì sao 3 điểm số lại sát nhau đến vậy?

3 con số CV F1_UP (0.5490 / 0.5488 / 0.5453) chênh nhau **rất ít**. Điều này bình thường và trung
thực: dự báo cổ phiếu **chỉ bằng dữ liệu giá** là bài toán khó, có một "trần" khó vượt. Việc 3
model khác nhau vẫn ra điểm gần nhau cho thấy **dữ liệu và cách làm quyết định nhiều hơn là chọn
model nào**. Đây là điểm nên nói thẳng khi trình bày, thay vì thổi phồng một model là "vượt trội".

---

## 6. Một điểm dễ nói nhầm khi bảo vệ (đọc kỹ)

Con số F1_UP ở cuối 3 sơ đồ này là **CV F1_UP đo trên TRAIN** — tức điểm "tự chấm trong lúc học".

Nó **KHÔNG phải** F1 đo trên tập TEST. Điểm trên TEST (điểm "thi thật") chỉ sinh ra ở **pipeline
chính thức**, là phần đi tiếp **sau** điểm dừng của các sơ đồ này (train model cuối → chấm trên
TEST → so sánh → chọn model → đem đi dự đoán).

Vậy nên khi trình bày 3 sơ đồ này, hãy nói:

> "Đây là luồng **thử tham số** cho từng model, kết thúc ở điểm CV F1_UP được ghi vào lịch sử.
> Việc chấm điểm trên TEST và chọn ra model cuối cùng nằm ở bước sau, không thuộc phạm vi 3 sơ đồ này."

Nói vậy vừa đúng, vừa tránh bị hỏi vặn.

---

## 7. Cách mở và chỉnh sơ đồ

- **Xem:** mở thẳng file `.html` bằng trình duyệt. Có nút đổi nền sáng/tối và menu xuất ảnh PNG/SVG
  (tiện đưa vào báo cáo).
- **Chỉnh nội dung:** sửa file `.json` cùng tên rồi dựng lại bằng renderer của skill *archify*:

  ```powershell
  node <archify>/renderers/workflow/render-workflow.mjs <file>.workflow.json <file>.html
  ```

- Số liệu trong sơ đồ lấy từ dataset phiên bản `f6cb3ac8f820` (20 features, mốc chia `2025-06-30`).
  Nếu cập nhật thêm dữ liệu mới thì các con số này sẽ đổi, nhớ chỉnh lại cho khớp.
