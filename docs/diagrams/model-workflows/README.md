# Lưu đồ tuning riêng cho từng model

Thư mục này chứa **3 lưu đồ riêng biệt**, mỗi model một file, mô tả luồng huấn luyện
của model đó **từ đầu đến lúc ra được điểm F1 và ghi vào lịch sử tuning** — KHÔNG đi tiếp
tới bước chọn final model hay phục vụ dự đoán.

| File | Model | Điểm CV F1_UP (ghi vào lịch sử) |
|------|-------|---:|
| `logistic-regression.html` | Logistic Regression | 0.5490 |
| `random-forest.html` | Random Forest | 0.5488 |
| `gradient-boosting.html` | Gradient Boosting | 0.5453 |

File `.json` cùng tên là dữ liệu nguồn để dựng lại sơ đồ bằng renderer archify:
```
node renderers/workflow/render-workflow.mjs <file>.workflow.json <file>.html
```

Tất cả số liệu lấy từ `experiments/tuning_history.csv` trên dataset hiện tại
(fingerprint `f6cb3ac8f820`, 20 feature, split date `2025-06-30`).

---

## Vì sao mỗi lưu đồ dừng ở "ghi vào lịch sử"?

Ý tưởng của bộ sơ đồ này là biểu diễn **vòng đời huấn luyện của MỘT model**, kết thúc
ngay khi model chạy Cross Validation xong và cho ra điểm F1. Vì vậy mỗi sơ đồ dừng tại
bước `Log CV F1_UP` (ghi một dòng vào `experiments/tuning_history.csv`).

**Điểm quan trọng cần hiểu đúng:** con số F1 tại điểm dừng là **CV F1_UP đo trên TRAIN**
(trung bình 5 fold), KHÔNG phải F1 trên TEST. F1 trên TEST + việc so sánh 3 model + chọn
`final_model.pkl` thuộc về pipeline chính thức — nằm **ngoài** phạm vi 3 sơ đồ này (xem
`docs/diagrams/v3/` cho luồng pipeline đầy đủ).

---

## Cấu trúc chung: 2 lane

Mỗi sơ đồ có 2 lane, đọc từ trên xuống:

1. **Prepare Data (dùng chung — giống hệt ở cả 3 model):**
   `Fetch` → `Clean` (396 mã đạt chuẩn, đã check cột OHLCV) → `Feature+Label` (20 feature
   + nhãn UP/NOT_UP) → `Time Split` (chia TRAIN/TEST theo `label_end_date`, mốc
   `2025-06-30`, TRAIN 417,806 dòng).

2. **Tuning Lab (riêng từng model):**
   `Set Params` → `Fit` (huấn luyện, cách xử lý mất cân bằng lớp riêng) → `Tune Threshold`
   (tối ưu ngưỡng F1_UP trên mỗi TRAIN fold, không rò rỉ) → `Log CV F1_UP` (ghi vào
   `tuning_history.csv`).

> Lane Prepare Data được vẽ **giống nhau** ở cả 3 file và gắn nhãn "shared - identical for
> all 3 models" ngay trên tiêu đề lane, để nhấn rằng **3 model dùng chung một dataset**,
> không model nào có dữ liệu riêng.

---

## 3 model khác nhau ở đâu?

Chỉ khác nhau ở **2 node** trong lane Tuning Lab (được đánh dấu tag `model-specific`), cộng
với card "What is different vs ..." ở chân mỗi sơ đồ:

| | Set Params | Fit (thuật toán + cân bằng lớp) | CV F1_UP |
|---|---|---|---:|
| **Logistic Regression** | `C=6.59e-05`, `solver=liblinear` | `StandardScaler → LogisticRegression`, `class_weight=balanced` | 0.5490 |
| **Random Forest** | `n_est=130, depth=4, leaf=25, feat=0.35` | `RandomForestClassifier`, `class_weight=balanced_subsample` | 0.5488 |
| **Gradient Boosting** | `n_est=120, lr=0.05, depth=2, subsample=1.0` | `GradientBoostingClassifier`, `sample_weight=balanced` (vì không có `class_weight`) | 0.5453 |

Ba điểm đọc nhanh từ bảng:

- **Thuật toán khác họ:** LR là model tuyến tính; RF là rừng cây bagging; GB là cây boosting
  tuần tự.
- **Cách cân bằng lớp khác nhau vì API scikit-learn khác nhau:** LR và RF có sẵn tham số
  `class_weight`; GB không có nên phải truyền `sample_weight` vào `fit()`.
- **Điểm CV rất sát nhau (0.545–0.549):** chênh lệch giữa 3 model nhỏ, phản ánh trần dự báo
  của dữ liệu OHLCV chứ không phải một model vượt trội hẳn.

---

## Node `Tune Threshold` (có ở cả 3 sơ đồ)

Nhãn UP/NOT_UP không dùng ngưỡng mặc định 0.5. Vì lớp UP là thiểu số (~39% TRAIN), ngưỡng
0.5 báo UP quá ít. Trong mỗi fold CV, hệ thống tối ưu ngưỡng cho F1_UP cao nhất **trên
TRAIN fold rồi áp lên VAL fold** — chọn ngưỡng không nhìn vào VAL nên không rò rỉ (tag
"no leak"). Điểm CV F1_UP ghi vào lịch sử đã phản ánh đúng ngưỡng này.

---

## Cách mở & xuất sơ đồ

Mở trực tiếp file `.html` bằng trình duyệt. Mỗi sơ đồ có:
- Nút đổi nền **sáng/tối** (góc trên, lưu vào `localStorage`).
- Menu **xuất ảnh**: sao chép/tải PNG (tối đa 4× độ phân giải), JPEG, WebP, hoặc SVG hai
  chế độ nền — hợp để đưa vào báo cáo/README.

Muốn chỉnh nội dung: sửa file `.json` tương ứng rồi dựng lại bằng renderer workflow của
skill archify (lệnh ở đầu file này).
