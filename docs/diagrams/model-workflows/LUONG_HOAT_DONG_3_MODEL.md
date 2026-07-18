# Luồng hoạt động của ba model

Tài liệu này giải thích bốn sơ đồ trong thư mục `model-workflows` cho người mới.
Ba sơ đồ đầu trả lời “mỗi model tuning tham số gì”; sơ đồ cuối trả lời “model nào
được chọn sau khi chấm TEST”.

## 1. Phần dữ liệu dùng chung

Cả ba model nhận đúng cùng một dữ liệu:

```text
Fetch -> Clean -> 20 Features + Label -> Time Split -> TRAIN
```

- `Fetch`: lấy OHLCV từ `vnstock/KBS`.
- `Clean`: giữ 396 mã đủ điều kiện và loại dòng OHLCV lỗi.
- `20 Features + Label`: tạo 20 đặc trưng và nhãn `UP/NOT_UP`; nhãn UP nghĩa là
  giá sau 5 phiên tăng trên 1%.
- `Time Split`: chia theo `label_end_date` tại `2025-06-30`, không shuffle.
- `TRAIN`: 417.806 dòng; TEST 95.022 dòng được giữ riêng cho bước cuối.

Phần này màu xám và có cùng node, vị trí, nội dung trong cả ba hình.

## 2. Hyperparameter nào thực sự được tuning?

Hyperparameter là nút điều chỉnh thay đổi giữa các lần thử. Mỗi model có một
khối lớn riêng:

| Model | Hyperparameter được thay đổi |
|---|---|
| Logistic Regression | `C`, `solver` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

Các giá trị sau không nằm trong khối hyperparameter:

- Logistic Regression: `StandardScaler`, `class_weight=balanced`,
  `max_iter=1000`, `random_state=42`.
- Random Forest: `class_weight=balanced_subsample`, `n_jobs=-1`,
  `random_state=42`.
- Gradient Boosting: `sample_weight=balanced`, `random_state=42`.
- Cả ba: `decision_threshold` là output của bước Tune Threshold.

Điểm này tránh nhầm xử lý dữ liệu/cân bằng lớp với tham số được thử.

## 3. CV trên TRAIN hoạt động ra sao?

Mỗi cấu hình đi qua `TimeSeriesSplit(n_splits=5, gap=5)` trên TRAIN:

1. Fit estimator mới trên train-fold.
2. Dùng prediction trên train-fold để tìm threshold tối đa F1_UP.
3. Áp threshold đó lên validation-fold.
4. Tính F1_UP validation.
5. Lấy trung bình năm validation-fold.

`gap=5` ở code hiện đếm năm dòng trên bảng nhiều mã, không phải năm ngày hoặc
năm phiên cho từng mã. Vì threshold được chọn bằng prediction in-sample trên
train-fold, tài liệu không gọi quy trình này “leak-free” tuyệt đối.

## 4. Ba cấu hình được chốt

Nguồn quyết định là `experiments/manual_config.json`, đã đối chiếu đúng selected
run trong `experiments/tuning_history.csv` và fingerprint `f6cb3ac8f820`.

| Model | Bộ tham số được chốt | Tuning CV F1_UP lúc chốt |
|---|---|---:|
| Logistic Regression | `C=6.5867e-05`, `solver=liblinear` | `0.5490` |
| Random Forest | `n_estimators=130`, `max_depth=4`, `min_samples_leaf=25`, `max_features=0.35` | `0.5491` |
| Gradient Boosting | `n_estimators=120`, `learning_rate=0.05`, `max_depth=2`, `subsample=1.0` | `0.5453` |

Random Forest `0.5491` là tuning CV lúc chốt. Con số `0.5488` thuộc official CV
rerun, không thay thế nó trên sơ đồ tuning. `max_features=0.35` là lựa chọn được
khóa, không phải nghiệm tốt nhất duy nhất vì `0.36/0.37` đồng hạng CV.

## 5. Tuning CV khác official CV và TEST

Ba lớp số liệu có mục đích khác nhau:

| Loại số | Dữ liệu | Mục đích |
|---|---|---|
| Tuning Lab CV lúc chốt | TRAIN | Chọn và lưu cấu hình thủ công |
| Official CV rerun | TRAIN | Chạy lại trong pipeline chính thức để tạo report/artifact |
| TEST F1_UP | TEST held-out | Xếp hạng và chọn Final Model |

Official CV hiện tại:

- Logistic Regression: `0.548977`.
- Random Forest: `0.548766`.
- Gradient Boosting: `0.545268`.

Sai lầm cần tránh: thấy Logistic Regression hoặc Random Forest có CV cao hơn rồi
kết luận đó là Final Model. Code không chọn Final bằng CV.

## 6. Chấm TEST và chọn Final Model

Sau khi đủ ba cấu hình:

```text
Best LR -----\
Best RF ------> Fit trên full TRAIN -> Tune threshold trên TRAIN
Best GB -----/                         -> Evaluate held-out TEST
                                          -> Compare -> Final Model
```

TEST metrics:

| Model | TEST F1_UP | TEST Recall_UP |
|---|---:|---:|
| Logistic Regression | `0.5024` | `0.9332` |
| Random Forest | `0.5041` | `0.9424` |
| Gradient Boosting | `0.5053` | `0.9176` |

Quy tắc trong code:

1. Loại Dummy Classifier khỏi danh sách ứng viên; Dummy chỉ là baseline.
2. Chọn TEST F1_UP cao hơn.
3. Chỉ khi F1_UP bằng nhau mới dùng Recall_UP cao hơn.
4. Nếu cả hai bằng nhau, ưu tiên model đơn giản: LR, rồi RF, rồi GB.

Gradient Boosting được chọn vì TEST F1_UP cao nhất `0.5053`. Random Forest có
Recall_UP cao hơn, nhưng Recall không được dùng vì F1_UP không hòa. Chênh F1 giữa
GB và RF chỉ khoảng `0.0013`, nên gọi “nhỉnh hơn”, không “vượt trội”.

## 7. Cách trình bày ngắn khi bảo vệ

> Ba model dùng cùng dữ liệu và cùng cơ chế CV. Khối màu riêng cho biết đúng
> hyperparameter được tuning; thiết lập cân bằng lớp nằm ở Fit. Cấu hình chốt được
> dựng lại và fit trên TRAIN, sau đó cả ba được chấm trên held-out TEST. Gradient
> Boosting có TEST F1_UP cao nhất nên được chọn làm Final Model; Dummy chỉ là baseline.

Mở lần lượt ba sơ đồ tuning, sau đó mở `final-model-selection.html` để kết thúc
câu chuyện. Không trộn tuning CV với TEST metrics.
