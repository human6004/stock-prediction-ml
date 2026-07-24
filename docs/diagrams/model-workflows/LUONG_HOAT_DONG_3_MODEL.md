# Luồng hoạt động ba model

## Phần chung

1. Nhận hyperparameter do người dùng nhập và kiểm tra giới hạn.
2. Dùng nguyên TRAIN frame có `trading_date` và `label_end_date`.
3. Chia 4 fold theo ngày giao dịch duy nhất (`CV_N_SPLITS = 4`, `TimeSeriesSplit`).
4. Bỏ đúng 5 phiên (`CV_GAP_SESSIONS`) trước validation và purge label overlap.
5. Fit model tạm từng fold; gộp xác suất out-of-fold rồi tự chọn một ngưỡng quyết định (OOF threshold, ràng buộc `predicted_up_ratio <= 0.50` và `precision >= base rate`); recompute F1/P/R từng fold tại ngưỡng đó; bỏ model tạm.
6. Lưu F1/Precision/Recall, mean/std, dải ngày, `decision_threshold`, thời gian và fingerprint.
7. Hiển thị metric và để người dùng tự chọn run áp dụng.

## Phần khác nhau

### Logistic Regression

```text
C > 0; solver = lbfgs hoặc liblinear
```

`C` quyết định regularization; `solver` quyết định thuật toán tối ưu. Pipeline có `StandardScaler` và class weight balanced.

### Random Forest

```text
n_estimators, max_depth, min_samples_leaf: số nguyên dương
max_features: sqrt/log2 hoặc số thực trong (0,1]
```

Các tham số điều khiển số cây, độ sâu, kích thước lá và số feature xét ở mỗi split.

### Gradient Boosting

```text
n_estimators, max_depth: số nguyên dương
learning_rate, subsample: số thực trong (0,1]
```

Các tham số điều khiển số stage, bước học, độ phức tạp cây và tỷ lệ mẫu. Balanced sample weight được truyền lúc fit.

## Sau tuning

Cấu hình người dùng chọn không phải Final Model. Nó chỉ cung cấp hyperparameters để fit candidate trên toàn TRAIN.

```text
Chosen LR -> LR candidate ┐
Chosen RF -> RF candidate ├-> VALIDATION selection
Chosen GB -> GB candidate ┘
```

Ba candidate được chấm trên VALIDATION, mỗi model tại `decision_threshold` riêng; chọn theo F1_UP, rồi Recall_UP, rồi đơn giản (LR→RF→GB). Winner được clone, refit TRAIN+VALIDATION, TEST một lần tại ngưỡng của artifact, rồi promote. TEST không chọn model và không có refit sau TEST.
