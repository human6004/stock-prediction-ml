# Luồng hoạt động ba model

## Phần chung

1. Nhận hyperparameter do người dùng nhập và kiểm tra giới hạn. Tuning Lab không giới hạn số run.
2. Lấy TRAIN frame có `trading_date` và `label_end_date`, nhưng CV chỉ giữ các phiên `>= CV_START_DATE` (`2021-01-01`); phiên cũ hơn bị bỏ im lặng.
3. Chia 4 fold theo ngày giao dịch duy nhất (`CV_N_SPLITS = 4`, `TimeSeriesSplit` trên mảng ngày unique, không phải trên row).
4. Bỏ đúng 5 phiên (`CV_GAP_SESSIONS`) trước validation và purge label overlap (`label_end_date < validation_start`).
5. Fit model tạm từng fold (clone mới mỗi fold); gộp xác suất out-of-fold rồi tự chọn một ngưỡng quyết định: grid `0.05..0.95` step `0.01`, 2 gate cứng `predicted_up_ratio <= 0.50` và `precision >= oof_up_rate`, tie-break `(F1, precision, threshold)`. Không thỏa gate thì **raise** (`No OOF threshold satisfies...`), run ghi status error - **không** fallback 0.5. Recompute F1/P/R từng fold tại ngưỡng đó; bỏ model tạm, chỉ `(y, P(UP))` sống sót.
6. Lưu F1/Precision/Recall, mean/std, dải ngày, `decision_threshold`, thời gian và fingerprint.
7. Provenance recheck 4 fold, ghi một dòng vào `experiments/tuning_history.csv`, hiển thị metric để người dùng tự chọn run áp dụng.

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

Các tham số điều khiển số cây, độ sâu, kích thước lá và số feature xét ở mỗi split. Cân bằng lớp bằng `class_weight="balanced_subsample"` (cố định, không phải hyperparameter), thêm `n_jobs=-1`.

### Gradient Boosting

```text
n_estimators, max_depth: số nguyên dương
learning_rate, subsample: số thực trong (0,1]
```

Các tham số điều khiển số stage, bước học, độ phức tạp cây và tỷ lệ mẫu. `GradientBoostingClassifier` **không có** `class_weight`, nên cân bằng lớp đến từ `sample_weight=compute_sample_weight("balanced", y)` truyền lúc fit - cùng cách trong từng CV fold và trong lần fit toàn TRAIN.

## Sau tuning

Cấu hình người dùng chọn không phải Final Model. Nó cung cấp hyperparameters để fit candidate trên toàn TRAIN, kèm `decision_threshold` và khối metric CV đã lưu. Pipeline **không** chạy lại CV: `_cv_from_selected` phát lại đúng số đã ghi trong `manual_config.json` (schema v4).

```text
Chosen LR -> LR candidate ┐
Chosen RF -> RF candidate ├-> VALIDATION selection
Chosen GB -> GB candidate ┘
```

Ba candidate được chấm trên VALIDATION, mỗi model tại `decision_threshold` riêng; chọn theo F1_UP desc, rồi Recall_UP desc, rồi `simplicity_rank` asc (LR→RF→GB). Winner được clone, fresh refit TRAIN+VALIDATION, TEST một lần tại ngưỡng mang theo, rồi atomic promote. TEST không chọn model và không có refit sau TEST.

Hai baseline gate **không đối xứng**:

| Mốc | Miss baseline |
|---|---|
| VALIDATION | `RuntimeError`, dừng hẳn pipeline |
| TEST | chỉ gắn warning flag, vẫn chạy tiếp |

`experiments/evaluation_registry.json` khóa TEST theo one-shot. Bốn state `started` / `evaluated` / `published` / `release_failed` đều tính là "TEST đã mở", không mở lại được.

Mốc TRAIN/VALIDATION/TEST là **rolling**: `resolve_protocol_dates` neo `test_end` vào phiên mới nhất trong dataset; `TRAIN_END_DATE` / `VALIDATION_END_DATE` / `TEST_END_DATE` trong settings chỉ là fallback.

Lưu ý thứ tự: `write_reports` chạy **trước** `atomic_model_release`. Release lỗi sẽ để lại report mô tả một model chưa publish, và snapshot đó cháy vĩnh viễn (`release_failed`).
