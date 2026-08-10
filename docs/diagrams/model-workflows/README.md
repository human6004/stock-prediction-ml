# Flowchart tuning và chọn Final Model - policy `rolling_recent_cv_oof_threshold`

Các sơ đồ hiện hành:

- [Logistic Regression](logistic-regression.html)
- [Random Forest](random-forest.html)
- [Gradient Boosting](gradient-boosting.html)
- [Chọn Final Model](final-model-selection.html)

Mỗi file HTML độc lập, có dark/light theme và menu export PNG/JPEG/WebP/SVG.

Ba sơ đồ tuning giữ cùng logic nhưng làm nổi bật hyperparameter riêng:

| Model | Hyperparameters người dùng nhập |
|---|---|
| Logistic Regression | `C`, `solver` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

`TUNABLE_PARAM_SCHEMA` (`config/settings.py`) là **domain validate cho form nhập tay**, không phải
search space: repo không dùng `GridSearchCV` / `RandomizedSearchCV` / `param_grid` nào. Người dùng
nhập một config vào Tuning Lab, `services/tuning_lab.validate_params` coerce kiểu + check biên,
rồi `evaluate_single_config` chạy CV cho đúng config đó.

Class balancing khác nhau theo family và **không** phải hyperparameter:

| Model | Cách cân bằng lớp |
|---|---|
| Logistic Regression | `class_weight="balanced"` trong Pipeline có `StandardScaler` |
| Random Forest | `class_weight="balanced_subsample"` |
| Gradient Boosting | không có `class_weight`; truyền `sample_weight` balanced lúc fit |

Flow chuẩn:

```text
user-entered configs (Tuning Lab, không giới hạn số run)
-> 4-fold date-purged CV trên TRAIN, chỉ các phiên >= CV_START_DATE (2021-01-01)
   (TimeSeriesSplit theo ngày giao dịch duy nhất, gap 5 phiên, purge label_end_date)
-> chọn một OOF decision threshold cho từng run (grid 0.05..0.95 step 0.01,
   2 gate cứng: predicted_up_ratio <= 0.50 và precision >= OOF up_rate;
   không thỏa gate thì raise, run ghi status error - KHÔNG fallback 0.5)
-> provenance recheck 4 fold + ghi một dòng vào experiments/tuning_history.csv
-> người dùng chọn config từng model (manual_config.json, schema v4)
-> pipeline fit ba candidates trên toàn TRAIN, KHÔNG chạy lại CV
   (metric CV lấy nguyên từ manual_config đã lưu)
-> chọn family trên VALIDATION (F1_UP desc, Recall_UP desc, simplicity LR<RF<GB)
   miss baseline VALIDATION -> RuntimeError, dừng hẳn
-> fresh refit winner trên TRAIN+VALIDATION
-> TEST đúng một lần tại threshold mang theo từ CV
   miss baseline TEST -> chỉ warning flag, không dừng
-> atomic promote đúng artifact vừa TEST
-> UI/CLI inference tại decision threshold của model
```

Mốc thời gian là **rolling**: `resolve_protocol_dates` neo `test_end` vào phiên mới nhất
trong dataset; `TRAIN_END_DATE` / `VALIDATION_END_DATE` / `TEST_END_DATE` trong settings
chỉ là fallback.

`experiments/evaluation_registry.json` khóa TEST theo one-shot. Bốn state
`started` / `evaluated` / `published` / `release_failed` đều tính là "TEST đã mở".

Lưu ý: `write_reports` chạy **trước** `atomic_model_release`, nên release lỗi sẽ để lại
report mô tả một model chưa được publish, và snapshot đó cháy vĩnh viễn.

Các export trong thư mục này mô tả luồng model hiện hành.
