# Flowchart tuning và chọn Final Model - protocol v3

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

Flow chuẩn:

```text
user-entered configs
-> 4-fold date-purged CV trên TRAIN (gap 5 phiên)
-> chọn OOF decision threshold cho từng run
-> người dùng chọn config từng model
-> fit ba candidates trên toàn TRAIN
-> chọn family trên VALIDATION (F1_UP, Recall_UP, simplicity LR→RF→GB)
-> fresh refit winner trên TRAIN+VALIDATION
-> TEST một lần tại threshold của artifact
-> atomic promote đúng artifact vừa TEST
-> UI/CLI inference tại decision threshold của model
```

Các export trong thư mục này mô tả luồng model hiện hành.
