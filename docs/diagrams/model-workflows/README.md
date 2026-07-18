# Lưu đồ tuning và chọn Final Model

Thư mục chứa bốn lưu đồ Archify. Ba lưu đồ đầu mô tả tuning riêng từng model;
lưu đồ cuối mô tả cách ba cấu hình đã chốt được fit trên TRAIN, đánh giá trên TEST
và chọn Final Model.

| File HTML | Nội dung |
|---|---|
| `logistic-regression.html` | Hyperparameter, CV lúc chốt và cấu hình Logistic Regression |
| `random-forest.html` | Hyperparameter, CV lúc chốt và cấu hình Random Forest |
| `gradient-boosting.html` | Hyperparameter, CV lúc chốt và cấu hình Gradient Boosting |
| `final-model-selection.html` | Đánh giá TEST và chọn Gradient Boosting làm Final Model |

File `*.workflow.json` cùng tên là nguồn duy nhất để render HTML. Không sửa HTML
bằng tay.

## Cách đọc ba lưu đồ tuning

Mỗi lưu đồ có cùng bốn lane:

```text
SHARED DATA FLOW
Fetch -> Clean -> 20 Features + Label -> Time Split
                                                |
                                                v
TRAIN -> HYPERPARAMETER ĐƯỢC THAY ĐỔI
      -> Fit model
      -> TimeSeriesSplit CV trên TRAIN
      -> Tune decision_threshold trên train-fold
      -> CV F1_UP lúc chốt
      -> BỘ THAM SỐ ĐƯỢC CHỐT
```

Phần dùng chung luôn màu xám. Khối hyperparameter, Fit, CV lúc chốt và cấu hình
chốt dùng màu riêng của model:

| Thành phần | Màu light theme | Archify type |
|---|---|---|
| Shared data flow | Slate `#64748b` | `external` |
| Logistic Regression | Cyan `#0891b2` | `frontend` |
| Random Forest | Amber `#d97706` | `cloud` |
| Gradient Boosting | Violet `#7c3aed` | `database` |
| TEST và Final Model | Emerald `#059669` | `backend` |

Tên model và nhãn lane luôn đi kèm màu; không cần dựa riêng vào màu để hiểu hình.

## Hyperparameter và thiết lập cố định

| Model | Hyperparameter được thay đổi | Thiết lập cố định ở node Fit |
|---|---|---|
| Logistic Regression | `C`, `solver` | `StandardScaler`, `class_weight=balanced`, `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | `class_weight=balanced_subsample`, `n_jobs=-1`, `random_state=42` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | `sample_weight=balanced`, `random_state=42` |

`decision_threshold` là output của bước Tune Threshold, không phải hyperparameter
model. Threshold được tối ưu theo F1_UP trên train-fold rồi áp dụng cho validation-fold.
`gap=5` hiện là năm dòng trên bảng TRAIN gộp nhiều mã, không phải năm phiên của
từng mã.

## Bảng nguồn sự thật

Ba loại điểm sau không được dùng thay nhau:

| Model | Cấu hình chốt | Tuning Lab CV lúc chốt | Official CV rerun | TEST F1_UP | TEST Recall_UP |
|---|---|---:|---:|---:|---:|
| Logistic Regression | `C=6.5867e-05`, `solver=liblinear` | `0.548995` | `0.548977` | `0.502401` | `0.933190` |
| Random Forest | `130`, `4`, `25`, `0.35` | `0.549132` | `0.548766` | `0.504059` | `0.942432` |
| Gradient Boosting | `120`, `0.05`, `2`, `1.0` | `0.545263` | `0.545268` | `0.505315` | `0.917595` |

Nguồn:

- Cấu hình chốt: `experiments/manual_config.json`.
- Tuning Lab CV lúc chốt: selected run tương ứng trong `experiments/tuning_history.csv`.
- Official CV rerun: `reports/pipeline_summary.json` và cột `cv_f1_up` trong
  `reports/model_comparison.csv`.
- TEST và cờ `selected`: `reports/model_comparison.csv`; đối chiếu
  `models/model_metadata.json` và `reports/pipeline_summary.json`.

Random Forest phải ghi `0.5491` trên sơ đồ tuning. `0.5488` vẫn đúng, nhưng chỉ
đúng cho official CV rerun. Cấu hình `max_features=0.35` là cấu hình được chốt;
`0.36` và `0.37` đồng hạng CV nên không gọi `0.35` là nghiệm tối ưu duy nhất.

## Chọn Final Model

`final-model-selection.html` thể hiện luồng:

```text
Best LR + Best RF + Best GB
-> dựng lại estimator và fit trên full TRAIN
-> chốt threshold trên TRAIN
-> đánh giá held-out TEST
-> loại Dummy khỏi danh sách ứng viên
-> TEST F1_UP giảm dần
-> nếu bằng nhau: Recall_UP giảm dần
-> nếu vẫn bằng: ưu tiên LR, rồi RF, rồi GB
-> Gradient Boosting
```

Gradient Boosting được chọn vì TEST F1_UP cao nhất `0.505315`. Chênh lệch với
Random Forest chỉ khoảng `0.001256`; dùng “cao nhất” hoặc “nhỉnh hơn”, không gọi
“vượt trội”. Random Forest có Recall_UP cao hơn, nhưng tie-break Recall không được
dùng vì F1_UP không bằng nhau. Dummy chỉ là baseline và bị loại trước khi xếp hạng.

## Render HTML

Chạy từ repo root:

```powershell
$renderer="$env:USERPROFILE\.codex\skills\archify\renderers\workflow\render-workflow.mjs"
rtk node $renderer "docs\diagrams\model-workflows\logistic-regression.workflow.json" "docs\diagrams\model-workflows\logistic-regression.html"
rtk node $renderer "docs\diagrams\model-workflows\random-forest.workflow.json" "docs\diagrams\model-workflows\random-forest.html"
rtk node $renderer "docs\diagrams\model-workflows\gradient-boosting.workflow.json" "docs\diagrams\model-workflows\gradient-boosting.html"
rtk node $renderer "docs\diagrams\model-workflows\final-model-selection.workflow.json" "docs\diagrams\model-workflows\final-model-selection.html"
```

Renderer phải kết thúc exit code `0`; AJV và layout validator phải pass.

## Xuất PNG cho Word

1. Mở HTML, chọn light theme.
2. Chọn `Export -> Download PNG`; renderer xuất 4x, `2880x2608`.
3. Lưu vào `docs/report_assets/` với tên `tuning_*.png` và
   `final_model_selection.png`.
4. Khi chèn Word, crop `17.18%` phía đáy để bỏ legend generic của Archify; không
   sửa HTML hoặc renderer.
5. Chèn rộng tối đa 6 inch và kiểm tra bằng bản render DOCX ở kích thước thật.

Cards dưới HTML chỉ giải thích bổ sung và không nằm trong PNG. Nội dung bắt buộc
đã được đặt trong lane/node của SVG.
