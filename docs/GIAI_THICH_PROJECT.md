# Giải thích project dự báo xu hướng cổ phiếu HOSE cho người mới

Tài liệu này được viết lại sau khi đọc lại code, report, model metadata, web demo, database schema và roadmap hiện tại.

- Project: `D:\study\niên luận\stock-prediction-ml`
- Roadmap: `D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md`
- Ngày đọc lại: `2026-06-27`
- Mục tiêu của tài liệu: giải thích từ đầu, coi bạn là người mới chưa biết project, chưa quen thuật ngữ chứng khoán và Machine Learning.

## 1. Project này làm gì, nói thật ngắn gọn?

Project này xây dựng một hệ thống Machine Learning để trả lời câu hỏi:

```text
Một mã cổ phiếu trên sàn HOSE có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không?
```

Kết quả chỉ có 2 lớp:

| Kết quả | Nghĩa |
|---|---|
| `UP` | Model dự báo giá đóng cửa sau 5 phiên có khả năng tăng hơn 1% so với ngày hiện tại |
| `NOT_UP` | Model dự báo giá không đạt điều kiện tăng hơn 1%; có thể giảm, đi ngang, hoặc tăng nhẹ dưới 1% |

Đây là project học thuật/niên luận. Nó không phải hệ thống tư vấn đầu tư, không tự mua bán cổ phiếu, không đảm bảo lợi nhuận.

## 2. Roadmap yêu cầu gì?

Roadmap mô tả một pipeline chuẩn:

```text
Thu thập dữ liệu HOSE
-> Lưu database
-> Làm sạch dữ liệu
-> Kiểm tra chất lượng dữ liệu
-> Tạo feature Machine Learning
-> Tạo nhãn UP / NOT_UP
-> Chia train/test theo thời gian
-> Cross Validation theo thời gian
-> Hyperparameter tuning
-> Đánh giá model
-> Chọn final_model.pkl
-> Web Flask demo dự báo
```

Code hiện tại đã triển khai đúng hướng chính của roadmap:

- Có dữ liệu HOSE dạng OHLCV.
- Có script cập nhật dữ liệu bằng `vnstock`.
- Có làm sạch dữ liệu và báo cáo chất lượng dữ liệu.
- Có lọc mã đủ tối thiểu 250 phiên giao dịch.
- Có tạo 15 feature kỹ thuật.
- Có tạo nhãn `UP / NOT_UP` với horizon 5 phiên và threshold 1%.
- Có chia train/test theo thời gian bằng `label_end_date`.
- Có `TimeSeriesSplit(gap=5)` cho Cross Validation.
- Có tuning Logistic Regression, Random Forest, Gradient Boosting.
- Có Dummy Classifier làm baseline.
- Có chọn final model theo `F1_UP`.
- Có Flask web demo và CLI dự báo.
- Có SQLite để sync dữ liệu/report và lưu lịch sử dự báo.

Điểm cần hiểu: trong code hiện tại, CSV vẫn là artefact chính của pipeline; SQLite là bản sync để lưu dữ liệu/report/prediction history.

## 3. Bức tranh tổng thể dễ hiểu

Hãy tưởng tượng project là một dây chuyền:

```text
Dữ liệu giá cổ phiếu thô
-> làm sạch
-> biến thành các con số mô tả xu hướng
-> tạo đáp án đúng/sai trong quá khứ
-> cho model học
-> kiểm tra model trên tương lai
-> chọn model tốt nhất
-> đưa model lên web demo để dự báo mã mới
```

Trong code, dây chuyền đó tương ứng:

```text
shared_dataset/hose_stock_raw.csv
-> data/processed/hose_stock_clean.csv
-> data/processed/hose_stock_features.csv
-> data/processed/ml_dataset.csv
-> models/*.pkl
-> models/final_model.pkl
-> reports/*.csv/.json/.png/.txt
-> app.py / scripts/predict_stock.py
```

## 4. Các thư mục và file quan trọng

| File/thư mục | Vai trò |
|---|---|
| `config/settings.py` | Cấu hình trung tâm: đường dẫn, horizon, threshold, feature list, split date, model/report paths |
| `scripts/run_pipeline.py` | Script chạy toàn bộ pipeline từ raw data đến model/report/database |
| `scripts/fetch_hose_data.py` | Cập nhật thêm dữ liệu OHLCV bằng `vnstock` nguồn `KBS` |
| `scripts/preprocess_data.py` | Chạy riêng bước làm sạch dữ liệu |
| `scripts/build_features.py` | Chạy riêng bước tạo feature, label và train/test split |
| `scripts/train_tune_models.py` | Chạy riêng bước train/tune model |
| `scripts/evaluate_models.py` | Chạy riêng bước đánh giá model |
| `scripts/select_final_model.py` | Chạy riêng bước chọn final model |
| `scripts/predict_stock.py` | Dự báo bằng command line |
| `scripts/finetune_model.py` | File cũ, đã deprecated; code bảo dùng `train_tune_models.py` |
| `scripts/build_academic_report.py` | Sinh file Word report trong `docs/` từ report/model hiện tại |
| `services/preprocessing.py` | Làm sạch dữ liệu và lọc mã đủ điều kiện |
| `services/feature_engineering.py` | Tạo feature, tạo target, chia train/test |
| `services/model_tuning.py` | Train Dummy và tuning 3 model chính |
| `services/model_evaluation.py` | Tính metric, chọn final model, ghi report |
| `services/prediction_service.py` | Tính feature mới nhất và dự báo một mã cổ phiếu |
| `services/database_service.py` | Sync dữ liệu/report vào SQLite và log lịch sử dự báo |
| `database/init_db.sql` | Schema SQLite |
| `app.py` | Flask web backend |
| `templates/` | Giao diện HTML của web |
| `static/style.css` | CSS của web |
| `models/final_model.pkl` | Model cuối đang được web/CLI sử dụng |
| `models/model_metadata.json` | Metadata: model name, feature order, threshold, metric, best params |
| `reports/model_comparison.csv` | Bảng so sánh các model |
| `reports/pipeline_summary.json` | Tóm tắt lần chạy pipeline mới nhất |

## 5. Dữ liệu OHLCV là gì?

Project dùng dữ liệu giá cổ phiếu theo ngày. OHLCV là viết tắt của:

| Cột | Nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu, ví dụ `FPT`, `HPG`, `SSI` |
| `trading_date` | Ngày giao dịch |
| `open` | Giá mở cửa trong phiên |
| `high` | Giá cao nhất trong phiên |
| `low` | Giá thấp nhất trong phiên |
| `close` | Giá đóng cửa trong phiên |
| `volume` | Khối lượng giao dịch |

Project hiện chỉ dùng OHLCV. Nó không dùng tin tức, báo cáo tài chính, dữ liệu realtime, dữ liệu intraday, sentiment analysis, LSTM, Transformer hay Deep Learning.

## 6. Các thông số cấu hình chính

Theo `config/settings.py`:

| Thông số | Giá trị hiện tại | Nghĩa dễ hiểu |
|---|---:|---|
| `RAW_DATA_PATH` | `D:\study\niên luận\shared_dataset\hose_stock_raw.csv` | File dữ liệu thô nằm ngoài repo |
| `SPLIT_DATE` | `2025-12-31` | Mốc thời gian chia train/test |
| `PREDICTION_HORIZON` | `5` | Dự báo sau 5 phiên giao dịch |
| `UP_THRESHOLD` | `0.01` | Tăng hơn 1% thì tính là `UP` |
| `MIN_TRADING_DAYS` | `250` | Mã có dưới 250 phiên hợp lệ bị loại khỏi train |
| `MIN_AVERAGE_VOLUME` | `0` | Hiện không lọc theo thanh khoản trung bình |
| `CV_N_SPLITS` | `5` | Cross Validation chia train thành 5 fold |
| `CV_GAP` | `5` | Chừa khoảng cách 5 mẫu giữa train và validation |
| `TUNING_N_ITER` | `12` | Mỗi model chính thử 12 bộ tham số |
| `TUNING_SCORING` | `f1` | Tuning ưu tiên F1 của lớp `UP` |
| `RANDOM_STATE` | `42` | Giúp kết quả random ổn định hơn |

`5 phiên giao dịch` không phải luôn là 5 ngày lịch. Nếu có cuối tuần hoặc ngày nghỉ, 5 phiên giao dịch có thể kéo dài hơn 5 ngày.

## 7. Số liệu dữ liệu hiện tại

Theo `reports/pipeline_summary.json`, `models/model_metadata.json` và các report hiện tại:

| Hạng mục | Giá trị |
|---|---:|
| Raw data | 548,074 dòng |
| Số mã trong raw data | 400 mã |
| Khoảng ngày raw data | 2019-08-14 đến 2026-06-19 |
| Dòng OHLC bất hợp lệ bị loại | 226 dòng |
| Clean data | 547,848 dòng |
| Mã sau làm sạch | 400 mã |
| Mã đủ điều kiện train | 396 mã |
| Mã bị loại | 4 mã |
| Feature data | 510,185 dòng |
| ML dataset sau khi có label | 508,205 dòng |
| Tỷ lệ `UP` trong ML dataset | 38.05% |
| Tỷ lệ `NOT_UP` trong ML dataset | 61.95% |

4 mã bị loại vì chưa đủ 250 phiên giao dịch:

| Mã | Số dòng | Khoảng dữ liệu | Lý do |
|---|---:|---|---|
| `CRV` | 136 | 2025-10-10 đến 2026-06-19 | `fewer_than_250_trading_days` |
| `TCX` | 164 | 2025-10-21 đến 2026-06-19 | `fewer_than_250_trading_days` |
| `VCK` | 124 | 2025-12-16 đến 2026-06-19 | `fewer_than_250_trading_days` |
| `VPX` | 127 | 2025-12-11 đến 2026-06-19 | `fewer_than_250_trading_days` |

`reports/fetch_report.json` cho biết lần fetch gần nhất thêm 3,629 dòng mới, raw data sau fetch là 548,074 dòng. Có 9 mã fetch lỗi trong lần đó, nhưng các mã này vẫn có dữ liệu cũ trong CSV nếu trước đó đã tồn tại.

## 8. Làm sạch dữ liệu là gì?

Làm sạch dữ liệu là bước biến dữ liệu thô thành dữ liệu đáng tin hơn trước khi train model.

Trong `services/preprocessing.py`, code làm các việc chính:

1. Kiểm tra file raw có tồn tại và đọc được không.
2. Kiểm tra có đủ cột bắt buộc: `symbol`, `trading_date`, `open`, `high`, `low`, `close`, `volume`.
3. Chuẩn hóa `symbol` thành chữ hoa.
4. Chuyển `trading_date` về kiểu ngày.
5. Chuyển `open`, `high`, `low`, `close`, `volume` về dạng số.
6. Xóa dòng thiếu dữ liệu.
7. Xóa dòng trùng theo `symbol + trading_date`.
8. Loại dòng có giá <= 0.
9. Loại dòng có `volume` âm.
10. Loại dòng OHLC sai logic, ví dụ `high` nhỏ hơn `open`, `close` hoặc `low`.
11. Sắp xếp theo `symbol`, `trading_date`.
12. Thống kê mỗi mã có bao nhiêu dòng, ngày bắt đầu/kết thúc, volume trung bình.
13. Lọc mã đủ điều kiện train: hiện cần tối thiểu 250 phiên giao dịch.

Output của bước này:

```text
data/processed/hose_stock_clean.csv
reports/data_quality_report.csv
reports/eligible_symbols.csv
reports/excluded_symbols.csv
```

## 9. Feature là gì?

Feature là dữ liệu đầu vào cho model.

Con người nhìn biểu đồ, đường trung bình, volume để đoán xu hướng. Model không nhìn biểu đồ trực tiếp, nên project biến lịch sử giá/volume thành các cột số. Các cột số đó gọi là feature.

Project hiện dùng 15 feature:

| Feature | Nghĩa dễ hiểu |
|---|---|
| `return_1d` | Giá đóng cửa thay đổi bao nhiêu so với phiên trước |
| `return_3d` | Giá thay đổi bao nhiêu trong 3 phiên gần nhất |
| `return_5d` | Giá thay đổi bao nhiêu trong 5 phiên gần nhất |
| `close_open_return` | Trong cùng ngày, giá đóng cửa cao/thấp hơn giá mở cửa bao nhiêu |
| `sma5` | Trung bình giá đóng cửa 5 phiên |
| `sma20` | Trung bình giá đóng cửa 20 phiên |
| `sma50` | Trung bình giá đóng cửa 50 phiên |
| `close_vs_sma20` | Giá hiện tại cao/thấp hơn SMA20 bao nhiêu |
| `sma20_vs_sma50` | SMA20 cao/thấp hơn SMA50 bao nhiêu |
| `rsi14` | Chỉ báo sức mạnh tương đối trong 14 phiên |
| `volatility_5d` | Độ biến động trong 5 phiên |
| `volatility_20d` | Độ biến động trong 20 phiên |
| `price_range` | Biên độ dao động trong ngày: `(high - low) / close` |
| `volume_change_1d` | Volume tăng/giảm bao nhiêu so với phiên trước |
| `volume_ratio_20` | Volume hiện tại so với volume trung bình 20 phiên |

Các feature được tính riêng cho từng `symbol`. Lịch sử của `FPT` không được trộn sang `HPG`, `SSI` hay mã khác.

## 10. Một vài feature quan trọng nên hiểu kỹ

### Return

`return_1d` được tính gần giống:

```text
return_1d = close(t) / close(t-1) - 1
```

Ví dụ:

- `0.02` nghĩa là tăng 2%.
- `-0.03` nghĩa là giảm 3%.

### SMA

SMA là `Simple Moving Average`, tức trung bình trượt đơn giản.

- `sma5`: trung bình 5 phiên, phản ánh ngắn hạn.
- `sma20`: trung bình 20 phiên, phản ánh trung hạn hơn.
- `sma50`: trung bình 50 phiên, phản ánh dài hơn.

Nếu giá hiện tại cao hơn SMA20, có thể hiểu là giá đang nằm trên mức trung bình 20 phiên.

### RSI

RSI là `Relative Strength Index`, chỉ báo sức mạnh tương đối, thường nằm từ 0 đến 100.

- Gần 70 trở lên: giá đã tăng mạnh trong giai đoạn gần đây.
- Gần 30 trở xuống: giá đã giảm mạnh.
- Gần 50: cân bằng hơn.

Trong code, nếu cả gain và loss đều bằng 0 thì RSI được gán 50; nếu loss bằng 0 thì RSI là 100.

### Volatility

Volatility là độ biến động. Giá dao động càng mạnh thì volatility càng cao. Cổ phiếu biến động cao thường khó dự báo hơn.

### Volume ratio

`volume_ratio_20 = volume hôm nay / volume trung bình 20 phiên`.

Ví dụ `volume_ratio_20 = 2` nghĩa là volume hôm nay gấp đôi trung bình 20 phiên.

## 11. Target, label, horizon, threshold là gì?

Target là đáp án mà model cần học.

Với mỗi mã cổ phiếu tại ngày `t`, project nhìn lại dữ liệu quá khứ để tạo feature, rồi nhìn sang tương lai 5 phiên để tạo đáp án:

```text
future_close_5d = close(t+5)
future_return_5d = close(t+5) / close(t) - 1
```

Sau đó tạo nhãn:

```text
target = 1 nếu future_return_5d > 0.01
target = 0 nếu future_return_5d <= 0.01
```

Quy ước:

```text
1 = UP
0 = NOT_UP
```

Ví dụ:

| Giá ngày t | Giá sau 5 phiên | Return | Label |
|---:|---:|---:|---|
| 100 | 103 | 3.0% | `UP` |
| 100 | 101.1 | 1.1% | `UP` |
| 100 | 100.8 | 0.8% | `NOT_UP` |
| 100 | 99 | -1.0% | `NOT_UP` |

`future_return_5d` chỉ dùng để tạo đáp án trong lúc train/test. Khi dự báo thật, model không được biết tương lai.

## 12. Data leakage là gì?

Data leakage là lỗi rất nguy hiểm trong Machine Learning: model vô tình nhìn thấy thông tin tương lai hoặc thông tin đáp án trong lúc train.

Ví dụ sai:

```text
Dùng close(t+5) làm feature để dự báo target tại ngày t.
```

Nếu làm vậy, model gần như được nhìn thấy đáp án trước. Kết quả test có thể đẹp giả, nhưng khi dùng thật sẽ tệ.

Project đang giảm leakage bằng các cách:

- Feature tại ngày `t` chỉ dùng dữ liệu từ ngày `t` trở về trước.
- `future_close_5d`, `future_return_5d`, `target`, `label_end_date` không nằm trong `FEATURE_COLUMNS`.
- Train/test chia theo thời gian, không random shuffle.
- Train set dùng `label_end_date <= 2025-12-31`.
- Test set dùng `label_end_date > 2025-12-31`.
- Cross Validation dùng `TimeSeriesSplit(gap=5)`.
- Logistic Regression dùng `Pipeline(StandardScaler, LogisticRegression)`, giúp scaler fit đúng trong từng fold.

## 13. Train/test split hiện tại

Project không dùng random split. Dữ liệu cổ phiếu là dữ liệu thời gian, nên quá khứ phải dùng để dự báo tương lai.

Theo `reports/train_test_summary.csv`:

| Tập | Dòng | Số mã | Khoảng `trading_date` | Label boundary | UP | NOT_UP |
|---|---:|---:|---|---|---:|---:|
| Train | 466,973 | 396 | 2019-10-23 đến 2025-12-24 | `label_end_date <= 2025-12-31` | 180,584 | 286,389 |
| Test | 41,232 | 394 | 2025-10-30 đến 2026-06-12 | `label_end_date > 2025-12-31` | 12,786 | 28,446 |

Vì sao test có `trading_date` bắt đầu từ 2025-10-30, trước split date 2025-12-31?

Vì project chia theo `label_end_date`, tức ngày kết thúc của nhãn `t+5`. Một dòng có ngày giao dịch trước split vẫn có thể nằm trong test nếu nhãn 5 phiên của nó kết thúc sau split. Cách này an toàn hơn cho bài toán có horizon 5 phiên.

## 14. Cross Validation là gì?

Cross Validation là cách chia tập train thành nhiều phần nhỏ để thử model có ổn định không.

Với dữ liệu bình thường, người ta có thể chia ngẫu nhiên. Nhưng với dữ liệu cổ phiếu, chia ngẫu nhiên dễ làm tương lai lọt vào quá khứ.

Project dùng:

```text
TimeSeriesSplit(n_splits=5, gap=5)
```

Ý nghĩa:

- Chỉ dùng trên tập train.
- Chia train thành 5 fold theo thứ tự thời gian.
- Fold sau có nhiều dữ liệu quá khứ hơn fold trước.
- `gap=5` bỏ qua 5 mẫu giữa train và validation để giảm rủi ro leakage do target dùng 5 phiên tương lai.

Theo `reports/cv_fold_results.csv`, mỗi model chính có 5 fold validation, mỗi fold validation có 77,828 mẫu.

## 15. Hyperparameter tuning là gì?

Model có 2 loại tham số:

| Loại | Nghĩa |
|---|---|
| Parameter | Thứ model tự học từ dữ liệu |
| Hyperparameter | Thứ người lập trình cấu hình trước khi train |

Ví dụ với Random Forest:

- `n_estimators`: số cây trong rừng.
- `max_depth`: độ sâu tối đa của mỗi cây.
- `min_samples_leaf`: số mẫu tối thiểu ở một lá.
- `max_features`: số feature được xét mỗi lần chia nhánh.

Tuning là quá trình thử nhiều bộ hyperparameter để tìm bộ tốt hơn.

Project dùng:

```text
RandomizedSearchCV
n_iter = 12
scoring = f1
cv = TimeSeriesSplit(n_splits=5, gap=5)
```

`RandomizedSearchCV` không thử hết mọi tổ hợp như Grid Search. Nó chọn ngẫu nhiên một số tổ hợp để tiết kiệm thời gian.

## 16. Các model trong project

Project train 4 model:

| Model | Vai trò |
|---|---|
| Dummy Classifier | Baseline tối thiểu, gần như chỉ đoán lớp phổ biến nhất |
| Logistic Regression | Model tuyến tính đơn giản, dễ giải thích |
| Random Forest | Nhiều cây quyết định cùng bỏ phiếu, hợp dữ liệu bảng |
| Gradient Boosting | Nhiều cây học tuần tự, cây sau cố sửa lỗi cây trước |

Chi tiết trong code:

- Dummy dùng `strategy="most_frequent"`.
- Logistic Regression nằm trong `Pipeline(StandardScaler(), LogisticRegression(...))`.
- Random Forest dùng `class_weight="balanced_subsample"`, `n_jobs=-1`.
- Gradient Boosting dùng `GradientBoostingClassifier`.

Bộ tham số tốt nhất hiện tại theo `reports/best_params.json`:

| Model | Best params |
|---|---|
| Logistic Regression | `model__C=0.01`, `model__solver=lbfgs` |
| Random Forest | `n_estimators=120`, `max_depth=8`, `min_samples_leaf=50`, `max_features=0.5` |
| Gradient Boosting | `n_estimators=60`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.7` |

## 17. Các metric đánh giá là gì?

### Accuracy

Accuracy là tỷ lệ dự báo đúng tổng thể:

```text
accuracy = số dự báo đúng / tổng số mẫu
```

Nhưng với project này, accuracy không đủ tốt để chọn model. Lý do: lớp `NOT_UP` nhiều hơn `UP`. Một model cứ đoán `NOT_UP` nhiều có thể accuracy khá cao nhưng không bắt được cổ phiếu tăng.

### Precision_UP

Precision_UP trả lời:

```text
Trong những lần model dự báo UP, bao nhiêu lần thật sự UP?
```

Precision cao nghĩa là khi model nói `UP`, nó ít báo động nhầm hơn.

### Recall_UP

Recall_UP trả lời:

```text
Trong tất cả trường hợp thật sự UP, model bắt được bao nhiêu?
```

Recall cao nghĩa là model ít bỏ sót trường hợp tăng hơn.

### F1_UP

F1_UP là chỉ số cân bằng giữa Precision_UP và Recall_UP.

Project chọn final model theo `F1_UP`, vì mục tiêu chính là học lớp `UP`, không phải chỉ đoán đúng lớp đông hơn.

### Confusion Matrix

Confusion Matrix là bảng đếm đúng/sai:

- Thật `NOT_UP`, đoán `NOT_UP`: đúng.
- Thật `NOT_UP`, đoán `UP`: sai kiểu false positive.
- Thật `UP`, đoán `NOT_UP`: sai kiểu false negative.
- Thật `UP`, đoán `UP`: đúng.

## 18. Kết quả model hiện tại

Theo `reports/model_comparison.csv`:

| Model | Accuracy | Precision_UP | Recall_UP | F1_UP | CV F1_UP | Chọn |
|---|---:|---:|---:|---:|---:|---|
| Dummy Classifier | 0.6899 | 0.0000 | 0.0000 | 0.0000 | - | Không |
| Logistic Regression | 0.6050 | 0.3777 | 0.4226 | 0.3989 | 0.4172 | Không |
| Random Forest | 0.5926 | 0.3831 | 0.5142 | 0.4391 | 0.4540 | Có |
| Gradient Boosting | 0.6905 | 0.5105 | 0.0476 | 0.0870 | 0.1860 | Không |

Final model hiện tại:

```text
models/final_model.pkl = Random Forest
```

Vì sao Dummy accuracy cao nhưng không chọn?

Vì Dummy gần như đoán lớp phổ biến nhất là `NOT_UP`. Nó có accuracy 0.6899 nhưng `F1_UP = 0`, nghĩa là không bắt được lớp `UP`.

Vì sao Gradient Boosting accuracy cao nhưng không chọn?

Vì `Recall_UP = 0.0476`, tức trong 100 trường hợp thật sự `UP`, nó chỉ bắt được khoảng 4-5 trường hợp. Nó quá né lớp `UP`.

Vì sao Random Forest được chọn?

Vì Random Forest có `F1_UP` cao nhất trên test set và `Recall_UP` cao nhất trong các model chính.

## 19. Confusion matrix hiện tại

Theo `reports/confusion_matrix.csv`:

| Thực tế / Dự báo | Dự báo NOT_UP | Dự báo UP |
|---|---:|---:|
| Thực tế NOT_UP | 17,861 | 10,585 |
| Thực tế UP | 6,212 | 6,574 |

Cách đọc:

- 17,861: thực tế `NOT_UP`, model đoán `NOT_UP`, đúng.
- 10,585: thực tế `NOT_UP`, model đoán `UP`, sai.
- 6,212: thực tế `UP`, model đoán `NOT_UP`, sai.
- 6,574: thực tế `UP`, model đoán `UP`, đúng.

Tổng test:

```text
17,861 + 10,585 + 6,212 + 6,574 = 41,232 dòng
```

Kết quả này không phải "siêu chính xác". Nó phản ánh bài toán dự báo cổ phiếu bằng OHLCV là khó. Điểm tốt của project là pipeline đúng, có kiểm soát leakage, có baseline, có tuning và đánh giá trung thực.

## 20. Feature importance hiện tại

Vì final model là Random Forest, project xuất được `reports/feature_importance.csv`.

Top feature quan trọng:

| Hạng | Feature | Importance |
|---:|---|---:|
| 1 | `volatility_20d` | 0.3115 |
| 2 | `return_1d` | 0.0892 |
| 3 | `volatility_5d` | 0.0845 |
| 4 | `close_vs_sma20` | 0.0824 |
| 5 | `return_3d` | 0.0735 |
| 6 | `volume_ratio_20` | 0.0685 |
| 7 | `sma20_vs_sma50` | 0.0542 |
| 8 | `volume_change_1d` | 0.0439 |

Cách hiểu: Random Forest đang dùng nhiều thông tin về biến động 20 phiên, return gần nhất, biến động 5 phiên, vị trí giá so với SMA20, return 3 phiên và volume.

Lưu ý quan trọng: feature importance không chứng minh nguyên nhân thị trường. Nó chỉ nói model đã dựa vào feature đó nhiều trong quá trình ra quyết định.

## 21. Database hiện tại

Roadmap đề xuất SQLite và code hiện tại có:

```text
database/stock_prediction.db
```

Schema nằm ở `database/init_db.sql`.

Các bảng hiện tại:

| Bảng | Vai trò |
|---|---|
| `raw_prices` | Dữ liệu OHLCV thô |
| `clean_prices` | Dữ liệu sau làm sạch |
| `features` | Feature kỹ thuật lưu dạng JSON theo từng mã/ngày |
| `tuning_results` | Kết quả tuning model |
| `model_evaluations` | Kết quả đánh giá model |
| `predictions` | Lịch sử dự báo từ Flask/CLI |

Theo lần sync trong `pipeline_summary.json`:

| Bảng/data | Số dòng sync |
|---|---:|
| Raw rows | 548,074 |
| Clean rows | 547,848 |
| Feature rows | 510,185 |
| Tuning rows | 3 |
| Evaluation rows | 4 |

Lưu ý nhỏ: `services/database_service.py` hiện ghi `best_params_json` trong SQLite là `{}`. Best params chi tiết nên xem ở `reports/best_params.json` và `models/model_metadata.json`.

## 22. Web demo hoạt động thế nào?

Khi chạy:

```powershell
python app.py
```

Web mở tại:

```text
http://127.0.0.1:5000
```

Luồng khi nhập mã, ví dụ `FPT`:

1. Web nhận mã cổ phiếu từ form.
2. Chuẩn hóa mã thành chữ hoa.
3. `prediction_service.py` đọc dữ liệu sạch.
4. Lọc dữ liệu của mã đó.
5. Tính lại feature mới nhất bằng `build_features`.
6. Lấy dòng feature có ngày mới nhất.
7. Load `models/final_model.pkl`.
8. Load `models/model_metadata.json`.
9. Lấy đúng thứ tự feature từ metadata.
10. Gọi `model.predict`.
11. Nếu model hỗ trợ thì gọi `predict_proba` để lấy xác suất lớp `UP`.
12. Ghi lịch sử dự báo vào SQLite.
13. Render kết quả trên HTML.

Web không train lại model. Web cũng không realtime. Nó dùng dữ liệu offline đã xử lý.

## 23. Xác suất lớp UP nghĩa là gì?

Nếu web hiện:

```text
Xác suất lớp UP: 51.2%
```

Không nên gọi đây là "độ tin cậy tuyệt đối".

Nên gọi là:

```text
Xác suất dự báo của model cho lớp UP.
```

Với Random Forest, xác suất này đến từ tổng hợp dự báo/xác suất của nhiều cây trong rừng.

## 24. Các lệnh chạy thường dùng

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Cập nhật dữ liệu raw bằng `vnstock`:

```powershell
python scripts/fetch_hose_data.py
```

Chạy toàn bộ pipeline:

```powershell
python scripts/run_pipeline.py
```

Chạy từng bước:

```powershell
python scripts/preprocess_data.py
python scripts/build_features.py
python scripts/train_tune_models.py
python scripts/evaluate_models.py
python scripts/select_final_model.py
python database/init_db.py
```

Dự báo bằng CLI:

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol FPT --log-db
```

Chạy web:

```powershell
python app.py
```

Sinh Word report học thuật:

```powershell
python scripts/build_academic_report.py
```

## 25. Output quan trọng nên biết

| File | Vai trò |
|---|---|
| `data/processed/hose_stock_clean.csv` | Dữ liệu đã làm sạch |
| `data/processed/hose_stock_features.csv` | Dữ liệu có 15 feature kỹ thuật |
| `data/processed/ml_dataset.csv` | Feature + target dùng cho train/test |
| `models/dummy.pkl` | Dummy baseline |
| `models/logistic_regression_tuned.pkl` | Logistic Regression sau tuning |
| `models/random_forest_tuned.pkl` | Random Forest sau tuning |
| `models/gradient_boosting_tuned.pkl` | Gradient Boosting sau tuning |
| `models/final_model.pkl` | Model cuối đang dùng |
| `models/model_metadata.json` | Metadata model, feature order, metric, best params |
| `reports/model_comparison.csv` | Bảng so sánh model |
| `reports/final_model_evaluation.csv` | Dòng kết quả của final model |
| `reports/classification_report.csv` | Precision/Recall/F1 theo từng lớp |
| `reports/confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV |
| `reports/confusion_matrix.png` | Ma trận nhầm lẫn dạng hình |
| `reports/feature_importance.csv` | Độ quan trọng feature |
| `reports/best_params.json` | Bộ hyperparameter tốt nhất |
| `reports/train_test_summary.csv` | Tóm tắt train/test split |
| `reports/pipeline_summary.json` | Tóm tắt toàn bộ pipeline |
| `database/stock_prediction.db` | SQLite sync dữ liệu/report và log prediction |
| `docs/bao_cao_project_hose_stock_prediction.docx` | Báo cáo Word được generate từ script |

## 26. Từ điển thuật ngữ nhanh

| Thuật ngữ | Nghĩa dễ hiểu |
|---|---|
| HOSE | Sở Giao dịch Chứng khoán TP.HCM |
| Phiên giao dịch | Một ngày thị trường mở cửa giao dịch |
| OHLCV | Open, High, Low, Close, Volume |
| Machine Learning | Cho máy học quy luật từ dữ liệu quá khứ |
| Classification | Bài toán phân loại; ở đây là `UP` hoặc `NOT_UP` |
| Binary classification | Phân loại 2 lớp |
| Feature | Cột đầu vào cho model |
| Target / Label | Đáp án model cần học |
| Horizon | Khoảng thời gian muốn dự báo trong tương lai |
| Threshold | Ngưỡng để quyết định nhãn; ở đây là 1% |
| Train set | Dữ liệu dùng để model học |
| Test set | Dữ liệu để kiểm tra cuối cùng |
| Cross Validation | Chia train thành nhiều fold để kiểm tra ổn định |
| TimeSeriesSplit | Cross Validation giữ thứ tự thời gian |
| Gap | Khoảng bỏ trống giữa train fold và validation fold |
| Hyperparameter | Tham số cấu hình trước khi train |
| Tuning | Thử nhiều hyperparameter để tìm bộ tốt |
| Baseline | Mốc so sánh tối thiểu |
| Dummy Classifier | Model rất đơn giản, dùng làm baseline |
| Logistic Regression | Model phân loại tuyến tính |
| Random Forest | Nhiều cây quyết định cùng bỏ phiếu |
| Gradient Boosting | Nhiều cây học tuần tự, cây sau sửa lỗi cây trước |
| Accuracy | Tỷ lệ dự báo đúng tổng thể |
| Precision | Trong những lần đoán một lớp, bao nhiêu lần đúng |
| Recall | Trong các mẫu thật sự thuộc một lớp, model bắt được bao nhiêu |
| F1 | Chỉ số cân bằng Precision và Recall |
| Confusion Matrix | Bảng đếm đúng/sai theo từng lớp |
| Feature Importance | Mức đóng góp tương đối của feature trong model |
| Data Leakage | Lỗi model nhìn thấy thông tin tương lai/đáp án |
| Overfitting | Model học quá sát train, nhưng dùng tương lai thì kém |
| Artifact | File sinh ra sau pipeline, ví dụ model/report |
| Metadata | Thông tin mô tả model, feature order, metric |
| Flask | Framework web Python dùng cho demo |
| SQLite | Database nhẹ, lưu trong một file |
| CLI | Command Line Interface, chạy bằng terminal |

## 27. Những câu nên nói khi bảo vệ

Nếu hỏi project làm gì:

```text
Project của em xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning. Bài toán được định nghĩa là phân loại nhị phân: dự báo một mã cổ phiếu có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không.
```

Nếu hỏi dữ liệu đầu vào:

```text
Em sử dụng dữ liệu OHLCV theo ngày, gồm mã cổ phiếu, ngày giao dịch, giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng giao dịch.
```

Nếu hỏi cách tạo nhãn:

```text
Với mỗi mã tại ngày t, em lấy giá đóng cửa sau 5 phiên, tính future_return_5d = close(t+5) / close(t) - 1. Nếu return lớn hơn 1% thì nhãn là UP, ngược lại là NOT_UP.
```

Nếu hỏi vì sao không random split:

```text
Dữ liệu cổ phiếu là dữ liệu chuỗi thời gian. Nếu random split thì dữ liệu tương lai có thể lọt vào train, gây data leakage. Vì vậy em chia train/test theo thời gian, cụ thể là theo label_end_date.
```

Nếu hỏi vì sao chọn Random Forest:

```text
Final model được chọn theo F1 của lớp UP trên test set. Random Forest có F1_UP cao nhất trong các model chính và Recall_UP cũng cao nhất, nên được chọn làm model cuối.
```

Nếu hỏi vì sao không chọn model accuracy cao nhất:

```text
Accuracy dễ bị lệch vì lớp NOT_UP nhiều hơn UP. Gradient Boosting và Dummy có accuracy cao nhưng bắt lớp UP kém. Project ưu tiên F1_UP vì mục tiêu là nhận diện trường hợp tăng hơn 1%.
```

Nếu hỏi có dùng database không:

```text
Có. Project dùng SQLite để sync raw data, clean data, features, tuning results, model evaluations và lưu lịch sử predictions. Tuy nhiên source chính để xem kết quả model hiện tại là các file report và model_metadata.json.
```

Nếu hỏi có phải khuyến nghị đầu tư không:

```text
Không. Đây là hệ thống demo học thuật để dự báo xu hướng theo dữ liệu lịch sử OHLCV. Kết quả chỉ mang tính tham khảo, không phải khuyến nghị mua bán.
```

## 28. Những điều không nên nói sai

Không nói:

```text
Project dự báo chính xác giá cổ phiếu.
```

Nên nói:

```text
Project dự báo xu hướng UP / NOT_UP theo ngưỡng 1% trong 5 phiên giao dịch.
```

Không nói:

```text
Xác suất model là độ tin cậy tuyệt đối.
```

Nên nói:

```text
Đó là xác suất dự báo của model cho lớp UP.
```

Không nói:

```text
5 phiên giao dịch là đúng 5 ngày.
```

Nên nói:

```text
5 phiên giao dịch là 5 ngày thị trường mở cửa; có thể dài hơn 5 ngày lịch nếu có cuối tuần/ngày nghỉ.
```

Không nói:

```text
Web chạy realtime.
```

Nên nói:

```text
Web demo dùng dữ liệu offline đã xử lý và model đã train sẵn.
```

Không nói:

```text
Feature importance chứng minh thị trường bị feature đó gây ra.
```

Nên nói:

```text
Feature importance chỉ cho biết model đã dựa vào feature đó nhiều, không chứng minh quan hệ nhân quả.
```

## 29. Một vài điểm cần biết trong code hiện tại

1. `docs/SO_DO_KIEN_TRUC_HE_THONG.md` mô tả kiến trúc đúng hướng, nhưng một vài dòng số liệu ngày test có thể cũ hơn `pipeline_summary.json`. Khi cần số mới nhất, ưu tiên `reports/pipeline_summary.json`, `reports/model_comparison.csv`, `reports/train_test_summary.csv`, `models/model_metadata.json`.

2. Trong `templates/index.html`, phần `details` kỹ thuật có dùng `result.selected_report_model` và `result.model_match`, nhưng `services/prediction_service.py` hiện chưa trả hai field này. Phần dự báo chính vẫn hoạt động dựa trên `final_model.pkl` và `model_metadata.json`.

3. `scripts/finetune_model.py` chỉ là thông báo deprecated. Muốn tuning thì dùng `scripts/train_tune_models.py` hoặc chạy full pipeline.

4. SQLite có bảng `tuning_results`, nhưng best params chi tiết nên xem trong `reports/best_params.json`.

5. Dữ liệu raw nằm ngoài repo ở `shared_dataset`, còn output đã xử lý nằm trong `data/processed/`.

## 30. Thứ tự học project cho dễ

Nếu bạn muốn học lại project này từ đầu, nên đi theo thứ tự:

1. Hiểu câu hỏi chính: có tăng hơn 1% sau 5 phiên không.
2. Hiểu OHLCV: open, high, low, close, volume.
3. Hiểu `future_return_5d` và cách tạo `target`.
4. Học 15 feature, đặc biệt return, SMA, RSI, volatility, volume ratio.
5. Hiểu vì sao phải chia train/test theo thời gian.
6. Hiểu data leakage là gì.
7. Hiểu Cross Validation và `TimeSeriesSplit(gap=5)`.
8. Hiểu Precision, Recall, F1, Confusion Matrix.
9. Đọc `reports/model_comparison.csv` để hiểu vì sao chọn Random Forest.
10. Đọc `services/feature_engineering.py` để hiểu feature/label.
11. Đọc `services/model_tuning.py` để hiểu train/tune model.
12. Đọc `services/prediction_service.py` và `app.py` để hiểu web/CLI dự báo.

Chỉ cần nắm chắc các ý trên, bạn đã hiểu lõi project đủ để đọc code, chạy demo và giải thích khi bảo vệ.
