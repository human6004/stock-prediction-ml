# Giải thích project dự báo xu hướng cổ phiếu HOSE cho người mới

Tài liệu này được viết lại sau khi đọc lại code, report, model metadata, web demo, database schema và roadmap hiện tại.

- Project: `D:\study\niên luận\stock-prediction-ml`
- Roadmap: `D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md`
- Ngày đọc lại: `2026-07-08`
- Mục tiêu của tài liệu: giải thích từ đầu, coi bạn là người mới chưa biết project, chưa quen thuật ngữ chứng khoán và Machine Learning.

> Cập nhật quan trọng ở lần đọc này: pipeline hiện lấy hyperparameter từ **Tuning Lab thủ công** (`experiments/manual_config.json`) thay cho `RandomizedSearchCV` tự động; mỗi model có **decision threshold** riêng được tinh chỉnh để tối ưu F1 lớp UP; và cách xử lý mất cân bằng lớp khác nhau theo model.

> ⚠️ **Lưu ý trạng thái dữ liệu (đọc kỹ trước khi trích số):** Code và dữ liệu đã xử lý hiện tại dùng **20 feature** (mới thêm 5 feature: `return_10d`, `return_20d`, `dist_high20`, `dist_low20`, `month`). Nhưng bộ **model/report đang nằm trên đĩa vẫn là run cũ ngày `2026-07-05` với 15 feature** (fingerprint `4c2259e97b3f`), vì pipeline chính thức **chưa chạy lại** sau khi thêm feature. Cụ thể: `ml_dataset.csv` đã được dựng lại ngày `2026-07-08` với đủ 20 cột; `experiments/manual_config.json` (fingerprint mới `7e88c028946b`) đã tune lại và chốt params mới cho **Logistic Regression** và **Random Forest**, nhưng **chưa chốt Gradient Boosting** → pipeline chưa đủ điều kiện chạy. Vì vậy mọi con số kết quả bên dưới (F1_UP, confusion matrix, feature importance, best params trong `reports/`) **thuộc run cũ 15 feature** và sẽ thay đổi khi bạn chốt đủ 3 model rồi chạy lại pipeline. Các mục có số cũ đều được đánh dấu **[RUN CŨ 15-FEATURE]**.

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
- Có tạo 20 feature kỹ thuật.
- Có tạo nhãn `UP / NOT_UP` với horizon 5 phiên và threshold 1%.
- Có chia train/test theo thời gian bằng `label_end_date`.
- Có `TimeSeriesSplit(gap=5)` cho Cross Validation.
- Có tuning Logistic Regression, Random Forest, Gradient Boosting qua Tuning Lab thủ công (`/tuning`).
- Có Dummy Classifier làm baseline.
- Có tinh chỉnh decision threshold riêng cho từng model để tối ưu F1 lớp UP.
- Có chọn final model theo `F1_UP` (tie-break bằng `Recall_UP` rồi độ đơn giản).
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
| `services/tuning_lab.py` | Backend Tuning Lab: validate tham số, chạy CV một bộ config, ghi lịch sử |
| `services/experiment_state.py` | Quản lý lịch sử tuning, cấu hình đã chốt, khóa (lock), fingerprint dataset |
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
| Raw data | 549,084 dòng |
| Số mã trong raw data | 400 mã |
| Khoảng ngày raw data | 2019-08-14 đến 2026-06-26 |
| Clean data | 548,858 dòng |
| Mã đủ điều kiện train | 396 mã |
| Mã bị loại | 4 mã |
| Feature data | 511,191 dòng |
| ML dataset sau khi có label | 509,211 dòng |

Tỷ lệ nhãn tính trên train/test (xem mục 13): train `UP` ≈ 38.7%, test `UP` ≈ 30.7%. Sự chênh này (test có ít UP hơn train) là một điểm đáng lưu ý — thị trường giai đoạn test tăng ít hơn giai đoạn train.

4 mã bị loại vì chưa đủ 250 phiên giao dịch (theo `reports/excluded_symbols.csv`):

| Mã | Số dòng | Khoảng dữ liệu | Lý do |
|---|---:|---|---|
| `CRV` | 140 | 2025-10-10 đến 2026-06-26 | `fewer_than_250_trading_days` |
| `TCX` | 164 | 2025-10-21 đến 2026-06-19 | `fewer_than_250_trading_days` |
| `VCK` | 124 | 2025-12-16 đến 2026-06-19 | `fewer_than_250_trading_days` |
| `VPX` | 127 | 2025-12-11 đến 2026-06-19 | `fewer_than_250_trading_days` |

`reports/fetch_report.json` (lần fetch `2026-06-28`) cho biết raw data hiện có 549,084 dòng trên 400 mã, khoảng ngày đến 2026-06-26. Lần đó `new_rows = 0` (không thêm dòng mới) và có 35 mã fetch lỗi (`RetryError`), nhưng các mã này vẫn giữ dữ liệu cũ trong CSV nếu trước đó đã tồn tại.

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

Project hiện dùng **20 feature** (trước đây là 15; 5 feature cuối bảng — `return_10d`, `return_20d`, `dist_high20`, `dist_low20`, `month` — là phần mới thêm trong lần cải tiến này, xem `config/settings.py::FEATURE_COLUMNS`):

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
| `return_10d` *(mới)* | Giá thay đổi bao nhiêu trong 10 phiên gần nhất — động lượng trung hạn |
| `return_20d` *(mới)* | Giá thay đổi bao nhiêu trong 20 phiên gần nhất — động lượng dài hơn |
| `dist_high20` *(mới)* | Giá hiện tại cách đỉnh cao nhất 20 phiên bao nhiêu: `(close / max_high_20) - 1` (thường ≤ 0) |
| `dist_low20` *(mới)* | Giá hiện tại cách đáy thấp nhất 20 phiên bao nhiêu: `(close / min_low_20) - 1` (thường ≥ 0) |
| `month` *(mới)* | Tháng của phiên giao dịch (1–12) — nắm bắt yếu tố mùa vụ theo tháng |

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

### 5 feature mới (bổ sung trong bản cải tiến)

- `return_10d`, `return_20d`: giống `return_5d` nhưng nhìn xa hơn — giá thay đổi bao
  nhiêu trong 10 và 20 phiên gần nhất. Cho model thêm góc nhìn xu hướng trung hạn.
- `dist_high20 = close / max(high, 20 phiên) - 1`: giá hiện tại đang cách đỉnh cao nhất
  20 phiên bao nhiêu. Bằng 0 nghĩa là đang ở đỉnh; âm nhiều nghĩa là đã rơi khá xa khỏi đỉnh.
- `dist_low20 = close / min(low, 20 phiên) - 1`: giá hiện tại cách đáy thấp nhất 20 phiên
  bao nhiêu. Bằng 0 nghĩa là đang ở đáy; dương nhiều nghĩa là đã bật lên khá xa khỏi đáy.
- `month`: tháng của phiên giao dịch (1–12). Feature lịch, để model có thể bắt yếu tố mùa vụ.

Hai feature `dist_high20` / `dist_low20` mô tả vị trí giá trong biên độ 20 phiên — thông
tin mà `close_vs_sma20` (so với trung bình) chưa nắm được.

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
| Test | 42,238 | 394 | 2025-10-30 đến 2026-06-19 | `label_end_date > 2025-12-31` | 12,977 | 29,261 |

Suy ra tỷ lệ `UP`: train ≈ 38.7% (180584/466973), test ≈ 30.7% (12977/42238). Test ít UP hơn train — mô hình phải dự báo trong giai đoạn thị trường tăng yếu hơn lúc học.

Theo `reports/pipeline_summary.json`, `overlap_rows = 0` (train và test không dính nhau), `train_label_end_max = 2025-12-31`, `test_label_end_min = 2026-01-05` — ranh giới nhãn tách sạch.

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

Điểm mới quan trọng: trong mỗi fold, code còn **tinh chỉnh decision threshold** (ngưỡng xác suất để quyết UP) ngay trên phần TRAIN của fold, rồi áp ngưỡng đó lên phần VAL của fold để đo F1. Nhờ vậy F1 đo được phản ánh đúng ngưỡng sẽ dùng khi phục vụ, và việc chọn ngưỡng không nhìn vào dữ liệu validation (không leakage). Chi tiết ở mục 15.

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

Cách tuning hiện tại (điểm thay đổi so với bản trước): project **không còn chạy `RandomizedSearchCV` tự động** trong pipeline. Thay vào đó dùng **Tuning Lab thủ công** trên web:

1. Vào trang `/tuning`, người dùng nhập bộ hyperparameter cho từng model (LR, RF, GB).
2. Bấm chạy → `services/tuning_lab.py::evaluate_single_config` chạy Cross Validation `TimeSeriesSplit(n_splits=5, gap=5)` trên TRAIN với đúng bộ tham số đó, trả về F1_UP trung bình các fold, và ghi một dòng vào `experiments/tuning_history.csv` (lịch sử thử nghiệm).
3. Khi thấy bộ nào tốt, bấm "dùng cấu hình này" → lưu vào `experiments/manual_config.json`, kèm dấu vân dữ liệu (`dataset_fingerprint`).
4. Khi đã chốt đủ cả 3 model (và fingerprint khớp dataset hiện tại), pipeline chính thức mới được phép chạy. Lúc chạy, `services/model_tuning.py::tune_models` **đọc lại các bộ tham số đã chốt** từ `manual_config.json` để huấn luyện, chứ không tự dò tham số.

`config/settings.py` vẫn còn `TUNING_N_ITER=12` và `TUNING_SCORING="f1"` như di sản cấu hình, nhưng luồng chốt tham số hiện do người dùng điều khiển qua Tuning Lab.

Ý nghĩa: cách này minh bạch hơn cho một đồ án — mỗi con số F1 trong `tuning_history.csv` gắn với một bộ tham số cụ thể và một phiên bản dataset cụ thể, dễ trình bày và tái lập.

## 16. Các model trong project

Project train 4 model:

| Model | Vai trò |
|---|---|
| Dummy Classifier | Baseline tối thiểu, gần như chỉ đoán lớp phổ biến nhất |
| Logistic Regression | Model tuyến tính đơn giản, dễ giải thích |
| Random Forest | Nhiều cây quyết định cùng bỏ phiếu, hợp dữ liệu bảng |
| Gradient Boosting | Nhiều cây học tuần tự, cây sau cố sửa lỗi cây trước |

Chi tiết trong code (`services/model_tuning.py`):

- Dummy dùng `strategy="most_frequent"` (threshold cố định 0.5).
- Logistic Regression nằm trong `Pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))`.
- Random Forest dùng `class_weight="balanced_subsample"`, `n_jobs=-1`.
- Gradient Boosting dùng `GradientBoostingClassifier`. `GradientBoostingClassifier` **không có** tham số `class_weight`, nên code truyền `sample_weight = compute_sample_weight("balanced", y)` vào lúc `.fit()` (cả trong CV lẫn khi fit cuối) để cân bằng lớp tương đương.

Điểm cần hiểu — **xử lý mất cân bằng lớp khác nhau theo model**: lớp UP ít hơn NOT_UP, nên mỗi model đều được cân bằng, nhưng bằng cơ chế khác nhau (LR/RF qua `class_weight`, GB qua `sample_weight`). Nếu không cân bằng, model dễ đoán toàn NOT_UP.

Bộ tham số hiện có **hai phiên bản** vì code vừa được nâng cấp lên 20 feature (xem hộp cảnh báo đầu tài liệu):

**(A) Params của run cũ đã sinh ra model đang phục vụ** — theo `reports/best_params.json` (fingerprint cũ `4c2259e97b3f`, 15 feature, chạy 2026-07-05). Đây là params đứng sau `final_model.pkl` và mọi số liệu ở mục 18–20:

| Model | Params (run cũ 15 feature) |
|---|---|
| Logistic Regression | `C=2.11e-06`, `solver=liblinear` |
| Random Forest | `n_estimators=80`, `max_depth=8`, `min_samples_leaf=100`, `max_features=0.4` |
| Gradient Boosting | `n_estimators=40`, `learning_rate=0.2`, `max_depth=2`, `subsample=0.6` |

**(B) Params vừa chốt lại trên dataset 20 feature** — theo `experiments/manual_config.json` (fingerprint mới `7e88c028946b`, chốt 2026-07-08). **Chưa chạy pipeline chính thức nên chưa có model/report mới**:

| Model | Params (đã chốt, 20 feature) | Trạng thái |
|---|---|---|
| Logistic Regression | `C=1.72e-04`, `solver=liblinear` | Đã chốt |
| Random Forest | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=40`, `max_features=0.4` | Đã chốt |
| Gradient Boosting | (chưa chốt) | **Còn thiếu** |

Vì Gradient Boosting chưa được chốt trong Tuning Lab nên điều kiện "đủ 3 model" chưa thỏa — pipeline chính thức chưa thể chạy lại. Khi chốt đủ GB và chạy `run_pipeline.py`, toàn bộ số liệu mục 18–20 sẽ được sinh lại theo 20 feature.

## 16b. Decision threshold (ngưỡng quyết định) là gì?

Mặc định, một model phân loại đoán UP khi `P(UP) >= 0.5`. Nhưng với dữ liệu mất cân bằng và mục tiêu ưu tiên bắt lớp UP, mốc 0.5 thường bỏ sót nhiều UP. Project vì vậy **tinh chỉnh ngưỡng riêng cho từng model**:

1. Trong mỗi fold CV, sau khi fit trên phần TRAIN của fold, code quét `precision_recall_curve` để tìm ngưỡng cho **F1 lớp UP cao nhất** (`services/model_tuning.py::tune_threshold`), giới hạn trong khoảng `[0.05, 0.95]`.
2. Ngưỡng đó được chọn **chỉ dựa trên TRAIN của fold**, rồi áp lên VAL của fold để đo — không nhìn trộm VAL, nên không leakage.
3. Khi fit model cuối trên toàn bộ TRAIN, ngưỡng tối ưu được tính lại và **lưu vào artifact + `model_metadata.json`** (`decision_threshold`).
4. Khi dự báo thật, model so `P(UP)` với ngưỡng đã lưu này, không dùng 0.5.

Ngưỡng của final model Random Forest đang phục vụ (run cũ 15 feature) là `decision_threshold ≈ 0.4049` (thấp hơn 0.5 → dễ gán UP hơn, giúp Recall_UP cao). Cơ chế tune ngưỡng vẫn giữ nguyên khi chạy lại pipeline 20 feature, chỉ giá trị ngưỡng sẽ được tính lại.

Hệ quả cần lưu ý: hạ ngưỡng làm **Recall_UP tăng mạnh** (bắt được nhiều UP) nhưng **Precision_UP và Accuracy giảm** (báo UP nhầm nhiều hơn). Đây là đánh đổi có chủ đích vì đề tài ưu tiên không bỏ sót cơ hội tăng.

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

## 18. Kết quả model (run cũ 15 feature — 2026-07-05)

> ⚠️ **Số liệu mục này thuộc run pipeline cũ ngày 2026-07-05, khi project mới có 15 feature.** Đó là kết quả đang nằm trong `reports/model_comparison.csv` và là model đang được `final_model.pkl` phục vụ. Sau khi bạn thêm 5 feature (nay 20 feature), pipeline **chưa chạy lại**, nên các con số dưới đây sẽ thay đổi khi chạy lại. Giữ lại để tham chiếu, không phải kết quả cuối của bản 20 feature.

Theo `reports/model_comparison.csv` (run 2026-07-05):

| Model | Accuracy | Precision_UP | Recall_UP | F1_UP | CV F1_UP | Chọn |
|---|---:|---:|---:|---:|---:|---|
| Dummy Classifier | 0.6928 | 0.0000 | 0.0000 | 0.0000 | - | Không |
| Logistic Regression | 0.3364 | 0.3115 | 0.9582 | 0.4701 | 0.5419 | Không |
| Random Forest | 0.4013 | 0.3281 | 0.9053 | 0.4816 | 0.5427 | Có |
| Gradient Boosting | 0.3850 | 0.3240 | 0.9219 | 0.4795 | 0.5441 | Không |

Final model hiện tại:

```text
models/final_model.pkl = Random Forest
```

Nhận xét chung về bảng này (khác hẳn bản trước, do đã áp decision threshold): cả 3 model chính giờ có **Recall_UP rất cao (0.90–0.96)** nhưng **Accuracy thấp (0.34–0.40)** và Precision_UP quanh 0.32. Đây là hệ quả trực tiếp của việc hạ ngưỡng để ưu tiên bắt lớp UP: model gán UP rất "hào phóng" → bắt gần hết UP thật nhưng cũng báo UP nhầm nhiều.

Vì sao Dummy không chọn?

Dummy đoán lớp phổ biến nhất là `NOT_UP`. Accuracy 0.6928 cao nhất bảng nhưng `F1_UP = 0` — không bắt được UP nào. `select_final_model` loại bỏ Dummy khỏi vòng chọn.

Vì sao Random Forest được chọn?

Quy tắc chọn (`services/model_evaluation.py::select_final_model`) xếp hạng theo thứ tự: **F1_UP giảm dần → Recall_UP giảm dần → độ đơn giản tăng dần** (LR đơn giản hơn RF, RF đơn giản hơn GB). Ba model chính rất sát nhau về F1_UP (RF 0.4816, GB 0.4795, LR 0.4701), và Random Forest có **F1_UP cao nhất** nên thắng.

Điểm cần trung thực khi trình bày: khoảng cách F1_UP giữa 3 model rất nhỏ (chênh ~0.01), nên "Random Forest tốt nhất" chỉ đúng ở mức sát sao trên tập test này, không phải vượt trội.

## 19. Confusion matrix (run cũ 15 feature — 2026-07-05)

> ⚠️ Cùng cảnh báo như mục 18: ma trận dưới đây là của run 15-feature ngày 2026-07-05, sẽ đổi khi chạy lại pipeline 20 feature.

Theo `reports/confusion_matrix.csv` (run 2026-07-05):

| Thực tế / Dự báo | Dự báo NOT_UP | Dự báo UP |
|---|---:|---:|
| Thực tế NOT_UP | 5,201 | 24,060 |
| Thực tế UP | 1,229 | 11,748 |

Cách đọc:

- 5,201: thực tế `NOT_UP`, model đoán `NOT_UP`, đúng.
- 24,060: thực tế `NOT_UP`, model đoán `UP`, sai (false positive — rất nhiều).
- 1,229: thực tế `UP`, model đoán `NOT_UP`, sai (false negative — rất ít).
- 11,748: thực tế `UP`, model đoán `UP`, đúng.

Tổng test:

```text
5,201 + 24,060 + 1,229 + 11,748 = 42,238 dòng
```

Cách đọc ma trận này rất rõ chiến lược của model: cột "Dự báo UP" chiếm đa số. Model bắt được 11,748/12,977 UP thật (Recall_UP ≈ 0.905 — rất cao) nhưng đổi lại báo UP nhầm 24,060 lần trên nền NOT_UP. Đây đúng là cái giá của decision threshold thấp: **ưu tiên không bỏ sót UP, chấp nhận nhiều cảnh báo sai**.

Kết quả này không phải "siêu chính xác". Nó phản ánh bài toán dự báo cổ phiếu bằng OHLCV là khó, cộng thêm chênh phân phối train/test (train UP 38.7% vs test UP 30.7%). Điểm tốt của project là pipeline đúng, có kiểm soát leakage, có baseline, có tinh chỉnh ngưỡng trung thực và đánh giá công bằng.

## 20. Feature importance hiện tại

Vì final model là Random Forest, project xuất được `reports/feature_importance.csv`.

> ⚠️ **Bảng dưới đây là của run cũ 2026-07-05, chỉ có 15 feature** (chưa có 5 feature mới `return_10d`, `return_20d`, `dist_high20`, `dist_low20`, `month`). Sau khi chạy lại pipeline trên 20 feature, thứ hạng importance sẽ thay đổi và có thêm dòng cho các feature mới.

Top feature quan trọng (run cũ 15-feature):

| Hạng | Feature | Importance |
|---:|---|---:|
| 1 | `volatility_20d` | 0.3136 |
| 2 | `return_1d` | 0.0909 |
| 3 | `close_vs_sma20` | 0.0879 |
| 4 | `volatility_5d` | 0.0842 |
| 5 | `return_3d` | 0.0748 |
| 6 | `volume_ratio_20` | 0.0694 |
| 7 | `sma20_vs_sma50` | 0.0535 |
| 8 | `volume_change_1d` | 0.0395 |

Cách hiểu: Random Forest đang dùng nhiều thông tin về biến động 20 phiên (chiếm ~31% một mình), return gần nhất, vị trí giá so với SMA20, biến động 5 phiên, return 3 phiên và volume. Các SMA thô (`sma5`, `sma20`, `sma50`) đóng góp thấp nhất — hợp lý vì các feature dạng "so sánh/tỷ lệ" mang thông tin hơn giá trị tuyệt đối.

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
| Raw rows | 549,084 |
| Clean rows | 548,858 |
| Feature rows | 511,191 |
| Tuning rows | 3 |
| Evaluation rows | 4 |

Lưu ý cơ chế sync (`services/database_service.py`): `raw_prices`, `clean_prices`, `features` được ghi kiểu **replace** (thay toàn bộ mỗi lần), còn `tuning_results` và `model_evaluations` ghi kiểu **append** (nối thêm) — nên số dòng hai bảng này tăng dần qua nhiều lần chạy. Best params chi tiết nên xem ở `reports/best_params.json` và `models/model_metadata.json` thay vì trong bảng SQLite.

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
10. Gọi `predict_proba` để lấy xác suất lớp `UP` (`P(UP)`).
11. So `P(UP)` với `decision_threshold` lấy từ metadata (≈0.4049): nếu `P(UP) >= ngưỡng` thì gán `UP`, ngược lại `NOT_UP`. Không dùng mốc 0.5 mặc định.
12. Ghi lịch sử dự báo vào SQLite (best-effort, lỗi được bỏ qua để không hỏng kết quả).
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

Chốt cấu hình tuning: mở web (`python app.py`) → vào trang `/tuning` → thử tham số từng model → bấm "dùng cấu hình này" cho đủ 3 model, rồi chạy pipeline từ chính trang đó hoặc bằng `python scripts/run_pipeline.py`.

## 25. Output quan trọng nên biết

| File | Vai trò |
|---|---|
| `data/processed/hose_stock_clean.csv` | Dữ liệu đã làm sạch |
| `data/processed/hose_stock_features.csv` | Dữ liệu có 20 feature kỹ thuật |
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
| `experiments/tuning_history.csv` | Lịch sử các lần thử tuning thủ công qua Tuning Lab |
| `experiments/manual_config.json` | Bộ tham số đã chốt cho 3 model, gắn fingerprint dataset |
| `docs/bao_cao_project_hose_stock_prediction.docx` | Báo cáo Word (bản đã tạo sẵn trong docs) |

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
| Decision threshold | Ngưỡng xác suất để quyết UP; ở đây tinh chỉnh riêng mỗi model thay cho 0.5 |
| Class imbalance | Mất cân bằng lớp: số mẫu UP ít hơn NOT_UP |
| class_weight / sample_weight | Cách tăng trọng số lớp thiểu số khi train (LR/RF dùng class_weight, GB dùng sample_weight) |
| Fingerprint | Chuỗi băm đại diện phiên bản dataset, dùng khớp cấu hình đã chốt |
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
Final model được chọn theo quy tắc: xếp hạng F1 của lớp UP trên test, hòa thì xét Recall_UP, rồi đến độ đơn giản của model. Ba model chính rất sát nhau, Random Forest có F1_UP cao nhất (0.4816) nên được chọn. Em cũng nói rõ khoảng cách với hai model kia rất nhỏ, khoảng 0.01.
```

Nếu hỏi vì sao không chọn model accuracy cao nhất:

```text
Accuracy dễ bị lệch vì lớp NOT_UP nhiều hơn UP. Dummy đoán toàn NOT_UP nên accuracy cao nhất nhưng F1_UP bằng 0. Project ưu tiên F1_UP vì mục tiêu là nhận diện trường hợp tăng hơn 1%.
```

Nếu hỏi vì sao Accuracy của các model chính lại thấp (khoảng 0.34–0.40):

```text
Vì em tinh chỉnh decision threshold để ưu tiên bắt lớp UP. Ngưỡng hạ xuống dưới 0.5 làm Recall_UP rất cao (khoảng 0.90) nhưng đổi lại model báo UP nhầm nhiều nên Accuracy và Precision giảm. Đây là đánh đổi có chủ đích theo mục tiêu đề tài.
```

Nếu hỏi tuning làm thế nào:

```text
Em dùng Tuning Lab trên web để thử từng bộ hyperparameter cho 3 model. Mỗi lần thử chạy TimeSeriesSplit CV trên tập train và ghi lại F1_UP vào lịch sử. Khi chốt đủ 3 model, pipeline chính thức đọc lại các bộ tham số đã chốt trong manual_config.json để huấn luyện, thay vì tự dò tham số.
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

3. `scripts/finetune_model.py` chỉ là thông báo deprecated. Việc chốt tham số nay làm qua Tuning Lab (`/tuning`); pipeline chính thức đọc `experiments/manual_config.json` để train.

4. SQLite có bảng `tuning_results`, nhưng best params chi tiết nên xem trong `reports/best_params.json`.

5. Dữ liệu raw nằm ngoài repo ở `shared_dataset`, còn output đã xử lý nằm trong `data/processed/`.

6. Điểm dễ gây hiểu nhầm khi đọc metric: Accuracy các model chính thấp (~0.34–0.40) **không phải model kém**, mà do decision threshold hạ thấp để ưu tiên Recall_UP. Luôn đọc Accuracy cùng với Recall_UP và ngưỡng `decision_threshold` trong `model_metadata.json`.

7. Pipeline có cơ chế chống chạy trùng và chống dùng lại tập TEST: `experiments/pipeline.lock` (khóa tiến trình), `experiments/test_evaluation_lock.json` (khóa TEST theo fingerprint). Muốn cố tình đánh giá lại cùng dataset thì đặt biến môi trường `STOCK_ALLOW_TEST_REEVAL=1`.

8. **Trạng thái quan trọng nhất hiện nay — code đã đi trước report một bước.** Code và `data/processed/ml_dataset.csv` đã cập nhật lên **20 feature** (thêm `return_10d`, `return_20d`, `dist_high20`, `dist_low20`, `month`), và `experiments/manual_config.json` (fingerprint mới `7e88c028946b`) đã tune lại params cho **Logistic Regression** (`C=0.00017`) và **Random Forest** (`n_estimators=100, max_depth=6, min_samples_leaf=40, max_features=0.4`). Nhưng **pipeline chính thức chưa chạy lại**: toàn bộ `models/*.pkl` và `reports/*` vẫn là kết quả run cũ ngày **2026-07-05** (15 feature, fingerprint `4c2259e97b3f`). Vì vậy mọi con số kết quả trong mục 16, 18, 19, 20 là của run cũ 15 feature — coi là tham chiếu, sẽ đổi sau khi chạy lại.

9. **Muốn chạy lại pipeline trên 20 feature thì còn thiếu 1 bước:** `manual_config.json` mới mới chốt 2/3 model (LR, RF), **chưa chốt Gradient Boosting**. Pipeline yêu cầu đủ cả 3 model khớp fingerprint hiện tại mới chạy được. Cần vào `/tuning` chốt nốt GB, rồi chạy `python scripts/run_pipeline.py` (hoặc bấm chạy trong trang tuning) để sinh lại report/model theo 20 feature. Sau khi chạy xong, nhớ cập nhật lại các con số trong tài liệu này.

## 30. Thứ tự học project cho dễ

Nếu bạn muốn học lại project này từ đầu, nên đi theo thứ tự:

1. Hiểu câu hỏi chính: có tăng hơn 1% sau 5 phiên không.
2. Hiểu OHLCV: open, high, low, close, volume.
3. Hiểu `future_return_5d` và cách tạo `target`.
4. Học 20 feature, đặc biệt return, SMA, RSI, volatility, volume ratio, và nhóm mới (return 10/20 phiên, khoảng cách đỉnh/đáy 20 phiên, tháng).
5. Hiểu vì sao phải chia train/test theo thời gian.
6. Hiểu data leakage là gì.
7. Hiểu Cross Validation và `TimeSeriesSplit(gap=5)`.
8. Hiểu Precision, Recall, F1, Confusion Matrix.
9. Đọc `reports/model_comparison.csv` để hiểu vì sao chọn Random Forest.
10. Đọc `services/feature_engineering.py` để hiểu feature/label.
11. Đọc `services/model_tuning.py` để hiểu train/tune model.
12. Đọc `services/prediction_service.py` và `app.py` để hiểu web/CLI dự báo.

Chỉ cần nắm chắc các ý trên, bạn đã hiểu lõi project đủ để đọc code, chạy demo và giải thích khi bảo vệ.
