# Giải thích project dự báo xu hướng cổ phiếu HOSE cho người mới

Tài liệu này được viết lại theo code hiện tại sau khi project đã được cleanup.

- Project: `D:\study\niên luận\v3\niên luận cơ sở ngành-cusor`
- Roadmap: `D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md`
- Ngày cập nhật tài liệu: 2026-06-12

Mục tiêu của tài liệu: giải thích project từ góc nhìn của một người mới, chưa rành Python, Machine Learning, pipeline, feature, model, metric. Bạn có thể lưu file này để học lại hoặc dùng làm nền khi chuẩn bị báo cáo/bảo vệ.

## 1. Project này làm gì?

Project này xây dựng một hệ thống hỗ trợ dự báo xu hướng cổ phiếu trên sàn HOSE bằng Machine Learning.

Câu hỏi chính của project là:

```text
Một mã cổ phiếu có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không?
```

Kết quả dự báo chỉ có 2 loại:

```text
UP     = có khả năng tăng hơn 1% sau 5 phiên giao dịch
NOT_UP = không tăng hơn 1% sau 5 phiên giao dịch
```

Ví dụ:

```text
Người dùng nhập FPT
→ hệ thống lấy dữ liệu gần nhất của FPT
→ tính các chỉ số kỹ thuật
→ nạp model đã train
→ trả về UP hoặc NOT_UP
```

Quan trọng: đây là project học thuật/demo, không phải khuyến nghị mua bán cổ phiếu.

## 2. Roadmap nói project nên có những gì?

Roadmap yêu cầu một pipeline khá đầy đủ:

```text
Thu thập dữ liệu HOSE
→ Lưu database
→ Làm sạch dữ liệu
→ Kiểm tra chất lượng dữ liệu
→ Tạo feature Machine Learning
→ Tạo nhãn UP / NOT_UP
→ Chia train/test theo thời gian
→ Cross Validation theo chuỗi thời gian
→ Hyperparameter tuning
→ Đánh giá model
→ Chọn final_model.pkl
→ Web demo dự báo
```

Code hiện tại đã làm đúng phần lớn luồng này:

- Có dữ liệu raw OHLCV toàn HOSE từ CSV.
- Có script fetch thêm dữ liệu bằng `vnstock`.
- Có bước làm sạch dữ liệu.
- Có bước lọc mã đủ tối thiểu 250 phiên giao dịch.
- Có bước tạo 15 feature.
- Có bước tạo nhãn `UP / NOT_UP`.
- Có bước chia train/test theo `label_end_date`.
- Có tuning bằng `RandomizedSearchCV`.
- Có `TimeSeriesSplit(gap=5)`.
- Có so sánh 4 model.
- Có chọn final model theo `F1_UP`.
- Có lưu model `.pkl`, metadata, report.
- Có SQLite database.
- Có Flask web demo.

## 3. Một câu về toàn bộ luồng project

Bạn có thể nhớ project như một dây chuyền:

```text
raw data
→ clean data
→ feature data
→ ML dataset có đáp án
→ train/test split
→ tune model
→ evaluate model
→ chọn final_model.pkl
→ Flask web dùng model để dự báo
```

Trong code hiện tại, file điều phối dây chuyền này là:

```text
scripts/run_pipeline.py
```

File này không tự làm hết mọi thứ. Nó gọi các hàm trong thư mục `services/`.

## 4. Cấu trúc thư mục dễ hiểu

```text
config/
```

Chứa cấu hình chung, quan trọng nhất là `settings.py`.

```text
services/
```

Chứa logic chính của project: làm sạch dữ liệu, tạo feature, train/tune model, đánh giá model, dự báo, sync database.

```text
scripts/
```

Chứa các file để chạy từng bước hoặc chạy toàn bộ pipeline.

```text
data/processed/
```

Chứa 3 dataset chính sau khi xử lý.

```text
models/
```

Chứa các model đã train và metadata.

```text
reports/
```

Chứa các báo cáo kết quả: so sánh model, confusion matrix, feature importance, train/test summary.

```text
database/
```

Chứa SQLite database và code kết nối database.

```text
templates/ + static/
```

Chứa giao diện web Flask.

```text
app.py
```

Backend Flask để chạy web demo.

## 5. Các file Python chính đang làm gì?

| File | Vai trò dễ hiểu |
|---|---|
| `config/settings.py` | Bảng cấu hình chung: đường dẫn, feature, model, split date, threshold |
| `scripts/run_pipeline.py` | Nút chạy tổng từ dữ liệu raw đến final model/report/database |
| `services/preprocessing.py` | Làm sạch dữ liệu và tạo `hose_stock_clean.csv` |
| `services/feature_engineering.py` | Tạo feature, tạo target, chia train/test |
| `services/model_tuning.py` | Train Dummy và tune 3 model chính |
| `services/model_evaluation.py` | Đánh giá model, chọn final model, ghi report |
| `services/prediction_service.py` | Dự báo một mã cổ phiếu khi người dùng nhập |
| `services/database_service.py` | Sync dữ liệu/report vào SQLite và lưu lịch sử dự báo |
| `scripts/fetch_hose_data.py` | Cập nhật dữ liệu mới bằng `vnstock` |
| `app.py` | Chạy web Flask |

## 6. `settings.py` là gì?

`config/settings.py` là file cấu hình trung tâm.

Bạn có thể hiểu nó như bảng điều khiển của project:

```text
Dữ liệu raw nằm ở đâu?
Dữ liệu sạch lưu ở đâu?
Model lưu ở đâu?
Report lưu ở đâu?
Dự báo mấy phiên?
Ngưỡng UP là bao nhiêu?
Train/test chia ngày nào?
Dùng feature nào?
Tune model bao nhiêu lần?
```

Các thông số quan trọng:

| Thông số | Giá trị | Nghĩa |
|---|---:|---|
| `SPLIT_DATE` | `2025-12-31` | Mốc chia train/test |
| `PREDICTION_HORIZON` | `5` | Dự báo sau 5 phiên giao dịch |
| `UP_THRESHOLD` | `0.01` | Tăng hơn 1% thì tính là `UP` |
| `MIN_TRADING_DAYS` | `250` | Mã có dưới 250 phiên sẽ bị loại khỏi train |
| `MIN_AVERAGE_VOLUME` | `0` | Hiện chưa lọc theo volume trung bình |
| `CV_N_SPLITS` | `5` | Cross Validation có 5 fold |
| `CV_GAP` | `5` | Chừa khoảng cách 5 mẫu để giảm leakage |
| `TUNING_N_ITER` | `12` | Mỗi model tune 12 bộ tham số |
| `TUNING_SCORING` | `f1` | Tune theo F1 của lớp UP |

Nếu đổi `UP_THRESHOLD = 0.02`, bài toán sẽ thành:

```text
UP nếu tăng hơn 2% sau 5 phiên
```

Nếu đổi `PREDICTION_HORIZON = 10`, bài toán sẽ thành:

```text
Dự báo sau 10 phiên giao dịch
```

Đổi các thông số này xong thì phải chạy lại pipeline để tạo label mới, train lại model, đánh giá lại model.

## 7. Dữ liệu OHLCV là gì?

Dữ liệu đầu vào là dữ liệu giao dịch theo ngày.

OHLCV là viết tắt của:

```text
O = Open   = giá mở cửa
H = High   = giá cao nhất
L = Low    = giá thấp nhất
C = Close  = giá đóng cửa
V = Volume = khối lượng giao dịch
```

Các cột bắt buộc:

| Cột | Nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu, ví dụ `FPT`, `VNM`, `SSI` |
| `trading_date` | Ngày giao dịch |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất trong ngày |
| `low` | Giá thấp nhất trong ngày |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch |

Project này chỉ dùng OHLCV. Nó không dùng tin tức, báo cáo tài chính, dữ liệu realtime, sentiment analysis, LSTM hay Transformer.

## 8. Dataset hiện tại gồm những file nào?

Sau cleanup, thư mục `data/processed/` chỉ còn 3 file chính:

| File | Ai tạo ra? | Vai trò |
|---|---|---|
| `hose_stock_clean.csv` | `services/preprocessing.py` | Dữ liệu OHLCV đã làm sạch |
| `hose_stock_features.csv` | `services/feature_engineering.py` | Dữ liệu sạch + 15 feature kỹ thuật |
| `ml_dataset.csv` | `services/feature_engineering.py` | Dữ liệu có feature + target, dùng train/test |

Luồng tạo file:

```text
shared_dataset/hose_stock_raw.csv
→ preprocessing.py
→ data/processed/hose_stock_clean.csv
→ feature_engineering.py
→ data/processed/hose_stock_features.csv
→ feature_engineering.py
→ data/processed/ml_dataset.csv
```

Trước đây project có vài file duplicate/legacy như:

```text
hose_stock_cleaned.csv
hose_stock_ml_dataset.csv
data/processed/hose_stock_raw.csv
```

Bản code hiện tại đã bỏ các file đó khỏi luồng chính.

## 9. Số liệu dataset hiện tại

Theo dữ liệu/report hiện tại:

| Hạng mục | Giá trị |
|---|---:|
| Raw CSV ở `shared_dataset` | 544,451 dòng |
| `hose_stock_clean.csv` | 544,225 dòng |
| `hose_stock_features.csv` | 506,596 dòng |
| `ml_dataset.csv` | 504,616 dòng |
| Số mã sau làm sạch | 400 |
| Số mã đủ điều kiện train | 396 |
| Số mã bị loại | 4 |

4 mã bị loại vì chưa đủ 250 phiên giao dịch:

```text
CRV, TCX, VCK, VPX
```

Lý do loại trong report là:

```text
fewer_than_250_trading_days
```

## 10. Làm sạch dữ liệu là gì?

Làm sạch dữ liệu nằm ở:

```text
services/preprocessing.py
```

Nó biến dữ liệu raw thành dữ liệu sạch:

```text
hose_stock_raw.csv
→ hose_stock_clean.csv
```

Các việc chính:

1. Kiểm tra file raw có tồn tại không.
2. Kiểm tra có đủ cột bắt buộc không.
3. Chuẩn hóa `symbol` thành chữ hoa.
4. Chuyển `trading_date` về dạng ngày.
5. Chuyển `open`, `high`, `low`, `close`, `volume` về dạng số.
6. Xóa dòng thiếu dữ liệu.
7. Xóa dòng trùng `symbol + trading_date`.
8. Loại giá không hợp lệ, ví dụ giá <= 0.
9. Loại volume âm.
10. Loại dòng OHLC sai logic, ví dụ `high` thấp hơn `close`.
11. Sắp xếp lại theo mã và ngày.
12. Thống kê từng mã để xem mã nào đủ điều kiện train.

File này cũng tạo các report:

```text
reports/data_quality_report.csv
reports/eligible_symbols.csv
reports/excluded_symbols.csv
```

Nói ngắn gọn:

```text
preprocessing.py = thằng rửa dữ liệu
```

## 11. Feature là gì?

Feature là các cột đầu vào cho model.

Con người nhìn biểu đồ giá, đường trung bình, volume để đoán xu hướng. Model không nhìn biểu đồ như con người, nên ta phải biến lịch sử giá/volume thành các con số. Những con số đó gọi là feature.

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
| `volatility_5d` | Độ biến động của lợi suất trong 5 phiên |
| `volatility_20d` | Độ biến động của lợi suất trong 20 phiên |
| `price_range` | Biên độ trong ngày: `(high - low) / close` |
| `volume_change_1d` | Volume thay đổi bao nhiêu so với phiên trước |
| `volume_ratio_20` | Volume hiện tại so với volume trung bình 20 phiên |

Các feature được tính riêng theo từng `symbol`, không trộn dữ liệu giữa các mã.

## 12. Giải thích vài feature quan trọng

`return_1d`:

```text
close(t) / close(t-1) - 1
```

Nếu `return_1d = 0.02`, nghĩa là giá tăng 2% so với phiên trước.

Nếu `return_1d = -0.03`, nghĩa là giá giảm 3% so với phiên trước.

`SMA` là Simple Moving Average, tức trung bình trượt đơn giản.

```text
sma5  = trung bình giá đóng cửa 5 phiên
sma20 = trung bình giá đóng cửa 20 phiên
sma50 = trung bình giá đóng cửa 50 phiên
```

`RSI` là Relative Strength Index, thường nằm trong khoảng 0 đến 100.

Cách hiểu phổ biến:

```text
RSI gần 70 trở lên = giá đã tăng mạnh gần đây
RSI gần 30 trở xuống = giá đã giảm mạnh gần đây
RSI quanh 50 = tương đối cân bằng
```

`volatility` là độ biến động. Cao nghĩa là giá dao động mạnh.

`volume_ratio_20 = 2` nghĩa là volume hôm nay gấp đôi volume trung bình 20 phiên.

## 13. Target, label, horizon, threshold là gì?

`Target` là đáp án model cần học.

Trong project này:

```text
target = 1 → UP
target = 0 → NOT_UP
```

`Horizon` là khoảng thời gian dự báo trong tương lai.

```text
PREDICTION_HORIZON = 5
```

Nghĩa là dự báo sau 5 phiên giao dịch.

`Threshold` là ngưỡng để quyết định có gọi là UP hay không.

```text
UP_THRESHOLD = 0.01
```

Nghĩa là tăng hơn 1%.

Công thức tạo nhãn:

```text
future_close_5d = close(t+5)
future_return_5d = close(t+5) / close(t) - 1

Nếu future_return_5d > 0.01 → target = 1 = UP
Nếu future_return_5d <= 0.01 → target = 0 = NOT_UP
```

Ví dụ:

| Giá ngày t | Giá sau 5 phiên | Return | Nhãn |
|---:|---:|---:|---|
| 100 | 103 | 3.0% | `UP` |
| 100 | 101.1 | 1.1% | `UP` |
| 100 | 100.8 | 0.8% | `NOT_UP` |
| 100 | 99 | -1.0% | `NOT_UP` |

Lưu ý rất quan trọng:

```text
future_return_5d chỉ dùng để tạo đáp án khi train/test.
Không được đưa future_return_5d vào feature.
```

Nếu đưa dữ liệu tương lai vào feature thì model đã nhìn thấy đáp án. Lỗi đó gọi là `data leakage`.

## 14. Train/test split là gì?

Machine Learning không nên đánh giá model trên chính dữ liệu đã học. Nếu làm vậy, model có thể chỉ học thuộc.

Ta chia dữ liệu thành:

| Tập | Vai trò |
|---|---|
| Train | Dữ liệu cho model học |
| Test | Dữ liệu giấu lại để kiểm tra model sau khi học |

Với cổ phiếu, phải chia theo thời gian:

```text
Train = quá khứ
Test  = tương lai
```

Không random shuffle, vì ngoài đời bạn không thể dùng dữ liệu tương lai để dự báo quá khứ.

Project hiện chia theo:

```text
Train: label_end_date <= 2025-12-31
Test : label_end_date > 2025-12-31
```

Report hiện tại:

| Tập | Dòng | Số mã | Khoảng trading_date | UP | NOT_UP |
|---|---:|---:|---|---:|---:|
| Train | 466,973 | 396 | 2019-10-23 đến 2025-12-24 | 180,584 | 286,389 |
| Test | 37,643 | 394 | 2025-10-30 đến 2026-05-29 | 11,683 | 25,960 |

Vì sao test có `trading_date` bắt đầu từ 2025-10-30 dù split date là 2025-12-31?

Vì code chia theo `label_end_date`, không phải chỉ theo `trading_date`. Một dòng ngày 2025-10-30 có thể có nhãn kết thúc sau split date nếu ngày `t+5` nằm sau 2025-12-31. Cách này giúp tránh rò rỉ nhãn tương lai.

## 15. Data leakage là gì?

Data leakage là khi model vô tình được học thông tin mà lúc dự báo thực tế nó không thể biết.

Ví dụ sai:

```text
Dùng close(t+5) làm feature để dự báo tại ngày t.
```

Như vậy model đã biết tương lai.

Project giảm leakage bằng cách:

- Feature tại ngày `t` chỉ dùng dữ liệu từ ngày `t` trở về trước.
- `future_close_5d`, `future_return_5d`, `target`, `label_end_date` không nằm trong `FEATURE_COLUMNS`.
- Train/test chia theo thời gian.
- Cross Validation dùng `TimeSeriesSplit(gap=5)`.
- Logistic Regression dùng `Pipeline(StandardScaler, LogisticRegression)`.

## 16. Cross Validation là gì?

Cross Validation là cách kiểm tra model nhiều lần trên các phần khác nhau của tập train.

Vì dữ liệu cổ phiếu là chuỗi thời gian, project dùng:

```text
TimeSeriesSplit(n_splits=5, gap=5)
```

Ý nghĩa:

- `n_splits=5`: chia train thành 5 lần kiểm tra.
- `gap=5`: bỏ qua 5 mẫu giữa train fold và validation fold để giảm leakage do nhãn 5 phiên.
- Fold sau dùng nhiều dữ liệu quá khứ hơn fold trước.

Cross Validation chỉ dùng trong tập train. Test set chỉ dùng để đánh giá cuối cùng.

## 17. Hyperparameter tuning là gì?

Hyperparameter là thông số cấu hình model trước khi train.

Ví dụ Random Forest có:

| Hyperparameter | Nghĩa |
|---|---|
| `n_estimators` | Số cây trong rừng |
| `max_depth` | Độ sâu tối đa của mỗi cây |
| `min_samples_leaf` | Số mẫu tối thiểu ở một node lá |
| `max_features` | Số feature được xét mỗi lần chia nhánh |

Tuning là thử nhiều bộ hyperparameter để tìm bộ tốt hơn.

Project dùng:

```text
RandomizedSearchCV
n_iter = 12
scoring = f1
```

Nghĩa là mỗi model thử 12 bộ tham số, dùng F1 của lớp `UP` để chọn bộ tốt nhất.

Bộ tham số tốt nhất hiện tại:

| Model | Best params |
|---|---|
| Logistic Regression | `C=0.01`, `solver=lbfgs` |
| Random Forest | `n_estimators=120`, `max_depth=8`, `min_samples_leaf=50`, `max_features=0.5` |
| Gradient Boosting | `n_estimators=60`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.7` |

## 18. Các model trong project

Project train/so sánh 4 model:

| ID | Model | Vai trò |
|---:|---|---|
| 1 | Dummy Classifier | Baseline tối thiểu |
| 2 | Logistic Regression | Model ML đơn giản, dễ giải thích |
| 3 | Random Forest | Model chính hiện được chọn |
| 4 | Gradient Boosting | Model cây boosting để so sánh |

### Dummy Classifier

Model này gần như không học gì thông minh. Nó dùng chiến lược `most_frequent`, tức là luôn đoán lớp phổ biến nhất.

Mục đích:

```text
Nếu model thật không tốt hơn Dummy, project có vấn đề.
```

### Logistic Regression

Model phân loại tuyến tính. Code dùng:

```text
StandardScaler → LogisticRegression
```

`StandardScaler` đưa các feature về thang đo dễ học hơn.

### Random Forest

Random Forest là nhiều cây quyết định cùng bỏ phiếu.

Ưu điểm:

- Hợp dữ liệu dạng bảng.
- Không cần scaling.
- Có feature importance.
- Dễ dùng trong project sinh viên.

Final model hiện tại là Random Forest.

### Gradient Boosting

Gradient Boosting cũng dùng nhiều cây, nhưng học tuần tự: cây sau cố sửa lỗi của cây trước.

Trong kết quả hiện tại, Gradient Boosting có accuracy cao nhưng bắt lớp `UP` rất kém, nên không được chọn.

## 19. Metric đánh giá model

`Accuracy`:

```text
Tỷ lệ dự báo đúng trên toàn bộ test set.
```

Nhưng accuracy không phải tiêu chí chính trong project này, vì dữ liệu bị lệch lớp. Nếu `NOT_UP` nhiều hơn, model cứ đoán `NOT_UP` cũng có accuracy khá cao.

`Precision_UP`:

```text
Trong những lần model dự báo UP, bao nhiêu lần đúng?
```

`Recall_UP`:

```text
Trong tất cả trường hợp thật sự UP, model bắt được bao nhiêu?
```

`F1_UP`:

```text
Chỉ số cân bằng Precision_UP và Recall_UP.
```

Project chọn final model theo:

```text
F1_UP cao nhất trên test set
→ nếu gần bằng thì Recall_UP cao hơn
→ nếu vẫn gần bằng thì model đơn giản hơn
```

## 20. Kết quả model hiện tại

Theo `reports/model_comparison.csv`:

| Model | Accuracy | Precision_UP | Recall_UP | F1_UP | CV F1_UP | Chọn |
|---|---:|---:|---:|---:|---:|---|
| Dummy Classifier | 0.6896 | 0.0000 | 0.0000 | 0.0000 | - | Không |
| Logistic Regression | 0.6030 | 0.3788 | 0.4364 | 0.4056 | 0.4172 | Không |
| Random Forest | 0.5900 | 0.3832 | 0.5267 | 0.4437 | 0.4540 | Có |
| Gradient Boosting | 0.6897 | 0.5014 | 0.0451 | 0.0828 | 0.1860 | Không |

Vì sao Dummy accuracy cao nhưng không được chọn?

Vì test set có nhiều `NOT_UP`. Dummy đoán toàn `NOT_UP` nên accuracy cao, nhưng nó không dự báo được `UP`, do đó `F1_UP = 0`.

Vì sao Gradient Boosting accuracy cao nhưng không được chọn?

Vì `Recall_UP = 0.0451`, tức là trong 100 trường hợp thật sự UP, nó chỉ bắt được khoảng 4 đến 5 trường hợp.

Vì sao Random Forest được chọn?

Vì Random Forest có `F1_UP` cao nhất trong các model chính và có `Recall_UP` tốt nhất.

Final model hiện tại:

```text
models/final_model.pkl = RandomForestClassifier
```

## 21. Confusion Matrix hiện tại

Confusion Matrix là bảng đếm đúng/sai theo từng lớp.

Final model hiện tại:

| Thực tế / Dự báo | Dự báo NOT_UP | Dự báo UP |
|---|---:|---:|
| Thực tế NOT_UP | 16,056 | 9,904 |
| Thực tế UP | 5,529 | 6,154 |

Cách đọc:

- 16,056: thực tế `NOT_UP`, model đoán `NOT_UP`, đúng.
- 9,904: thực tế `NOT_UP`, model đoán `UP`, sai.
- 5,529: thực tế `UP`, model đoán `NOT_UP`, sai.
- 6,154: thực tế `UP`, model đoán `UP`, đúng.

Tổng:

```text
16,056 + 9,904 + 5,529 + 6,154 = 37,643 dòng test
```

## 22. Feature importance hiện tại

Vì final model là Random Forest, project có thể xuất `feature_importance.csv`.

Top 5 feature quan trọng nhất:

| Hạng | Feature | Importance |
|---:|---|---:|
| 1 | `volatility_20d` | 0.3115 |
| 2 | `return_1d` | 0.0892 |
| 3 | `volatility_5d` | 0.0845 |
| 4 | `close_vs_sma20` | 0.0824 |
| 5 | `return_3d` | 0.0735 |

Cách hiểu:

```text
Model đang dựa nhiều vào độ biến động 20 phiên, return gần nhất,
biến động 5 phiên, vị trí giá so với SMA20 và return 3 phiên.
```

Feature importance không chứng minh nguyên nhân thị trường. Nó chỉ nói model đã dùng feature đó nhiều khi ra quyết định.

## 23. Database hiện tại

Database SQLite:

```text
database/stock_prediction.db
```

Các bảng:

| Bảng | Vai trò |
|---|---|
| `raw_prices` | Dữ liệu raw OHLCV |
| `clean_prices` | Dữ liệu sạch |
| `features` | Feature dạng JSON theo từng mã/ngày |
| `tuning_results` | Kết quả tuning |
| `model_evaluations` | Kết quả đánh giá model |
| `predictions` | Lịch sử dự báo |

Số dòng database hiện tại:

| Bảng | Dòng |
|---|---:|
| `raw_prices` | 544,451 |
| `clean_prices` | 544,225 |
| `features` | 506,596 |
| `tuning_results` | 9 |
| `model_evaluations` | 12 |
| `predictions` | 2 |

Lưu ý: các bảng `tuning_results` và `model_evaluations` có thể có nhiều dòng do các lần sync append. Khi xem kết quả model hiện tại, ưu tiên:

```text
reports/model_comparison.csv
models/model_metadata.json
reports/final_model_evaluation.csv
```

## 24. Web demo hoạt động thế nào?

Chạy web:

```powershell
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

Luồng khi nhập mã cổ phiếu:

1. Người dùng nhập mã, ví dụ `FPT`.
2. Web chuẩn hóa thành chữ hoa.
3. `prediction_service.py` đọc `hose_stock_clean.csv`.
4. Lọc dữ liệu của mã đó.
5. Tính lại feature mới nhất bằng `build_features`.
6. Load `models/final_model.pkl`.
7. Load `models/model_metadata.json` để lấy đúng thứ tự feature.
8. Gọi `model.predict(...)` để ra `UP` hoặc `NOT_UP`.
9. Nếu model hỗ trợ, gọi `predict_proba(...)` để lấy xác suất lớp `UP`.
10. Lưu kết quả vào bảng `predictions`.
11. Render kết quả ra `templates/index.html`.

Web không train lại model. Web cũng không realtime. Web chỉ dùng dữ liệu offline đã xử lý và model đã lưu.

## 25. Xác suất lớp UP nghĩa là gì?

Nếu web hiện:

```text
Xác suất lớp UP: 50.1%
```

Không nên gọi là độ tin cậy tuyệt đối.

Nên gọi là:

```text
Xác suất dự báo của model cho lớp UP.
```

Với Random Forest, xác suất này đến từ tổng hợp kết quả của nhiều cây trong rừng.

## 26. `run_pipeline.py` có cần thiết không?

Không bắt buộc tuyệt đối, nhưng rất nên có.

Bạn có thể chạy từng bước:

```powershell
python scripts/preprocess_data.py
python scripts/build_features.py
python scripts/train_tune_models.py
python scripts/evaluate_models.py
python scripts/select_final_model.py
python database/init_db.py
```

Nhưng `run_pipeline.py` gom lại thành một lệnh:

```powershell
python scripts/run_pipeline.py
```

Nó giúp:

- Không quên bước.
- Không chạy sai thứ tự.
- Dễ tái lập kết quả.
- Dễ chứng minh project có pipeline đầy đủ.
- Phù hợp khi báo cáo niên luận.

## 27. Các lệnh chạy thường dùng

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

Chạy web:

```powershell
python app.py
```

Dự báo CLI:

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol FPT --log-db
```

## 28. Các output quan trọng

| File | Vai trò |
|---|---|
| `data/processed/hose_stock_clean.csv` | Dữ liệu sạch |
| `data/processed/hose_stock_features.csv` | Dữ liệu có 15 feature |
| `data/processed/ml_dataset.csv` | Dữ liệu train/test có target |
| `models/final_model.pkl` | Model cuối cùng web đang dùng |
| `models/model_metadata.json` | Metadata của final model |
| `reports/model_comparison.csv` | Bảng so sánh 4 model |
| `reports/final_model_evaluation.csv` | Kết quả final model |
| `reports/classification_report.csv` | Precision/Recall/F1 theo từng lớp |
| `reports/confusion_matrix.csv` | Bảng đúng/sai |
| `reports/confusion_matrix.png` | Hình confusion matrix |
| `reports/feature_importance.csv` | Độ quan trọng feature |
| `reports/train_test_summary.csv` | Tóm tắt train/test |
| `reports/best_params.json` | Tham số tốt nhất sau tuning |
| `database/stock_prediction.db` | SQLite database |

## 29. Những điểm nên nói khi bảo vệ

Bạn có thể nói:

```text
Project của em xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning.
Bài toán được định nghĩa là phân loại nhị phân: dự báo cổ phiếu có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không.
Dữ liệu đầu vào là OHLCV theo ngày.
Em tạo các feature kỹ thuật như return, SMA, RSI, volatility và volume ratio.
Dữ liệu được chia train/test theo thời gian bằng label_end_date để tránh data leakage.
Project train và tuning 3 model chính: Logistic Regression, Random Forest, Gradient Boosting, đồng thời dùng Dummy Classifier làm baseline.
Final model được chọn theo F1_UP trên test set, không chọn theo accuracy đơn thuần.
Model hiện tại được chọn là Random Forest.
Web Flask chỉ dùng để demo kết quả dự báo, không phải hệ thống realtime và không phải khuyến nghị đầu tư.
```

## 30. Câu hỏi dễ bị hỏi và cách trả lời

Nếu hỏi: Vì sao không chọn model accuracy cao nhất?

```text
Vì dữ liệu bị lệch lớp, NOT_UP nhiều hơn UP. Nếu chỉ chọn accuracy, model có thể đoán NOT_UP rất nhiều nhưng không bắt được lớp UP. Project quan tâm xu hướng tăng nên ưu tiên F1_UP.
```

Nếu hỏi: Vì sao Dummy accuracy cao?

```text
Vì Dummy đoán lớp phổ biến nhất là NOT_UP. Trong test set, NOT_UP chiếm đa số nên accuracy cao, nhưng F1_UP bằng 0 vì không dự báo được UP.
```

Nếu hỏi: Vì sao Gradient Boosting không được chọn?

```text
Vì recall của lớp UP rất thấp. Nó có accuracy cao nhưng gần như không bắt được các trường hợp thật sự UP.
```

Nếu hỏi: Dự báo này dùng để đầu tư được không?

```text
Không. Đây là hệ thống hỗ trợ/demo học thuật. Kết quả chỉ mang tính tham khảo, không phải khuyến nghị mua bán.
```

Nếu hỏi: Web có realtime không?

```text
Không. Web dùng dữ liệu offline đã xử lý và model đã train sẵn.
```

Nếu hỏi: 5 phiên giao dịch có phải 5 ngày không?

```text
Không hẳn. 5 phiên giao dịch là 5 ngày có giao dịch thật trên sàn, không tính cuối tuần hoặc ngày nghỉ.
```

Nếu hỏi: Có dùng database không?

```text
Có. Project dùng SQLite để lưu raw data, clean data, features, tuning results, model evaluations và predictions. Tuy nhiên kết quả model hiện tại nên xem ở report và model_metadata.
```

## 31. Thứ tự học lại project

Nếu bạn mới học, nên đi theo thứ tự:

1. Hiểu câu hỏi chính: tăng hơn 1% sau 5 phiên hay không.
2. Hiểu OHLCV.
3. Hiểu raw → clean → features → ml_dataset.
4. Hiểu cách tạo `future_return_5d`.
5. Hiểu `target = 1` là `UP`, `target = 0` là `NOT_UP`.
6. Học 15 feature.
7. Hiểu train/test theo thời gian.
8. Hiểu data leakage.
9. Hiểu Precision, Recall, F1, Confusion Matrix.
10. Đọc `reports/model_comparison.csv`.
11. Đọc `services/feature_engineering.py`.
12. Đọc `services/model_tuning.py`.
13. Đọc `services/prediction_service.py`.
14. Chạy web và thử dự báo vài mã.

Chỉ cần nắm chắc những ý trên là bạn đã hiểu phần lõi của project.

## 32. Kết luận cực ngắn

Project này không phải là một app đầu tư hoàn chỉnh.

Nó là một pipeline Machine Learning cho bài toán:

```text
Dùng dữ liệu OHLCV quá khứ
→ tạo feature kỹ thuật
→ học quy luật UP / NOT_UP
→ chọn model tốt nhất theo F1_UP
→ demo dự báo trên web Flask
```

Hiện tại:

```text
Final model = Random Forest
Horizon = 5 phiên
Threshold = 1%
Feature count = 15
Train rows = 466,973
Test rows = 37,643
F1_UP = 0.4437
```
