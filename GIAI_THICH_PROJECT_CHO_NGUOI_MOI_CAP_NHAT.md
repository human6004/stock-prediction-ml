# Giải thích project dự báo xu hướng cổ phiếu HOSE cho người mới

Tài liệu này được viết lại theo code và report hiện tại trong project:

- Project: `D:\study\niên luận\v3\niên luận cơ sở ngành-cusor`
- Roadmap: `D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md`
- Ngày đọc lại: 2026-06-12

Mục tiêu: giúp bạn hiểu project đang làm gì, các thuật ngữ nghĩa là gì, thông số nào quan trọng, feature nào được dùng, vì sao chọn model hiện tại, và khi bảo vệ có thể nói thế nào.

## 1. Project này nói một câu là gì?

Project này xây dựng một hệ thống Machine Learning để trả lời câu hỏi:

```text
Một mã cổ phiếu trên sàn HOSE có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không?
```

Nếu model dự báo:

- `UP`: model cho rằng giá có khả năng tăng hơn 1% sau 5 phiên giao dịch.
- `NOT_UP`: model cho rằng giá không đạt điều kiện tăng hơn 1%. Nó có thể giảm, đi ngang, hoặc tăng nhẹ dưới 1%.

Đây là hệ thống hỗ trợ/demo học thuật, không phải khuyến nghị mua bán cổ phiếu.

## 2. Roadmap yêu cầu gì?

Roadmap mô tả pipeline chuẩn:

```text
Thu thập dữ liệu HOSE
→ Lưu database
→ Làm sạch dữ liệu
→ Kiểm tra chất lượng dữ liệu
→ Tạo feature
→ Tạo nhãn UP / NOT_UP
→ Chia train/test theo thời gian
→ Cross Validation theo thời gian
→ Hyperparameter tuning
→ Đánh giá model
→ Chọn final_model.pkl
→ Web Flask demo dự báo
```

Code hiện tại đã triển khai phần lớn pipeline này:

- Có đọc/cập nhật dữ liệu HOSE từ CSV và `vnstock`.
- Có làm sạch dữ liệu.
- Có lọc mã đủ tối thiểu 250 phiên.
- Có tạo 15 feature kỹ thuật.
- Có tạo nhãn `UP / NOT_UP`.
- Có chia train/test theo `label_end_date`.
- Có tuning 3 model chính bằng `RandomizedSearchCV`.
- Có chọn final model theo `F1_UP`.
- Có lưu model, report, metadata.
- Có SQLite để sync dữ liệu/report và lưu lịch sử dự báo.
- Có Flask web demo.

## 3. Những file quan trọng

```text
config/settings.py
```

Chứa cấu hình chính: đường dẫn dữ liệu, horizon, threshold, feature list, split date, số fold CV, đường dẫn model/report.

```text
scripts/run_pipeline.py
```

Script chạy toàn bộ dây chuyền: check roadmap, đọc dataset, clean, feature, label, split, tune model, evaluate, chọn final model, ghi report, sync database.

```text
services/preprocessing.py
```

Kiểm tra và làm sạch dữ liệu OHLCV.

```text
services/feature_engineering.py
```

Tạo feature, tạo target `UP / NOT_UP`, chia train/test theo thời gian.

```text
services/model_tuning.py
```

Train Dummy, tune Logistic Regression, Random Forest, Gradient Boosting.

```text
services/model_evaluation.py
```

Tính metric, chọn model cuối, ghi report, confusion matrix, feature importance, metadata.

```text
services/prediction_service.py
```

Tính feature mới nhất cho một mã cổ phiếu và gọi `final_model.pkl` để dự báo.

```text
app.py
```

Flask web backend. Có trang nhập mã, trang đánh giá model, route xem confusion matrix.

```text
database/
```

Chứa SQLite database và script sync dữ liệu/report.

## 4. Dữ liệu OHLCV là gì?

OHLCV là dữ liệu giá và khối lượng giao dịch theo ngày:

| Cột | Nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu, ví dụ `FPT`, `VNM`, `SSI` |
| `trading_date` | Ngày giao dịch |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất trong ngày |
| `low` | Giá thấp nhất trong ngày |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch |

Project chỉ dùng dữ liệu OHLCV. Nó không dùng tin tức, báo cáo tài chính, dữ liệu realtime, sentiment, LSTM hay deep learning.

## 5. Các thông số cấu hình chính

Theo `config/settings.py`:

| Thông số | Giá trị hiện tại | Nghĩa |
|---|---:|---|
| `SPLIT_DATE` | `2025-12-31` | Mốc chia train/test |
| `PREDICTION_HORIZON` | `5` | Dự báo sau 5 phiên giao dịch |
| `UP_THRESHOLD` | `0.01` | Tăng hơn 1% thì tính là `UP` |
| `MIN_TRADING_DAYS` | `250` | Mã có dưới 250 dòng dữ liệu bị loại khỏi train |
| `MIN_AVERAGE_VOLUME` | `0` | Hiện chưa lọc theo thanh khoản trung bình |
| `CV_N_SPLITS` | `5` | Cross Validation có 5 fold |
| `CV_GAP` | `5` | Chừa khoảng cách 5 mẫu giữa train và validation |
| `TUNING_N_ITER` | `12` | Mỗi model thử 12 bộ tham số |
| `TUNING_SCORING` | `f1` | Tuning ưu tiên F1 của lớp `UP` |
| `RANDOM_STATE` | `42` | Giúp kết quả random ổn định hơn |

## 6. Số liệu dữ liệu hiện tại

Theo các file output/report hiện tại:

| Hạng mục | Giá trị |
|---|---:|
| Raw CSV trong `shared_dataset` | 544,451 dòng |
| Dữ liệu sạch `hose_stock_clean.csv` | 544,225 dòng |
| Dữ liệu feature `hose_stock_features.csv` | 506,596 dòng |
| ML dataset `ml_dataset.csv` | 504,616 dòng |
| Số mã sau làm sạch | 400 mã |
| Số mã đủ điều kiện train | 396 mã |
| Số mã bị loại | 4 mã |

4 mã bị loại vì chưa đủ 250 phiên:

```text
CRV, TCX, VCK, VPX
```

## 7. Làm sạch dữ liệu là làm gì?

Trong `services/preprocessing.py`, code làm các việc sau:

1. Kiểm tra dataset có đủ cột bắt buộc không.
2. Chuẩn hóa `symbol` thành chữ hoa.
3. Ép `trading_date` về kiểu ngày.
4. Ép `open`, `high`, `low`, `close`, `volume` về dạng số.
5. Xóa dòng thiếu dữ liệu.
6. Xóa dòng trùng `symbol + trading_date`.
7. Loại giá không hợp lệ, ví dụ giá <= 0.
8. Loại volume âm.
9. Loại dòng OHLC bất hợp lệ, ví dụ `high` nhỏ hơn `open/close/low`.
10. Sắp xếp theo `symbol`, `trading_date`.
11. Lập báo cáo chất lượng dữ liệu.

Sau đó code lọc mã đủ điều kiện train. Hiện tiêu chí chính là có tối thiểu 250 phiên giao dịch.

## 8. Feature là gì?

Feature là cột đầu vào cho model.

Con người nhìn biểu đồ giá, đường trung bình, volume để đoán xu hướng. Model không nhìn biểu đồ trực tiếp, nên ta biến lịch sử giá/volume thành các con số. Những con số đó là feature.

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
| `close_vs_sma20` | Giá hiện tại đang cao/thấp hơn SMA20 bao nhiêu |
| `sma20_vs_sma50` | SMA20 cao/thấp hơn SMA50 bao nhiêu |
| `rsi14` | Chỉ báo sức mạnh tương đối 14 phiên |
| `volatility_5d` | Độ biến động 5 phiên |
| `volatility_20d` | Độ biến động 20 phiên |
| `price_range` | Biên độ dao động trong ngày: `(high - low) / close` |
| `volume_change_1d` | Volume tăng/giảm bao nhiêu so với phiên trước |
| `volume_ratio_20` | Volume hiện tại so với volume trung bình 20 phiên |

Các feature được tính riêng theo từng mã cổ phiếu, không trộn lịch sử của mã này sang mã khác.

## 9. Một vài feature quan trọng nên hiểu kỹ

`return_1d = close(t) / close(t-1) - 1`

Nếu bằng `0.02`, giá tăng 2% so với phiên trước. Nếu bằng `-0.03`, giá giảm 3%.

`SMA` là Simple Moving Average, nghĩa là trung bình trượt đơn giản.

- `sma5`: nhìn ngắn hạn.
- `sma20`: nhìn trung hạn.
- `sma50`: nhìn dài hơn.

`RSI` thường dao động từ 0 đến 100:

- Gần 70 trở lên: giá đã tăng mạnh trong giai đoạn gần đây.
- Gần 30 trở xuống: giá đã giảm mạnh.
- Gần 50: tương đối cân bằng.

`volatility` là độ biến động. Cao nghĩa là giá dao động mạnh, thường khó dự báo hơn.

`volume_ratio_20 = 2` nghĩa là volume hôm nay gấp đôi trung bình 20 phiên.

## 10. Target, label, horizon, threshold

Target là đáp án mà model cần học.

Với mỗi mã cổ phiếu, tại ngày `t`, code tính:

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

| Giá tại ngày t | Giá sau 5 phiên | Return | Label |
|---:|---:|---:|---|
| 100 | 103 | 3.0% | `UP` |
| 100 | 101.1 | 1.1% | `UP` |
| 100 | 100.8 | 0.8% | `NOT_UP` |
| 100 | 99 | -1.0% | `NOT_UP` |

`future_return_5d` chỉ dùng để tạo đáp án trong lúc train/test. Nó không được đưa vào feature. Nếu đưa nó vào feature thì model đã nhìn thấy tương lai, gọi là data leakage.

## 11. Train/test split hiện tại

Code chia dữ liệu theo `label_end_date`, không chia random.

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

Vì project chia theo `label_end_date`, tức ngày kết thúc của nhãn t+5. Một dòng có ngày giao dịch trước split vẫn có thể bị đưa vào test nếu nhãn của nó kết thúc sau split. Cách này giúp giảm nguy cơ dùng thông tin tương lai trong train.

## 12. Data leakage là gì?

Data leakage là khi model vô tình được học thông tin mà ngoài đời nó không thể biết tại thời điểm dự báo.

Ví dụ sai:

```text
Dùng close(t+5) làm feature để dự báo target tại ngày t.
```

Khi đó model đã biết đáp án tương lai, kết quả đánh giá sẽ đẹp giả.

Project giảm leakage bằng cách:

- Feature tại ngày `t` chỉ dùng dữ liệu từ `t` trở về trước.
- `future_close_5d`, `future_return_5d`, `target`, `label_end_date` không nằm trong `FEATURE_COLUMNS`.
- Train/test chia theo thời gian, không shuffle.
- Cross Validation dùng `TimeSeriesSplit(gap=5)`.
- Logistic Regression dùng `Pipeline(StandardScaler, LogisticRegression)` để scaler fit trong pipeline.

## 13. Cross Validation là gì?

Cross Validation là cách chia tập train thành nhiều phần nhỏ để thử model ổn định không.

Với dữ liệu chứng khoán, không nên dùng KFold random vì sẽ làm lẫn quá khứ và tương lai.

Project dùng:

```text
TimeSeriesSplit(n_splits=5, gap=5)
```

Nghĩa là:

- Chia train thành 5 fold theo thứ tự thời gian.
- Fold sau dùng nhiều dữ liệu quá khứ hơn fold trước.
- `gap=5` bỏ qua 5 mẫu giữa train và validation để giảm rò rỉ do nhãn 5 phiên.

## 14. Hyperparameter tuning là gì?

Hyperparameter là tham số cấu hình model trước khi train.

Ví dụ với Random Forest:

- `n_estimators`: số cây.
- `max_depth`: độ sâu tối đa của cây.
- `min_samples_leaf`: số mẫu tối thiểu ở mỗi lá.
- `max_features`: số feature được xét mỗi lần chia cây.

Tuning là thử nhiều bộ hyperparameter để tìm bộ tốt hơn.

Project dùng:

```text
RandomizedSearchCV
n_iter = 12
scoring = f1
```

Bộ tham số tốt nhất hiện tại:

| Model | Best params |
|---|---|
| Logistic Regression | `C=0.01`, `solver=lbfgs` |
| Random Forest | `n_estimators=120`, `max_depth=8`, `min_samples_leaf=50`, `max_features=0.5` |
| Gradient Boosting | `n_estimators=60`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.7` |

## 15. Các model trong project

Project train 4 model:

| ID | Model | Vai trò |
|---:|---|---|
| 1 | Dummy Classifier | Baseline tối thiểu |
| 2 | Logistic Regression | Model đơn giản, dễ giải thích |
| 3 | Random Forest | Model chính hiện được chọn |
| 4 | Gradient Boosting | Model cây boosting để so sánh |

Dummy Classifier dùng chiến lược `most_frequent`, nghĩa là gần như luôn đoán lớp xuất hiện nhiều nhất. Nó không thông minh, chỉ dùng để làm mốc so sánh.

Logistic Regression là model phân loại tuyến tính. Vì các feature có thang đo khác nhau, code đặt `StandardScaler` trước Logistic Regression.

Random Forest là nhiều cây quyết định cùng bỏ phiếu. Nó hợp dữ liệu dạng bảng, không cần scaling, và có thể xuất feature importance.

Gradient Boosting cũng dùng nhiều cây nhưng học tuần tự, cây sau cố sửa lỗi của cây trước.

## 16. Các metric đánh giá

`Accuracy` là tỷ lệ dự báo đúng tổng thể.

```text
accuracy = số dự báo đúng / tổng số mẫu
```

Nhưng trong project này không nên chọn model chỉ bằng accuracy, vì lớp `NOT_UP` nhiều hơn `UP`. Model cứ đoán `NOT_UP` nhiều cũng có accuracy nhìn đẹp.

`Precision_UP` trả lời:

```text
Trong những lần model dự báo UP, bao nhiêu lần đúng?
```

`Recall_UP` trả lời:

```text
Trong tất cả trường hợp thật sự UP, model bắt được bao nhiêu?
```

`F1_UP` cân bằng Precision_UP và Recall_UP.

Project chọn final model theo `F1_UP`, vì mục tiêu là bắt lớp tăng `UP`, không phải chỉ đoán đúng tổng thể.

## 17. Kết quả model hiện tại

Theo `reports/model_comparison.csv`:

| Model | Accuracy | Precision_UP | Recall_UP | F1_UP | CV F1_UP | Chọn |
|---|---:|---:|---:|---:|---:|---|
| Dummy Classifier | 0.6896 | 0.0000 | 0.0000 | 0.0000 | - | Không |
| Logistic Regression | 0.6030 | 0.3788 | 0.4364 | 0.4056 | 0.4172 | Không |
| Random Forest | 0.5900 | 0.3832 | 0.5267 | 0.4437 | 0.4540 | Có |
| Gradient Boosting | 0.6897 | 0.5014 | 0.0451 | 0.0828 | 0.1860 | Không |

Vì sao Dummy accuracy cao nhưng không được chọn?

Vì test set có nhiều `NOT_UP`. Dummy đoán lớp phổ biến nhất nên accuracy cao, nhưng không bắt được `UP`, nên `F1_UP = 0`.

Vì sao Gradient Boosting accuracy cao nhưng không được chọn?

Vì `Recall_UP = 0.0451`, tức trong 100 trường hợp thật sự `UP`, nó chỉ bắt được khoảng 4 đến 5 trường hợp. Nó quá né lớp `UP`.

Vì sao Random Forest được chọn?

Vì Random Forest có `F1_UP` cao nhất trong các model thật sự, đồng thời `Recall_UP` cũng cao nhất. Final model hiện tại là:

```text
models/final_model.pkl = RandomForestClassifier
```

## 18. Confusion matrix hiện tại

Confusion matrix của final model:

| Thực tế / Dự báo | Dự báo NOT_UP | Dự báo UP |
|---|---:|---:|
| Thực tế NOT_UP | 16,056 | 9,904 |
| Thực tế UP | 5,529 | 6,154 |

Cách đọc:

- 16,056: thực tế `NOT_UP`, model đoán `NOT_UP`, đúng.
- 9,904: thực tế `NOT_UP`, model đoán `UP`, sai.
- 5,529: thực tế `UP`, model đoán `NOT_UP`, sai.
- 6,154: thực tế `UP`, model đoán `UP`, đúng.

Tổng test:

```text
16,056 + 9,904 + 5,529 + 6,154 = 37,643 dòng
```

## 19. Feature importance hiện tại

Vì final model là Random Forest, project xuất được `feature_importance.csv`.

Top 5 feature quan trọng nhất:

| Thứ hạng | Feature | Importance |
|---:|---|---:|
| 1 | `volatility_20d` | 0.3115 |
| 2 | `return_1d` | 0.0892 |
| 3 | `volatility_5d` | 0.0845 |
| 4 | `close_vs_sma20` | 0.0824 |
| 5 | `return_3d` | 0.0735 |

Cách hiểu: model đang dựa nhiều vào biến động 20 phiên, lợi suất gần nhất, biến động 5 phiên, vị trí giá so với SMA20, và lợi suất 3 phiên.

Feature importance không chứng minh nguyên nhân thị trường. Nó chỉ nói model đã dùng feature đó nhiều khi ra quyết định.

## 20. Database hiện tại

Roadmap yêu cầu SQLite, và code hiện tại đã có:

```text
database/stock_prediction.db
```

Các bảng:

| Bảng | Vai trò |
|---|---|
| `raw_prices` | Dữ liệu OHLCV thô |
| `clean_prices` | Dữ liệu sau làm sạch |
| `features` | Feature dạng JSON theo từng mã/ngày |
| `tuning_results` | Kết quả tuning |
| `model_evaluations` | Kết quả đánh giá model |
| `predictions` | Lịch sử dự báo từ CLI/web |

Số dòng database hiện tại:

| Bảng | Dòng |
|---|---:|
| `raw_prices` | 544,451 |
| `clean_prices` | 544,225 |
| `features` | 506,596 |
| `tuning_results` | 3 |
| `model_evaluations` | 4 |
| `predictions` | 2 |

Lưu ý: nguồn đáng tin nhất cho kết quả model hiện tại vẫn là `reports/model_comparison.csv` và `models/model_metadata.json`. Bảng `model_evaluations` trong SQLite có thể là bản sync từ lần chạy trước.

## 21. Web demo hoạt động thế nào?

Khi chạy:

```powershell
python app.py
```

Web mở tại:

```text
http://127.0.0.1:5000
```

Luồng khi nhập mã, ví dụ `FPT`:

1. Web nhận `symbol`.
2. Chuẩn hóa thành chữ hoa.
3. Đọc dữ liệu clean.
4. Lọc các dòng của mã đó.
5. Tính lại feature mới nhất bằng `build_features`.
6. Lấy dòng feature mới nhất.
7. Load `models/final_model.pkl`.
8. Lấy đúng thứ tự feature từ `model_metadata.json`.
9. Gọi `model.predict`.
10. Nếu model hỗ trợ thì gọi `predict_proba` để lấy xác suất lớp `UP`.
11. Lưu dự báo vào bảng `predictions`.
12. Hiển thị kết quả trên HTML.

Web không train lại model. Web cũng không tự cập nhật realtime. Nó dự báo dựa trên dữ liệu offline đã xử lý.

## 22. Xác suất lớp UP nghĩa là gì?

Nếu web hiện:

```text
Xác suất lớp UP: 50.1%
```

Không nên gọi đây là độ tin cậy tuyệt đối.

Nên gọi là:

```text
Xác suất dự báo của model cho lớp UP.
```

Với Random Forest, xác suất này đến từ tổng hợp xác suất/phiếu của nhiều cây.

## 23. Các lệnh chạy thường dùng

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

## 24. Những output quan trọng

| File | Vai trò |
|---|---|
| `data/processed/hose_stock_clean.csv` | Dữ liệu sạch |
| `data/processed/hose_stock_features.csv` | Dữ liệu có feature, web có thể dùng để lấy metadata |
| `data/processed/ml_dataset.csv` | Dữ liệu feature + target dùng train/test |
| `models/final_model.pkl` | Model cuối cùng đang dùng |
| `models/model_metadata.json` | Thông tin model, feature order, metric |
| `reports/model_comparison.csv` | Bảng so sánh model |
| `reports/final_model_evaluation.csv` | Kết quả final model |
| `reports/confusion_matrix.csv` | Bảng đúng/sai |
| `reports/confusion_matrix.png` | Hình confusion matrix |
| `reports/feature_importance.csv` | Độ quan trọng feature |
| `reports/best_params.json` | Bộ tham số tốt nhất |
| `reports/train_test_summary.csv` | Tóm tắt train/test |

## 25. Một vài lưu ý khi học/bảo vệ

Không nói project dự báo giá chính xác. Nên nói project dự báo xu hướng `UP / NOT_UP`.

Không nói xác suất model là độ tin cậy tuyệt đối. Nên nói là xác suất dự báo của model.

Không chọn model theo accuracy. Project chọn theo `F1_UP`.

Không nói 5 phiên giao dịch là đúng 5 ngày lịch. Cuối tuần, ngày nghỉ, hoặc mã giao dịch thưa có thể làm khoảng ngày thực tế dài hơn.

Không nói web realtime. Web dùng dữ liệu offline đã xử lý.

Không nói kết quả là khuyến nghị đầu tư. Đây là demo học thuật.

## 26. Một vài điểm code nên nhớ

File `GIAI_THICH_PROJECT_CHO_NGUOI_MOI.md` và `SO_DO_KIEN_TRUC_HE_THONG.md` trong repo có vài chi tiết cũ, ví dụ từng nói chưa có SQLite hoặc mốc split khác. Khi cần trình bày, hãy ưu tiên tài liệu cập nhật này cùng `README.md`, `config/settings.py`, `reports/model_comparison.csv`, `models/model_metadata.json`.

Trong `templates/index.html`, phần technical panel có nhắc `selected_report_model` và `model_match`, nhưng `services/prediction_service.py` hiện chưa trả hai field này. Vì vậy phần kỹ thuật đó có thể hiển thị chưa chuẩn. Phần dự báo chính vẫn dùng `final_model.pkl` và `model_metadata.json`.

Trong SQLite, `tuning_results.best_params_json` hiện đang ghi `{}` trong `database_service.py`; best params chi tiết nằm ở `reports/best_params.json` và `models/model_metadata.json`.

## 27. Câu trả lời mẫu khi bảo vệ

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
Với mỗi mã tại ngày t, em lấy giá đóng cửa sau 5 phiên, tính future_return_5d = close(t+5) / close(t) - 1. Nếu giá tăng hơn 1% thì nhãn là UP, ngược lại là NOT_UP.
```

Nếu hỏi vì sao không random split:

```text
Dữ liệu cổ phiếu là dữ liệu chuỗi thời gian. Nếu random split thì dữ liệu tương lai có thể lọt vào train, gây data leakage. Vì vậy em chia train/test theo thời gian, cụ thể là theo label_end_date.
```

Nếu hỏi vì sao chọn Random Forest:

```text
Final model được chọn theo F1 của lớp UP trên test set. Random Forest có F1_UP cao nhất trong các model chính, nên được chọn làm model cuối.
```

Nếu hỏi vì sao Gradient Boosting accuracy cao nhưng không chọn:

```text
Gradient Boosting có accuracy cao nhưng recall của lớp UP rất thấp, nghĩa là gần như không bắt được các trường hợp thật sự tăng. Vì project quan tâm lớp UP nên không chọn theo accuracy đơn thuần.
```

Nếu hỏi có dùng database không:

```text
Có. Project có SQLite để lưu raw data, clean data, features, tuning results, model evaluations và predictions. Tuy nhiên source chính để xem kết quả model hiện tại là các file report và metadata.
```

Nếu hỏi có khuyến nghị đầu tư không:

```text
Không. Đây là hệ thống demo hỗ trợ dự báo xu hướng phục vụ niên luận, kết quả chỉ mang tính tham khảo và không phải khuyến nghị mua bán.
```

## 28. Thứ tự học lại project

1. Hiểu câu hỏi chính: có tăng hơn 1% sau 5 phiên không.
2. Hiểu OHLCV.
3. Hiểu cách tạo `future_return_5d` và `target`.
4. Học 15 feature, nhất là return, SMA, RSI, volatility, volume ratio.
5. Hiểu train/test theo thời gian và data leakage.
6. Hiểu Precision, Recall, F1, Confusion Matrix.
7. Đọc `reports/model_comparison.csv` để hiểu vì sao chọn Random Forest.
8. Đọc `services/feature_engineering.py` để hiểu feature và label.
9. Đọc `services/model_tuning.py` để hiểu train/tune model.
10. Đọc `services/prediction_service.py` và `app.py` để hiểu web dự báo.

Chỉ cần nắm chắc 10 ý này là bạn đã hiểu lõi project.
