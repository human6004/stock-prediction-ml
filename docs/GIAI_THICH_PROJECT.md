# Giải thích project dự báo xu hướng cổ phiếu HOSE cho người mới

Tài liệu này được viết lại sau khi đọc kỹ code, report, model metadata, web demo, database schema và trạng thái pipeline hiện tại. Mục tiêu: giải thích từ đầu, coi bạn là người **chưa biết gì** về project, chưa quen thuật ngữ chứng khoán và Machine Learning.

- Project: `D:\study\niên luận\stock-prediction-ml`
- Roadmap: `D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md`
- Ngày đọc lại: `2026-07-17`

> **Trạng thái hiện tại (đọc trước khi trích số):** Pipeline chính thức đã chạy trên bộ **20 feature**, raw data mới nhất đến `2026-07-10`, fingerprint ML dataset `f6cb3ac8f820`. **Final model đang phục vụ là `Gradient Boosting`**, `trained_at = 2026-07-17`. TEST metrics, confusion matrix, feature importance và model metadata bên dưới lấy từ official pipeline run này. Fetch report và các điểm CV lúc tuning là các run riêng, nên tài liệu luôn ghi rõ nguồn khi số có thể khác.

---

## 0. Project trong 5 phút

Nếu chỉ nhớ 8 ý, hãy nhớ:

1. Project **không dự báo giá chính xác**. Nó phân loại một mã thành `UP` hoặc `NOT_UP`.
2. `UP` nghĩa là giá đóng cửa ở **dòng giao dịch thứ 5 kế tiếp của chính mã đó** tăng hơn 1%; `NOT_UP` là các trường hợp còn lại.
3. Đầu vào chỉ là OHLCV theo ngày. Code biến chúng thành 20 feature kỹ thuật.
4. Dữ liệu quá khứ được chia thành TRAIN và TEST bằng `label_end_date`.
5. Tuning dùng CV trên TRAIN để thử cấu hình. Điểm lưu lịch sử là **CV F1_UP**, không phải TEST F1_UP.
6. Official pipeline fit model, đánh giá TEST rồi chọn final model theo TEST F1_UP.
7. Final model hiện tại là Gradient Boosting; TEST F1_UP khoảng `0.5053`. Kết quả chỉ ở mức khiêm tốn, không phải hệ thống dự báo chắc thắng.
8. Web/CLI chỉ tải model đã train và dự báo dữ liệu offline mới nhất; không train lại và không realtime.

Một số từ xuất hiện xuyên suốt:

| Từ | Hiểu đơn giản |
|---|---|
| Model | Công thức/quy luật máy đã học từ dữ liệu |
| Train / fit | Cho model học từ dữ liệu có đáp án |
| Validation | Phần dữ liệu TRAIN tạm giữ lại để kiểm tra lúc CV |
| Test | Tập tách riêng dùng ở bước đánh giá và chọn final model hiện tại |
| Fold | Một lần chia TRAIN thành phần học và phần validation |
| `label_end_date` | Ngày của dòng tương lai dùng để kết thúc việc tạo nhãn |
| CV F1_UP | F1 lớp UP trung bình trên các validation fold thuộc TRAIN |
| Fingerprint | Chữ ký ngắn để đối chiếu data/config giữa các run |
| Artifact | File sinh ra sau pipeline, ví dụ `.pkl`, CSV report, metadata |

Đọc lần đầu: đọc mục 0, 1, 3, 11, 13-18 và 22. Các mục còn lại dùng để tra chi tiết, chạy project hoặc chuẩn bị bảo vệ.

---

## 1. Project này làm gì, nói thật ngắn gọn?

Project xây dựng một hệ thống Machine Learning để trả lời **một câu hỏi duy nhất**:

```text
Giá của một mã HOSE ở bước quan sát thứ 5 tiếp theo có tăng hơn 1% hay không?
```

Kết quả chỉ có 2 lớp:

| Kết quả | Nghĩa |
|---|---|
| `UP` | Model dự báo giá đóng cửa ở dòng thứ 5 kế tiếp có khả năng **tăng hơn 1%** so với hiện tại |
| `NOT_UP` | Model dự báo **không đạt** điều kiện tăng hơn 1%; có thể giảm, đi ngang, hoặc tăng nhẹ dưới 1% |

Ví dụ một dòng tại ngày `t` có `close=100`. Khi tạo dữ liệu học, nếu dòng thứ 5 kế tiếp của cùng mã có `close=102`, return là 2% nên đáp án thật là `UP`. Khi dự báo thật, giá tương lai chưa tồn tại; model chỉ nhìn 20 feature tại `t`, sinh `P(UP)`, rồi so xác suất này với decision threshold để trả `UP/NOT_UP`.

Đây là project học thuật/niên luận. Nó **không** phải hệ thống tư vấn đầu tư, **không** tự mua bán cổ phiếu, **không** đảm bảo lợi nhuận.

---

## 2. Roadmap yêu cầu gì?

Roadmap mô tả một pipeline (dây chuyền xử lý) chuẩn:

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

Code hiện tại đã triển khai đúng hướng chính:

- Có dữ liệu HOSE dạng OHLCV.
- Có script cập nhật dữ liệu bằng thư viện `vnstock`.
- Có làm sạch dữ liệu và báo cáo chất lượng dữ liệu.
- Có lọc mã đủ tối thiểu 250 phiên giao dịch.
- Có tạo **20 feature** kỹ thuật.
- Có tạo nhãn `UP / NOT_UP` với horizon 5 dòng quan sát của từng mã và threshold 1%.
- Có chia train/test theo thời gian bằng `label_end_date`.
- Có `TimeSeriesSplit(gap=5)` cho Cross Validation.
- Có Tuning Lab (`/tuning`) để chạy CV từng cấu hình và chốt cấu hình; repo cũng có script thí nghiệm tự động thử nhiều cấu hình qua cùng backend.
- Có Dummy Classifier làm baseline.
- Có tinh chỉnh **decision threshold** riêng cho từng model để tối ưu F1 lớp UP.
- Có chọn final model theo `F1_UP` (tie-break bằng `Recall_UP` rồi độ đơn giản).
- Có Flask web demo và CLI dự báo.
- Có SQLite để sync dữ liệu/report và lưu lịch sử dự báo.

Điểm cần hiểu: trong code, **CSV vẫn là artefact chính** của pipeline; SQLite chỉ là bản sync để lưu dữ liệu/report/prediction history.

---

## 3. Bức tranh tổng thể dễ hiểu

Project có **4 luồng tách biệt**. Hiểu chỗ này sẽ tránh nhầm tuning, train và dự báo.

### Luồng A - Chuẩn bị dữ liệu

```text
shared_dataset/hose_stock_raw.csv
-> làm sạch OHLCV
-> tạo 20 feature
-> tạo label UP / NOT_UP
-> data/processed/ml_dataset.csv
-> chia TRAIN / TEST theo label_end_date
```

### Luồng B - Tuning cấu hình, chỉ chấm trên TRAIN

```text
Chọn một bộ hyperparameter
-> TimeSeriesSplit trên TRAIN
-> CV F1_UP
-> experiments/tuning_history.csv
-> người dùng chốt cấu hình vào manual_config.json
```

### Luồng C - Official pipeline huấn luyện và đánh giá

```text
Đọc 3 cấu hình đã chốt
-> tính lại CV trên TRAIN
-> fit từng model trên toàn TRAIN
-> đánh giá các model trên TEST
-> chọn final theo TEST F1_UP
-> models/final_model.pkl + reports/* + SQLite sync
```

### Luồng D - Dự báo bằng model đã có

```text
Người dùng nhập mã
-> lấy dữ liệu sạch mới nhất của mã
-> tính 20 feature
-> load final_model.pkl + model_metadata.json
-> tính P(UP)
-> so với decision_threshold
-> trả UP / NOT_UP và có thể log SQLite
```

Luồng D **không chạy lại luồng B/C**. Vì vậy bấm dự báo trên web nhanh hơn train model và không làm thay đổi model.

---

## 4. Các thư mục và file quan trọng

| File/thư mục | Vai trò |
|---|---|
| `config/settings.py` | Cấu hình trung tâm: đường dẫn, horizon, threshold, feature list, split date, model/report paths |
| `scripts/run_pipeline.py` | Chạy toàn bộ pipeline từ raw data đến model/report/database |
| `scripts/fetch_hose_data.py` | Cập nhật thêm dữ liệu OHLCV bằng `vnstock`, nguồn `KBS` |
| `scripts/preprocess_data.py` | Chạy riêng bước làm sạch dữ liệu |
| `scripts/build_features.py` | Chạy riêng bước tạo feature, label và train/test split |
| `scripts/train_tune_models.py` | Chạy riêng bước train/tune model |
| `scripts/evaluate_models.py` | Debug bước đánh giá; đọc TEST nhưng không kiểm tra TEST lock |
| `scripts/select_final_model.py` | Debug bước evaluate/chọn final; đọc TEST nhưng không kiểm tra TEST lock |
| `scripts/predict_stock.py` | Dự báo bằng command line |
| `scripts/finetune_model.py` | File cũ, đã deprecated; code bảo dùng `train_tune_models.py` |
| `services/tuning_lab.py` | Backend Tuning Lab: validate tham số, chạy CV một bộ config, ghi lịch sử |
| `services/experiment_state.py` | Quản lý lịch sử tuning, cấu hình đã chốt, khóa (lock), fingerprint dataset |
| `services/preprocessing.py` | Làm sạch dữ liệu và lọc mã đủ điều kiện |
| `services/feature_engineering.py` | Tạo feature, tạo target, chia train/test |
| `services/model_tuning.py` | Train Dummy và train 3 model chính từ tham số đã chốt |
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

---

## 5. Dữ liệu OHLCV là gì?

Project dùng dữ liệu giá cổ phiếu **theo ngày**. OHLCV là viết tắt của:

| Cột | Nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu, ví dụ `FPT`, `HPG`, `SSI` |
| `trading_date` | Ngày giao dịch |
| `open` | Giá mở cửa trong phiên |
| `high` | Giá cao nhất trong phiên |
| `low` | Giá thấp nhất trong phiên |
| `close` | Giá đóng cửa trong phiên |
| `volume` | Khối lượng giao dịch |

Project **chỉ dùng OHLCV**. Nó không dùng tin tức, báo cáo tài chính, dữ liệu realtime, dữ liệu intraday, sentiment analysis, LSTM, Transformer hay Deep Learning.

---

## 6. Các thông số cấu hình chính

Theo `config/settings.py`:

| Thông số | Giá trị hiện tại | Nghĩa dễ hiểu |
|---|---:|---|
| `RAW_DATA_PATH` | `...\shared_dataset\hose_stock_raw.csv` | File dữ liệu thô nằm ngoài repo |
| `SPLIT_DATE` | `2025-06-30` | Mốc thời gian chia train/test |
| `PREDICTION_HORIZON` | `5` | Lấy dòng quan sát thứ 5 kế tiếp của từng mã |
| `UP_THRESHOLD` | `0.01` | Tăng hơn 1% thì tính là `UP` |
| `MIN_TRADING_DAYS` | `250` | Mã có dưới 250 phiên hợp lệ bị loại khỏi train |
| `MIN_AVERAGE_VOLUME` | `0` | Hiện không lọc theo thanh khoản trung bình |
| `CV_N_SPLITS` | `5` | Cross Validation chia train thành 5 fold |
| `CV_GAP` | `5` | Chừa khoảng cách 5 mẫu giữa train và validation |
| `TUNING_N_ITER` | `12` | Di sản cấu hình (xem mục 15) |
| `TUNING_SCORING` | `f1` | Tuning ưu tiên F1 của lớp `UP` |
| `RANDOM_STATE` | `42` | Giúp kết quả random ổn định, tái lập được |

Trong code, horizon `5` được cài bằng `shift(-5)` **riêng cho từng mã**: lấy dòng quan sát thứ 5 kế tiếp của mã đó. Với mã giao dịch đều, nó gần với 5 phiên thị trường. Với mã có dữ liệu thưa, ngừng giao dịch hoặc thiếu ngày, khoảng thời gian lịch có thể dài hơn rất nhiều.

---

## 7. Số liệu dữ liệu hiện tại

Theo `reports/pipeline_summary.json` (run mới nhất):

| Hạng mục | Giá trị |
|---|---:|
| Raw data | 552,738 dòng |
| Số mã trong raw data | 400 mã |
| Khoảng ngày raw data | 2019-08-14 đến 2026-07-10 |
| Clean data | 552,512 dòng (loại 226 dòng OHLC sai logic) |
| Mã đủ điều kiện train | 396 mã |
| Mã bị loại | 4 mã |
| Feature data | 514,808 dòng |
| ML dataset sau khi có label | 512,828 dòng |
| Tỷ lệ nhãn UP toàn bộ dataset | ≈ 37.9% |

Vì sao số dòng giảm qua từng bước:

- Raw → clean: loại 226 dòng OHLC sai logic.
- Clean all → clean đủ điều kiện train: loại 592 dòng thuộc 4 mã có dưới 250 quan sát, từ 552,512 còn 551,920 dòng.
- Clean đủ điều kiện → feature: còn 514,808 dòng. Phần giảm chủ yếu là các dòng đầu mỗi mã chưa đủ cửa sổ rolling dài nhất 50 dòng; code cũng loại mọi dòng feature còn NaN/inf.
- Feature → ML dataset: 5 dòng cuối mỗi mã chưa có dòng tương lai thứ 5 để tạo label, nên bị loại. `396 mã × 5 dòng = 1,980 dòng`, đúng bằng `514,808 - 512,828`.

4 mã bị loại vì chưa đủ 250 phiên giao dịch (theo `reports/excluded_symbols.csv`):

| Mã | Số dòng | Khoảng dữ liệu | Lý do |
|---|---:|---|---|
| `CRV` | 147 | 2025-10-10 đến 2026-07-10 | `fewer_than_250_trading_days` |
| `TCX` | 174 | 2025-10-21 đến 2026-07-10 | `fewer_than_250_trading_days` |
| `VCK` | 134 | 2025-12-16 đến 2026-07-10 | `fewer_than_250_trading_days` |
| `VPX` | 137 | 2025-12-11 đến 2026-07-10 | `fewer_than_250_trading_days` |

`reports/fetch_report.json` (lần fetch `2026-07-11`) ghi `new_rows = 3657`, tức tải được 3,657 record trước khử trùng. Sau merge và `drop_duplicates`, raw data tăng ròng 3,654 dòng: từ 549,084 lên 552,738; ngày mới nhất `2026-07-10`. Có 4 mã fetch lỗi (`BCG`, `LGC`, `TCD`, `VNE` — kiểu `RetryError`), nhưng các mã này vẫn giữ dữ liệu cũ trong CSV nếu trước đó đã tồn tại.

---

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
10. Loại dòng OHLC sai logic (ví dụ `high` nhỏ hơn `open`, `close` hoặc `low`).
11. Sắp xếp theo `symbol`, `trading_date`.
12. Thống kê mỗi mã: số dòng, ngày bắt đầu/kết thúc, volume trung bình.
13. Lọc mã đủ điều kiện train: cần tối thiểu 250 phiên giao dịch.

Output của bước này:

```text
data/processed/hose_stock_clean.csv
reports/data_quality_report.csv
reports/eligible_symbols.csv
reports/excluded_symbols.csv
```

---

## 9. Feature là gì?

Feature là **dữ liệu đầu vào** cho model.

Con người nhìn biểu đồ, đường trung bình, volume để đoán xu hướng. Model không nhìn biểu đồ trực tiếp, nên project biến lịch sử giá/volume thành các cột số. Các cột số đó gọi là feature.

Project dùng **20 feature** (theo `config/settings.py::FEATURE_COLUMNS`):

Các hậu tố `5d/10d/20d/50` trong code thực tế đếm số **dòng có dữ liệu của từng mã**. Tài liệu đôi lúc gọi gọn là phiên; với mã dữ liệu thưa, chúng không tương đương số ngày thị trường liên tiếp.

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
| `return_10d` | Giá thay đổi bao nhiêu trong 10 phiên gần nhất — động lượng trung hạn |
| `return_20d` | Giá thay đổi bao nhiêu trong 20 phiên gần nhất — động lượng dài hơn |
| `dist_high20` | Giá hiện tại cách đỉnh cao nhất 20 phiên bao nhiêu: `(close / max_high_20) - 1` (thường ≤ 0) |
| `dist_low20` | Giá hiện tại cách đáy thấp nhất 20 phiên bao nhiêu: `(close / min_low_20) - 1` (thường ≥ 0) |
| `month` | Tháng của phiên giao dịch (1–12) — nắm bắt yếu tố mùa vụ theo tháng |

Các feature được tính **riêng cho từng `symbol`**. Lịch sử của `FPT` không được trộn sang `HPG`, `SSI` hay mã khác (xem `services/feature_engineering.py`, dùng `groupby("symbol")`).

---

## 10. Một vài feature quan trọng nên hiểu kỹ

### Return

`return_1d` được tính:

```text
return_1d = close(t) / close(t-1) - 1
```

Ví dụ: `0.02` = tăng 2%; `-0.03` = giảm 3%.

### SMA (Simple Moving Average — trung bình trượt đơn giản)

- `sma5`: trung bình 5 phiên, phản ánh ngắn hạn.
- `sma20`: trung bình 20 phiên, trung hạn.
- `sma50`: trung bình 50 phiên, dài hơn.

Nếu giá hiện tại cao hơn SMA20, có thể hiểu là giá đang nằm trên mức trung bình 20 phiên.

### RSI (Relative Strength Index)

Chỉ báo sức mạnh tương đối, thường nằm từ 0 đến 100.

- Gần 70 trở lên: giá đã tăng mạnh trong giai đoạn gần đây.
- Gần 30 trở xuống: giá đã giảm mạnh.
- Gần 50: cân bằng hơn.

Trong code, nếu cả gain và loss đều bằng 0 thì RSI được gán 50; nếu loss bằng 0 thì RSI là 100.

### Volatility (độ biến động)

Giá dao động càng mạnh thì volatility càng cao. Cổ phiếu biến động cao thường khó dự báo hơn.

### Volume ratio

```text
volume_ratio_20 = volume hôm nay / volume trung bình 20 phiên
```

Ví dụ `volume_ratio_20 = 2` = volume hôm nay gấp đôi trung bình 20 phiên.

### Nhóm feature vị trí giá

- `return_10d`, `return_20d`: giống `return_5d` nhưng nhìn xa hơn — cho model thêm góc nhìn xu hướng trung hạn.
- `dist_high20 = close / max(high, 20 phiên) - 1`: giá hiện tại cách **đỉnh** cao nhất 20 phiên bao nhiêu. Bằng 0 = đang ở đỉnh; âm nhiều = đã rơi khá xa khỏi đỉnh.
- `dist_low20 = close / min(low, 20 phiên) - 1`: giá hiện tại cách **đáy** thấp nhất 20 phiên bao nhiêu. Bằng 0 = đang ở đáy; dương nhiều = đã bật lên khá xa khỏi đáy.
- `month`: tháng của phiên giao dịch (1–12). Feature lịch, để model bắt yếu tố mùa vụ.

`dist_high20` / `dist_low20` mô tả **vị trí giá trong biên độ 20 phiên** — thông tin mà `close_vs_sma20` (so với trung bình) chưa nắm được.

---

## 11. Target, label, horizon, threshold là gì?

Target là **đáp án** mà model cần học.

Với mỗi mã tại dòng/ngày `t`, project nhìn lại các dòng quá khứ để tạo feature, rồi lấy dòng thứ 5 kế tiếp để tạo đáp án:

```text
future_close_5d  = close(t+5)
future_return_5d = close(t+5) / close(t) - 1
```

Sau đó tạo nhãn:

```text
target = 1 (UP)     nếu future_return_5d > 0.01
target = 0 (NOT_UP) nếu future_return_5d <= 0.01
```

Ví dụ:

| Giá tại t | Giá ở dòng thứ 5 kế tiếp | Return | Label |
|---:|---:|---:|---|
| 100 | 103 | 3.0% | `UP` |
| 100 | 101.1 | 1.1% | `UP` |
| 100 | 100.8 | 0.8% | `NOT_UP` |
| 100 | 99 | -1.0% | `NOT_UP` |

`future_return_5d` chỉ dùng để tạo đáp án trong lúc train/test. **Khi dự báo thật, model không được biết tương lai.** Số `5` ở đây là dòng quan sát thứ 5 tiếp theo của từng `symbol`, không đảm bảo đúng 5 ngày thị trường nếu mã có dữ liệu thưa.

### Hai threshold hoàn toàn khác nhau

| Threshold | Giá trị hiện tại | Dùng lúc nào? |
|---|---:|---|
| `UP_THRESHOLD` | `0.01` (1%) | Tạo **đáp án thật**: return tương lai >1% thì target=`UP` |
| `decision_threshold` | `≈0.4129` với final GB | Đổi **xác suất model** thành dự báo: `P(UP) >= 0.4129` thì prediction=`UP` |

Ví dụ: giá từ 100 lên 102 ở dòng thứ 5 kế tiếp tạo target `UP` vì tăng 2% > 1%. Khi dùng model thật, model có thể trả `P(UP)=0.45`; vì `0.45 >= 0.4129`, kết quả dự báo cũng là `UP`. Hai phép so này diễn ra ở hai thời điểm khác nhau và không được gọi lẫn nhau.

---

## 12. Data leakage là gì?

Data leakage là lỗi rất nguy hiểm: model **vô tình nhìn thấy thông tin tương lai** hoặc thông tin đáp án trong lúc train.

Ví dụ sai: dùng `close(t+5)` làm feature để dự báo target tại ngày `t`. Nếu làm vậy, model gần như được nhìn thấy đáp án trước → kết quả test đẹp giả, nhưng dùng thật sẽ tệ.

Project có các biện pháp **giảm** leakage:

- Feature tại ngày `t` chỉ dùng dữ liệu từ ngày `t` trở về trước.
- `future_close_5d`, `future_return_5d`, `target`, `label_end_date` **không** nằm trong `FEATURE_COLUMNS` (hàm `verify_data_pipeline` kiểm tra và ném lỗi nếu bị lẫn).
- Train/test chia theo **thời gian**, không random shuffle.
- Train dùng `label_end_date <= 2025-06-30`; test dùng `label_end_date > 2025-06-30`.
- Cross Validation dùng `TimeSeriesSplit(gap=5)`.
- Logistic Regression dùng `Pipeline(StandardScaler, LogisticRegression)`, giúp scaler fit đúng trong từng fold.

Các biện pháp này chưa loại bỏ leakage hoàn toàn. CV hiện chạy trên bảng gộp nhiều mã và `gap=5` chỉ là 5 dòng; mục 14 giải thích giới hạn cụ thể.

---

## 13. Train/test split hiện tại

Project **không** dùng random split. Code dùng `label_end_date` để bảo đảm nhãn của TRAIN kết thúc không muộn hơn split date.

Theo `reports/train_test_summary.csv`:

| Tập | Dòng | Số mã | Khoảng `trading_date` | Label boundary | UP | NOT_UP |
|---|---:|---:|---|---|---:|---:|
| Train | 417,806 | 395 | 2019-10-23 đến 2025-06-23 | `label_end_date <= 2025-06-30` | 163,278 | 254,528 |
| Test | 95,022 | 396 | 2025-03-03 đến 2026-07-03 | `label_end_date > 2025-06-30` | 31,163 | 63,859 |

Suy ra tỷ lệ `UP`: TRAIN ≈ 39.1% (163278/417806), TEST ≈ 32.8% (31163/95022). Phân phối TEST có ít nhãn UP hơn TRAIN; đây là distribution shift cần lưu ý, dù hai tập chưa tách tuyệt đối theo `trading_date`.

Theo `reports/pipeline_summary.json`: `overlap_rows = 0` chỉ có nghĩa không dùng lại cùng index dòng; `train_label_end_max = 2025-06-30`, `test_label_end_min = 2025-07-01` cho thấy ranh giới **kết thúc nhãn** tách nhau. Nó không chứng minh hai tập không chồng khoảng `trading_date`.

**Vì sao TEST có `trading_date` bắt đầu từ 2025-03-03, trước split date 2025-06-30?**

Project chia theo `label_end_date`, không chia trực tiếp theo `trading_date`. Ví dụ mã `TTE` có dòng tham chiếu `2025-03-03`, nhưng vì dữ liệu mã này rất thưa nên dòng thứ 5 kế tiếp tận `2025-08-25`; dòng đó thuộc TEST. Đây không phải 5 phiên liên tiếp của toàn thị trường.

**Giới hạn cần nói thật:** split hiện bảo vệ ranh giới kết thúc nhãn (`TRAIN label_end_date <= 2025-06-30`, TEST ngược lại), nhưng không bảo đảm mọi ngày tham chiếu của TEST đều sau mọi ngày tham chiếu của TRAIN. Dataset hiện có 89 dòng TEST thuộc 42 mã với `trading_date <=` ngày TRAIN lớn nhất. Vì vậy không nên mô tả split hiện tại là tách thời gian tuyệt đối hoặc hoàn toàn không leakage.

---

## 14. Cross Validation là gì?

Cross Validation là cách chia tập train thành nhiều phần nhỏ để kiểm tra model có ổn định không.

Với dữ liệu bình thường có thể chia ngẫu nhiên. Nhưng với dữ liệu cổ phiếu, chia ngẫu nhiên dễ làm tương lai lọt vào quá khứ.

Project dùng:

```text
TimeSeriesSplit(n_splits=5, gap=5)
```

Đọc metric nhanh trước khi đi tiếp: Precision_UP hỏi "đã báo UP thì đúng bao nhiêu", Recall_UP hỏi "UP thật thì bắt được bao nhiêu", còn F1_UP cân bằng hai chỉ số đó. Mục 17 giải thích chi tiết.

Ý nghĩa:

- Chỉ dùng trên tập train.
- Chia train thành 5 fold theo thứ tự thời gian.
- Fold sau có nhiều dữ liệu quá khứ hơn fold trước.
- `gap=5` bỏ qua 5 **dòng của bảng gộp**, không phải 5 ngày, không phải 5 phiên cho từng mã.

Trong mỗi fold, code tinh chỉnh **decision threshold** trên prediction của chính TRAIN fold, rồi áp ngưỡng đó lên VAL để đo F1. Threshold không dùng trực tiếp nhãn VAL/TEST, nhưng vẫn có thể khớp quá sát TRAIN vì chưa có một phần dữ liệu riêng để chọn ngưỡng. Chi tiết ở mục 16.1.

**Giới hạn quan trọng:** vì bảng gộp hàng trăm mã rồi chia theo vị trí dòng, `gap=5` không tạo khoảng cách 5 phiên. Fold có thể chung ngày và label window của TRAIN có thể lấn vào VAL, nên CV F1_UP có thể lạc quan. Bằng chứng và hướng sửa kỹ thuật nằm ở mục 29.

---

## 15. Hyperparameter tuning là gì?

Model có 2 loại tham số:

| Loại | Nghĩa |
|---|---|
| Parameter | Thứ model **tự học** từ dữ liệu |
| Hyperparameter | Thứ người lập trình **cấu hình trước** khi train |

Ví dụ với Random Forest: `n_estimators` (số cây), `max_depth` (độ sâu tối đa), `min_samples_leaf` (số mẫu tối thiểu ở một lá), `max_features` (số feature xét mỗi lần chia nhánh).

Tuning là quá trình thử nhiều bộ hyperparameter để tìm bộ tốt hơn.

**Cách tuning/chốt cấu hình hiện tại:** official pipeline **không** chạy `RandomizedSearchCV`. Tuning Lab cho phép đánh giá từng cấu hình bằng cùng một hàm CV:

1. Vào trang `/tuning`, nhập bộ hyperparameter cho từng model (LR, RF, GB).
2. Bấm chạy → `services/tuning_lab.py::evaluate_single_config` chỉ nhận tập TRAIN. Hàm chia TRAIN thành 5 fold thời gian; mỗi fold học trên phần `fold-train`, đo F1_UP trên phần `fold-validation`, rồi lấy trung bình 5 fold thành **CV F1_UP** và ghi một dòng vào `experiments/tuning_history.csv`.
3. Khi thấy bộ nào tốt, bấm "dùng cấu hình này" → lưu vào `experiments/manual_config.json`, kèm dấu vân dữ liệu (`dataset_fingerprint`).
4. Khi đã chốt đủ cả 3 model (và fingerprint khớp dataset hiện tại), pipeline chính thức mới được phép chạy. Lúc chạy, `services/model_tuning.py::tune_models` **đọc lại các bộ tham số đã chốt** từ `manual_config.json` để huấn luyện, chứ không tự dò tham số.

Repo còn có `experiments/lr_2h_search.py`, `rf_2h_auto_search.py`, `gb_3h_ext_search.py` để tự động gọi cùng `evaluate_single_config` với nhiều cấu hình. Vì vậy **khâu thử nghiệm không hoàn toàn làm tay**; phần thủ công quan trọng là người dùng xem kết quả và bấm chọn cấu hình cuối. Code cho phép chọn bất kỳ run hợp lệ, không tự bắt buộc chọn điểm cao nhất.

> Lưu ý về CV score: khi chạy pipeline, code **luôn tính lại CV trên TRAIN hiện tại** (`run_cv_metrics`) chứ không tin con số CV đã lưu trong `manual_config.json` — con số trong config chỉ để lưu vết (provenance).

### Hai loại F1 rất dễ nhầm trong project

Hãy hình dung:

- **TRAIN** là bộ tài liệu dùng để học và làm bài kiểm tra thử.
- Các phần **validation bên trong TRAIN** là 5 bài kiểm tra thử của Cross Validation.
- **TEST** là tập đánh giá/chọn final model, được giữ khỏi quá trình chọn hyperparameter.

Hai luồng cần tách riêng:

```text
TUNING / THỬ CẤU HÌNH
TRAIN -> 5-fold CV -> CV F1_UP
      -> tuning_history.csv -> người dùng chốt params

OFFICIAL PIPELINE
params đã chốt -> tính lại CV trên TRAIN
                -> fit model trên toàn TRAIN
                -> predict TEST -> TEST F1_UP
                -> so sánh 3 model -> final_model.pkl
```

Điểm cần hiểu: `.fit()` chỉ làm model học, **chưa tự sinh ra F1**. Muốn có F1 phải lấy dự đoán so với nhãn thật:

- Trong Tuning Lab, nhãn thật dùng để chấm nằm ở các `fold-validation` thuộc TRAIN → kết quả là **CV F1_UP**.
- Trong bước evaluate chính thức, nhãn thật dùng để chấm nằm ở TEST → kết quả là **TEST F1_UP**.

Hai metric cùng dùng công thức F1 cho lớp UP, nhưng trả lời hai câu hỏi khác nhau:

| Metric | Dữ liệu chấm điểm | Dùng để làm gì? | Nằm ở đâu? |
|---|---|---|---|
| `CV F1_UP` | 5 validation fold bên trong TRAIN | So sánh các bộ hyperparameter, lưu lịch sử tuning | `tuning_history.csv::cv_f1_up_mean` |
| `TEST F1_UP` | Tập TEST tách riêng | So sánh 3 model đã chốt và chọn final model | `model_comparison.csv::f1_up` |

Ví dụ số hiện tại cũng cho thấy hai loại này khác nhau:

| Model | CV F1_UP lúc chọn trong Tuning Lab | CV F1_UP pipeline tính lại | TEST F1_UP |
|---|---:|---:|---:|
| Logistic Regression | 0.5490 | 0.5490 | 0.5024 |
| Random Forest | 0.5491 | 0.5488 | 0.5041 |
| Gradient Boosting | 0.5453 | 0.5453 | 0.5053 |

Hai cột CV đều đo trên TRAIN nhưng đến từ **hai lần chạy khác nhau**: lần người dùng thử/chọn cấu hình trong Tuning Lab và lần pipeline chính thức tính lại CV. Vì vậy khi trích số phải nói rõ nguồn. Đặc biệt, `0.5491` là CV F1_UP của run Random Forest được lưu/chọn trong history; `0.5488` là CV F1_UP do pipeline tính lại, không phải TEST F1_UP.

**Kết luận cho 3 lưu đồ tuning của LR, RF và GB:** lưu đồ dừng tại `CV F1_UP -> tuning_history.csv`. Các bước fit cuối trên toàn TRAIN, evaluate TEST, so sánh model và chọn final model nằm ngoài 3 lưu đồ đó.

`config/settings.py` vẫn còn `TUNING_N_ITER=12` và `TUNING_SCORING="f1"` như di sản cấu hình, nhưng luồng chốt tham số hiện do người dùng điều khiển qua Tuning Lab.

Ý nghĩa: mỗi dòng `tuning_history.csv` lưu config, điểm, fold, thời gian và fingerprint nên dễ truy vết. Muốn tái lập đúng còn phải giữ nguyên nội dung data, code và thứ tự dòng; fingerprint hiện không hash toàn bộ nội dung.

---

## 16. Các model trong project

Project train 4 model:

| Model | Vai trò |
|---|---|
| Dummy Classifier | Baseline tối thiểu, chỉ đoán lớp phổ biến nhất |
| Logistic Regression | Model tuyến tính đơn giản, dễ giải thích |
| Random Forest | Nhiều cây quyết định cùng bỏ phiếu, hợp dữ liệu bảng |
| Gradient Boosting | Nhiều cây học tuần tự, cây sau cố sửa lỗi cây trước |

Chi tiết trong code (`services/model_tuning.py`):

- Dummy dùng `strategy="most_frequent"` (threshold cố định 0.5).
- Logistic Regression nằm trong `Pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))`.
- Random Forest dùng `class_weight="balanced_subsample"`, `n_jobs=-1`.
- Gradient Boosting dùng `GradientBoostingClassifier`. Class này **không có** tham số `class_weight`, nên code truyền `sample_weight = compute_sample_weight("balanced", y)` vào lúc `.fit()` (cả trong CV lẫn khi fit cuối) để cân bằng lớp tương đương.

**Xử lý mất cân bằng lớp khác nhau theo model:** lớp UP ít hơn NOT_UP, nên mỗi model đều được cân bằng, nhưng bằng cơ chế khác nhau (LR/RF qua `class_weight`, GB qua `sample_weight`). Nếu không cân bằng, model dễ đoán toàn NOT_UP.

**Bộ tham số đã chốt** (theo `reports/best_params.json` và `experiments/manual_config.json`, fingerprint `f6cb3ac8f820`):

| Model | Params đã chốt (20 feature) |
|---|---|
| Logistic Regression | `C=6.59e-05`, `solver=liblinear` |
| Random Forest | `n_estimators=130`, `max_depth=4`, `min_samples_leaf=25`, `max_features=0.35` |
| Gradient Boosting | `n_estimators=120`, `learning_rate=0.05`, `max_depth=2`, `subsample=1.0` |

---

## 16.1. Decision threshold (ngưỡng quyết định) là gì?

Mặc định, model phân loại đoán UP khi `P(UP) >= 0.5`. Project không giữ cứng mốc này mà **tìm ngưỡng làm F1_UP trên TRAIN cao nhất** cho từng model:

1. Trong mỗi fold CV, sau khi fit trên phần TRAIN của fold, code quét `precision_recall_curve` để tìm ngưỡng cho **F1 lớp UP cao nhất** (`services/model_tuning.py::tune_threshold`), giới hạn trong khoảng `[0.05, 0.95]`.
2. Ngưỡng đó được chọn từ prediction trên chính TRAIN fold, rồi áp lên VAL để đo. Nó không dùng trực tiếp nhãn VAL/TEST, nhưng vẫn là tuning in-sample và có thể overfit threshold.
3. Khi fit model cuối trên toàn bộ TRAIN, ngưỡng tối ưu được tính lại và **lưu vào artifact + `model_metadata.json`** (`decision_threshold`).
4. Khi dự báo thật, model so `P(UP)` với ngưỡng đã lưu này, **không dùng 0.5**.

Ngưỡng của final model **Gradient Boosting** đang phục vụ là `decision_threshold ≈ 0.4129`. Vì thấp hơn 0.5, model gán UP nhiều hơn; trên TEST hiện tại kết quả quan sát được là Recall_UP cao.

**Hệ quả quan sát được trên TEST:** Recall_UP tăng mạnh nhưng Precision_UP và Accuracy thấp vì báo UP nhầm nhiều. Code trực tiếp tối ưu F1_UP, **không đặt mục tiêu Recall riêng**.

---

## 16.2. Tại sao bộ tham số đã chốt là "tốt nhất"?

Đây là câu hỏi hay gặp khi bảo vệ: *"Sao biết params ở mục 16 là tốt nhất?"* Trả lời trung thực gồm 2 phần: (1) tốt nhất theo **tiêu chí nào**, và (2) tốt nhất trong **phạm vi nào**.

### Tiêu chí: CV F1_UP trung bình 5 fold

Mỗi lần thử một bộ tham số, code chạy `TimeSeriesSplit(n_splits=5, gap=5)` trên TRAIN, tính F1 lớp UP ở từng fold rồi lấy **trung bình** (`cv_f1_up_mean`). Con số này được ghi vào `experiments/tuning_history.csv`. UI có thể đánh dấu run cao nhất, hòa thì ưu tiên `std` thấp hơn; tuy nhiên **người dùng mới là người chốt** run vào `manual_config.json`.

Quan trọng: khâu chọn hyperparameter không đọc TEST. Sau khi chốt cấu hình, official pipeline dùng TEST để **vừa báo cáo metric, vừa chọn final model trong 3 model chính**. Vì vậy TEST không tham gia hyperparameter tuning, nhưng cũng không còn là holdout hoàn toàn độc lập sau bước chọn model.

### Bằng chứng: các bộ đã thử và điểm số

Trích từ `experiments/tuning_history.csv` với fingerprint hiện tại `f6cb3ac8f820`: có 177 run Random Forest, 7 run Gradient Boosting và chỉ 1 run Logistic Regression. Dòng **in đậm** là bộ người dùng đã chốt; không có nghĩa code tự chọn nó.

**Random Forest** (đã thử 177 bộ) — top theo CV F1_UP:

| n_estimators | max_depth | min_samples_leaf | max_features | CV F1_UP | std |
|---:|---:|---:|---:|---:|---:|
| **130** | **4** | **25** | **0.35** | **0.5491** | **0.0205** |
| 130 | 4 | 25 | 0.37 | 0.5491 | 0.0205 |
| 130 | 4 | 32 | 0.36 | 0.5491 | 0.0206 |
| 130 | 4 | 20 | 0.35 | 0.5491 | 0.0206 |

**Gradient Boosting** (đã thử 7 bộ) — top theo CV F1_UP:

| n_estimators | learning_rate | max_depth | subsample | CV F1_UP | std |
|---:|---:|---:|---:|---:|---:|
| **120** | **0.05** | **2** | **1.0** | **0.5453** | **0.0200** |
| 150 | 0.05 | 3 | 0.5 | 0.5449 | 0.0220 |
| 80 | 0.1 | 3 | 0.5 | 0.5444 | 0.0218 |
| 60 | 0.3 | 2 | 0.5 | 0.5427 | 0.0218 |

**Logistic Regression**: bộ chốt `C=6.59e-05`, `solver=liblinear` được chạy lại trên fingerprint hiện tại và cho CV F1_UP ≈ 0.5490. History hiện tại chỉ có một run LR, nên không đủ bằng chứng để gọi đây là C cao điểm nhất trên chính dataset hiện tại; các lần quét nhiều C nằm chủ yếu ở fingerprint/log thí nghiệm cũ.

Điểm cần thấy: với RF nhiều bộ gần như hòa nhau (0.5491) — chênh lệch nằm ở số lẻ thứ 4. Nghĩa là quanh vùng `max_depth=4`, `min_samples_leaf≈20–32`, `max_features≈0.35–0.37`, model đã "bão hòa": chỉnh thêm gần như không cải thiện. Bộ được chốt chỉ là một đại diện tốt của vùng ổn định đó, **không** phải điểm duy nhất đúng.

### Ý nghĩa từng giá trị được chốt (vì sao chúng hợp lý)

| Giá trị | Ý nghĩa & vì sao hợp lý với bài toán này |
|---|---|
| RF `max_depth=4` (nông) | Cây nông = model đơn giản, **chống overfit**. Dữ liệu OHLCV nhiễu mạnh, cây sâu dễ học thuộc nhiễu quá khứ. |
| RF `min_samples_leaf=25` | Mỗi lá cần ≥25 mẫu → không tách nhánh theo vài điểm cá biệt → ổn định hơn. |
| RF `max_features=0.35` | Mỗi lần chia chỉ xét 35% feature → các cây khác nhau hơn → rừng đa dạng, giảm phương sai. |
| GB `learning_rate=0.05` (nhỏ) | Học chậm, từng bước nhỏ → tổng quát tốt hơn, ít overfit hơn learning_rate lớn. |
| GB `max_depth=2` (rất nông) | Mỗi cây cơ sở chỉ học quan hệ nông; boosting cộng dồn nhiều cây để tăng sức biểu diễn. `max_depth=2` không phải decision stump (stump thường depth 1) và không bảo đảm tuyệt đối không overfit. |
| LR `C=6.59e-05` (rất nhỏ) | `C` nhỏ = **regularization rất mạnh** → ép hệ số về gần 0, model rất "thận trọng". Trên dữ liệu nhiễu, điều này lại cho F1_UP cao và ổn định. |

Mẫu số chung: cả 3 model đều được chốt về phía **đơn giản / regularization mạnh**. Đây là lựa chọn hợp lý cho dữ liệu cổ phiếu nhiễu, nơi model phức tạp dễ học thuộc quá khứ mà kém khi gặp tương lai.

### Phạm vi: "tốt nhất trong số đã thử", không phải tối ưu tuyệt đối

Cần nói thẳng khi bảo vệ: project **không** dùng `RandomizedSearchCV`, nhưng có script tự động quét các danh sách cấu hình do người làm định trước. "Tốt nhất" chỉ nên hiểu là cấu hình người dùng đã chọn trong phạm vi run đã thử và lưu vết, không phải tối ưu toàn cục. Với fingerprint hiện tại, bằng chứng mạnh nhất thuộc RF/GB; LR chỉ có một run trong `tuning_history.csv` hiện tại.

---

## 17. Các metric đánh giá là gì?

### Accuracy

```text
accuracy = số dự báo đúng / tổng số mẫu
```

Với project này, accuracy **không đủ tốt** để chọn model, vì lớp `NOT_UP` nhiều hơn `UP`. Một model cứ đoán `NOT_UP` nhiều có thể accuracy khá cao nhưng không bắt được cổ phiếu tăng.

### Precision_UP

> Trong những lần model dự báo UP, bao nhiêu lần **thật sự** UP?

Precision cao = khi model nói `UP`, nó ít báo động nhầm hơn.

### Recall_UP

> Trong tất cả trường hợp thật sự UP, model **bắt được** bao nhiêu?

Recall cao = model ít bỏ sót trường hợp tăng.

### F1_UP

F1_UP là chỉ số cân bằng giữa Precision_UP và Recall_UP. Project chọn final model theo `F1_UP`, vì mục tiêu chính là học lớp `UP`.

### Confusion Matrix

Bảng đếm đúng/sai: thật NOT_UP đoán NOT_UP (đúng), thật NOT_UP đoán UP (false positive), thật UP đoán NOT_UP (false negative), thật UP đoán UP (đúng).

---

## 18. Kết quả model (run mới nhất, 20 feature)

Theo `reports/model_comparison.csv`:

| Model | Accuracy | Precision_UP | Recall_UP | F1_UP | CV F1_UP | Chọn |
|---|---:|---:|---:|---:|---:|---|
| Dummy Classifier | 0.6720 | 0.0000 | 0.0000 | 0.0000 | - | Không |
| Logistic Regression | 0.3938 | 0.3437 | 0.9332 | 0.5024 | 0.5490 | Không |
| Random Forest | 0.3918 | 0.3440 | 0.9424 | 0.5041 | 0.5488 | Không |
| **Gradient Boosting** | **0.4108** | **0.3487** | **0.9176** | **0.5053** | **0.5453** | **Có** |

Final model hiện tại:

```text
models/final_model.pkl = Gradient Boosting
```

**Nhận xét chung:** cả 3 model chính có **Recall_UP rất cao (0.92–0.94)** nhưng **Accuracy thấp (0.39–0.41)** và Precision_UP quanh 0.34. Decision threshold thấp là nguyên nhân lớn: model gán UP rất "hào phóng" nên bắt gần hết UP thật nhưng báo UP nhầm nhiều. Tuy nhiên không thể dùng threshold để kết luận model chắc chắn tốt; TEST F1_UP chỉ quanh 0.50, nên hiệu năng vẫn khiêm tốn.

**Vì sao Dummy không chọn?**
Dummy đoán lớp phổ biến nhất là `NOT_UP`. Accuracy 0.6720 cao nhất bảng nhưng `F1_UP = 0` — không bắt được UP nào. `select_final_model` loại Dummy khỏi vòng chọn.

Dummy hiện tại cũng là baseline khá yếu cho metric F1_UP. Nếu có baseline đơn giản **luôn đoán UP**, từ tỷ lệ lớp TEST hiện tại ta tính được F1_UP ≈ `0.4939`; final GB đạt `0.5053`, chỉ hơn khoảng `0.0114`. Code chưa xuất baseline luôn-UP này, nhưng phép so cho thấy lợi thế của final model còn nhỏ.

**Vì sao Gradient Boosting được chọn?**
Quy tắc chọn (`services/model_evaluation.py::select_final_model`) xếp hạng theo thứ tự: **F1_UP giảm dần → Recall_UP giảm dần → độ đơn giản tăng dần** (LR đơn giản hơn RF, RF đơn giản hơn GB). Ba model chính rất sát nhau về F1_UP (GB 0.5053, RF 0.5041, LR 0.5024), và Gradient Boosting có **F1_UP cao nhất** nên thắng — dù nó là model "phức tạp nhất" trong tie-break.

**Điểm cần trung thực khi trình bày:** khoảng cách F1_UP giữa 3 model **rất nhỏ (chênh ~0.003)**, nên "Gradient Boosting tốt nhất" chỉ đúng ở mức sát sao trên tập TEST này, không phải vượt trội. Về CV F1_UP thì Logistic Regression nhỉnh nhất; GB chỉ thắng TEST F1_UP. Vì chính TEST F1_UP được dùng để chọn GB, TEST metric này có selection bias và không còn là ước lượng hoàn toàn độc lập cho model sau chọn. Thiết kế chặt hơn sẽ chọn model family bằng CV/validation riêng, rồi chỉ đánh giá một model đã khóa trên TEST.

---

## 19. Confusion matrix (run mới nhất)

Theo `reports/confusion_matrix.csv`:

| Thực tế / Dự báo | Dự báo NOT_UP | Dự báo UP |
|---|---:|---:|
| Thực tế NOT_UP | 10,440 | 53,419 |
| Thực tế UP | 2,568 | 28,595 |

Cách đọc:

- 10,440: thực tế `NOT_UP`, đoán `NOT_UP`, **đúng**.
- 53,419: thực tế `NOT_UP`, đoán `UP`, **sai** (false positive — rất nhiều).
- 2,568: thực tế `UP`, đoán `NOT_UP`, **sai** (false negative — ít).
- 28,595: thực tế `UP`, đoán `UP`, **đúng**.

Tổng test:

```text
10,440 + 53,419 + 2,568 + 28,595 = 95,022 dòng
```

Ma trận này cho thấy cột "Dự báo UP" chiếm đa số. Model bắt được 28,595/31,163 UP thật (Recall_UP ≈ 0.918 — rất cao) nhưng báo UP nhầm 53,419 lần trên nền NOT_UP. Đây là kết quả quan sát được khi ngưỡng tối ưu F1 trên TRAIN thấp hơn 0.5, không phải bằng chứng code trực tiếp tối ưu Recall.

Kết quả này **không phải "siêu chính xác"**. Nó phản ánh bài toán dự báo cổ phiếu bằng OHLCV là khó. Project có pipeline đầy đủ, report, baseline và cơ chế khóa TEST; đồng thời vẫn còn các giới hạn ở cách chia CV, baseline, tuning threshold và việc dùng TEST để chọn final model. Các giới hạn này được tổng hợp ở mục 29.

---

## 20. Feature importance hiện tại

Final model là Gradient Boosting, nên project xuất được `reports/feature_importance.csv`.

Top feature quan trọng (run mới nhất):

| Hạng | Feature | Importance |
|---:|---|---:|
| 1 | `volatility_20d` | 0.4236 |
| 2 | `month` | 0.1796 |
| 3 | `return_20d` | 0.0763 |
| 4 | `return_1d` | 0.0713 |
| 5 | `volume_ratio_20` | 0.0561 |
| 6 | `return_3d` | 0.0493 |
| 7 | `volume_change_1d` | 0.0323 |
| 8 | `close_open_return` | 0.0257 |

Cách hiểu: trong model đã fit này, `volatility_20d` chiếm ~42% importance và `month` ~18%, sau đó là return dài/ngắn hạn và volume. Đa số SMA có importance thấp, đặc biệt `sma50`, `sma5`; `rsi14` thấp nhất. `sma20` đứng khoảng giữa bảng (hạng 11/20), nên không nên gom toàn bộ SMA vào nhóm thấp nhất hoặc tự suy ra nguyên nhân nếu chưa có thí nghiệm ablation.

**Lưu ý quan trọng:** feature importance **không** chứng minh nguyên nhân thị trường. Nó chỉ nói model đã dựa vào feature đó nhiều trong quá trình ra quyết định.

---

## 21. Database hiện tại

Roadmap đề xuất SQLite và code hiện tại có:

```text
database/stock_prediction.db
```

`database/init_db.sql` mô tả schema khởi tạo/dự kiến. Các bảng nghiệp vụ:

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
| Raw rows | 552,738 |
| Clean rows | 552,512 |
| Feature rows | 514,808 |
| Tuning rows ghi thêm lần này | 3 |
| Evaluation rows ghi thêm lần này | 4 |

**Cơ chế sync** (`services/database_service.py`): `raw_prices`, `clean_prices`, `features` ghi kiểu **replace**; `tuning_results` và `model_evaluations` ghi kiểu **append**. Vì thế số `3/4` trên là số dòng ghi thêm của run mới nhất, không phải tổng số dòng đang có trong DB. DB hiện có 24 tuning rows, 44 evaluation rows và 13 prediction rows.

`pandas.to_sql(if_exists="replace")` sẽ drop/recreate ba bảng raw/clean/features, nên schema runtime của chúng có thể mất các ràng buộc `id`, PK, UNIQUE, NOT NULL đã khai báo trong `init_db.sql`. Do đó không nên coi `init_db.sql` là mô tả tuyệt đối của DB sau sync. Best params chi tiết vẫn nên xem ở `reports/best_params.json` và `models/model_metadata.json`.

---

## 22. Web demo hoạt động thế nào?

Khi chạy `python app.py`, web mở tại `http://127.0.0.1:5000`.

Luồng khi nhập mã, ví dụ `FPT` (theo `services/prediction_service.py`):

1. Web nhận mã cổ phiếu từ form.
2. Chuẩn hóa mã thành chữ hoa.
3. Đọc dữ liệu sạch (`hose_stock_clean.csv`).
4. Lọc dữ liệu của mã đó.
5. Tính lại feature bằng `build_features`.
6. Lấy dòng feature có ngày mới nhất.
7. Load `models/final_model.pkl`.
8. Load `models/model_metadata.json`.
9. Lấy đúng thứ tự feature từ metadata.
10. Gọi `predict_proba` để lấy xác suất lớp `UP` (`P(UP)`).
11. So `P(UP)` với `decision_threshold` lấy từ metadata (≈0.4129): nếu `P(UP) >= ngưỡng` thì gán `UP`, ngược lại `NOT_UP`. **Không dùng mốc 0.5 mặc định.**
12. Ghi lịch sử dự báo vào SQLite (best-effort, lỗi được bỏ qua để không hỏng kết quả).
13. Render kết quả trên HTML.

Ngoài trang dự báo (`/`), web còn có:

- `/evaluation`: bảng so sánh model + confusion matrix.
- `/tuning`: Tuning Lab để thử/chốt tham số và chạy pipeline.
- `/tuning/fetch-data` + `/tuning/fetch-status`: làm mới dữ liệu (fetch OHLCV → preprocess → build features) và theo dõi tiến trình.

Web **không** train lại model khi dự báo, **không** realtime. Nó dùng dữ liệu offline đã xử lý.

---

## 23. Xác suất lớp UP nghĩa là gì?

Nếu web hiện `Xác suất lớp UP: 51.2%`, **không** nên gọi đây là "độ tin cậy tuyệt đối". Nên gọi là:

```text
Xác suất dự báo của model cho lớp UP.
```

Với Gradient Boosting, xác suất này đến từ tổng hợp các cây học tuần tự trong model.

Project chưa chạy bước **probability calibration** như calibration curve, Brier score hoặc `CalibratedClassifierCV`. Vì vậy `P(UP)=51.2%` chỉ là output `predict_proba` để xếp hạng/so threshold; chưa thể diễn giải rằng trong thực tế chắc chắn có đúng 51.2% trường hợp sẽ UP.

---

## 24. Các lệnh chạy thường dùng

### Chạy demo bằng artifact có sẵn

Repo hiện đã có processed data, final model và report. Người mới chỉ cần tạo môi trường, cài thư viện rồi chạy web:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Mở `http://127.0.0.1:5000`. Dự báo CLI:

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol FPT --log-db
```

### Khi thật sự muốn cập nhật dữ liệu và retrain

Điều kiện trước:

- File raw ngoài repo phải tồn tại đúng đường dẫn `RAW_DATA_PATH` trong `config/settings.py`.
- Dataset mới phải được build xong.
- Phải chốt đủ cấu hình LR/RF/GB cho đúng fingerprint mới.
- TEST fingerprint đó chưa bị khóa.

Quy trình an toàn cho người mới:

```powershell
python scripts/refresh_data.py
python app.py
```

Sau đó vào `/tuning`, thử/chọn đủ 3 cấu hình, rồi dùng nút **chạy pipeline** trên trang. Route web kiểm tra fingerprint, config, fetch/pipeline lock và TEST lock trước khi khởi chạy.

**Cảnh báo trạng thái hiện tại:** fingerprint `f6cb3ac8f820` đã được đánh giá TEST và khóa. Không chạy lại `python scripts/run_pipeline.py` trên trạng thái hiện tại. Trong code hiện tại, CLI gọi `cleanup_outputs()` trước khi kiểm tra TEST lock; chạy lại có thể xóa processed data/model/report rồi mới báo lỗi. `STOCK_ALLOW_TEST_REEVAL=1` chỉ dành cho debug có chủ đích, không dùng để cải thiện điểm sau khi đã xem TEST.

### Script từng bước

```powershell
python scripts/preprocess_data.py
python scripts/build_features.py
python scripts/train_tune_models.py
python database/init_db.py
```

`preprocess_data.py` và `build_features.py` chuẩn bị dữ liệu. `train_tune_models.py` chỉ dùng TRAIN nhưng yêu cầu `manual_config.json` đầy đủ, đúng fingerprint.

Không dùng `evaluate_models.py` hoặc `select_final_model.py` như lệnh thử đi thử lại: cả hai đọc TEST nhưng không kiểm tra `test_evaluation_lock.json`. Chúng chỉ phù hợp cho debug/maintenance khi bạn hiểu rõ tác động.

---

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
| `models/final_model.pkl` | Model cuối đang dùng (hiện = Gradient Boosting) |
| `models/model_metadata.json` | Metadata model, feature order, metric, best params, decision_threshold |
| `reports/model_comparison.csv` | Bảng so sánh model |
| `reports/final_model_evaluation.csv` | Dòng kết quả của final model |
| `reports/classification_report.csv` | Precision/Recall/F1 theo từng lớp |
| `reports/confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV |
| `reports/confusion_matrix.png` | Ma trận nhầm lẫn dạng hình |
| `reports/feature_importance.csv` | Độ quan trọng feature |
| `reports/best_params.json` | Bộ hyperparameter đã chốt |
| `reports/train_test_summary.csv` | Tóm tắt train/test split |
| `reports/pipeline_summary.json` | Tóm tắt toàn bộ pipeline |
| `database/stock_prediction.db` | SQLite sync dữ liệu/report và log prediction |
| `experiments/tuning_history.csv` | Lịch sử các lần đánh giá cấu hình từ UI hoặc script thí nghiệm dùng chung backend CV |
| `experiments/manual_config.json` | Bộ tham số đã chốt cho 3 model, gắn fingerprint dataset |
| `docs/bao_cao_project_hose_stock_prediction.docx` | Báo cáo Word (bản đã tạo sẵn trong docs) |

### Vì sao không chèn comment trực tiếp vào file output?

- `models/final_model.pkl` là file nhị phân do `joblib` tạo. Không mở sửa hoặc chèn chữ; làm vậy sẽ hỏng model.
- `models/model_metadata.json` và `reports/pipeline_summary.json` là dữ liệu JSON cho code đọc. Chuẩn JSON không hỗ trợ comment.
- `reports/model_comparison.csv` và các CSV khác là bảng máy đọc. Thêm dòng comment sẽ làm sai cấu trúc hoặc thành dữ liệu giả.
- Muốn hiểu các artifact này, đọc bảng trên, comment trong code sinh file và các phần pipeline tương ứng của tài liệu này.

---

## 26. Từ điển thuật ngữ nhanh

| Thuật ngữ | Nghĩa dễ hiểu |
|---|---|
| HOSE | Sở Giao dịch Chứng khoán TP.HCM |
| Phiên giao dịch | Một ngày thị trường mở cửa; lưu ý code horizon thực tế đếm dòng có dữ liệu của từng mã |
| OHLCV | Open, High, Low, Close, Volume |
| Machine Learning | Cho máy học quy luật từ dữ liệu quá khứ |
| Classification | Bài toán phân loại; ở đây là `UP` hoặc `NOT_UP` |
| Binary classification | Phân loại 2 lớp |
| Feature | Cột đầu vào cho model |
| Target / Label | Đáp án model cần học |
| Horizon | Số bước tương lai; code hiện lấy dòng thứ 5 kế tiếp của từng mã |
| Threshold (nhãn) | Ngưỡng return để quyết định nhãn; ở đây là 1% |
| Train set | Dữ liệu dùng để model học |
| Test set | Dữ liệu tách riêng; project hiện dùng để đánh giá và chọn final model |
| Cross Validation | Chia train thành nhiều fold để kiểm tra ổn định |
| TimeSeriesSplit | Cross Validation giữ thứ tự thời gian |
| Gap | Khoảng bỏ trống giữa train fold và validation fold; `gap=5` hiện là 5 dòng bảng gộp |
| Hyperparameter | Tham số cấu hình trước khi train |
| Tuning | Thử nhiều hyperparameter để tìm bộ tốt |
| Decision threshold | Ngưỡng xác suất để quyết UP; tinh chỉnh riêng mỗi model thay cho 0.5 |
| Class imbalance | Mất cân bằng lớp: số mẫu UP ít hơn NOT_UP |
| class_weight / sample_weight | Cách tăng trọng số lớp thiểu số khi train (LR/RF dùng class_weight, GB dùng sample_weight) |
| Fingerprint | Chữ ký từ row count, max date và config; không hash toàn bộ nội dung dataset |
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

---

## 27. Những câu nên nói khi bảo vệ

**Project làm gì:**
> Project của em xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning. Bài toán là phân loại nhị phân: giá ở bước quan sát thứ 5 tiếp theo của cùng mã có tăng hơn 1% hay không. Với mã giao dịch đều, bước này gần với 5 phiên thị trường; mã dữ liệu thưa có thể kéo dài hơn.

**Dữ liệu đầu vào:**
> Em sử dụng dữ liệu OHLCV theo ngày, gồm mã cổ phiếu, ngày giao dịch, giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng giao dịch.

**Cách tạo nhãn:**
> Với mỗi mã tại ngày t, code lấy close ở dòng thứ 5 kế tiếp của chính mã đó, tính future_return_5d = close(t+5)/close(t) - 1. Nếu return lớn hơn 1% thì nhãn là UP, ngược lại là NOT_UP.

**Vì sao không random split:**
> Dữ liệu cổ phiếu là chuỗi thời gian nên random split dễ làm tương lai lọt vào train. Project chia theo label_end_date với mốc 2025-06-30 để giữ mọi nhãn TRAIN kết thúc trước hoặc tại mốc đó. Em cũng thừa nhận split hiện chưa tách tuyệt đối trading_date giữa TRAIN/TEST đối với mã dữ liệu thưa.

**Vì sao chọn Gradient Boosting:**
> Final model được chọn theo TEST F1_UP, hòa thì xét Recall_UP rồi độ đơn giản. Ba model rất sát nhau; Gradient Boosting đạt 0.5053 nên được chọn. Đây chỉ là chiến thắng nhỏ trên TEST hiện tại, và vì TEST tham gia chọn model nên metric sau chọn có selection bias.

**Vì sao không chọn model accuracy cao nhất:**
> Accuracy dễ bị lệch vì lớp NOT_UP nhiều hơn UP. Dummy đoán toàn NOT_UP nên accuracy cao nhất nhưng F1_UP bằng 0. Project ưu tiên F1_UP vì mục tiêu là nhận diện trường hợp tăng hơn 1%.

**Vì sao Accuracy các model chính thấp (~0.39–0.41):**
> Decision threshold dưới 0.5 làm model dự báo UP nhiều, nên Recall_UP cao nhưng false positive nhiều, kéo Accuracy và Precision xuống. Đây là một nguyên nhân lớn, nhưng không đủ để khẳng định model tốt; phải đọc thêm F1, confusion matrix và baseline luôn-UP.

**Tuning làm thế nào:**
> Backend Tuning Lab đánh giá mỗi cấu hình bằng TimeSeriesSplit trên TRAIN, lấy trung bình F1_UP của 5 validation fold thành CV F1_UP và ghi history. UI cho thử từng cấu hình; repo cũng có script gọi tự động nhiều cấu hình. Người dùng chốt đủ 3 config rồi pipeline mới fit model, evaluate TEST và chọn final.

**Giới hạn phương pháp hiện tại:**
> Em không khẳng định hệ thống hoàn toàn không leakage. TimeSeriesSplit đang chạy trên bảng gộp nhiều mã và gap=5 chỉ là 5 dòng, nên fold có thể chồng ngày và label window. Threshold cũng được chọn trên chính dữ liệu model vừa fit. Hướng cải thiện là split theo ngày, loại train row có label lấn validation và chọn threshold trên prediction ngoài mẫu.

**Có dùng database không:**
> Có. Project dùng SQLite để sync raw data, clean data, features, tuning results, model evaluations và lưu lịch sử predictions. Tuy nhiên source chính để xem kết quả model là các file report và model_metadata.json.

**Có phải khuyến nghị đầu tư không:**
> Không. Đây là hệ thống demo học thuật để dự báo xu hướng theo dữ liệu lịch sử OHLCV. Kết quả chỉ mang tính tham khảo, không phải khuyến nghị mua bán.

---

## 28. Những điều không nên nói sai

| Không nói | Nên nói |
|---|---|
| Project dự báo chính xác giá cổ phiếu. | Project phân loại UP / NOT_UP theo ngưỡng 1% ở dòng quan sát thứ 5 kế tiếp của mã. |
| Xác suất model là độ tin cậy tuyệt đối. | Đó là xác suất dự báo của model cho lớp UP. |
| Code luôn dự báo đúng 5 ngày thị trường mở cửa. | Code lấy dòng thứ 5 kế tiếp của từng mã; mã dữ liệu thưa có thể kéo dài nhiều tuần/tháng. |
| Web chạy realtime. | Web demo dùng dữ liệu offline đã xử lý và model đã train sẵn. |
| Feature importance chứng minh thị trường bị feature đó gây ra. | Feature importance chỉ cho biết model đã dựa vào feature đó nhiều, không chứng minh quan hệ nhân quả. |
| F1 lưu trong tuning history là F1 trên TEST. | History lưu CV F1_UP trung bình trên các validation fold thuộc TRAIN; TEST F1_UP chỉ sinh ở bước evaluate sau khi đã chốt cấu hình. |
| `gap=5` nghĩa là cách đúng 5 phiên thị trường. | Với bảng gộp hiện tại, `gap=5` chỉ là 5 dòng và chưa đủ purge horizon 5 bước. |
| TEST chỉ dùng để báo cáo. | Project hiện dùng TEST để vừa tính metric vừa chọn final model. |
| Toàn bộ tuning đều nhập tay trên web. | UI hỗ trợ nhập/chốt; các script thí nghiệm cũng tự động gọi cùng backend để thử nhiều config. |
| Pipeline hiện hoàn toàn không leakage. | Project có biện pháp giảm leakage nhưng CV/split hiện vẫn còn giới hạn đã nêu ở mục 14 và 29. |

---

## 29. Giới hạn và điểm kỹ thuật cần biết

1. Khi cần số mới nhất, ưu tiên `reports/pipeline_summary.json`, `reports/model_comparison.csv`, `reports/train_test_summary.csv`, `models/model_metadata.json`. `README.md` hiện vẫn ghi final model là Random Forest và là thông tin cũ; final hiện tại theo metadata/report là Gradient Boosting. Một số sơ đồ/tài liệu khác cũng có thể cũ hơn artifact.

2. Trong `templates/index.html`, phần `details` kỹ thuật có dùng `result.selected_report_model` và `result.model_match`, nhưng `services/prediction_service.py` hiện chưa trả hai field này. Phần dự báo chính vẫn hoạt động dựa trên `final_model.pkl` và `model_metadata.json`.

3. `scripts/finetune_model.py` chỉ là thông báo deprecated. Việc chốt tham số nay làm qua Tuning Lab (`/tuning`); pipeline chính thức đọc `experiments/manual_config.json` để train.

4. SQLite có bảng `tuning_results`, nhưng best params chi tiết nên xem trong `reports/best_params.json`.

5. Dữ liệu raw nằm **ngoài** repo ở `shared_dataset`, còn output đã xử lý nằm trong `data/processed/`.

6. Accuracy thấp một phần lớn vì ngưỡng tối ưu F1 trên TRAIN thấp hơn 0.5 và làm model báo UP nhiều; code không tối ưu Recall trực tiếp. TEST F1_UP final chỉ `0.5053`; baseline tính thêm kiểu luôn-UP khoảng `0.4939`, nên mức cải thiện nhỏ.

7. Pipeline có `pipeline.lock` và `test_evaluation_lock.json`, nhưng CLI kiểm tra TEST lock **sau** `cleanup_outputs()`. Rerun fingerprint đã khóa có thể xóa artifact trước khi abort. Route web kiểm tra trước và an toàn hơn. Hai script `evaluate_models.py` / `select_final_model.py` không kiểm tra TEST lock.

8. **Trạng thái hiện tại: code, dữ liệu và report đã đồng bộ.** Pipeline chính thức đã chạy xong trên **20 feature** (`trained_at = 2026-07-17`, fingerprint `f6cb3ac8f820`). Cả 3 model đã chốt tham số, final model = **Gradient Boosting**. Không còn tình trạng "report lệch code" như các lần trước.

9. Fingerprint gắn vào `manual_config.json` được hash từ row count, max trading date, split/horizon/threshold và feature list; nó **không hash nội dung OHLCV**. Sửa giá nhưng giữ số dòng/max date có thể không đổi fingerprint. Nếu fingerprint thật sự đổi, cần chốt lại đủ 3 model trước official pipeline.

10. CV hiện chia bảng nhiều mã theo vị trí dòng. `gap=5` không phải 5 ngày; audit hiện tại cho thấy fold có thể chung ngày và train label window lấn vào VAL. CV F1_UP có thể lạc quan. Cần grouped time split + purge theo `label_end_date` để sửa.

11. Outer TRAIN/TEST split bảo vệ `label_end_date`, nhưng không tách tuyệt đối `trading_date`; mã thưa như TTE làm TEST có reference date sớm hơn một số TRAIN rows.

12. TEST đang dùng để chọn final model trong ba ứng viên. Đây không phải hyperparameter leakage, nhưng làm TEST mất vai trò holdout hoàn toàn độc lập sau chọn model.

13. Decision threshold được tune từ prediction in-sample của fold-train và toàn TRAIN. Thiết kế chặt hơn dùng OOF hoặc một validation tầng trong.

14. `database/init_db.sql` chỉ là schema khởi tạo. Ba bảng được pandas ghi `replace` có thể mất PK/UNIQUE/NOT NULL sau sync.

---

## 30. Thứ tự học project cho dễ

1. Hiểu câu hỏi chính: dòng thứ 5 kế tiếp của mã có tăng hơn 1% không.
2. Hiểu OHLCV: open, high, low, close, volume.
3. Hiểu `future_return_5d` và cách tạo `target`.
4. Học 20 feature, đặc biệt return, SMA, RSI, volatility, volume ratio, và nhóm vị trí giá (return 10/20 phiên, khoảng cách đỉnh/đáy 20 phiên, tháng).
5. Phân biệt hai threshold: 1% để tạo target và ~0.4129 để đổi P(UP) thành prediction.
6. Hiểu hai luồng riêng: tuning trên TRAIN và official evaluate/select trên TEST.
7. Hiểu data leakage và giới hạn split hiện tại.
8. Hiểu Cross Validation; nhớ `gap=5` hiện chỉ là 5 dòng.
9. Hiểu Precision, Recall, F1, Confusion Matrix và baseline.
10. Đọc `reports/model_comparison.csv` để hiểu vì sao chọn Gradient Boosting và vì sao lợi thế nhỏ.
11. Đọc `services/feature_engineering.py` để hiểu feature/label.
12. Đọc `services/model_tuning.py` để hiểu CV, fit và tune ngưỡng.
13. Đọc `services/prediction_service.py` và `app.py` để hiểu web/CLI dự báo.

Nắm chắc các ý trên là bạn đã hiểu lõi project đủ để đọc code, chạy demo và giải thích khi bảo vệ.
