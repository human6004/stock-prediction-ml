# HOSE Stock Trend Prediction

Project niên luận: Xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu trên sàn HOSE bằng Machine Learning.

## 1. Project này làm gì

Hệ thống đọc dữ liệu OHLCV toàn sàn HOSE từ CSV, làm sạch dữ liệu, tạo feature, tạo nhãn `UP / NOT_UP`, tune 3 model chính (LogReg, RF, GB) với TimeSeriesSplit(gap=5), chọn model tốt nhất theo `F1_UP`, rồi demo dự báo bằng Flask.

Kết quả dự báo trả lời câu hỏi: một mã cổ phiếu có tăng hơn 1% trong 5 phiên giao dịch tiếp theo hay không.

## 2. Dataset và roadmap

Dataset chính:

```text
D:\study\niên luận\shared_dataset\hose_stock_raw.csv
```

Roadmap chính:

```text
D:\study\niên luận\shared_dataset\roadmap_nien_luan_HOSE_5_phien.md
```

Cập nhật dữ liệu mới qua `scripts/fetch_hose_data.py` (vnstock).

## 3. Luật tạo nhãn

Tính riêng theo từng `symbol`:

```text
future_return_5d = close(t+5) / close(t) - 1
label_end_date = trading_date tại t+5
```

- `UP` nếu `future_return_5d > 0.01` (1%)
- `NOT_UP` nếu `future_return_5d <= 0.01`

## 4. Chia train / test (không leakage)

Chia theo `label_end_date` (không shuffle):

```text
SPLIT_DATE = 2025-12-31
train: label_end_date <= 2025-12-31
test:  label_end_date > 2025-12-31
```

Cấu hình CV/tuning trong `config/settings.py`: `CV_N_SPLITS=5`, `CV_GAP=5`, `TUNING_N_ITER=12`.

## 5. Cài đặt

```powershell
cd "D:\study\niên luận\v3\niên luận cơ sở ngành-cusor"
python -m venv .venv
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 6. Chạy pipeline đầy đủ

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/fetch_hose_data.py
python scripts/run_pipeline.py
```

Hoặc chạy từng bước:

```powershell
python scripts/preprocess_data.py
python scripts/build_features.py
python scripts/train_tune_models.py
python scripts/evaluate_models.py
python scripts/select_final_model.py
python database/init_db.py
```

Dự báo CLI (feature on-the-fly):

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol SSI --log-db
```

## 7. File output chính

```text
data/processed/hose_stock_clean.csv
data/processed/hose_stock_features.csv
data/processed/ml_dataset.csv
reports/eligible_symbols.csv
reports/excluded_symbols.csv
reports/train_test_summary.csv
reports/tuning_results.csv
reports/best_params.json
reports/cv_fold_results.csv
reports/model_comparison.csv
reports/classification_report.csv
reports/confusion_matrix.csv
reports/confusion_matrix.png
reports/hyperparameter_explanation.md
reports/pipeline_summary.json
models/dummy.pkl
models/logistic_regression_tuned.pkl
models/random_forest_tuned.pkl
models/gradient_boosting_tuned.pkl
models/final_model.pkl
models/model_metadata.json
database/stock_prediction.db
```

## 8. Chạy web

```powershell
python app.py
```

- Trang dự báo: http://127.0.0.1:5000
- Trang đánh giá: http://127.0.0.1:5000/evaluation

Web dùng `services/prediction_service.py` để tính feature on-the-fly và đọc `models/model_metadata.json`.

## 9. Model và tiêu chí chọn

| model_id | Model |
|----------|--------|
| 1 | Dummy Classifier (baseline) |
| 2 | Logistic Regression (tuned) |
| 3 | Random Forest (tuned) |
| 4 | Gradient Boosting (tuned) |

Chọn model cuối (id 2–4): `F1_UP` test cao nhất → `Recall_UP` → độ đơn giản (LogReg < RF < GB).

## 10. Cấu trúc thư mục

```text
config/           settings.py
services/         preprocessing, feature_engineering, model_tuning, model_evaluation, prediction_service, database_service
scripts/          preprocess_data, build_features, train_tune_models, evaluate_models, select_final_model, predict_stock, run_pipeline, fetch_hose_data
database/         init_db.sql, db_connection.py, init_db.py
templates/        Flask HTML
models/           .pkl + model_metadata.json
reports/          báo cáo đánh giá
app.py            Flask backend
```
