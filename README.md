# HOSE Stock Trend Prediction

Project niên luận xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning.

Bài toán hiện tại: dự báo một mã cổ phiếu có tăng hơn `1%` trong `5` phiên giao dịch tiếp theo hay không (`UP` / `NOT_UP`).

## Tài liệu chính

- [Giải thích project](docs/GIAI_THICH_PROJECT.md)
- [Sơ đồ kiến trúc hệ thống](docs/SO_DO_KIEN_TRUC_HE_THONG.md)
- [Các sơ đồ HTML export](docs/diagrams/)

## Luồng chính

```text
shared raw CSV
-> clean data
-> build technical features
-> create UP/NOT_UP labels
-> split train/test by label_end_date
-> tune and evaluate models
-> select final model
-> sync SQLite
-> Flask/CLI prediction
```

Model cuối hiện tại là `Random Forest`, được chọn theo `F1_UP` trên tập test. Các artefact demo trong `models/`, `reports/` và `data/processed/` được giữ lại để chạy web và phục vụ bảo vệ.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy pipeline

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

## Dự báo

CLI:

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol SSI --log-db
```

Web:

```powershell
python app.py
```

- Trang dự báo: http://127.0.0.1:5000
- Trang đánh giá: http://127.0.0.1:5000/evaluation

## Cấu trúc thư mục

```text
config/           cấu hình đường dẫn, feature, split, model
data/processed/   dữ liệu đã xử lý phục vụ demo/pipeline
database/         SQLite schema, connection, sync script
docs/             tài liệu giải thích và sơ đồ kiến trúc
models/           model đã train và metadata
reports/          báo cáo đánh giá model/pipeline
scripts/          các lệnh chạy từng bước và full pipeline
services/         logic xử lý dữ liệu, feature, tuning, evaluation, prediction
static/           CSS cho Flask web
templates/        HTML cho Flask web
app.py            Flask backend
```

## Output quan trọng

```text
data/processed/hose_stock_clean.csv
data/processed/hose_stock_features.csv
data/processed/ml_dataset.csv
models/final_model.pkl
models/model_metadata.json
reports/model_comparison.csv
reports/confusion_matrix.csv
reports/confusion_matrix.png
reports/feature_importance.csv
reports/pipeline_summary.json
database/stock_prediction.db
```
