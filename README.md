# HOSE Stock Trend Prediction

Project niên luận xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning.

Bài toán hiện tại: dự báo một mã cổ phiếu có tăng hơn `1%` trong `5` phiên giao dịch tiếp theo hay không (`UP` / `NOT_UP`).

## Tài liệu chính

- [Giải thích project](docs/GIAI_THICH_PROJECT.md)
- [Sơ đồ kiến trúc hệ thống](docs/SO_DO_KIEN_TRUC_HE_THONG.md)
- [Các sơ đồ HTML export](docs/diagrams/)
- [Tuning ba model và chọn Final Model](docs/diagrams/model-workflows/README.md)

## Luồng chính

```text
shared raw CSV
-> clean data
-> build technical features
-> create exact common-market t+5 UP/NOT_UP labels
-> TRAIN / VALIDATION / TEST by fixed dates
-> manually try and select one config/model with purged date CV on TRAIN
-> select model family on VALIDATION
-> refit winner on TRAIN+VALIDATION
-> evaluate TEST once and publish the same artifact
-> sync SQLite
-> Flask/CLI prediction
```

Policy hiện hành là `v3_recent4_oof_threshold`: `UP` nghĩa là giá đúng phiên thị trường `t+5` tăng hơn `1%`. Tuning dùng time-series CV 4 fold từ `2021-01-01`, purge 5 phiên và threshold OOF riêng cho từng model. TRAIN kết thúc `30/06/2025`, VALIDATION kết thúc `31/03/2026`, TEST chính thức được đóng băng từ `01/04/2026` đến `03/07/2026`. Dữ liệu mới hơn chỉ phục vụ inference. Model family được chọn trên VALIDATION; TEST chỉ đánh giá một lần model đã chọn và refit.

Tuning history dùng fingerprint riêng của TRAIN. TEST lock dùng fingerprint của snapshot TRAIN+VALIDATION+TEST đóng băng. Vì vậy refresh dữ liệu inference không mở lại TEST.

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

Chuẩn bị dataset riêng:

```powershell
python scripts/preprocess_data.py
python scripts/build_features.py
```

Sau đó thử hyperparameter và tự chọn một run hợp lệ cho từng model tại `/tuning`. Final Model chỉ
được chọn và publish bằng `python scripts/run_pipeline.py`; không chạy riêng
từng bước VALIDATION/TEST vì sẽ phá TEST lock và tính nhất quán artifact.

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
reports/final_model_evaluation.csv
reports/cv_fold_results.csv
reports/confusion_matrix.csv
reports/confusion_matrix.png
reports/feature_importance.csv
reports/pipeline_summary.json
database/stock_prediction.db
```
