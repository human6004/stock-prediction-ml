# Sơ đồ kiến trúc hệ thống dự báo xu hướng cổ phiếu HOSE

Tài liệu này được tổng hợp từ code hiện tại và roadmap của project.

## 1. Kiến trúc tổng thể hiện tại

```mermaid
flowchart LR
    A["Nguồn dữ liệu OHLCV HOSE"] --> B["hose_stock_raw.csv"]
    C["scripts/fetch_hose_data.py"] --> B
    B --> D["scripts/run_pipeline.py"]
    E["config/settings.py"] --> C
    E --> D
    F["Roadmap markdown"] --> D

    D --> G["hose_stock_clean.csv"]
    D --> H["hose_stock_features.csv"]
    D --> I["ml_dataset.csv"]
    D --> J["models/*.pkl"]
    D --> K["reports/*.csv, *.png, *.json, *.txt"]

    I --> L["Train/Test split theo label_end_date"]
    L --> M["Train 4 model"]
    M --> N["Chọn model theo F1_UP"]
    N --> O["models/final_model.pkl"]

    O --> P["app.py Flask"]
    H --> P
    K --> P
    P --> Q["templates/index.html"]
    P --> R["templates/evaluation.html"]
    S["static/style.css"] --> Q
    S --> R
    Q --> T["Người dùng nhập mã cổ phiếu"]
    R --> U["Bảng đánh giá model"]
```

### Cách đọc sơ đồ

- `hose_stock_raw.csv` là dữ liệu OHLCV gốc.
- `fetch_hose_data.py` dùng `vnstock` để cập nhật thêm dữ liệu mới vào raw CSV.
- `run_pipeline.py` là phần xử lý chính: làm sạch, tạo feature, tạo nhãn, train model, đánh giá, xuất report.
- `final_model.pkl` là model cuối được web sử dụng.
- `app.py` là Flask backend, nhận mã cổ phiếu từ người dùng và trả kết quả `UP / NOT_UP`.
- `templates/` và `static/` là phần giao diện web.

## 2. Pipeline xử lý dữ liệu và train model

```mermaid
flowchart TD
    A["hose_stock_raw.csv"] --> B["Kiểm tra cột bắt buộc"]
    B --> C["Làm sạch dữ liệu"]
    C --> D["hose_stock_clean.csv"]
    C --> E["Lọc mã đủ tối thiểu 250 phiên"]
    E --> F["Tạo 15 feature kỹ thuật"]
    F --> G["hose_stock_features.csv"]
    G --> H["Tạo future_return_5d"]
    H --> I["Gắn nhãn target: UP hoặc NOT_UP"]
    I --> J["ml_dataset.csv"]
    J --> K["Chia train/test theo SPLIT_DATE"]
    K --> L["Train set: label_end_date <= SPLIT_DATE"]
    K --> M["Test set: label_end_date > SPLIT_DATE"]
    L --> N["Train Dummy Classifier"]
    L --> O["Train Logistic Regression"]
    L --> P["Train Random Forest"]
    L --> Q["Train Gradient Boosting"]
    M --> R["Đánh giá trên test set"]
    N --> R
    O --> R
    P --> R
    Q --> R
    R --> S["model_comparison.csv"]
    R --> T["confusion_matrix.csv và PNG"]
    R --> U["feature_importance.csv"]
    S --> V["Chọn model có F1_UP cao nhất"]
    V --> W["models/final_model.pkl"]
```

### Ý chính

Pipeline không train trực tiếp từ file raw. Nó đi qua nhiều bước:

```text
raw data
→ cleaned data
→ feature data
→ ML dataset có target
→ train/test split
→ train model
→ evaluate
→ final_model.pkl
```

`ml_dataset.csv` chứa cả train và test. Code chia train/test trong RAM bằng `label_end_date`:

```text
Train: label_end_date <= SPLIT_DATE
Test : label_end_date > SPLIT_DATE
```

Theo code hiện tại:

```text
SPLIT_DATE = 2025-12-31
```

Theo report hiện tại:

```text
Train: 2019-10-23 đến 2025-12-31
Test : 2026-01-05 đến 2026-05-29
```

## 3. Luồng dự báo trên web

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant Web as Flask app.py
    participant Features as hose_stock_features.csv
    participant Model as final_model.pkl
    participant Report as model_comparison.csv
    participant UI as HTML template

    User->>Web: Nhập mã cổ phiếu, ví dụ FPT
    Web->>Web: Chuẩn hóa symbol thành chữ hoa
    Web->>Features: Đọc dữ liệu feature
    Features-->>Web: Các dòng của symbol
    Web->>Web: Lấy dòng mới nhất theo trading_date
    Web->>Model: Load model cuối
    Web->>Model: predict và predict_proba
    Model-->>Web: UP hoặc NOT_UP, xác suất lớp UP
    Web->>Report: Đọc tên model đang được chọn
    Report-->>Web: Random Forest hoặc model selected
    Web->>UI: Render kết quả
    UI-->>User: Hiển thị dự báo và giải thích
```

### Ý chính

Khi chạy web, hệ thống không train lại model. Web chỉ:

1. Đọc feature đã xử lý.
2. Lấy dòng mới nhất của mã cổ phiếu.
3. Load `final_model.pkl`.
4. Dự báo `UP / NOT_UP`.
5. Hiển thị xác suất lớp `UP` nếu model hỗ trợ.

## 4. Các thành phần chính và vai trò

| Thành phần | Vai trò |
|---|---|
| `config/settings.py` | Cấu hình đường dẫn, feature, model, split date, threshold |
| `scripts/fetch_hose_data.py` | Cập nhật dữ liệu OHLCV mới bằng `vnstock` và append vào raw CSV |
| `scripts/run_pipeline.py` | Pipeline chính: clean, feature, label, train, evaluate |
| `scripts/train_tune_models.py` | Tune LogReg, RF, GB với TimeSeriesSplit(gap=5) |
| `data/processed/hose_stock_clean.csv` | Dữ liệu OHLCV đã làm sạch |
| `data/processed/hose_stock_features.csv` | Dữ liệu có 15 feature (metadata web, DB sync) |
| `data/processed/ml_dataset.csv` | Dữ liệu có feature + target, dùng để train/test |
| `models/final_model.pkl` | Model cuối cùng đang triển khai |
| `reports/model_comparison.csv` | Bảng so sánh 4 model |
| `reports/confusion_matrix.csv` | Bảng đúng/sai của final model |
| `reports/feature_importance.csv` | Mức độ quan trọng của feature |
| `app.py` | Flask backend |
| `templates/index.html` | Trang nhập mã và xem dự báo |
| `templates/evaluation.html` | Trang xem bảng đánh giá model |
| `static/style.css` | Giao diện CSS |

## 5. Kiến trúc theo roadmap và trạng thái hiện tại

Roadmap đề xuất kiến trúc đầy đủ hơn:

```mermaid
flowchart TD
    A["vnstock hoặc vnstock3"] --> B["SQLite database"]
    B --> C["stock_prices_raw"]
    C --> D["stock_prices_clean"]
    D --> E["stock_features"]
    E --> F["Train model"]
    F --> G["trained_models và model_evaluations"]
    F --> H["models/final_model.pkl"]
    H --> I["Flask web demo"]
    I --> J["predictions table"]
```

Code hiện tại:

| Phần trong roadmap | Trạng thái hiện tại |
|---|---|
| Fetch dữ liệu bằng `vnstock` | Đã có `scripts/fetch_hose_data.py` |
| SQLite database | Chưa có trong code hiện tại |
| Lưu raw/clean/features vào database | Hiện lưu bằng CSV |
| Train nhiều model | Đã có |
| Chọn model tốt nhất theo `F1_UP` | Đã có |
| Lưu `final_model.pkl` | Đã có |
| Web Flask demo | Đã có |
| Lưu lịch sử predictions | Chưa thấy trong code hiện tại |

## 6. Sơ đồ một câu

```text
vnstock/raw CSV
→ làm sạch dữ liệu
→ tạo feature
→ tạo nhãn UP/NOT_UP
→ chia train/test theo thời gian
→ train và đánh giá model
→ chọn Random Forest làm final model
→ Flask web dùng final_model.pkl để dự báo mã người dùng nhập
```

