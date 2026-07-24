# Sơ đồ kiến trúc hệ thống HOSE ML - protocol v2

## 1. Kiến trúc tổng thể

```mermaid
flowchart LR
    A["Raw HOSE OHLCV"] --> B["Preprocessing"]
    B --> C["Clean OHLCV"]
    C --> D["20 technical features"]
    C --> E["Exact common-market t+5 labels"]
    D --> F["ML dataset"]
    E --> F
    F --> G["Fixed TRAIN / VALIDATION / TEST"]
    G --> H["Tuning Lab: user-entered configs"]
    H --> I["Manual choice LR / RF / GB"]
    I --> J["TRAIN candidates"]
    J --> K["VALIDATION selection"]
    K --> L["TRAIN+VALIDATION refit"]
    L --> M["One-time TEST"]
    M --> N["Atomic final_model.pkl + metadata"]
    N --> O["Flask / CLI inference"]
    O --> P["UP / NOT_UP at score 0.5"]

    F --> Q["CSV reports"]
    Q --> R["SQLite sync"]
    O --> R
```

## 2. Ranh giới dữ liệu

```mermaid
flowchart TD
    A["ML dataset có trading_date và label_end_date"] --> B{"Ranh giới"}
    B -->|"label_end <= 30/06/2025"| C["TRAIN"]
    B -->|"trading > 30/06/2025 và label_end <= 31/03/2026"| D["VALIDATION"]
B -->|"31/03/2026 < trading <= 03/07/2026"| E["TEST đóng băng"]
    B -->|"label cắt qua boundary"| F["PURGED"]
    C --> G["5-fold date CV, gap 5 sessions"]
    D --> H["Chọn LR / RF / GB"]
    E --> I["Chỉ đánh giá winner đã refit"]
```

## 3. Ba pipeline tuning

Luồng giống nhau; hộp hyperparameter là phần khác nhau bắt buộc làm nổi bật.

```mermaid
flowchart LR
    A["Logistic Regression"] --> B["User enters C and solver"]
    B --> C["Date-purged CV runs"]
    C --> D["User selects LR config"]

    E["Random Forest"] --> F["n_estimators\nmax_depth\nmin_samples_leaf\nmax_features"]
    F --> G["Date-purged CV runs"]
    G --> H["User selects RF config"]

    I["Gradient Boosting"] --> J["n_estimators\nlearning_rate\nmax_depth\nsubsample"]
    J --> K["Date-purged CV runs"]
    K --> L["User selects GB config"]

    D --> M["Fit full TRAIN candidates"]
    H --> M
    L --> M
    M --> N["VALIDATION selection"]
```

Chi tiết giới hạn tham số nằm tại [Giải thích project](GIAI_THICH_PROJECT.md#7-tuning-ba-model).

## 4. Final Model lifecycle

```mermaid
flowchart LR
    A["Best params per family"] --> B["Fresh LR candidate on TRAIN"]
    A --> C["Fresh RF candidate on TRAIN"]
    A --> D["Fresh GB candidate on TRAIN"]
    B --> E["VALIDATION metrics"]
    C --> E
    D --> E
    E --> F["F1_UP -> Recall_UP -> LR/RF/GB"]
    F --> G["Clone winner"]
    G --> H["Fit TRAIN+VALIDATION"]
    H --> I["TEST once at threshold 0.5"]
    I --> J["Compare always-UP / always-NOT_UP"]
    J --> K["Atomic promote exact tested artifact"]
    K --> L["Write metadata and TEST lock"]
```

## 5. Inference

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant UI as Flask UI / CLI
    participant Service as prediction_service
    participant Data as Clean OHLCV
    participant Model as final_model.pkl
    participant Meta as model_metadata.json

    User->>UI: Chọn mã
    UI->>Service: predict_symbol / predict_symbols
    Service->>Data: Tính feature đến ngày tham chiếu
    Service->>Model: Load exact tested artifact
    Service->>Meta: Kiểm policy/fingerprint
    Model-->>Service: Điểm UP
    Service->>Service: score >= 0.5 ? UP : NOT_UP
    Service-->>UI: Kết quả + 5 phiên dự kiến + baseline warning
    UI-->>User: Biểu đồ và Điểm UP
```

## 6. Report contract

| File | Nội dung |
|---|---|
| `tuning_results.csv` | CV best config trên TRAIN |
| `cv_fold_results.csv` | Metric và dải ngày từng fold |
| `model_comparison.csv` | Ba candidates và baselines trên VALIDATION |
| `final_model_evaluation.csv` | Final Model và hai baselines trên TEST |
| `model_metadata.json` | Policy, fingerprint, feature order, Validation/TEST metrics, baseline status |

Artifact hiện có vẫn là legacy cho tới khi Tuning Lab đạt `12/12` cho cả ba model và pipeline v2 hoàn tất. Không có metric v2 được ghi cứng trong tài liệu này.
