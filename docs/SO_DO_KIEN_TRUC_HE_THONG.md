# Sơ đồ kiến trúc hệ thống HOSE ML - policy `rolling_recent_cv_oof_threshold`

## 1. Kiến trúc tổng thể

```mermaid
flowchart LR
    A["Raw HOSE OHLCV"] --> B["Preprocessing"]
    B --> C["Clean OHLCV"]
    C --> D["20 technical features"]
    C --> E["Exact common-market t+5 labels"]
    D --> F["ML dataset"]
    E --> F
    F --> G["Rolling TRAIN / VALIDATION / TEST"]
    G --> H["Tuning Lab: user-entered configs"]
    H --> I["Manual choice LR / RF / GB"]
    I --> J["TRAIN candidates"]
    J --> K["VALIDATION selection"]
    K --> L["TRAIN+VALIDATION refit"]
    L --> M["One-time TEST"]
    M --> N["Atomic final_model.pkl + metadata"]
    N --> O["Flask / CLI inference"]
    O --> P["UP / NOT_UP at decision_threshold"]

    F --> Q["CSV reports"]
    Q --> O
```

## 2. Ranh giới dữ liệu

```mermaid
flowchart TD
    A["ML dataset có trading_date và label_end_date"] --> B{"Ranh giới rolling"}
    B -->|"label_end <= train_end_date"| C["TRAIN"]
    B -->|"trading > train_end_date và label_end <= validation_end_date"| D["VALIDATION"]
    B -->|"validation_end_date < trading <= test_end_date"| E["TEST đóng băng"]
    B -->|"label cắt qua boundary"| F["PURGED"]
    C --> G["4-fold date CV, gap 5 sessions"]
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
    H --> I["TEST once at decision_threshold"]
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
    Service->>Service: score >= decision_threshold ? UP : NOT_UP
    Service-->>UI: Kết quả + 5 phiên dự kiến + baseline warning
    UI-->>User: Biểu đồ và Điểm UP
```

## 6. Chatbot Action Decision

```mermaid
flowchart LR
    A["User /chat"] --> B["POST /api/chat\nmessage + history"]
    B --> C["Validate payload\ntối đa 6 history message"]
    C --> D["LLM call 1\nJSON decision"]
    D --> E["Validate 5 action\n+ arguments"]
    E -->|"GENERAL_CHAT / OUT_OF_SCOPE"| I["Deterministic formatter"]
    E -->|"3 action dữ liệu"| F["Fixed dispatcher"]
    F --> G["chatbot_tools\nreadiness + scope"]
    G -->|"STOCK_SIGNAL / STOCK_RANKING"| H["prediction_service\nML inference"]
    H --> G
    G --> I
    I -->|"action dữ liệu có kết quả\nhoặc GENERAL_CHAT / OUT_OF_SCOPE"| L["LLM call 2 compose\nviết văn từ JSON backend"]
    L --> J["answer + sources + warnings\n+ hai mốc ngày"]
    I -->|"mã ngoài scope / compose tắt, lỗi"| J
    J --> K["Render bằng textContent"]
```

LLM call 1 chỉ trả JSON `{action, arguments}`. Backend chọn handler cố định, kiểm symbol scope, gọi ML model và format số liệu thật; LLM call 2 (compose) diễn đạt lại câu trả lời — action dữ liệu từ đúng JSON số liệu đó (chỉ khi có kết quả thật), `GENERAL_CHAT`/`OUT_OF_SCOPE` từ `kind`/`reason` kèm danh sách năng lực, không có số liệu nào — lỗi/timeout thì giữ bản formatter/câu cố định, tắt bằng `CHATBOT_COMPOSE=0`. Call decision không nhận CSV, report, code hoặc artifact; không có keyword router dự phòng hay tool loop. `GENERAL_CHAT` và `OUT_OF_SCOPE` bỏ qua dispatcher dữ liệu, không chạm ML; mã ngoài scope giữ câu deterministic, không compose.

## 7. Report contract

| File | Nội dung |
|---|---|
| `tuning_results.csv` | CV best config trên TRAIN |
| `cv_fold_results.csv` | Metric và dải ngày từng fold |
| `model_comparison.csv` | Ba candidates và baselines trên VALIDATION |
| `final_model_evaluation.csv` | Final Model và hai baselines trên TEST |
| `model_metadata.json` | Policy, fingerprint, feature order, Validation/TEST metrics, baseline status |

Mốc TRAIN/VALIDATION/TEST không ghi cứng trong tài liệu này: chúng do `services/protocol_dates.resolve_protocol_dates` suy ra từ dataset (rolling) và được ghi lại trong `model_metadata.json` dưới các key `split_date` (bằng `train_end_date`), `train_end_date`, `validation_end_date`, `test_end_date`. Tương tự, `decision_threshold` lấy từ artifact tuning rồi ghi vào metadata, không phải hằng số 0.5.

Artifact đang phục vụ (`models/final_model.pkl`) hiện mang `policy_id = legacy_pre_validation_baseline_gate`, tức được publish trước khi policy `rolling_recent_cv_oof_threshold` có hiệu lực. `prediction_service` phát hiện lệch này và trả `baseline_warning` cho UI. Số liệu cụ thể tra trực tiếp trong `models/model_metadata.json` và `reports/`, không ghi lại ở đây để tránh lệch.
