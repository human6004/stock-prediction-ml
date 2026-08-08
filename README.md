# HOSE Stock Trend Prediction

Project niên luận xây dựng hệ thống hỗ trợ dự báo xu hướng cổ phiếu HOSE bằng Machine Learning.

Bài toán hiện tại: dự báo một mã cổ phiếu có tăng hơn `1%` trong `5` phiên giao dịch tiếp theo hay không (`UP` / `NOT_UP`).

## Tài liệu chính

- [Giải thích project](docs/GIAI_THICH_PROJECT.md)
- [Sơ đồ kiến trúc hệ thống](docs/SO_DO_KIEN_TRUC_HE_THONG.md)
- [Các sơ đồ HTML export](docs/diagrams/)
- [Tuning ba model và chọn Final Model](docs/diagrams/model-workflows/README.md)
- [Chatbot structured context injection (SCI)](docs/CHATBOT_RAG_MUC_B.md)
- `docs/slides/` - slide bảo vệ (`thuyet_trinh_nien_luan.pptx`), outline và script dựng slide
- `docs/report_render/` - script sinh bản báo cáo `.docx` từ nội dung theo chương

## Luồng chính

```text
shared raw CSV
-> clean data
-> build technical features
-> create exact common-market t+5 UP/NOT_UP labels
-> TRAIN / VALIDATION / TEST by rolling dates
-> manually try and select one config/model with purged date CV on TRAIN
-> select model family on VALIDATION
-> refit winner on TRAIN+VALIDATION
-> evaluate TEST once and publish the same artifact
-> write CSV/JSON reports
-> Flask/CLI prediction
```

Policy hiện hành là `rolling_recent_cv_oof_threshold`: `UP` nghĩa là giá đúng phiên thị trường `t+5` tăng hơn `1%`. Tuning dùng time-series CV 4 fold từ `2021-01-01`, purge 5 phiên và threshold OOF riêng cho từng model. Ba mốc TRAIN/VALIDATION/TEST không cố định trong tài liệu mà được suy ra theo cửa sổ rolling từ phiên mới nhất của dataset (`services/protocol_dates.py`); giá trị hằng trong `config/settings.py` chỉ là fallback. Mốc thực tế của snapshot đã publish luôn nằm trong `models/model_metadata.json` (`split_date`, `validation_end_date`, `test_end_date`). Dữ liệu mới hơn `test_end_date` chỉ phục vụ inference. Model family được chọn trên VALIDATION; TEST chỉ đánh giá một lần model đã chọn và refit.

Tuning history dùng fingerprint riêng của TRAIN. TEST lock dùng fingerprint của snapshot TRAIN+VALIDATION+TEST đóng băng. Vì vậy refresh dữ liệu inference không mở lại TEST.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` liệt kê 9 dependency (`pandas`, `numpy`, `scikit-learn`, `joblib`, `Flask`,
`matplotlib`, `vnstock`, `openai`, `python-dotenv`) nhưng **chưa pin version** nào. Cài lại ở máy
khác sẽ lấy bản mới nhất trên PyPI, nên môi trường chưa reproducible; muốn dựng lại đúng bản đã
dùng thì phải tự pin (ví dụ `pip freeze > requirements.lock.txt`).

Chatbot cần cấu hình LLM qua `.env` ở gốc repo (`config/settings.py` đọc bằng python-dotenv).
Copy từ `.env.example` rồi điền 3 biến:

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
```

Thiếu `.env` thì pipeline và các trang dự báo vẫn chạy, chỉ riêng chatbot không gọi được LLM.

Chatbot chạy theo structured context injection: rule-based intent routing khớp keyword để chọn nguồn, server tự dựng context từ signal/artifact/report đã publish rồi gọi provider đúng một lần, không gửi tool schema, raw CSV, code hoặc pickle. Dock và `/chat` dùng chung hội thoại/state trong `sessionStorage`; reload còn, đóng tab thì mất. Sources, warnings, release status và grounding vẫn do server kiểm soát.

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

Hoặc gộp cả ba bước fetch + preprocess + build features bằng một lệnh:

```powershell
python scripts/refresh_data.py
```

Sau đó thử hyperparameter và tự chọn một run hợp lệ cho từng model tại `/tuning`. Final Model chỉ
được chọn và publish bằng `python scripts/run_pipeline.py`; không chạy riêng
từng bước VALIDATION/TEST vì sẽ phá TEST lock và tính nhất quán artifact.

## Dự báo

CLI:

```powershell
python scripts/predict_stock.py --symbol FPT
python scripts/predict_stock.py --symbol SSI
```

`predict_stock.py` là script duy nhất có argparse và chỉ nhận một tham số `--symbol` (bắt buộc).

Web:

```powershell
python app.py
```

- Trang dự báo: http://127.0.0.1:5000
- Trang đánh giá: http://127.0.0.1:5000/evaluation
- Trang chatbot: http://127.0.0.1:5000/chat

`app.py` khai báo tổng cộng 14 route, không chỉ 6 trang HTML. Ngoài các trang trên còn có
`POST /predict` (`app.py:476`, nhận form từ trang chủ hoặc link `?symbol=`), `/compare`,
`/screener`, `/tuning` cùng nhóm action `/tuning/*`, ảnh `/reports/confusion_matrix.png`,
và endpoint JSON `POST /api/chat` phục vụ chatbot.

## Cấu trúc thư mục

```text
config/           cấu hình đường dẫn, feature, split, model
data/processed/   dữ liệu đã xử lý phục vụ demo/pipeline
docs/             tài liệu giải thích và sơ đồ kiến trúc
experiments/      state của Tuning Lab (history, config, registry, archive)
models/           model đã train và metadata
reports/          báo cáo đánh giá model/pipeline
scripts/          các lệnh chạy từng bước và full pipeline
services/         logic xử lý dữ liệu, feature, tuning, evaluation, prediction
static/           CSS/JS cho Flask web
templates/        HTML cho Flask web
tests/            pytest cho pipeline, prediction, tuning, chatbot, UI
app.py            Flask backend
.env.example      mẫu 3 biến LLM (không chứa giá trị thật)
.env              cấu hình LLM thật của máy local, không commit
```

`experiments/archive/` là snapshot đóng băng của các release trước (config, tuning history,
model và report đã publish). Không nằm trong luồng chạy: không code path nào đọc thư mục này,
nó chỉ giữ bằng chứng để đối chiếu giữa các lần thay đổi protocol.

## Chạy test

Repo không có `conftest.py` cũng không có `pytest.ini` / `pyproject.toml` / `setup.cfg`, và
không test module nào tự thêm project root vào `sys.path`. Vì vậy phải truyền `PYTHONPATH`
trỏ về root; nếu thiếu, collection có thể lỗi `ModuleNotFoundError` khi import `app`,
`config`, `services`, `scripts`.

PowerShell:

```powershell
$env:PYTHONPATH="."; python -m pytest tests -q
```

bash / CI:

```bash
PYTHONPATH=. python -m pytest tests -q
```

Lệnh trên chạy toàn bộ suite trong `tests/`; không khóa cứng số lượng test trong tài liệu.

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
```
