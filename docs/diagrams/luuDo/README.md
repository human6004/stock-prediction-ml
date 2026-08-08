# Lưu đồ kiến trúc và luồng hệ thống

Bộ sơ đồ mô tả hệ thống dự đoán cổ phiếu HOSE: pipeline ML offline, runtime prediction và chatbot Action Decision. Mỗi file HTML độc lập, có dark/light theme và menu export PNG/JPEG/WebP/SVG. Nguồn là các file JSON cùng tên, render bằng Archify — không sửa HTML/SVG bằng tay.

## Tổng quan hệ thống

- [01 - Kiến trúc runtime local](01-kien-truc-runtime-local.html) — bức tranh tổng: Offline ML + Runtime Prediction + Chatbot. `Published Release` là điểm nối chung; Offline ML ghi artifact, Prediction và Chatbot chỉ đọc artifact.

## Offline ML pipeline

- [02 - Luồng dữ liệu pipeline](02-luong-du-lieu-pipeline.html) — dataflow offline: Raw → Clean → Features/Labels → Rolling split (boundary suy từ phiên mới nhất: VALIDATION 274 ngày, TEST 94 ngày). CV 4-fold purged chạy ở Tuning Lab trên TRAIN; official pipeline chỉ replay config đã chốt, chọn candidate trên VALIDATION, refit TRAIN+VALIDATION, TEST đúng một lần rồi publish artifact.
- [03 - Workflow tuning đến release](03-workflow-tuning-den-release.html) — `run_pipeline.py` chính thức (không nhận tham số CLI): lock token-owned, 2 fingerprint, config gate, validation gate, TEST gate, publish, `release_failed`. Phân biệt precheck ở Flask với revalidation trong pipeline chính thức; ghi reports trước khi publish nên cả tập không nguyên tử.
- [05 - Lifecycle fingerprint và TEST lock](05-lifecycle-fingerprint-va-test-lock.html) — vòng đời kép: fingerprint train/experiment, config stale, `evaluation_registry.json` là gate TEST duy nhất (không còn file TEST lock hay env escape hatch), `started` → `evaluated` → `published`, và `release_failed` cháy snapshot vĩnh viễn.

## Runtime prediction

- [04 - Sequence suy luận predict](04-sequence-suy-luan-predict.html) — runtime prediction: Browser/CLI → Flask → `prediction_service` → metadata + clean data + `final_model` → kết quả. Route Flask không đọc PKL trực tiếp; `prediction_service` load in-process mỗi request cho `/predict` và `/compare`, chỉ `/screener` cache theo mtime. Không retrain ở runtime.

## Runtime Chatbot Action Decision

- [06 - Sequence chatbot Action Decision](06-sequence-chatbot-action-decision.html) — `/chat` → Flask `/api/chat` → đúng một LLM call trả JSON decision → validate 5 action → fixed dispatcher → read-only data/ML handler → deterministic formatter. Không có call LLM thứ hai; provider không nhận CSV/report/code/artifact. Chỉ trang `/chat` giữ transcript tab và gửi 6 message cuối.

## Nguyên tắc

- Sửa JSON nguồn trước, render lại HTML bằng renderer Archify đúng mode (architecture / dataflow / workflow / sequence / lifecycle).
- Không thêm component không tồn tại trong code. Chatbot không dùng RAG, Vector DB, embedding, LangChain, agent loop hoặc chat database.
- Khi thuật ngữ/flow đổi, chỉ giữ một sơ đồ chatbot canonical và sửa đồng thời mọi inbound reference cùng `meta.output` trong JSON nguồn.
