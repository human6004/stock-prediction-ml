# Lưu đồ kiến trúc và luồng hệ thống

Bộ sơ đồ mô tả hệ thống dự đoán cổ phiếu HOSE: pipeline ML offline, runtime prediction và runtime Chatbot RAG. Mỗi file HTML độc lập, có dark/light theme và menu export PNG/JPEG/WebP/SVG. Nguồn là các file JSON cùng tên, render bằng Archify — không sửa HTML/SVG bằng tay.

## Tổng quan hệ thống

- [01 - Kiến trúc runtime local](01-kien-truc-runtime-local.html) — bức tranh tổng: Offline ML + Runtime Prediction + Chatbot RAG. `Published Release` là điểm nối chung; Offline ML ghi artifact, Prediction và Chatbot chỉ đọc artifact.

## Offline ML pipeline

- [02 - Luồng dữ liệu pipeline](02-luong-du-lieu-pipeline.html) — dataflow offline: Raw → Clean → Features/Labels → Rolling split. CV chỉ trong TRAIN, chọn candidate trên VALIDATION, refit TRAIN+VALIDATION, TEST đúng một lần rồi publish artifact.
- [03 - Workflow tuning đến release](03-workflow-tuning-den-release.html) — `run_pipeline.py` chính thức: lock, fingerprint, config gate, validation gate, TEST gate, publish, `release_failed`. Phân biệt precheck ở Flask với revalidation trong pipeline chính thức; các nhánh dừng an toàn.
- [05 - Lifecycle fingerprint và TEST lock](05-lifecycle-fingerprint-va-test-lock.html) — vòng đời kép: fingerprint, config stale, TEST lock, `evaluated`, `published`, `release_failed`.

## Runtime prediction

- [04 - Sequence suy luận predict](04-sequence-suy-luan-predict.html) — runtime prediction: Browser/CLI → Flask → `prediction_service` → metadata + clean data + `final_model` → kết quả. Flask không nạp model trực tiếp, chỉ đọc artifact, không retrain.

## Runtime Chatbot RAG

- [06 - Sequence chatbot RAG](06-sequence-chatbot-rag.html) — Chatbot Mức B: Browser `/chat` → Flask `/api/chat` → `chatbot_service` → LLM tool calling → `chatbot_tools` → Release Gate → artifact/`prediction_service` → grounding validation → câu trả lời. Có nhánh social/advice cục bộ; 6 tool đọc; giới hạn 3 vòng LLM / 5 tool call / 30 giây. Chatbot chỉ đọc, không train, không chạy `run_pipeline.py`.

## Nguyên tắc

- Sửa JSON nguồn trước, render lại HTML bằng renderer Archify đúng mode (architecture / dataflow / workflow / sequence / lifecycle).
- Không thêm component không tồn tại trong code. Không có Vector DB, LangChain hay chat database.
- Không sửa code app/model/pipeline. Không xóa sơ đồ cũ, giữ nguyên tên file.
