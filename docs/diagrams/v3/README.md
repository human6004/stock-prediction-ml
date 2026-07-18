# Giải thích 5 sơ đồ kiến trúc v3

Thư mục này chứa 5 sơ đồ Archify đã được đối chiếu lại với mã nguồn và artifact hiện có
ngày **2026-07-17**. Mỗi sơ đồ gồm một file JSON nguồn và một file HTML tự chứa, có đổi
theme sáng/tối và xuất PNG/JPEG/WebP/SVG.

| Sơ đồ | Câu hỏi chính |
|---|---|
| `stock-architecture.html` | Hệ thống gồm thành phần nào và chúng phụ thuộc nhau ra sao? |
| `stock-dataflow.html` | Dữ liệu biến đổi từ OHLCV đến dự đoán như thế nào? |
| `stock-workflow.html` | Refresh, tuning, pipeline và serving chạy theo thứ tự nào? |
| `stock-sequence.html` | Một request `POST /predict` đi qua những thành phần nào? |
| `stock-lifecycle.html` | Một lần chạy pipeline chính thức bị chặn, thất bại hoặc hoàn tất ra sao? |

## Snapshot dùng trong sơ đồ

Số liệu lấy từ `reports/pipeline_summary.json`, `models/model_metadata.json`,
`experiments/manual_config.json` và `experiments/test_evaluation_lock.json`.

| Hạng mục | Giá trị hiện tại |
|---|---:|
| Raw OHLCV | 552.738 dòng, 400 mã |
| Clean CSV | 552.512 dòng |
| Mã đủ điều kiện train | 396 mã; 4 mã bị loại |
| Feature CSV | 514.808 dòng, 20 feature |
| ML dataset | 512.828 dòng |
| Split date | `2025-06-30`, theo `label_end_date` |
| TRAIN / TEST | 417.806 / 95.022 dòng |
| Dataset fingerprint | `f6cb3ac8f820` |
| Final model | Gradient Boosting |
| TEST F1_UP / Recall_UP | 0,5053 / 0,9176 |
| Decision threshold | 0,4129 |

## 1. Kiến trúc runtime

`stock-architecture.html` mô tả hệ thống đang chạy, không chỉ pipeline ML.

- Flask là monolith server-rendered bằng Jinja, bind `127.0.0.1:5000`. Không có SPA hay
  REST API JSON riêng.
- Browser dùng các trang dự đoán, đánh giá, tuning và trạng thái refresh. CLI dùng chung
  `Prediction Service` nhưng không đi qua Flask/Jinja.
- Refresh dữ liệu chạy nền bằng `subprocess.Popen`. Request trả về ngay rồi trang status
  đọc log để hiển thị tiến độ.
- Pipeline chính thức chạy bằng `subprocess.run`. Request Flask chờ tiến trình con hoàn tất.
- `vnstock/KBS` và raw CSV nằm ngoài repo. Processed CSV, PKL, metadata và report là nguồn
  dữ liệu chính trong repo.
- SQLite chỉ mirror raw/clean/features/tuning/evaluation và lưu prediction history. SQLite
  không chứa file model hoặc report.
- Experiment State quản lý tuning history, manual config, dataset fingerprint, test lock,
  fetch lock, pipeline lock và log lần chạy.

## 2. Luồng dữ liệu

`stock-dataflow.html` theo dõi lineage từ trái sang phải:

```text
vnstock/KBS
-> shared raw CSV
-> clean CSV + quality/eligible/excluded lists
-> 20 technical features
-> target UP/NOT_UP + label_end_date
-> TRAIN/TEST split
-> manual configs + 4 candidate artifacts
-> TEST evaluation + Gradient Boosting release
-> Flask/CLI inference + SQLite audit
```

Các điểm dễ nhầm:

- Clean CSV giữ toàn bộ 400 mã hợp lệ. Danh sách eligible mới quyết định 396 mã nào đi vào
  feature dataset dùng train.
- `future_close_5d` và `future_return_5d` chỉ tạo target, không thuộc 20 feature đầu vào.
- TRAIN gồm dòng có `label_end_date <= 2025-06-30`; TEST gồm dòng có
  `label_end_date > 2025-06-30`.
- Manual config cấp tham số cho Logistic Regression, Random Forest và Gradient Boosting.
  Dummy Classifier chỉ là baseline, không chạy CV như ba model tunable.
- Final release gồm `final_model.pkl`, `model_metadata.json` và reports. Serving còn đọc
  clean history để tính feature mới nhất cho mã được yêu cầu.

## 3. Workflow vận hành

`stock-workflow.html` có **ba entrypoint độc lập**, không phải một chuỗi bắt buộc:

1. `Refresh Request`: cập nhật dữ liệu tùy chọn.
2. `Tune Request`: thử và chốt cấu hình cho từng model.
3. `Official Run`: chạy pipeline chính thức bằng một POST riêng.

Refresh không tự mở tuning. Chọn config cũng không tự chạy pipeline.

Để đường nối dễ đọc, Flow C dùng bố cục snake: preflight đọc trái sang phải, child pipeline
đọc phải sang trái, rồi publish đọc trái sang phải. Dependency tùy chọn, retry, blocked,
failed và prediction logging được ghi trong tag/card thay vì vẽ mũi tên vòng qua nhiều lane.

### Luồng 1: refresh tùy chọn

```text
POST /tuning/fetch-data
-> kiểm fetch/pipeline lock
-> Popen refresh_data.py + ghi PID/log
-> fetch KBS, retry từng mã, merge/dedupe raw CSV
-> preprocess clean data
-> build feature + label + split files
-> status page đọc PID/log/report mỗi 5 giây
-> completed hoặc failed
```

Worker không lưu fingerprint. Tuning Lab tính lại dataset signature khi đọc dataset; chỉ cần
tuning lại nếu signature thay đổi.

### Luồng 2: tuning thủ công

```text
nhập params cho một model
-> validate server-side
-> TimeSeriesSplit trên TRAIN
-> append history với status ok/error
-> chọn một run ok thuộc signature hiện tại
-> lưu config cho một model
-> lặp đến khi đủ LR + RF + GB
```

Manual config lưu params và provenance CV. Pipeline chính thức vẫn chạy lại CV; ba model
tunable LR/RF/GB tune final decision threshold trên toàn TRAIN, còn Dummy giữ threshold 0,5.

### Luồng 3: pipeline chính thức

```text
POST /tuning/run-pipeline
-> UI preflight: đủ 3 config + current signature + TEST unused + locks free
-> subprocess.run + child PID lock
-> roadmap/raw check + cleanup + rebuild clean/features/labels/split
-> recompute signature
-> child recheck: TEST + config schema/fingerprint/3 keys
-> Dummy fit-only + CV/fit LR/RF/GB
-> TEST evaluate all 4; exclude Dummy before selection
-> write final model + metadata + reports
-> best-effort SQLite sync
-> write pipeline summary + TEST lock
-> model ready; release PID lock trong finally
```

Preflight bị từ chối trả HTTP 400/409 và không chạy child. Exception trong child tạo nonzero
exit, ghi log và vẫn release pipeline lock.

### Serving sau release

Serving không phải bước train tiếp theo. Khi có request sau này, Web/CLI đọc clean history,
`final_model.pkl` và metadata, dựng feature mới nhất rồi inference. Flask thử ghi prediction
log; CLI chỉ ghi khi có `--log-db`; lỗi logging không làm prediction thất bại.

## 4. Sequence dự đoán

`stock-sequence.html` chỉ vẽ request Flask để không trộn semantics của CLI:

1. Browser gửi `POST /predict` với mã cổ phiếu.
2. Flask gọi `predict_symbol()`.
3. Prediction Service đọc clean CSV, lọc lịch sử mã và gọi `build_features()`.
4. Service lấy feature row mới nhất, đúng thứ tự feature trong metadata.
5. Service đọc `final_model.pkl` và `model_metadata.json`.
6. Gradient Boosting object trong memory chạy `predict_proba()`.
7. `P(UP) >= 0.4129` cho kết quả `UP`; thấp hơn cho `NOT_UP`.
8. Service trả result dict. Flask thử ghi SQLite rồi render trang kết quả.

Nếu ghi SQLite lỗi, Flask vẫn trả dự đoán. CLI dùng cùng `predict_symbol()` và chỉ ghi DB khi
có cờ `--log-db`.

## 5. Lifecycle pipeline chính thức

`stock-lifecycle.html` mô tả một lần chạy, gồm main phases, guard/side effect và terminal
outcomes.

- `Requested -> Launch Guard`: UI kiểm config/fingerprint, test lock, fetch lock và pipeline
  lock trước khi spawn child.
- `Prepare Data`: child cleanup artifact cũ, đọc raw, clean, build feature/label và split.
- `Internal Test Guard`: fingerprint được tính lại sau rebuild rồi test lock được kiểm tra.
- `Train + Test`: chạy CV/fitting, đánh giá TEST và chọn model.
- `Write Outputs`: ghi model, metadata, report, thử sync SQLite và ghi test lock.
- `Blocked`: guard từ chối trước train.
- `Failed`: exception thoát khỏi pipeline; PID lock vẫn được release trong `finally`.
- `Model Ready`: file release hoàn tất và dùng được cho serving.

## Giới hạn cần hiểu đúng

1. `TimeSeriesSplit(gap=5)` là gap 5 **dòng** trên bảng TRAIN nhiều mã đã sắp theo thời
   gian; không thể gọi chính xác là 5 phiên giao dịch của từng mã.
2. Khi chấm CV, threshold được tune trên TRAIN fold rồi áp vào VAL fold. Khi fit artifact
   cuối, threshold được tune lại trên toàn TRAIN và lưu cùng artifact.
3. Test lock bảo vệ đường chạy official pipeline/UI. `evaluate_models.py` và
   `select_final_model.py` chạy trực tiếp không kiểm lock này.
4. `run_pipeline.py` gọi cleanup trước internal test-lock check. UI precheck tránh tình huống
   rerun cùng fingerprint làm mất artifact trước khi bị chặn; chạy script trực tiếp không có
   lớp bảo vệ sớm đó.

## Render lại bằng Archify

Chạy từ thư mục skill Archify:

```powershell
node renderers/architecture/render-architecture.mjs <repo>\docs\diagrams\v3\stock-architecture.architecture.json <repo>\docs\diagrams\v3\stock-architecture.html
node renderers/dataflow/render-dataflow.mjs <repo>\docs\diagrams\v3\stock-dataflow.dataflow.json <repo>\docs\diagrams\v3\stock-dataflow.html
node renderers/workflow/render-workflow.mjs <repo>\docs\diagrams\v3\stock-workflow.workflow.json <repo>\docs\diagrams\v3\stock-workflow.html
node renderers/sequence/render-sequence.mjs <repo>\docs\diagrams\v3\stock-sequence.sequence.json <repo>\docs\diagrams\v3\stock-sequence.html
node renderers/lifecycle/render-lifecycle.mjs <repo>\docs\diagrams\v3\stock-lifecycle.lifecycle.json <repo>\docs\diagrams\v3\stock-lifecycle.html
```
