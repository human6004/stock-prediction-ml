# Bao cao AI - Slide thuyet trinh nien luan CT239H

Ban xuat: `docs/slides/thuyet_trinh_nien_luan.pptx` (24 slide, 16:9, font Arial).
Moi so lieu tren slide doc truc tiep tu file trong repo luc build, khong hardcode.

## 1. Cau truc 24 slide

| # | Slide | Muc | Hinh/Bang |
|---|-------|-----|-----------|
| 1 | TRƯỜNG ĐẠI HỌC CẦN THƠ - TRƯỜNG CÔNG NGHỆ THÔNG TIN & TRUYỀN THÔNG | NIÊN LUẬN CƠ SỞ - HỌC PHẦN CT239H | 1 hinh |
| 2 | Nội dung trình bày | Bốn chương theo guideline CT239H | - |
| 3 | HOSE có 400 mã mỗi phiên, và kết quả dự báo dễ bị rò rỉ dữ liệu | Chương 1.1 - Đặt vấn đề | 1 bang |
| 4 | Mục tiêu: phân loại UP/NOT_UP cho phiên t+5, kèm giao thức kiểm chứng | Chương 1.2 - Mục tiêu và phạm vi | 1 bang |
| 5 | Cơ sở lý thuyết: ba họ mô hình bảng, đo bằng F1 cho lớp UP | Chương 2 - Cơ sở lý thuyết và công cụ | 1 bang |
| 6 | Từ 554.897 dòng thô còn 510.862 dòng có nhãn hợp lệ | Chương 3 - Làm sạch và lọc mã đủ điều kiện | 1 hinh |
| 7 | Nhãn t+5 tính trên lịch phiên chung, tỷ lệ UP chỉ 37,6% | Chương 3 - Định nghĩa nhãn chống rò rỉ | 1 hinh |
| 8 | 20 feature kỹ thuật, mỗi feature chỉ dùng dòng hiện tại và quá khứ | Chương 3 - Feature engineering | 1 bang |
| 9 | Kiến trúc: pipeline offline tách khỏi tầng phục vụ dự báo | Chương 3.2.1 - Kiến trúc ứng dụng | 1 hinh |
| 10 | Chatbot dùng structured context injection (SCI) | Chương 3 - Chức năng chatbot | 1 hinh, 1 bang |
| 11 | Demo: chatbot trả lời có số liệu, từ chối câu ngoài phạm vi | Chương 3 - Ảnh chụp phiên làm việc thật | 1 hinh |
| 12 | Chia theo thời gian, purge nhãn vắt biên và khóa TEST bằng fingerprint | Chương 3 - Giao thức thực nghiệm | 1 hinh, 1 bang |
| 13 | Tuning bằng CV 4 fold theo ngày, purge 5 phiên, ngưỡng chọn từ OOF | Chương 3 - Thiết lập thực nghiệm | 1 hinh, 1 bang |
| 14 | Siêu tham số chốt: cây nông, lá lớn để giảm overfit | Chương 3 - Cấu hình model và độ ổn định CV | 1 hinh, 1 bang |
| 15 | Tuning Lab ghi lại 350 lần thử cấu hình, chọn tay rồi khóa lại | Chương 3 - Quy trình thử siêu tham số | 1 bang |
| 16 | Random Forest thắng trên VALIDATION với F1_UP 0,4773 | Chương 3.3.2 - Chọn họ mô hình | 1 hinh, 1 bang |
| 17 | Trên TEST, F1_UP 0,3754 vẫn dưới baseline always-UP 0,3839 | Chương 3.3.2 - Đánh giá TEST một lần | 1 hinh, 1 bang |
| 18 | Model bắt được 2.589 phiên UP nhưng báo động sai 6.371 lần | Chương 3.3.2 - Phân tích lỗi | 1 hinh |
| 19 | Biến động 20 phiên là feature quan trọng nhất, chiếm 19,7% | Chương 3.3.2 - Diễn giải mô hình | 1 hinh |
| 20 | Sản phẩm: sáu trang web đọc lại đúng artifact đã publish | Chương 3 - Chức năng hệ thống | 1 bang |
| 21 | Bộ kiểm thử phủ giao thức dữ liệu, chọn model, chatbot và UI | Chương 3.3.1 - Kế hoạch kiểm thử | 1 bang |
| 22 | Hạn chế lớn nhất: chất lượng dự báo chưa vượt baseline | Chương 4.3 - Hạn chế | - |
| 23 | Hướng phát triển: mở rộng feature và hiệu chỉnh xác suất | Chương 4.4 - Hướng phát triển | - |
| 24 | Bốn mục tiêu kỹ thuật đạt, mục tiêu vượt baseline không đạt | Chương 4.1 - Kết luận | 1 bang |

Thoi luong tinh theo 15 phut: slide 1-5 khoang 3 phut, slide 6-11 khoang 4 phut,
slide 12-15 khoang 3 phut, slide 16-19 khoang 3 phut, slide 20-24 khoang 2 phut.

## 2. Quyet dinh da chon

1. **Bam theo 4 chuong guideline.** Guideline CT239H (`CT239H-Guideline (2).docx`)
   chi quy dinh dinh dang va noi dung BAO CAO, khong co rang buoc rieng cho buoi
   bao ve (khong noi thoi luong, khong noi tieu chi cham). Vi vay slide di theo
   dung thu tu 4 chuong cua guideline de hoi dong doi chieu de.
2. **24 slide cho 15 phut**, trung binh 40 giay/slide. Da tach slide han che va
   huong phat trien thanh 2 slide vi gop lai vuot 6 bullet. Da them slide rieng
   cho sieu tham so va do on dinh CV, vi hoi dong hay hoi "tham so cu the la gi".
3. **Bao cao ket qua that: model CHUA vuot baseline.** TEST F1_UP 0,3754 so voi
   Always UP 0,3839 (`reports/final_model_evaluation.csv`). Day la huong an toan
   nhat: bao cao goc va `models/model_metadata.json` deu ghi `baseline_passed=false`,
   neu slide che con so nay thi hoi doi chieu se thanh mau thuan.
4. **Metric chinh la F1_UP, khong dung accuracy lam diem ban.** Always NOT_UP dat
   accuracy 0,7625 tren TEST, nen accuracy don doc gay hieu nham.
5. **Ve lai 10 chart de dong bo mau va font.** `docs/slides/make_charts.py` ve 9 chart
   tu `reports/*.csv` va `pipeline_summary.json`; `docs/slides/make_chatbot_chart.py`
   ve `chatbot_flow.png` theo contract SCI cua code.
   Giu lai 2 anh co san dung tren slide: `architecture_overview.png` (so do kien truc) va
   `logo_ctu.png`. Tong cong 13 anh duoc chen vao slide: 10 chart tu ve + 2 anh co san
   + 1 screenshot `chatbot_demo.png`. Trong `img/` co 14 tep; rieng
   `dataflow_pipeline.png` khong duoc dung.
6. **Confusion matrix tu ve lai** (`img/confusion_matrix_test.png`) vi ban trong
   `reports/confusion_matrix.png` khong co nhan tieng Viet va co ty le nho.
7. **So lieu chi nam trong bang/chart**, bullet khong chua con so dai. Tieu de slide
   phat bieu ket luan (vi du slide 17: "Tren TEST, F1_UP 0,3754 van duoi baseline
   always-UP 0,3839").
8. **Layout don gian**: moi slide la textbox + anh roi + bang, khong dung placeholder
   theme, khong group shape, de import Canva khong lech.
9. **Mau**: navy `#10243B`, teal `#0E7C86`, do canh bao `#B43A2E` cho so lieu khong
   dat. Font Arial toan bo (kiem lai trong pptx: chi 1 font).
10. **Speaker notes 4 cau/slide**, dat trong notes slide, viet o ngoi thu nhat de
    doc truc tiep khi bao ve.
11. **Chi tao slide demo cho trang `/chat`**, la trang duy nhat da co screenshot chup
    that trong repo (`img/chatbot_demo.png`, slide 11). Cac trang web con lai chua co
    anh chup, nen slide 9 mo ta kien truc bang so do co san thay vi anh demo.
12. **Anh demo chatbot chup that, khong dung mockup.** `docs/slides/shoot_chat.py`
    bat Flask that, dieu khien Chrome qua DevTools Protocol, go 3 cau hoi vao trang
    `/chat` roi chup. Ba cau: hoi model va ket qua TEST, hoi tin hieu FPT, hoi P/E va
    tin tuc (cau nay bi tu choi dung theo pham vi). Toan van hoi dap luu tai
    `docs/slides/_chat_transcript.txt` de doi chieu.
13. **Khong sua bat ky file nao ngoai `docs/slides/`.** Bao cao goc va code giu nguyen.

## 3. Cho hong (can bo sung thu cong)

1. **Chi co screenshot trang `/chat`** (slide 11, anh `img/chatbot_demo.png` chup
   that tu Flask dang chay). Chua co anh trang `/predict`, `/compare`, `/screener`
   va `/evaluation`; neu muon them slide demo cho cac trang do thi chup roi noi toi chen.
2. **Khong co so do "3 nguon leakage"** dang hinh; slide 3 dien dat bang bullet.
3. **Khong do thoi gian chay pipeline** trong repo (khong co log timing), nen khong
   co slide hieu nang. Neu hoi dong hoi, tra loi "chua do".
4. **Khong co ket qua kiem thu hieu nang / user acceptance** (bao cao muc 4.3 tu ghi
   nhan la chua lam). Slide 21 ghi ro dieu nay.
5. **So test thay doi theo code**, nen slide khong ghi cung tong so. Repo chua co file
   junit/coverage; truoc khi bao ve, chay `$env:PYTHONPATH="."; python -m pytest tests -q`
   va chi trinh bay ket qua cua lan chay do neu can.
6. **month la feature quan trong thu 2** nhung repo khong co phan tich vi sao;
   slide 19 chi neu day la diem can than trong, chua co bang chung dinh luong.
7. **Chatbot co so do HTML mo ta luong SCI** tai
   `docs/diagrams/luuDo/06-sequence-chatbot-rag.html`; slide 10 dung chart tu ve
   `img/chatbot_flow.png` mo ta rule-based intent routing, server-built context
   va mot LLM call.
8. **Chua do do chinh xac cau tra loi cua chatbot** (khong co eval set, khong co log
   hoi dap trong repo). Slide 10 chi noi ve rang buoc runtime va grounding;
   neu hoi dong hoi "chatbot tra loi dung bao nhieu phan tram", tra loi "chua do".
9. **`models/model_metadata.json` hien khong co `training_symbols`**, nen chatbot roi
   ve symbol scope doc tu `reports/eligible_symbols.csv` (nhanh fallback trong
   `_load_scope`). Khong sai, nhung neu bi hoi thi phai giai thich dung nhanh nay.

## 4. Rui ro hoi dong de bat bi

1. **"Model khong vuot baseline thi de tai co gia tri gi?"** Slide 17 va 24 tra loi:
   gia tri nam o giao thuc chong ro ri va tinh tai lap. Nen chuan bi noi thang:
   neu noi giao thuc (nhan theo tung ma, CV tron ngay, cham TEST nhieu lan) thi
   con so se dep hon nhung khong dung.
2. **"Tai sao chon Random Forest khi Gradient Boosting co CV F1_UP cao hon?"**
   CV: GB 0,4714 > RF 0,4703, nhung tieu chi chon la F1_UP tren VALIDATION, o do
   RF 0,4773 > GB 0,4734. Slide 13 va 16 co ca hai bang.
3. **"Metadata ghi policy_id = legacy_pre_validation_baseline_gate, con README noi
   rolling_recent_cv_oof_threshold"**. Artifact dang cong bo duoc import tu lan danh gia
   truoc (`experiments/evaluation_registry.json` ghi ro "Imported prior TEST
   evaluation without re-evaluation"). Slide khong noi ve policy_id de tranh mau
   thuan; neu bi hoi, giai thich dung nhu tren.
4. **Moc thoi gian TEST (13/04/2026 - 13/07/2026)** la moc suy dong tu phien moi nhat
   cua dataset, khong phai hang so trong `config/settings.py` (3 hang so do chi la
   fallback, va README khong con neu moc TEST co dinh nao). Slide dung moc thuc te
   trong `reports/split_summary.csv`, trung voi `models/model_metadata.json`.
5. **Ty le UP tren TEST chi 23,8%** so voi 37,6% toan bo du lieu. Do la ly do chinh
   precision tut xuong 0,289. Slide 18 da noi truoc diem nay.
6. **Feature month** co the bi hoi la ro ri chu ky; tra loi: month chi la thang
   duong lich cua chinh dong do, khong dung thong tin tuong lai, nhung dung y kien
   la no co the phan anh che do thi truong trong giai doan train.
7. **"Chatbot nay co dung RAG khong?"** Tra loi: khong phai RAG. Day la structured
   context injection: khong co vector DB, khong co embedding, khong co corpus van ban.
   Server khop keyword de chon nguon du lieu (rule-based intent routing, 6 handler
   dong), doc artifact da publish (`model_metadata.json`, `reports/*.csv`), tiem vao
   prompt dang CONTEXT_JSON roi goi LLM dung mot lan; provider khong nhan tool schema.
   Grounding validation cua server chan so khong co trong context.
8. **"Tuning tay 350 run co phai la khoa hoc?"** Tra loi: moi run duoc ghi vao
   `experiments/tuning_history.csv` kem params va F1_UP tung fold, nen truy vet duoc;
   va toan bo tuning chi cham tren TRAIN. Diem yeu that su can thua nhan: so run lech
   nhau giua ba model (LogReg 269, RF 51, GB 30), nen khong the noi ba model duoc do
   voi cung ngan sach tim kiem.
9. **"Web co tu train lai model khong?"** Trang `/predict`, `/compare`, `/screener`,
   `/evaluation` chi doc hien vat. Nhung `/tuning/run-pipeline` trong app.py THUC SU
   chay duoc pipeline chinh thuc; no bi chan boi ba dieu kien (du config 3 model,
   dung dataset fingerprint, snapshot chua tung duoc danh gia). Slide 20 va notes
   cua no da noi dung nhu vay, khong noi "khong trang nao train lai".

## 5. Import vao Canva

1. Mo link Canva dich, chon **File > Import > Upload files**, chon
   `docs/slides/thuyet_trinh_nien_luan.pptx`. Canva se tao design moi tu file nay;
   noi dung cu trong design dich khong bi ghi de.
2. Canva khong import speaker notes trong moi truong hop. Sau khi import, kiem
   phan **Notes** o slide 1 va 17; neu trong, copy lai tu ban pptx (mo bang
   PowerPoint hoac chay `python docs/slides/selfcheck.py` roi lay tu
   `docs/slides/selfcheck_report.txt`).
3. Font Arial co san tren Canva. Neu Canva doi sang font khac, chon toan bo slide
   va set lai Arial.
4. Slide nen chinh tay sau import:
   - Slide 1 (bia): kiem logo CTU khong bi keo lech ty le.
   - Slide 8 va 21: bang 7-8 dong, Canva hay lam chu nho lai; kiem con doc duoc.
   - Slide 12, 13, 14, 16, 17: moi slide co ca bang va chart, kiem khong de len nhau.
5. Neu can dung lai file: `.\.venv\Scripts\python.exe docs/slides/make_charts.py`
   roi `.\.venv\Scripts\python.exe docs/slides/build_pptx.py`.

## 6. Ket qua tu kiem

Chay `docs/slides/selfcheck.py` (doc lai chinh file pptx da xuat):

- 24 slide, ty le 12191695x6858000 EMU = 1,7777 (16:9).
- 0 slide vuot 6 bullet; 0 bullet tu 12 tu tro len.
- 24/24 slide co speaker notes, moi slide 4 cau.
- 13/13 anh duoc chen ton tai tren disk va da nhung vao file (kiem ca duong dan va blob).
  `docs/slides/img/` co 14 tep; `dataflow_pipeline.png` khong duoc slide nao dung nen khong nhung.
- Chi mot font duoc dung: Arial.
- `total_problems=0` trong `docs/slides/selfcheck_report.txt`.

Nam cau hoi phan bien, tra loi chi bang text doc ra tu file pptx:

1. *Du lieu o dau?* Slide 3 va 6: 400 ma, 554.897 dong OHLCV, 2019-08-14 den
   2026-07-20, tai qua vnstock luu CSV chia se; nguon ghi o chan slide.
2. *Sao chon model nay?* Slide 13, 14 va 16: CV 4 fold chon sieu tham so (n_estimators=130,
   max_depth=8, min_samples_leaf=100, max_features=0,2), chon ho model
   theo F1_UP tren VALIDATION, RF 0,4773 cao nhat.
3. *Ket qua so baseline the nao?* Slide 17: RF 0,3754 so Always UP 0,3839 va
   Always NOT_UP 0,0000 tren TEST.
4. *Co leakage khong?* Slide 7, 8, 12, 13: nhan theo lich phien chung, 4 cot tuong
   lai bi chan khoi feature, purge 1.883 + 1.775 dong o hai bien, TEST cham mot lan
   co fingerprint khoa.
5. *Han che gi?* Slide 22: chua vuot baseline, chi dung gia va khoi luong, nguong co
   dinh, du lieu tinh, web chua co xac thuc.
