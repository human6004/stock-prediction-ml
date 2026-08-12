# Outline slide bao ve nien luan CT239H

Thoi luong gia dinh: 15 phut trinh bay + Q&A. So slide: 24 (22 noi dung + bia + ket luan/Q&A).
Ty le 16:9, font Arial, layout tieu de + noi dung, hinh chen dang anh roi.

Guideline CT239H chi quy dinh dinh dang va noi dung BAO CAO (4 chuong: Gioi thieu,
Co so ly thuyet, Ket qua ung dung, Ket luan), khong co rang buoc rieng cho buoi bao ve.
Vi vay slide di theo dung thu tu 4 chuong cua guideline, moi chuong mot khoi slide.

| # | Slide | Chuong / muc | Nguon so lieu |
|---|-------|--------------|---------------|
| 1 | TRƯỜNG ĐẠI HỌC CẦN THƠ - TRƯỜNG CÔNG NGHỆ THÔNG TIN & TRUYỀN THÔNG | NIÊN LUẬN CƠ SỞ - HỌC PHẦN CT239H | - |
| 2 | Nội dung trình bày | Bốn chương theo guideline CT239H | - |
| 3 | HOSE có 400 mã mỗi phiên, và kết quả dự báo dễ bị rò rỉ dữ liệu | Chương 1.1 - Đặt vấn đề | reports/pipeline_summary.json |
| 4 | Mục tiêu: phân loại UP/NOT_UP cho phiên t+5, kèm giao thức kiểm chứng | Chương 1.2 - Mục tiêu và phạm vi | config/settings.py, reports/pipeline_summary.json |
| 5 | Cơ sở lý thuyết: ba họ mô hình bảng, đo bằng F1 cho lớp UP | Chương 2 - Cơ sở lý thuyết và công cụ | requirements.txt, services/model_tuning.py |
| 6 | Từ 554.897 dòng thô còn 510.862 dòng có nhãn hợp lệ | Chương 3 - Làm sạch và lọc mã đủ điều kiện | reports/pipeline_summary.json (dataset_report, clean_report), scripts/fetch_hose_data.py |
| 7 | Nhãn t+5 tính trên lịch phiên chung, tỷ lệ UP chỉ 37,6% | Chương 3 - Định nghĩa nhãn chống rò rỉ | services/feature_engineering.py, reports/pipeline_summary.json |
| 8 | 20 feature kỹ thuật, mỗi feature chỉ dùng dòng hiện tại và quá khứ | Chương 3 - Feature engineering | config/settings.py (FEATURE_COLUMNS), services/feature_engineering.py |
| 9 | Kiến trúc: pipeline offline tách khỏi tầng phục vụ dự báo | Chương 3.2.1 - Kiến trúc ứng dụng | docs/report_assets/architecture_overview.png, app.py |
| 10 | Chatbot Action-Decision: LLM chọn action, backend giữ số liệu | Chương 3 - Chức năng chatbot | docs/CHATBOT_ARCHITECTURE.md, services/chatbot_service.py, services/chatbot_tools.py |
| 11 | Demo: chatbot trả lời có số liệu, từ chối câu ngoài phạm vi | Chương 3 - Ảnh chụp phiên làm việc thật | ảnh chụp 127.0.0.1:5000/chat, docs/slides/shoot_chat.py |
| 12 | Chia theo thời gian, purge nhãn vắt biên và khóa TEST bằng fingerprint | Chương 3 - Giao thức thực nghiệm | reports/split_summary.csv, reports/pipeline_summary.json |
| 13 | Tuning bằng CV 4 fold theo ngày, purge 5 phiên, ngưỡng chọn từ OOF | Chương 3 - Thiết lập thực nghiệm | reports/tuning_results.csv, reports/cv_fold_results.csv, config/settings.py |
| 14 | Siêu tham số chốt: cây nông, lá lớn để giảm overfit | Chương 3 - Cấu hình model và độ ổn định CV | reports/best_params.json, reports/cv_fold_results.csv, models/model_metadata.json |
| 15 | Tuning Lab ghi lại 350 lần thử cấu hình, chọn tay rồi khóa lại | Chương 3 - Quy trình thử siêu tham số | experiments/tuning_history.csv, experiments/manual_config.json, services/tuning_lab.py |
| 16 | Random Forest thắng trên VALIDATION với F1_UP 0,4773 | Chương 3.3.2 - Chọn họ mô hình | reports/model_comparison.csv, models/model_metadata.json |
| 17 | Trên TEST, F1_UP 0,3754 vẫn dưới baseline always-UP 0,3839 | Chương 3.3.2 - Đánh giá TEST một lần | reports/final_model_evaluation.csv, models/model_metadata.json |
| 18 | Model bắt được 2.589 phiên UP nhưng báo động sai 6.371 lần | Chương 3.3.2 - Phân tích lỗi | reports/confusion_matrix.csv, reports/classification_report.csv |
| 19 | Biến động 20 phiên là feature quan trọng nhất, chiếm 19,7% | Chương 3.3.2 - Diễn giải mô hình | reports/feature_importance.csv |
| 20 | Sản phẩm: sáu trang web đọc lại đúng artifact đã publish | Chương 3 - Chức năng hệ thống | app.py (route), templates/, services/prediction_service.py |
| 21 | Bộ kiểm thử phủ giao thức dữ liệu, model, chatbot và UI | Chương 3.3.1 - Kế hoạch kiểm thử | tests/, reports/pipeline_summary.json |
| 22 | Hạn chế lớn nhất: chất lượng dự báo chưa vượt baseline | Chương 4.3 - Hạn chế | báo cáo mục 4.3, reports/final_model_evaluation.csv |
| 23 | Hướng phát triển: mở rộng feature và hiệu chỉnh xác suất | Chương 4.4 - Hướng phát triển | báo cáo mục 4.4 |
| 24 | Bốn mục tiêu kỹ thuật đạt, mục tiêu vượt baseline không đạt | Chương 4.1 - Kết luận | báo cáo mục 4.1, models/model_metadata.json |

## Nguyen tac noi dung
- Toi da 6 bullet/slide, moi bullet duoi 12 tu.
- So lieu nam trong bang hoac chart, khong nhet vao bullet.
- Tieu de slide phat bieu ket luan, khong phai nhan suong.
- Speaker notes 3-5 cau moi slide.
- Khong bao cao ket qua khong co trong repo; ket qua that la model CHUA vuot baseline.

## Tep sinh ra
- docs/slides/thuyet_trinh_nien_luan.pptx (ban xuat de import Canva)
- docs/slides/build_pptx.py (script dung slide, doc so lieu tu reports/)
- docs/slides/make_charts.py (script ve chart tu du lieu that)
- docs/slides/make_chatbot_chart.py (script ve so do luong chatbot)
- docs/slides/selfcheck.py + selfcheck_report.txt (doc lai pptx va kiem rang buoc)
- docs/slides/img/ (14 anh, slide dung 13: 10 chart tu ve, 1 anh chup that
  chatbot_demo.png, 2 anh co san architecture_overview.png va logo_ctu.png.
  Rieng dataflow_pipeline.png co trong img/ nhung khong slide nao chen)
