"""Front matter: bia, abstract, muc luc, danh muc bang/hinh/viet tat."""

from __future__ import annotations

from docx_builder import DocxBuilder

SINH_VIEN = "Lê Trần Hiếu Nhân"
MSSV = "B2308203"
HOC_VI_GVHD = "ThS."
GIANG_VIEN = "Phan Phương Lan"
NGANH = "Công nghệ thông tin (Chương trình chất lượng cao)"
KHOA = "49"
LOP = "Niên luận cơ sở - CT239H"
HOC_PHAN = "NIÊN LUẬN CƠ SỞ - CÁC ĐỀ TÀI CƠ BẢN"
MA_HOC_PHAN = "CT239H"
HOC_KY = "HỌC KỲ III, NĂM HỌC 2025 - 2026"
DIA_DIEM_NAM = "CẦN THƠ, 2026"


def _cover_block(doc: DocxBuilder, assets) -> None:
    """Mot trang bia theo layout va cac truong bat buoc cua quy dinh CT239H."""
    doc.paragraph("BỘ GIÁO DỤC VÀ ĐÀO TẠO", style="CoverText")
    doc.paragraph("TRƯỜNG ĐẠI HỌC CẦN THƠ", style="CoverText")
    doc.paragraph("TRƯỜNG CÔNG NGHỆ THÔNG TIN & TRUYỀN THÔNG", style="CoverText")
    logo = assets / "logo_ctu.png"
    if logo.exists():
        doc.image(logo, width_cm=3.8, page_center=True)
    else:
        doc.paragraph("", style="CoverText")
    doc.paragraph("HỌC PHẦN: NIÊN LUẬN CƠ SỞ - CÁC ĐỀ TÀI CƠ BẢN", style="CoverHeading")
    doc.paragraph(f"MÃ HỌC PHẦN: {MA_HOC_PHAN}", style="CoverSub")
    doc.paragraph(f"NGÀNH: {NGANH.upper()}", style="CoverSub")
    doc.paragraph(f"KHÓA: {KHOA}", style="CoverSub")
    doc.paragraph("", style="CoverText")
    doc.paragraph("Đề tài", style="CoverSub")
    doc.paragraph(
        "DỰ BÁO XU HƯỚNG GIÁ CỔ PHIẾU HOSE BẰNG MACHINE LEARNING",
        style="CoverTitle",
    )
    doc.paragraph("", style="CoverText")
    doc.paragraph("Giảng viên hướng dẫn:", style="CoverSub")
    doc.paragraph(f"{HOC_VI_GVHD} {GIANG_VIEN}", style="CoverSub")
    doc.paragraph("", style="CoverText")
    doc.paragraph("Sinh viên thực hiện:", style="CoverSub")
    doc.paragraph(f"1. {SINH_VIEN} - MSSV: {MSSV}", style="CoverSub")
    doc.paragraph(f"Lớp: {LOP} - Khóa {KHOA}", style="CoverSub")
    doc.paragraph("", style="CoverText")
    doc.paragraph(HOC_KY, style="CoverSub")
    doc.paragraph(DIA_DIEM_NAM, style="CoverSub")


def build_cover(doc: DocxBuilder, assets) -> None:
    """Bia ngoai va bia trong, giong cach template dung hai trang bia."""
    _cover_block(doc, assets)
    doc.page_break()
    _cover_block(doc, assets)
    doc.page_break()


ABSTRACT_TEXT = (
    "Niên luận xây dựng hệ thống học máy dự báo xu hướng ngắn hạn của cổ phiếu HOSE: xác "
    "định một mã có tăng hơn 1% tại đúng phiên giao dịch thứ năm sau ngày tham chiếu hay "
    "không, tương ứng hai nhãn UP và NOT_UP. Từ 554.897 dòng OHLCV của 400 mã, pipeline sinh "
    "20 đặc trưng kỹ thuật, gán nhãn theo lịch phiên chung của thị trường, chia "
    "TRAIN/VALIDATION/TEST theo thời gian có purge nhãn, kiểm định chéo bốn fold cách nhau "
    "năm phiên, chọn họ mô hình trên VALIDATION và đánh giá TEST đúng một lần. Ba mô hình "
    "được so sánh là Logistic Regression, Random Forest và Gradient Boosting. Random Forest "
    "được chọn với F1_UP 0,4773 trên VALIDATION, đạt F1_UP 0,3754 và độ chính xác 0,5766 "
    "trên TEST gồm 20.350 dòng, vẫn thấp hơn baseline luôn dự báo UP (0,3839), nên hệ thống "
    "ghi baseline_passed = false và cảnh báo trên giao diện. Sản phẩm gồm ứng dụng web Flask "
    "cho dự báo, so sánh, xếp hạng, đánh giá, Tuning Lab và chatbot chỉ dùng dữ liệu nội bộ. "
    "Giá trị chính là quy trình thực nghiệm chống rò rỉ dữ liệu."
)


def build_work_assignment(doc: DocxBuilder) -> None:
    """Muc phan cong cong viec theo template; nien luan nay do mot sinh vien thuc hien."""
    doc.heading(1, "PHÂN CÔNG CÔNG VIỆC")
    doc.paragraph(
        "Niên luận do một sinh viên thực hiện, nên toàn bộ các công việc dưới đây đều thuộc "
        "trách nhiệm của cá nhân. Bảng này ghi lại các nhóm công việc chính và sản phẩm "
        "tương ứng để thuần tiện đối chiếu với nội dung báo cáo."
    )
    doc.table(
        ["STT", "Công việc", "Người thực hiện", "Sản phẩm"],
        [
            ["1", "Thu thập và làm sạch dự liệu OHLCV HOSE",
             f"{SINH_VIEN} ({MSSV})", "scripts/fetch_hose_data.py, services/preprocessing.py"],
            ["2", "Sinh 20 đặc trưng kỹ thuật và gán nhãn t+5",
             f"{SINH_VIEN} ({MSSV})", "services/feature_engineering.py, ml_dataset.csv"],
            ["3", "Thiết kế giao thức chống rò rỉ và chia dự liệu",
             f"{SINH_VIEN} ({MSSV})", "services/protocol_dates.py, services/time_splitting.py"],
            ["4", "Tinh chỉnh siêu tham số ba mô hình",
             f"{SINH_VIEN} ({MSSV})", "services/model_tuning.py, services/tuning_lab.py"],
            ["5", "Chọn mô hình, đánh giá TEST và phát hành hiện vật",
             f"{SINH_VIEN} ({MSSV})", "services/model_evaluation.py, scripts/run_pipeline.py"],
            ["6", "Xây dựng ứng dụng web Flask và chatbot",
             f"{SINH_VIEN} ({MSSV})", "app.py, templates/, services/chatbot_*.py"],
            ["7", "Viết kiểm thứ tự động và tài liệu",
             f"{SINH_VIEN} ({MSSV})", "tests/ (183 ca), docs/, báo cáo này"],
        ],
        widths=[700, 3500, 2100, 2670],
        caption="Phân công công việc trong niên luận",
    )
    doc.page_break()

def build_abstract(doc: DocxBuilder) -> None:
    doc.heading(1, "TÓM TẮT")
    doc.paragraph(ABSTRACT_TEXT)
    doc.paragraph(
        "Từ khóa: dự báo xu hướng cổ phiếu, HOSE, phân loại nhị phân, kiểm định chéo "
        "chuỗi thời gian, chống rò rỉ dữ liệu, Random Forest, Flask.",
        italic=True,
    )
    doc.page_break()


def build_toc(doc: DocxBuilder) -> None:
    doc.heading(1, "MỤC LỤC")
    doc.paragraph(
        "Mục lục dưới đây được sinh tự động bằng trường TOC của Word ở bốn cấp tiêu đề. "
        "Sau khi mở tài liệu, nhấn Ctrl+A rồi F9 và chọn Update entire table để cập nhật "
        "số trang.",
        italic=True,
    )
    doc.toc(levels="1-4")
    doc.page_break()

    doc.heading(1, "DANH MỤC BẢNG")
    doc.table_of_captions("Bang")
    doc.page_break()

    doc.heading(1, "DANH MỤC HÌNH")
    doc.table_of_captions("Hinh")
    doc.page_break()


ABBREVIATIONS = [
    ["API", "Application Programming Interface", "Giao diện lập trình ứng dụng"],
    ["CLI", "Command Line Interface", "Giao diện dòng lệnh"],
    ["CSV", "Comma Separated Values", "Tệp dữ liệu phân tách bằng dấu phẩy"],
    ["CV", "Cross Validation", "Kiểm định chéo"],
    ["EMA/SMA", "Simple Moving Average", "Trung bình động đơn giản"],
    ["F1_UP", "F1 score of the UP class", "Điểm F1 của lớp UP"],
    ["GB", "Gradient Boosting", "Mô hình tăng cường độ dốc"],
    ["HOSE", "Ho Chi Minh Stock Exchange", "Sàn giao dịch chứng khoán TP. Hồ Chí Minh"],
    ["LLM", "Large Language Model", "Mô hình ngôn ngữ lớn"],
    ["LR", "Logistic Regression", "Hồi quy logistic"],
    ["ML", "Machine Learning", "Học máy"],
    ["OHLCV", "Open High Low Close Volume", "Giá mở, cao, thấp, đóng và khối lượng"],
    ["OOF", "Out Of Fold", "Dự báo ngoài fold trong kiểm định chéo"],
    ["RF", "Random Forest", "Rừng ngẫu nhiên"],
    ["RSI", "Relative Strength Index", "Chỉ số sức mạnh tương đối"],
    ["SRS", "Software Requirement Specification", "Đặc tả yêu cầu phần mềm"],
    ["UI", "User Interface", "Giao diện người dùng"],
]


def build_abbreviations(doc: DocxBuilder) -> None:
    doc.heading(1, "DANH MỤC TỪ VIẾT TẮT")
    doc.table(
        ["Viết tắt", "Dạng đầy đủ", "Nghĩa sử dụng trong báo cáo"],
        ABBREVIATIONS,
        widths=[1300, 3300, 4470],
        caption="Danh mục từ viết tắt dùng trong báo cáo",
    )
    doc.page_break()
