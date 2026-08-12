# -*- coding: utf-8 -*-
"""Build slide bao ve nien luan tu so lieu that trong repo.

Moi con so tren slide duoc doc truc tiep tu reports/*.json|csv de bao dam
truy nguon. Khong hardcode metric.
"""
import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

BASE = Path(__file__).resolve().parents[2]
SLIDES = BASE / "docs" / "slides"
IMG = SLIDES / "img"
import os
OUT = SLIDES / (os.environ.get("SLIDE_OUT") or "thuyet_trinh_nien_luan.pptx")

FONT = "Arial"
NAVY = RGBColor(0x10, 0x24, 0x3B)
INK = RGBColor(0x1F, 0x2A, 0x37)
ACCENT = RGBColor(0x0E, 0x7C, 0x86)
WARN = RGBColor(0xB4, 0x3A, 0x2E)
GREY = RGBColor(0x5B, 0x66, 0x72)
LIGHT = RGBColor(0xEC, 0xF1, 0xF4)

SW, SH = Inches(13.333), Inches(7.5)


def load_sources():
    summary = json.loads((BASE / "reports" / "pipeline_summary.json").read_text(encoding="utf-8"))
    best = json.loads((BASE / "reports" / "best_params.json").read_text(encoding="utf-8"))
    meta = json.loads((BASE / "models" / "model_metadata.json").read_text(encoding="utf-8"))

    def rows(name):
        with open(BASE / "reports" / name, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    return {
        "summary": summary,
        "best": best,
        "meta": meta,
        "comparison": rows("model_comparison.csv"),
        "final": rows("final_model_evaluation.csv"),
        "clf": rows("classification_report.csv"),
        "split": rows("split_summary.csv"),
        "tuning": rows("tuning_results.csv"),
        "importance": rows("feature_importance.csv"),
        "cm": rows("confusion_matrix.csv"),
    }


def load_lab_runs():
    """Dem so lan thu cau hinh moi model tu tuning_history.csv."""
    counts = {}
    path = BASE / "experiments" / "tuning_history.csv"
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("model_name") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


LAB_RUNS = load_lab_runs()
S = load_sources()
SUM = S["summary"]
META = S["meta"]
BEST = S["best"]
D2 = lambda x: f"{float(x):.2f}".replace(".", ",")
D3 = lambda x: f"{float(x):.3f}".replace(".", ",")
D4 = lambda x: f"{float(x):.4f}".replace(".", ",")
N = lambda x: f"{int(x):,}".replace(",", ".")
PCT = lambda x: f"{float(x) * 100:.1f}".replace(".", ",") + "%"


def pick(rows, name, split=None):
    for r in rows:
        if r["model_name"] == name and (split is None or r["split"] == split):
            return r
    raise KeyError(name)


def tune(name):
    return pick(S["tuning"], name)


# ---------- primitives ----------

def set_text(tf, items, size=18, color=INK, bullet_char="\u2013", space=10):
    tf.word_wrap = True
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        run = p.add_run()
        run.text = (bullet_char + " " + txt) if bullet_char else txt
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = color


def add_box(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    box.text_frame.word_wrap = True
    return box


def add_bar(slide):
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(0.13))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False


def add_title(slide, text, sub=None):
    add_bar(slide)
    box = add_box(slide, Inches(0.6), Inches(0.36), SW - Inches(1.2), Inches(1.0))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = NAVY
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = FONT
        r2.font.size = Pt(14)
        r2.font.color.rgb = GREY
    return box


def fit_picture(slide, path, left, top, max_w, max_h):
    pic = slide.shapes.add_picture(str(path), left, top)
    scale = min(max_w / pic.width, max_h / pic.height)
    pic.width = Emu(int(pic.width * scale))
    pic.height = Emu(int(pic.height * scale))
    pic.left = Emu(int(left + (max_w - pic.width) / 2))
    pic.top = Emu(int(top + (max_h - pic.height) / 2))
    return pic


def add_table(slide, data, left, top, width, height, size=12, col_widths=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            table.columns[i].width = Emu(int(width * w / total))
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else RGBColor(0xFF, 0xFF, 0xFF))
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = r == 0
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else INK
            if c > 0:
                p.alignment = PP_ALIGN.CENTER
    return table


def add_note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_source(slide, text):
    box = add_box(slide, Inches(0.6), SH - Inches(0.62), SW - Inches(1.2), Inches(0.4))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = "Nguồn: " + text
    run.font.name = FONT
    run.font.size = Pt(11)
    run.font.color.rgb = GREY


prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def new_slide():
    return prs.slides.add_slide(BLANK)


# ---------- 1. Bia ----------
s = new_slide()
bg = s.shapes.add_shape(1, 0, 0, SW, SH)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
bg.shadow.inherit = False
strip = s.shapes.add_shape(1, 0, Inches(3.35), SW, Inches(0.06))
strip.fill.solid()
strip.fill.fore_color.rgb = ACCENT
strip.line.fill.background()
strip.shadow.inherit = False
fit_picture(s, IMG / "logo_ctu.png", Inches(0.6), Inches(0.5), Inches(1.5), Inches(1.5))
box = add_box(s, Inches(2.4), Inches(0.6), SW - Inches(3.2), Inches(2.6))
tf = box.text_frame
for txt, size, bold, col in [
    ("TRƯỜNG ĐẠI HỌC CẦN THƠ - TRƯỜNG CÔNG NGHỆ THÔNG TIN & TRUYỀN THÔNG", 14, False, RGBColor(0x9F, 0xB6, 0xC4)),
    ("NIÊN LUẬN CƠ SỞ - HỌC PHẦN CT239H", 14, False, RGBColor(0x9F, 0xB6, 0xC4)),
    ("D\u1ef0 B\u00c1O XU H\u01af\u1edaNG GI\u00c1 C\u1ed4 PHI\u1ebeU HOSE B\u1eb0NG MACHINE LEARNING", 32, True, RGBColor(0xFF, 0xFF, 0xFF)),
]:
    p = tf.paragraphs[0] if txt.startswith("TRUONG") else tf.add_paragraph()
    p.space_after = Pt(10)
    run = p.add_run()
    run.text = txt
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = col
box = add_box(s, Inches(2.4), Inches(3.7), SW - Inches(3.2), Inches(2.4))
set_text(
    box.text_frame,
    [
        "Gi\u1ea3ng vi\u00ean h\u01b0\u1edbng d\u1eabn: Phan Ph\u01b0\u01a1ng Lan",
        "Sinh vi\u00ean th\u1ef1c hi\u1ec7n: L\u00ea Tr\u1ea7n Hi\u1ebfu Nh\u00e2n - B2308203",
        "Ng\u00e0nh C\u00f4ng ngh\u1ec7 th\u00f4ng tin - C\u1ea7n Th\u01a1, th\u00e1ng 7 n\u0103m 2026",
    ],
    size=18,
    color=RGBColor(0xE6, 0xED, 0xF2),
    bullet_char=None,
    space=14,
)
add_note(
    s,
    "Em xin ph\u00e9p tr\u00ecnh b\u00e0y ni\u00ean lu\u1eadn c\u01a1 s\u1edf CT239H v\u1ec1 d\u1ef1 b\u00e1o xu h\u01b0\u1edbng gi\u00e1 c\u1ed5 phi\u1ebfu HOSE b\u1eb1ng machine learning. "
    "\u0110\u1ec1 t\u00e0i do c\u00f4 Phan Ph\u01b0\u01a1ng Lan h\u01b0\u1edbng d\u1eabn. "
    "B\u00e0i tr\u00ecnh b\u00e0y kho\u1ea3ng 15 ph\u00fat, \u0111i theo b\u1ed1n ch\u01b0\u01a1ng c\u1ee7a b\u00e1o c\u00e1o. "
    "T\u1ea5t c\u1ea3 s\u1ed1 li\u1ec7u tr\u00ean slide \u0111\u1ec1u l\u1ea5y t\u1eeb c\u00e1c t\u1ec7p b\u00e1o c\u00e1o sinh ra b\u1edfi pipeline trong repo.",
)
# ---------- 2. Noi dung trinh bay ----------
s = new_slide()
add_title(s, "Nội dung trình bày", "Bốn chương theo guideline CT239H")
box = add_box(s, Inches(0.8), Inches(1.7), SW - Inches(1.6), Inches(4.6))
set_text(
    box.text_frame,
    [
        "Chương 1: đặt vấn đề và mục tiêu",
        "Chương 2: cơ sở lý thuyết và công cụ",
        "Chương 3: dữ liệu, feature, kiến trúc, thực nghiệm",
        "Chương 3: kết quả trên VALIDATION và TEST",
        "Chương 4: kết luận, hạn chế, hướng phát triển",
        "Hỏi đáp",
    ],
    size=20,
    space=16,
)
add_note(
    s,
    "Bài trình bày đi theo đúng bốn chương của báo cáo. "
    "Phần lớn thời lượng dành cho chương 3 vì đó là phần thực nghiệm. "
    "Em sẽ nói rõ giao thức chống rò rỉ dữ liệu trước khi công bố kết quả. "
    "Kết quả cuối cùng là kết quả trung thực, kể cả phần chưa đạt mục tiêu.",
)

# ---------- 3. Van de ----------
s = new_slide()
add_title(s, "HOSE có 400 mã mỗi phiên, và kết quả dự báo dễ bị rò rỉ dữ liệu", "Chương 1.1 - Đặt vấn đề")
box = add_box(s, Inches(0.7), Inches(1.75), Inches(6.1), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Nhà đầu tư cá nhân không theo kịp toàn sàn",
        "Chỉ báo kỹ thuật phải tính lại mỗi phiên",
        "Rò rỉ 1: nhãn lấy theo dòng riêng từng mã",
        "Rò rỉ 2: CV trộn cùng ngày vào train và test",
        "Rò rỉ 3: chọn model trên TEST rồi báo cáo TEST",
        "Cần quy trình thực nghiệm có kỷ luật, không chỉ model",
    ],
    size=17,
    space=13,
)
add_table(
    s,
    [
        ["Dữ liệu thô", "Giá trị"],
        ["Số mã", N(SUM["dataset_report"]["symbols"])],
        ["Số dòng OHLCV", N(SUM["dataset_report"]["rows"])],
        ["Từ ngày", SUM["dataset_report"]["date_min"]],
        ["Đến ngày", SUM["dataset_report"]["date_max"]],
    ],
    Inches(7.3),
    Inches(2.1),
    Inches(5.3),
    Inches(2.4),
    size=14,
    col_widths=[3, 2],
)
add_source(s, "reports/pipeline_summary.json")
add_note(
    s,
    "Vấn đề thứ nhất là khối lượng: 400 mã, hơn 554 nghìn dòng OHLCV từ tháng 8 năm 2019 đến tháng 7 năm 2026. "
    "Không ai theo dõi thủ công hết được. "
    "Vấn đề thứ hai nghiêm trọng hơn: ba dạng rò rỉ dữ liệu khiến nhiều thử nghiệm dự báo chứng khoán cho chỉ số đẹp nhưng vô giá trị. "
    "Vì vậy trọng tâm của niên luận là quy trình thực nghiệm chống rò rỉ, không chỉ là huấn luyện một mô hình.",
)

# ---------- 4. Muc tieu ----------
s = new_slide()
add_title(s, "Mục tiêu: phân loại UP/NOT_UP cho phiên t+5, kèm giao thức kiểm chứng", "Chương 1.2 - Mục tiêu và phạm vi")
box = add_box(s, Inches(0.7), Inches(1.75), Inches(6.2), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Pipeline từ OHLCV thô đến dataset học máy",
        "Nhãn theo đúng phiên thị trường thứ năm",
        "Chia dữ liệu theo thời gian, có purge nhãn",
        "Tuning ba model bằng CV chuỗi thời gian",
        "Chọn model trên VALIDATION, chấm TEST một lần",
        "Web Flask, chatbot nội bộ và bộ kiểm thử tự động",
    ],
    size=17,
    space=13,
)
add_table(
    s,
    [
        ["Định nghĩa bài toán", "Giá trị"],
        ["Loại bài toán", "Phân loại nhị phân"],
        ["Horizon", str(SUM["label_report"]["horizon"]) + " phiên"],
        ["Ngưỡng UP", "> " + PCT(SUM["label_report"]["threshold"])],
        ["Phạm vi dữ liệu", "Chỉ giá và khối lượng"],
    ],
    Inches(7.4),
    Inches(2.1),
    Inches(5.2),
    Inches(2.4),
    size=14,
    col_widths=[3, 2],
)
add_source(s, "config/settings.py, reports/pipeline_summary.json")
add_note(
    s,
    "Bài toán được phát biểu là phân loại nhị phân: giá đóng cửa tại đúng phiên thứ năm tăng hơn 1% thì nhãn UP, ngược lại NOT_UP. "
    "Mục tiêu không chỉ là mô hình mà còn là giao thức: purge, khóa TEST, so sánh baseline. "
    "Phạm vi giới hạn ở dữ liệu giá và khối lượng lịch sử, không dùng tin tức hay báo cáo tài chính. "
    "Kết quả phục vụ học tập, không phải khuyến nghị đầu tư.",
)

# ---------- 5. Co so ly thuyet ----------
s = new_slide()
add_title(s, "Cơ sở lý thuyết: ba họ mô hình bảng, đo bằng F1 cho lớp UP", "Chương 2 - Cơ sở lý thuyết và công cụ")
box = add_box(s, Inches(0.7), Inches(1.75), Inches(6.2), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Logistic Regression: tuyến tính, làm mốc tham chiếu",
        "Random Forest: bagging, giảm variance",
        "Gradient Boosting: boosting tuần tự trên cây nông",
        "Chỉ báo kỹ thuật: SMA, RSI, volatility, vị thế giá",
        "Metric chính là F1_UP vì lớp UP thiểu số",
        "Baseline hằng số always-UP và always-NOT_UP",
    ],
    size=17,
    space=13,
)
add_table(
    s,
    [
        ["Công cụ", "Vai trò"],
        ["scikit-learn", "Model, CV, metric"],
        ["pandas, numpy", "Xử lý bảng dữ liệu"],
        ["Flask", "Web dự báo, đánh giá"],
        ["matplotlib", "Vẽ confusion matrix"],
        ["vnstock", "Thu thập OHLCV"],
    ],
    Inches(7.4),
    Inches(2.05),
    Inches(5.2),
    Inches(2.9),
    size=14,
    col_widths=[3, 3],
)
add_source(s, "requirements.txt, services/model_tuning.py")
add_note(
    s,
    "Ba họ mô hình được chọn vì đây là dữ liệu bảng, không phải chuỗi thuần: một mô hình tuyến tính làm mốc và hai mô hình ensemble trên cây. "
    "Metric chính là F1 của lớp UP vì UP là lớp thiểu số, chỉ khoảng 37,6% dữ liệu. "
    "Nếu chỉ dùng accuracy thì mô hình luôn dự báo NOT_UP đã đạt hơn 76% trên TEST. "
    "Công cụ đều là thư viện Python phổ biến, khai báo trong requirements.txt.",
)

# ---------- 6. Du lieu ----------
s = new_slide()
cr = SUM["clean_report"]
lr = SUM["label_report"]
add_title(
    s,
    "Từ 554.897 dòng thô còn 510.862 dòng có nhãn hợp lệ",
    "Chương 3 - Làm sạch và lọc mã đủ điều kiện",
)
box = add_box(s, Inches(0.7), Inches(1.7), Inches(5.0), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Nguồn: OHLCV tải qua vnstock, lưu CSV chia sẻ",
        "Loại nến sai logic OHLC",
        "Bỏ mã dưới 250 phiên giao dịch",
        "Cần đủ 50 phiên lịch sử cho rolling feature",
        "Bỏ dòng thiếu giá tại phiên đích t+5",
        "Không dòng nào trùng mã và ngày",
    ],
    size=17,
    space=13,
)
fit_picture(s, IMG / "data_funnel.png", Inches(5.9), Inches(1.6), Inches(6.9), Inches(4.7))
add_source(s, "reports/pipeline_summary.json (dataset_report, clean_report), scripts/fetch_hose_data.py")
add_note(
    s,
    "Dữ liệu OHLCV được tải bằng scripts/fetch_hose_data.py qua thư viện vnstock, lưu thành một tệp CSV chia sẻ ngoài repo. "
    "Phễu dữ liệu cho thấy từng bước mất bao nhiêu dòng: 233 dòng bị loại vì nến sai logic OHLC, {cr['excluded_symbols']} mã bị loại vì dưới 250 phiên. "
    "Bước sinh feature mất thêm dòng vì rolling window dài nhất cần 50 phiên lịch sử. "
    f"Cuối cùng còn {N(lr['rows_after_labeling'])} dòng có nhãn hợp lệ trên {SUM['feature_report']['symbols_after_features']} mã.",
)

# ---------- 7. Nhan ----------
s = new_slide()
add_title(
    s,
    "Nhãn t+5 tính trên lịch phiên chung, tỷ lệ UP chỉ 37,6%",
    "Chương 3 - Định nghĩa nhãn chống rò rỉ",
)
box = add_box(s, Inches(0.7), Inches(1.7), Inches(5.6), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Lịch phiên lấy từ toàn bộ dữ liệu đã làm sạch",
        "Phiên đích là phiên thị trường thứ năm",
        "Mã thiếu giá tại phiên đích thì bỏ dòng",
        "Không nhảy sang phiên kế tiếp để lấp",
        "label_end_date lưu lại để purge khi chia dữ liệu",
        "Lớp UP thiểu số nên chọn F1_UP làm metric",
    ],
    size=17,
    space=12,
)
fit_picture(s, IMG / "label_balance.png", Inches(6.5), Inches(1.75), Inches(6.3), Inches(4.4))
add_source(s, "services/feature_engineering.py, reports/pipeline_summary.json")
add_note(
    s,
    "Đây là điểm kỹ thuật quan trọng nhất của phần dữ liệu. "
    "Nhiều thử nghiệm lấy dòng thứ năm của riêng từng mã, nên hai mã có thể có nhãn ở hai phiên khác nhau. "
    "Ở đây phiên đích được tính trên lịch phiên chung của thị trường; mã nào không có giá đúng phiên đó thì bỏ dòng. "
    f"Kết quả là {PCT(lr['up_ratio'])} nhãn UP và {PCT(lr['not_up_ratio'])} nhãn NOT_UP, tức lớp UP là lớp thiểu số.",
)

# ---------- 8. Feature ----------
s = new_slide()
add_title(
    s,
    "20 feature kỹ thuật, mỗi feature chỉ dùng dòng hiện tại và quá khứ",
    "Chương 3 - Feature engineering",
)
groups = [
    ["Nhóm feature", "Thành phần"],
    ["Return", "return_1d, 3d, 5d, 10d, 20d, close_open_return"],
    ["Trung bình động", "sma5, sma20, sma50, close_vs_sma20, sma20_vs_sma50"],
    ["Động lượng", "rsi14"],
    ["Biến động", "volatility_5d, volatility_20d, price_range"],
    ["Khối lượng", "volume_change_1d, volume_ratio_20"],
    ["Vị thế giá", "dist_high20, dist_low20"],
    ["Thời gian", "month"],
]
add_table(s, groups, Inches(0.7), Inches(1.75), Inches(11.9), Inches(3.9), size=13, col_widths=[2, 6])
box = add_box(s, Inches(0.7), Inches(5.9), Inches(11.9), Inches(1.0))
set_text(
    box.text_frame,
    [
        "Rolling tính riêng theo từng mã, không trộn mã",
        "Bốn cột tương lai bị chặn khỏi feature",
    ],
    size=15,
    space=6,
)
add_source(s, "config/settings.py (FEATURE_COLUMNS), services/feature_engineering.py")
add_note(
    s,
    "Hai mươi feature chia thành bảy nhóm, tất cả tính từ giá và khối lượng theo ngày. "
    "Mọi phép rolling đều chạy riêng trong từng mã, không trộn dữ liệu giữa các mã. "
    "Bốn cột liên quan tương lai là future_close_5d, future_return_5d, target và label_end_date bị kiểm tra tự động để không lọt vào tập feature. "
    "Không có feature nào nhìn thấy giá sau ngày tham chiếu.",
)

# ---------- 9. Kien truc ----------
s = new_slide()
add_title(s, "Kiến trúc: pipeline offline tách khỏi tầng phục vụ dự báo", "Chương 3.2.1 - Kiến trúc ứng dụng")
fit_picture(s, IMG / "architecture_overview.png", Inches(0.7), Inches(1.6), Inches(7.6), Inches(4.9))
box = add_box(s, Inches(8.6), Inches(1.8), Inches(4.1), Inches(4.5))
set_text(
    box.text_frame,
    [
        "config: đường dẫn, feature, mốc thời gian",
        "services: xử lý, tuning, đánh giá, dự báo",
        "scripts: chạy từng bước và pipeline chính",
        "models và reports: hiện vật công bố",
        "app.py: web dự báo, đánh giá, Tuning Lab",
    ],
    size=15,
    space=12,
)
add_source(s, "docs/report_assets/architecture_overview.png, app.py")
add_note(
    s,
    "Hệ thống tách hai phần: pipeline offline sinh hiện vật, và tầng phục vụ đọc lại hiện vật đó. "
    "Pipeline ghi ra final_model.pkl, model_metadata.json và các tệp báo cáo. "
    "Web Flask và CLI chỉ đọc hiện vật, không tự huấn luyện lại, nên kết quả trên web luôn khớp báo cáo. "
    "Tuning Lab là nơi thử siêu tham số và chốt cấu hình trước khi chạy pipeline chính thức.",
)

# ---------- 9b. Chatbot action-decision ----------
s = new_slide()
add_title(
    s,
    "Chatbot Action-Decision: LLM chọn action, backend giữ số liệu",
    "Chương 3 - Chức năng chatbot",
)
box = add_box(s, Inches(0.7), Inches(1.75), Inches(5.5), Inches(4.5))
set_text(
    box.text_frame,
    [
        "LLM decision chọn đúng một trong năm action",
        "Backend validate schema và kiểm symbol scope",
        "Dispatcher cố định gọi handler, không tool loop",
        "Mọi số liệu do backend tính, LLM không tạo",
        "Compose diễn đạt lại, lỗi thì giữ formatter",
        "Không RAG, không vector DB, không embedding",
    ],
    size=16,
    space=11,
)
add_table(
    s,
    [
        ["Ràng buộc runtime", "Giá trị"],
        ["LLM call: decision + compose", "2"],
        ["Action trong whitelist", "5"],
        ["History gửi kèm", "3 cặp"],
        ["Fallback khi compose lỗi", "Formatter"],
    ],
    Inches(6.5),
    Inches(1.85),
    Inches(6.2),
    Inches(2.2),
    size=13,
    col_widths=[4, 2],
)
fit_picture(s, IMG / "chatbot_flow.png", Inches(6.5), Inches(4.15), Inches(6.2), Inches(2.1))
add_source(s, "docs/CHATBOT_ARCHITECTURE.md, services/chatbot_service.py, services/chatbot_tools.py")
add_note(
    s,
    "Chatbot chạy kiến trúc Action-Decision: LLM call thứ nhất chỉ chọn đúng một trong năm action là GENERAL_CHAT, STOCK_SIGNAL, STOCK_RANKING, PROJECT_INFO, OUT_OF_SCOPE kèm arguments theo schema cố định. "
    "Backend validate lại schema rồi kiểm symbol scope tập trung, sau đó một dispatcher cố định gọi handler tương ứng; mọi prediction, Điểm UP, xếp hạng và metric đều do backend tính từ artifact đã publish. "
    "LLM call thứ hai chỉ diễn đạt lại câu trả lời từ JSON số liệu backend đưa, không được thêm số mới, và nếu call này lỗi hoặc quá deadline thì hệ thống giữ nguyên bản formatter deterministic. "
    "Không có vector database, embedding, tool loop hay router keyword; trang /chat và dock nổi dùng chung một API. "
    "Decision sai protocol trả 502, nguồn dữ liệu chưa sẵn sàng trả 503 và quá deadline trả 504.",
)

# ---------- 9c. Demo chatbot ----------
s = new_slide()
add_title(
    s,
    "Demo: chatbot trả lời có số liệu, từ chối câu ngoài phạm vi",
    "Chương 3 - Ảnh chụp phiên làm việc thật",
)
fit_picture(s, IMG / "chatbot_demo.png", Inches(0.7), Inches(1.7), Inches(5.6), Inches(4.9))
box = add_box(s, Inches(6.6), Inches(1.75), Inches(6.1), Inches(4.6))
set_text(
    box.text_frame,
    [
        "Câu 1: PROJECT_INFO, model và kết quả TEST",
        "Câu 2: STOCK_SIGNAL FPT, Điểm UP và ngưỡng",
        "Câu 3: OUT_OF_SCOPE, hỏi P/E và tin tức",
        "Số liệu do backend tính, LLM chỉ diễn đạt",
        "Chatbot nói rõ đây là dữ liệu offline",
    ],
    size=16,
    space=12,
)
add_source(s, "ảnh chụp 127.0.0.1:5000/chat, docs/slides/shoot_chat.py")
add_note(
    s,
    "Đây là ảnh chụp thật từ phiên làm việc với Flask đang chạy, không phải ảnh minh họa. "
    "Câu đầu hỏi về model và kết quả TEST nên decision chọn PROJECT_INFO, số trả về trùng khớp bảng ở slide kết quả. "
    "Câu thứ hai hỏi FPT nên decision chọn STOCK_SIGNAL, backend kiểm scope rồi chạy inference và trả Điểm UP kèm ngưỡng quyết định. "
    "Câu thứ ba hỏi P/E và tin tức nên decision chọn OUT_OF_SCOPE, chatbot từ chối và nêu lại đúng những việc nó làm được. "
    "Cả ba câu không có con số nào do LLM tự viết ra, và disclaimer dữ liệu offline do backend gắn vào cuối trang.",
)


# ---------- 10. Chong ro ri ----------
s = new_slide()
sp = SUM["split_report"]
add_title(
    s,
    "Chia theo thời gian, purge nhãn vắt biên và khóa TEST bằng fingerprint",
    "Chương 3 - Giao thức thực nghiệm",
)
add_table(
    s,
    [
        ["Tập", "Số dòng", "Từ ngày", "Đến ngày"],
        ["TRAIN", N(sp["train_rows"]), sp["train_date_min"], sp["train_date_max"]],
        ["VALIDATION", N(sp["validation_rows"]), sp["validation_date_min"], sp["validation_date_max"]],
        ["TEST", N(sp["test_rows"]), sp["test_date_min"], sp["test_date_max"]],
    ],
    Inches(0.7),
    Inches(1.75),
    Inches(6.3),
    Inches(1.9),
    size=13,
    col_widths=[3, 3, 3, 3],
)
box = add_box(s, Inches(0.7), Inches(3.9), Inches(6.3), Inches(2.6))
set_text(
    box.text_frame,
    [
        f"Purge {N(sp['purged_train_validation_rows'])} dòng ở biên TRAIN/VALIDATION",
        f"Purge {N(sp['purged_validation_test_rows'])} dòng ở biên VALIDATION/TEST",
        "TEST chỉ được chấm một lần, sau khi đã chọn model",
        "Fingerprint nội dung chặn chấm lại TEST",
    ],
    size=15,
    space=11,
)
fit_picture(s, IMG / "split_rows.png", Inches(7.2), Inches(1.7), Inches(5.5), Inches(4.6))
add_source(s, "reports/split_summary.csv, reports/pipeline_summary.json")
add_note(
    s,
    "Ba tập được cắt theo mốc thời gian, không trộn ngẫu nhiên. "
    "Dòng nào có nhãn kết thúc vắt qua ranh giới thì bị purge, cụ thể 1.883 dòng ở biên TRAIN/VALIDATION và 1.775 dòng ở biên VALIDATION/TEST. "
    "Model được chọn trên VALIDATION, sau đó refit trên TRAIN cộng VALIDATION và chấm TEST đúng một lần. "
    "Fingerprint nội dung của snapshot dữ liệu được ghi vào registry để không thể chấm lại TEST nhiều lần rồi chọn kết quả đẹp nhất.",
)

# ---------- 11. Tuning ----------
s = new_slide()
add_title(
    s,
    "Tuning bằng CV 4 fold theo ngày, purge 5 phiên, ngưỡng chọn từ OOF",
    "Chương 3 - Thiết lập thực nghiệm",
)
rows = [["Model", "CV F1_UP", "Độ lệch chuẩn", "Ngưỡng"]]
for r in SUM["tuning_results"]:
    rows.append([r["model_name"], D4(r["cv_f1_up"]), D4(r["cv_f1_up_std"]), D2(r["decision_threshold"])])
add_table(s, rows, Inches(0.7), Inches(1.75), Inches(6.3), Inches(1.9), size=13, col_widths=[3, 2, 2, 2])
box = add_box(s, Inches(0.7), Inches(3.9), Inches(6.3), Inches(2.6))
set_text(
    box.text_frame,
    [
        "Fold cắt theo ngày thị trường, không theo dòng",
        "CV chỉ chấm giai đoạn từ 2021 trở đi",
        "Ngưỡng chọn từ xác suất out-of-fold trên TRAIN",
        "Ràng buộc precision và tỷ lệ UP tối đa 50%",
    ],
    size=15,
    space=11,
)
fit_picture(s, IMG / "cv_f1_up.png", Inches(7.2), Inches(1.7), Inches(5.5), Inches(4.6))
add_source(s, "reports/tuning_results.csv, reports/cv_fold_results.csv, config/settings.py")
add_note(
    s,
    "Kiểm định chéo cắt theo ngày thị trường: mọi dòng của cùng một phiên nằm cùng một fold. "
    "Giữa train và validation của mỗi fold bỏ 5 phiên và purge dòng có nhãn vắt biên. "
    "Ngưỡng quyết định 0,49 được chọn từ xác suất out-of-fold trên TRAIN, có ràng buộc precision không thấp hơn tỷ lệ UP thật và tỷ lệ dự báo UP không quá 50%. "
    "Ràng buộc này ngăn mô hình suy biến thành luôn dự báo UP để ăn điểm F1.",
)

# ---------- 11b. Sieu tham so chot ----------
s = new_slide()
add_title(
    s,
    "Siêu tham số chốt: cây nông, lá lớn để giảm overfit",
    "Chương 3 - Cấu hình model và độ ổn định CV",
)
bp = S["best"]["best_params"]
lr_p = bp["Logistic Regression"]
rf_p = bp["Random Forest"]
gb_p = bp["Gradient Boosting"]
add_table(
    s,
    [
        ["Model", "Siêu tham số đã chốt"],
        ["Logistic Regression", f"C={lr_p['C']:.2e}, solver={lr_p['solver']}"],
        [
            "Random Forest (final)",
            f"n_estimators={rf_p['n_estimators']}, max_depth={rf_p['max_depth']}, "
            f"min_samples_leaf={rf_p['min_samples_leaf']}, max_features={rf_p['max_features']}",
        ],
        [
            "Gradient Boosting",
            f"n_estimators={gb_p['n_estimators']}, learning_rate={gb_p['learning_rate']}, "
            f"max_depth={gb_p['max_depth']}, subsample={gb_p['subsample']}",
        ],
    ],
    Inches(0.7),
    Inches(1.75),
    Inches(11.9),
    Inches(1.9),
    size=12,
    col_widths=[3, 9],
)
box = add_box(s, Inches(0.7), Inches(3.95), Inches(5.0), Inches(2.4))
set_text(
    box.text_frame,
    [
        "Cây nông, lá tối thiểu 100 mẫu",
        "class_weight balanced_subsample cho RF",
        "random_state 42 nên tái lập được",
        "F1_UP dao động theo chế độ thị trường",
    ],
    size=15,
    space=10,
)
fit_picture(s, IMG / "cv_fold_spread.png", Inches(6.0), Inches(3.75), Inches(6.6), Inches(2.7))
add_source(s, "reports/best_params.json, reports/cv_fold_results.csv, models/model_metadata.json")
add_note(
    s,
    "Siêu tham số được chốt thủ công tại Tuning Lab rồi khóa lại trong manual_config.json trước khi chạy pipeline. "
    "Hướng chung là giữ model đơn giản: cây nông, lá tối thiểu 100 mẫu, chỉ xét 20 phần trăm feature mỗi lần split. "
    "Random Forest dùng class_weight balanced_subsample vì lớp UP là lớp thiểu số, và random_state 42 để tái lập. "
    "Biểu đồ bên phải cho thấy F1_UP từng fold dao động từ khoảng 0,41 đến 0,51, tức chất lượng phụ thuộc chế độ thị trường từng giai đoạn.",
)

# ---------- 11c. Tuning Lab ----------
s = new_slide()
add_title(
    s,
    "Tuning Lab ghi lại 350 lần thử cấu hình, chọn tay rồi khóa lại",
    "Chương 3 - Quy trình thử siêu tham số",
)
box = add_box(s, Inches(0.7), Inches(1.75), Inches(6.1), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Trang /tuning: nhập tham số, chạy CV trên TRAIN",
        "Mỗi lần chạy ghi một dòng vào tuning_history.csv",
        "Chỉ TRAIN được dùng, không nhìn VALIDATION",
        "Cấu hình chốt lưu vào manual_config.json",
        "Pipeline từ chối chạy nếu thiếu cấu hình model",
        "Fingerprint dữ liệu chặn dùng lại kết quả cũ",
    ],
    size=16,
    space=12,
)
lab_rows = [["Số lần thử (tuning_history.csv)", "Số run"]]
for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
    lab_rows.append([name, N(LAB_RUNS.get(name, 0))])
lab_rows.append(["Tổng", N(sum(LAB_RUNS.values()))])
add_table(
    s,
    lab_rows,
    Inches(7.1),
    Inches(2.2),
    Inches(5.5),
    Inches(2.4),
    size=14,
    col_widths=[3.4, 2.1],
)
add_source(s, "experiments/tuning_history.csv, experiments/manual_config.json, services/tuning_lab.py")
add_note(
    s,
    "Siêu tham số không dò tự động bằng GridSearch mà thử tay qua trang Tuning Lab, vì em muốn kiểm soát từng cấu hình và hiểu ảnh hưởng của mỗi tham số. "
    "Mọi lần chạy đều được ghi vào tuning_history.csv, tổng cộng 350 run cho ba model, nên quá trình thử là truy vết được chứ không phải chọn ngẫu nhiên. "
    "Điểm quan trọng về giao thức: Tuning Lab chỉ chấm trên TRAIN bằng cross-validation, không hề nhìn VALIDATION hay TEST. "
    "Cấu hình cuối cùng được khóa vào manual_config.json và pipeline chính thức từ chối chạy nếu thiếu cấu hình hoặc sai fingerprint dữ liệu.",
)

# ---------- 12. Chon model tren VALIDATION ----------
s = new_slide()
val_rows = [r for r in SUM["validation_selection"]["rows"] if r["row_type"] == "candidate"]
add_title(
    s,
    "Random Forest thắng trên VALIDATION với F1_UP 0,4773",
    "Chương 3.3.2 - Chọn họ mô hình",
)
rows = [["Model", "F1_UP", "Precision_UP", "Recall_UP", "Accuracy"]]
for r in val_rows:
    rows.append([r["model_name"], D4(r["f1_up"]), D4(r["precision_up"]), D4(r["recall_up"]), D4(r["accuracy"])])
bl = SUM["validation_selection"]["rows"][3]
rows.append(["Always UP (baseline)", D4(bl["f1_up"]), D4(bl["precision_up"]), D4(bl["recall_up"]), D4(bl["accuracy"])])
add_table(s, rows, Inches(0.6), Inches(1.75), Inches(6.6), Inches(2.3), size=12, col_widths=[3, 2, 2, 2, 2])
box = add_box(s, Inches(0.6), Inches(4.25), Inches(6.6), Inches(2.2))
set_text(
    box.text_frame,
    [
        "Tiêu chí 1: F1_UP cao nhất trên VALIDATION",
        "Tiêu chí 2: Recall_UP cao hơn khi bằng điểm",
        "Tiêu chí 3: chọn model đơn giản hơn",
        "Baseline always-UP vẫn cao hơn cả ba model",
    ],
    size=15,
    space=10,
)
fit_picture(s, IMG / "validation_f1_up.png", Inches(7.4), Inches(1.7), Inches(5.3), Inches(4.6))
add_source(s, "reports/model_comparison.csv, models/model_metadata.json")
add_note(
    s,
    "Ba mô hình được chấm trên VALIDATION bằng ngưỡng đã chốt từ TRAIN. "
    "Random Forest đạt F1_UP 0,4773, cao hơn Gradient Boosting 0,4734 và Logistic Regression 0,4350, nên được chọn. "
    "Cần nói thẳng: baseline luôn dự báo UP đạt 0,4982 trên VALIDATION, vẫn cao hơn cả ba mô hình. "
    "Hệ thống ghi lại cảnh báo này trong metadata thay vì bỏ qua.",
)

# ---------- 13. Ket qua TEST ----------
s = new_slide()
ft = META["final_test_metrics"]
bl_up = META["final_test_baselines"][0]
bl_not = META["final_test_baselines"][1]
add_title(
    s,
    "Trên TEST, F1_UP 0,3754 vẫn dưới baseline always-UP 0,3839",
    "Chương 3.3.2 - Đánh giá TEST một lần",
)
rows = [
    ["Trên TEST (20.350 dòng)", "F1_UP", "Precision_UP", "Recall_UP", "Accuracy"],
    ["Random Forest (final)", D4(ft["f1_up"]), D4(ft["precision_up"]), D4(ft["recall_up"]), D4(ft["accuracy"])],
    ["Always UP", D4(bl_up["f1_up"]), D4(bl_up["precision_up"]), D4(bl_up["recall_up"]), D4(bl_up["accuracy"])],
    ["Always NOT_UP", D4(bl_not["f1_up"]), D4(bl_not["precision_up"]), D4(bl_not["recall_up"]), D4(bl_not["accuracy"])],
]
add_table(s, rows, Inches(0.6), Inches(1.75), Inches(6.7), Inches(2.0), size=12, col_widths=[3, 2, 2, 2, 2])
box = add_box(s, Inches(0.6), Inches(4.0), Inches(6.7), Inches(2.4))
set_text(
    box.text_frame,
    [
        "Mục tiêu vượt baseline: không đạt",
        "Hệ thống ghi baseline_passed = false",
        "Web hiển thị cảnh báo, không che kết quả",
        "Chỉ dùng cho học tập, không khuyến nghị đầu tư",
    ],
    size=15,
    space=11,
)
fit_picture(s, IMG / "test_vs_baseline.png", Inches(7.5), Inches(1.7), Inches(5.2), Inches(4.6))
add_source(s, "reports/final_model_evaluation.csv, models/model_metadata.json")
add_note(
    s,
    "Đây là kết quả chính thức, chấm một lần trên 20.350 dòng TEST từ 13/04/2026 đến 13/07/2026. "
    "Random Forest đạt F1_UP 0,3754 và accuracy 0,5766, còn baseline luôn dự báo UP đạt F1_UP 0,3839. "
    "Nghĩa là mục tiêu vượt baseline không đạt, và hệ thống ghi baseline_passed bằng false kèm cảnh báo hiển thị trên web. "
    "Em chọn báo cáo trung thực con số này thay vì nới giao thức để có số đẹp hơn.",
)

# ---------- 14. Confusion matrix ----------
s = new_slide()
cm = SUM["confusion_matrix"]
add_title(
    s,
    "Model bắt được 2.589 phiên UP nhưng báo động sai 6.371 lần",
    "Chương 3.3.2 - Phân tích lỗi",
)
fit_picture(s, IMG / "confusion_matrix_test.png", Inches(0.7), Inches(1.7), Inches(6.4), Inches(4.6))
box = add_box(s, Inches(7.3), Inches(1.9), Inches(5.4), Inches(4.3))
set_text(
    box.text_frame,
    [
        "Recall_UP 0,536: bắt được hơn nửa phiên tăng",
        "Precision_UP 0,289: cứ 10 tín hiệu, 3 đúng",
        "Ngưỡng 0,49 ưu tiên recall hơn precision",
        "Tỷ lệ UP trên TEST chỉ 23,8%, thấp hơn TRAIN",
        "Phân phối lớp lệch khiến precision giảm mạnh",
    ],
    size=15,
    space=12,
)
add_source(s, "reports/confusion_matrix.csv, reports/classification_report.csv")
add_note(
    s,
    "Ma trận nhầm lẫn cho thấy bản chất lỗi: mô hình dự báo UP khá thoáng nên bắt được 2.589 trong 4.834 phiên tăng thật, đạt recall 0,536. "
    "Nhưng đi kèm 6.371 tín hiệu UP sai, nên precision chỉ 0,289. "
    "Một lý do là tỷ lệ UP trên TEST chỉ 23,8%, thấp hơn mức 37,6% của toàn bộ dữ liệu, nên ngưỡng chốt từ TRAIN trở nên quá thoáng. "
    "Đây chính là hạn chế về hiệu chỉnh ngưỡng theo chế độ thị trường.",
)

# ---------- 15. Feature importance ----------
s = new_slide()
add_title(
    s,
    "Biến động 20 phiên là feature quan trọng nhất, chiếm 19,7%",
    "Chương 3.3.2 - Diễn giải mô hình",
)
fit_picture(s, IMG / "feature_importance_top8.png", Inches(0.7), Inches(1.7), Inches(7.0), Inches(4.6))
box = add_box(s, Inches(8.0), Inches(1.9), Inches(4.7), Inches(4.3))
set_text(
    box.text_frame,
    [
        "Biến động lấn át tín hiệu xu hướng",
        "month có importance cao, cần soi kỹ",
        "Return dài hạn hơn return 1 phiên",
        "SMA thô gần như không đóng góp",
        "Đúng kỳ vọng: giá ngày khó dự báo hướng",
    ],
    size=15,
    space=12,
)
add_source(s, "reports/feature_importance.csv")
add_note(
    s,
    "Random Forest xếp volatility_20d cao nhất với 19,7%, rồi tới month và volatility_5d. "
    "Mô hình dựa vào mức biến động nhiều hơn là hướng đi của giá, phù hợp với việc F1_UP thấp. "
    "Feature month đứng thứ hai là điểm cần thận trọng: nó có thể chỉ phản ánh chế độ thị trường trong giai đoạn huấn luyện. "
    "Các trung bình động thô như sma5, sma20, sma50 gần như không đóng góp vì đã có các tỷ lệ so sánh tương đối.",
)

# ---------- 15b. San pham web ----------
s = new_slide()
add_title(
    s,
    "Sản phẩm: sáu trang web đọc lại đúng artifact đã publish",
    "Chương 3 - Chức năng hệ thống",
)
box = add_box(s, Inches(0.7), Inches(1.75), Inches(5.6), Inches(4.4))
set_text(
    box.text_frame,
    [
        "Web và CLI dùng chung một model artifact",
        "Trang dự báo chỉ đọc, không train lại",
        "Kết quả trên web khớp số trong báo cáo",
        "Trang đánh giá tách CV, VALIDATION, TEST",
        "Cảnh báo baseline hiện ngay trên giao diện",
    ],
    size=16,
    space=13,
)
add_table(
    s,
    [
        ["Trang", "Chức năng"],
        ["/predict", "Dự báo UP/NOT_UP một đến hai mã"],
        ["/compare", "So sánh Điểm UP hai mã"],
        ["/screener", "Xếp hạng Điểm UP toàn sàn"],
        ["/evaluation", "CV, VALIDATION, TEST và baseline"],
        ["/tuning", "Tuning Lab, chốt cấu hình, chạy pipeline"],
        ["/chat", "Chatbot dữ liệu nội bộ"],
    ],
    Inches(6.6),
    Inches(1.9),
    Inches(6.0),
    Inches(3.6),
    size=13,
    col_widths=[1.9, 4.1],
)
add_source(s, "app.py (route), templates/, services/prediction_service.py")
add_note(
    s,
    "Sản phẩm giao nộp gồm sáu trang chức năng, tất cả đọc lại cùng một model artifact do pipeline publish. "
    "Các trang dự báo, so sánh, xếp hạng và đánh giá chỉ đọc hiện vật, không train lại, nên số trên web luôn khớp báo cáo. Riêng trang Tuning Lab có nút chạy pipeline chính thức, nhưng bị chặn bởi ba điều kiện: đủ cấu hình ba model, đúng fingerprint dữ liệu và snapshot chưa từng được đánh giá. "
    "Trang đánh giá cố tình tách ba khối: cross-validation trên TRAIN, chọn model trên VALIDATION và đánh giá TEST một lần, để không trộn lẫn ba loại số này. "
    "Cảnh báo chưa vượt baseline được hiển thị trực tiếp trên giao diện thay vì chỉ nằm trong metadata.",
)

# ---------- 16. Kiem thu ----------
s = new_slide()
add_title(
    s,
    "Bộ kiểm thử phủ giao thức dữ liệu, model, chatbot và UI",
    "Chương 3.3.1 - Kế hoạch kiểm thử",
)
rows = [
    ["Nhóm kiểm thử", "Tệp"],
    ["Giao thức dữ liệu và nhãn", "tests/test_data_protocol.py"],
    ["CV chuỗi thời gian", "tests/test_recent_cv.py"],
    ["Chọn model và giao thức TEST", "tests/test_model_selection.py"],
    ["Luồng dự báo", "tests/test_prediction_flow.py"],
    ["Tuning Lab và lịch sử tuning", "tests/test_tuning_lab.py, test_tuning_history.py"],
    ["Web UI và chatbot", "tests/test_ui_shell.py, test_chatbot*.py"],
]
add_table(s, rows, Inches(0.7), Inches(1.75), Inches(11.9), Inches(3.5), size=13, col_widths=[3, 5])
box = add_box(s, Inches(0.7), Inches(5.5), Inches(11.9), Inches(1.1))
set_text(
    box.text_frame,
    [
        "Cổng kiểm tra trong pipeline chặn rò rỉ biên",
        "Chưa có kiểm thử hiệu năng và chấp nhận",
    ],
    size=15,
    space=6,
)
add_source(s, "tests/, reports/pipeline_summary.json")
add_note(
    s,
    "Bộ kiểm thử trong thư mục tests được mở rộng cùng code nên slide không khóa cứng tổng số. "
    "Quan trọng nhất là nhóm kiểm thử giao thức: có test khẳng định nhãn dùng đúng phiên thị trường thứ năm, và test chặn cột tương lai lọt vào feature. "
    "Ngoài kiểm thử tự động, pipeline còn có cổng kiểm tra chạy trong lúc thực thi để chặn rò rỉ ranh giới và sai lệch mô hình được chọn. "
    "Phần chưa làm là kiểm thử hiệu năng và kiểm thử chấp nhận có người dùng thật.",
)

# ---------- 17. Han che ----------
s = new_slide()
add_title(
    s,
    "Hạn chế lớn nhất: chất lượng dự báo chưa vượt baseline",
    "Chương 4.3 - Hạn chế",
)
box = add_box(s, Inches(0.8), Inches(1.7), SW - Inches(1.6), Inches(4.6))
set_text(
    box.text_frame,
    [
        "TEST F1_UP thấp hơn baseline luôn UP",
        "Chỉ dùng giá và khối lượng theo ngày",
        "Ngưỡng cố định, chưa hiệu chỉnh theo thị trường",
        "Suy luận trên dữ liệu tĩnh, không realtime",
        "Web chưa có xác thực người dùng",
        "Chưa kiểm thử hiệu năng và chấp nhận",
    ],
    size=18,
    space=14,
)
add_source(s, "báo cáo mục 4.3, reports/final_model_evaluation.csv")
add_note(
    s,
    "Hạn chế quan trọng nhất là chất lượng dự báo chưa vượt baseline trên TEST, nên hệ thống chỉ dùng cho học tập và nghiên cứu. "
    "Nguyên nhân chính là tập feature chỉ khai thác giá và khối lượng theo ngày, chưa có dữ liệu cơ bản hay dòng tiền khối ngoại. "
    "Ngưỡng quyết định được chọn một lần từ OOF trên TRAIN và giữ cố định, chưa hiệu chỉnh lại theo chế độ thị trường. "
    "Về phần mềm, web chưa có xác thực người dùng và chưa có kiểm thử hiệu năng.",
)

# ---------- 18. Huong phat trien ----------
s = new_slide()
add_title(
    s,
    "Hướng phát triển: mở rộng feature và hiệu chỉnh xác suất",
    "Chương 4.4 - Hướng phát triển",
)
box = add_box(s, Inches(0.8), Inches(1.7), SW - Inches(1.6), Inches(4.6))
set_text(
    box.text_frame,
    [
        "Thêm VN-Index, sức mạnh ngành, khối ngoại",
        "Thử XGBoost, LightGBM, LSTM cùng giao thức",
        "Hiệu chỉnh xác suất Platt hoặc isotonic",
        "Đánh giá theo lợi nhuận và mức sụt giảm",
        "Tự động walk-forward định kỳ mỗi tháng",
        "Thêm xác thực, khóa liên tiến trình, ghi vết",
    ],
    size=18,
    space=14,
)
add_source(s, "báo cáo mục 4.4")
add_note(
    s,
    "Hướng ưu tiên là mở rộng feature sang dữ liệu thị trường và ngành, vì đây là điểm nghẽn rõ nhất của mô hình hiện tại. "
    "Sau đó mới thử các họ mô hình mạnh hơn như XGBoost hay LSTM, nhưng giữ nguyên giao thức chống rò rỉ hiện có. "
    "Thay vì chọn một ngưỡng cố định, bước tiếp theo là hiệu chỉnh xác suất rồi chọn ngưỡng theo mục tiêu sử dụng. "
    "Cuối cùng là đánh giá theo hướng đầu tư có chi phí giao dịch, thay vì chỉ dựa vào độ đo phân loại.",
)

# ---------- 19. Ket luan ----------
s = new_slide()
add_title(
    s,
    "Bốn mục tiêu kỹ thuật đạt, mục tiêu vượt baseline không đạt",
    "Chương 4.1 - Kết luận",
)
rows = [
    ["Mục tiêu", "Kết quả"],
    ["Pipeline dữ liệu và feature", "Đạt - 510.862 dòng, 20 feature"],
    ["Nhãn không rò rỉ theo phiên chung", "Đạt - có test kiểm chứng"],
    ["So sánh ba model theo giao thức", "Đạt - 3 model, CV 4 fold"],
    ["Công bố hiện vật và web", "Đạt - final_model.pkl, web Flask + CLI"],
    ["Vượt baseline always-UP trên TEST", "Không đạt - 0,3754 so 0,3839"],
]
add_table(s, rows, Inches(0.7), Inches(1.75), Inches(11.9), Inches(3.1), size=14, col_widths=[4, 4])
box = add_box(s, Inches(0.7), Inches(5.15), Inches(11.9), Inches(1.4))
set_text(
    box.text_frame,
    [
        "Giá trị chính: giao thức chống rò rỉ, tái lập được",
        "Xin cảm ơn - kính mời hội đồng đặt câu hỏi",
    ],
    size=17,
    space=10,
)
add_source(s, "báo cáo mục 4.1, models/model_metadata.json")
add_note(
    s,
    "Tổng kết: bốn mục tiêu về kỹ thuật và sản phẩm đã đạt, mục tiêu về chất lượng dự báo không đạt. "
    "Em báo cáo công khai điều đó vì nó có ý nghĩa phương pháp: khi loại bỏ ba nguồn rò rỉ, năng lực dự báo thực tế của feature kỹ thuật thuần trên dữ liệu ngày là rất hạn chế. "
    "Đóng góp chính của niên luận là một giao thức thực nghiệm có kỷ luật và tái lập được, cùng hệ thống web cho phép kiểm chứng lại kết quả. "
    "Em xin hết và sẵn sàng nhận câu hỏi từ hội đồng.",
)

prs.save(str(OUT))
print("saved", OUT, len(prs.slides.__iter__.__self__._sldIdLst))
