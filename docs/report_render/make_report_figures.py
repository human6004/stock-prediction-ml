"""Sinh cac hinh bo sung cho bao cao: so do Gantt ke hoach va so do use case.

Chay: python docs/report_render/make_report_figures.py
Ket qua ghi vao docs/report_assets/ duoi dang PNG 200 dpi, font Arial de khop
voi quy dinh trinh bay CT239H.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ASSETS = Path(__file__).resolve().parent.parent / "report_assets"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1f2933"
GRID = "#d7dce2"

# ----------------------------------------------------------------------------
# 1. So do Gantt
# ----------------------------------------------------------------------------
# (ten cong viec, tuan bat dau (1-based), so tuan, nhom)
GANTT_TASKS = [
    ("Khảo sát đề tài, xác định bài toán", 1, 2, "plan"),
    ("Thu thập và làm sạch dữ liệu OHLCV", 2, 2, "data"),
    ("Sinh 20 đặc trưng kỹ thuật, gán nhãn t+5", 3, 2, "data"),
    ("Thiết kế giao thức chống rò rỉ, chia tập", 4, 2, "data"),
    ("Tinh chỉnh siêu tham số ba mô hình", 5, 3, "model"),
    ("Chọn ngưỡng OOF, chọn họ mô hình", 7, 2, "model"),
    ("Đánh giá TEST một lần, phát hành hiện vật", 8, 2, "model"),
    ("Xây dựng ứng dụng web Flask", 6, 4, "app"),
    ("Xây dựng chatbot dữ liệu nội bộ", 9, 2, "app"),
    ("Viết kiểm thử tự động", 10, 2, "app"),
    ("Viết báo cáo và hoàn thiện tài liệu", 10, 3, "plan"),
]

GROUP_COLORS = {
    "plan": "#7f8c9b",
    "data": "#2f7fb8",
    "model": "#2e8b6f",
    "app": "#c2793a",
}
GROUP_LABELS = {
    "plan": "Chuẩn bị và báo cáo",
    "data": "Dữ liệu và đặc trưng",
    "model": "Mô hình và đánh giá",
    "app": "Ứng dụng và kiểm thử",
}
TOTAL_WEEKS = 12


def make_gantt(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    rows = len(GANTT_TASKS)

    for index, (name, start, span, group) in enumerate(GANTT_TASKS):
        y = rows - index - 1
        ax.add_patch(
            patches.FancyBboxPatch(
                (start - 1 + 0.06, y + 0.18),
                span - 0.12,
                0.64,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=GROUP_COLORS[group],
                edgecolor="none",
            )
        )
        ax.text(
            start - 1 + span / 2,
            y + 0.5,
            f"T{start}-T{start + span - 1}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )

    ax.set_xlim(0, TOTAL_WEEKS)
    ax.set_ylim(0, rows)
    ax.set_yticks([rows - i - 0.5 for i in range(rows)])
    ax.set_yticklabels([task[0] for task in GANTT_TASKS], fontsize=11, color=INK)
    ax.set_xticks(range(TOTAL_WEEKS + 1))
    ax.set_xticklabels([f"{w}" for w in range(TOTAL_WEEKS + 1)], fontsize=10, color=INK)
    ax.set_xlabel("Tuần thực hiện (học kỳ III, năm học 2025 - 2026)", fontsize=12, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=11,
               markerfacecolor=color, markeredgecolor="none", label=GROUP_LABELS[key])
        for key, color in GROUP_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. So do use case
# ----------------------------------------------------------------------------
USE_CASES = [
    ("UC-01", "Dự báo xu hướng một mã"),
    ("UC-02", "So sánh hai mã"),
    ("UC-03", "Xếp hạng toàn sàn"),
    ("UC-04", "Xem kết quả đánh giá mô hình"),
    ("UC-05", "Tra cứu qua chatbot"),
    ("UC-06", "Tinh chỉnh siêu tham số (Tuning Lab)"),
    ("UC-07", "Chạy pipeline và phát hành mô hình"),
    ("UC-08", "Cập nhật dữ liệu OHLCV"),
]
# use case nao thuoc ve actor nao
INVESTOR_UC = {"UC-01", "UC-02", "UC-03", "UC-04", "UC-05"}
ANALYST_UC = {"UC-04", "UC-06", "UC-07", "UC-08"}


def _actor(ax, x: float, y: float, label: str) -> None:
    head_r = 0.16
    ax.add_patch(patches.Circle((x, y + 0.82), head_r, fill=False, lw=1.8, ec=INK))
    ax.plot([x, x], [y + 0.66, y + 0.16], color=INK, lw=1.8)
    ax.plot([x - 0.3, x + 0.3], [y + 0.52, y + 0.52], color=INK, lw=1.8)
    ax.plot([x, x - 0.26], [y + 0.16, y - 0.3], color=INK, lw=1.8)
    ax.plot([x, x + 0.26], [y + 0.16, y - 0.3], color=INK, lw=1.8)
    ax.text(x, y - 0.55, label, ha="center", va="top", fontsize=12,
            color=INK, fontweight="bold")


def make_use_case(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.6))

    boundary_x0, boundary_x1 = 3.1, 8.6
    ax.add_patch(
        patches.Rectangle(
            (boundary_x0, 0.35), boundary_x1 - boundary_x0, 7.1,
            fill=False, ec=INK, lw=1.6,
        )
    )
    ax.text(
        (boundary_x0 + boundary_x1) / 2, 7.22,
        "Hệ thống dự báo xu hướng giá cổ phiếu HOSE",
        ha="center", va="center", fontsize=12.5, color=INK, fontweight="bold",
    )

    positions: dict[str, tuple[float, float]] = {}
    top, step = 6.55, 0.79
    for index, (code, name) in enumerate(USE_CASES):
        cx = (boundary_x0 + boundary_x1) / 2
        cy = top - index * step
        positions[code] = (cx, cy)
        ax.add_patch(
            patches.Ellipse((cx, cy), 4.9, 0.62, facecolor="#eef3f8",
                            edgecolor=INK, lw=1.3)
        )
        ax.text(cx, cy, f"{code}. {name}", ha="center", va="center",
                fontsize=11, color=INK)

    investor = (1.35, 4.6)
    analyst = (10.15, 2.7)
    _actor(ax, investor[0], investor[1], "Người dùng\n(nhà đầu tư)")
    _actor(ax, analyst[0], analyst[1], "Người quản trị\nmô hình")

    for code in USE_CASES:
        pass
    for code, _ in USE_CASES:
        cx, cy = positions[code]
        if code in INVESTOR_UC:
            ax.annotate(
                "", xy=(cx - 2.5, cy), xytext=(investor[0] + 0.35, investor[1] + 0.4),
                arrowprops=dict(arrowstyle="-", color="#5a6672", lw=1.1),
            )
        if code in ANALYST_UC:
            ax.annotate(
                "", xy=(cx + 2.5, cy), xytext=(analyst[0] - 0.35, analyst[1] + 0.4),
                arrowprops=dict(arrowstyle="-", color="#5a6672", lw=1.1),
            )

    # quan he <<include>>: UC-07 include UC-06 (phai chot cau hinh truoc khi chay)
    x6, y6 = positions["UC-06"]
    x7, y7 = positions["UC-07"]
    ax.annotate(
        "", xy=(x6 + 1.9, y6 - 0.12), xytext=(x7 + 1.9, y7 + 0.12),
        arrowprops=dict(arrowstyle="->", color="#5a6672", lw=1.1, linestyle="dashed"),
    )
    ax.text(x7 + 2.15, (y6 + y7) / 2, "<<include>>", fontsize=9.5,
            color="#5a6672", ha="left", va="center")

    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. So do kien truc tong the (ve lai voi chu lon de doc duoc o khung 15,5 cm)
# ----------------------------------------------------------------------------
LAYER_FACE = "#f5f8fb"

# (nhan lop, y0, mau nen node, mau vien node, [(noi dung node, fontsize)])
ARCH_BANDS = [
    (
        "Lớp trình bày",
        7.75,
        "#dce9f5",
        "#2f7fb8",
        [
            ("Người dùng\n(trình duyệt, dòng lệnh)", 11.5),
            ("app.py\ncác route Flask", 12.0),
            ("templates/ + static/\nJinja2, Chart.js", 11.5),
        ],
    ),
    (
        "Lớp dịch vụ (services/)",
        5.5,
        "#dbeee5",
        "#2e8b6f",
        [
            ("model_tuning.py\ntuning_lab.py\nCV, chọn ngưỡng OOF", 11.0),
            ("model_evaluation.py\nVALIDATION, TEST một lần", 11.0),
            ("prediction_service.py\nchatbot_service.py\ndự báo, hỏi đáp", 11.0),
        ],
    ),
    (
        "Lớp dữ liệu",
        3.25,
        "#f6e6d6",
        "#c2793a",
        [
            ("preprocessing.py\nlàm sạch OHLCV", 11.5),
            ("feature_engineering.py\n20 đặc trưng, nhãn t+5", 11.0),
            ("time_splitting.py\npurged CV, gap 5 phiên", 11.0),
        ],
    ),
    (
        "Nguồn dữ liệu",
        1.0,
        "#e9ecef",
        "#7f8c9b",
        [
            ("hose_stock_raw.csv (ngoài repo)\n554.897 dòng, 400 mã", 11.5),
            ("data/processed/\nclean, features, ml_dataset", 11.5),
        ],
    ),
]

ARCH_ARTIFACTS = [
    ("models/\nfinal_model.pkl\n(Random Forest)\nmodel_metadata.json", 8.35),
    ("reports/\nCSV, JSON, PNG\nF1_UP CV 0,4703", 6.0),
    ("experiments/\nfingerprint f162b3e8\nlịch sử, khóa TEST", 3.75),
]

ARCH_CONFIG_LINES = [
    "• 20 đặc trưng",
    "• thứ tự cột cố định",
    "• ngưỡng UP 1%",
    "• tầm dự báo 5 phiên",
    "• 4 fold, purge 5 phiên",
    "• đường dẫn tệp",
    "• mã giao thức",
]


def _arch_box(ax, cx, cy, w, h, text, face, edge, fontsize, bold=False) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=face, edgecolor=edge, lw=1.5,
        )
    )
    ax.text(
        cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK,
        fontweight="bold" if bold else "normal", linespacing=1.35,
    )


def make_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.3))

    band_x0, band_x1 = 2.60, 10.40
    band_h, node_h = 1.70, 1.06

    for label, y0, face, edge, nodes in ARCH_BANDS:
        ax.add_patch(
            patches.Rectangle(
                (band_x0, y0), band_x1 - band_x0, band_h,
                facecolor=LAYER_FACE, edgecolor=GRID, lw=1.2,
            )
        )
        ax.text(
            band_x0 + 0.14, y0 + band_h - 0.22, label,
            ha="left", va="center", fontsize=13.0, color=INK, fontweight="bold",
        )
        count = len(nodes)
        span = (band_x1 - band_x0) - 0.28
        cell = span / count
        for index, (text, fontsize) in enumerate(nodes):
            cx = band_x0 + 0.14 + cell * (index + 0.5)
            _arch_box(ax, cx, y0 + 0.60, cell - 0.16, node_h, text, face, edge, fontsize)

    # cot cau hinh ben trai
    cfg_x0, cfg_x1 = 0.25, 2.35
    ax.add_patch(
        patches.FancyBboxPatch(
            (cfg_x0, 1.00), cfg_x1 - cfg_x0, 8.45,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor="#fdf3d8", edgecolor="#b8912f", lw=1.6,
        )
    )
    ax.text(
        (cfg_x0 + cfg_x1) / 2, 8.95, "config/settings.py",
        ha="center", va="center", fontsize=12.5, color=INK, fontweight="bold",
    )
    ax.text(
        (cfg_x0 + cfg_x1) / 2, 8.45, "Nguồn cấu hình\nduy nhất",
        ha="center", va="center", fontsize=11.0, color=INK, linespacing=1.35,
    )
    for index, line in enumerate(ARCH_CONFIG_LINES):
        ax.text(
            cfg_x0 + 0.14, 7.55 - index * 0.52, line,
            ha="left", va="center", fontsize=10.5, color=INK,
        )

    # cot hien vat ben phai
    art_x0, art_x1 = 10.75, 13.05
    ax.add_patch(
        patches.Rectangle(
            (art_x0, 1.00), art_x1 - art_x0, 8.45,
            facecolor=LAYER_FACE, edgecolor=GRID, lw=1.2,
        )
    )
    ax.text(
        (art_x0 + art_x1) / 2, 9.23, "Lớp lưu trữ hiện vật",
        ha="center", va="center", fontsize=13.0, color=INK, fontweight="bold",
    )
    for text, cy in ARCH_ARTIFACTS:
        _arch_box(
            ax, (art_x0 + art_x1) / 2, cy, art_x1 - art_x0 - 0.28, 1.45,
            text, "#eee3f2", "#7d5ba6", 10.5,
        )

    band_cx = (band_x0 + band_x1) / 2
    flow = dict(arrowstyle="-|>", color="#41505f", lw=2.0,
                mutation_scale=22, shrinkA=0, shrinkB=0)

    # luong du lieu di len giua cac lop
    for y_from, y_to, label in [
        (2.70, 3.25, "đọc CSV thô"),
        (4.95, 5.50, "ml_dataset.csv"),
        (7.20, 7.75, "kết quả dự báo"),
    ]:
        ax.annotate("", xy=(band_cx, y_to), xytext=(band_cx, y_from),
                    arrowprops=flow)
        ax.text(band_cx + 0.16, (y_from + y_to) / 2, label,
                ha="left", va="center", fontsize=10.5, color="#41505f")

    # dich vu ghi va doc hien vat
    ax.annotate("", xy=(art_x0, 6.60), xytext=(band_x1, 6.60), arrowprops=flow)
    ax.text((band_x1 + art_x0) / 2, 6.78, "ghi", ha="center", va="bottom",
            fontsize=10.5, color="#41505f")
    ax.annotate("", xy=(band_x1, 5.95), xytext=(art_x0, 5.95), arrowprops=flow)
    ax.text((band_x1 + art_x0) / 2, 5.72, "đọc", ha="center", va="top",
            fontsize=10.5, color="#41505f")

    # cau hinh cap cho moi lop
    for _, y0, _, _, _ in ARCH_BANDS:
        ax.annotate(
            "", xy=(band_x0, y0 + band_h / 2), xytext=(cfg_x1, y0 + band_h / 2),
            arrowprops=dict(arrowstyle="-|>", color="#b8912f", lw=1.5,
                            linestyle="dashed", mutation_scale=18,
                            shrinkA=0, shrinkB=0),
        )

    ax.set_xlim(0.05, 13.25)
    ax.set_ylim(0.55, 9.60)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_gantt(ASSETS / "gantt_plan.png")
    make_use_case(ASSETS / "use_case_diagram.png")
    make_architecture(ASSETS / "architecture_overview.png")
    print("wrote gantt_plan.png, use_case_diagram.png, architecture_overview.png")


if __name__ == "__main__":
    main()
