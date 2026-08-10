"""Sinh luu do va so do tuan tu cho muc 3.2.3 (thiet ke chi tiet).

Chay: python docs/report_render/make_flow_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "report_assets"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1f2933"
LINE = "#41505f"

KINDS = {
    "start": ("#dce9f5", "#2f7fb8"),
    "end": ("#dbeee5", "#2e8b6f"),
    "proc": ("#ffffff", "#41505f"),
    "dec": ("#fdf3dd", "#c2793a"),
    "drop": ("#fbe4e4", "#b4534b"),
    "note": ("#f5f8fb", "#7f8c9b"),
}


def _box(ax, cx, cy, w, h, text, kind="proc", fontsize=10.5) -> None:
    face, edge = KINDS[kind]
    if kind in {"start", "end"}:
        style = "round,pad=0.02,rounding_size=0.34"
    else:
        style = "round,pad=0.02,rounding_size=0.06"
    ax.add_patch(patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle=style,
        facecolor=face, edgecolor=edge, lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=INK, linespacing=1.3, zorder=4)


def _diamond(ax, cx, cy, w, h, text, fontsize=10.0) -> None:
    face, edge = KINDS["dec"]
    ax.add_patch(patches.Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
        closed=True, facecolor=face, edgecolor=edge, lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=INK, linespacing=1.3, zorder=4)


def _route(ax, points, label=None, label_offset=(0.12, 0.12), dashed=False,
           color=LINE, fontsize=9.5) -> None:
    """Ve duong gap khuc qua danh sach diem, dat mui nhui o doan cuoi."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.7, zorder=2,
            linestyle="--" if dashed else "-", solid_capstyle="round")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                mutation_scale=18,
                                linestyle="--" if dashed else "-"),
                zorder=2)
    if label:
        mx = (points[0][0] + points[1][0]) / 2 + label_offset[0]
        my = (points[0][1] + points[1][1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="left", va="center", fontsize=fontsize,
                color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.92))


def _finish(fig, ax, out: Path, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 3.2.3.1
def make_flow_labeling(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 9.0))
    cx, sx = 4.0, 8.6
    w, h = 5.2, 0.82
    dw, dh = 5.4, 1.12

    ys = [12.30, 11.05, 9.80, 8.55, 7.15, 5.75, 4.35, 2.95, 1.78, 0.55]
    _box(ax, cx, ys[0], w, h, "Bảng OHLCV đã làm sạch", "start", 11)
    _box(ax, cx, ys[1], w, h,
         "Lấy lịch phiên chung của toàn sàn\nmarket_dates = sorted(unique(trading_date))")
    _box(ax, cx, ys[2], w, h,
         "Dịch lịch đi 5 phiên\nfuture_by_date = zip(dates, dates.shift(-5))")
    _box(ax, cx, ys[3], w, h,
         "Gán label_end_date cho từng dòng\ntheo lịch chung, không theo lịch riêng của mã")
    _diamond(ax, cx, ys[4], dw, dh, "Còn đủ 5 phiên\ntương lai trên lịch?")
    _box(ax, cx, ys[5], w, h,
         "Ghép giá đóng cửa tại phiên đích\ntheo khóa (symbol, label_end_date)")
    _diamond(ax, cx, ys[6], dw, dh, "Mã có giá tại\nđúng phiên đích?")
    _box(ax, cx, ys[7], w, h,
         "future_return_5d =\nfuture_close_5d / close − 1")
    _box(ax, cx, ys[8], w, h,
         "target = 1 nếu future_return_5d > 1%\nngược lại target = 0")
    _box(ax, cx, ys[9], w, h,
         "ml_dataset.csv\n510.862 dòng, tỷ lệ UP 37,61%", "end", 11)

    for i in range(len(ys) - 1):
        top = ys[i] - (dh if ys[i] in (ys[4], ys[6]) else h) / 2
        bottom = ys[i + 1] + (dh if ys[i + 1] in (ys[4], ys[6]) else h) / 2
        label = "Có" if ys[i] in (ys[4], ys[6]) else None
        _route(ax, [(cx, top), (cx, bottom)], label, label_offset=(0.14, 0.0))

    _box(ax, sx, ys[4], 3.1, 0.86, "Loại dòng\n1.768 dòng", "drop", 10)
    _route(ax, [(cx + dw / 2, ys[4]), (sx - 1.55, ys[4])], "Không",
           label_offset=(0.05, 0.22))
    _box(ax, sx, ys[6], 3.1, 0.86, "Loại dòng\n4.468 dòng", "drop", 10)
    _route(ax, [(cx + dw / 2, ys[6]), (sx - 1.55, ys[6])], "Không",
           label_offset=(0.05, 0.22))

    _finish(fig, ax, out, (0.3, 10.5), (0.0, 12.95))


# ---------------------------------------------------------------- 3.2.3.2
def make_flow_purged_cv(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.4))
    cx = 4.3
    w, h = 5.6, 0.84
    dw, dh = 5.8, 1.14

    ys = [10.75, 9.50, 8.25, 7.00, 5.75, 4.50, 3.25, 1.95, 0.60]
    _box(ax, cx, ys[0], w, h, "Tập TRAIN của ml_dataset.csv", "start", 11)
    _box(ax, cx, ys[1], w, h,
         "Rút danh sách ngày giao dịch duy nhất\nmarket_dates = np.sort(dates.unique())")
    _box(ax, cx, ys[2], w, h,
         "Bỏ các ngày trước 01/01/2021\nđể cắt vùng dữ liệu quá cũ")
    _box(ax, cx, ys[3], w, h,
         "TimeSeriesSplit(n_splits=4, gap=5)\nchia trên mảng ngày, không chia theo dòng")
    _box(ax, cx, ys[4], w, h,
         "validation_start = ngày đầu tiên\ncủa khối kiểm định trong fold")
    _box(ax, cx, ys[5], w, h,
         "PURGE: train_mask = dates.isin(train_dates)\nAND label_ends < validation_start")
    _box(ax, cx, ys[6], w, h,
         "validation_mask = dates.isin(validation_dates)\nlấy trọn ngày, không cắt giữa phiên")
    _diamond(ax, cx, ys[7], dw, dh, "Đã sinh đủ 4 fold?")
    _box(ax, cx, ys[8], w, h,
         "4 cặp chỉ số (train, validation)\nkhông rò rỉ nhãn qua ranh giới", "end", 11)

    for i in range(len(ys) - 1):
        top = ys[i] - (dh if ys[i] == ys[7] else h) / 2
        bottom = ys[i + 1] + (dh if ys[i + 1] == ys[7] else h) / 2
        label = "Rồi" if ys[i] == ys[7] else None
        _route(ax, [(cx, top), (cx, bottom)], label, label_offset=(0.14, 0.0))

    back_x = cx + dw / 2 + 1.35
    _route(ax, [(cx + dw / 2, ys[7]), (back_x, ys[7]), (back_x, ys[4]),
                (cx + w / 2, ys[4])], "Chưa", label_offset=(0.06, 0.24))
    ax.text(back_x + 0.12, (ys[4] + ys[7]) / 2, "Lặp fold\ntiếp theo",
            ha="left", va="center", fontsize=9.5, color=LINE, linespacing=1.3)

    _finish(fig, ax, out, (0.5, 11.6), (0.0, 11.4))


# ---------------------------------------------------------------- 3.2.3.3
def make_flow_threshold(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.1, 8.6))
    cx, sx = 4.6, 10.15
    w, h = 5.6, 0.84
    dw, dh = 5.8, 1.18

    ys = [11.30, 10.05, 8.90, 7.60, 6.20, 4.90, 3.70, 2.45, 1.20]
    _box(ax, cx, ys[0], w, h,
         "Xác suất out-of-fold của cả 4 fold", "start", 11)
    _box(ax, cx, ys[1], w, h,
         "Gom thành một vector xác suất duy nhất\nmỗi dòng TRAIN được dự báo đúng một lần")
    _box(ax, cx, ys[2], w, h, "Đặt ngưỡng thử t = 0,05")
    _box(ax, cx, ys[3], w, h,
         "Dự báo UP nếu p ≥ t\ntính up_rate, precision_UP, F1_UP")
    _diamond(ax, cx, ys[4], dw, dh, "up_rate ≤ 0,50?")
    _diamond(ax, cx, ys[5], dw, dh, "precision_UP ≥\ntỷ lệ UP thực tế?")
    _box(ax, cx, ys[6], w, h, "Đưa (t, F1_UP) vào danh sách ứng viên")
    _diamond(ax, cx, ys[7], dw, dh, "t + 0,01 ≤ 0,95?")
    _box(ax, cx, ys[8], w, h,
         "Chọn t có F1_UP lớn nhất\nkết quả: ngưỡng quyết định 0,49", "end", 11)

    dia_ys = {ys[4], ys[5], ys[7]}
    for i in range(len(ys) - 1):
        top = ys[i] - (dh if ys[i] in dia_ys else h) / 2
        bottom = ys[i + 1] + (dh if ys[i + 1] in dia_ys else h) / 2
        if ys[i] in (ys[4], ys[5]):
            label = "Đạt"
        elif ys[i] == ys[7]:
            label = "Hết dải quét"
        else:
            label = None
        _route(ax, [(cx, top), (cx, bottom)], label, label_offset=(0.14, 0.0))

    reject_left = sx - 1.50
    for yy in (ys[4], ys[5]):
        _box(ax, sx, yy, 3.0, 0.9, "Bỏ ngưỡng t\nkhông xét F1_UP", "drop", 10)
        _route(ax, [(cx + dw / 2, yy), (reject_left, yy)], "Không",
               label_offset=(-0.32, 0.26))

    back_x = 0.55
    _route(ax, [(cx - dw / 2, ys[7]), (back_x, ys[7]), (back_x, ys[3]),
                (cx - w / 2, ys[3])], "Còn ngưỡng", label_offset=(-0.62, 0.28))
    ax.text(back_x - 0.14, (ys[3] + ys[7]) / 2, "Tăng t\nthêm 0,01",
            ha="right", va="center", fontsize=9.5, color=LINE, linespacing=1.3)

    _finish(fig, ax, out, (-0.55, 11.95), (0.4, 11.95))


# ---------------------------------------------------------------- 3.2.3.5
def make_flow_tuning_lab(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.8))
    cx, sx = 4.3, 9.2
    w, h = 5.6, 0.86
    dw, dh = 5.8, 1.18

    ys = [11.55, 10.25, 8.95, 7.65, 6.45, 5.25, 4.05, 2.70, 1.35, 0.20]
    _box(ax, cx, ys[0], w, h,
         "Người dùng nhập siêu tham số tại /tuning", "start", 11)
    _diamond(ax, cx, ys[1], dw, dh,
             "Hợp lệ theo\nTUNABLE_PARAM_SCHEMA?")
    _diamond(ax, cx, ys[2], dw, dh, "Đang có tác vụ\nnền nào chạy?")
    _box(ax, cx, ys[3], w, h,
         "Chạy CV 4 fold trên luồng nền\nvới đúng bộ tham số vừa nhập")
    _box(ax, cx, ys[4], w, h, "Chọn ngưỡng quyết định từ xác suất OOF")
    _box(ax, cx, ys[5], w, h,
         "Ghi experiments/tuning_history.csv\nmã giao thức, fingerprint TRAIN, tham số JSON")
    _box(ax, cx, ys[6], w, h,
         "Chốt cấu hình vào manual_config.json\ncho mô hình tương ứng")
    _diamond(ax, cx, ys[7], dw, dh,
             "is_config_complete: đủ 3 mô hình,\nđúng schema, giao thức, fingerprint?")
    _box(ax, cx, ys[8], w, h,
         "Cho phép scripts/run_pipeline.py\ncông bố mô hình", "end", 11)

    dia_ys = {ys[1], ys[2], ys[7]}
    for i in range(len(ys) - 2):
        top = ys[i] - (dh if ys[i] in dia_ys else h) / 2
        bottom = ys[i + 1] + (dh if ys[i + 1] in dia_ys else h) / 2
        if ys[i] == ys[1]:
            label = "Hợp lệ"
        elif ys[i] == ys[2]:
            label = "Rảnh"
        elif ys[i] == ys[7]:
            label = "Đủ"
        else:
            label = None
        _route(ax, [(cx, top), (cx, bottom)], label, label_offset=(0.14, 0.0))

    _box(ax, sx, ys[1], 3.2, 0.86, "Trả lỗi,\nkhông chạy", "drop", 10)
    _route(ax, [(cx + dw / 2, ys[1]), (sx - 1.60, ys[1])], "Sai",
           label_offset=(-0.34, 0.26))
    _box(ax, sx, ys[2], 3.2, 0.86, "Từ chối: chỉ một\ntác vụ mỗi lúc", "drop", 10)
    _route(ax, [(cx + dw / 2, ys[2]), (sx - 1.60, ys[2])], "Có",
           label_offset=(-0.34, 0.26))
    _box(ax, sx, ys[7], 3.2, 0.86, "Pipeline dừng\nngay từ đầu", "drop", 10)
    _route(ax, [(cx + dw / 2, ys[7]), (sx - 1.60, ys[7])], "Thiếu",
           label_offset=(-0.34, 0.26))

    _finish(fig, ax, out, (0.4, 11.1), (-0.3, 12.2))


# ------------------------------------------------------- so do tuan tu chung
def _sequence(ax, lifelines, messages, top=0.0, step=0.72, box_h=0.62):
    """Ve so do tuan tu. messages: (i, j, text, kind) voi kind in {call,ret,self}."""
    n = len(lifelines)
    xs = [i * (10.0 / max(n - 1, 1)) for i in range(n)]
    depth = top - step * (len(messages) + 1)
    for x, name in zip(xs, lifelines):
        _box(ax, x, top, 2.28, box_h, name, "note", 10.0)
        ax.plot([x, x], [top - box_h / 2, depth], color="#aeb8c2", lw=1.2,
                linestyle=(0, (4, 3)), zorder=1)
    for k, (i, j, text, kind) in enumerate(messages):
        y = top - step * (k + 1)
        if kind == "self":
            x = xs[i]
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.10, y - 0.17), 0.52, 0.34,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor="none", edgecolor=LINE, lw=1.4, zorder=2))
            ax.annotate("", xy=(x + 0.04, y - 0.17), xytext=(x + 0.10, y - 0.17),
                        arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4,
                                        mutation_scale=14), zorder=2)
            ax.text(x + 0.74, y, text, ha="left", va="center", fontsize=9.8,
                    color=INK, linespacing=1.25, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.95))
            continue
        dashed = kind == "ret"
        ax.annotate("", xy=(xs[j], y), xytext=(xs[i], y),
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.5,
                                    mutation_scale=16,
                                    linestyle="--" if dashed else "-"), zorder=2)
        ax.text((xs[i] + xs[j]) / 2, y + 0.16, text, ha="center", va="bottom",
                fontsize=9.8, color=INK, linespacing=1.25, zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.95))
    return depth


# ---------------------------------------------------------------- 3.2.3.6
def make_sequence_predict(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.2))
    lifelines = ["Người dùng", "app.py\n(route Flask)",
                 "prediction_\nservice.py", "models/\nhiện vật",
                 "data/processed/\nhose_stock_clean"]
    messages = [
        (0, 1, "1. POST /predict với 1–2 mã cổ phiếu", "call"),
        (1, 2, "2. predict_symbols(symbols)", "call"),
        (2, 3, "3. đọc final_model.pkl và model_metadata.json", "call"),
        (3, 2, "4. model, feature_order, decision_threshold", "ret"),
        (2, 4, "5. đọc OHLCV đã làm sạch của đúng các mã", "call"),
        (4, 2, "6. bảng giá theo phiên", "ret"),
        (2, 2, "7. sinh 20 đặc trưng, lấy dòng mới nhất mỗi mã", "self"),
        (2, 2, "8. xếp cột theo feature_order trong metadata", "self"),
        (2, 3, "9. predict_proba(x_latest)", "call"),
        (3, 2, "10. probability_up", "ret"),
        (2, 2, "11. gán UP nếu p ≥ 0,49, kèm cảnh báo baseline", "self"),
        (2, 1, "12. nhãn, Điểm UP, ngày tham chiếu, phiên dự kiến", "ret"),
        (1, 0, "13. trang kết quả kèm biểu đồ Chart.js", "ret"),
    ]
    depth = _sequence(ax, lifelines, messages)
    _finish(fig, ax, out, (-1.5, 11.5), (depth - 0.35, 0.7))


# ---------------------------------------------------------------- 3.2.3.7
def make_sequence_chatbot(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    lifelines = ["Người dùng", "Trang chat\n(dock, /chat)",
                 "chatbot_\nservice.py", "chatbot_tools.py\nhandler dữ liệu",
                 "Nhà cung cấp\nLLM"]
    messages = [
        (0, 1, "1. câu hỏi tối đa 1.000 ký tự", "call"),
        (1, 2, "2. POST /api/chat kèm tối đa 6 tin nhắn lịch sử", "call"),
        (2, 2, "3. định tuyến ý định bằng luật (rule-based)", "self"),
        (2, 3, "4. gọi trực tiếp handler dữ liệu tương ứng", "call"),
        (3, 2, "5. artifact, report, tín hiệu đã công bố hoặc suy ra", "ret"),
        (2, 2, "6. dựng context tối đa 16.000 ký tự", "self"),
        (2, 4, "7. gọi mô hình đúng một lần, không gửi tools", "call"),
        (4, 2, "8. câu trả lời tối đa 1.000 ký tự", "ret"),
        (2, 2, "9. kiểm release, nguồn, cặp mã–trường–giá trị", "self"),
        (2, 2, "10. chặn khuyến nghị mua bán trực tiếp", "self"),
        (2, 1, "11. câu trả lời đã kiểm hoặc thông báo từ chối", "ret"),
        (1, 0, "12. hiển thị, lưu vào sessionStorage dùng chung", "ret"),
    ]
    depth = _sequence(ax, lifelines, messages)
    _finish(fig, ax, out, (-1.5, 11.5), (depth - 0.35, 0.7))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_flow_labeling(ASSETS / "flow_labeling.png")
    make_flow_purged_cv(ASSETS / "flow_purged_cv.png")
    make_flow_threshold(ASSETS / "flow_threshold.png")
    make_flow_tuning_lab(ASSETS / "flow_tuning_lab.png")
    make_sequence_predict(ASSETS / "sequence_predict.png")
    make_sequence_chatbot(ASSETS / "sequence_chatbot.png")
    print("wrote 6 flow/sequence figures")


if __name__ == "__main__":
    main()
