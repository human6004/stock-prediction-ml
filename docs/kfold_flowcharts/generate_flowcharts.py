#!/usr/bin/env python3
"""Generate 3 distinct 4-Fold CV Hyperparameter Tuning Flowcharts for HOSE Stock Prediction ML.

Models:
1. Logistic Regression (model_id=2)
2. Random Forest (model_id=3)
3. Gradient Boosting (model_id=4)

Outputs:
- docs/kfold_flowcharts/flow_tuning_logistic_regression.drawio (.png)
- docs/kfold_flowcharts/flow_tuning_random_forest.drawio (.png)
- docs/kfold_flowcharts/flow_tuning_gradient_boosting.drawio (.png)
"""

from __future__ import annotations

import html
import pathlib
import subprocess
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Drawio binary resolution
DRAWIO_BIN = r"C:\Users\nhan\AppData\Local\Programs\draw.io\draw.io.EXE"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color Palette matching Figure 3.3 / make_flow_figures.py
INK = "#1f2933"
LINE = "#41505f"
BG_WHITE = "#ffffff"

COLORS = {
    "start": {"fill": "#dce9f5", "stroke": "#2f7fb8"},
    "end": {"fill": "#dbeee5", "stroke": "#2e8b6f"},
    "proc": {"fill": "#ffffff", "stroke": "#41505f"},
    "dec": {"fill": "#fdf3dd", "stroke": "#c2793a"},
    "drop": {"fill": "#fbe4e4", "stroke": "#b4534b"},
    "banner": {"fill": "#f5f8fb", "stroke": "#7f8c9b"},
}

MODELS_CONFIG = [
    {
        "id": "logistic_regression",
        "title": "Logistic Regression",
        "file_base": "flow_tuning_logistic_regression",
        "caption": "Hình 3.x. Lưu đồ tinh chỉnh Logistic Regression trên kiểm định chéo 4 fold",
        "start_text": "Tập TRAIN (20 đặc trưng + target)&#xa;+ Siêu tham số Logistic Regression (C, solver)",
        "fit_text": "Clone Pipeline(StandardScaler + LogisticRegression)&#xa;class_weight=&quot;balanced&quot;, max_iter=1000, random_state=42&#xa;Fit fold k; thu y_val và P(UP)",
        "end_text": "Chốt cấu hình Logistic Regression&#xa;C = 2,16e-05; solver = liblinear&#xa;F1_UP = 0,4571; Precision_UP = 0,4097; Recall_UP = 0,5419&#xa;Ngưỡng quyết định = 0,49",
    },
    {
        "id": "random_forest",
        "title": "Random Forest",
        "file_base": "flow_tuning_random_forest",
        "caption": "Hình 3.x. Lưu đồ tinh chỉnh Random Forest trên kiểm định chéo 4 fold",
        "start_text": "Tập TRAIN (20 đặc trưng + target)&#xa;+ Siêu tham số Random Forest&#xa;(n_estimators, max_depth, min_samples_leaf, max_features)",
        "fit_text": "Clone RandomForestClassifier (không dùng scaler)&#xa;class_weight=&quot;balanced_subsample&quot;, n_jobs=-1, random_state=42&#xa;Fit fold k; thu y_val và P(UP)",
        "end_text": "Chốt cấu hình Random Forest&#xa;n_estimators = 108; max_depth = 24&#xa;min_samples_leaf = 100; max_features = 0,2&#xa;F1_UP = 0,4755; Precision_UP = 0,4190; Recall_UP = 0,5574&#xa;Ngưỡng quyết định = 0,48",
    },
    {
        "id": "gradient_boosting",
        "title": "Gradient Boosting",
        "file_base": "flow_tuning_gradient_boosting",
        "caption": "Hình 3.x. Lưu đồ tinh chỉnh Gradient Boosting trên kiểm định chéo 4 fold",
        "start_text": "Tập TRAIN (20 đặc trưng + target)&#xa;+ Siêu tham số Gradient Boosting&#xa;(n_estimators, learning_rate, max_depth, subsample)",
        "fit_text": "Clone GradientBoostingClassifier, random_state=42&#xa;(không class_weight; dùng sample_weight=&quot;balanced&quot;)&#xa;Fit fold k; thu y_val và P(UP)",
        "end_text": "Chốt cấu hình Gradient Boosting&#xa;n_estimators = 100; learning_rate = 0,4&#xa;max_depth = 2; subsample = 0,5&#xa;F1_UP = 0,4765; Precision_UP = 0,4227; Recall_UP = 0,5499&#xa;Ngưỡng quyết định = 0,49",
    },
]


def build_drawio_xml(model_cfg: dict) -> str:
    """Build exact draw.io XML matching Figure 3.3 geometry and style."""
    center_x = 460
    main_w = 480
    left_x = center_x - main_w // 2  # 220

    # Diamonds have same width as main process boxes (480px) for the wide elegant diamond style of Fig 3.3
    dec_w = 480
    dec_x = center_x - dec_w // 2  # 220

    # Banner
    banner_y = 28
    banner_h = 36
    banner_w = 600

    # 1: Start (Blue)
    start_y = 86
    start_h = 58

    # 2: Sort Panel
    p1_y = 168
    p1_h = 48

    # 3: Unique dates & filter < 2021
    p2_y = 240
    p2_h = 48

    # 4: TimeSeriesSplit + Purge
    p3_y = 312
    p3_h = 66

    # 5: Decision 1 (Train/val fold rỗng?)
    dec1_y = 402
    dec1_h = 66

    # 5_drop: Reject (Red box on the right)
    drop_x = 760
    drop_w = 175
    drop_h = 56
    drop_y = dec1_y + (dec1_h - drop_h) // 2

    # 6: Set k = 1
    p4_y = 492
    p4_h = 44

    # 7: Fit model on fold k
    fit_y = 560
    fit_h = 68

    # 8: Decision 2 (k < 4 ?)
    dec2_y = 652
    dec2_h = 64

    # 9: OOF vector combine
    oof_y = 740
    oof_h = 50

    # 10: OOF Threshold selection
    th_y = 814
    th_h = 72

    # 11: Apply threshold on 4 folds
    eval_y = 910
    eval_h = 50

    # 12: End (Green)
    end_y = 984
    end_h = 84

    # 13: Caption
    cap_y = 1092
    cap_h = 28

    # Left Loop routing
    loop_x = 90  # X position of the vertical return line

    nodes = []

    # Banner
    banner_x = center_x - banner_w // 2
    nodes.append(
        f'<mxCell id="banner" value="Đề tài: Dự báo xu hướng giá cổ phiếu HOSE bằng Machine Learning" '
        f'style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={COLORS["banner"]["fill"]};'
        f'strokeColor={COLORS["banner"]["stroke"]};strokeWidth=1.2;fontColor={INK};'
        f'fontFamily=Arial;fontSize=11.5;fontStyle=1;" vertex="1" parent="1">'
        f'<mxGeometry x="{banner_x}" y="{banner_y}" width="{banner_w}" height="{banner_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 1. Start Node (rounded pill-like)
    nodes.append(
        f'<mxCell id="start" value="{model_cfg["start_text"]}" '
        f'style="rounded=1;arcSize=24;whiteSpace=wrap;html=1;fillColor={COLORS["start"]["fill"]};'
        f'strokeColor={COLORS["start"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=11;fontStyle=1;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{start_y}" width="{main_w}" height="{start_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 2. Sort panel
    nodes.append(
        f'<mxCell id="p1" value="Sắp xếp panel theo (trading_date, symbol)&#xa;hàm sort_panel_frame" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{p1_y}" width="{main_w}" height="{p1_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 3. Unique dates & date filter
    nodes.append(
        f'<mxCell id="p2" value="Rút danh sách ngày giao dịch duy nhất&#xa;Bỏ các ngày trước 01/01/2021" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{p2_y}" width="{main_w}" height="{p2_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 4. TimeSeriesSplit + Purge
    nodes.append(
        f'<mxCell id="p3" value="TimeSeriesSplit(n_splits=4, gap=5)&#xa;Chia trên mảng ngày, không chia theo dòng&#xa;Purge: loại train row có label_end_date ≥ validation_start" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{p3_y}" width="{main_w}" height="{p3_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 5. Decision 1 (Empty fold?)
    nodes.append(
        f'<mxCell id="dec1" value="Train hoặc val fold rỗng?" '
        f'style="rhombus;whiteSpace=wrap;html=1;fillColor={COLORS["dec"]["fill"]};'
        f'strokeColor={COLORS["dec"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.2;" vertex="1" parent="1">'
        f'<mxGeometry x="{dec_x}" y="{dec1_y}" width="{dec_w}" height="{dec1_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 5_drop: Reject (Red box on the right)
    nodes.append(
        f'<mxCell id="drop1" value="Raise lỗi (ValueError)&#xa;Không chấm cấu hình" '
        f'style="rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={COLORS["drop"]["fill"]};'
        f'strokeColor={COLORS["drop"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{drop_x}" y="{drop_y}" width="{drop_w}" height="{drop_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 6. Set k = 1
    nodes.append(
        f'<mxCell id="p4" value="Khởi tạo vòng lặp: Đặt k = 1" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{p4_y}" width="{main_w}" height="{p4_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 7. Fit model on fold k
    nodes.append(
        f'<mxCell id="fit" value="{model_cfg["fit_text"]}" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{fit_y}" width="{main_w}" height="{fit_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 8. Decision 2 (k < 4 ?)
    nodes.append(
        f'<mxCell id="dec2" value="k &lt; 4 ?" '
        f'style="rhombus;whiteSpace=wrap;html=1;fillColor={COLORS["dec"]["fill"]};'
        f'strokeColor={COLORS["dec"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.2;" vertex="1" parent="1">'
        f'<mxGeometry x="{dec_x}" y="{dec2_y}" width="{dec_w}" height="{dec2_h}" as="geometry" />'
        f'</mxCell>'
    )

    # Text annotation on diamond 2 left exit: "Còn fold"
    nodes.append(
        f'<mxCell id="lbl_con_fold" value="Còn fold" '
        f'style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=bottom;'
        f'fontColor={LINE};fontFamily=Arial;fontSize=10;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x - 75}" y="{dec2_y + dec2_h // 2 - 20}" width="65" height="18" as="geometry" />'
        f'</mxCell>'
    )

    # Text annotation on the left vertical return line: "Tăng k thêm 1"
    loop_label_y = (fit_y + fit_h // 2 + dec2_y + dec2_h // 2) // 2
    nodes.append(
        f'<mxCell id="loop_lbl" value="Tăng k&#xa;thêm 1" '
        f'style="text;html=1;strokeColor=none;fillColor=none;align=right;verticalAlign=middle;'
        f'fontColor={LINE};fontFamily=Arial;fontSize=9.5;linespacing=1.2;" vertex="1" parent="1">'
        f'<mxGeometry x="{loop_x - 66}" y="{loop_label_y - 14}" width="60" height="28" as="geometry" />'
        f'</mxCell>'
    )

    # 9. OOF vector combine
    nodes.append(
        f'<mxCell id="p5" value="Gom xác suất 4 fold thành một vector OOF&#xa;Mỗi dòng TRAIN được dự báo đúng một lần" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{oof_y}" width="{main_w}" height="{oof_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 10. OOF Threshold selection
    nodes.append(
        f'<mxCell id="p6" value="Chọn một ngưỡng t từ toàn bộ OOF&#xa;Quét t = 0,05 → 0,95, bước 0,01&#xa;Bỏ t nếu up_rate &gt; 0,50 hoặc precision_UP &lt; tỷ lệ UP thực tế&#xa;Chọn t có F1_UP lớn nhất" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{th_y}" width="{main_w}" height="{th_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 11. Apply threshold on 4 folds
    nodes.append(
        f'<mxCell id="p7" value="Áp ngưỡng t chung cho 4 fold&#xa;Tính F1_UP, Precision_UP, Recall_UP trung bình" '
        f'style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={COLORS["proc"]["fill"]};'
        f'strokeColor={COLORS["proc"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{eval_y}" width="{main_w}" height="{eval_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 12. End Node (rounded pill-like)
    nodes.append(
        f'<mxCell id="end" value="{model_cfg["end_text"]}" '
        f'style="rounded=1;arcSize=24;whiteSpace=wrap;html=1;fillColor={COLORS["end"]["fill"]};'
        f'strokeColor={COLORS["end"]["stroke"]};strokeWidth=1.5;fontColor={INK};'
        f'fontFamily=Arial;fontSize=10.5;fontStyle=1;linespacing=1.25;" vertex="1" parent="1">'
        f'<mxGeometry x="{left_x}" y="{end_y}" width="{main_w}" height="{end_h}" as="geometry" />'
        f'</mxCell>'
    )

    # 13. Caption
    nodes.append(
        f'<mxCell id="caption" value="{model_cfg["caption"]}" '
        f'style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;'
        f'fontColor={INK};fontFamily=Times New Roman;fontSize=12.5;fontStyle=2;" vertex="1" parent="1">'
        f'<mxGeometry x="{center_x - 300}" y="{cap_y}" width="600" height="{cap_h}" as="geometry" />'
        f'</mxCell>'
    )

    edges = []

    def add_edge(edge_id, src, tgt, label="", exit_pos=(0.5, 1), entry_pos=(0.5, 0)):
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={LINE};strokeWidth=1.5;fontColor={LINE};fontFamily=Arial;fontSize=10;"
            f"exitX={exit_pos[0]};exitY={exit_pos[1]};exitDx=0;exitDy=0;"
            f"entryX={entry_pos[0]};entryY={entry_pos[1]};entryDx=0;entryDy=0;"
        )
        if label:
            style += "labelBackgroundColor=#ffffff;"
        lbl_val = html.escape(label) if label else ""
        edges.append(
            f'<mxCell id="{edge_id}" value="{lbl_val}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
            f'<mxGeometry relative="1" as="geometry" />'
            f'</mxCell>'
        )

    # Downward straight edges
    add_edge("e_start_p1", "start", "p1")
    add_edge("e_p1_p2", "p1", "p2")
    add_edge("e_p2_p3", "p2", "p3")
    add_edge("e_p3_dec1", "p3", "dec1")

    # Decision 1 branches
    # Branch Right: Có -> drop1
    add_edge("e_dec1_drop", "dec1", "drop1", label="Có", exit_pos=(1, 0.5), entry_pos=(0, 0.5))
    # Branch Down: Không -> p4
    add_edge("e_dec1_p4", "dec1", "p4", label="Không", exit_pos=(0.5, 1), entry_pos=(0.5, 0))

    # p4 -> fit
    add_edge("e_p4_fit", "p4", "fit")

    # fit -> dec2
    add_edge("e_fit_dec2", "fit", "dec2")

    # Decision 2 branches
    # Branch Down: Hết 4 fold -> p5
    add_edge("e_dec2_p5", "dec2", "p5", label="Hết 4 fold", exit_pos=(0.5, 1), entry_pos=(0.5, 0))

    # Branch Left (Loop back to fit): loop_x -> fit
    loop_edge = (
        f'<mxCell id="e_loop" value="" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;'
        f'strokeColor={LINE};strokeWidth=1.5;fontColor={LINE};fontFamily=Arial;fontSize=10;'
        f'exitX=0;exitY=0.5;exitDx=0;exitDy=0;'
        f'entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="dec2" target="fit">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<Array as="points">'
        f'<mxPoint x="{loop_x}" y="{dec2_y + dec2_h // 2}" />'
        f'<mxPoint x="{loop_x}" y="{fit_y + fit_h // 2}" />'
        f'</Array>'
        f'</mxGeometry>'
        f'</mxCell>'
    )
    edges.append(loop_edge)

    # p5 -> p6 -> p7 -> end
    add_edge("e_p5_p6", "p5", "p6")
    add_edge("e_p6_p7", "p6", "p7")
    add_edge("e_p7_end", "p7", "end")

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" version="31.1.8">
  <diagram name="{model_cfg['title']}" id="{model_cfg['id']}">
    <mxGraphModel dx="1200" dy="1400" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="960" pageHeight="1150" background="#ffffff" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {chr(10).join("        " + n for n in nodes)}
        {chr(10).join("        " + e for e in edges)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    return xml_content


def main():
    print(f"Generating 3 flowcharts into: {OUTPUT_DIR}")
    for model_cfg in MODELS_CONFIG:
        drawio_path = OUTPUT_DIR / f"{model_cfg['file_base']}.drawio"
        png_path = OUTPUT_DIR / f"{model_cfg['file_base']}.png"

        xml = build_drawio_xml(model_cfg)
        drawio_path.write_text(xml, encoding="utf-8")
        print(f"-> Wrote {drawio_path.name}")

        # Export PNG using draw.io CLI (scale 2 for high crispness, width capped)
        cmd = [
            DRAWIO_BIN,
            "-x",
            "-f",
            "png",
            "-s",
            "2",
            "-b",
            "10",
            "-o",
            str(png_path),
            str(drawio_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error exporting {png_path.name}: {res.stderr}")
        else:
            print(f"-> Exported {png_path.name} (RC: {res.returncode})")

    print("\nAll 3 flowcharts generated and exported successfully!")


if __name__ == "__main__":
    main()
