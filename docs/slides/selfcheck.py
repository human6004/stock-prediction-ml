"""Doc lai file pptx da xuat va kiem cac rang buoc trinh bay."""
import re
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu

BASE = Path(__file__).resolve().parents[2]
SLIDES = BASE / "docs" / "slides"
import os
OUT = SLIDES / os.environ.get("SLIDE_OUT","thuyet_trinh_nien_luan.pptx")
REPORT = SLIDES / os.environ.get("SELFCHECK_OUT","selfcheck_report.txt")

prs = Presentation(str(OUT))
lines = []
problems = []
lines.append(f"file={OUT}")
lines.append(f"size_bytes={OUT.stat().st_size}")
lines.append(f"slides={len(prs.slides.__iter__.__self__._sldIdLst)}")
lines.append(f"slide_w_emu={prs.slide_width} slide_h_emu={prs.slide_height}")
ratio = prs.slide_width / prs.slide_height
lines.append(f"aspect_ratio={ratio:.4f} (16:9 = 1.7778)")
if abs(ratio - 16 / 9) > 0.01:
    problems.append("aspect ratio khong phai 16:9")

fonts = set()
for idx, slide in enumerate(prs.slides, start=1):
    bullets = []
    texts = []
    pics = 0
    tables = 0
    for shape in slide.shapes:
        if shape.shape_type == 13 or shape.__class__.__name__ == "Picture":
            pics += 1
            blip = shape._element.blipFill.blip
            rid = blip.rEmbed
            part = slide.part.related_part(rid)
            name = Path(str(part.partname)).name
            found = list((SLIDES / "img").glob("*"))
            lines.append(f"  IMG: {name} embedded_bytes={len(part.blob)}")
            if len(part.blob) == 0:
                problems.append(f"s{idx}: anh rong {name}")
        if getattr(shape, "has_table", False) and shape.has_table:
            tables += 1
            tbl = shape.table
            lines.append(f"  TABLE {len(tbl.rows)}x{len(tbl.columns)}")
            for r in tbl.rows:
                cells = [c.text.strip() for c in r.cells]
                lines.append("    | " + " | ".join(cells))
                for c in r.cells:
                    for p in c.text_frame.paragraphs:
                        for run in p.runs:
                            if run.font.name:
                                fonts.add(run.font.name)
        if not getattr(shape, "has_text_frame", False):
            continue
        tf = shape.text_frame
        for p in tf.paragraphs:
            txt = "".join(run.text for run in p.runs).strip()
            for run in p.runs:
                if run.font.name:
                    fonts.add(run.font.name)
            if not txt:
                continue
            if txt.startswith("\u2013 ") or txt.startswith("\u2022 "):
                bullets.append(txt[2:].strip())
            else:
                texts.append(txt)
    notes = ""
    if slide.has_notes_slide:
        notes = slide.notes_slide.notes_text_frame.text.strip()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", notes) if s.strip()]
    lines.append(
        f"=== SLIDE {idx} | bullets={len(bullets)} pics={pics} tables={tables} notes_sent={len(sents)}"
    )
    for t in texts:
        lines.append(f"  TXT: {t}")
    for b in bullets:
        lines.append(f"  BUL({len(b.split())}w): {b}")
    lines.append(f"  NOTES: {notes}")

    if len(bullets) > 6:
        problems.append(f"s{idx}: {len(bullets)} bullets > 6")
    for b in bullets:
        if len(b.split()) > 12:
            problems.append(f"s{idx}: bullet {len(b.split())} tu: {b}")
    if not notes:
        problems.append(f"s{idx}: thieu speaker notes")
    elif len(sents) < 3 or len(sents) > 6:
        problems.append(f"s{idx}: notes {len(sents)} cau (yeu cau 3-5)")

lines.append(f"FONTS={sorted(fonts)}")
non_arial = [f for f in fonts if f != "Arial"]
if non_arial:
    problems.append(f"font khac Arial: {non_arial}")

# Kiem moi anh duoc build_pptx.py tham chieu co ton tai tren disk.
build_src = (SLIDES / "build_pptx.py").read_text(encoding="utf-8")
refs = set(re.findall(r'(?:IMG|ASSETS)\s*/\s*"([^"]+)"', build_src))
lines.append(f"IMAGE_REFS={sorted(refs)}")
for name in sorted(refs):
    candidates = [SLIDES / "img" / name, BASE / "docs" / "report_assets" / name]
    if not any(c.exists() and c.stat().st_size > 0 for c in candidates):
        problems.append(f"anh tham chieu khong ton tai tren disk: {name}")

lines.append("")
lines.append("=== PROBLEMS ===")
for p in problems:
    lines.append(p)
lines.append(f"total_problems={len(problems)}")
REPORT.write_text("\n".join(lines), encoding="utf-8")
