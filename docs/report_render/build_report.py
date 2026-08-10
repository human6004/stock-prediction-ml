"""Build the CT239H report .docx from the section modules."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from docx_builder import DocxBuilder
from content_front import (
    build_cover,
    build_abstract,
    build_toc,
    build_abbreviations,
    build_work_assignment,
)
from content_ch12 import build_chapter1, build_chapter2
from content_ch3a import build_srs
from content_ch3b import build_design
from content_ch3c import build_testing
from content_ch4 import build_chapter4, build_references, build_appendices

DOCS_DIR = HERE.parent
ASSETS = DOCS_DIR / "report_assets"
DIAGRAMS = DOCS_DIR / "diagrams"
OUTPUT = DOCS_DIR / "bao_cao_project_hose_stock_prediction.docx"


def main() -> None:
    doc = DocxBuilder()
    build_cover(doc, ASSETS)
    build_abstract(doc)
    build_toc(doc)
    build_abbreviations(doc)
    build_work_assignment(doc)
    build_chapter1(doc, ASSETS)
    build_chapter2(doc, ASSETS)
    build_srs(doc, ASSETS)
    build_design(doc, ASSETS, DIAGRAMS)
    build_testing(doc, ASSETS)
    build_chapter4(doc)
    build_references(doc)
    build_appendices(doc, ASSETS, DIAGRAMS)
    doc.save(OUTPUT)
    print("written bytes=%d" % OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()