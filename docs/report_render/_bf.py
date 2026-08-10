import sys, importlib
from pathlib import Path
sys.path.insert(0, ".")
import build_report as br
importlib.reload(br)
from docx_builder import DocxBuilder
from content_front import build_cover, build_abstract, build_toc, build_abbreviations, build_work_assignment
from content_ch12 import build_chapter1, build_chapter2
from content_ch3a import build_srs
from content_ch3b import build_design
from content_ch3c import build_testing
from content_ch4 import build_chapter4, build_references, build_appendices

out = br.DOCS_DIR / "bao_cao_project_hose_stock_prediction_FIXED.docx"
doc = DocxBuilder()
build_cover(doc, br.ASSETS)
build_abstract(doc)
build_toc(doc)
build_abbreviations(doc)
build_work_assignment(doc)
build_chapter1(doc, br.ASSETS)
build_chapter2(doc, br.ASSETS)
build_srs(doc, br.ASSETS)
build_design(doc, br.ASSETS, br.DIAGRAMS)
build_testing(doc, br.ASSETS)
build_chapter4(doc)
build_references(doc)
build_appendices(doc, br.ASSETS, br.DIAGRAMS)
doc.save(out)
print("written", out.stat().st_size)