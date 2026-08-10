import sys, importlib
from pathlib import Path
sys.path.insert(0, ".")
import build_report as br
br.OUTPUT = Path("_preview.docx")
br.main()