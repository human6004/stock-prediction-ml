import re, zipfile, pathlib
d = pathlib.Path(r"D:\study\niên luận\stock-prediction-ml\docs")
z = zipfile.ZipFile(d / "bao_cao_project_hose_stock_prediction.docx")
doc = z.read("word/document.xml").decode("utf-8")
st = z.read("word/styles.xml").decode("utf-8")

out = []
# section margins
m = re.search(r"<w:sectPr>.*?</w:sectPr>", doc, re.S)
out.append("SECT: " + m.group(0))

# cover styles definitions
for sid in ("Normal", "CoverText", "CoverHeading", "CoverSub", "CoverTitle", "Heading1", "Caption"):
    ms = re.search(r'<w:style [^>]*w:styleId="%s">.*?</w:style>' % sid, st, re.S)
    if ms:
        body = ms.group(0)
        ind = re.search(r"<w:ind[^/]*/>", body)
        jc = re.search(r'<w:jc w:val="(\w+)"/>', body)
        out.append("%-13s jc=%s ind=%s" % (sid, jc.group(1) if jc else "-", ind.group(0) if ind else "-"))

# first 25 paragraphs raw pPr
paras = re.findall(r"<w:p(?: [^>]*)?>.*?</w:p>", doc, re.S)
out.append("total paras=%d" % len(paras))
for i, p in enumerate(paras[:24]):
    txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))[:46]
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", p, re.S)
    out.append("%02d | %s | %s" % (i, txt or ("[IMG]" if "w:drawing" in p else ""), ppr.group(0) if ppr else "-"))

pathlib.Path(r"D:\study\niên luận\stock-prediction-ml\docs\report_render\_insp.txt").write_text("\n".join(out), encoding="utf-8")