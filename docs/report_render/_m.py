import re, zipfile
z = zipfile.ZipFile("_preview.docx")
d = z.read("word/document.xml").decode("utf-8")
s = z.read("word/styles.xml").decode("utf-8")
for name in ("CoverText","CoverHeading","CoverSub","CoverTitle","Heading1","Caption"):
    m = re.search(r'w:styleId="%s".*?</w:style>' % name, s, re.S)
    b = m.group(0) if m else ""
    ind = re.search(r"<w:ind[^/]*/>", b)
    print(name, "->", ind.group(0) if ind else "NO-IND")
paras = re.findall(r"<w:p>.*?</w:p>", d, re.S)
print("--- first 8 cover paras ---")
for i, p in enumerate(paras[:8]):
    txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))[:40]
    ind = re.search(r"<w:ind[^/]*/>", p)
    print(i, (ind.group(0) if ind else "no-ind"), "|", txt if txt.strip() else ("[IMG]" if "drawing" in p else "(blank)"))