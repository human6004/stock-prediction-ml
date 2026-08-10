import pathlib, re
b = pathlib.Path(r"D:\study\niên luận\stock-prediction-ml\docs\report_render\docx_builder.py")
s = b.read_text(encoding="utf-8")

# 1) Cac style bia can giua theo trang giay: bu lech le trai/phai (1980-1018)/2 = 481 twip.
for sid in ("CoverTitle", "CoverText", "CoverHeading", "CoverSub"):
    pat = re.compile(r'(w:styleId="%s">.*?)<w:ind w:left="0" w:right="0" w:firstLine="0"/>' % sid, re.S)
    s2 = pat.sub(r'\1<w:ind w:left="-481" w:right="481" w:firstLine="0"/>', s, count=1)
    assert s2 != s, sid
    s = s2

# 2) Anh tren bia cung phai bu lech; them tham so page_center cho image().
old_sig = "def image(self, path: Path, caption: str | None = None, *, width_cm: float | None = None) -> None:"
new_sig = ("def image(\n        self,\n        path: Path,\n        caption: str | None = None,\n"
           "        *,\n        width_cm: float | None = None,\n        page_center: bool = False,\n    ) -> None:")
assert old_sig in s
s = s.replace(old_sig, new_sig, 1)

old_p = '\'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/></w:pPr>\''
new_p = ('\'<w:p><w:pPr><w:jc w:val="center"/>\'\n            + (\'<w:ind w:left="-481" w:right="481"/>\' if page_center else "")\n'
         '            + \'<w:spacing w:before="120" w:after="60"/></w:pPr>\'')
assert old_p in s
s = s.replace(old_p, new_p, 1)

b.write_text(s, encoding="utf-8")
print("builder ok")