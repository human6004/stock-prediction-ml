"""Minimal OOXML (.docx) writer built on the standard library only.

Repo khong cai duoc python-docx (pip khong co temp dir kha dung), nen module nay
sinh truc tiep WordprocessingML: styles theo guideline CT239H, heading 4 cap,
field TOC/PAGE tu dong, bang kieu booktabs, cong thuc toan OMML (Word equation)
va anh PNG nhung kem caption SEQ.
"""

from __future__ import annotations

import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

EMU_PER_CM = 360000
MAX_IMAGE_WIDTH_CM = 15.5
# Chieu rong vung chu voi le trai 3cm / phai 2cm tren kho A4 (twips).
TEXT_WIDTH_TWIPS = 9072

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)

MATH_FONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


class M:
    """Cac khoi OMML co ban, dung nhu mini-DSL kieu LaTeX cho content_*.py.

    Moi ham tra ve mot chuoi XML OMML; ghep chuoi la ghep bieu thuc theo chieu
    ngang. Quy uoc: bien viet nghieng (mac dinh cua math run), ten ham va chu
    tieng Viet dung ``M.t`` de ra chu dung (upright).
    """

    @staticmethod
    def r(text: str) -> str:
        """Math run nghieng mac dinh (bien, toan tu)."""
        return f'<m:r>{MATH_FONT}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'

    @staticmethod
    def t(text: str) -> str:
        """Math run CHU DUNG cho ten ham (log, std, max) va chu thich."""
        return (
            '<m:r><m:rPr><m:sty m:val="p"/></m:rPr>'
            f'{MATH_FONT}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'
        )

    @staticmethod
    def frac(num: str, den: str) -> str:
        return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"

    @staticmethod
    def sub(base: str, subscript: str) -> str:
        return f"<m:sSub><m:e>{base}</m:e><m:sub>{subscript}</m:sub></m:sSub>"

    @staticmethod
    def sup(base: str, superscript: str) -> str:
        return f"<m:sSup><m:e>{base}</m:e><m:sup>{superscript}</m:sup></m:sSup>"

    @staticmethod
    def subsup(base: str, subscript: str, superscript: str) -> str:
        return (
            f"<m:sSubSup><m:e>{base}</m:e><m:sub>{subscript}</m:sub>"
            f"<m:sup>{superscript}</m:sup></m:sSubSup>"
        )

    @staticmethod
    def paren(body: str, *, beg: str = "(", end: str = ")") -> str:
        pr = ""
        if beg != "(" or end != ")":
            pr = f'<m:begChr m:val="{escape(beg)}"/><m:endChr m:val="{escape(end)}"/>'
        return f"<m:d><m:dPr>{pr}</m:dPr><m:e>{body}</m:e></m:d>"

    @staticmethod
    def total(lo: str, hi: str, body: str) -> str:
        """Tong sigma voi can duoi/tren dat tren-duoi (nhu \\sum cua LaTeX)."""
        sup_hide = '<m:supHide m:val="1"/>' if not hi else ""
        sup_part = f"<m:sup>{hi}</m:sup>" if hi else "<m:sup/>"
        return (
            '<m:nary><m:naryPr><m:chr m:val="\u2211"/><m:limLoc m:val="undOvr"/>'
            f'<m:grow m:val="1"/>{sup_hide}</m:naryPr>'
            f"<m:sub>{lo}</m:sub>{sup_part}<m:e>{body}</m:e></m:nary>"
        )

    @staticmethod
    def cases(rows: list[str]) -> str:
        """Dinh nghia theo truong hop, dau ngoac nhon mot ben nhu \\begin{cases}."""
        arr = "".join(f"<m:e>{row}</m:e>" for row in rows)
        return (
            '<m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val=""/></m:dPr>'
            f"<m:e><m:eqArr>{arr}</m:eqArr></m:e></m:d>"
        )

    @staticmethod
    def norm(body: str) -> str:
        return M.paren(body, beg="\u2016", end="\u2016")


def png_size(path: Path) -> tuple[int, int]:
    """Doc width/height tu IHDR cua file PNG."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


@dataclass
class DocxBuilder:
    body: list[str] = field(default_factory=list)
    images: list[tuple[str, Path]] = field(default_factory=list)
    _image_ids: dict[str, str] = field(default_factory=dict)
    _drawing_id: int = 0
    _caption_prefix: str | None = None
    _caption_reset: set[str] = field(default_factory=set)
    _caption_counter: dict[str, int] = field(default_factory=dict)
    header_topic: str = "Đề tài: Dự báo xu hướng giá cổ phiếu HOSE bằng Machine Learning"

    # ---------- text ----------
    def _run(self, text: str, *, bold=False, italic=False, mono=False) -> str:
        props = []
        if mono:
            props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/>')
        if bold:
            props.append("<w:b/>")
        if italic:
            props.append("<w:i/>")
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        return (
            f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
        )

    def paragraph(
        self,
        text: str = "",
        *,
        style: str | None = None,
        bold=False,
        italic=False,
        align: str | None = None,
        spacing_after: int | None = None,
    ) -> None:
        props = []
        if style:
            props.append(f'<w:pStyle w:val="{style}"/>')
        if align:
            props.append(f'<w:jc w:val="{align}"/>')
        if spacing_after is not None:
            props.append(f'<w:spacing w:after="{spacing_after}"/>')
        ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
        run = self._run(text, bold=bold, italic=italic) if text else ""
        self.body.append(f"<w:p>{ppr}{run}</w:p>")

    def heading(self, level: int, text: str, *, page_break=False) -> None:
        if level not in (1, 2, 3, 4):
            raise ValueError("Heading level must be 1..4")
        self._track_caption_chapter(level, text)
        props = [f'<w:pStyle w:val="Heading{level}"/>']
        if page_break:
            props.insert(0, "<w:pageBreakBefore/>")
        self.body.append(
            f"<w:p><w:pPr>{''.join(props)}</w:pPr>{self._run(text)}</w:p>"
        )

    def bullets(self, items: list[str], *, numbered=False) -> None:
        for index, item in enumerate(items, start=1):
            marker = f"{index}. " if numbered else "- "
            self.body.append(
                "<w:p><w:pPr><w:ind w:left=\"567\" w:hanging=\"283\"/>"
                "<w:spacing w:after=\"60\"/></w:pPr>"
                f"{self._run(marker + item)}</w:p>"
            )

    def code_block(self, text: str) -> None:
        for line in text.strip("\n").split("\n"):
            self.body.append(
                '<w:p><w:pPr><w:pStyle w:val="CodeBlock"/></w:pPr>'
                f"{self._run(line or ' ', mono=True)}</w:p>"
            )

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    # ---------- math ----------
    def equation(self, body: str, number: str | None = None) -> None:
        """Cong thuc hien thi kieu LaTeX: can giua dong, so thu tu can phai.

        ``body`` la chuoi OMML ghep tu cac khoi ``M.*``. Tab giua (vi tri nua
        chieu rong vung chu) can giua bieu thuc, tab phai day so cong thuc ra
        sat le phai — dung layout cua \\begin{equation}.
        """
        center = TEXT_WIDTH_TWIPS // 2
        tabs = (
            f'<w:tabs><w:tab w:val="center" w:pos="{center}"/>'
            f'<w:tab w:val="right" w:pos="{TEXT_WIDTH_TWIPS}"/></w:tabs>'
        )
        number_runs = ""
        if number:
            number_runs = "<w:r><w:tab/></w:r>" + self._run(f"({number})")
        self.body.append(
            f'<w:p><w:pPr><w:pStyle w:val="Equation"/>{tabs}</w:pPr>'
            f"<w:r><w:tab/></w:r><m:oMath>{body}</m:oMath>{number_runs}</w:p>"
        )

    def where(self, text: str) -> None:
        """Dong 'trong do ...' giai thich ky hieu ngay duoi cong thuc."""
        self.body.append(
            '<w:p><w:pPr><w:ind w:left="284" w:right="0" w:firstLine="0"/>'
            '<w:spacing w:after="80"/></w:pPr>'
            + self._run(text, italic=False)
            + "</w:p>"
        )

    # ---------- fields ----------
    def _field(self, instruction: str, placeholder: str) -> str:
        # KHONG dat w:dirty="true": co nay bat Word tinh lai field ngay trong
        # luc dan trang, voi field PAGE o footer se gay vong lap repagination
        # lam ExportAsFixedFormat (xuat PDF) treo vo han. settings.xml da co
        # <w:updateFields w:val="true"/> de Word tu cap nhat field khi mo file.
        return (
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> {escape(instruction)} </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:t>{escape(placeholder)}</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        )

    def toc(self, *, levels="1-4") -> None:
        instruction = f'TOC \\o "{levels}" \\h \\z \\u'
        self.body.append(
            "<w:p>"
            + self._field(instruction, "Nhan Ctrl+A roi F9 de cap nhat muc luc.")
            + "</w:p>"
        )

    def table_of_captions(self, seq_name: str) -> None:
        instruction = f'TOC \\h \\z \\c "{seq_name}"'
        self.body.append(
            "<w:p>"
            + self._field(instruction, f"Nhan F9 de cap nhat danh muc {seq_name}.")
            + "</w:p>"
        )

    LABELS = {"Bang": "B\u1ea3ng", "Hinh": "H\u00ecnh"}

    CHAPTER_RE = re.compile(r"^CHƯƠNG\s+(\d+)")
    APPENDIX_RE = re.compile(r"^Phụ lục\s+([A-Z])\b")

    def _track_caption_chapter(self, level: int, text: str) -> None:
        """Suy ra tien to so chuong (hoac ky hieu phu luc) tu tieu de vua ghi.

        Heading 1 dang "CHUONG 3: ..." -> tien to "3"; heading 2 dang
        "Phu luc B. ..." -> tien to "B"; cac heading 1 cua front matter
        (TOM TAT, MUC LUC, PHAN CONG CONG VIEC...) -> khong co tien to nen
        caption o phan nay danh so phang.
        """
        stripped = text.strip()
        if level == 1:
            match = self.CHAPTER_RE.match(stripped)
            self.set_caption_chapter(match.group(1) if match else None)
        elif level == 2:
            match = self.APPENDIX_RE.match(stripped)
            if match:
                self.set_caption_chapter(match.group(1))

    def set_caption_chapter(self, prefix: str | None) -> None:
        """Doi tien to chuong cho caption; SEQ se reset ve 1 o caption ke tiep."""
        if prefix == self._caption_prefix:
            return
        self._caption_prefix = prefix
        self._caption_reset = {"Bang", "Hinh"}
        self._caption_counter = {}

    def caption(self, seq_name: str, text: str, *, keep_next: bool = False) -> None:
        """Caption dung field SEQ nen Word tu danh so va sinh danh muc.

        So thu tu ke thua so chuong theo quy dinh CT239H: caption in ra
        "Bang 2.1" bang cach ghep tien to chuong (van ban thuong) voi field SEQ
        duoc reset o dau moi chuong. Doan van van chua field SEQ nen truong
        TOC \\c "Bang" va TOC \\c "Hinh" tiep tuc sinh duoc danh muc.

        Kieu LaTeX: nhan va so ("Bang 2.1.") in dam, phan mo ta chu thuong.
        ``keep_next`` dung cho caption DAT TREN bang de caption khong bi tach
        khoi bang khi sang trang.
        """
        label = self.LABELS.get(seq_name, seq_name)
        instruction = f"SEQ {seq_name} \\* ARABIC"
        if seq_name in self._caption_reset:
            instruction += " \\r 1"
            self._caption_reset.discard(seq_name)
            self._caption_counter[seq_name] = 1
        else:
            self._caption_counter[seq_name] = self._caption_counter.get(seq_name, 0) + 1
        number = str(self._caption_counter[seq_name])
        prefix = f"{self._caption_prefix}." if self._caption_prefix else ""
        keep = "<w:keepNext/>" if keep_next else ""
        bold_field = (
            '<w:r><w:rPr><w:b/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:rPr><w:b/></w:rPr><w:instrText xml:space="preserve"> {escape(instruction)} </w:instrText></w:r>'
            '<w:r><w:rPr><w:b/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:rPr><w:b/></w:rPr><w:t>{escape(number)}</w:t></w:r>'
            '<w:r><w:rPr><w:b/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>'
        )
        self.body.append(
            f'<w:p><w:pPr><w:pStyle w:val="Caption"/>{keep}<w:jc w:val="center"/></w:pPr>'
            + self._run(f"{label} {prefix}", bold=True)
            + bold_field
            + self._run(".", bold=True)
            + self._run(f" {text}")
            + "</w:p>"
        )

    # ---------- table ----------
    def table(
        self,
        header: list[str],
        rows: list[list[str]],
        *,
        widths: list[int] | None = None,
        caption: str | None = None,
    ) -> None:
        """Bang kieu booktabs cua LaTeX: chi co ba duong ke ngang (toprule dam,
        midrule manh duoi hang tieu de, bottomrule dam), khong ke doc, khong to
        nen. Caption dat TREN bang theo dung quy uoc bang cua LaTeX.
        """
        if caption:
            self.caption("Bang", caption, keep_next=True)
        column_count = len(header)
        widths = widths or [round(9070 / column_count)] * column_count
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        xml = [
            "<w:tbl><w:tblPr>"
            '<w:tblW w:w="5000" w:type="pct"/>'
            "<w:tblBorders>"
            '<w:top w:val="single" w:sz="12" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="12" w:color="000000"/>'
            "</w:tblBorders>"
            '<w:tblCellMar>'
            '<w:top w:w="50" w:type="dxa"/><w:bottom w:w="50" w:type="dxa"/>'
            '<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
            "</w:tblCellMar>"
            '<w:tblLayout w:type="fixed"/>'
            "</w:tblPr>"
            f"<w:tblGrid>{grid}</w:tblGrid>"
        ]
        xml.append(self._table_row(header, widths, header=True))
        for row in rows:
            xml.append(self._table_row([str(cell) for cell in row], widths))
        xml.append("</w:tbl>")
        self.body.append("".join(xml))
        self.paragraph("", spacing_after=0)

    def _table_row(self, cells: list[str], widths: list[int], *, header=False) -> str:
        parts = ["<w:tr>"]
        if header:
            parts.insert(1, "<w:trPr><w:tblHeader/></w:trPr>")
        for index, cell in enumerate(cells):
            # Midrule cua booktabs: duong ke manh duy nhat nam duoi hang tieu de.
            borders = (
                '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:color="000000"/></w:tcBorders>'
                if header
                else ""
            )
            parts.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[index]}" w:type="dxa"/>{borders}'
                "</w:tcPr>"
                '<w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>'
                f"{self._run(cell, bold=header)}</w:p></w:tc>"
            )
        parts.append("</w:tr>")
        return "".join(parts)

    # ---------- image ----------
    def image(
        self,
        path: Path,
        caption: str | None = None,
        *,
        width_cm: float | None = None,
        page_center: bool = False,
    ) -> None:
        path = Path(path)
        key = str(path.resolve())
        if key not in self._image_ids:
            rel_id = f"rIdImg{len(self._image_ids) + 1}"
            self._image_ids[key] = rel_id
            self.images.append((rel_id, path))
        rel_id = self._image_ids[key]

        pixel_width, pixel_height = png_size(path)
        target_cm = width_cm or min(MAX_IMAGE_WIDTH_CM, pixel_width / 96 * 2.54)
        target_cm = min(target_cm, MAX_IMAGE_WIDTH_CM)
        width_emu = int(target_cm * EMU_PER_CM)
        height_emu = int(width_emu * pixel_height / pixel_width)

        self._drawing_id += 1
        doc_pr_id = self._drawing_id
        self.body.append(
            '<w:p><w:pPr><w:jc w:val="center"/>'
            + ('<w:ind w:left="-481" w:right="481"/>' if page_center else "")
            + '<w:spacing w:before="120" w:after="60"/></w:pPr>'
            "<w:r><w:drawing>"
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{doc_pr_id}" name="Picture {doc_pr_id}"/>'
            "<a:graphic><a:graphicData "
            'uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            "<pic:pic><pic:nvPicPr>"
            f'<pic:cNvPr id="{doc_pr_id}" name="image{doc_pr_id}.png"/><pic:cNvPicPr/>'
            "</pic:nvPicPr>"
            f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/>'
            "</a:stretch></pic:blipFill>"
            "<pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/>"
            f'<a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            "</pic:pic></a:graphicData></a:graphic></wp:inline>"
            "</w:drawing></w:r></w:p>"
        )
        if caption:
            self.caption("Hinh", caption)

    # ---------- package ----------
    def _document_xml(self) -> str:
        # Le theo dung template CT239H: trai 3cm, tren/phai/duoi 2cm, kho A4.
        sect_pr = (
            "<w:sectPr>"
            '<w:headerReference w:type="default" r:id="rIdHeader"/>'
            '<w:footerReference w:type="default" r:id="rIdFooter"/>'
            '<w:titlePg/>'
            '<w:pgSz w:w="11907" w:h="16840"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1701" '
            'w:header="720" w:footer="576" w:gutter="0"/>'
            '<w:pgNumType w:start="1"/><w:cols w:space="720"/>'
            "</w:sectPr>"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f"<w:document {NS}><w:body>{''.join(self.body)}{sect_pr}</w:body></w:document>"
        )

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        media = []
        rels = [
            '<Relationship Id="rIdStyles" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>',
            '<Relationship Id="rIdSettings" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" '
            'Target="settings.xml"/>',
            '<Relationship Id="rIdHeader" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" '
            'Target="header1.xml"/>',
            '<Relationship Id="rIdFooter" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            'Target="footer1.xml"/>',
        ]
        for index, (rel_id, path) in enumerate(self.images, start=1):
            name = f"media/image{index}.png"
            media.append((f"word/{name}", path))
            rels.append(
                f'<Relationship Id="{rel_id}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="{name}"/>'
            )

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", CONTENT_TYPES)
            zf.writestr("_rels/.rels", ROOT_RELS)
            zf.writestr("docProps/core.xml", CORE_XML)
            zf.writestr("docProps/app.xml", APP_XML)
            zf.writestr("word/document.xml", self._document_xml())
            zf.writestr("word/styles.xml", STYLES_XML)
            zf.writestr("word/settings.xml", SETTINGS_XML)
            zf.writestr("word/header1.xml", self._header_xml())
            zf.writestr("word/footer1.xml", self._footer_xml())
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(rels)
                + "</Relationships>",
            )
            for arc_name, source in media:
                zf.writestr(arc_name, source.read_bytes())
        return output_path

    def _footer_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f"<w:ftr {NS}>"
            '<w:p><w:pPr><w:jc w:val="center"/>'
            '<w:ind w:left="0" w:right="0" w:firstLine="0"/></w:pPr>'
            + self._field("PAGE", "1")
            + "</w:p></w:ftr>"
        )

    def _header_xml(self) -> str:
        text = self.header_topic
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f"<w:hdr {NS}>"
            '<w:p><w:pPr><w:jc w:val="right"/><w:ind w:left="0" w:right="0" w:firstLine="0"/>'
            '<w:spacing w:after="0"/>'
            '<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="808080"/></w:pBdr>'
            "</w:pPr>"
            f'<w:r><w:rPr><w:i/><w:sz w:val="22"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
            "</w:p></w:hdr>"
        )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>Bao cao nien luan - Du bao xu huong co phieu HOSE</dc:title>
<dc:creator>Sinh vien thuc hien</dc:creator>
<cp:revision>1</cp:revision>
</cp:coreProperties>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
<Application>Microsoft Office Word</Application>
</Properties>"""

SETTINGS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:zoom w:percent="100"/>
<w:updateFields w:val="true"/>
<w:defaultTabStop w:val="720"/>
</w:settings>"""

STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
<w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/><w:lang w:val="vi-VN"/>
</w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/>
<w:jc w:val="both"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
<w:name w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/>
<w:ind w:left="0" w:right="0" w:firstLine="567"/><w:jc w:val="both"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
<w:color w:val="000000"/><w:sz w:val="26"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading1">
<w:name w:val="heading 1"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120" w:line="271" w:lineRule="auto"/>
<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Arial" w:hAnsi="Arial" w:cs="Arial"/>
<w:b/><w:caps/><w:color w:val="000000"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading2">
<w:name w:val="heading 2"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="200" w:after="60" w:line="271" w:lineRule="auto"/>
<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Arial" w:hAnsi="Arial" w:cs="Arial"/>
<w:b/><w:caps/><w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading3">
<w:name w:val="heading 3"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="160" w:after="60" w:line="271" w:lineRule="auto"/>
<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="left"/><w:outlineLvl w:val="2"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Arial" w:hAnsi="Arial" w:cs="Arial"/>
<w:b/><w:i/><w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading4">
<w:name w:val="heading 4"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="140" w:after="60" w:line="271" w:lineRule="auto"/>
<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="left"/><w:outlineLvl w:val="3"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Arial" w:hAnsi="Arial" w:cs="Arial"/>
<w:i/><w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TOC1">
<w:name w:val="toc 1"/><w:uiPriority w:val="39"/>
<w:pPr><w:spacing w:after="95" w:line="240" w:lineRule="auto"/>
<w:ind w:left="25" w:right="129" w:hanging="10"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TOC2">
<w:name w:val="toc 2"/><w:uiPriority w:val="39"/>
<w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/>
<w:ind w:left="240" w:right="129" w:firstLine="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TOC3">
<w:name w:val="toc 3"/><w:uiPriority w:val="39"/>
<w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/>
<w:ind w:left="480" w:right="129" w:firstLine="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TOC4">
<w:name w:val="toc 4"/><w:uiPriority w:val="39"/>
<w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/>
<w:ind w:left="720" w:right="129" w:firstLine="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:i/><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Caption">
<w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:jc w:val="center"/><w:ind w:left="0" w:right="0" w:firstLine="0"/>
<w:spacing w:before="60" w:after="180" w:line="240" w:lineRule="auto"/></w:pPr>
<w:rPr><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Equation">
<w:name w:val="Equation"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:pPr><w:jc w:val="left"/><w:ind w:left="0" w:right="0" w:firstLine="0"/>
<w:spacing w:before="80" w:after="120" w:line="276" w:lineRule="auto"/></w:pPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TableText">
<w:name w:val="Table Text"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="left"/><w:ind w:left="0" w:right="0" w:firstLine="0"/>
<w:spacing w:before="40" w:after="40" w:line="240" w:lineRule="auto"/></w:pPr>
<w:rPr><w:sz w:val="22"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CodeBlock">
<w:name w:val="Code Block"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="left"/><w:ind w:left="284" w:right="0" w:firstLine="0"/>
<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>
<w:pBdr><w:left w:val="single" w:sz="12" w:space="8" w:color="BFBFBF"/></w:pBdr>
<w:shd w:val="clear" w:color="auto" w:fill="F7F7F7"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CoverTitle">
<w:name w:val="Cover Title"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="center"/><w:ind w:left="-481" w:right="481" w:firstLine="0"/>
<w:spacing w:before="0" w:after="120" w:line="259" w:lineRule="auto"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="40"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CoverText">
<w:name w:val="Cover Text"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="center"/><w:ind w:left="-481" w:right="481" w:firstLine="0"/>
<w:spacing w:before="0" w:after="0" w:line="259" w:lineRule="auto"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CoverHeading">
<w:name w:val="Cover Heading"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="center"/><w:ind w:left="-481" w:right="481" w:firstLine="0"/>
<w:spacing w:before="0" w:after="60" w:line="259" w:lineRule="auto"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CoverSub">
<w:name w:val="Cover Sub"/><w:basedOn w:val="Normal"/>
<w:pPr><w:jc w:val="center"/><w:ind w:left="-481" w:right="481" w:firstLine="0"/>
<w:spacing w:before="0" w:after="40" w:line="269" w:lineRule="auto"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="30"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph">
<w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="34"/><w:qFormat/>
<w:pPr><w:ind w:left="720" w:right="0" w:firstLine="0"/><w:contextualSpacing/></w:pPr>
</w:style>
</w:styles>"""
