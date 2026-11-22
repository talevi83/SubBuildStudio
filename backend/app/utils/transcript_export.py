"""Markdown (.md) and Word (.docx) transcript exporters with speaker grouping and RTL support."""
from dataclasses import dataclass
import io
import re
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape
import zipfile

from app.utils.rtl_fixer import contains_rtl, strip_directional_marks
from app.utils.srt_formatter import _get_seg_attr

_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _clean_text(text: Optional[str]) -> str:
    clean = strip_directional_marks(text or "").strip()
    return _ILLEGAL_XML_RE.sub("", clean)


def _format_clock(seconds: float) -> str:
    total = int(round(max(0.0, float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _speaker_label(speaker_id: Optional[str], speakers: Optional[Dict[str, Dict[str, Any]]]) -> Optional[str]:
    if not speaker_id:
        return None
    if isinstance(speakers, dict):
        info = speakers.get(speaker_id)
        if isinstance(info, dict) and str(info.get("label") or "").strip():
            return _ILLEGAL_XML_RE.sub("", str(info["label"]).strip())
    return str(speaker_id)


@dataclass
class _TranscriptBlock:
    kind: str  # DIALOGUE | ON_SCREEN | COMMS | FOREIGN
    start: float
    end: float
    speaker_id: Optional[str]
    speaker_label: Optional[str]
    primary_lines: List[str]
    secondary_lines: List[str]


def _group_blocks(
    segments: List[Any],
    *,
    use_translated: bool,
    bilingual: bool,
    speakers: Optional[Dict[str, Dict[str, Any]]],
    max_turn_gap: float = 6.0,
) -> List[_TranscriptBlock]:
    sorted_segs = sorted(
        segments or [],
        key=lambda s: (float(_get_seg_attr(s, "start_time", 0.0)), int(_get_seg_attr(s, "index", 0) or 0)),
    )
    blocks: List[_TranscriptBlock] = []
    for seg in sorted_segs:
        orig = _clean_text(_get_seg_attr(seg, "original_text", ""))
        trans = _clean_text(_get_seg_attr(seg, "translated_text", ""))
        if bilingual:
            primary = (trans or orig).replace("\n", " ")
            secondary = orig.replace("\n", " ") if (trans and orig and trans != orig) else ""
        else:
            primary = ((trans if use_translated else orig) or orig or trans).replace("\n", " ")
            secondary = ""
        if not primary:
            continue

        seg_type = str(_get_seg_attr(seg, "segment_type", "DIALOGUE") or "DIALOGUE")
        speaker_id = _get_seg_attr(seg, "speaker", None)
        label = _speaker_label(speaker_id, speakers)
        start = float(_get_seg_attr(seg, "start_time", 0.0))
        end = float(_get_seg_attr(seg, "end_time", start))

        can_merge = (
            bool(blocks)
            and seg_type == "DIALOGUE"
            and blocks[-1].kind == "DIALOGUE"
            and speaker_id is not None
            and blocks[-1].speaker_id == speaker_id
            and (start - blocks[-1].end) <= max_turn_gap
        )
        if can_merge:
            blocks[-1].end = max(blocks[-1].end, end)
            blocks[-1].primary_lines.append(primary)
            if secondary:
                blocks[-1].secondary_lines.append(secondary)
        else:
            blocks.append(
                _TranscriptBlock(
                    kind=seg_type,
                    start=start,
                    end=end,
                    speaker_id=speaker_id,
                    speaker_label=label,
                    primary_lines=[primary],
                    secondary_lines=[secondary] if secondary else [],
                )
            )
    return blocks


def serialize_to_markdown(
    segments: List[Any],
    *,
    title: str = "Transcript",
    language: Optional[str] = None,
    duration: Optional[float] = None,
    use_translated: bool = False,
    bilingual: bool = False,
    speakers: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Serialize subtitle segments to a readable, speaker-grouped Markdown transcript."""
    clean_title = _clean_text(title) or "Transcript"
    blocks = _group_blocks(
        segments,
        use_translated=use_translated,
        bilingual=bilingual,
        speakers=speakers,
    )
    meta_parts: List[str] = []
    if language:
        meta_parts.append(f"**Language:** {_clean_text(language)}")
    if duration and duration > 0:
        meta_parts.append(f"**Duration:** {_format_clock(duration)}")
    meta_parts.append(f"**Segments:** {len(segments or [])}")

    out: List[str] = [f"# {clean_title}", "", " · ".join(meta_parts), ""]
    for b in blocks:
        ts = f"[{_format_clock(b.start)}]"
        body = " ".join(b.primary_lines)
        if b.kind == "ON_SCREEN":
            onscreen_text = body if (body.startswith("[") and body.endswith("]")) else f"[{body}]"
            out.append(f"> {ts} **[ON_SCREEN]** {onscreen_text}")
            if b.secondary_lines:
                out.append(f"> *{' '.join(b.secondary_lines)}*")
            out.append("")
            continue

        header_bits = [ts]
        if b.speaker_label:
            header_bits.append(f"**{b.speaker_label}**")
        elif b.kind in ("COMMS", "FOREIGN"):
            header_bits.append(f"**[{b.kind}]**")
        out.append(" ".join(header_bits) + "  ")
        out.append(body)
        if b.secondary_lines:
            out.append("")
            out.append(f"*{' '.join(b.secondary_lines)}*")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="36"/>
      <w:szCs w:val="36"/>
    </w:rPr>
  </w:style>
</w:styles>"""


def _docx_run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: Optional[str] = None,
    size_half_pt: Optional[int] = None,
    rtl: bool = False,
) -> str:
    rpr: List[str] = []
    if bold:
        rpr.append("<w:b/><w:bCs/>")
    if italic:
        rpr.append("<w:i/><w:iCs/>")
    if color:
        rpr.append(f'<w:color w:val="{xml_escape(color)}"/>')
    if size_half_pt:
        rpr.append(f'<w:sz w:val="{size_half_pt}"/><w:szCs w:val="{size_half_pt}"/>')
    if rtl:
        rpr.append("<w:rtl/>")
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    safe = xml_escape(_ILLEGAL_XML_RE.sub("", text or ""))
    return f'<w:r>{rpr_xml}<w:t xml:space="preserve">{safe}</w:t></w:r>'


def _docx_paragraph(runs: List[str], *, rtl: bool = False, style: Optional[str] = None, space_after: int = 120) -> str:
    ppr: List[str] = []
    if style:
        ppr.append(f'<w:pStyle w:val="{xml_escape(style)}"/>')
    if rtl:
        ppr.append("<w:bidi/>")
    ppr.append(f'<w:spacing w:after="{space_after}"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>"
    return f"<w:p>{ppr_xml}{''.join(runs)}</w:p>"


def serialize_to_docx(
    segments: List[Any],
    *,
    title: str = "Transcript",
    language: Optional[str] = None,
    duration: Optional[float] = None,
    use_translated: bool = False,
    bilingual: bool = False,
    rtl: bool = False,
    speakers: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bytes:
    """Build a zero-dependency OOXML .docx transcript with full RTL BiDi support."""
    clean_title = _clean_text(title) or "Transcript"
    blocks = _group_blocks(
        segments,
        use_translated=use_translated,
        bilingual=bilingual,
        speakers=speakers,
    )

    paragraphs: List[str] = [
        _docx_paragraph(
            [_docx_run(clean_title, bold=True, size_half_pt=34, rtl=contains_rtl(clean_title))],
            rtl=contains_rtl(clean_title),
            style="Title",
            space_after=160,
        )
    ]

    meta_parts: List[str] = []
    if language:
        meta_parts.append(f"Language: {_clean_text(language)}")
    if duration and duration > 0:
        meta_parts.append(f"Duration: {_format_clock(duration)}")
    meta_parts.append(f"Segments: {len(segments or [])}")
    paragraphs.append(
        _docx_paragraph(
            [_docx_run(" · ".join(meta_parts), color="666666", size_half_pt=19)],
            space_after=200,
        )
    )

    for b in blocks:
        ts = f"[{_format_clock(b.start)}] "
        body = " ".join(b.primary_lines)
        is_rtl_para = bool(rtl or contains_rtl(body) or contains_rtl(b.speaker_label or ""))
        runs: List[str] = [_docx_run(ts, color="6B7280", size_half_pt=18)]
        if b.kind == "ON_SCREEN":
            runs.append(_docx_run("[ON_SCREEN] ", bold=True, color="0891B2", size_half_pt=20))
        elif b.speaker_label:
            runs.append(_docx_run(f"{b.speaker_label}: ", bold=True, color="4F46E5", rtl=contains_rtl(b.speaker_label)))
        elif b.kind in ("COMMS", "FOREIGN"):
            runs.append(_docx_run(f"[{b.kind}] ", bold=True, color="059669", size_half_pt=20))
        runs.append(_docx_run(body, rtl=is_rtl_para))
        paragraphs.append(_docx_paragraph(runs, rtl=is_rtl_para, space_after=80 if b.secondary_lines else 140))

        if b.secondary_lines:
            sec = " ".join(b.secondary_lines)
            sec_rtl = contains_rtl(sec)
            paragraphs.append(
                _docx_paragraph(
                    [_docx_run(sec, italic=True, color="52525B", size_half_pt=20, rtl=sec_rtl)],
                    rtl=sec_rtl,
                    space_after=140,
                )
            )

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}<w:sectPr/></w:body>"
        "</w:document>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", _RELS_XML)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS_XML)
        zf.writestr("word/styles.xml", _STYLES_XML)
        zf.writestr("word/document.xml", doc_xml)
    return buf.getvalue()
