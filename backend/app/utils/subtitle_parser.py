"""
Read a subtitle file the user already has (SRT, WebVTT or ASS/SSA) for "sync existing subtitles".

* Decoding: BOM (UTF-8 / UTF-16), else strict UTF-8, else the best-scoring legacy code page
  (Windows-1255 Hebrew, Windows-1256 Arabic, Windows-1251 Cyrillic), else Windows-1252.
  The setup screen can re-decode the same bytes with an encoding the user picks.
* Cues come back in file order with formatting removed but line breaks kept. A cue positioned at
  the top of the picture ({\\an8}, ASS alignment 7-9, VTT "line:0") is usually a sign or title, so
  it is marked with position="top" - a hint for matching on-screen text later.
* The language is guessed from the script (Hebrew/Arabic/Cyrillic/Greek letters, else English).
"""
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


class SubtitleParseError(ValueError):
    pass


@dataclass
class ParsedCue:
    start: float
    end: float
    text: str
    position: str = "bottom"     # "top" when the file places the cue at the top of the picture
    kind: str = "speech"         # "sound" for hearing-impaired descriptions such as [door slams] or ♪ ...

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- decoding

_HEBREW = re.compile(r"[\u05d0-\u05ea]")
_HEBREW_NIKUD = re.compile(r"[\u0591-\u05c7]")
# Points written on a letter belong to the word; points with no letter before them are what
# Cyrillic capitals (Windows-1251) look like when read as Windows-1255.
_HEBREW_ATTACHED_NIKUD = re.compile(r"(?<=[\u05d0-\u05ea])[\u0591-\u05c7]+")
_HEBREW_PURE_WORDS = re.compile(r"(?<![A-Za-z\u00c0-\u00ff])[\u05d0-\u05ea]{2,}(?![A-Za-z\u00c0-\u00ff])")
_MIXED_LATIN_HEBREW = re.compile(r"[A-Za-z\u00c0-\u00ff][\u05d0-\u05ea]|[\u05d0-\u05ea][A-Za-z\u00c0-\u00ff]")

_ARABIC = re.compile(r"[\u0621-\u063a\u0641-\u064a]")
_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f]")
_ARABIC_PURE_WORDS = re.compile(r"(?<![A-Za-z\u00c0-\u00ff])[\u0621-\u063a\u0641-\u064a]{2,}(?![A-Za-z\u00c0-\u00ff])")
_MIXED_LATIN_ARABIC = re.compile(r"[A-Za-z\u00c0-\u00ff][\u0600-\u06ff]|[\u0600-\u06ff][A-Za-z\u00c0-\u00ff]")

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_CYRILLIC_PURE_WORDS = re.compile(r"(?<![A-Za-z\u00c0-\u00ff])[\u0400-\u04ff]{2,}(?![A-Za-z\u00c0-\u00ff])")
_MIXED_LATIN_CYRILLIC = re.compile(r"[A-Za-z\u00c0-\u00ff][\u0400-\u04ff]|[\u0400-\u04ff][A-Za-z\u00c0-\u00ff]")
# Lowercase followed by uppercase inside a word ("гСНИЗ") never happens in real Cyrillic text,
# but it is what Arabic (Windows-1256) bytes look like when read as Windows-1251.
_CYRILLIC_CASE_FLIP = re.compile(r"[\u0430-\u044f\u0450-\u045f][\u0410-\u042f\u0400-\u040f]")
_GREEK = re.compile(r"[Ͱ-Ͽ]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u00ff]")

_SUPPORTED_ENCODINGS: Dict[str, Tuple[str, str]] = {
    "utf-8": ("utf-8-sig", "utf-8"),
    "utf8": ("utf-8-sig", "utf-8"),
    "utf-8-sig": ("utf-8-sig", "utf-8-sig"),
    "utf-16": ("utf-16", "utf-16"),
    "utf-16-le": ("utf-16-le", "utf-16-le"),
    "utf-16-be": ("utf-16-be", "utf-16-be"),
    "windows-1255": ("cp1255", "windows-1255"),
    "cp1255": ("cp1255", "windows-1255"),
    "iso-8859-8": ("iso-8859-8", "iso-8859-8"),
    "windows-1256": ("cp1256", "windows-1256"),
    "cp1256": ("cp1256", "windows-1256"),
    "windows-1252": ("cp1252", "windows-1252"),
    "cp1252": ("cp1252", "windows-1252"),
    "latin-1": ("cp1252", "windows-1252"),
    "iso-8859-1": ("cp1252", "windows-1252"),
    "windows-1251": ("cp1251", "windows-1251"),
    "cp1251": ("cp1251", "windows-1251"),
    "windows-1250": ("cp1250", "windows-1250"),
    "cp1250": ("cp1250", "windows-1250"),
    "windows-1253": ("cp1253", "windows-1253"),
    "cp1253": ("cp1253", "windows-1253"),
    "windows-1254": ("cp1254", "windows-1254"),
    "cp1254": ("cp1254", "windows-1254"),
}


def decode_subtitle_bytes(data: bytes, encoding: Optional[str] = None) -> Tuple[str, str]:
    """Bytes -> (text, encoding name). Honors an explicit encoding when given."""
    if encoding and encoding.strip().lower() not in ("", "auto"):
        key = encoding.strip().lower()
        if key not in _SUPPORTED_ENCODINGS:
            raise SubtitleParseError(f"Unsupported subtitle encoding: {encoding}")
        codec, canonical = _SUPPORTED_ENCODINGS[key]
        if canonical == "utf-8" and data.startswith(b"\xef\xbb\xbf"):
            return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
        return data.decode(codec, errors="replace"), canonical

    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace"), "utf-16"
    # UTF-16 without a BOM: every other byte of ASCII text is zero
    sample = data[:4000]
    if sample and sample.count(b"\x00") > len(sample) // 4:
        enc = "utf-16-le" if sample[1:2] == b"\x00" else "utf-16-be"
        return data.decode(enc, errors="replace"), enc
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    # Legacy single-byte code pages share the 0xC0-0xFF range, so the same bytes read as
    # plausible letters in several scripts. Score each candidate by how much of the text forms
    # clean words in its script, minus signs of a wrong guess (replacement characters, letters
    # glued to Latin, stray Hebrew points / Arabic marks, impossible letter case).
    # Ties go to the earlier candidate: Hebrew text read as Windows-1251 is valid-looking
    # lowercase Cyrillic, while Russian read as Windows-1255 breaks on capitals and on letters
    # Windows-1255 does not have, so Russian always scores strictly higher on Russian files.
    candidates = []

    hebrew = data.decode("cp1255", errors="replace")
    hebrew_bare = _HEBREW_ATTACHED_NIKUD.sub("", hebrew)
    candidates.append((
        hebrew,
        "windows-1255",
        sum(1 for ch in hebrew_bare if ch.isalpha()),
        sum(len(w) for w in _HEBREW_PURE_WORDS.findall(hebrew_bare)),
        3 * len(_MIXED_LATIN_HEBREW.findall(hebrew_bare))
        + 2 * len(_HEBREW_NIKUD.findall(hebrew_bare))
        + 4 * hebrew.count("\ufffd"),
    ))

    arabic = data.decode("cp1256", errors="replace")
    candidates.append((
        arabic,
        "windows-1256",
        sum(1 for ch in arabic if ch.isalpha()),
        sum(len(w) for w in _ARABIC_PURE_WORDS.findall(arabic)),
        3 * len(_MIXED_LATIN_ARABIC.findall(arabic))
        + 2 * len(_ARABIC_DIACRITICS.findall(arabic))
        + 4 * arabic.count("\ufffd"),
    ))

    cyrillic = data.decode("cp1251", errors="replace")
    candidates.append((
        cyrillic,
        "windows-1251",
        sum(1 for ch in cyrillic if ch.isalpha()),
        sum(len(w) for w in _CYRILLIC_PURE_WORDS.findall(cyrillic)),
        3 * len(_MIXED_LATIN_CYRILLIC.findall(cyrillic))
        + 3 * len(_CYRILLIC_CASE_FLIP.findall(cyrillic))
        + 4 * cyrillic.count("\ufffd"),
    ))

    best = None
    best_score = 0
    for text, name, letters, pure, penalty in candidates:
        score = pure - penalty
        if score > best_score and letters and pure / letters > 0.15:
            best, best_score = (text, name), score
    if best is not None:
        return best
    return data.decode("cp1252", errors="replace"), "windows-1252"


def detect_language(cues: List[ParsedCue]) -> str:
    """Language name from the dominant script of the cue text."""
    text = " ".join(c.text for c in cues)
    counts = {
        "Hebrew": len(_HEBREW.findall(text)),
        "Arabic": len(_ARABIC.findall(text)),
        "Russian": len(_CYRILLIC.findall(text)),
        "Greek": len(_GREEK.findall(text)),
        "English": len(_LATIN.findall(text)),
    }
    letters = sum(counts.values())
    if not letters:
        return "English"
    # A non-Latin script that makes up a real share of the letters wins (names stay in Latin)
    other = max((k for k in counts if k != "English"), key=counts.get)
    return other if counts[other] >= 0.3 * letters else "English"


# --------------------------------------------------------------------------- text cleanup

_HTML = re.compile(r"<[^>]+>")
_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")
_TOP_TAG = re.compile(r"\{\\an?[789]\}|\{[^}]*\\an[789][^}]*\}")
_SOUND = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\)|[♪♫#*]+.*)\s*$")
_SPACES = re.compile(r"[ \t]+")


def _clean(text: str) -> str:
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    text = _ASS_OVERRIDE.sub("", _HTML.sub("", text))
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _kind(text: str) -> str:
    """'sound' when every line is a bracketed description or music note (nothing is spoken)."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "sound" if lines and all(_SOUND.match(line) for line in lines) else "speech"


# --------------------------------------------------------------------------- formats

_TIME = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_CUE_TIME = re.compile(_TIME + r"\s*-->\s*" + _TIME + r"(.*)$")


def _seconds(h: Optional[str], m: str, s: str, frac: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000


def _parse_timed_blocks(content: str, vtt: bool) -> List[ParsedCue]:
    """SRT and WebVTT share the "start --> end" block structure."""
    cues: List[ParsedCue] = []
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = block.split("\n")
        for i, line in enumerate(lines):
            match = _CUE_TIME.search(line)
            if not match:
                continue
            g = match.groups()
            settings_part = g[8] or ""
            raw = "\n".join(lines[i + 1:])
            top = bool(_TOP_TAG.search(raw)) or (vtt and re.search(r"\bline:(0|[0-9](\.\d+)?%|1[0-9]%)", settings_part))
            text = _clean(raw)
            if text:
                start, end = _seconds(*g[0:4]), _seconds(*g[4:8])
                cues.append(ParsedCue(start, max(end, start + 0.1), text, "top" if top else "bottom", _kind(text)))
            break
    return cues


def _parse_ass(content: str) -> List[ParsedCue]:
    styles_top: Dict[str, bool] = {}
    fields: List[str] = []
    style_fields: List[str] = []
    cues: List[ParsedCue] = []
    section = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.lower()
            continue
        if line.lower().startswith("format:"):
            names = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            if "styles" in section:
                style_fields = names
            elif "events" in section:
                fields = names
            continue
        if line.lower().startswith("style:") and style_fields:
            values = [v.strip() for v in line.split(":", 1)[1].split(",")]
            row = dict(zip(style_fields, values))
            styles_top[row.get("name", "")] = row.get("alignment", "2") in ("7", "8", "9")
            continue
        if line.lower().startswith("dialogue:") and fields:
            values = line.split(":", 1)[1].split(",", len(fields) - 1)
            row = dict(zip(fields, [v.strip() if k < len(fields) - 1 else v for k, v in enumerate(values)]))
            try:
                start = _ass_time(row.get("start", ""))
                end = _ass_time(row.get("end", ""))
            except ValueError:
                continue
            raw = row.get("text", "")
            text = _clean(raw)
            if not text:
                continue
            top = bool(_TOP_TAG.search(raw)) or styles_top.get(row.get("style", ""), False)
            cues.append(ParsedCue(start, max(end, start + 0.1), text, "top" if top else "bottom", _kind(text)))
    return cues


def _ass_time(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[.,](\d{1,3})", value.strip())
    if not match:
        raise ValueError(value)
    h, m, s, cs = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs.ljust(2, "0")) / (10 ** len(cs))


def parse_subtitle_text(content: str, file_name: str = "") -> Tuple[List[ParsedCue], str]:
    """Text -> (cues sorted by start, format name). Raises SubtitleParseError if nothing is found."""
    content = content.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    name = file_name.lower()
    head = content.lstrip()[:200].lower()
    if name.endswith((".ass", ".ssa")) or "[script info]" in head or "[events]" in content.lower():
        cues, fmt = _parse_ass(content), "ass"
    elif name.endswith(".vtt") or head.startswith("webvtt"):
        cues, fmt = _parse_timed_blocks(content, vtt=True), "vtt"
    else:
        cues, fmt = _parse_timed_blocks(content, vtt=False), "srt"
    if not cues:
        raise SubtitleParseError("No subtitles found - supported formats are SRT, WebVTT and ASS/SSA.")
    cues.sort(key=lambda c: (c.start, c.end))
    return cues, fmt


def parse_subtitle_file(data: bytes, file_name: str = "", encoding: Optional[str] = None) -> Dict:
    """Bytes of an uploaded file -> everything the setup screen and the sync job need."""
    if len(data) > 20 * 1024 * 1024:
        raise SubtitleParseError("The subtitle file is too large (over 20 MB).")
    text, used_encoding = decode_subtitle_bytes(data, encoding=encoding)
    cues, fmt = parse_subtitle_text(text, file_name)
    return {
        "file_name": file_name,
        "format": fmt,
        "encoding": used_encoding,
        "language": detect_language(cues),
        "cue_count": len(cues),
        "first_start": cues[0].start,
        "last_end": max(c.end for c in cues),
        "top_count": sum(1 for c in cues if c.position == "top"),
        "sound_count": sum(1 for c in cues if c.kind == "sound"),
        "cues": [c.to_dict() for c in cues],
    }
