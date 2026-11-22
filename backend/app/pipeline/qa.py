"""Deterministic post-translation quality checks (no model calls).

Flags defects that are cheap to detect and easy to miss by eye: untranslated Latin
words in a Hebrew line, glossary names/terms rendered differently from the saved
glossary, and 2nd-person pronoun/verb gender mismatches.
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

_HEBREW = re.compile(r"[֐-׿]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Short Latin tokens that are normal inside Hebrew subtitles
_LATIN_ALLOWED = {"ok", "okay", "tv", "dna", "cia", "fbi", "usa", "uss", "uav", "gps", "vip", "sos", "eta"}

# Hebrew "at" (you, f.) with a masculine future form, and "atah" (you, m.) with a feminine one
_MASC_FUT = "תהיה תעשה תגיד תלך תבוא תראה תדע תקבל תיקח תרצה תוכל תצטרך תישאר תשאר תחכה תעזור תשמע תבין".split()
_FEM_FUT = ("תהיי תעשי תגידי תלכי תבואי תראי תדעי תקבלי תיקחי תרצי תוכלי "
            "תצטרכי תישארי תשארי תחכי תעזרי תשמעי תביני").split()
_NB = r"(?<![֐-׿])"
_NA = r"(?![֐-׿])"
_FEM_PRON_MASC_VERB = re.compile(_NB + "את" + r"\s+(" + "|".join(_MASC_FUT) + ")" + _NA)
_MASC_PRON_FEM_VERB = re.compile(_NB + "אתה" + r"\s+(" + "|".join(_FEM_FUT) + ")" + _NA)


@dataclass
class QAIssue:
    segment_id: int
    index: int
    kind: str      # latin | name | term | agreement | missing | overlap | timing
    message: str


def _contains_word(text: str, word: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(word.lower()) + r"(?![\w])", text.lower()) is not None


def check_segment(
    seg_id: int,
    index: int,
    original: str,
    translated: Optional[str],
    segment_type: str,
    failed: bool,
    glossary: Optional[Dict],
    translation_enabled: bool = True,
) -> List[QAIssue]:
    issues: List[QAIssue] = []
    if segment_type == "ON_SCREEN" and not translated:
        return issues

    has_translated = bool((translated or "").strip())
    if failed or not has_translated:
        if translation_enabled:
            if (original or "").strip():
                issues.append(QAIssue(seg_id, index, "missing", "No translation"))
            return issues
        text = translated if has_translated else (original or "")
    else:
        text = translated or ""

    if _HEBREW.search(text):
        if has_translated:
            for m in _LATIN_WORD.finditer(text):
                w = m.group(0)
                if len(w) < 2 or w.isupper() or w.lower() in _LATIN_ALLOWED:
                    continue
                issues.append(QAIssue(seg_id, index, "latin", f"Untranslated word: {w}"))
                break

        for pat, label in ((_FEM_PRON_MASC_VERB, "feminine pronoun + masculine verb"),
                           (_MASC_PRON_FEM_VERB, "masculine pronoun + feminine verb")):
            m = pat.search(text)
            if m:
                issues.append(QAIssue(seg_id, index, "agreement", f"Pronoun/verb mismatch ({label}): {m.group(0)}"))

    if glossary and has_translated:
        for c in glossary.get("characters", []) or []:
            src, tgt = (c.get("name") or "").strip(), (c.get("target_name") or "").strip()
            if src and tgt and _contains_word(original or "", src) and tgt not in text:
                issues.append(QAIssue(seg_id, index, "name", f"{src} should be {tgt}"))
        for t in glossary.get("terms", []) or []:
            src, tgt = (t.get("source") or "").strip(), (t.get("translation") or "").strip()
            if src and tgt and _contains_word(original or "", src) and tgt not in text:
                issues.append(QAIssue(seg_id, index, "term", f"{src} should be {tgt}"))
    return issues


def _has_timing(r) -> bool:
    return (
        getattr(r, "start_time", None) is not None
        and getattr(r, "end_time", None) is not None
    )


def _overlap_issues(rows: List) -> Dict[int, QAIssue]:
    overlaps: Dict[int, QAIssue] = {}
    lanes = (
        [r for r in rows if _has_timing(r) and getattr(r, "segment_type", None) != "ON_SCREEN"],
        [r for r in rows if _has_timing(r) and getattr(r, "segment_type", None) == "ON_SCREEN"],
    )
    for lane in lanes:
        lane.sort(key=lambda r: (float(r.start_time), int(getattr(r, "index", 0))))
        prev = None
        for cur in lane:
            if prev is not None and float(cur.start_time) < float(prev.end_time) - 0.001:
                overlap_sec = float(prev.end_time) - float(cur.start_time)
                overlaps[cur.id] = QAIssue(
                    cur.id,
                    cur.index,
                    "overlap",
                    f"Overlaps with line #{prev.index} ({overlap_sec:.2f}s)",
                )
            if prev is None or float(cur.end_time) > float(prev.end_time):
                prev = cur
    return overlaps


def check_rows(rows: Iterable, glossary: Optional[Dict], translation_enabled: bool = True) -> List[QAIssue]:
    row_list = list(rows)
    overlap_by_id = _overlap_issues(row_list)
    out: List[QAIssue] = []
    for r in row_list:
        out.extend(check_segment(r.id, r.index, r.original_text, r.translated_text, r.segment_type,
                                 bool(r.translation_failed), glossary, translation_enabled=translation_enabled))
        if r.id in overlap_by_id:
            out.append(overlap_by_id[r.id])
        if _has_timing(r):
            dur = float(r.end_time) - float(r.start_time)
            if dur < 0.5:
                out.append(QAIssue(r.id, r.index, "timing", f"Very short duration ({dur:.2f}s < 0.5s)"))
            elif dur > 8.0:
                out.append(QAIssue(r.id, r.index, "timing", f"Very long duration ({dur:.1f}s > 8.0s)"))
    return out
