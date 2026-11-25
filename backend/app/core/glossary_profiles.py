"""
Series glossary profiles: characters (gender, name spelling) and terms that are saved once and
reused by every episode of a show, so each episode's glossary pass only has to add what is new.

Profiles are small JSON files under `<storage>/glossaries/`.
"""
import json
import re
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.private_files import write_text_atomic
from app.utils.filenames import defuse_reserved

_MAX_NAME = 80


def _dir():
    return settings.storage_dir / "glossaries"


def clean_name(name: Optional[str]) -> str:
    """A profile name as the user typed it, trimmed and without path characters."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", (name or "")).strip()[:_MAX_NAME].rstrip(" .")
    return defuse_reserved(cleaned)  # "NUL" would name the NUL device on Windows


def _path(name: str):
    return _dir() / f"{clean_name(name)}.json"


def list_profiles() -> List[str]:
    try:
        return sorted(p.stem for p in _dir().glob("*.json"))
    except OSError:
        return []


def load_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    name = clean_name(name)
    if not name:
        return None
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_profile(name: str, glossary: Dict[str, Any]) -> None:
    name = clean_name(name)
    if not name:
        raise ValueError("Profile name is empty")
    _dir().mkdir(parents=True, exist_ok=True)
    write_text_atomic(_path(name), json.dumps(glossary, ensure_ascii=False, indent=2))


def merge_glossary(known: Optional[Dict[str, Any]], fresh: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Combine the saved profile with a newly built glossary: whatever the profile already says
    wins, and the new glossary only adds characters and terms the profile does not have.
    """
    if not known:
        return fresh
    if not fresh:
        return known
    chars = list(known.get("characters") or [])
    seen = {(c.get("name") or "").strip().lower() for c in chars}
    for c in fresh.get("characters") or []:
        key = (c.get("name") or "").strip().lower()
        if key and key not in seen:
            chars.append(c)
            seen.add(key)
    terms = list(known.get("terms") or [])
    seen_terms = {(t.get("source") or "").strip().lower() for t in terms}
    for t in fresh.get("terms") or []:
        key = (t.get("source") or "").strip().lower()
        if key and key not in seen_terms:
            terms.append(t)
            seen_terms.add(key)
    return {
        "summary": known.get("summary") or fresh.get("summary") or "",
        "characters": chars,
        "terms": terms,
        "register_notes": known.get("register_notes") or fresh.get("register_notes") or "",
    }


_SECTION_HEADERS = {
    "context", "characters", "character", "cast", "names", "terms", "terminology", "glossary",
    "notes", "note", "synopsis", "summary", "genre", "setting", "tone", "register", "places",
    "speaker", "speakers", "http", "https",
}
_GENDER_HINT_RE = re.compile(
    r"\(\s*(male|female|man|woman|masculine|feminine|group|mixed group|זכר|נקבה|רבים)\s*\)",
    re.IGNORECASE,
)
_EXPLICIT_PAIR_RE = re.compile(r"^\s*([^=\n\r→]{1,70}?)\s*(?:=|->|=>|→)\s*(.+?)\s*$")
_COLON_PAIR_RE = re.compile(r"^\s*([^:\n\r=→]{1,50}?)\s*:\s*([^:\n\r]{1,60}?)\s*$")


def _normalize_gender_hint(raw: str) -> str:
    val = raw.strip().lower()
    if val in ("male", "man", "masculine", "זכר"):
        return "male"
    if val in ("female", "woman", "feminine", "נקבה"):
        return "female"
    if val in ("group", "mixed group", "רבים"):
        return "mixed group"
    return "unknown"


def combine_context_notes(context_notes: Optional[str], initial_prompt: Optional[str]) -> str:
    """Combine `initial_prompt` and `context_notes` without duplicating identical text."""
    parts: List[str] = []
    seen: set[str] = set()
    for item in (initial_prompt, context_notes):
        cleaned = (item or "").strip()
        if cleaned and cleaned.casefold() not in seen:
            parts.append(cleaned)
            seen.add(cleaned.casefold())
    return "\n".join(parts)


def parse_user_glossary(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract explicit `Source = Target` / `Source -> Target` pairs from user context/glossary notes.

    Returned entries are marked `locked: True` so downstream merging and batch pruning treat
    user-supplied names and terminology as authoritative across every translation batch.
    """
    if not raw_text or not raw_text.strip():
        return None

    characters: List[Dict[str, Any]] = []
    terms: List[Dict[str, Any]] = []
    seen_chars: set[str] = set()
    seen_terms: set[str] = set()

    chunks: List[str] = []
    for line in raw_text.splitlines():
        line_s = line.strip().lstrip("-•* ").strip()
        if not line_s:
            continue
        # Allow comma/semicolon-separated `A = B, C = D` on a single line
        sub_parts = re.split(r"\s*[;,]\s*(?=[^;,=\n\r→]{1,50}\s*(?:=|->|=>|→))", line_s)
        chunks.extend(p.strip() for p in sub_parts if p.strip())

    for chunk in chunks:
        m = _EXPLICIT_PAIR_RE.match(chunk)
        if not m:
            m_colon = _COLON_PAIR_RE.match(chunk)
            if not m_colon:
                continue
            cand_src, cand_tgt = m_colon.group(1).strip(), m_colon.group(2).strip()
            if (
                cand_src.casefold() in _SECTION_HEADERS
                or len(cand_src.split()) > 4
                or cand_src.isdigit()
                or "," in cand_tgt
            ):
                continue
            src, tgt_raw = cand_src, cand_tgt
        else:
            src, tgt_raw = m.group(1).strip(), m.group(2).strip()

        if not src or not tgt_raw or src.casefold() in _SECTION_HEADERS:
            continue

        gender = "unknown"
        g_match = _GENDER_HINT_RE.search(tgt_raw)
        if g_match:
            gender = _normalize_gender_hint(g_match.group(1))
            tgt_raw = _GENDER_HINT_RE.sub("", tgt_raw).strip()
        else:
            g_src = _GENDER_HINT_RE.search(src)
            if g_src:
                gender = _normalize_gender_hint(g_src.group(1))
                src = _GENDER_HINT_RE.sub("", src).strip()

        note = ""
        note_match = re.search(r"\(([^()]{1,80})\)\s*$", tgt_raw)
        if note_match:
            note = note_match.group(1).strip()
            tgt_raw = tgt_raw[:note_match.start()].strip()

        tgt = tgt_raw.strip(" ,;")
        src = src.strip(" ,;")
        if not src or not tgt:
            continue

        if gender != "unknown":
            key = src.casefold()
            if key not in seen_chars:
                characters.append({
                    "name": src,
                    "target_name": tgt,
                    "gender": gender,
                    "notes": note,
                    "speaker_id": "",
                    "locked": True,
                })
                seen_chars.add(key)
        else:
            key = src.casefold()
            if key not in seen_terms:
                terms.append({
                    "source": src,
                    "translation": tgt,
                    "notes": note,
                    "locked": True,
                })
                seen_terms.add(key)

    if not characters and not terms:
        return None
    return {
        "summary": "",
        "characters": characters,
        "terms": terms,
        "register_notes": "",
    }
