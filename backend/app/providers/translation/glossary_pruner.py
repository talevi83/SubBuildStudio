"""
Dynamic per-batch glossary pruning to reduce repeated input tokens per Gemini request.

Keeps the core cast (first `max_core_characters` characters) plus any character or terminology
entry mentioned in the current batch or its surrounding sliding context window. Never mutates
the source glossary dict.
"""
import re
from typing import Any, Dict, Optional, Sequence


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts = [
            item.get("original_text") or item.get("text") or "",
            item.get("speaker_info") or "",
            item.get("addressee_info") or "",
            item.get("addressee_guess") or "",
            item.get("speaker") or "",
            item.get("addressee") if isinstance(item.get("addressee"), str) else "",
        ]
        return " ".join(str(p) for p in parts if p)
    parts = [
        getattr(item, "original_text", "") or getattr(item, "text", "") or "",
        getattr(item, "speaker_info", "") or "",
        getattr(item, "addressee_info", "") or "",
        getattr(item, "addressee_guess", "") or "",
        getattr(item, "speaker", "") or "",
    ]
    return " ".join(str(p) for p in parts if p)


def _build_window_text(
    batch_texts: Sequence[Any],
    previous_context: Optional[Sequence[Any]] = None,
    lookahead_context: Optional[Sequence[Any]] = None,
) -> str:
    parts = [_extract_text(x) for x in (batch_texts or ())]
    if previous_context:
        parts.extend(_extract_text(x) for x in previous_context)
    if lookahead_context:
        parts.extend(_extract_text(x) for x in lookahead_context)
    return "\n".join(p for p in parts if p).casefold()


def _character_matches(char: Dict[str, Any], window_cf: str, window_tokens: set[str]) -> bool:
    speaker_id = (char.get("speaker_id") or "").strip().casefold()
    if speaker_id and speaker_id in window_tokens:
        return True
    for key in ("name", "target_name"):
        val = (char.get(key) or "").strip()
        if not val:
            continue
        val_cf = val.casefold()
        if val_cf in window_cf:
            return True
        for token in re.findall(r"[^\W_]+", val_cf):
            if len(token) >= 3 and token in window_tokens:
                return True
    return False


def _term_matches(term: Dict[str, Any], window_cf: str, window_tokens: Optional[set[str]] = None) -> bool:
    if term.get("locked"):
        return True
    source = (term.get("source") or "").strip()
    if not source:
        return False
    src_cf = source.casefold()
    if src_cf in window_cf:
        return True
    if window_tokens:
        for tok in re.findall(r"[^\W_]+", src_cf):
            if len(tok) >= 4:
                if tok in window_tokens:
                    return True
                stem = tok.rstrip("s")
                if len(stem) >= 4 and any(wt.startswith(stem) for wt in window_tokens):
                    return True
    return False


def prune_glossary_for_batch(
    glossary: Optional[Dict[str, Any]],
    batch_texts: Sequence[Any],
    previous_context: Optional[Sequence[Any]] = None,
    lookahead_context: Optional[Sequence[Any]] = None,
    max_core_characters: int = 6,
) -> Optional[Dict[str, Any]]:
    """
    Return a filtered shallow copy of `glossary` tailored to a single translation batch:
      - First `max_core_characters` characters (main cast) plus any `locked` user characters.
      - Any additional character whose full name or name token (>= 3 chars) appears
        case-insensitively in `batch_texts` + `previous_context` + `lookahead_context`.
      - Any `locked` user terminology item or terminology item matching that window.

    Does not mutate `glossary`. Returns None if `glossary` is None or empty.
    """
    if not glossary:
        return None

    window_cf = _build_window_text(batch_texts, previous_context, lookahead_context)
    window_tokens = set(re.findall(r"[^\W_]+", window_cf))

    all_chars = list(glossary.get("characters") or [])
    core_cutoff = max(0, max_core_characters)
    pruned_chars = []
    for idx, char in enumerate(all_chars):
        if not isinstance(char, dict):
            continue
        if char.get("locked") or idx < core_cutoff or _character_matches(char, window_cf, window_tokens):
            pruned_chars.append(dict(char))

    all_terms = list(glossary.get("terms") or [])
    pruned_terms = [
        dict(term)
        for term in all_terms
        if isinstance(term, dict) and _term_matches(term, window_cf, window_tokens)
    ]

    pruned: Dict[str, Any] = dict(glossary)
    pruned["characters"] = pruned_chars
    pruned["terms"] = pruned_terms
    return pruned
