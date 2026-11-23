"""Cross-reference acoustic voiceprints with dialogue/timeline cues and AI to identify character speakers."""
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_SPEAKER_LABEL_RE = re.compile(r"^Speaker \d+$", re.IGNORECASE)
_SDH_PREFIX_RE = re.compile(r"^\s*(?:\[)?([A-Za-z\u0590-\u05FF][A-Za-z0-9\u0590-\u05FF .'\-]{1,28})(?:\])?\s*:\s*\S")
_SELF_INTRO_RE = re.compile(
    r"\b(?:i['’]?m|i am|my name is|this is|call me)\s+([A-Za-z][A-Za-z .'\-]{1,28})",
    re.IGNORECASE,
)
_DUAL_SPEAKER_LINE_RE = re.compile(r"(?:^|\n)\s*[-–—]\s*\S.*\n\s*[-–—]\s*\S")

_PLURAL_PHRASE_RE = re.compile(
    r"\b(?:you\s+(?:guys|all|two|three|four|both|people|folks)|"
    r"(?:all|both|none|some|any|each)\s+of\s+you|"
    r"y['’]?all|"
    r"ladies\s+and\s+gentlemen)\b",
    re.IGNORECASE,
)

_STOP_ALIASES = frozenset({
    "and", "the", "with", "from", "for", "staff", "group", "kids", "children", "classmates",
    "bullies", "security", "plant", "school", "person", "people", "crowd", "voice", "voices",
    "unknown", "character", "characters", "narrator", "mixed", "male", "female", "guy", "girl",
    "man", "woman", "boy", "lady", "boss", "doctor", "officer", "agent", "teacher", "student",
    "mom", "dad", "mother", "father", "sister", "brother", "friend", "friends", "e.g", "etc",
})
_GENERIC_GROUP_WORDS = frozenset({
    "staff", "group", "classmates", "bullies", "security", "crowd", "kids", "students",
})


class SpeakerCharacterMatch(BaseModel):
    speaker_id: str
    character_name: str = ""
    target_name: str = ""
    gender: Literal["male", "female", "unknown"] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: str = ""
    merge_into_speaker_id: str = ""


class SpeakerMatchResponse(BaseModel):
    matches: List[SpeakerCharacterMatch] = Field(default_factory=list)


def _seg_get(seg: Any, field: str, default: Any = None) -> Any:
    return seg.get(field, default) if isinstance(seg, dict) else getattr(seg, field, default)


def _seg_set(seg: Any, field: str, value: Any) -> None:
    if isinstance(seg, dict):
        seg[field] = value
    else:
        setattr(seg, field, value)


def _fmt_ts(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _clean_text(text: Optional[str]) -> str:
    return " ".join((text or "").split()).strip()


def _extract_character_aliases(characters: Sequence[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any], List[str]]]:
    """Build disambiguated name/title aliases per named glossary character."""
    raw_entries: List[Tuple[int, Dict[str, Any], set[str], set[str]]] = []
    token_owners: Dict[str, set[int]] = defaultdict(set)

    for idx, ch in enumerate(characters or ()):
        if not isinstance(ch, dict):
            continue
        name = str(ch.get("name") or "").strip()
        if not name:
            continue
        gender = str(ch.get("gender") or "unknown").strip().lower()
        name_lower = name.lower()
        if gender == "mixed group" or any(w in name_lower.split() for w in _GENERIC_GROUP_WORDS):
            continue

        bare_tokens: set[str] = set()
        titled_tokens: set[str] = set()

        # Split slash / comma / parenthetical variants: "Raymond / Ray Mack", "Alex (Alexander) Mack"
        paren_parts = re.findall(r"\(([^)]+)\)", name)
        stripped_parens = re.sub(r"\([^)]*\)", " ", name)
        variants = [stripped_parens, *paren_parts]

        # Detect surname from full name if multi-word (ignoring slash options before last word)
        words_clean = [w.strip(".,!?\"'") for w in stripped_parens.replace("/", " ").split() if w.strip(".,!?\"'")]
        surname = words_clean[-1].lower() if len(words_clean) >= 2 and len(words_clean[-1]) >= 3 else ""

        for part in variants:
            for chunk in re.split(r"[/,;]+", part):
                chunk_clean = " ".join(chunk.strip(" .,\"'").split())
                if len(chunk_clean) < 3:
                    continue
                low = chunk_clean.lower()
                if low not in _STOP_ALIASES:
                    bare_tokens.add(low)
                for tok in re.split(r"\s+", low):
                    tok = tok.strip(".,!?\"'")
                    if len(tok) >= 3 and tok not in _STOP_ALIASES:
                        bare_tokens.add(tok)

        if surname and surname not in _STOP_ALIASES:
            if gender == "male":
                titled_tokens.update({f"mr. {surname}", f"mr {surname}"})
            elif gender == "female":
                titled_tokens.update({f"mrs. {surname}", f"mrs {surname}", f"ms. {surname}", f"ms {surname}", f"miss {surname}"})
            else:
                titled_tokens.update({f"mr. {surname}", f"mr {surname}", f"mrs. {surname}", f"mrs {surname}", f"ms. {surname}", f"ms {surname}", f"miss {surname}", f"dr. {surname}", f"dr {surname}"})

        for tok in bare_tokens:
            token_owners[tok].add(idx)
        raw_entries.append((idx, ch, bare_tokens, titled_tokens))

    resolved: List[Tuple[int, Dict[str, Any], List[str]]] = []
    for idx, ch, bare_tokens, titled_tokens in raw_entries:
        # Drop ambiguous single-word tokens shared across 2+ characters (e.g., family surname "Mack")
        unique_aliases = [t for t in bare_tokens if len(token_owners.get(t, ())) <= 1 or " " in t]
        all_aliases = sorted(set(unique_aliases) | titled_tokens, key=len, reverse=True)
        if all_aliases:
            resolved.append((idx, ch, all_aliases))
    return resolved


def _match_alias_text(phrase: str, alias_entries: Sequence[Tuple[int, Dict[str, Any], List[str]]]) -> Optional[int]:
    low = " ".join(phrase.lower().split()).strip(" .,:;!?\"'[]")
    if not low:
        return None
    for idx, _ch, aliases in alias_entries:
        for al in aliases:
            if low == al or re.search(rf"(?<![a-z0-9\u0590-\u05FF]){re.escape(al)}(?![a-z0-9\u0590-\u05FF])", low):
                return idx
    return None


def compute_conversational_clues(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Dict[str, Any]]],
    characters: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[Dict[str, List[str]], List[SpeakerCharacterMatch]]:
    """
    Analyze SDH prefixes, self-introductions, and direct vocative address/reply turns across
    dialogue segments to gather per-speaker character clues and conservative deterministic matches.
    """
    dialogue = sorted(
        (s for s in (segments or ()) if (_seg_get(s, "segment_type", "DIALOGUE") or "DIALOGUE") != "ON_SCREEN"),
        key=lambda s: (float(_seg_get(s, "start_time", 0.0)), int(_seg_get(s, "index", 0))),
    )
    alias_entries = _extract_character_aliases(characters or ())
    if not dialogue or not alias_entries:
        return {}, []

    alias_to_char: List[Tuple[str, int]] = []
    char_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, ch, aliases in alias_entries:
        char_by_idx[idx] = ch
        for al in aliases:
            alias_to_char.append((al, idx))
    alias_to_char.sort(key=lambda item: len(item[0]), reverse=True)

    scores: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    clues: Dict[str, List[str]] = defaultdict(list)

    def _add_clue(sp_id: str, note: str) -> None:
        bucket = clues[sp_id]
        if note not in bucket and len(bucket) < 8:
            bucket.append(note)

    for i, seg in enumerate(dialogue):
        sp = (_seg_get(seg, "speaker") or "").strip()
        text = _clean_text(_seg_get(seg, "original_text", ""))
        if not text:
            continue
        ts = _fmt_ts(float(_seg_get(seg, "start_time", 0.0)))

        # 1. SDH / explicit speaker tag prefix ("ALEX: ...")
        m_sdh = _SDH_PREFIX_RE.match(text)
        if m_sdh and sp:
            c_idx = _match_alias_text(m_sdh.group(1), alias_entries)
            if c_idx is not None:
                cname = str(char_by_idx[c_idx].get("name") or "")
                scores[sp][c_idx] += 4.0
                _add_clue(sp, f"SDH tag '{m_sdh.group(1)}' -> {cname} at [{ts}]")

        # 2. Explicit self-introduction ("I'm Annie", "My name is Alex")
        for m_self in _SELF_INTRO_RE.finditer(text):
            c_idx = _match_alias_text(m_self.group(1), alias_entries)
            if c_idx is not None and sp:
                cname = str(char_by_idx[c_idx].get("name") or "")
                scores[sp][c_idx] += 3.5
                _add_clue(sp, f"Self-intro '{text[:50]}' -> {cname} at [{ts}]")

        # 3. Vocative direct address at start or end of utterance
        matched_vocatives: set[int] = set()
        for al, c_idx in alias_to_char:
            if c_idx in matched_vocatives:
                continue
            esc = re.escape(al)
            lead_re = rf"^(?:hey|hi|hello|okay|ok|well|oh|wait|look|listen|sorry|excuse me|please)?[\s,!?\-]*\b{esc}\b(?:\s*[,!?]|\s*$)"
            tail_re = rf",\s*\b{esc}\b[.!?]*$"
            if re.search(lead_re, text, re.IGNORECASE) or re.search(tail_re, text, re.IGNORECASE):
                matched_vocatives.add(c_idx)
                cname = str(char_by_idx[c_idx].get("name") or "")
                if sp:
                    scores[sp][c_idx] -= 2.0
                    _add_clue(sp, f"Addressed '{al}' ({cname}) at [{ts}]")

                # Next speaker replying immediately is strong positive signal for the addressed character
                if i + 1 < len(dialogue):
                    nxt = dialogue[i + 1]
                    nxt_sp = (_seg_get(nxt, "speaker") or "").strip()
                    gap = float(_seg_get(nxt, "start_time", 0.0)) - float(_seg_get(seg, "end_time", 0.0))
                    if nxt_sp and nxt_sp != sp and gap <= 4.0:
                        scores[nxt_sp][c_idx] += 2.0
                        _add_clue(nxt_sp, f"Replied after '{text[:45]}' -> likely {cname} at [{ts}]")

                # Previous speaker just finishing before someone greets/calls them by name
                if i - 1 >= 0:
                    prv = dialogue[i - 1]
                    prv_sp = (_seg_get(prv, "speaker") or "").strip()
                    gap = float(_seg_get(seg, "start_time", 0.0)) - float(_seg_get(prv, "end_time", 0.0))
                    if prv_sp and prv_sp != sp and gap <= 3.0:
                        scores[prv_sp][c_idx] += 1.2
                        _add_clue(prv_sp, f"Spoke right before being called '{al}' ({cname}) at [{ts}]")

    # Emit conservative deterministic matches (strict threshold + margin, 1 primary speaker per character)
    best_by_char: Dict[int, Tuple[float, float, str]] = {}
    for sp_id, char_scores in scores.items():
        if speakers is not None and sp_id not in speakers:
            continue
        ranked = sorted(char_scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            continue
        top_c, top_val = ranked[0]
        second_val = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_val - max(0.0, second_val)
        if top_val >= 3.0 and margin >= 1.5:
            prev = best_by_char.get(top_c)
            if prev is None or top_val > prev[0]:
                best_by_char[top_c] = (top_val, margin, sp_id)

    matches: List[SpeakerCharacterMatch] = []
    for c_idx, (top_val, margin, sp_id) in best_by_char.items():
        ch = char_by_idx[c_idx]
        gender_raw = str(ch.get("gender") or "unknown").strip().lower()
        gender: Literal["male", "female", "unknown"] = (
            gender_raw if gender_raw in ("male", "female") else "unknown"
        )
        conf: Literal["high", "medium", "low"] = "high" if (top_val >= 4.0 and margin >= 2.0) else "medium"
        matches.append(
            SpeakerCharacterMatch(
                speaker_id=sp_id,
                character_name=str(ch.get("name") or "").strip(),
                target_name=str(ch.get("target_name") or "").strip(),
                gender=gender,
                confidence=conf,
                evidence="; ".join(clues.get(sp_id, [])[:2]) or "Conversational name/turn match",
            )
        )
    return dict(clues), matches


def format_speaker_identity(
    speaker_id: Optional[str],
    speakers: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Format a speaker profile into a concise prompt tag like `'SPEAKER_01 (שרה, female)'`."""
    if not speaker_id:
        return None
    info = (speakers or {}).get(speaker_id)
    if not isinstance(info, dict):
        return speaker_id
    label = str(info.get("label") or "").strip()
    gender = str(info.get("gender") or "").strip().lower()
    parts: List[str] = []
    if label and not _DEFAULT_SPEAKER_LABEL_RE.match(label):
        parts.append(label)
    if gender in ("male", "female"):
        parts.append(gender)
    return f"{speaker_id} ({', '.join(parts)})" if parts else speaker_id


def _is_informative_speaker(
    speaker_id: str,
    speakers: Optional[Dict[str, Dict[str, Any]]],
) -> bool:
    info = (speakers or {}).get(speaker_id)
    if not isinstance(info, dict):
        return False
    gender = str(info.get("gender") or "").strip().lower()
    label = str(info.get("label") or "").strip()
    return gender in ("male", "female") or (bool(label) and not _DEFAULT_SPEAKER_LABEL_RE.match(label))


def _match_vocative_character(
    text: str,
    alias_to_char: Sequence[Tuple[str, int]],
    char_by_idx: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for al, c_idx in alias_to_char:
        esc = re.escape(al)
        lead_re = rf"^(?:hey|hi|hello|okay|ok|well|oh|wait|look|listen|sorry|excuse me|please)?[\s,!?\-]*\b{esc}\b(?:\s*[,!?]|\s*$)"
        tail_re = rf",\s*\b{esc}\b[.!?]*$"
        mid_re = rf",\s*\b{esc}\b\s*[,!?]"
        if (
            re.search(lead_re, text, re.IGNORECASE)
            or re.search(tail_re, text, re.IGNORECASE)
            or re.search(mid_re, text, re.IGNORECASE)
        ):
            return char_by_idx[c_idx]
    return None


def infer_segment_addressees(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Dict[str, Any]]] = None,
    characters: Optional[Sequence[Dict[str, Any]]] = None,
    max_turn_gap: float = 4.5,
) -> Dict[int, str]:
    """
    Infer the listener/addressee (`to`) per dialogue segment index, as a display string.

    Distinguishes who is speaking (`speaker`) from who is being addressed (`addressee`) so
    gendered target languages conjugate 2nd-person forms for the listener rather than the speaker.
    A thin view over `app.pipeline.addressee.resolve_addressees`, which also scores how sure each
    answer is; anything below a usable guess is left out here.
    """
    from app.pipeline.addressee import MEDIUM, resolve_addressees

    out: Dict[int, str] = {}
    for seg_idx, record in resolve_addressees(segments, speakers, characters, max_turn_gap=max_turn_gap).items():
        shown = record.display(speakers) if record.confidence >= MEDIUM else None
        if shown:
            out[seg_idx] = shown
    return out


def build_speaker_dossier(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Dict[str, Any]]],
    glossary: Optional[Dict[str, Any]],
    clue_summaries: Optional[Dict[str, List[str]]] = None,
    max_speakers: int = 18,
) -> str:
    """Build a compact conversational dossier per acoustic speaker cluster for LLM character matching."""
    dialogue = sorted(
        (s for s in (segments or ()) if (_seg_get(s, "segment_type", "DIALOGUE") or "DIALOGUE") != "ON_SCREEN"),
        key=lambda s: (float(_seg_get(s, "start_time", 0.0)), int(_seg_get(s, "index", 0))),
    )
    by_sp_indices: Dict[str, List[int]] = defaultdict(list)
    sp_durations: Dict[str, float] = defaultdict(float)
    for pos, seg in enumerate(dialogue):
        sp = (_seg_get(seg, "speaker") or "").strip()
        if sp:
            by_sp_indices[sp].append(pos)
            dur = max(0.0, float(_seg_get(seg, "end_time", 0.0)) - float(_seg_get(seg, "start_time", 0.0)))
            sp_durations[sp] += dur

    alias_entries = _extract_character_aliases((glossary or {}).get("characters") or ())

    lines: List[str] = []
    if glossary and glossary.get("summary"):
        lines.append(f"SHOW SUMMARY: {glossary['summary']}")
    chars = (glossary or {}).get("characters") or []
    if chars:
        lines.append("\nKNOWN CHARACTERS:")
        for ch in chars:
            if not isinstance(ch, dict) or not ch.get("name"):
                continue
            tname = ch.get("target_name") or ch["name"]
            g = ch.get("gender") or "unknown"
            note = f" — {ch['notes']}" if ch.get("notes") else ""
            lines.append(f"- {ch['name']} (target: {tname}, gender: {g}){note}")

    ranked_speakers = sorted(by_sp_indices.keys(), key=lambda sid: len(by_sp_indices[sid]), reverse=True)[:max_speakers]
    lines.append("\nACOUSTIC VOICE CLUSTERS (sorted by line count):")
    for sid in ranked_speakers:
        positions = by_sp_indices[sid]
        sp_meta = (speakers or {}).get(sid) or {}
        cur_label = str(sp_meta.get("label") or sid)
        cur_gender = str(sp_meta.get("gender") or "unknown")
        first_ts = _fmt_ts(float(_seg_get(dialogue[positions[0]], "start_time", 0.0)))
        last_ts = _fmt_ts(float(_seg_get(dialogue[positions[-1]], "start_time", 0.0)))
        lines.append(
            f"\n[{sid}] lines={len(positions)}, duration={sp_durations[sid]:.1f}s, "
            f"span={first_ts}-{last_ts}, current_label={cur_label}, current_gender={cur_gender}"
        )
        sp_clues = (clue_summaries or {}).get(sid) or []
        if sp_clues:
            lines.append("  Clues: " + " | ".join(sp_clues[:4]))

        # Score candidate lines to select up to 7 diverse, high-signal turns across the timeline
        scored_pos: List[Tuple[float, int]] = []
        for rank_idx, pos in enumerate(positions):
            seg = dialogue[pos]
            txt = _clean_text(_seg_get(seg, "original_text", ""))
            sc = 0.0
            if len(txt) >= 18:
                sc += 1.0
            if "?" in txt:
                sc += 0.5
            if _SELF_INTRO_RE.search(txt) or _SDH_PREFIX_RE.match(txt):
                sc += 3.0
            if _match_alias_text(txt, alias_entries) is not None:
                sc += 2.0
            # Light spread preference by bucket
            scored_pos.append((sc + (rank_idx % 5) * 0.05, pos))

        # Pick evenly across positions while favoring high-signal lines
        selected: List[int] = []
        used_buckets: set[int] = set()
        for _sc, pos in sorted(scored_pos, key=lambda kv: (-kv[0], kv[1])):
            bucket = int((pos / max(1, len(dialogue))) * 7)
            if bucket not in used_buckets or len(selected) < 4:
                selected.append(pos)
                used_buckets.add(bucket)
            if len(selected) >= 7:
                break
        if len(selected) < min(7, len(positions)):
            for _sc, pos in sorted(scored_pos, key=lambda kv: (-kv[0], kv[1])):
                if pos not in selected:
                    selected.append(pos)
                if len(selected) >= 7:
                    break

        lines.append("  Sample exchanges:")
        for pos in sorted(selected):
            cur = dialogue[pos]
            cur_t = float(_seg_get(cur, "start_time", 0.0))
            parts: List[str] = []
            if pos - 1 >= 0:
                prv = dialogue[pos - 1]
                if cur_t - float(_seg_get(prv, "end_time", 0.0)) <= 10.0:
                    psp = (_seg_get(prv, "speaker") or "?").strip()
                    parts.append(f"    prev [{psp} {_fmt_ts(float(_seg_get(prv, 'start_time', 0.0)))}]: {_clean_text(_seg_get(prv, 'original_text', ''))}")
            parts.append(f"  * [{sid} {_fmt_ts(cur_t)}]: {_clean_text(_seg_get(cur, 'original_text', ''))}")
            if pos + 1 < len(dialogue):
                nxt = dialogue[pos + 1]
                if float(_seg_get(nxt, "start_time", 0.0)) - float(_seg_get(cur, "end_time", 0.0)) <= 10.0:
                    nsp = (_seg_get(nxt, "speaker") or "?").strip()
                    parts.append(f"    next [{nsp} {_fmt_ts(float(_seg_get(nxt, 'start_time', 0.0)))}]: {_clean_text(_seg_get(nxt, 'original_text', ''))}")
            lines.extend(parts)

    return "\n".join(lines)


async def _call_ai_speaker_matcher(
    dossier: str,
    target_language: str,
    *,
    api_key_override: Optional[str] = None,
    usage: Any = None,
    flex_state: Any = None,
) -> Optional[List[SpeakerCharacterMatch]]:
    from app.core.user_settings import load_settings
    from app.providers.llm import call_text_json, resolve_provider_spec

    app_settings = load_settings()
    cfg = app_settings.glossary if app_settings.glossary.model else app_settings.translation
    if not cfg.model:
        return None
    spec = resolve_provider_spec(
        cfg.provider,
        cfg.model,
        key_id=cfg.key_id,
        api_key_override=api_key_override if cfg.provider == "google" else None,
        thinking=cfg.thinking,
        flex=bool(cfg.flex),
    )
    if not spec.api_key and not spec.is_local:
        return None

    system_prompt = (
        "You match acoustic speaker voice clusters (SPEAKER_XX) to characters from a show/movie glossary.\n"
        "You will be given the character list and a dossier of each SPEAKER_XX cluster showing its line count, "
        "conversational clues (vocatives/replies/self-introductions), and key dialogue exchanges with adjacent lines.\n"
        "Rules:\n"
        "1. Cross-reference who speaks to whom, who gets addressed by name and replies next, character roles, and tone.\n"
        f"2. Set `character_name` (from KNOWN CHARACTERS or clearly identified role), `target_name` (in {target_language}), "
        "`gender` ('male', 'female', or 'unknown'), `confidence` ('high', 'medium', or 'low'), and a short `evidence`.\n"
        "3. Return 'high' or 'medium' ONLY when conversational context strongly supports the match. "
        "If unsure, leave `character_name` empty ('') and `confidence`='low' — NEVER guess wildly.\n"
        "4. Only set `merge_into_speaker_id` when two clusters unmistakably belong to the exact same person "
        "(e.g. a smaller cluster is clearly a split fragment of a larger cluster of the same character)."
    )
    resp = await call_text_json(
        spec,
        system_instruction=system_prompt,
        prompt=dossier,
        schema_cls=SpeakerMatchResponse,
        temperature=0.1,
        usage=usage,
        stage="glossary",
        flex_state=flex_state,
    )
    parsed = SpeakerMatchResponse.model_validate_json(resp.text)
    return parsed.matches


async def match_speakers_to_characters(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Dict[str, Any]]],
    glossary: Optional[Dict[str, Any]],
    *,
    target_language: str = "Hebrew",
    use_ai: bool = True,
    overwrite_manual: bool = False,
    auto_merge: bool = True,
    api_key_override: Optional[str] = None,
    usage: Any = None,
    flex_state: Any = None,
) -> Dict[str, Any]:
    """
    Identify which character each acoustic SPEAKER_XX cluster represents and optionally merge duplicates.

    Runs deterministic vocative/turn-taking matching first, then upgrades with structured AI matching
    when configured (non-fatal on offline/unconfigured/error). Mutates `speakers`, `segments`, and
    `glossary["characters"]` in-place.
    """
    if not speakers or not segments:
        return {"matched": 0, "merged": 0, "method": "deterministic"}

    chars = (glossary or {}).get("characters") or []
    clue_summaries, det_matches = compute_conversational_clues(segments, speakers, chars)
    method = "deterministic"

    ai_matches: Optional[List[SpeakerCharacterMatch]] = None
    if use_ai and os.environ.get("SUBBUILD_OFFLINE") != "1":
        try:
            dossier = build_speaker_dossier(segments, speakers, glossary, clue_summaries)
            ai_matches = await _call_ai_speaker_matcher(
                dossier,
                target_language,
                api_key_override=api_key_override,
                usage=usage,
                flex_state=flex_state,
            )
            if ai_matches is not None:
                method = "ai"
        except Exception as exc:
            logger.info(f"AI speaker-character cross-matching skipped (using deterministic clues): {exc}")

    # Combine deterministic + AI matches (AI wins when confident; deterministic high-confidence fills gaps)
    merged_by_sp: Dict[str, SpeakerCharacterMatch] = {m.speaker_id: m for m in det_matches}
    if ai_matches:
        for m in ai_matches:
            sid = (m.speaker_id or "").strip()
            if not sid or sid not in speakers:
                continue
            existing = merged_by_sp.get(sid)
            if m.character_name.strip() and m.confidence in ("high", "medium"):
                merged_by_sp[sid] = m
            elif m.merge_into_speaker_id.strip() and existing is None:
                merged_by_sp[sid] = m
            elif existing is not None and existing.confidence == "high":
                # Keep deterministic high-confidence match if AI left it blank or low
                continue

    matched_count = 0
    for sid, m in merged_by_sp.items():
        if sid not in speakers or not isinstance(speakers[sid], dict):
            continue
        if m.confidence not in ("high", "medium"):
            continue
        cname = m.character_name.strip()
        tname = m.target_name.strip() or cname
        if not cname:
            continue

        profile = speakers[sid]
        cur_label = str(profile.get("label") or "").strip()
        is_default = (not cur_label) or bool(_DEFAULT_SPEAKER_LABEL_RE.match(cur_label))
        if overwrite_manual or is_default:
            profile["label"] = tname
        cur_gender = str(profile.get("gender") or "unknown").strip().lower()
        from_voice = profile.get("gender_source") == "voice"
        if m.gender in ("male", "female") and (overwrite_manual or from_voice or cur_gender in ("", "unknown")):
            profile["gender"] = m.gender
            profile.pop("gender_source", None)  # matched to a character: the transcript's gender wins
        profile["character_name"] = cname
        profile["confidence"] = m.confidence
        if m.evidence:
            profile["evidence"] = m.evidence.strip()
        matched_count += 1

    # Count cues per active speaker to pick primary cluster and safely merge smaller duplicates
    sp_counts: Counter[str] = Counter()
    for seg in segments:
        if (_seg_get(seg, "segment_type", "DIALOGUE") or "DIALOGUE") == "ON_SCREEN":
            continue
        sp = (_seg_get(seg, "speaker") or "").strip()
        if sp:
            sp_counts[sp] += 1

    merged_map: Dict[str, str] = {}
    if auto_merge:
        # Explicit high-confidence merge targets from AI
        for sid, m in merged_by_sp.items():
            target = (m.merge_into_speaker_id or "").strip()
            if (
                m.confidence == "high"
                and target
                and target != sid
                and target in speakers
                and sid in speakers
                and sp_counts.get(target, 0) >= sp_counts.get(sid, 0)
            ):
                merged_map[sid] = target

        # Same high-confidence normalized character_name across multiple clusters -> fold smaller into largest
        by_char_key: Dict[str, List[str]] = defaultdict(list)
        for sid, m in merged_by_sp.items():
            if sid in merged_map or sid not in speakers:
                continue
            key = m.character_name.strip().lower()
            if key and m.confidence == "high":
                by_char_key[key].append(sid)
        for _key, sids in by_char_key.items():
            if len(sids) < 2:
                continue
            sids_sorted = sorted(sids, key=lambda s: sp_counts.get(s, 0), reverse=True)
            primary = sids_sorted[0]
            for dup in sids_sorted[1:]:
                merged_map[dup] = primary

    # Resolve transitive merge chains safely without cycles
    def _resolve_target(sid: str) -> str:
        cur = sid
        visited = {cur}
        while cur in merged_map:
            nxt = merged_map[cur]
            if nxt in visited or nxt not in speakers:
                break
            visited.add(nxt)
            cur = nxt
        return cur

    final_merges = {src: _resolve_target(src) for src in merged_map if _resolve_target(src) != src}
    merged_count = 0
    if final_merges:
        for seg in segments:
            sp = (_seg_get(seg, "speaker") or "").strip()
            if sp in final_merges:
                _seg_set(seg, "speaker", final_merges[sp])
        for src in final_merges:
            if src in speakers:
                speakers.pop(src, None)
                merged_count += 1
        # Recompute counts after merging
        sp_counts.clear()
        for seg in segments:
            if (_seg_get(seg, "segment_type", "DIALOGUE") or "DIALOGUE") == "ON_SCREEN":
                continue
            sp = (_seg_get(seg, "speaker") or "").strip()
            if sp:
                sp_counts[sp] += 1

    # Sync primary speaker_id back onto matching glossary characters
    if chars:
        best_sp_for_char: Dict[str, str] = {}
        for sid, profile in speakers.items():
            cname = str(profile.get("character_name") or "").strip().lower()
            if not cname:
                continue
            prev_sid = best_sp_for_char.get(cname)
            if prev_sid is None or sp_counts.get(sid, 0) > sp_counts.get(prev_sid, 0):
                best_sp_for_char[cname] = sid

        for ch in chars:
            if not isinstance(ch, dict):
                continue
            existing_sid = str(ch.get("speaker_id") or "").strip()
            if existing_sid in final_merges:
                ch["speaker_id"] = final_merges[existing_sid]
                existing_sid = ch["speaker_id"]
            if existing_sid and existing_sid in speakers and not overwrite_manual:
                continue
            keys = [
                str(ch.get("name") or "").strip().lower(),
                str(ch.get("target_name") or "").strip().lower(),
            ]
            for k in keys:
                if k and k in best_sp_for_char:
                    ch["speaker_id"] = best_sp_for_char[k]
                    break

    return {"matched": matched_count, "merged": merged_count, "method": method}
