"""
Automatic zero-config local gender pre-tagging via Ollama (runs when `cost_saver_mode == True`).

When translating to Hebrew or Arabic, pings `http://127.0.0.1:11434/api/tags` with a 2.0s timeout.
If Ollama is not running or has no models, silently skips in 0s with no errors.
If Ollama is available, selects a local text model (preferring qwen2.5 / gemma / llama) and asks it
only about the lines the deterministic resolver (`app.pipeline.addressee`) left open: lines with a
2nd-person form, an imperative or a 1st-person self-reference whose listener / speaker gender is
still unknown. Each of those is sent together with the dialogue around it, so the model sees who
was named and who answers instead of an isolated sentence. Requests use `format="json"`, and
`keep_alive=0` on the final one (to free VRAM immediately); the result is
`[Speaker: M/F, To: M/F/G]` hints for the translator.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

# Lines of dialogue sent along with each asked line, on either side, and the most lines one
# request may carry (small local models run with a small context window)
PRETAG_CONTEXT_LINES = 3
PRETAG_MAX_SCENE_LINES = 32

PREFERRED_FAMILIES = ("qwen2.5", "gemma", "llama")
PRETAG_BATCH_SIZE = 15


def _is_ambiguous(text: Optional[str]) -> bool:
    from app.pipeline.addressee import is_gender_sensitive

    return is_gender_sensitive(text)


def is_gendered_target_language(target_lang: Optional[str]) -> bool:
    if not target_lang:
        return False
    norm = target_lang.strip().lower()
    return norm in ("hebrew", "he", "עברית", "arabic", "ar", "ערבית", "العربية")


def pick_ollama_model(models_payload: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Choose the best available local text model from Ollama's `/api/tags` list."""
    names: List[str] = []
    for item in models_payload or ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if "embed" in lower or "nomic" in lower or "bge" in lower:
            continue
        names.append(name)
    if not names:
        return None
    for pref in PREFERRED_FAMILIES:
        for name in names:
            if pref in name.lower():
                return name
    return names[0]


def _field(seg: Any, name: str, default: Any = None) -> Any:
    return seg.get(name, default) if isinstance(seg, dict) else getattr(seg, name, default)


def _speaker_desc(sp_id: Optional[str], speakers: Optional[Dict[str, Any]]) -> Optional[str]:
    if not sp_id:
        return None
    sp_info = (speakers or {}).get(sp_id) if isinstance(speakers, dict) else None
    sp_gender = sp_info.get("gender") if isinstance(sp_info, dict) else None
    sp_label = sp_info.get("label") if isinstance(sp_info, dict) else None
    if sp_gender in ("male", "female"):
        return f"{sp_label or sp_id} ({sp_gender})"
    return sp_label or str(sp_id)


def find_ambiguous_segments(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Any]] = None,
    addressees: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Return segments whose translation depends on the speaker's or the listener's gender:
    1st/2nd-person forms, imperatives ("Come here.") and bare adjective questions ("Ready?")."""
    seg_list = list(segments)
    ambiguous: List[Dict[str, Any]] = []
    for pos, seg in enumerate(seg_list):
        idx = int(_field(seg, "index", 0))
        text = _field(seg, "original_text", "")
        if text and _is_ambiguous(text):
            item: Dict[str, Any] = {"index": idx, "text": str(text)}
            sp_id = _field(seg, "speaker")
            sp_desc = _speaker_desc(sp_id, speakers)
            if sp_desc:
                item["speaker"] = sp_desc
            addr = (addressees or {}).get(idx) or _field(seg, "addressee_info")
            if addr:
                item["addressee"] = str(addr)
            else:
                if pos > 0:
                    prev_sp = _field(seg_list[pos - 1], "speaker")
                    if prev_sp and prev_sp != sp_id:
                        p_desc = _speaker_desc(prev_sp, speakers)
                        if p_desc:
                            item["prev_speaker"] = p_desc
                if pos + 1 < len(seg_list):
                    nxt_sp = _field(seg_list[pos + 1], "speaker")
                    if nxt_sp and nxt_sp != sp_id:
                        n_desc = _speaker_desc(nxt_sp, speakers)
                        if n_desc:
                            item["next_speaker"] = n_desc
            ambiguous.append(item)
    return ambiguous


def _normalize_gender_code(val: Any) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip().upper()
    if s in ("M", "MALE", "MAN"):
        return "M"
    if s in ("F", "FEMALE", "WOMAN"):
        return "F"
    if s in ("G", "GROUP", "PLURAL", "BOTH", "ALL", "MIXED", "MIXED GROUP"):
        return "G"
    return None


def format_gender_tag(speaker: Any, addressee: Any) -> Optional[str]:
    sp = _normalize_gender_code(speaker) or "?"
    to = _normalize_gender_code(addressee) or "?"
    if sp == "?" and to == "?":
        return None
    return f"[Speaker: {sp}, To: {to}]"


def parse_gender_tag(tag: str) -> tuple[Optional[str], Optional[str]]:
    """Parse `[Speaker: M/F/?, To: M/F/G/?]` into `(speaker_gender, addressee_label)`."""
    m = re.search(r"Speaker:\s*([MFG?])\s*,\s*To:\s*([MFG?])", tag or "", re.IGNORECASE)
    if not m:
        return None, None
    sp_map = {"M": "male", "F": "female"}
    to_map = {"M": "male", "F": "female", "G": "group (plural)"}
    return sp_map.get(m.group(1).upper()), to_map.get(m.group(2).upper())


async def detect_ollama_model(timeout: float = 2.0) -> Optional[str]:
    """Ping local Ollama; return the chosen model name or None if unreachable / empty."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code != 200:
                return None
            data = resp.json()
            models = data.get("models") if isinstance(data, dict) else None
            return pick_ollama_model(models or [])
    except Exception as e:
        logger.debug(f"Local Ollama not available ({e}); skipping local gender pre-tagging")
        return None


def _listener_code(value: Any) -> Optional[str]:
    """M / F / G for a listener that is already settled: a display string such as
    `SPEAKER_02 (דוד, male)` / `group (plural)`, or a confident addressee record."""
    if value is None:
        return None
    if isinstance(value, str):
        low = value.lower()
        if "group" in low or "plural" in low:
            return "G"
        if "(female)" in low or ", female)" in low or low.strip() == "female":
            return "F"
        if "(male)" in low or ", male)" in low or low.strip() == "male":
            return "M"
        return None
    from app.pipeline.addressee import HIGH, AddresseeRecord

    rec = value if isinstance(value, AddresseeRecord) else AddresseeRecord.from_dict(value)
    if rec is None or rec.confidence < HIGH:
        return None
    return {"m": "M", "f": "F", "pl": "G"}.get(rec.form or "")


def build_scene_batches(
    asked: Sequence[int],
    context: Sequence[Any],
    speakers: Optional[Dict[str, Any]] = None,
    known_to: Optional[Dict[int, str]] = None,
    batch_size: int = PRETAG_BATCH_SIZE,
    context_lines: int = PRETAG_CONTEXT_LINES,
    max_scene_lines: int = PRETAG_MAX_SCENE_LINES,
) -> List[Dict[str, Any]]:
    """Group the asked segment indices into requests, each with the dialogue around its lines.

    Returns `[{"ask": [indices], "lines": [{"index", "text", "speaker"?, "to"?, "ask"?}]}]`; the lines
    are consecutive runs of the scene, so the model can see who was named and who replies.
    """
    dialogue = [s for s in context if (_field(s, "segment_type", "DIALOGUE") or "DIALOGUE") != "ON_SCREEN"]
    pos_of = {int(_field(s, "index", 0)): pos for pos, s in enumerate(dialogue)}
    wanted = [idx for idx in asked if idx in pos_of]

    def _window(idx: int) -> range:
        pos = pos_of[idx]
        return range(max(0, pos - context_lines), min(len(dialogue), pos + context_lines + 1))

    groups: List[Tuple[List[int], set]] = []
    cur_ask: List[int] = []
    cur_pos: set = set()
    for idx in wanted:
        grown = cur_pos | set(_window(idx))
        if cur_ask and (len(cur_ask) >= batch_size or len(grown) > max_scene_lines):
            groups.append((cur_ask, cur_pos))
            cur_ask, cur_pos = [], set()
            grown = set(_window(idx))
        cur_ask.append(idx)
        cur_pos = grown
    if cur_ask:
        groups.append((cur_ask, cur_pos))

    batches: List[Dict[str, Any]] = []
    for ask, positions in groups:
        ask_set = set(ask)
        lines: List[Dict[str, Any]] = []
        for pos in sorted(positions):
            seg = dialogue[pos]
            idx = int(_field(seg, "index", 0))
            line: Dict[str, Any] = {"index": idx, "text": str(_field(seg, "original_text", "") or "")[:200]}
            sp_desc = _speaker_desc(_field(seg, "speaker"), speakers)
            if sp_desc:
                line["speaker"] = sp_desc
            to_code = (known_to or {}).get(idx)
            if to_code:
                line["to"] = to_code
            if idx in ask_set:
                line["ask"] = True
            lines.append(line)
        batches.append({"ask": ask, "lines": lines})
    return batches


# Lines on either side of an asked line in which a named character grounds a listener hint
GROUNDING_WINDOW = 2


def _character_name_patterns(glossary: Optional[Dict[str, Any]]) -> List[Tuple["re.Pattern[str]", str]]:
    """`(name regex, M|F)` for every glossary character with a known gender (name and aliases)."""
    from app.pipeline.speaker_match import _extract_character_aliases

    chars = [c for c in (glossary or {}).get("characters") or [] if isinstance(c, dict)]
    out: List[Tuple["re.Pattern[str]", str]] = []
    for _idx, ch, aliases in _extract_character_aliases(chars):
        code = _normalize_gender_code(ch.get("gender"))
        if code not in ("M", "F"):
            continue
        for alias in aliases:
            alias = str(alias or "").strip()
            if len(alias) >= 3:
                out.append((re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE), code))
    return out


def grounded_listener_hint(
    answer: Optional[str],
    line_text: str,
    neighbour_texts: Sequence[str],
    name_patterns: Sequence[Tuple["re.Pattern[str]", str]],
) -> Optional[str]:
    """Keep a local model's listener answer (M/F/G) only when the transcript itself supports it.

    A small local model guesses "you" genders from thin air (job_320de88c99: 198 hints, systemic
    inversions). An M/F answer survives only if a glossary character of that gender is named in the
    line or within `GROUNDING_WINDOW` lines of it; a G answer only if the line addresses a group.
    """
    code = _normalize_gender_code(answer)
    if code is None:
        return None
    if code == "G":
        from app.pipeline.addressee import _PLURAL_VOCATIVE_RE, _YOURSELVES_RE

        return "G" if (_PLURAL_VOCATIVE_RE.search(line_text or "") or _YOURSELVES_RE.search(line_text or "")) else None
    texts = [line_text or "", *neighbour_texts]
    for pattern, gender in name_patterns:
        if gender == code and any(pattern.search(t) for t in texts):
            return code
    return None


async def pretag_gender_hints(
    segments: Sequence[Any],
    target_lang: str = "Hebrew",
    glossary: Optional[Dict[str, Any]] = None,
    batch_size: int = PRETAG_BATCH_SIZE,
    usage: Any = None,
    cloud_model: Optional[str] = None,
    on_batch: Optional[Any] = None,
    speakers: Optional[Dict[str, Any]] = None,
    addressees: Optional[Dict[int, Any]] = None,
    context_segments: Optional[Sequence[Any]] = None,
) -> Dict[int, str]:
    """
    Query local Ollama for the still-open lines and return `{segment_index: "[Speaker: M/F, To: M/F/G]"}`.
    Silently returns `{}` if Ollama is not running, has no models, or if any network error occurs.

    `addressees` holds what is already known about each line's listener (display strings or
    addressee records by segment index): a settled listener is not asked about again and is shown to
    the model as evidence. `context_segments` is the whole ordered job, for the lines around each
    asked one (defaults to `segments`).
    """
    if not is_gendered_target_language(target_lang) or not segments:
        return {}

    from app.pipeline.addressee import needs_listener_gender, needs_speaker_gender

    known_gender_by_index: Dict[int, str] = {}
    if isinstance(speakers, dict) and speakers:
        for seg in (context_segments or segments):
            sp_id = _field(seg, "speaker")
            if sp_id and isinstance(speakers.get(sp_id), dict):
                code = _normalize_gender_code(speakers[sp_id].get("gender"))
                if code:
                    known_gender_by_index[int(_field(seg, "index", 0))] = code

    known_to_by_index: Dict[int, str] = {}
    for seg_idx, addr_val in (addressees or {}).items():
        code = _listener_code(addr_val)
        if code:
            known_to_by_index[int(seg_idx)] = code

    # Ask only about what is still open: a listener nobody resolved, a speaker of unknown gender
    asked: List[int] = []
    spoken = [s for s in segments if (_field(s, "segment_type", "DIALOGUE") or "DIALOGUE") != "ON_SCREEN"]
    for item in find_ambiguous_segments(spoken):
        idx, text = item["index"], item["text"]
        open_listener = needs_listener_gender(text) and idx not in known_to_by_index
        open_speaker = needs_speaker_gender(text) and idx not in known_gender_by_index
        if open_listener or open_speaker:
            asked.append(idx)
    if not asked:
        return {}

    model = await detect_ollama_model(timeout=2.0)
    if not model:
        return {}

    cast_lines = []
    for c in (glossary or {}).get("characters") or []:
        if isinstance(c, dict) and c.get("name"):
            cast_lines.append(f"{c.get('name')}: {c.get('gender', 'unknown')}")
    cast_summary = ", ".join(cast_lines[:12]) if cast_lines else "none"

    batches = build_scene_batches(
        asked, context_segments or segments, speakers=speakers, known_to=known_to_by_index, batch_size=batch_size
    )
    hints: Dict[int, str] = {}
    name_patterns = _character_name_patterns(glossary)
    dropped_ungrounded = 0

    try:
        from app.core.model_catalog import estimated_thinking_tokens, request_cost

        ref_model = cloud_model or "gemini-3.8-flash"
        async with httpx.AsyncClient(timeout=90.0) as client:
            for batch_idx, batch in enumerate(batches):
                is_last = batch_idx == len(batches) - 1
                ask_set = set(batch["ask"])
                prompt = (
                    "These are consecutive subtitle lines of a scene. For each line marked \"ask\": true, infer the "
                    "speaker gender (M, F, or ?) and the listener/addressee ('to') gender/number (M, F, G for plural group, or ?).\n"
                    "Note: speaker gender is ONLY who is speaking; addressee ('to') is who they are talking TO "
                    "(e.g. 'Sarah, are you ready?' -> to=F; 'You guys' -> to=G). Use the surrounding lines: who was just "
                    "named, who spoke right before and who answers. Lines that already carry 'to' are settled; use them "
                    "as evidence. Answer ? when the lines do not tell.\n"
                    f"Known characters: {cast_summary}\n"
                    'Return JSON with one item per asked line: {"items": [{"index": <int>, "speaker": "M"|"F"|"?", "to": "M"|"F"|"G"|"?"}]}\n'
                    f"Lines: {json.dumps(batch['lines'], ensure_ascii=False)}"
                )
                body: Dict[str, Any] = {
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                }
                if is_last:
                    body["keep_alive"] = 0

                resp = await client.post(OLLAMA_GENERATE_URL, json=body)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                raw_text = payload.get("response") if isinstance(payload, dict) else None
                if not raw_text:
                    continue
                parsed = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
                items = parsed.get("items") if isinstance(parsed, dict) else None
                batch_tagged = 0
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict) or "index" not in item:
                            continue
                        try:
                            seg_idx = int(item["index"])
                        except (TypeError, ValueError):
                            continue
                        if seg_idx not in ask_set:
                            continue  # a context line: the model was not asked about it
                        # The local model's own speaker-gender guess is never used: only a speaker
                        # gender that is already known (diarization + glossary) is passed on
                        sp_raw = known_gender_by_index.get(seg_idx)
                        to_raw = item.get("to") or item.get("addressee") or item.get("addressee_gender")
                        if seg_idx in known_to_by_index:
                            to_raw = known_to_by_index[seg_idx]
                        else:
                            texts = [str(ln.get("text") or "") for ln in batch["lines"]]
                            pos = next((k for k, ln in enumerate(batch["lines"]) if ln.get("index") == seg_idx), None)
                            if pos is None:
                                continue
                            near = texts[max(0, pos - GROUNDING_WINDOW):pos] + texts[pos + 1:pos + 1 + GROUNDING_WINDOW]
                            grounded = grounded_listener_hint(to_raw, texts[pos], near, name_patterns)
                            if to_raw and _normalize_gender_code(to_raw) and grounded is None:
                                dropped_ungrounded += 1
                            to_raw = grounded
                        tag = format_gender_tag(sp_raw, to_raw)
                        if tag:
                            hints[seg_idx] = tag
                            batch_tagged += 1

                if usage is not None and hasattr(usage, "record_local_ai"):
                    in_tok = int(payload.get("prompt_eval_count") or max(1, len(prompt) // 4)) if isinstance(payload, dict) else max(1, len(prompt) // 4)
                    out_tok = int(payload.get("eval_count") or max(1, len(str(raw_text)) // 4)) if isinstance(payload, dict) else max(1, len(str(raw_text)) // 4)
                    think_tok = estimated_thinking_tokens(ref_model, "low")
                    saved_tok = in_tok + out_tok + think_tok
                    saved_cost = request_cost(ref_model, in_tok, out_tok + think_tok) or 0.0
                    usage.record_local_ai(
                        model=model,
                        requests=1,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        segments_tagged=batch_tagged,
                        saved_tokens=saved_tok,
                        saved_cost_usd=saved_cost,
                    )
                if callable(on_batch):
                    await on_batch()
    except Exception as e:
        logger.debug(f"Local gender pre-tagger stopped early ({e}); continuing with Gemini")
        return hints

    if hints or dropped_ungrounded:
        logger.info(
            f"Local Ollama ({model}) pre-tagged gender hints for {len(hints)} segment(s); "
            f"dropped {dropped_ungrounded} listener guess(es) no character name or group address supports"
        )
    return hints


def format_hints_for_context_notes(hints: Dict[int, str]) -> str:
    """Format `{index: '[Speaker: M, To: F]'}` into a compact context block for GeminiTranslator."""
    if not hints:
        return ""
    lines = [f"- #{idx}: {tag}" for idx, tag in sorted(hints.items())]
    return "Local speaker/addressee gender hints:\n" + "\n".join(lines)
