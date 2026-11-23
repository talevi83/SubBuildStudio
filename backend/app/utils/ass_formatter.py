"""Advanced SubStation Alpha (.ass) subtitle serializer with styling, speaker colors, and word/karaoke highlighting."""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.utils.rtl_fixer import PDF, RLE, RLM, apply_rtl_fix, contains_rtl, strip_directional_marks
from app.utils.srt_formatter import _ensure_brackets, _get_seg_attr, _is_bracketed_line, split_long_lines

_HEX_COLOR_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")
_PUNCT_STRIP = ".,!?;:\"'“”‘’״׳()[]{}—–-…"

_ASS_POS_MAP = {
    "top_center": 8,
    "top_left": 7,
    "top_right": 9,
    "middle_left": 4,
    "middle_right": 6,
    "middle_center": 5,
}

_FALLBACK_SPEAKER_PALETTE = (
    "#38BDF8",
    "#F472B6",
    "#4ADE80",
    "#FBBF24",
    "#C084FC",
    "#FB923C",
    "#2DD4BF",
    "#F87171",
)


@dataclass
class AssStyleConfig:
    """Visual styling options for .ass export and FFmpeg/libass burn-in."""
    font_family: str = "Heebo"
    font_size_px: int = 18
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: float = 1.6
    background_color: str = "#000000"
    background_opacity: int = 0
    margin_v: int = 16
    bold: bool = False
    color_speakers: bool = False
    highlight_mode: str = "none"  # none | word | karaoke
    highlight_color: str = "#FACC15"


def format_ass_timestamp(seconds: float) -> str:
    """Convert float seconds to ASS timecode: H:MM:SS.cc (centiseconds)."""
    total_cs = int(round(max(0.0, float(seconds)) * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    """Escape literal braces/backslashes and convert newlines into ASS hard line breaks (\\N)."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    escaped = normalized.replace("\\", "\u2216").replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\n", "\\N")


def _hex_to_ass_bgr(hex_color: str, fallback: str = "FFFFFF") -> str:
    match = _HEX_COLOR_RE.match((hex_color or "").strip())
    raw = match.group(1) if match else fallback
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _norm_word(word: str) -> str:
    return strip_directional_marks(word or "").strip().strip(_PUNCT_STRIP).lower()


def _extract_stored_words(seg: Any) -> Optional[List[Dict[str, Any]]]:
    raw_words = _get_seg_attr(seg, "words")
    if isinstance(raw_words, list) and raw_words:
        return raw_words
    words_json = _get_seg_attr(seg, "words_json")
    if isinstance(words_json, str) and words_json.strip():
        try:
            parsed = json.loads(words_json)
            if isinstance(parsed, list) and parsed:
                return parsed
        except (ValueError, TypeError):
            pass
    return None


def resolve_word_timings(
    text: str,
    start_time: float,
    end_time: float,
    stored_words: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Pair display words with stored word timestamps when available, else distribute by character length."""
    clean = strip_directional_marks(text or "").strip()
    if not clean:
        return []

    tokens: List[Tuple[str, bool]] = []
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    for line_idx, line in enumerate(lines):
        for w_idx, w in enumerate(line.split()):
            tokens.append((w, line_idx > 0 and w_idx == 0))

    if not tokens:
        return []

    start_f = max(0.0, float(start_time))
    end_f = max(start_f + 0.05, float(end_time))

    valid_stored: List[Tuple[str, float, float]] = []
    if stored_words:
        for item in stored_words:
            if isinstance(item, dict):
                w = str(item.get("w") or item.get("word") or "").strip()
                s = item.get("s", item.get("start"))
                e = item.get("e", item.get("end"))
            else:
                w = str(getattr(item, "w", getattr(item, "word", "")) or "").strip()
                s = getattr(item, "s", getattr(item, "start", None))
                e = getattr(item, "e", getattr(item, "end", None))
            if w and s is not None and e is not None:
                valid_stored.append((w, float(s), float(e)))

    use_stored = False
    if len(valid_stored) == len(tokens):
        # Verify the stored tokens align with the display text (e.g., untranslated / matching edits)
        matches = sum(
            1 for (disp, _), (sw, _, _) in zip(tokens, valid_stored)
            if _norm_word(disp) == _norm_word(sw) or not _norm_word(disp)
        )
        if matches == len(tokens):
            use_stored = True

    out: List[Dict[str, Any]] = []
    if use_stored:
        cursor = start_f
        for idx, ((disp, nl), (_, s, e)) in enumerate(zip(tokens, valid_stored)):
            ws = max(cursor, min(end_f - 0.02, s))
            we = max(ws + 0.02, min(end_f, e))
            if idx == len(tokens) - 1:
                we = max(we, end_f)
            out.append({"w": disp, "s": round(ws, 3), "e": round(we, 3), "newline_before": nl})
            cursor = we
        return out

    # Proportional fallback weighted by visible character length
    weights = [max(1, len(_norm_word(disp) or disp)) for disp, _ in tokens]
    total_w = sum(weights)
    span = end_f - start_f
    cursor = start_f
    for idx, ((disp, nl), w_len) in enumerate(zip(tokens, weights)):
        if idx == len(tokens) - 1:
            we = end_f
        else:
            we = cursor + span * (w_len / total_w)
        out.append({"w": disp, "s": round(cursor, 3), "e": round(max(cursor + 0.02, we), 3), "newline_before": nl})
        cursor = we
    return out


def _safe_style_id(speaker_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(speaker_id or "")).strip("_")
    return f"Speaker_{cleaned or 'Unknown'}"


def _safe_actor_name(label: str) -> str:
    return (label or "").replace(",", " ").replace("\n", " ").replace("\r", " ").strip()


def _format_plain_ass_text(text: str, apply_rtl: bool, max_chars_per_line: int) -> str:
    wrapped = split_long_lines(text, max_chars_per_line=max_chars_per_line)
    if apply_rtl:
        wrapped = apply_rtl_fix(wrapped)
    return escape_ass_text(wrapped)


def _wrap_rtl_line_if_needed(escaped_line: str, plain_line: str, apply_rtl: bool) -> str:
    if apply_rtl and contains_rtl(plain_line):
        return f"{RLE}{RLM}{escaped_line}{RLM}{PDF}"
    return escaped_line


def _build_highlight_events(
    text: str,
    start_time: float,
    end_time: float,
    style_name: str,
    actor_name: str,
    cfg: AssStyleConfig,
    base_hex_color: str,
    apply_rtl: bool,
    max_chars_per_line: int,
    stored_words: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    wrapped = split_long_lines(text, max_chars_per_line=max_chars_per_line)
    words = resolve_word_timings(wrapped, start_time, end_time, stored_words=stored_words)
    if not words:
        plain = _format_plain_ass_text(text, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
        return [f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},{style_name},{actor_name},0,0,0,,{plain}"]

    # Group words into logical lines so RTL embedding wraps each line cleanly
    lines_words: List[List[Tuple[int, Dict[str, Any]]]] = []
    for idx, wt in enumerate(words):
        if idx == 0 or wt["newline_before"]:
            lines_words.append([])
        lines_words[-1].append((idx, wt))

    if cfg.highlight_mode == "karaoke":
        line_strs: List[str] = []
        for line_group in lines_words:
            parts: List[str] = []
            plain_line = " ".join(wt["w"] for _, wt in line_group)
            for _, wt in line_group:
                cs = max(1, int(round((wt["e"] - wt["s"]) * 100)))
                parts.append(f"{{\\kf{cs}}}{escape_ass_text(wt['w'])}")
            joined = " ".join(parts)
            line_strs.append(_wrap_rtl_line_if_needed(joined, plain_line, apply_rtl))
        body = "\\N".join(line_strs)
        return [f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},{style_name},{actor_name},0,0,0,,{body}"]

    # highlight_mode == "word": emit one Dialogue slice per active word keeping the full sentence visible
    hi_bgr = _hex_to_ass_bgr(cfg.highlight_color, "15CCFA")
    base_bgr = _hex_to_ass_bgr(base_hex_color, "FFFFFF")
    base_bold_num = 1 if cfg.bold else 0
    events: List[str] = []
    for active_idx, wt in enumerate(words):
        slice_start = wt["s"] if active_idx > 0 else start_time
        slice_end = words[active_idx + 1]["s"] if active_idx + 1 < len(words) else end_time
        if slice_end <= slice_start:
            slice_end = max(slice_start + 0.02, wt["e"])

        line_strs = []
        for line_group in lines_words:
            parts = []
            plain_line = " ".join(item["w"] for _, item in line_group)
            for idx, item in line_group:
                w_esc = escape_ass_text(item["w"])
                if idx == active_idx:
                    parts.append(f"{{\\1c{hi_bgr}\\b1}}{w_esc}{{\\1c{base_bgr}\\b{base_bold_num}}}")
                else:
                    parts.append(w_esc)
            joined = " ".join(parts)
            line_strs.append(_wrap_rtl_line_if_needed(joined, plain_line, apply_rtl))
        body = "\\N".join(line_strs)
        events.append(
            f"Dialogue: 0,{format_ass_timestamp(slice_start)},{format_ass_timestamp(slice_end)},{style_name},{actor_name},0,0,0,,{body}"
        )
    return events


def serialize_to_ass(
    segments: List[Any],
    *,
    use_translated: bool = False,
    apply_rtl: bool = True,
    max_chars_per_line: int = 42,
    onscreen_position: str = "top_center",
    bilingual: bool = False,
    style: Optional[AssStyleConfig] = None,
    speakers: Optional[Dict[str, Any]] = None,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> str:
    """Serialize subtitle segments into an Advanced SubStation Alpha (.ass) document."""
    from app.core.video_embed import (
        ALLOWED_BURN_FONTS,
        MAX_BURN_FONT_PX,
        MIN_BURN_FONT_PX,
        _burn_font_name,
        hex_to_ass_color,
    )

    cfg = style or AssStyleConfig()
    font_name = ALLOWED_BURN_FONTS.get((cfg.font_family or "").strip().lower()) or _burn_font_name()
    clamped_px = max(MIN_BURN_FONT_PX, min(MAX_BURN_FONT_PX, int(cfg.font_size_px)))
    ass_font_size = max(24, round(clamped_px * 3.2))
    orig_font_size = max(20, round(ass_font_size * 0.82))
    onscreen_font_size = max(22, round(ass_font_size * 0.9))

    outline_w = max(0.0, min(6.0, float(cfg.outline_width)))
    scaled_outline = round(outline_w * 2.0, 2)
    margin_v = max(8, min(120, int(cfg.margin_v)))
    scaled_margin_v = round(margin_v * 3.0)
    bold_flag = -1 if cfg.bold else 0

    bg_op = max(0, min(100, int(cfg.background_opacity)))
    if bg_op > 0:
        border_style = 3
        outline_col = hex_to_ass_color(cfg.background_color, bg_op)
        back_col = outline_col
        outline_val: Any = 10
        shadow_val: Any = 0
    else:
        border_style = 1
        outline_col = hex_to_ass_color(cfg.outline_color, 100)
        back_col = "&H80000000"
        outline_val = scaled_outline
        shadow_val = 1.5

    def _primary_secondary(base_hex: str) -> Tuple[str, str]:
        if cfg.highlight_mode == "karaoke":
            return hex_to_ass_color(cfg.highlight_color, 100), hex_to_ass_color(base_hex, 100)
        return hex_to_ass_color(base_hex, 100), hex_to_ass_color(cfg.highlight_color, 100)

    def _style_line(
        name: str,
        *,
        font: str = font_name,
        size: int = ass_font_size,
        base_hex: str = cfg.text_color,
        bold: int = bold_flag,
        italic: int = 0,
        b_style: int = border_style,
        o_col: str = outline_col,
        b_col: str = back_col,
        o_val: Any = outline_val,
        s_val: Any = shadow_val,
        alignment: int = 2,
        m_v: int = scaled_margin_v,
    ) -> str:
        pri, sec = _primary_secondary(base_hex)
        return (
            f"Style: {name},{font},{size},{pri},{sec},{o_col},{b_col},"
            f"{bold},{italic},0,0,100,100,0,0,{b_style},{o_val},{s_val},{alignment},40,40,{m_v},1"
        )

    onscreen_align = _ASS_POS_MAP.get(onscreen_position or "top_center", 8)
    styles = [
        _style_line("Default"),
        _style_line("Comms", base_hex="#67E8F9", italic=-1),
        _style_line(
            "OnScreen",
            size=onscreen_font_size,
            base_hex=cfg.text_color,
            b_style=3,
            o_col="&H40000000",
            b_col="&H40000000",
            o_val=10,
            s_val=0,
            alignment=onscreen_align,
            m_v=36,
        ),
        _style_line("Foreign", italic=-1),
        _style_line("OriginalSub", size=orig_font_size, base_hex="#E4E4E7", bold=0),
    ]

    speaker_map: Dict[str, Dict[str, str]] = {}
    if isinstance(speakers, dict):
        for spk_id, info in speakers.items():
            if isinstance(info, dict):
                speaker_map[str(spk_id)] = {
                    "label": str(info.get("character") or info.get("label") or spk_id),
                    "color": str(info.get("color") or ""),
                }

    speaker_style_names: Dict[str, Tuple[str, str]] = {}
    if cfg.color_speakers:
        seen_ids: List[str] = list(speaker_map.keys())
        for seg in segments:
            sid = _get_seg_attr(seg, "speaker")
            if sid and sid not in seen_ids:
                seen_ids.append(str(sid))
        for idx, sid in enumerate(seen_ids):
            raw_color = (speaker_map.get(sid) or {}).get("color") or ""
            if not _HEX_COLOR_RE.match(raw_color.strip()):
                raw_color = _FALLBACK_SPEAKER_PALETTE[idx % len(_FALLBACK_SPEAKER_PALETTE)]
            style_id = _safe_style_id(sid)
            speaker_style_names[sid] = (style_id, raw_color)
            styles.append(_style_line(style_id, base_hex=raw_color))

    ordered = sorted(
        segments,
        key=lambda s: (
            float(_get_seg_attr(s, "start_time", 0.0)),
            (_get_seg_attr(s, "segment_type") or "DIALOGUE") != "ON_SCREEN",
        ),
    )

    events: List[str] = []
    for seg in ordered:
        translated = _get_seg_attr(seg, "translated_text")
        original = _get_seg_attr(seg, "original_text", "")
        clean_orig = strip_directional_marks((original or "").strip())
        clean_trans = strip_directional_marks((translated or "").strip())
        seg_type = (_get_seg_attr(seg, "segment_type") or "DIALOGUE").upper()
        start_time = float(_get_seg_attr(seg, "start_time", 0.0))
        end_time = float(_get_seg_attr(seg, "end_time", 0.0))
        spk_id = _get_seg_attr(seg, "speaker")
        actor_raw = (speaker_map.get(str(spk_id)) or {}).get("label") if spk_id else ""
        actor_name = _safe_actor_name(actor_raw or (str(spk_id) if spk_id else ""))
        stored_words = _extract_stored_words(seg)

        if seg_type == "COMMS":
            base_style, base_hex = "Comms", "#67E8F9"
        elif seg_type == "FOREIGN":
            base_style, base_hex = "Foreign", cfg.text_color
        else:
            base_style, base_hex = "Default", cfg.text_color

        if cfg.color_speakers and spk_id and str(spk_id) in speaker_style_names and seg_type != "ON_SCREEN":
            base_style, base_hex = speaker_style_names[str(spk_id)]

        if bilingual and clean_orig and clean_trans and clean_orig != clean_trans and seg_type != "ON_SCREEN":
            orig_esc = _format_plain_ass_text(clean_orig, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
            trans_esc = _format_plain_ass_text(clean_trans, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
            body = f"{{\\rOriginalSub}}{orig_esc}\\N{{\\r{base_style}}}{trans_esc}"
            events.append(
                f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},{base_style},{actor_name},0,0,0,,{body}"
            )
            continue

        raw_text = translated if (use_translated and translated) else original
        clean_text = strip_directional_marks((raw_text or "").strip())
        if not clean_text:
            continue

        raw_lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]

        if seg_type == "ON_SCREEN" or (raw_lines and all(_is_bracketed_line(ln) for ln in raw_lines)):
            bracketed = _ensure_brackets(clean_text)
            body = _format_plain_ass_text(bracketed, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
            events.append(
                f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},OnScreen,{actor_name},0,0,0,,{body}"
            )
        elif len(raw_lines) >= 2 and _is_bracketed_line(raw_lines[0]):
            top_body = _format_plain_ass_text(_ensure_brackets(raw_lines[0]), apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
            events.append(
                f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},OnScreen,{actor_name},0,0,0,,{top_body}"
            )
            bot_text = "\n".join(raw_lines[1:])
            if cfg.highlight_mode in ("word", "karaoke"):
                events.extend(_build_highlight_events(
                    bot_text, start_time, end_time, base_style, actor_name, cfg, base_hex,
                    apply_rtl, max_chars_per_line, stored_words=stored_words,
                ))
            else:
                bot_body = _format_plain_ass_text(bot_text, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
                events.append(
                    f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},{base_style},{actor_name},0,0,0,,{bot_body}"
                )
        else:
            if cfg.highlight_mode in ("word", "karaoke"):
                events.extend(_build_highlight_events(
                    clean_text, start_time, end_time, base_style, actor_name, cfg, base_hex,
                    apply_rtl, max_chars_per_line, stored_words=stored_words,
                ))
            else:
                body = _format_plain_ass_text(clean_text, apply_rtl=apply_rtl, max_chars_per_line=max_chars_per_line)
                events.append(
                    f"Dialogue: 0,{format_ass_timestamp(start_time)},{format_ass_timestamp(end_time)},{base_style},{actor_name},0,0,0,,{body}"
                )

    header = [
        "[Script Info]",
        "; Script generated by SubBuild Studio",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "Collisions: Normal",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ]
    return "\n".join(header)
