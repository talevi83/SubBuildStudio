"""
Embedded subtitle tracks: pick a video's forced/"signs" track or its full dialogue track and
extract it with ffmpeg into ON_SCREEN / DIALOGUE segment dicts.

Used by the on-screen text stage (forced track instead of a frame scan), the audio stage
(dialogue track instead of Whisper) and the sync stage (reference timing).
"""
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.db_models import SegmentType
from app.models.schemas import SubtitleStreamInfo
from app.pipeline.onscreen_merge import make_onscreen_segment
from app.pipeline.stages.audio_stage import get_ffmpeg_bin
from app.utils.subtitle_parser import SubtitleParseError, parse_subtitle_text
from app.core.ffmpeg_safety import safe_input


def pick_forced_track(streams: Sequence[SubtitleStreamInfo]) -> Optional[SubtitleStreamInfo]:
    """The best text-based forced/"signs" subtitle stream, if any."""
    candidates = [s for s in streams if s.looks_forced and s.text_based]
    if not candidates:
        return None
    # Prefer the explicit forced disposition, then English, then the first one
    candidates.sort(key=lambda s: (not s.forced, (s.language or "") not in ("eng", "en"), s.index))
    return candidates[0]


_PURE_MUSIC_LINE = re.compile(r"^[\s♪♫#*+\-—–~]+$")
_INLINE_MUSIC_NOTES = re.compile(r"[♪♫]")
_SDH_SOUND_KEYWORDS = (
    "music", "song", "singing", "lyrics", "applause", "cheer", "cheering",
    "laugh", "laughter", "chuckle", "chuckles", "chuckling", "scoff", "scoffs", "scoffing",
    "giggle", "snicker", "snickers", "groan", "groans", "groaning", "grunt", "grunts", "grunting",
    "sigh", "sighs", "sighing", "gasp", "gasps", "gasping", "pant", "pants", "panting",
    "cough", "coughs", "coughing", "choke", "chokes", "choking", "exclaim", "exclaiming",
    "scream", "screams", "screaming", "yell", "yells", "yelling", "shout", "shouts",
    "whisper", "whispers", "whispering", "murmur", "mutter", "stammer", "stammers",
    "clatter", "clatters", "bang", "thud", "crash", "thunder", "rumbl", "wind", "rain",
    "siren", "alarm", "ring", "ringing", "beep", "buzz", "click", "creak",
    "footstep", "footsteps", "breath", "breathing", "snort", "sob", "sobs", "sobbing", "crying",
    "silence", "inaudible", "indistinct", "muffled", "overlapping", "speaking",
    "over radio", "on radio", "on tv", "on phone", "sound", "sounds", "noise", "engine", "explosion",
)
_SDH_PARENS_SPAN = re.compile(r"\(([^()]{1,70})\)|\[([^\[\]]{1,70})\]")
_SDH_ALL_CAPS_SPEAKER = re.compile(
    r"^(\s*[-–—]?\s*)([A-Z][A-Z0-9\s.'\-/#]{1,30}):(?:\s+(?=\S)|\s*$)"
)
_ACRONYM_SKIP = {"RE", "PS", "NB", "AM", "PM", "TV", "ID", "OK", "US", "UK", "EU", "UN", "FBI", "CIA", "MI5", "MI6", "DNA", "FYI", "HUD", "ETA", "POV"}
_SDH_SPEAKER_LABEL = re.compile(
    r"^(\s*[-–—]?\s*)\[([a-zA-Z0-9\s._'-]{2,25})\]\s*:?\s*(?=\S)",
)


def _is_sdh_stage_direction(inner: str, is_sdh: bool, is_brackets: bool) -> bool:
    """Return True if parenthesized/bracketed text is a non-speech SDH sound/action cue."""
    clean = (inner or "").strip()
    if not clean:
        return True
    low = clean.lower()
    if any(kw in low for kw in _SDH_SOUND_KEYWORDS):
        return True
    # In SDH mode, ALL-CAPS parenthetical descriptions like (SCOFFS), (CHOKING) are non-speech actions;
    # keep bracketed [Title / Date] cues untouched unless they matched an explicit sound keyword above.
    if is_sdh and not is_brackets:
        letters = [ch for ch in clean if ch.isalpha()]
        if letters and all(ch.isupper() for ch in letters) and len(clean.split()) <= 7:
            return True
    return False


def clean_embedded_dialogue_text(
    text: str,
    is_sdh: bool = False,
    strip_speaker_labels: bool = False,
) -> str:
    """Strip formatting tags, music note lines, and SDH noise while preserving dialogue and turns."""
    clean = _TAGS.sub("", text or "").strip()
    if not clean:
        return ""
    raw_lines = [rl.strip() for rl in clean.splitlines() if rl.strip()]
    dropped_any = False
    lines: List[str] = []

    for l in raw_lines:
        if _PURE_MUSIC_LINE.match(l):
            dropped_any = True
            continue
        if not l.strip("♪♫#*~ \t"):
            dropped_any = True
            continue

        def _strip_parens(m: re.Match[str]) -> str:
            paren_txt = m.group(1)
            brack_txt = m.group(2)
            inner = paren_txt if paren_txt is not None else (brack_txt or "")
            if _is_sdh_stage_direction(inner, is_sdh=is_sdh, is_brackets=(brack_txt is not None)):
                return ""
            return m.group(0)

        l = _SDH_PARENS_SPAN.sub(_strip_parens, l)
        l = _SDH_SPEAKER_LABEL.sub(r"\1", l.strip())

        if strip_speaker_labels:
            m_spk = _SDH_ALL_CAPS_SPEAKER.match(l)
            if m_spk and m_spk.group(2).strip().upper() not in _ACRONYM_SKIP:
                l = (m_spk.group(1) or "") + l[m_spk.end():]

        l = _INLINE_MUSIC_NOTES.sub("", l)
        l = re.sub(r"\s{2,}", " ", l).strip()
        if not l or _PURE_MUSIC_LINE.match(l):
            dropped_any = True
            continue
        lines.append(l)

    # Remove orphaned leading "- " when a multi-line turn lost a sound-effect line
    if len(lines) == 1 and dropped_any:
        lines[0] = re.sub(r"^[-–—]\s*", "", lines[0]).strip()

    return "\n".join(l for l in lines if l).strip()


def pick_dialogue_track(
    streams: Sequence[SubtitleStreamInfo],
    source_language: Optional[str] = "auto",
    audio_streams: Sequence[Any] = (),
    selected_audio_stream: int = 0,
    selected_subtitle_stream: Optional[int] = None,
) -> Optional[SubtitleStreamInfo]:
    """
    Find the best embedded text-based full dialogue subtitle track, if any.
    If `selected_subtitle_stream` is explicitly given, return that text-based track directly.
    Otherwise, prefers clean dialogue over SDH, matching source or audio stream language,
    and falls back to any available text-based dialogue track (including foreign languages).
    """
    if selected_subtitle_stream is not None:
        explicit = next(
            (s for s in streams if s.index == selected_subtitle_stream and s.text_based),
            None,
        )
        if explicit is not None:
            return explicit

    candidates = [s for s in streams if s.text_based and not s.looks_forced]
    if not candidates:
        return None

    # Exclude commentary / audio description tracks
    filtered = []
    for s in candidates:
        t = (s.title or "").lower()
        if any(bad in t for bad in ("commentary", "director", "description", "audio desc")):
            continue
        filtered.append(s)
    if not filtered:
        filtered = candidates

    from app.utils.languages import language_code, language_name

    desired_code: Optional[str] = None
    if source_language and source_language.lower() not in ("auto", "none"):
        desired_code = language_code(source_language) or source_language.lower()
    elif audio_streams:
        audio = next((a for a in audio_streams if getattr(a, "index", None) == selected_audio_stream), None)
        if audio and getattr(audio, "language", None) and audio.language.lower() not in ("und", "unknown"):
            desired_code = language_code(audio.language) or audio.language.lower()

    def _lang_match_score(s: SubtitleStreamInfo) -> int:
        s_code = language_code(s.language) or (s.language or "").lower()
        s_title = (s.title or "").lower()
        if desired_code:
            desired_name = language_name(desired_code, "").lower()
            if s_code == desired_code or (desired_name and desired_name in s_title) or desired_code in s_title:
                return 0
            if s_code in ("und", ""):
                return 1
            return 2
        if s_code == "en" or "english" in s_title:
            return 0
        if s_code in ("und", ""):
            return 1
        return 2

    filtered.sort(
        key=lambda s: (
            _lang_match_score(s),
            1 if s.looks_sdh else 0,
            0 if s.default else 1,
            s.index,
        )
    )
    return filtered[0]


_LANG_TITLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "en": ("english", "eng"),
    "ar": ("arabic", "ara", "عربي", "العربية"),
    "ru": ("russian", "rus", "русский"),
    "pl": ("polish", "pol", "polski"),
    "es": ("spanish", "spa", "español", "espanol"),
    "fr": ("french", "fra", "fre", "français", "francais"),
}


def pick_track_by_language(
    streams: Sequence[SubtitleStreamInfo],
    lang_code: str,
) -> Optional[SubtitleStreamInfo]:
    """Return the best embedded text-based dialogue track matching `lang_code`, or None."""
    from app.utils.languages import language_code, language_name

    target_code = (language_code(lang_code) or lang_code or "").strip().lower()
    if not target_code:
        return None
    target_name = language_name(target_code, "").lower()
    aliases = _LANG_TITLE_ALIASES.get(target_code, ())

    matches: List[SubtitleStreamInfo] = []
    for s in streams:
        if not s.text_based or s.looks_forced:
            continue
        title = (s.title or "").lower()
        if any(bad in title for bad in ("commentary", "director", "description", "audio desc")):
            continue
        s_code = (language_code(s.language) or (s.language or "")).strip().lower()
        title_hit = (
            (target_name and target_name in title)
            or any(al in title for al in aliases)
        )
        if s_code == target_code or (s_code in ("und", "unknown", "") and title_hit) or title_hit:
            matches.append(s)

    if not matches:
        return None
    matches.sort(key=lambda s: (1 if s.looks_sdh else 0, 0 if s.default else 1, s.index))
    return matches[0]


_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _ts(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(content: str) -> List[Dict[str, Any]]:
    """Parse SRT text into [{start_time, end_time, text}] (formatting tags removed)."""
    try:
        cues, _ = parse_subtitle_text(content, "track.srt")
    except SubtitleParseError:
        return []
    return [{"start_time": c.start, "end_time": c.end, "text": c.text} for c in cues]


async def extract_track_as_srt(video_path: str, stream: SubtitleStreamInfo) -> str:
    """Extract an embedded subtitle stream as raw UTF-8 SRT text via ffmpeg."""
    process = await asyncio.create_subprocess_exec(
        get_ffmpeg_bin(), "-v", "error", *safe_input(str(Path(video_path).resolve())),
        "-map", f"0:s:{stream.index}", "-f", "srt", "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Could not extract subtitle stream {stream.index}: {stderr.decode(errors='ignore')[-500:]}")
    return stdout.decode("utf-8", errors="ignore").replace("\ufeff", "").strip()


async def extract_forced_track(video_path: str, stream: SubtitleStreamInfo) -> List[Dict[str, Any]]:
    """Convert one embedded subtitle stream to ON_SCREEN segment dicts via ffmpeg."""
    process = await asyncio.create_subprocess_exec(
        get_ffmpeg_bin(), "-v", "error", *safe_input(str(Path(video_path).resolve())),
        "-map", f"0:s:{stream.index}", "-f", "srt", "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Could not extract subtitle stream {stream.index}: {stderr.decode(errors='ignore')[-500:]}")
    return [
        make_onscreen_segment(c["start_time"], c["end_time"], c["text"], bracket=False)
        for c in parse_srt(stdout.decode("utf-8", errors="ignore"))
    ]


async def extract_dialogue_track(
    video_path: str,
    stream: SubtitleStreamInfo,
    duration: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract embedded subtitle stream via ffmpeg into:
    - dialogue_segments: clean dialogue cues
    - onscreen_segments: top-of-screen titles (marked with {\\an8} in SRT/ASS)
    """
    process = await asyncio.create_subprocess_exec(
        get_ffmpeg_bin(), "-v", "error", *safe_input(str(Path(video_path).resolve())),
        "-map", f"0:s:{stream.index}", "-f", "srt", "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Could not extract subtitle stream {stream.index}: {stderr.decode(errors='ignore')[-500:]}")

    content = stdout.decode("utf-8", errors="ignore").replace("\ufeff", "").strip()
    blocks = re.split(r"\r?\n\s*\r?\n", content)

    dialogue: List[Dict[str, Any]] = []
    onscreen: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            match = _SRT_TIME.search(line)
            if not match:
                continue
            g = match.groups()
            start_ts = _ts(*g[:4])
            end_ts = _ts(*g[4:])
            raw_text = "\n".join(lines[i + 1:])
            is_top = "{\\an8}" in raw_text or "\\an8" in raw_text
            cleaned = clean_embedded_dialogue_text(raw_text, is_sdh=stream.looks_sdh)
            if not cleaned:
                continue
            if is_top:
                onscreen.append(make_onscreen_segment(start_ts, end_ts, cleaned, bracket=True))
            else:
                dialogue.append({
                    "index": len(dialogue) + 1,
                    "start_time": round(start_ts, 3),
                    "end_time": round(max(end_ts, start_ts + 0.4), 3),
                    "original_text": cleaned,
                    "translated_text": None,
                    "translation_failed": False,
                    "segment_type": SegmentType.DIALOGUE.value,
                    "confidence": 1.0,
                })
            break

    return dialogue, onscreen
