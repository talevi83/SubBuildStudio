"""
Recover speech the voice-activity detector (Silero VAD) skipped.

Silero VAD is reliable on clean dialogue, but on TV/film audio it misses speech under gunfire,
engines and music (shouted commands in action scenes) and band-limited radio voices. Whisper
never hears what the VAD removed, so those lines are simply missing from the subtitles.

The recovery pass re-runs Whisper *without* VAD, only over the stretches of audio no subtitle
covers (faster-whisper `clip_timestamps`), and keeps a decoded line only if it passes strict
hallucination checks — decoding music or effects without a VAD is exactly when Whisper invents
text ("Thank you for watching", repeated words, ...).
"""
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MIN_GAP = 1.2            # seconds of uncovered audio worth re-listening to
EDGE = 0.15              # keep away from the neighbours' first/last words
MIN_LEVEL_DB = -42.0     # quieter than this (RMS, dBFS) is treated as silence
MAX_CLIP = 28.0          # split long gaps so each clip fits one Whisper window
MAX_WORD_COVER = 1.2     # clamp single words stretched across silence so they don't hide gaps

# Strict acceptance thresholds for recovered lines (decoded without VAD).
# Require `no_speech_prob < 0.30`, `avg_logprob > -0.55`, and `mean_word_prob > 0.65` so orchestral
# music, crowd noise, and broadcast silence cannot slip hallucinated dialogue into the transcript.
MIN_AVG_LOGPROB = -0.55
MAX_NO_SPEECH = 0.30
MAX_COMPRESSION = 2.4
MIN_WORD_PROB = 0.65
MIN_DURATION = 0.2
# Short lines ("Okay.", "Marcy, please!") are where hallucinations hide: held to an even tighter bar
SHORT_WORDS = 3
SHORT_MIN_AVG_LOGPROB = -0.50
SHORT_MIN_WORD_PROB = 0.68
SHORT_MAX_NO_SPEECH = 0.25
# A line stretched over far more audio than its words need is Whisper bridging silence
STRETCH_PER_WORD = 2.0
STRETCH_SLACK = 2.0

# Shouted commands in action scenes: one or two words, usually over gunfire, often repeated
# ("Fire! ... Fire! ... Fire!"). Word timestamps for a shout can be under 0.2s, so duration is
# allowed below MIN_DURATION while still enforcing strict speech & logprob bounds.
COMMAND_WORDS = {
    "fire", "move", "go", "clear", "stop", "hold", "down", "contact", "grenade", "ready", "aim",
    "cover", "reload", "moving", "set", "freeze", "breach", "push", "back", "run", "duck", "incoming",
    "sniper", "medic", "wait", "now", "out", "no", "hey", "watch", "look", "whoa", "wow",
}
COMMAND_PHRASES = {
    "get down", "man down", "cease fire", "hold fire", "fire at will", "frag out", "open fire",
    "hands up", "drop it", "fall back", "take cover", "head shot", "on me", "move out", "let's go",
    "cover me", "moving up", "go go go", "move move", "stop stop", "fire fire",
}
COMMAND_MIN_WORD_PROB = 0.65
COMMAND_MAX_NO_SPEECH = 0.30
COMMAND_MIN_AVG_LOGPROB = -0.55

# A prompt/hotword list echoed back as "speech" (seen with radio hotwords on silence)
_PROMPT_WORDS = {"over", "copy", "roger", "wilco", "sitrep", "klicks", "callsign", "actual", "say", "again", "radio", "check"}
_PROMPT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "is", "are", "was", "were", "it", "this", "that", "as", "be",
}

# Whisper's well-known inventions on non-speech audio (normalized, lower case, no punctuation)
_HALLUCINATIONS = {
    "thank you for watching", "thanks for watching", "thank you so much for watching",
    "please subscribe", "subscribe to my channel", "like and subscribe",
    "subtitles by the amara org community", "transcription by castingwords",
    "you", "bye", "the end",
    "thank you", "thanks", "thank you very much", "thank you so much",
    "we'll be right back", "be right back", "we'll see you next time", "i'll see you next time",
    "see you next time", "see you in the next video", "see you in the next one",
    "all rights reserved",
}
# Broadcast/watermark hallucination patterns safe to block in BOTH primary and gap-recovery passes.
# These are specific enough to never occur in real dialogue, so they may match anywhere in a segment.
_WATERMARK_PATTERNS = [
    re.compile(r"[©®™]"),
    re.compile(r"\bbf[\s\-]*watch\b", re.IGNORECASE),
    re.compile(
        r"\b(?:amara\.org|opensubtitles|subscene|addic7ed|podnapisi|castingwords|yify|yts\.mx)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:www\.[a-z0-9\-]+\.[a-z]{2,}|[a-z0-9\-]+\.(?:org|net)/)\b", re.IGNORECASE),
]
# Credit lines ("Subtitles by ...", "Synced and corrected by ...", "Copyright 2021"). They are only a
# watermark when the WHOLE short segment is the credit line: the same verbs appear in real dialogue
# ("The cord not yet cut, they ripped from me.", "The letter was transcribed by a monk").
_CREDIT_LINE_RE = re.compile(
    r"^\W*(?:"
    r"(?:subtitles?|captions?|subtitled|captioned|transcript(?:ion)?|transcribed|translated|"
    r"synced|synchro(?:nized)?|resynced|corrected|ripped|encoded)"
    r"(?:\s+and\s+(?:synced|corrected|edited|encoded|timed))?"
    r"\s+by\b"
    r"|copyright\b"
    r"|(?:bf[\s\-]*watch\s+)?tv\s*(?:19|20)\d{2}\W*$"
    r")",
    re.IGNORECASE,
)
_CREDIT_LINE_MAX_WORDS = 8
# YouTube-style outros. Real dialogue can say "Thank you for watching" too, so these are only treated
# as hallucinations near the very end of the media (or when the position is unknown, e.g. a lone
# gap-recovery clip judged on its own).
_OUTRO_PHRASES = frozenset({
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
})
# Phrases that are pure subtitle-community credits: never dialogue, so position does not matter
_CREDIT_PHRASES = frozenset({
    "subtitles by the amara org community",
    "transcription by castingwords",
})
OUTRO_TAIL_MIN_SECONDS = 180.0   # the outro window is at least the last 3 minutes ...
OUTRO_TAIL_FRACTION = 0.04       # ... or the last 4% of the media, whichever is longer ...
OUTRO_TAIL_MAX_SECONDS = 420.0   # ... but never more than the last 7 minutes
_HALLUCINATION_PATTERNS = [
    re.compile(r"\b(subtitles?|captions?|transcri\w+|amara|opensubtitles|subscene|addic7ed|subscribe|copyright)\b"),
    re.compile(r"\b(bf\s*watch|watch\s*tv|tv\s*(?:19|20)\d{2})\b"),
    re.compile(r"\b(link in|in the description|description below|check out my|my channel|next video)\b"),
]
_MUSIC = re.compile(r"[♪♫]")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def in_outro_window(start: Optional[float], media_duration: Optional[float]) -> bool:
    """True when `start` lies inside the final minutes of the media (where outro artifacts live).

    Unknown position or duration counts as inside, so callers without timing keep the old behaviour.
    """
    if start is None or not media_duration or media_duration <= 0:
        return True
    tail = min(OUTRO_TAIL_MAX_SECONDS, max(OUTRO_TAIL_MIN_SECONDS, OUTRO_TAIL_FRACTION * float(media_duration)))
    return float(start) >= float(media_duration) - tail


def is_watermark_hallucination(
    text: str,
    start: Optional[float] = None,
    media_duration: Optional[float] = None,
) -> bool:
    """True when `text` is a known broadcast/subtitle watermark or a YouTube outro artifact.

    Rules, from strictest to loosest:
    * unmistakable watermark tokens (``©``, subtitle-site names, URLs) match anywhere;
    * credit lines ("Subtitles by ...", "Copyright 2021") must be the whole, short segment;
    * outro phrases ("Thanks for watching") must be the whole segment AND sit in the final
      minutes of the media when `start` and `media_duration` are given.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if any(p.search(raw) for p in _WATERMARK_PATTERNS):
        return True
    norm = normalize(raw)
    if norm in _CREDIT_PHRASES:
        return True
    if _CREDIT_LINE_RE.search(raw) and len(norm.split()) <= _CREDIT_LINE_MAX_WORDS:
        return True
    return norm in _OUTRO_PHRASES and in_outro_window(start, media_duration)


def find_gaps(
    covered: Iterable[Tuple[float, float]],
    duration: float,
    exclude: Iterable[Tuple[float, float]] = (),
    min_gap: float = MIN_GAP,
    edge: float = EDGE,
) -> List[Tuple[float, float]]:
    """Uncovered stretches of [0, duration] at least `min_gap` long, outside `exclude`."""
    busy = sorted([(s, e) for s, e in covered] + [(s, e) for s, e in exclude])
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in busy:
        if s - cursor >= min_gap + 2 * edge:
            gaps.append((round(cursor + (edge if cursor > 0 else 0.0), 3), round(s - edge, 3)))
        cursor = max(cursor, e)
    if duration - cursor >= min_gap + edge:
        gaps.append((round(cursor + (edge if cursor > 0 else 0.0), 3), round(duration, 3)))
    return gaps


def split_long(gaps: Sequence[Tuple[float, float]], max_len: float = MAX_CLIP) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in gaps:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def level_db(audio: np.ndarray, start: float, end: float, sample_rate: int = 16000) -> float:
    piece = audio[int(start * sample_rate):int(end * sample_rate)]
    if piece.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(piece.astype(np.float64)))))
    return 20 * np.log10(max(rms, 1e-6))


def is_command(norm: str) -> bool:
    """A normalized shouted command ("fire", "go go go", "cease fire")."""
    words = norm.split()
    if not words or len(words) > 3:
        return False
    if norm in COMMAND_PHRASES:
        return True
    return all(w in COMMAND_WORDS for w in words)


def rejection_reason(
    text: str,
    avg_logprob: Optional[float],
    no_speech_prob: Optional[float],
    compression_ratio: Optional[float],
    word_probs: Sequence[float],
    duration: float,
    neighbours: Sequence[str] = (),
    prompt_text: Optional[str] = None,
) -> Optional[str]:
    """None if a recovered line looks like real speech, else a short reason (for the job log)."""
    norm = normalize(text)
    if _MUSIC.search(text):
        return "known hallucination (music notes)"
    if is_watermark_hallucination(text):
        return "known hallucination (watermark)"
    if not norm:
        return "empty"
    if norm in _HALLUCINATIONS or any(p.search(norm) for p in _HALLUCINATION_PATTERNS):
        return "known hallucination"
    if len(set(norm.split()) & _PROMPT_WORDS) >= 4:
        return "echo of the prompt words"
    if prompt_text:
        prompt_norm = normalize(prompt_text)
        prompt_vocab = {w for w in prompt_norm.split() if len(w) >= 3 and w not in _PROMPT_STOPWORDS}
        line_vocab = [w for w in norm.split() if len(w) >= 3 and w not in _PROMPT_STOPWORDS]
        if prompt_norm and norm == prompt_norm:
            return "echo of the prompt words"
        if len(line_vocab) >= 4 and prompt_vocab and sum(1 for w in line_vocab if w in prompt_vocab) >= max(4, int(0.8 * len(line_vocab))):
            return "echo of the prompt words"
    words = norm.split()
    mean_word_prob = float(np.mean(word_probs)) if word_probs else None
    if duration > STRETCH_PER_WORD * len(words) + STRETCH_SLACK:
        return f"stretched over {duration:.1f}s for {len(words)} word(s)"

    if is_command(norm):
        # Shouts may be shorter than MIN_DURATION and repeat a neighbour, yet must pass strict speech gates
        if avg_logprob is not None and avg_logprob <= COMMAND_MIN_AVG_LOGPROB:
            return f"low confidence (logprob {avg_logprob:.2f})"
        if no_speech_prob is not None and no_speech_prob >= COMMAND_MAX_NO_SPEECH:
            return f"probably not speech (no_speech {no_speech_prob:.2f})"
        if mean_word_prob is not None and mean_word_prob <= COMMAND_MIN_WORD_PROB:
            return f"unsure words (mean {mean_word_prob:.2f})"
        return None

    if duration < MIN_DURATION:
        return "too short"
    short = len(words) <= SHORT_WORDS
    min_logprob = SHORT_MIN_AVG_LOGPROB if short else MIN_AVG_LOGPROB
    max_no_speech = SHORT_MAX_NO_SPEECH if short else MAX_NO_SPEECH
    min_word_prob = SHORT_MIN_WORD_PROB if short else MIN_WORD_PROB
    if avg_logprob is not None and avg_logprob <= min_logprob:
        return f"low confidence (logprob {avg_logprob:.2f})"
    if no_speech_prob is not None and no_speech_prob >= max_no_speech:
        return f"probably not speech (no_speech {no_speech_prob:.2f})"
    if compression_ratio is not None and compression_ratio > MAX_COMPRESSION:
        return f"repetitive (compression {compression_ratio:.1f})"
    if mean_word_prob is not None and mean_word_prob <= min_word_prob:
        return f"unsure words (mean {mean_word_prob:.2f})"
    # "Go, go, go!" / "It's okay, it's okay" are real; hallucinated loops run much longer
    if len(words) >= 8 and len(set(words)) <= len(words) // 4:
        return "repeated words"
    if norm in {normalize(n) for n in neighbours}:
        return "duplicate of a neighbouring line"
    return None


def covered_intervals(segments: Sequence, join: float = 0.3) -> List[Tuple[float, float]]:
    """Where speech was actually recognized. Word timestamps are used when available: a batched
    Whisper segment can span audio the VAD removed in the middle of it (e.g. 0:57-1:25 around a
    shooting range full of shouted commands), and the segment bounds would hide that gap.
    Single words whose timestamps were stretched across silence by Whisper are clamped so they
    cannot mask an uncovered speech gap."""
    spans: List[Tuple[float, float]] = []
    for seg in segments:
        words = [w for w in (getattr(seg, "words", None) or []) if w.end > w.start]
        if words:
            for i, w in enumerate(words):
                ws, we = float(w.start), float(w.end)
                if we - ws > MAX_WORD_COVER:
                    if i == 0 and len(words) > 1:
                        ws = we - MAX_WORD_COVER
                    else:
                        we = ws + MAX_WORD_COVER
                spans.append((ws, we))
        else:
            spans.append((float(seg.start_time), float(seg.end_time)))
    spans.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + join:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def neighbours_of(start: float, end: float, segments: Sequence[Dict], window: float = 6.0) -> List[str]:
    return [s["text"] for s in segments if s["end"] >= start - window and s["start"] <= end + window]
