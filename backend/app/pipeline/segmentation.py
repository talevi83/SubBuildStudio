"""
Segmentation: turn raw Whisper output into broadcast-style subtitle blocks.

Whisper's own segments are decoding windows, not subtitles: they often end mid-sentence
("Just to" / "come down with the virus"), leave single-word orphans and carry timings that
are far too short to read. This module therefore:

1. Flattens every word of every Whisper segment into one timeline (clamping words whose
   timestamps were stretched across silence).
2. Removes runaway repetition loops (a known Whisper hallucination).
3. Groups words into units. A Whisper boundary only counts when the sentence really ends
   there (punctuation) or there is a pause, so mid-sentence cuts are re-joined. Sentences
   Whisper put in the same segment stay together (it rarely mixes speakers inside one
   segment, so we never merge sentences across segments — that could glue two speakers).
4. Splits long units into blocks with a small dynamic program that prefers sentence and
   clause boundaries and pauses, avoids tiny fragments and keeps blocks balanced.
5. Merges dangling fragments ("But", "Put") into their neighbours.
6. Re-times everything: no overlaps, a minimum gap of ~2 frames, display time extended
   toward the target reading speed, small gaps closed to avoid flicker.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.db_models import SegmentType
from app.pipeline.context import RawSegment
from app.utils.srt_formatter import calculate_cps, format_balanced_lines, split_long_lines
from app.utils.text_rules import (
    BREAK_BEFORE_WORDS,
    WEAK_END_WORDS,
    bare,
    ends_clause,
    ends_sentence,
    has_spoken_content,
)

# Pauses (seconds) that end a sentence even without punctuation
_HARD_PAUSE = 1.2
_SEGMENT_PAUSE = 0.6
# Blocks are chained (gap closed) when the gap to the next one is shorter than this
_CHAIN_GAP = 0.5
# Maximum time a subtitle may appear before the first word is spoken
_MAX_LEAD_IN = 0.3
# A speaker change with less silence than this and no punctuation is a diarization flicker, not a
# turn: people do not hand over mid-phrase without a pause ("Every night I" / "see a lot of devils")
_FALSE_TURN_GAP = 0.25
# An unfinished sentence may continue after a dramatic pause this long ("lot of devils" ... 1.8s ...
# "coming for us.") and still belongs in the same subtitle, within the line and duration limits
_CONTINUATION_MAX_GAP = 4.0
# Across a detected speaker change, only a clearly continuing sentence (lower-case start) may merge
_CROSS_SPEAKER_CONTINUATION_GAP = 1.5
# Overlap (seconds) that marks a repeated boundary word as a chunking duplicate rather than a stutter
_DUPLICATE_OVERLAP = 0.02


@dataclass
class _Tok:
    text: str            # includes the leading space exactly as Whisper produced it
    start: float
    end: float
    prob: float = 1.0
    seg_end: bool = False  # last word of a Whisper segment
    speaker: Optional[str] = None


@dataclass
class _Block:
    toks: List[_Tok]
    start: float = 0.0
    end: float = 0.0

    @property
    def text(self) -> str:
        return _join(self.toks)

    @property
    def speech_start(self) -> float:
        return self.toks[0].start

    @property
    def speech_end(self) -> float:
        return self.toks[-1].end


def _join(toks: List[_Tok]) -> str:
    return "".join(t.text for t in toks).strip()


def _fits_two_lines(text: str, max_line: int) -> bool:
    """True if the text fits on one line, or can be broken at a space into two lines that fit."""
    if len(text) <= max_line:
        return True
    if len(text) > 2 * max_line + 1:
        return False
    words = text.split()
    if len(words) < 2:
        return False  # unspaced scripts: the line splitter cuts anywhere
    left = 0
    for i in range(len(words) - 1):
        left += len(words[i]) + (1 if i else 0)
        right = len(text) - left - 1
        if left <= max_line and right <= max_line:
            return True
        if left > max_line:
            break
    return False


# ---------------------------------------------------------------------------
# 1. Tokens
# ---------------------------------------------------------------------------

def _max_word_duration(word: str) -> float:
    return max(1.0, 0.09 * len(word.strip()) + 0.6)


PAUSE_SPLIT = 1.5  # seconds of silence inside a Whisper segment that end it


def split_at_pauses(raw_segments: List[RawSegment], max_gap: float = PAUSE_SPLIT) -> List[RawSegment]:
    """Split segments at long pauses between their words and order everything by time.

    Batched Whisper decodes VAD-merged chunks, so one segment can hold words from either side of
    a long stretch the VAD removed (0:57 "Let me off the ship" ... 1:24 "Weapons on set!"). Lines
    recovered later inside that stretch must land between the two parts, not after them."""
    out: List[RawSegment] = []
    for raw in raw_segments:
        words = [w for w in (raw.words or []) if w.word and w.word.strip()]
        if len(words) < 2:
            out.append(raw)
            continue
        groups: List[List[Any]] = [[words[0]]]
        for prev, cur in zip(words, words[1:]):
            if float(cur.start) - float(prev.end) > max_gap:
                groups.append([])
            groups[-1].append(cur)
        if len(groups) == 1:
            out.append(raw)
            continue
        for g in groups:
            out.append(RawSegment(
                index=raw.index,
                start_time=float(g[0].start),
                end_time=float(g[-1].end),
                text="".join(w.word for w in g).strip(),
                confidence=raw.confidence,
                words=list(g),
            ))
    out.sort(key=lambda s: s.start_time)
    return out


_TERMINAL_PUNCT = ('.', '?', '!', '\u2026', '."', '?"', '!"', '.\u201d', '?\u201d', '!\u201d', '.\u05f4', '?\u05f4', '!\u05f4')
_ANY_PUNCT = _TERMINAL_PUNCT + (',', ';', ':', ',"', ',\u201d', ',\u05f4')
_ORPHAN_MAX_WORDS = 3
_ORPHAN_LEAD = 0.2  # seconds each reattached word is placed before the next line's first word
_ORPHAN_TAIL_PAD = 0.2  # seconds each reattached ending word extends the previous line
_UNPUNCTUATED_OPENERS = {
    "hey", "hi", "hello", "yo", "well", "look", "listen", "wait", "oh",
    "היי", "שלום", "תשמע", "תשמעי", "תראה", "תראי", "רגע", "חכה", "חכי", "תקשיב", "תקשיבי",
}


def _is_unspaced_script(ch: str) -> bool:
    """True for CJK/Japanese/Korean/Thai scripts where consecutive segments join without a space."""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
        or 0x0E00 <= code <= 0x0E7F
    )


def _orphan_tail(words: List[Any]) -> List[Any]:
    """Trailing words that start a sentence finished elsewhere ("aren't we? Feet" -> ["Feet"]).

    Words after the last sentence-ending punctuation; if that still leaves more than
    _ORPHAN_MAX_WORDS, fall back to the last punctuation of any kind (comma, semicolon):
    "Hey, Commodore, don't" -> ["don't"]. An empty result means the tail is a finished sentence.
    """
    for marks in (_TERMINAL_PUNCT, _ANY_PUNCT):
        tail: List[Any] = []
        for w in reversed(words):
            if w.word.strip().endswith(marks):
                break
            tail.insert(0, w)
        if len(tail) == len(words) and marks is _TERMINAL_PUNCT and any(
            w.word.strip().endswith(_ANY_PUNCT) for w in words
        ):
            continue  # "Hey, Commodore, don't": the commas say only the last word is the orphan
        if 0 < len(tail) <= _ORPHAN_MAX_WORDS:
            return tail
    # Fallback for longer completely unpunctuated Whisper segments that end on a greeting/opener
    # right before a pause (e.g. "let's get to work got a lot to do let's go hey" | "pete just in time")
    if len(words) >= 4 and not any(w.word.strip().endswith(_ANY_PUNCT) for w in words):
        last_bare = bare(words[-1].word)
        if last_bare in _UNPUNCTUATED_OPENERS:
            return [words[-1]]
    return []


def _orphan_head(words: List[Any]) -> List[Any]:
    """Leading 1..3 words that finish a sentence started in the previous segment
    ("officials. I thought..." -> ["officials."], "it tonight. You getting..." -> ["it", "tonight."]).
    """
    head: List[Any] = []
    for w in words[:_ORPHAN_MAX_WORDS]:
        head.append(w)
        if w.word.strip().endswith(_TERMINAL_PUNCT):
            # Only treat as an orphan tail of the previous sentence if another sentence
            # follows in the current segment.
            if len(head) < len(words):
                return head
            return []
    return []


def _different_speakers(a: RawSegment, b: RawSegment) -> bool:
    """True when both segments carry a known speaker and they differ: their words never move across."""
    sa, sb = getattr(a, "speaker", None), getattr(b, "speaker", None)
    return bool(sa and sb and sa != sb)


def reattach_orphan_tails(raw_segments: List[RawSegment], max_gap: float = 4.0) -> List[RawSegment]:
    """Move a 1..3 word sentence ending stranded at the start of segment `b` back to segment `a`.

    When Whisper or VAD splits right before the final word(s) of a sentence ("I really can't make" /
    "it tonight. You getting busy with someone?" or "Military" / "officials. I thought..."), the
    next subtitle begins with the last word of the previous sentence followed by a period and a
    new sentence. Reattach that 1..3 word tail to the previous segment so each sentence stays whole.
    """
    out = [s for s in raw_segments if s.words]
    plain = [s for s in raw_segments if not s.words]
    for a, b in zip(out, out[1:]):
        if not a.words or not b.words:
            continue
        if a.words[-1].word.strip().endswith(_TERMINAL_PUNCT) or _different_speakers(a, b):
            continue
        gap = float(b.words[0].start) - float(a.words[-1].end)
        if gap > max_gap:
            continue
        head = _orphan_head(b.words)
        if not head:
            continue
        a_end = float(a.words[-1].end)
        next_start = float(b.words[len(head)].start)
        for i, w in enumerate(head):
            # If the word was snapped across a pause onto the next sentence, pull its timing
            # back to immediately follow `a` while never crossing `next_start`.
            if float(w.start) - a_end > 0.6:
                w.start = round(min(next_start - 0.05 * (len(head) - i), a_end + 0.05 + _ORPHAN_TAIL_PAD * i), 3)
                w.end = round(min(next_start - 0.02, w.start + _ORPHAN_TAIL_PAD), 3)
            else:
                w.end = round(min(float(w.end), max(float(w.start) + 0.1, next_start - 0.02)), 3)
            a_end = float(w.end)
        a.words = a.words + head
        a.text = "".join(w.word for w in a.words).strip()
        a.end_time = float(a.words[-1].end)
        b.words = b.words[len(head):]
        if b.words:
            b.text = "".join(w.word for w in b.words).strip()
            b.start_time = float(b.words[0].start)
    kept = [s for s in out if s.words] + plain
    kept.sort(key=lambda s: s.start_time)
    return kept


def reattach_orphan_leads(raw_segments: List[RawSegment], min_gap: float = _SEGMENT_PAUSE) -> List[RawSegment]:
    """Move a sentence opener stamped at the tail of the previous line across the pause to its line.

    Batched Whisper's word alignment smears across silences the VAD removed: the first word(s) of
    an utterance get the timestamp of the end of the previous one ("...aren't we? Feet" at 11:40,
    "dry on the east beach." at 12:01). The pause split then leaves that stray word on the wrong
    side of the cut, so the subtitle shows a dangling fragment seconds before its sentence. When a
    line ends mid-sentence and the next line starts across a pause, the unfinished tail is
    moved (and retimed) to open the next line.
    """
    out = [s for s in raw_segments if s.words]
    plain = [s for s in raw_segments if not s.words]
    for a, b in zip(out, out[1:]):
        if not a.words or not b.words:
            continue
        gap = float(b.words[0].start) - float(a.words[-1].end)
        if gap <= min_gap or _different_speakers(a, b):
            continue
        tail = _orphan_tail(a.words)
        if not tail:
            continue
        b0 = float(b.words[0].start)
        floor = float(a.words[-1].end) + 0.01 if len(tail) < len(a.words) else float(a.start_time)
        for i, w in enumerate(tail):
            w.start = max(floor, b0 - _ORPHAN_LEAD * (len(tail) - i))
            w.end = min(b0, w.start + 0.15)
        a.words = a.words[:len(a.words) - len(tail)]
        b.words = tail + b.words
        b.text = "".join(w.word for w in b.words).strip()
        b.start_time = float(b.words[0].start)
        if a.words:
            a.text = "".join(w.word for w in a.words).strip()
            a.end_time = float(a.words[-1].end)
    kept = [s for s in out if s.words] + plain
    kept.sort(key=lambda s: s.start_time)
    return kept


def _tokens_from_raw(raw_segments: List[RawSegment]) -> List[_Tok]:
    toks: List[_Tok] = []
    for raw in raw_segments:
        words = [w for w in (raw.words or []) if w.word and w.word.strip()]
        seg_toks: List[_Tok] = []
        raw_sp = getattr(raw, "speaker", None)
        if words:
            for w in words:
                seg_toks.append(_Tok(w.word, float(w.start), float(w.end), float(w.probability), speaker=getattr(w, "speaker", None) or raw_sp))
        else:
            # No word timestamps: spread the segment's time over its words by length.
            parts = raw.text.split()
            if not parts:
                continue
            total = sum(len(p) + 1 for p in parts)
            span = max(0.1, raw.end_time - raw.start_time)
            t = raw.start_time
            for p in parts:
                d = span * (len(p) + 1) / total
                seg_toks.append(_Tok(" " + p, t, t + d, raw.confidence, speaker=raw_sp))
                t += d

        # Make sure words from consecutive Whisper segments don't get glued together
        if toks and seg_toks and not seg_toks[0].text[:1].isspace():
            prev_last = toks[-1].text[-1:]
            first = seg_toks[0].text[:1]
            if prev_last.isalnum() and first.isalnum() and not _is_unspaced_script(prev_last) and not _is_unspaced_script(first):
                seg_toks[0].text = " " + seg_toks[0].text

        for i, tok in enumerate(seg_toks):
            if tok.end < tok.start:
                tok.end = tok.start
            # Whisper sometimes stretches a word across a following/preceding silence.
            limit = _max_word_duration(tok.text)
            if tok.end - tok.start > limit:
                if i == 0 and len(seg_toks) > 1:
                    tok.start = tok.end - limit  # first word: silence was before it
                else:
                    tok.end = tok.start + limit  # otherwise the silence follows it
        if seg_toks:
            seg_toks[-1].seg_end = True
        toks.extend(seg_toks)

    # Chunked/aligned decoding can emit the boundary word twice with overlapping timestamps
    # ("sending" | "sending"). A real stutter is spoken twice, one after the other, so only an
    # overlapping repeat is dropped.
    deduped: List[_Tok] = []
    for tok in toks:
        if deduped:
            prev = deduped[-1]
            key = bare(tok.text).lower()
            if key and key == bare(prev.text).lower() and tok.start < prev.end - _DUPLICATE_OVERLAP:
                prev.end = max(prev.end, tok.end)
                prev.seg_end = prev.seg_end or tok.seg_end
                continue
        deduped.append(tok)
    toks = deduped

    # Keep the timeline monotonic
    for prev, cur in zip(toks, toks[1:]):
        if cur.start < prev.start:
            cur.start = prev.start
        if cur.end < cur.start:
            cur.end = cur.start
    return toks


# ---------------------------------------------------------------------------
# 2. Repetition loops
# ---------------------------------------------------------------------------

def _collapse_repetitions(toks: List[_Tok]) -> List[_Tok]:
    """Collapse runaway loops: an n-gram repeated back-to-back many times is kept once."""
    if len(toks) < 8:
        return toks
    keys = [bare(t.text) for t in toks]
    out: List[_Tok] = []
    i = 0
    n_total = len(toks)
    while i < n_total:
        collapsed = False
        for n in range(8, 0, -1):
            if i + n * 2 > n_total:
                continue
            gram = keys[i:i + n]
            if not any(gram):
                continue
            reps = 1
            while keys[i + reps * n: i + (reps + 1) * n] == gram:
                reps += 1
            min_reps = 8 if n == 1 else 4
            if reps >= min_reps:
                out.extend(toks[i:i + n])
                i += reps * n
                collapsed = True
                break
        if not collapsed:
            out.append(toks[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# 3. Units
# ---------------------------------------------------------------------------

def _next_sentence_len(toks: List[_Tok], start_idx: int) -> int:
    count = 0
    for j in range(start_idx, len(toks)):
        count += 1
        if ends_sentence(toks[j].text) or toks[j].seg_end:
            break
    return count


def _split_units(toks: List[_Tok]) -> List[List[_Tok]]:
    units: List[List[_Tok]] = []
    cur: List[_Tok] = []
    for i, tok in enumerate(toks):
        cur.append(tok)
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if nxt is None:
            break
        gap = nxt.start - tok.end
        is_paired_exclamation = (
            len(cur) == 1
            and _next_sentence_len(toks, i + 1) == 1
            and gap < _SEGMENT_PAUSE
            and not tok.seg_end
        )
        speaker_switch = bool(
            tok.speaker and nxt.speaker and tok.speaker != nxt.speaker
            and (ends_sentence(tok.text) or ends_clause(tok.text) or gap >= _FALSE_TURN_GAP)
        )
        if (
            speaker_switch
            or gap >= _HARD_PAUSE
            or (tok.seg_end and (ends_sentence(tok.text) or gap >= _SEGMENT_PAUSE))
            or (ends_sentence(tok.text) and not is_paired_exclamation)
        ):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


# ---------------------------------------------------------------------------
# 4. Split long units into blocks
# ---------------------------------------------------------------------------

def _break_cost(toks: List[_Tok], k: int) -> float:
    """Cost of breaking between toks[k] and toks[k+1] (lower is better)."""
    tok, nxt = toks[k], toks[k + 1]
    cost = 6.0
    if ends_sentence(tok.text):
        cost -= 8
    elif ends_clause(tok.text):
        cost -= 5
    cost -= min(4.0, max(0.0, nxt.start - tok.end) * 8)
    if tok.seg_end:
        cost -= 1.5
    if bare(nxt.text) in BREAK_BEFORE_WORDS:
        cost -= 2
    if bare(tok.text) in WEAK_END_WORDS:
        cost += 5
    # Strongly discourage breaking 1 or 2 words before the end of a sentence
    # (which strands the last word(s) + period at the start of the next subtitle).
    if not ends_sentence(tok.text):
        if ends_sentence(nxt.text) and k + 2 < len(toks):
            cost += 7.0
        elif k + 2 < len(toks) and not ends_sentence(nxt.text) and ends_sentence(toks[k + 2].text):
            cost += 4.5
    # Also discourage breaking 1 word after a sentence boundary (stranding the opener at the end).
    if k >= 1 and ends_sentence(toks[k - 1].text) and not ends_sentence(tok.text):
        cost += 6.0
    return cost


def _split_unit(toks: List[_Tok], max_line: int, max_duration: float) -> List[List[_Tok]]:
    n = len(toks)
    max_block_chars = 2 * max_line
    text = _join(toks)
    if _fits_two_lines(text, max_line) and toks[-1].end - toks[0].start <= max_duration:
        return [toks]

    inf = float("inf")
    best = [inf] * (n + 1)
    prev = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            part = toks[i:j]
            part_text = _join(part)
            length = len(part_text)
            dur = part[-1].end - part[0].start
            if j - i > 1 and (length > max_block_chars or dur > max_duration):
                break  # extending further left only makes it longer
            if j - i > 1 and not _fits_two_lines(part_text, max_line):
                continue
            if best[i] == inf:
                continue
            cost = best[i] + 10.0 + 6.0 * (length / max_block_chars) ** 2
            if length < 12:
                cost += 8
            elif length < 20:
                cost += 3
            if dur < 0.7:
                cost += 3
            if i > 0:
                cost += _break_cost(toks, i - 1)
            if cost < best[j]:
                best[j], prev[j] = cost, i

    parts: List[List[_Tok]] = []
    j = n
    while j > 0:
        i = prev[j]
        parts.append(toks[i:j])
        j = i
    parts.reverse()
    return parts


# ---------------------------------------------------------------------------
# 5. Merge dangling fragments & micro-segments
# ---------------------------------------------------------------------------

_MICRO_MERGE_MAX_GAP = 0.4
_SHORT_SENTENCE_CHARS = 20
_MIN_DURATION_BUFFER = 0.05


def _dominant_speaker(toks: List[_Tok]) -> Optional[str]:
    dur_by_sp: Dict[str, float] = {}
    for t in toks:
        if t.speaker:
            dur_by_sp[t.speaker] = dur_by_sp.get(t.speaker, 0.0) + max(0.05, t.end - t.start)
    return max(dur_by_sp.items(), key=lambda kv: kv[1])[0] if dur_by_sp else None


def _has_terminal_sentence_end(text: str) -> bool:
    """True if `text` ends with terminal sentence punctuation (`.`, `?`, `!`), excluding cut-off ellipses."""
    flat = " ".join((text or "").split())
    if not flat:
        return False
    stripped = flat.rstrip("\"'”’׳״)]}")
    if stripped.endswith(("...", "…")):
        return False
    words = flat.split()
    if not words:
        return False
    if ends_sentence(words[-1]):
        last_clean = words[-1].rstrip("\"'”’׳״)]}")
        return not last_clean.endswith(("...", "…"))
    return False


def _has_dialogue_turns(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if any(ln.startswith(("-", "–", "—")) for ln in lines):
        return True
    return " -" in f" {text or ''}"


def _is_bracketed_cue(text: str) -> bool:
    s = (text or "").strip()
    return (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")"))


def _is_fragment(block: _Block) -> bool:
    if _is_bracketed_cue(block.text):
        return False
    words = block.text.split()
    return len(words) <= 2 and not _has_terminal_sentence_end(block.text)


def _continues_sentence(a_text: str, b_text: str) -> bool:
    """True when `b_text` reads as the rest of the unfinished sentence `a_text`.

    `a_text` must end with no sentence or clause punctuation at all, and `b_text` must start in
    lower case or follow a word that cannot end a sentence ("the", "to", "of", ...).
    Scripts without letter case only qualify through the weak-word rule.
    """
    a_words = (a_text or "").split()
    b_flat = (b_text or "").lstrip()
    if not a_words or not b_flat:
        return False
    last = a_words[-1]
    if ends_sentence(last) or ends_clause(last) or last.endswith(("-", "—", "–")):
        return False
    if _is_bracketed_cue(a_text) or _is_bracketed_cue(b_text) or b_flat.startswith(("-", "–", "—")):
        return False
    first = b_flat[0]
    return first.islower() or bare(last).lower() in WEAK_END_WORDS


def _lowercase_continuation(a_text: str, b_text: str) -> bool:
    b_flat = (b_text or "").lstrip()
    return bool(b_flat) and b_flat[0].islower() and _continues_sentence(a_text, b_text)


def _can_merge(a: _Block, b: _Block, max_line: int, max_duration: float, allow_speaker_change: bool = False) -> bool:
    if _is_bracketed_cue(a.text) or _is_bracketed_cue(b.text):
        return False
    sp_a, sp_b = _dominant_speaker(a.toks), _dominant_speaker(b.toks)
    if sp_a and sp_b and sp_a != sp_b and not allow_speaker_change:
        return False
    combined = _join(a.toks + b.toks)
    if _has_dialogue_turns(a.text) or _has_dialogue_turns(b.text):
        return False
    return (
        len(combined) <= 2 * max_line
        and _fits_two_lines(combined, max_line)
        and b.speech_end - a.speech_start <= max_duration
    )


def _should_merge_micro_pair(cur_text: str, nxt_text: str, gap: float, max_gap: float = _MICRO_MERGE_MAX_GAP,
                             short_sentence_chars: int = _SHORT_SENTENCE_CHARS) -> bool:
    """True when consecutive cues are close enough and grammatically eligible to merge."""
    if gap > max_gap + 1e-6 or gap < -0.03:
        return False
    if _is_bracketed_cue(cur_text) or _is_bracketed_cue(nxt_text):
        return False
    cur_flat = " ".join(cur_text.split())
    nxt_flat = " ".join(nxt_text.split())
    if not cur_flat or not nxt_flat:
        return False
    if _has_terminal_sentence_end(cur_flat):
        return len(cur_flat) < short_sentence_chars and len(nxt_flat) < short_sentence_chars
    return True


def _merge_fragments(blocks: List[_Block], max_line: int, max_duration: float) -> List[_Block]:
    changed = True
    while changed:
        changed = False
        out: List[_Block] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is not None:
                gap = nxt.speech_start - cur.speech_end
                continuation = _continues_sentence(cur.text, nxt.text) and gap <= _CONTINUATION_MAX_GAP
                cross_speaker = _lowercase_continuation(cur.text, nxt.text) and gap <= _CROSS_SPEAKER_CONTINUATION_GAP
                if (
                    (_is_fragment(cur) and gap <= 1.0)
                    or _should_merge_micro_pair(cur.text, nxt.text, gap)
                    or continuation
                ) and _can_merge(cur, nxt, max_line, max_duration, allow_speaker_change=cross_speaker):
                    out.append(_Block(cur.toks + nxt.toks))
                    i += 2
                    changed = True
                    continue
            # A dangling fragment that can't join the next block may join the previous one
            if out and _is_fragment(cur) and not _has_terminal_sentence_end(out[-1].text):
                prev_block = out[-1]
                if cur.speech_start - prev_block.speech_end <= 1.0 and _can_merge(prev_block, cur, max_line, max_duration):
                    out[-1] = _Block(prev_block.toks + cur.toks)
                    i += 1
                    changed = True
                    continue
            out.append(cur)
            i += 1
        blocks = out
    return blocks


# ---------------------------------------------------------------------------
# 6. Timing
# ---------------------------------------------------------------------------

def _retime(
    blocks: List[_Block],
    min_duration: float,
    max_duration: float,
    target_cps: float,
    max_cps: float,
    min_gap: float,
) -> None:
    for b in blocks:
        b.start, b.end = b.speech_start, max(b.speech_end, b.speech_start + 0.1)

    # A. No overlaps, at least min_gap between blocks
    for prev, cur in zip(blocks, blocks[1:]):
        if cur.start < prev.end + min_gap:
            trimmed = cur.start - min_gap
            if trimmed - prev.start >= 0.5:
                prev.end = trimmed
            else:
                cur.start = prev.end + min_gap
                cur.end = max(cur.end, cur.start + 0.3)

    n = len(blocks)
    chars = [len(b.text.replace("\n", "")) for b in blocks]

    # B. Extend display time toward the target reading speed / minimum duration
    for i, b in enumerate(blocks):
        desired = min(max_duration, max(min_duration, chars[i] / target_cps))
        if b.end - b.start < desired:
            limit = blocks[i + 1].start - min_gap if i + 1 < n else float("inf")
            b.end = max(b.end, min(b.start + desired, limit))

    # C. Still too fast? Let it appear slightly before the speech starts.
    for i, b in enumerate(blocks):
        dur = b.end - b.start
        need = chars[i] / max_cps - dur
        if need > 0:
            floor = blocks[i - 1].end + min_gap if i > 0 else 0.0
            shift = min(_MAX_LEAD_IN, need, max(0.0, b.start - floor))
            b.start -= shift

    # D. Close small gaps so subtitles don't flicker off and on
    for i in range(n - 1):
        b, nxt = blocks[i], blocks[i + 1]
        gap = nxt.start - b.end
        if min_gap < gap < _CHAIN_GAP and nxt.start - min_gap - b.start <= max_duration:
            b.end = nxt.start - min_gap


# ---------------------------------------------------------------------------
# 7. Post-processing pass on subtitle segment dicts
# ---------------------------------------------------------------------------

def _join_segment_texts(a_text: str, b_text: str) -> str:
    a_flat = " ".join((a_text or "").split())
    b_flat = " ".join((b_text or "").split())
    if not a_flat:
        return b_flat
    if not b_flat:
        return a_flat
    if _is_unspaced_script(a_flat[-1]) and _is_unspaced_script(b_flat[0]):
        return f"{a_flat}{b_flat}"
    return f"{a_flat} {b_flat}"


def merge_micro_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = _MICRO_MERGE_MAX_GAP,
    max_chars_per_line: int = 42,
    short_sentence_chars: int = _SHORT_SENTENCE_CHARS,
    max_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Merge consecutive short spoken segments separated by <= `max_gap` seconds.

    Merges segment `i` and `i+1` when:
      1. Both are spoken cues of the same `segment_type` (never `ON_SCREEN` or dual-speaker turns)
         and do not carry conflicting `speaker` IDs.
      2. The gap between `end_time` of `i` and `start_time` of `i+1` is <= `max_gap` (and, if
         word-level timestamps are present on both, the true speech gap is also <= `max_gap`).
      3. Combined text fits standard two-line subtitle limits (`<= 2 * max_chars_per_line` and
         breakable into 2 lines of `<= max_chars_per_line` each).
      4. Segment `i` does not end with terminal sentence punctuation (`.`, `?`, `!`) unless
         both segments are very short (`< short_sentence_chars` characters each).
    """
    if len(segments) < 2:
        return [dict(s) for s in segments]

    eff_max_dur = settings.max_subtitle_duration if max_duration is None else max_duration
    work = [dict(s) for s in segments]
    changed = True
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(work):
            cur = work[i]
            nxt = work[i + 1] if i + 1 < len(work) else None
            if nxt is not None:
                cur_type = cur.get("segment_type") or SegmentType.DIALOGUE.value
                nxt_type = nxt.get("segment_type") or SegmentType.DIALOGUE.value
                cur_sp = cur.get("speaker")
                nxt_sp = nxt.get("speaker")
                cur_text = cur.get("original_text") or ""
                nxt_text = nxt.get("original_text") or ""
                time_gap = float(nxt["start_time"]) - float(cur["end_time"])
                cur_words = cur.get("words") or []
                nxt_words = nxt.get("words") or []
                speech_gap = (
                    float(nxt_words[0]["s"]) - float(cur_words[-1]["e"])
                    if (cur_words and nxt_words)
                    else time_gap
                )
                combined = _join_segment_texts(cur_text, nxt_text)
                total_dur = float(nxt["end_time"]) - float(cur["start_time"])

                if (
                    cur_type == SegmentType.DIALOGUE.value
                    and nxt_type == SegmentType.DIALOGUE.value
                    and not (cur_sp and nxt_sp and cur_sp != nxt_sp)
                    and not _has_dialogue_turns(cur_text)
                    and not _has_dialogue_turns(nxt_text)
                    and not _is_bracketed_cue(cur_text)
                    and not _is_bracketed_cue(nxt_text)
                    and total_dur <= eff_max_dur
                    and len(combined) <= 2 * max_chars_per_line
                    and _fits_two_lines(combined, max_chars_per_line)
                    and _should_merge_micro_pair(cur_text, nxt_text, time_gap, max_gap, short_sentence_chars)
                    and _should_merge_micro_pair(cur_text, nxt_text, speech_gap, max_gap, short_sentence_chars)
                ):
                    formatted = format_balanced_lines(combined, max_chars_per_line=max_chars_per_line)
                    merged_seg = dict(cur)
                    merged_seg["start_time"] = round(float(cur["start_time"]), 3)
                    merged_seg["end_time"] = round(float(nxt["end_time"]), 3)
                    merged_seg["original_text"] = formatted
                    if cur.get("translated_text") and nxt.get("translated_text"):
                        merged_trans = _join_segment_texts(cur["translated_text"], nxt["translated_text"])
                        merged_seg["translated_text"] = format_balanced_lines(
                            merged_trans, max_chars_per_line=max_chars_per_line
                        )
                    else:
                        merged_seg["translated_text"] = cur.get("translated_text") or nxt.get("translated_text")
                    merged_seg["confidence"] = round(
                        (float(cur.get("confidence", 1.0)) + float(nxt.get("confidence", 1.0))) / 2.0, 2
                    )
                    merged_seg["words"] = (list(cur_words) + list(nxt_words)) if (cur_words and nxt_words) else None
                    merged_seg["speaker"] = cur_sp or nxt_sp
                    dur = max(0.1, merged_seg["end_time"] - merged_seg["start_time"])
                    merged_seg["cps"] = calculate_cps(merged_seg.get("translated_text") or formatted, dur)
                    out.append(merged_seg)
                    i += 2
                    changed = True
                    continue
            out.append(cur)
            i += 1
        work = out

    for idx, seg in enumerate(work, start=1):
        seg["index"] = idx
    return work


def merge_subsecond_cues(
    segments: List[Dict[str, Any]],
    min_duration: Optional[float] = None,
    max_gap: float = 0.35,
    max_chars_per_line: int = 42,
    max_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Merge sub-1.0s spoken dialogue fragments into an adjacent cue when `gap <= max_gap`.

    Prevents single-word micro-splits (e.g. 0.7s `'לך'` / `'אל אביך.'` or `'תסתכל'` / `'על'` / `'לי!'`)
    while formatting merged text cleanly with `\\n` inside a single subtitle block.
    """
    if len(segments) < 2:
        return [dict(s) for s in segments]

    eff_min_dur = settings.min_subtitle_duration if min_duration is None else min_duration
    eff_max_dur = settings.max_subtitle_duration if max_duration is None else max_duration
    work = [dict(s) for s in segments]

    def _can_pair_merge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_type = a.get("segment_type") or SegmentType.DIALOGUE.value
        b_type = b.get("segment_type") or SegmentType.DIALOGUE.value
        if a_type != SegmentType.DIALOGUE.value or b_type != SegmentType.DIALOGUE.value:
            return False
        a_orig = a.get("original_text") or ""
        b_orig = b.get("original_text") or ""
        a_trans = a.get("translated_text") or ""
        b_trans = b.get("translated_text") or ""
        a_active = a_trans or a_orig
        b_active = b_trans or b_orig
        if not a_active.strip() or not b_active.strip():
            return False
        if _is_bracketed_cue(a_active) or _is_bracketed_cue(b_active) or _is_bracketed_cue(a_orig) or _is_bracketed_cue(b_orig):
            return False
        if _has_dialogue_turns(a_active) or _has_dialogue_turns(b_active):
            return False

        time_gap = float(b["start_time"]) - float(a["end_time"])
        if time_gap > max_gap + 1e-6 or time_gap < -0.03:
            return False
        a_words = a.get("words") or []
        b_words = b.get("words") or []
        if a_words and b_words:
            speech_gap = float(b_words[0]["s"]) - float(a_words[-1]["e"])
            if speech_gap > max_gap + 1e-6 or speech_gap < -0.03:
                return False

        a_dur = float(a["end_time"]) - float(a["start_time"])
        b_dur = float(b["end_time"]) - float(b["start_time"])
        if a_dur >= eff_min_dur and b_dur >= eff_min_dur:
            return False
        if float(b["end_time"]) - float(a["start_time"]) > eff_max_dur:
            return False

        a_flat = " ".join(a_active.split())
        b_flat = " ".join(b_active.split())
        a_sp, b_sp = a.get("speaker"), b.get("speaker")
        if a_sp and b_sp and a_sp != b_sp:
            # Allow merging across a speaker flip ONLY when one side is a 1-word unpunctuated mid-phrase fragment
            one_word_midphrase = (
                (len(a_flat.split()) == 1 and not _has_terminal_sentence_end(a_flat))
                or (len(b_flat.split()) == 1 and not _has_terminal_sentence_end(b_flat))
            ) and time_gap <= 0.25
            if not one_word_midphrase:
                return False

        # Do not glue two independent longer sentences together if the first already ended a full sentence
        if _has_terminal_sentence_end(a_flat) and not (len(a_flat) < _SHORT_SENTENCE_CHARS and len(b_flat) < _SHORT_SENTENCE_CHARS):
            return False

        if a_orig and b_orig:
            comb_orig = _join_segment_texts(a_orig, b_orig)
            if len(comb_orig) > 2 * max_chars_per_line or not _fits_two_lines(comb_orig, max_chars_per_line):
                return False
        if a_trans and b_trans:
            comb_trans = _join_segment_texts(a_trans, b_trans)
            if len(comb_trans) > 2 * max_chars_per_line or not _fits_two_lines(comb_trans, max_chars_per_line):
                return False
        return True

    def _do_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(a)
        merged.pop("id", None)  # Force DB segment refresh when merged after initial persistence
        merged["start_time"] = round(float(a["start_time"]), 3)
        merged["end_time"] = round(float(b["end_time"]), 3)
        comb_orig = _join_segment_texts(a.get("original_text") or "", b.get("original_text") or "")
        merged["original_text"] = format_balanced_lines(comb_orig, max_chars_per_line=max_chars_per_line)
        if a.get("translated_text") and b.get("translated_text"):
            comb_trans = _join_segment_texts(a["translated_text"], b["translated_text"])
            merged["translated_text"] = format_balanced_lines(comb_trans, max_chars_per_line=max_chars_per_line)
        else:
            raw_tr = a.get("translated_text") or b.get("translated_text")
            merged["translated_text"] = (
                format_balanced_lines(raw_tr, max_chars_per_line=max_chars_per_line) if raw_tr else None
            )
        merged["translation_failed"] = bool(a.get("translation_failed") and b.get("translation_failed"))
        a_w, b_w = a.get("words") or [], b.get("words") or []
        merged["words"] = (list(a_w) + list(b_w)) if (a_w and b_w) else None
        a_dur = max(0.05, float(a["end_time"]) - float(a["start_time"]))
        b_dur = max(0.05, float(b["end_time"]) - float(b["start_time"]))
        merged["speaker"] = (a.get("speaker") if a_dur >= b_dur else b.get("speaker")) or a.get("speaker") or b.get("speaker")
        dur = max(0.1, merged["end_time"] - merged["start_time"])
        merged["cps"] = calculate_cps(merged.get("translated_text") or merged["original_text"], dur)
        return merged

    changed = True
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(work):
            cur = work[i]
            nxt = work[i + 1] if i + 1 < len(work) else None
            if nxt is not None and _can_pair_merge(cur, nxt):
                out.append(_do_merge(cur, nxt))
                i += 2
                changed = True
                continue
            if out and (float(cur["end_time"]) - float(cur["start_time"]) < eff_min_dur) and _can_pair_merge(out[-1], cur):
                out[-1] = _do_merge(out[-1], cur)
                i += 1
                changed = True
                continue
            out.append(cur)
            i += 1
        work = out

    for idx, seg in enumerate(work, start=1):
        seg["index"] = idx
    return work


def enforce_min_duration(
    segments: List[Dict[str, Any]],
    min_duration: Optional[float] = None,
    buffer_sec: float = _MIN_DURATION_BUFFER,
    max_chars_per_line: int = 42,
) -> List[Dict[str, Any]]:
    """Ensure every spoken subtitle lasts at least `min_duration` (up to next start - `buffer_sec`)
    and format lines cleanly into at most 2 balanced lines."""
    if not segments:
        return []

    eff_min_dur = settings.min_subtitle_duration if min_duration is None else min_duration
    result = [dict(s) for s in segments]

    spoken_indices = [
        idx for idx, s in enumerate(result)
        if (s.get("segment_type") or SegmentType.DIALOGUE.value) != SegmentType.ON_SCREEN.value
    ]

    for pos, seg_idx in enumerate(spoken_indices):
        seg = result[seg_idx]
        orig = seg.get("original_text") or ""
        if orig:
            seg["original_text"] = format_balanced_lines(orig, max_chars_per_line=max_chars_per_line)
        trans = seg.get("translated_text")
        if trans:
            seg["translated_text"] = format_balanced_lines(trans, max_chars_per_line=max_chars_per_line)

        start = float(seg["start_time"])
        end = float(seg["end_time"])
        if end - start < eff_min_dur:
            ceiling = (
                float(result[spoken_indices[pos + 1]]["start_time"]) - buffer_sec
                if pos + 1 < len(spoken_indices)
                else float("inf")
            )
            if ceiling > end:
                end = min(start + eff_min_dur, ceiling)
            # If still shorter than eff_min_dur and there is room before start_time, allow a short lead-in
            if end - start < eff_min_dur:
                floor = (
                    float(result[spoken_indices[pos - 1]]["end_time"]) + buffer_sec
                    if pos > 0
                    else 0.0
                )
                needed = eff_min_dur - (end - start)
                lead = min(_MAX_LEAD_IN, needed, max(0.0, start - floor))
                if lead > 0:
                    start -= lead

        seg["start_time"] = round(max(0.0, start), 3)
        seg["end_time"] = round(max(seg["start_time"] + 0.1, end), 3)
        text_for_cps = seg.get("translated_text") or seg.get("original_text") or ""
        seg["cps"] = calculate_cps(text_for_cps, max(0.1, seg["end_time"] - seg["start_time"]))

    for idx, seg in enumerate(result, start=1):
        seg["index"] = idx
    return result


def postprocess_subtitle_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = _MICRO_MERGE_MAX_GAP,
    min_duration: Optional[float] = None,
    buffer_sec: float = _MIN_DURATION_BUFFER,
    max_chars_per_line: int = 42,
    short_sentence_chars: int = _SHORT_SENTENCE_CHARS,
    max_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run micro-segment & sub-second merging followed by minimum duration enforcement and 2-line balancing."""
    merged = merge_micro_segments(
        segments,
        max_gap=max_gap,
        max_chars_per_line=max_chars_per_line,
        short_sentence_chars=short_sentence_chars,
        max_duration=max_duration,
    )
    merged = merge_subsecond_cues(
        merged,
        min_duration=min_duration,
        max_gap=min(max_gap, 0.35),
        max_chars_per_line=max_chars_per_line,
        max_duration=max_duration,
    )
    return enforce_min_duration(
        merged,
        min_duration=min_duration,
        buffer_sec=buffer_sec,
        max_chars_per_line=max_chars_per_line,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_raw_segments(
    raw_segments: List[RawSegment],
    max_chars_per_line: int = 42,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    target_cps: Optional[float] = None,
    max_cps: Optional[float] = None,
    min_gap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Turn raw Whisper segments into refined subtitle dicts (see module docstring)."""
    if not raw_segments:
        return []

    min_duration = settings.min_subtitle_duration if min_duration is None else min_duration
    max_duration = settings.max_subtitle_duration if max_duration is None else max_duration
    target_cps = settings.target_cps if target_cps is None else target_cps
    max_cps = settings.max_cps_warning if max_cps is None else max_cps
    min_gap = settings.min_gap_seconds if min_gap is None else min_gap
    prepared = reattach_orphan_tails(reattach_orphan_leads(split_at_pauses(raw_segments)))
    toks = _collapse_repetitions(_tokens_from_raw(prepared))
    # A standalone "." / "..." token aligned to a breath would otherwise glue onto real words
    # and stretch the neighbouring cue across silence.
    toks = [t for t in toks if has_spoken_content(t.text)] if any(has_spoken_content(t.text) for t in toks) else []
    if not toks:
        return []

    blocks: List[_Block] = []
    for unit in _split_units(toks):
        for part in _split_unit(unit, max_chars_per_line, max_duration):
            # A block of bare punctuation (".", ". .", "...") is an ASR breath/noise artifact
            if has_spoken_content(_join(part)):
                blocks.append(_Block(part))

    blocks = _merge_fragments(blocks, max_chars_per_line, max_duration)
    _retime(blocks, min_duration, max_duration, target_cps, max_cps, min_gap)

    refined: List[Dict[str, Any]] = []
    for b in blocks:
        text = format_balanced_lines(b.text, max_chars_per_line=max_chars_per_line)
        start, end = round(max(0.0, b.start), 3), round(b.end, 3)
        words_payload = [
            {"w": t.text.strip(), "s": round(max(0.0, t.start), 3), "e": round(max(t.start, t.end), 3)}
            for t in b.toks
            if t.text.strip()
        ]
        dominant_sp = _dominant_speaker(b.toks)
        refined.append({
            "index": len(refined) + 1,
            "start_time": start,
            "end_time": end,
            "original_text": text,
            "translated_text": None,
            "translation_failed": False,
            "segment_type": SegmentType.DIALOGUE.value,
            "confidence": round(sum(t.prob for t in b.toks) / len(b.toks), 2),
            "cps": calculate_cps(text, max(0.1, end - start)),
            "words": words_payload or None,
            "speaker": dominant_sp,
        })

    refined = merge_micro_segments(
        refined,
        max_gap=_MICRO_MERGE_MAX_GAP,
        max_chars_per_line=max_chars_per_line,
        max_duration=max_duration,
    )
    from app.utils.speaker_split import split_dual_speaker_segments

    split_segs = split_dual_speaker_segments(refined, max_chars_per_line=max_chars_per_line, min_gap=min_gap)
    return enforce_min_duration(
        split_segs,
        min_duration=min_duration,
        buffer_sec=max(_MIN_DURATION_BUFFER, min_gap),
        max_chars_per_line=max_chars_per_line,
    )
