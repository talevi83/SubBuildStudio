"""Cross-language pair selection, temporal alignment, and quality checks for Dual-Source subtitles."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.utils.subtitle_parser import SubtitleParseError, parse_subtitle_text

_LANG_NORM = {
    "en": "en", "eng": "en", "english": "en",
    "ar": "ar", "ara": "ar", "arabic": "ar",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "pl": "pl", "pol": "pl", "polish": "pl",
    "es": "es", "spa": "es", "spanish": "es",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
}

DEFAULT_SECONDARY_FOR_AR = ("es", "fr", "ru")
DEFAULT_GENDER_FALLBACK_FOR_EN = ("ru", "pl", "es", "fr")


@dataclass
class TimedCue:
    """Single parsed subtitle cue with normalized timing and text."""
    index: int
    start: float
    end: float
    text: str
    top_positioned: bool = False


@dataclass
class AlignedCue:
    """Temporally paired primary + anchor subtitle item ready for refine and translation stages."""
    index: int
    start_time: float
    end_time: float
    original_text: str
    reference_text: Optional[str]
    alignment_kind: str   # "EN_AR" | "EN_ONLY" | "AR_ONLY" | "EN_ANCHOR" | "AR_SECONDARY" | "AR_ONLY_DIALOGUE" | "SINGLE_SOURCE"
    segment_type: str     # "DIALOGUE" | "ON_SCREEN"


def normalize_lang_code(lang: Optional[str]) -> str:
    low = (lang or "").strip().lower()
    return _LANG_NORM.get(low, low)


def _ordered_candidates(defaults: Sequence[str], custom_priority: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    if custom_priority:
        for item in custom_priority:
            norm = normalize_lang_code(item)
            if norm in defaults and norm not in out:
                out.append(norm)
    for d in defaults:
        if d not in out:
            out.append(d)
    return out


def select_subtitle_pair(
    available: Dict[str, Any],
    fallback_priority: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Select the highest-value primary + gender/context anchor pair from available subtitle languages."""
    present: Dict[str, str] = {}
    for raw_k, raw_v in (available or {}).items():
        code = normalize_lang_code(raw_k)
        txt = str(raw_v or "").strip() if raw_v is not None else ""
        if code and txt and code not in present:
            present[code] = txt

    if "en" in present and "ar" in present:
        return {"pair_mode": "en_ar", "primary_lang": "en", "anchor_lang": "ar"}

    if "ar" in present and "en" not in present:
        for sec in _ordered_candidates(DEFAULT_SECONDARY_FOR_AR, fallback_priority):
            if sec in present:
                return {"pair_mode": "ar_secondary", "primary_lang": "ar", "anchor_lang": sec}
        return {"pair_mode": "single_source", "primary_lang": "ar", "anchor_lang": None}

    if "en" in present and "ar" not in present:
        for fb in _ordered_candidates(DEFAULT_GENDER_FALLBACK_FOR_EN, fallback_priority):
            if fb in present:
                return {"pair_mode": "en_gender_fallback", "primary_lang": "en", "anchor_lang": fb}
        return {"pair_mode": "single_source", "primary_lang": "en", "anchor_lang": None}

    if present:
        only_lang = next(iter(present.keys()))
        return {"pair_mode": "single_source", "primary_lang": only_lang, "anchor_lang": None}

    return {"pair_mode": "none", "primary_lang": None, "anchor_lang": None}


def parse_srt_cues(
    srt_text: str,
    *,
    clean_sdh: bool = False,
    strip_speaker_labels: bool = False,
) -> List[TimedCue]:
    """Decode raw SRT text into chronologically ordered TimedCue items."""
    if not srt_text or not srt_text.strip():
        return []
    try:
        raw_cues, _ = parse_subtitle_text(srt_text, file_name="subtitles.srt")
    except SubtitleParseError:
        return []

    if clean_sdh:
        from app.pipeline.embedded_tracks import clean_embedded_dialogue_text

    cues: List[TimedCue] = []
    for idx, cue in enumerate(raw_cues, start=1):
        text = (cue.text or "").strip()
        if clean_sdh:
            text = clean_embedded_dialogue_text(
                text,
                is_sdh=True,
                strip_speaker_labels=strip_speaker_labels,
            )
        if not text:
            continue
        start = round(max(0.0, float(cue.start)), 3)
        end = round(max(start + 0.1, float(cue.end)), 3)
        cues.append(
            TimedCue(
                index=idx,
                start=start,
                end=end,
                text=text,
                top_positioned=(cue.position == "top"),
            )
        )
    cues.sort(key=lambda c: (c.start, c.end, c.index))
    for pos, cue in enumerate(cues, start=1):
        cue.index = pos
    return cues


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return min(a_end, b_end) - max(a_start, b_start)


def _edge_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end >= b_start and b_end >= a_start:
        return 0.0
    return min(abs(a_start - b_end), abs(b_start - a_end))


def _join_texts(parts: Sequence[str]) -> str:
    cleaned = [" ".join((p or "").split()) for p in parts if p and p.strip()]
    return " / ".join(cleaned)


def align_dual_cues(
    primary_cues: Sequence[TimedCue],
    anchor_cues: Sequence[TimedCue] = (),
    *,
    pair_mode: str = "en_ar",
    window_sec: float = 1.5,
) -> Tuple[List[AlignedCue], Dict[str, int]]:
    """Temporally pair primary cues against anchor cues according to `pair_mode`."""
    prim_sorted = sorted(primary_cues, key=lambda c: (c.start, c.end))
    anch_sorted = sorted(anchor_cues, key=lambda c: (c.start, c.end))

    if pair_mode == "single_source" or not anch_sorted:
        single = [
            AlignedCue(
                index=pos,
                start_time=c.start,
                end_time=c.end,
                original_text=c.text,
                reference_text=None,
                alignment_kind="SINGLE_SOURCE",
                segment_type="ON_SCREEN" if c.top_positioned else "DIALOGUE",
            )
            for pos, c in enumerate(prim_sorted, start=1)
        ]
        stats = {
            "en_total": len(prim_sorted),
            "ar_total": 0,
            "primary_total": len(prim_sorted),
            "anchor_total": 0,
            "paired_count": 0,
            "en_only_count": len(prim_sorted),
            "ar_only_count": 0,
        }
        return single, stats

    anch_best_prim: Dict[int, int] = {}
    anch_has_any_overlap: Dict[int, bool] = {}
    for a_idx, anc in enumerate(anch_sorted):
        best_p_idx: Optional[int] = None
        best_score = 0.0
        for p_idx, pri in enumerate(prim_sorted):
            if pri.end <= anc.start:
                continue
            if pri.start >= anc.end:
                break
            ov = _overlap(pri.start, pri.end, anc.start, anc.end)
            if ov > 0.0:
                anch_has_any_overlap[a_idx] = True
                union = max(1e-6, max(pri.end, anc.end) - min(pri.start, anc.start))
                score = ov / union
                if score > best_score:
                    best_score = score
                    best_p_idx = p_idx
        if best_p_idx is not None:
            anch_best_prim[a_idx] = best_p_idx

    prim_to_anch: Dict[int, List[int]] = {i: [] for i in range(len(prim_sorted))}
    matched_anch: set[int] = set()
    for a_idx, p_idx in anch_best_prim.items():
        prim_to_anch[p_idx].append(a_idx)
        matched_anch.add(a_idx)

    for p_idx, pri in enumerate(prim_sorted):
        if prim_to_anch[p_idx]:
            continue
        p_mid = 0.5 * (pri.start + pri.end)
        best_a_idx: Optional[int] = None
        best_dist = window_sec + 1e-6
        for a_idx, anc in enumerate(anch_sorted):
            if a_idx in matched_anch:
                continue
            a_mid = 0.5 * (anc.start + anc.end)
            mid_dist = abs(a_mid - p_mid)
            edge_dist = _edge_distance(pri.start, pri.end, anc.start, anc.end)
            if mid_dist <= window_sec and edge_dist <= window_sec and mid_dist < best_dist:
                best_dist = mid_dist
                best_a_idx = a_idx
        if best_a_idx is not None:
            prim_to_anch[p_idx].append(best_a_idx)
            matched_anch.add(best_a_idx)

    paired_kind = {
        "en_ar": "EN_AR",
        "en_gender_fallback": "EN_ANCHOR",
        "ar_secondary": "AR_SECONDARY",
    }.get(pair_mode, "EN_AR")
    unmatched_prim_kind = "AR_ONLY_DIALOGUE" if pair_mode == "ar_secondary" else "EN_ONLY"

    aligned: List[AlignedCue] = []
    paired_count = 0
    prim_only_count = 0
    for p_idx, pri in enumerate(prim_sorted):
        a_indices = sorted(prim_to_anch[p_idx], key=lambda i: (anch_sorted[i].start, anch_sorted[i].end))
        seg_type = "ON_SCREEN" if pri.top_positioned else "DIALOGUE"
        if a_indices:
            ref_text = _join_texts([anch_sorted[i].text for i in a_indices])
            paired_count += 1
            aligned.append(
                AlignedCue(
                    index=0,
                    start_time=pri.start,
                    end_time=pri.end,
                    original_text=pri.text,
                    reference_text=ref_text or None,
                    alignment_kind=paired_kind,
                    segment_type=seg_type,
                )
            )
        else:
            prim_only_count += 1
            aligned.append(
                AlignedCue(
                    index=0,
                    start_time=pri.start,
                    end_time=pri.end,
                    original_text=pri.text,
                    reference_text=None,
                    alignment_kind=unmatched_prim_kind,
                    segment_type=seg_type,
                )
            )

    # In EN+AR mode, unmatched AR cues outside English timestamps are strong ON_SCREEN title candidates
    ar_only_count = 0
    if pair_mode == "en_ar":
        for a_idx, anc in enumerate(anch_sorted):
            if a_idx in matched_anch or anch_has_any_overlap.get(a_idx):
                continue
            ar_only_count += 1
            aligned.append(
                AlignedCue(
                    index=0,
                    start_time=anc.start,
                    end_time=anc.end,
                    original_text=anc.text,
                    reference_text=anc.text,
                    alignment_kind="AR_ONLY",
                    segment_type="ON_SCREEN",
                )
            )

    aligned.sort(key=lambda c: (c.start_time, c.end_time))
    for pos, item in enumerate(aligned, start=1):
        item.index = pos

    stats: Dict[str, int] = {
        "en_total": len(prim_sorted),
        "ar_total": len(anch_sorted),
        "primary_total": len(prim_sorted),
        "anchor_total": len(anch_sorted),
        "paired_count": paired_count,
        "en_only_count": prim_only_count,
        "ar_only_count": ar_only_count,
    }
    return aligned, stats


def align_en_ar_cues(
    en_cues: Sequence[TimedCue],
    ar_cues: Sequence[TimedCue],
    window_sec: float = 1.5,
) -> Tuple[List[AlignedCue], Dict[str, int]]:
    """Pair EN cues with overlapping or nearby AR cues, flagging unmatched AR cues as ON_SCREEN."""
    aligned, stats = align_dual_cues(en_cues, ar_cues, pair_mode="en_ar", window_sec=window_sec)
    # Preserve concise 5-key stats contract expected by callers
    return aligned, {
        "en_total": stats["en_total"],
        "ar_total": stats["ar_total"],
        "paired_count": stats["paired_count"],
        "en_only_count": stats["en_only_count"],
        "ar_only_count": stats["ar_only_count"],
    }


def evaluate_alignment_quality(
    aligned: Sequence[AlignedCue],
    duration_sec: float,
    stats: Dict[str, int],
    *,
    pair_mode: str,
    primary_synced: bool = True,
) -> Dict[str, Any]:
    """Assess subtitle coverage, pairing confidence, and large uncovered gaps for Smart ASR Fallback."""
    total = len(aligned)
    dur = max(0.0, float(duration_sec or 0.0))
    primary_total = max(1, int(stats.get("primary_total", stats.get("en_total", total)) or 0))
    paired = int(stats.get("paired_count", 0) or 0)
    pair_ratio = round(paired / primary_total, 3) if pair_mode in ("en_ar", "en_gender_fallback", "ar_secondary") else 1.0

    ordered = sorted(aligned, key=lambda c: (c.start_time, c.end_time))
    if ordered and dur > 0.0:
        span_cov = max(0.0, ordered[-1].end_time - ordered[0].start_time) / dur
        coverage_ratio = round(min(1.0, span_cov), 3)
    else:
        coverage_ratio = 1.0 if ordered else 0.0

    max_gap_sec = 0.0
    for i in range(1, len(ordered)):
        gap = max(0.0, ordered[i].start_time - ordered[i - 1].end_time)
        if gap > max_gap_sec:
            max_gap_sec = gap
    max_gap_sec = round(max_gap_sec, 1)

    reason: Optional[str] = None
    needs_full_asr_fallback = False
    if dur >= 30.0 and total < 3:
        needs_full_asr_fallback = True
        reason = f"Only {total} aligned subtitle cue(s) found for a {dur:.0f}s video"
    elif dur >= 120.0 and coverage_ratio < 0.25:
        needs_full_asr_fallback = True
        reason = f"Subtitle span covers only {coverage_ratio:.0%} of {dur:.0f}s video"

    anchor_unreliable = (
        pair_mode in ("en_ar", "en_gender_fallback", "ar_secondary")
        and primary_total >= 5
        and pair_ratio < 0.35
    )
    needs_gap_asr_supplement = (
        not needs_full_asr_fallback
        and dur >= 120.0
        and max_gap_sec >= 90.0
    )
    if needs_gap_asr_supplement and reason is None:
        reason = f"Uncovered speech gap of {max_gap_sec:.0f}s detected inside aligned subtitles"
    elif not primary_synced and reason is None:
        reason = "Subtitle audio sync failed; using unverified timestamps"

    return {
        "ok": not needs_full_asr_fallback and not needs_gap_asr_supplement and not anchor_unreliable,
        "needs_full_asr_fallback": needs_full_asr_fallback,
        "needs_gap_asr_supplement": needs_gap_asr_supplement,
        "anchor_unreliable": anchor_unreliable,
        "coverage_ratio": coverage_ratio,
        "pair_ratio": pair_ratio,
        "max_gap_sec": max_gap_sec,
        "reason": reason,
    }
