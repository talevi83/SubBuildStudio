"""Final Cut Pro XML (FCPXML 1.9) exporter for DaVinci Resolve, Premiere Pro, and Final Cut Pro."""
from fractions import Fraction
import re
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from app.utils.rtl_fixer import apply_rtl_fix
from app.utils.srt_formatter import _get_seg_attr, split_long_lines

_HEX_COLOR_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")

# Canonical FCPXML frameDuration pairs (numerator, denominator)
_KNOWN_FPS: List[Tuple[float, int, int]] = [
    (23.976, 1001, 24000),
    (24.0, 100, 2400),
    (25.0, 100, 2500),
    (29.97, 1001, 30000),
    (30.0, 100, 3000),
    (50.0, 100, 5000),
    (59.94, 1001, 60000),
    (60.0, 100, 6000),
]


def _resolve_frame_rate(fps: float) -> Tuple[int, int]:
    """Return rational (frame_num, frame_den) seconds per frame."""
    safe_fps = float(fps) if fps and fps > 0 else 25.0
    for target, num, den in _KNOWN_FPS:
        if abs(safe_fps - target) < 0.02:
            return num, den
    den = max(100, int(round(safe_fps * 100)))
    return 100, den


def _frames_for_time(seconds: float, frame_dur: Fraction, min_frames: int = 0) -> int:
    frames = int(round(Fraction(max(0.0, float(seconds))) / frame_dur))
    return max(min_frames, frames)


def _format_rational(frames: int, num: int, den: int) -> str:
    if frames <= 0:
        return "0s"
    frac = Fraction(frames * num, den)
    if frac.denominator == 1:
        return f"{frac.numerator}s"
    return f"{frac.numerator}/{frac.denominator}s"


def _hex_to_fcpxml_rgba(hex_color: str, fallback: str = "1 1 1 1") -> str:
    match = _HEX_COLOR_RE.match((hex_color or "").strip())
    if not match:
        return fallback
    raw = match.group(1)
    r = int(raw[0:2], 16) / 255.0
    g = int(raw[2:4], 16) / 255.0
    b = int(raw[4:6], 16) / 255.0
    return f"{r:.4f} {g:.4f} {b:.4f} 1"


def _xml_safe_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text or "")


def serialize_to_fcpxml(
    segments: List[Any],
    *,
    title: str = "SubBuild Studio",
    duration: Optional[float] = None,
    fps: float = 25.0,
    use_translated: bool = False,
    apply_rtl: bool = True,
    max_chars_per_line: int = 42,
    speakers: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Serialize subtitle segments to FCPXML 1.9 with title clips and NLE markers."""
    frame_num, frame_den = _resolve_frame_rate(fps)
    frame_dur = Fraction(frame_num, frame_den)
    frame_dur_attr = f"{frame_num}/{frame_den}s"

    sorted_segs = sorted(segments or [], key=lambda s: (float(_get_seg_attr(s, "start_time", 0.0)), int(_get_seg_attr(s, "index", 0) or 0)))

    max_seg_end = max(
        (float(_get_seg_attr(s, "end_time", 0.0)) for s in sorted_segs),
        default=1.0,
    )
    total_seconds = max(float(duration or 0.0), max_seg_end, 1.0)
    total_frames = _frames_for_time(total_seconds, frame_dur, min_frames=1)
    total_dur_str = _format_rational(total_frames, frame_num, frame_den)

    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name="FFVideoFormat1080p",
        frameDuration=frame_dur_attr,
        width="1920",
        height="1080",
    )
    ET.SubElement(
        resources,
        "effect",
        id="r2",
        name="Basic Title",
        uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",
    )

    clean_title = _xml_safe_text(title).strip() or "SubBuild Studio"
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=clean_title)
    project = ET.SubElement(event, "project", name=clean_title)
    sequence = ET.SubElement(project, "sequence", format="r1", duration=total_dur_str)
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(spine, "gap", name="Gap", offset="0s", duration=total_dur_str)

    speaker_map = speakers if isinstance(speakers, dict) else {}
    prev_speaker: Optional[str] = None
    style_counter = 0

    for seg in sorted_segs:
        orig = str(_get_seg_attr(seg, "original_text", "") or "").strip()
        trans = str(_get_seg_attr(seg, "translated_text", "") or "").strip()
        raw_text = (trans if use_translated else orig) or orig or trans
        if not raw_text:
            continue

        seg_type = str(_get_seg_attr(seg, "segment_type", "DIALOGUE") or "DIALOGUE")
        speaker_id = _get_seg_attr(seg, "speaker", None)
        start_s = float(_get_seg_attr(seg, "start_time", 0.0))
        end_s = max(start_s + 0.04, float(_get_seg_attr(seg, "end_time", start_s + 0.1)))

        start_frames = _frames_for_time(start_s, frame_dur, min_frames=0)
        end_frames = _frames_for_time(end_s, frame_dur, min_frames=start_frames + 1)
        if end_frames <= start_frames:
            end_frames = start_frames + 1
        offset_str = _format_rational(start_frames, frame_num, frame_den)
        dur_str = _format_rational(end_frames - start_frames, frame_num, frame_den)

        formatted = split_long_lines(_xml_safe_text(raw_text), max_chars_per_line=max_chars_per_line)
        if apply_rtl:
            formatted = apply_rtl_fix(formatted)

        style_counter += 1
        style_id = f"ts{style_counter}"
        lane = "2" if seg_type == "ON_SCREEN" else "1"
        clip_name = formatted.splitlines()[0][:40] if formatted else f"Cue {style_counter}"

        sp_info = speaker_map.get(speaker_id) if speaker_id else None
        sp_label = str(sp_info.get("label") or speaker_id).strip() if isinstance(sp_info, dict) else (str(speaker_id) if speaker_id else "")
        sp_color = _hex_to_fcpxml_rgba(sp_info.get("color", ""), "1 1 1 1") if isinstance(sp_info, dict) else "1 1 1 1"
        if seg_type == "ON_SCREEN" and sp_color == "1 1 1 1":
            sp_color = "0.4039 0.9098 0.9765 1"

        title_el = ET.SubElement(
            gap,
            "title",
            ref="r2",
            lane=lane,
            offset=offset_str,
            name=clip_name,
            start=offset_str,
            duration=dur_str,
        )
        text_el = ET.SubElement(title_el, "text")
        ts_el = ET.SubElement(text_el, "text-style", ref=style_id)
        ts_el.text = formatted
        ts_def = ET.SubElement(title_el, "text-style-def", id=style_id)
        ET.SubElement(
            ts_def,
            "text-style",
            font="Heebo",
            fontSize="56",
            fontFace="Regular",
            fontColor=sp_color,
            alignment="center",
        )

        # Timeline marker on speaker switches or special cues (COMMS / ON_SCREEN / FOREIGN)
        marker_parts: List[str] = []
        if seg_type in ("COMMS", "ON_SCREEN", "FOREIGN"):
            marker_parts.append(seg_type)
        if speaker_id and speaker_id != prev_speaker:
            marker_parts.append(sp_label or str(speaker_id))
        if marker_parts:
            ET.SubElement(
                gap,
                "marker",
                start=offset_str,
                duration=_format_rational(1, frame_num, frame_den),
                value=_xml_safe_text(" · ".join(marker_parts)),
            )
        if speaker_id:
            prev_speaker = str(speaker_id)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n'
