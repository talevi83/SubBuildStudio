"""Audio and cross-track synchronization helpers for Dual-Source subtitles."""
import importlib.util
import logging
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.utils.subtitle_parser import SubtitleParseError, parse_subtitle_text

logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(
    r"(?:offset\s*seconds|shifted\s*(?:subtitles\s*)?by)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"(?:framerate\s*scale\s*factor|scale\s*factor)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_CANDIDATE_SCALES: Tuple[float, ...] = (
    1.0,
    25.0 / 23.976,
    23.976 / 25.0,
    25.0 / 24.0,
    24.0 / 25.0,
)


def detect_sync_tools() -> Dict[str, bool]:
    """Check whether local CLI audio-subtitle sync engines are installed."""
    ffsubsync_ok = bool(shutil.which("ffsubsync") or importlib.util.find_spec("ffsubsync"))
    alass_ok = bool(shutil.which("alass") or shutil.which("alass-cli"))
    return {"ffsubsync": ffsubsync_ok, "alass": alass_ok}


def _ffsubsync_command(reference_path: Path, srt_path: Path, output_path: Path) -> Optional[List[str]]:
    exe = shutil.which("ffsubsync")
    if exe:
        return [exe, str(reference_path), "--srtin", str(srt_path), "--srtout", str(output_path)]
    if importlib.util.find_spec("ffsubsync") is not None:
        return [
            sys.executable,
            "-m",
            "ffsubsync",
            str(reference_path),
            "--srtin",
            str(srt_path),
            "--srtout",
            str(output_path),
        ]
    return None


def _alass_command(reference_path: Path, srt_path: Path, output_path: Path) -> Optional[List[str]]:
    exe = shutil.which("alass") or shutil.which("alass-cli")
    if exe:
        return [exe, str(reference_path), str(srt_path), str(output_path)]
    return None


def _parse_sync_metrics(output: str) -> Dict[str, Optional[float]]:
    offset_match = _OFFSET_RE.search(output or "")
    scale_match = _SCALE_RE.search(output or "")
    return {
        "offset_seconds": round(float(offset_match.group(1)), 3) if offset_match else 0.0,
        "scale_factor": float(scale_match.group(1)) if scale_match else None,
    }


def _format_srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000.0)))
    h = ms_total // 3_600_000
    ms_total %= 3_600_000
    m = ms_total // 60_000
    ms_total %= 60_000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_shifted_srt(
    cues: Sequence[Any],
    output_path: Path,
    shifts: Sequence[float],
    scale_factor: float,
) -> bool:
    blocks: List[str] = []
    for idx, (cue, shift_sec) in enumerate(zip(cues, shifts), start=1):
        t0 = max(0.0, float(cue.start) * scale_factor + shift_sec)
        t1 = max(t0 + 0.1, float(cue.end) * scale_factor + shift_sec)
        text = (cue.text or "").strip()
        if not text:
            continue
        if getattr(cue, "position", None) == "top" and "{\\an8}" not in text:
            text = f"{{\\an8}}{text}"
        blocks.append(f"{idx}\n{_format_srt_ts(t0)} --> {_format_srt_ts(t1)}\n{text}")
    if not blocks:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return True


def _cues_activity_vector(
    cues: Sequence[Tuple[float, float]],
    n_bins: int,
    dt: float,
) -> np.ndarray:
    vec = np.zeros(n_bins, dtype=np.float32)
    for start, end in cues:
        i0 = max(0, int(math.floor(start / dt)))
        i1 = min(n_bins, int(math.ceil(end / dt)))
        if i1 > i0:
            vec[i0:i1] = 1.0
    kernel = np.array([0.2, 0.5, 1.0, 0.5, 0.2], dtype=np.float32)
    smoothed = np.convolve(vec, kernel, mode="same")
    return np.clip(smoothed, 0.0, 1.0)


def _fft_best_shift(
    ref_vec: np.ndarray,
    tgt_spans: Sequence[Tuple[float, float]],
    *,
    dt: float = 0.05,
    max_offset_sec: float = 180.0,
    scales: Sequence[float] = _CANDIDATE_SCALES,
) -> Tuple[float, float, float]:
    """Find (best_shift_sec, best_scale, pearson_corr) aligning target spans onto ref_vec."""
    n_ref = len(ref_vec)
    ref_zero = ref_vec.astype(np.float32) - float(np.mean(ref_vec))
    ref_norm = float(np.sqrt(np.sum(ref_zero * ref_zero)))
    if ref_norm < 1e-6:
        return 0.0, 1.0, 0.0

    max_bins = max(1, int(round(max_offset_sec / dt)))
    n_fft = 1 << int(math.ceil(math.log2(max(8, n_ref * 2))))
    ref_fft = np.fft.rfft(ref_zero, n=n_fft)

    best_score = -1.0
    best_shift = 0.0
    best_scale = 1.0
    unit_score = 0.0

    for scale in scales:
        scaled_spans = [(s * scale, e * scale) for s, e in tgt_spans]
        max_t = max((e for _, e in scaled_spans), default=0.0)
        n_tgt = max(n_ref, int(math.ceil(max_t / dt)) + 1)
        if n_tgt * 2 > n_fft:
            n_fft = 1 << int(math.ceil(math.log2(n_tgt * 2)))
            ref_fft = np.fft.rfft(ref_zero, n=n_fft)
        tgt_vec = _cues_activity_vector(scaled_spans, n_tgt, dt)
        tgt_zero = tgt_vec - float(np.mean(tgt_vec))
        tgt_norm = float(np.sqrt(np.sum(tgt_zero * tgt_zero)))
        if tgt_norm < 1e-6:
            continue

        corr = np.fft.irfft(ref_fft * np.conj(np.fft.rfft(tgt_zero, n=n_fft)), n=n_fft)
        denom = ref_norm * tgt_norm
        # Positive lags: 0..max_bins, negative lags: n_fft - max_bins .. n_fft - 1
        pos_idx = np.arange(0, min(max_bins + 1, n_fft))
        neg_idx = np.arange(max(0, n_fft - max_bins), n_fft)
        best_pos_i = int(pos_idx[int(np.argmax(corr[pos_idx]))])
        best_neg_i = int(neg_idx[int(np.argmax(corr[neg_idx]))])
        if corr[best_pos_i] >= corr[best_neg_i]:
            lag_bins = best_pos_i
            peak_corr = float(corr[best_pos_i]) / denom
        else:
            lag_bins = best_neg_i - n_fft
            peak_corr = float(corr[best_neg_i]) / denom

        if abs(scale - 1.0) < 1e-6:
            unit_score = peak_corr

        # Prefer scale=1.0 unless an alternate FPS scale beats it decisively
        effective = peak_corr if abs(scale - 1.0) < 1e-6 else peak_corr * 0.96
        if effective > best_score and (abs(scale - 1.0) < 1e-6 or peak_corr > unit_score * 1.08):
            best_score = peak_corr
            best_shift = round(lag_bins * dt, 3)
            best_scale = scale

    return best_shift, best_scale, best_score


def _refine_cue_overlap(
    ref_spans: Sequence[Tuple[float, float]],
    tgt_spans: Sequence[Tuple[float, float]],
    coarse_shift: float,
    scale: float,
) -> float:
    """Refine offset within +-0.4s at 20ms resolution by maximizing cue temporal overlap."""
    ref_sorted = sorted(ref_spans)
    best_shift = coarse_shift
    best_ov = -1.0
    step = 0.02
    offset = coarse_shift - 0.4
    while offset <= coarse_shift + 0.4 + 1e-9:
        ov = 0.0
        j_start = 0
        for rs, re_ in ref_sorted:
            # Coarse linear sweep keeps inner loop small
            while j_start < len(tgt_spans) and tgt_spans[j_start][1] * scale + offset < rs - 5.0:
                j_start += 1
            for k in range(max(0, j_start - 3), len(tgt_spans)):
                ts = tgt_spans[k][0] * scale + offset
                te = tgt_spans[k][1] * scale + offset
                if ts >= re_:
                    break
                cur = min(re_, te) - max(rs, ts)
                if cur > 0.0:
                    ov += cur
        if ov > best_ov + 1e-6 or (abs(ov - best_ov) <= 1e-6 and abs(offset - coarse_shift) < abs(best_shift - coarse_shift)):
            best_ov = ov
            best_shift = round(offset, 3)
        offset += step
    return best_shift


def sync_srt_to_reference_cues(
    reference_srt_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    max_offset_sec: float = 180.0,
) -> Dict[str, Any]:
    """Synchronize an SRT track against an already-synced reference SRT via FFT cue cadence."""
    try:
        ref_raw = reference_srt_path.read_text(encoding="utf-8", errors="replace")
        tgt_raw = srt_path.read_text(encoding="utf-8", errors="replace")
        ref_cues, _ = parse_subtitle_text(ref_raw, file_name="ref.srt")
        tgt_cues, _ = parse_subtitle_text(tgt_raw, file_name="tgt.srt")
    except Exception:
        return {"synced": False, "engine": "cue_cadence", "offset_seconds": 0.0, "scale_factor": None}

    if len(ref_cues) < 5 or len(tgt_cues) < 5:
        return {"synced": False, "engine": "cue_cadence", "offset_seconds": 0.0, "scale_factor": None}

    ref_spans = [(float(c.start), float(c.end)) for c in ref_cues]
    tgt_spans = sorted((float(c.start), float(c.end)) for c in tgt_cues)
    dt = 0.05
    n_bins = max(40, int(math.ceil(max(e for _, e in ref_spans) / dt)) + 20)
    ref_vec = _cues_activity_vector(ref_spans, n_bins, dt)

    coarse_shift, scale, corr = _fft_best_shift(ref_vec, tgt_spans, dt=dt, max_offset_sec=max_offset_sec)
    if corr < 0.12:
        return {"synced": False, "engine": "cue_cadence", "offset_seconds": 0.0, "scale_factor": None}

    fine_shift = _refine_cue_overlap(ref_spans, tgt_spans, coarse_shift, scale)
    cue_shifts: List[float] = [fine_shift] * len(tgt_cues)

    # Piecewise local refinement handles TV release commercial-break / bumper offset jumps
    ref_max = max((e for _, e in ref_spans), default=0.0)
    if ref_max >= 180.0 and len(tgt_cues) >= 20:
        win_sec = 240.0
        for idx_c, cue in enumerate(tgt_cues):
            t_mid = float(cue.start) * scale + fine_shift
            # Compute once per window bucket and cache by bucket index
            if idx_c == 0 or int(t_mid // win_sec) != int((float(tgt_cues[idx_c - 1].start) * scale + fine_shift) // win_sec):
                b = max(0, int(t_mid // win_sec))
                w0 = b * win_sec
                w1 = w0 + win_sec
                pad = 30.0
                r_i0 = max(0, int(math.floor(max(0.0, w0 - pad) / dt)))
                r_i1 = min(n_bins, int(math.ceil((w1 + pad) / dt)))
                win_start = r_i0 * dt
                local_tgts = [
                    (float(c.start) * scale + fine_shift - win_start, float(c.end) * scale + fine_shift - win_start)
                    for c in tgt_cues
                    if w0 <= float(c.start) * scale + fine_shift < w1
                ]
                local_refs = [
                    (rs, re_)
                    for rs, re_ in ref_spans
                    if (w0 - pad) <= rs <= (w1 + pad)
                ]
                bucket_shift = fine_shift
                if len(local_tgts) >= 6 and (r_i1 - r_i0) >= 80:
                    delta_coarse, _, l_corr = _fft_best_shift(
                        ref_vec[r_i0:r_i1],
                        local_tgts,
                        dt=dt,
                        max_offset_sec=15.0,
                        scales=(1.0,),
                    )
                    if l_corr >= 0.25:
                        local_tgts_abs = [
                            (float(c.start), float(c.end))
                            for c in tgt_cues
                            if w0 <= float(c.start) * scale + fine_shift < w1
                        ]
                        bucket_shift = _refine_cue_overlap(
                            local_refs or ref_spans,
                            local_tgts_abs,
                            round(fine_shift + delta_coarse, 3),
                            scale,
                        )
            cue_shifts[idx_c] = bucket_shift

    if not _write_shifted_srt(tgt_cues, output_path, cue_shifts, scale):
        return {"synced": False, "engine": "cue_cadence", "offset_seconds": 0.0, "scale_factor": None}

    scale_out = None if abs(scale - 1.0) < 1e-6 else round(scale, 5)
    median_shift = round(float(np.median(np.array(cue_shifts, dtype=np.float32))), 3)
    logger.info(
        f"Synced '{srt_path.name}' against reference cues with cue_cadence "
        f"(offset={median_shift:+.3f}s, scale={scale_out or 1.0}, corr={corr:.2f})"
    )
    return {
        "synced": True,
        "engine": "cue_cadence",
        "offset_seconds": median_shift,
        "scale_factor": scale_out,
        "warning": None,
    }


def _sync_srt_to_wav_envelope(
    wav_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    max_offset_sec: float = 180.0,
) -> Optional[Dict[str, Any]]:
    """Built-in fallback synchronizer using PCM WAV RMS energy envelope vs. subtitle cadence."""
    if not wav_path.is_file() or wav_path.stat().st_size < 64:
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            if sampwidth != 2 or rate <= 0 or n_frames < rate * 2:
                return None
            dt = 0.05
            hop = max(1, int(round(rate * dt)))
            n_bins = max(1, n_frames // hop)
            total_frames = n_bins * hop
            raw = wf.readframes(total_frames)
    except Exception:
        return None

    try:
        samples = np.frombuffer(raw, dtype=np.int16)
        usable = (len(samples) // n_channels) * n_channels
        if usable <= 0:
            return None
        mono = samples[:usable].reshape(-1, n_channels).mean(axis=1)
        bins_count = len(mono) // hop
        if bins_count < 40:
            return None
        framed = mono[: bins_count * hop].reshape(bins_count, hop)
        rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-6)
        energy = np.log1p(rms).astype(np.float32)
        med = float(np.median(energy))
        std = float(np.std(energy)) + 1e-6
        speech_proxy = np.clip((energy - med) / (2.0 * std), 0.0, 1.5).astype(np.float32)

        tgt_raw = srt_path.read_text(encoding="utf-8", errors="replace")
        tgt_cues, _ = parse_subtitle_text(tgt_raw, file_name="tgt.srt")
        if len(tgt_cues) < 5:
            return None
        tgt_spans = [(float(c.start), float(c.end)) for c in tgt_cues]

        shift, scale, corr = _fft_best_shift(speech_proxy, tgt_spans, dt=dt, max_offset_sec=max_offset_sec)
        if corr < 0.08:
            return None
        if not _write_shifted_srt(tgt_cues, output_path, [shift] * len(tgt_cues), scale):
            return None
        scale_out = None if abs(scale - 1.0) < 1e-6 else round(scale, 5)
        logger.info(
            f"Synced '{srt_path.name}' to audio with audio_envelope "
            f"(offset={shift:+.3f}s, scale={scale_out or 1.0}, corr={corr:.2f})"
        )
        return {
            "synced": True,
            "engine": "audio_envelope",
            "offset_seconds": shift,
            "scale_factor": scale_out,
            "warning": None,
        }
    except Exception as exc:
        logger.warning(f"Built-in audio envelope sync failed for '{srt_path.name}': {exc}")
        return None


def sync_srt_to_audio(
    reference_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    timeout: float = 300.0,
    reference_srt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Synchronize an SRT file against extracted audio/video via ffsubsync, alass, or built-in sync."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [
        ("ffsubsync", _ffsubsync_command(reference_path, srt_path, output_path)),
        ("alass", _alass_command(reference_path, srt_path, output_path)),
    ]

    last_error: Optional[str] = None
    for engine, cmd in attempts:
        if cmd is None:
            continue
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            combined_out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
                metrics = _parse_sync_metrics(combined_out)
                logger.info(
                    f"Synced '{srt_path.name}' with {engine} (offset={metrics['offset_seconds']:+.3f}s)"
                )
                return {
                    "synced": True,
                    "engine": engine,
                    "offset_seconds": metrics["offset_seconds"],
                    "scale_factor": metrics["scale_factor"],
                    "warning": None,
                }
            last_error = f"{engine} exited with code {proc.returncode}"
            logger.warning(f"Audio sync attempt with {engine} failed for '{srt_path.name}': {last_error}")
        except Exception as exc:
            last_error = f"{engine}: {exc}"
            logger.warning(f"Audio sync error with {engine} for '{srt_path.name}': {exc}")

    # Built-in fallback 1: align against an already-synced reference SRT (ideal when EN is embedded)
    if reference_srt_path is not None and Path(reference_srt_path).is_file():
        cue_res = sync_srt_to_reference_cues(Path(reference_srt_path), srt_path, output_path)
        if cue_res.get("synced"):
            return cue_res

    # Built-in fallback 2: correlate SRT speech intervals against extracted WAV energy envelope
    env_res = _sync_srt_to_wav_envelope(reference_path, srt_path, output_path)
    if env_res and env_res.get("synced"):
        return env_res

    # Degrade gracefully when no sync engine could verify timestamps
    shutil.copyfile(srt_path, output_path)
    warning = (
        f"Audio sync tool unavailable or failed ({last_error or 'neither ffsubsync nor alass found'}); "
        "using original subtitle timestamps."
    )
    logger.warning(warning)
    return {
        "synced": False,
        "engine": "none",
        "offset_seconds": 0.0,
        "scale_factor": None,
        "warning": warning,
    }
