"""Smart Audio Map: pre-ASR acoustic timeline scanner and selective window conditioner.

Scans the extracted 16 kHz PCM soundtrack in a fast vectorized pass (~1-2s on CPU for a
2-hour movie) to identify specific time stretches that need localized conditioning:
  * ``whisper``: faint or whispered speech (-48..-25.5 dBFS with sibilant/formant presence)
    -> localized soft speech gain (+up to 9 dB) without cutting high-frequency consonants.
  * ``comms``: band-limited radio/walkie-talkie/telephone speech (>=86% power in 280-3600 Hz,
    brick-wall roll-off below 220 Hz and above 4300 Hz)
    -> localized radio bandpass + spectral hiss floor reduction + gentle compression, WITHOUT
       injecting global military hotwords.
  * ``music_masked``: speech buried under sustained broadband music/SFX
    -> selective Demucs vocal isolation ONLY on those windows (seconds instead of hours).

Every sample outside the detected windows remains 100% untouched (``OFF``), and window edges
are blended with an 80 ms raised-cosine crossfade so Whisper's VAD never sees splice clicks.
Speaker diarization and F0 voice-gender estimation continue reading the original unconditioned
WAV (`ctx.raw_audio_wav_path`).
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from app.core.executables import executable
from app.core.ffmpeg_safety import safe_input
from app.pipeline.context import PipelineCancelled
from app.services.audio_isolation import (
    is_vocal_isolation_available,
    isolate_vocals,
)

logger = logging.getLogger(__name__)

AudioMode = Literal["off", "auto", "on"]
WindowKind = Literal["whisper", "comms", "music_masked"]

VALID_AUDIO_MODES = ("off", "auto", "on")

_FRAME_SEC = 0.50
_HOP_SEC = 0.25
_CROSSFADE_SEC = 0.080
_WINDOW_PAD_SEC = 0.35
_MERGE_GAP_SEC = 0.75
_MAX_SELECTIVE_DEMUCS_WINDOWS = 24
_MAX_SELECTIVE_DEMUCS_TOTAL_SEC = 600.0


def normalize_audio_mode(mode: Optional[str], legacy_flag: bool = False) -> AudioMode:
    """Resolve an explicit `'off'|'auto'|'on'` mode with fallback to legacy boolean flags."""
    cleaned = (mode or "").strip().lower()
    if cleaned == "auto":
        return "auto"
    if cleaned == "on" or legacy_flag:
        return "on"
    return "off"


def resolve_audio_modes(
    *,
    vocal_isolation: bool = False,
    vocal_isolation_mode: Optional[str] = None,
    comms_boost: bool = False,
    comms_boost_mode: Optional[str] = None,
) -> Tuple[AudioMode, AudioMode]:
    """Return `(effective_vocal_mode, effective_comms_mode)` preserving backward compatibility."""
    v_mode = normalize_audio_mode(vocal_isolation_mode, bool(vocal_isolation))
    c_mode = normalize_audio_mode(comms_boost_mode, bool(comms_boost))
    return v_mode, c_mode


@dataclass
class AudioWindow:
    start: float
    end: float
    kind: WindowKind
    confidence: float = 0.75

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SmartAudioMapReport:
    duration_sec: float = 0.0
    windows: List[AudioWindow] = field(default_factory=list)
    conditioned_wav_path: Optional[str] = None
    used_demucs: bool = False

    @property
    def clean_ratio(self) -> float:
        if self.duration_sec <= 0.0:
            return 1.0
        covered = sum(w.duration for w in self.windows)
        return max(0.0, min(1.0, 1.0 - (covered / self.duration_sec)))

    def counts_by_kind(self) -> Dict[str, Tuple[int, float]]:
        out: Dict[str, Tuple[int, float]] = {}
        for w in self.windows:
            cnt, dur = out.get(w.kind, (0, 0.0))
            out[w.kind] = (cnt + 1, round(dur + w.duration, 2))
        return out

    def summary_line(self) -> str:
        if not self.windows:
            return f"Smart Audio Map: 100.0% clean speech/silence (all filters stayed OFF across {self.duration_sec:.1f}s)"
        counts = self.counts_by_kind()
        parts = [f"{self.clean_ratio * 100:.1f}% untouched (OFF)"]
        for kind in ("whisper", "comms", "music_masked"):
            if kind in counts:
                c, d = counts[kind]
                parts.append(f"{kind}={c} window(s) ({d:.1f}s)")
        return "Smart Audio Map: " + ", ".join(parts)


def _read_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if not raw:
        return np.zeros(0, dtype=np.float32), sr
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        frame_count = len(samples) // channels
        samples = samples[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)
    return samples, sr


def _write_mono_pcm16(wav_path: Path, samples: np.ndarray, sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(samples * 32767.0), -32768.0, 32767.0).astype(np.int16)
    tmp = wav_path.with_name(f"{wav_path.stem}.tmp{wav_path.suffix}")
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        tmp.replace(wav_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _coalesce_mask(
    mask: np.ndarray,
    hop_sec: float,
    frame_sec: float,
    duration_sec: float,
    kind: WindowKind,
    min_duration: float,
    pad_sec: float = _WINDOW_PAD_SEC,
    merge_gap: float = _MERGE_GAP_SEC,
) -> List[AudioWindow]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    raw_spans: List[Tuple[float, float]] = []
    run_start = int(idx[0])
    prev = int(idx[0])
    for cur_raw in idx[1:]:
        cur = int(cur_raw)
        if (cur - prev) * hop_sec <= merge_gap:
            prev = cur
        else:
            raw_spans.append((run_start * hop_sec, prev * hop_sec + frame_sec))
            run_start = cur
            prev = cur
    raw_spans.append((run_start * hop_sec, prev * hop_sec + frame_sec))

    merged: List[AudioWindow] = []
    for st, et in raw_spans:
        if (et - st) < min_duration:
            continue
        p_st = max(0.0, st - pad_sec)
        p_et = min(duration_sec, et + pad_sec)
        if merged and p_st - merged[-1].end <= 0.25:
            merged[-1].end = max(merged[-1].end, p_et)
        else:
            merged.append(AudioWindow(start=round(p_st, 3), end=round(p_et, 3), kind=kind))
    return merged


def _resolve_overlapping_windows(windows: Sequence[AudioWindow]) -> List[AudioWindow]:
    """Resolve overlaps deterministically by priority (`comms` > `music_masked` > `whisper`)."""
    if not windows:
        return []
    priority = {"comms": 3, "music_masked": 2, "whisper": 1}
    ordered = sorted(windows, key=lambda w: (w.start, -priority[w.kind]))
    result: List[AudioWindow] = []
    for cur in ordered:
        if cur.end - cur.start < 0.35:
            continue
        if not result:
            result.append(AudioWindow(cur.start, cur.end, cur.kind, cur.confidence))
            continue
        prev = result[-1]
        if cur.start >= prev.end:
            result.append(AudioWindow(cur.start, cur.end, cur.kind, cur.confidence))
            continue
        if cur.kind == prev.kind:
            prev.end = max(prev.end, cur.end)
            continue
        if priority[cur.kind] > priority[prev.kind]:
            left_end = cur.start
            right_start = cur.end
            old_end = prev.end
            old_kind = prev.kind
            if left_end - prev.start >= 0.45:
                prev.end = left_end
                result.append(AudioWindow(cur.start, cur.end, cur.kind, cur.confidence))
            else:
                prev.start = min(prev.start, cur.start)
                prev.end = cur.end
                prev.kind = cur.kind
            if old_end - right_start >= 0.55:
                result.append(AudioWindow(right_start, old_end, old_kind, prev.confidence))
        else:
            if cur.end - prev.end >= 0.55:
                result.append(AudioWindow(prev.end, cur.end, cur.kind, cur.confidence))
    return result


def analyze_audio_timeline(
    wav_path: str | Path,
    *,
    detect_whispers: bool = True,
    detect_comms: bool = False,
    detect_music_masked: bool = False,
    surround_center_extracted: bool = False,
) -> List[AudioWindow]:
    """Scan a 16 kHz WAV file and return localized windows requiring targeted conditioning.

    By default `detect_comms=False` so `auto` mode never applies narrowband radio cuts to muffled
    or reverberant cinema speech; whisper detection adapts to the soundtrack's own 88th-percentile
    dialogue loudness so standard -27..-31 LKFS Atmos dialogue stays 100% untouched (`OFF`).
    """
    path = Path(wav_path)
    if not path.is_file():
        return []
    samples, sr = _read_mono_float32(path)
    if samples.size == 0 or sr <= 0:
        return []

    duration_sec = samples.size / float(sr)
    win_len = max(64, int(round(_FRAME_SEC * sr)))
    hop_len = max(32, int(round(_HOP_SEC * sr)))
    if samples.size < win_len:
        return []

    n_frames = 1 + (samples.size - win_len) // hop_len
    # Process in 10-minute chunks so peak temporary RAM stays under ~50 MB even on 3-hour movies
    chunk_frames = max(1, int(round(600.0 / _HOP_SEC)))

    rms_db_all = np.empty(n_frames, dtype=np.float32)
    low_260_share = np.empty(n_frames, dtype=np.float32)
    comms_band_share = np.empty(n_frames, dtype=np.float32)
    high_3800_share = np.empty(n_frames, dtype=np.float32)
    whisper_band_share = np.empty(n_frames, dtype=np.float32)
    flatness_all = np.empty(n_frames, dtype=np.float32)
    inband_flatness_all = np.empty(n_frames, dtype=np.float32)
    sub_rms_cv = np.empty(n_frames, dtype=np.float32)

    hann = np.hanning(win_len).astype(np.float32)
    n_fft = 1 << int(np.ceil(np.log2(win_len)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))

    idx_low_260 = freqs < 260.0
    idx_comms = (freqs >= 280.0) & (freqs <= 3500.0)
    idx_high_3800 = freqs >= 3800.0
    idx_whisper = (freqs >= 550.0) & (freqs <= 6800.0)

    for start_f in range(0, n_frames, chunk_frames):
        end_f = min(n_frames, start_f + chunk_frames)
        offsets = np.arange(start_f, end_f, dtype=np.int64)[:, None] * hop_len + np.arange(win_len, dtype=np.int64)[None, :]
        frames = samples[offsets]
        frames = frames - frames.mean(axis=1, keepdims=True)

        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        rms_db_all[start_f:end_f] = 20.0 * np.log10(np.maximum(rms, 1e-6))

        sub_count = 8
        sub_len = win_len // sub_count
        if sub_len >= 16:
            trimmed = frames[:, : sub_count * sub_len].reshape(end_f - start_f, sub_count, sub_len)
            s_rms = np.sqrt(np.mean(trimmed * trimmed, axis=2) + 1e-12)
            sub_rms_cv[start_f:end_f] = np.std(s_rms, axis=1) / (np.mean(s_rms, axis=1) + 1e-7)
        else:
            sub_rms_cv[start_f:end_f] = 0.5

        spec = np.fft.rfft(frames * hann[None, :], n=n_fft, axis=1)
        power = (spec.real * spec.real + spec.imag * spec.imag) + 1e-12
        total_p = np.sum(power, axis=1) + 1e-12

        low_260_share[start_f:end_f] = np.sum(power[:, idx_low_260], axis=1) / total_p
        comms_band_share[start_f:end_f] = np.sum(power[:, idx_comms], axis=1) / total_p
        high_3800_share[start_f:end_f] = np.sum(power[:, idx_high_3800], axis=1) / total_p
        whisper_band_share[start_f:end_f] = np.sum(power[:, idx_whisper], axis=1) / total_p

        geo_mean = np.exp(np.mean(np.log(power[:, 1:]), axis=1))
        arith_mean = np.mean(power[:, 1:], axis=1) + 1e-12
        flatness_all[start_f:end_f] = geo_mean / arith_mean

        comms_bins = power[:, idx_comms]
        if comms_bins.shape[1] > 2:
            ib_geo = np.exp(np.mean(np.log(comms_bins), axis=1))
            ib_arith = np.mean(comms_bins, axis=1) + 1e-12
            inband_flatness_all[start_f:end_f] = ib_geo / ib_arith
        else:
            inband_flatness_all[start_f:end_f] = 0.0

    windows: List[AudioWindow] = []

    # 1. Strict narrowband static radio / walkie-talkie detection (only when explicitly requested):
    # Requires brick-wall 3.8 kHz codec roll-off (<1.0%), zero bass (<1.5%), >=93% energy in 280-3500 Hz,
    # AND measurable radio static/carrier noise floor (`inband_flatness_all >= 0.010`) so clean indoor
    # or historical muffled dialogue (`inband_flatness < 0.003`) never false-positives as radio.
    if detect_comms:
        comms_mask = (
            (rms_db_all >= -42.0)
            & (comms_band_share >= 0.93)
            & (low_260_share <= 0.015)
            & (high_3800_share <= 0.010)
            & (inband_flatness_all >= 0.010)
            & (flatness_all <= 0.45)
            & (sub_rms_cv >= 0.12)
        )
        windows.extend(
            _coalesce_mask(comms_mask, _HOP_SEC, _FRAME_SEC, duration_sec, "comms", min_duration=1.00)
        )

    # 2. Whisper / faint-dialogue detection calibrated relative to the track's own dialogue loudness
    if detect_whispers:
        active_frames = rms_db_all[rms_db_all >= -48.0]
        if active_frames.size >= 8:
            speech_ref_db = float(np.percentile(active_frames, 88))
        else:
            speech_ref_db = -22.0
        # Only boost stretches at least 9.5 dB quieter than normal speech in this mix (and <= -31.5 dBFS)
        whisper_ceiling_db = min(-31.5, speech_ref_db - 9.5)

        vad_soft_mask = np.ones(n_frames, dtype=bool)
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            low_stamps = get_speech_timestamps(
                samples,
                VadOptions(threshold=0.16, min_speech_duration_ms=120, min_silence_duration_ms=150, speech_pad_ms=80),
            )
            vad_soft_mask[:] = False
            for st in low_stamps:
                f0 = max(0, int((st["start"] / float(sr) - _FRAME_SEC) / _HOP_SEC))
                f1 = min(n_frames, int(np.ceil((st["end"] / float(sr)) / _HOP_SEC)) + 1)
                vad_soft_mask[f0:f1] = True
        except Exception:
            vad_soft_mask = sub_rms_cv >= 0.24

        whisper_mask = (
            vad_soft_mask
            & (rms_db_all >= -49.0)
            & (rms_db_all <= whisper_ceiling_db)
            & (whisper_band_share >= 0.56)
            & (low_260_share <= 0.45)
            & (flatness_all <= 0.56)
            & (sub_rms_cv >= 0.18)
        )
        windows.extend(
            _coalesce_mask(whisper_mask, _HOP_SEC, _FRAME_SEC, duration_sec, "whisper", min_duration=0.60)
        )

    # 3. Music-masked speech detection for selective Demucs vocal isolation
    if detect_music_masked:
        kernel = 10
        if n_frames >= kernel:
            pad_rms = np.pad(rms_db_all, (kernel // 2, kernel - kernel // 2 - 1), mode="edge")
            rolling_min_db = np.minimum.reduce([pad_rms[k : k + n_frames] for k in range(kernel)])
        else:
            rolling_min_db = rms_db_all

        bed_floor_db = -21.0 if surround_center_extracted else -27.5
        non_comms_share = low_260_share + high_3800_share
        music_masked = (
            (rolling_min_db >= bed_floor_db)
            & (rms_db_all >= bed_floor_db + 1.5)
            & (comms_band_share >= 0.28)
            & (comms_band_share <= 0.78)
            & (non_comms_share >= 0.34)
            & (sub_rms_cv >= 0.12)
            & (sub_rms_cv <= 0.55)
            & (flatness_all <= 0.38)
        )
        cand_music = _coalesce_mask(
            music_masked, _HOP_SEC, _FRAME_SEC, duration_sec, "music_masked", min_duration=1.75, pad_sec=0.50
        )
        if len(cand_music) > _MAX_SELECTIVE_DEMUCS_WINDOWS:
            cand_music = sorted(cand_music, key=lambda w: w.duration, reverse=True)[:_MAX_SELECTIVE_DEMUCS_WINDOWS]
            cand_music.sort(key=lambda w: w.start)
        acc = 0.0
        capped_music: List[AudioWindow] = []
        for w in cand_music:
            if acc + w.duration > _MAX_SELECTIVE_DEMUCS_TOTAL_SEC:
                break
            capped_music.append(w)
            acc += w.duration
        windows.extend(capped_music)

    return _resolve_overlapping_windows(windows)


def _enhance_whisper_slice(seg: np.ndarray, sr: int) -> np.ndarray:
    """Pure linear peak-safe gain lift for faint whispers (zero EQ warping or nonlinear distortion)."""
    if seg.size == 0:
        return seg
    centered = (seg - float(np.mean(seg))).astype(np.float32)
    rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
    peak = float(np.max(np.abs(centered)) + 1e-8)
    target_rms = 10.0 ** (-24.0 / 20.0)  # ~0.063 (-24 dBFS, natural cinema speech level)
    desired_gain = min(2.5, max(1.0, target_rms / max(rms, 1e-4)))  # Cap at +8.0 dB
    safe_gain = min(desired_gain, 0.85 / peak)
    return (centered * max(1.0, safe_gain)).astype(np.float32)


def _enhance_comms_slice(seg: np.ndarray, sr: int) -> np.ndarray:
    """Gentle linear bandpass & peak-safe leveling for narrowband radio without spectral-subtraction artifacts."""
    if seg.size == 0:
        return seg
    n_fft = 1 << int(np.ceil(np.log2(max(64, seg.size))))
    spec = np.fft.rfft(seg, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))

    low_ramp = np.clip((freqs - 180.0) / 70.0, 0.0, 1.0)
    high_ramp = np.clip((4800.0 - freqs) / 500.0, 0.0, 1.0)
    band_mask = (low_ramp * high_ramp).astype(np.float32)
    cleaned = np.fft.irfft(spec * band_mask, n=n_fft)[: seg.size].astype(np.float32)

    rms = float(np.sqrt(np.mean(cleaned * cleaned) + 1e-12))
    peak = float(np.max(np.abs(cleaned)) + 1e-8)
    target_rms = 10.0 ** (-22.0 / 20.0)
    desired_gain = min(2.5, max(1.0, target_rms / max(rms, 1e-4)))
    safe_gain = min(desired_gain, 0.85 / peak)
    return (cleaned * max(1.0, safe_gain)).astype(np.float32)


def _extract_source_slice_stereo(
    source_media_path: str,
    out_wav: Path,
    audio_stream_index: int,
    start_sec: float,
    duration_sec: float,
) -> bool:
    """Extract a clean full-band 44.1 kHz stereo slice directly from the source container for Demucs."""
    src = Path(source_media_path)
    if not src.is_file():
        return False
    cmd = [
        executable("ffmpeg"),
        "-y",
        "-nostats",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{max(0.2, duration_sec):.3f}",
        *safe_input(str(src.resolve())),
        "-map",
        f"0:a:{audio_stream_index}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out_wav.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=120)
    return res.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 1024


def _splice_with_crossfade(dst: np.ndarray, processed: np.ndarray, i0: int, i1: int, fade_samples: int) -> None:
    """Splice `processed` into `dst[i0:i1]` using raised-cosine crossfades at both edges."""
    seg_len = i1 - i0
    if seg_len <= 0 or processed.size == 0:
        return
    if processed.size != seg_len:
        x_old = np.linspace(0.0, 1.0, processed.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, seg_len, dtype=np.float32)
        processed = np.interp(x_new, x_old, processed).astype(np.float32)

    f = min(fade_samples, seg_len // 2)
    if f <= 1:
        dst[i0:i1] = processed
        return

    ramp_in = (0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, f, dtype=np.float32)))).astype(np.float32)
    ramp_out = ramp_in[::-1]

    orig_head = dst[i0 : i0 + f].copy()
    orig_tail = dst[i1 - f : i1].copy()

    dst[i0:i1] = processed
    dst[i0 : i0 + f] = orig_head * (1.0 - ramp_in) + processed[:f] * ramp_in
    dst[i1 - f : i1] = processed[-f:] * ramp_out + orig_tail * (1.0 - ramp_out)


def condition_audio_by_map(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    windows: Sequence[AudioWindow],
    *,
    source_media_path: Optional[str] = None,
    audio_stream_index: int = 0,
    model_name: str = "htdemucs",
    cancel_flag: Optional[threading.Event] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> SmartAudioMapReport:
    """Apply localized conditioning strictly inside `windows` and return to `OFF` everywhere else."""
    in_path = Path(input_wav_path)
    out_path = Path(output_wav_path)
    samples, sr = _read_mono_float32(in_path)
    duration_sec = samples.size / float(sr) if sr > 0 else 0.0

    if not windows or samples.size == 0 or sr <= 0:
        return SmartAudioMapReport(
            duration_sec=duration_sec,
            windows=[],
            conditioned_wav_path=str(in_path),
            used_demucs=False,
        )

    out_samples = samples.copy()
    fade_samples = max(16, int(round(_CROSSFADE_SEC * sr)))
    demucs_ready = is_vocal_isolation_available()
    used_demucs = False
    applied_windows: List[AudioWindow] = []

    with tempfile.TemporaryDirectory(prefix="subbuild_smartmap_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        total_w = len(windows)
        for idx, win in enumerate(windows):
            if cancel_flag is not None and cancel_flag.is_set():
                raise PipelineCancelled()
            i0 = max(0, min(samples.size, int(round(win.start * sr))))
            i1 = max(i0, min(samples.size, int(round(win.end * sr))))
            if i1 - i0 < 64:
                continue

            if on_progress is not None:
                pct = 10.0 + (idx / max(1, total_w)) * 85.0
                on_progress(pct, f"Smart Audio Map: conditioning {win.kind} window ({win.start:.1f}s-{win.end:.1f}s)...")

            raw_slice = samples[i0:i1]
            if win.kind == "whisper":
                proc = _enhance_whisper_slice(raw_slice, sr)
                _splice_with_crossfade(out_samples, proc, i0, i1, fade_samples)
                applied_windows.append(win)
            elif win.kind == "comms":
                proc = _enhance_comms_slice(raw_slice, sr)
                _splice_with_crossfade(out_samples, proc, i0, i1, fade_samples)
                applied_windows.append(win)
            elif win.kind == "music_masked":
                if not demucs_ready:
                    # Fallback when Demucs is not installed: mild speech-band clarity boost
                    proc = _enhance_whisper_slice(raw_slice, sr)
                    _splice_with_crossfade(out_samples, proc, i0, i1, fade_samples)
                    applied_windows.append(win)
                    continue
                margin = 0.45
                ext_start = max(0.0, win.start - margin)
                ext_end = min(duration_sec, win.end + margin)
                slice_in = tmp_root / f"win_{idx}_in.wav"
                slice_out = tmp_root / f"win_{idx}_vocals.wav"
                extracted_stereo = False
                if source_media_path:
                    extracted_stereo = _extract_source_slice_stereo(
                        source_media_path,
                        slice_in,
                        audio_stream_index=audio_stream_index,
                        start_sec=ext_start,
                        duration_sec=ext_end - ext_start,
                    )
                if not extracted_stereo:
                    e0 = max(0, int(round(ext_start * sr)))
                    e1 = min(samples.size, int(round(ext_end * sr)))
                    _write_mono_pcm16(slice_in, samples[e0:e1], sr)

                isolate_vocals(
                    str(slice_in),
                    str(slice_out),
                    model_name=model_name,
                    source_media_path=None,  # slice-specific, do not collide with whole-file cache
                    cancel_flag=cancel_flag,
                )
                iso_samples, iso_sr = _read_mono_float32(slice_out)
                if iso_samples.size > 0 and iso_sr > 0:
                    trim_head = int(round((win.start - ext_start) * iso_sr))
                    trim_len = int(round((win.end - win.start) * iso_sr))
                    trimmed = iso_samples[max(0, trim_head) : max(0, trim_head) + max(1, trim_len)]
                    if trimmed.size > 0:
                        _splice_with_crossfade(out_samples, trimmed, i0, i1, fade_samples)
                        used_demucs = True
                        applied_windows.append(win)

    if not applied_windows:
        return SmartAudioMapReport(
            duration_sec=duration_sec,
            windows=[],
            conditioned_wav_path=str(in_path),
            used_demucs=False,
        )

    _write_mono_pcm16(out_path, out_samples, sr)
    if on_progress is not None:
        on_progress(100.0, "Smart Audio Map conditioning complete")
    return SmartAudioMapReport(
        duration_sec=duration_sec,
        windows=applied_windows,
        conditioned_wav_path=str(out_path),
        used_demucs=used_demucs,
    )
