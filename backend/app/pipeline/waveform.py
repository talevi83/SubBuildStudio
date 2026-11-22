"""
Lightweight audio amplitude envelope extraction and caching for the Studio waveform timeline.

Extracts 50 peaks/second (20 ms bins) from an existing job WAV or directly from the selected
audio stream of the source video via FFmpeg, and caches the normalized JSON under
`storage/regions/<job_id>.waveform.json`.
"""
import asyncio
import json
import logging
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.ffmpeg_safety import safe_input
from app.core.private_files import write_text_atomic
from app.pipeline.stages.audio_stage import build_audio_filter, get_ffmpeg_bin

logger = logging.getLogger(__name__)

PEAKS_PER_SECOND = 50
WAVEFORM_SAMPLE_RATE = 8000
SAMPLES_PER_BIN = WAVEFORM_SAMPLE_RATE // PEAKS_PER_SECOND  # 160 samples per 20 ms bin
_NORM_FLOOR = 800  # Minimum int16 scale so near-silence does not blow up into full-height noise

# Serialize duplicate concurrent extractions for the same job without blocking the event loop
_job_locks: Dict[str, asyncio.Lock] = {}


def cache_path(job_id: str) -> Path:
    """Return the path of the cached waveform JSON for a job."""
    return settings.storage_dir / "regions" / f"{job_id}.waveform.json"


def _run_ffmpeg_pcm(cmd: List[str]) -> bytes:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=flags)
    if res.returncode != 0:
        err = (res.stderr or b"").decode(errors="ignore")[-800:]
        raise RuntimeError(f"FFmpeg waveform extraction failed: {err}")
    return res.stdout or b""


def _extract_pcm_s16le(
    job_id: str,
    video_path: str,
    audio_stream: int,
    channels: int,
    comms_boost: bool,
) -> bytes:
    ffmpeg = get_ffmpeg_bin()
    wav_path = settings.storage_dir / "audio" / f"{job_id}.wav"
    if wav_path.is_file():
        try:
            cmd = [
                ffmpeg,
                "-v", "error",
                "-nostats",
                *safe_input(str(wav_path.resolve())),
                "-vn",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(WAVEFORM_SAMPLE_RATE),
                "-ac", "1",
                "pipe:1",
            ]
            return _run_ffmpeg_pcm(cmd)
        except Exception as exc:
            logger.debug(f"Could not read waveform from cached WAV {wav_path}: {exc}")

    dialogue_mix = channels >= 6
    resolved_video = str(Path(video_path).resolve())

    def _build_video_cmd(use_dialogue_mix: bool) -> List[str]:
        cmd = [
            ffmpeg,
            "-v", "error",
            "-nostats",
            *safe_input(resolved_video),
            "-map", f"0:a:{audio_stream}",
            "-vn",
        ]
        audio_filter = build_audio_filter(use_dialogue_mix, comms_boost)
        if audio_filter:
            cmd += ["-af", audio_filter]
        cmd += [
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(WAVEFORM_SAMPLE_RATE),
            "-ac", "1",
            "pipe:1",
        ]
        return cmd

    try:
        return _run_ffmpeg_pcm(_build_video_cmd(dialogue_mix))
    except RuntimeError:
        if not dialogue_mix:
            raise
        logger.warning("Waveform dialogue downmix failed; retrying with default downmix")
        return _run_ffmpeg_pcm(_build_video_cmd(False))


def compute_peaks_from_pcm(pcm_bytes: bytes) -> List[float]:
    """Convert raw 8 kHz mono s16le PCM bytes into normalized 0.0..1.0 peaks (50/sec)."""
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return []

    samples = array("h")
    samples.frombytes(pcm_bytes[:usable])
    if sys.byteorder != "little":
        samples.byteswap()

    total_samples = len(samples)
    raw_peaks: List[int] = []
    for offset in range(0, total_samples, SAMPLES_PER_BIN):
        chunk = samples[offset : offset + SAMPLES_PER_BIN]
        if not chunk:
            continue
        hi = max(chunk)
        lo = min(chunk)
        amp = hi if hi >= -lo else -lo
        if amp > 32767:
            amp = 32767
        raw_peaks.append(amp)

    if not raw_peaks:
        return []

    sorted_peaks = sorted(raw_peaks)
    p995_idx = min(len(sorted_peaks) - 1, int(len(sorted_peaks) * 0.995))
    scale = float(max(_NORM_FLOOR, sorted_peaks[p995_idx]))
    return [round(min(1.0, p / scale), 3) for p in raw_peaks]


def _read_cached_sync(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("peaks"), list):
            return data
    except (OSError, ValueError):
        return None
    return None


def _build_and_cache_sync(
    job_id: str,
    video_path: str,
    audio_stream: int,
    channels: int,
    comms_boost: bool,
    duration: float,
) -> Dict[str, Any]:
    path = cache_path(job_id)
    cached = _read_cached_sync(path)
    if cached is not None:
        return cached

    pcm = _extract_pcm_s16le(job_id, video_path, audio_stream, channels, comms_boost)
    peaks = compute_peaks_from_pcm(pcm)
    computed_duration = round(len(peaks) / PEAKS_PER_SECOND, 3) if peaks else float(duration or 0.0)
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "peaks_per_second": PEAKS_PER_SECOND,
        "duration": computed_duration,
        "peaks": peaks,
    }
    try:
        write_text_atomic(path, json.dumps(payload, separators=(",", ":")))
    except OSError as exc:
        logger.warning(f"Could not cache waveform for job {job_id}: {exc}")
    return payload


async def get_or_create_waveform(
    job_id: str,
    video_path: str,
    audio_stream: int = 0,
    channels: int = 2,
    comms_boost: bool = False,
    duration: float = 0.0,
) -> Dict[str, Any]:
    """Load cached waveform peaks or extract them from the job's audio without blocking the loop."""
    path = cache_path(job_id)
    cached = await asyncio.to_thread(_read_cached_sync, path)
    if cached is not None:
        return cached

    lock = _job_locks.setdefault(job_id, asyncio.Lock())
    try:
        async with lock:
            return await asyncio.to_thread(
                _build_and_cache_sync,
                job_id,
                video_path,
                audio_stream,
                channels,
                comms_boost,
                duration,
            )
    finally:
        if not lock.locked():
            _job_locks.pop(job_id, None)
