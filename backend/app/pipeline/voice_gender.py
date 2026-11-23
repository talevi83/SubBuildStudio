"""Guess each diarized speaker's gender from the pitch of their voice.

Hebrew and Arabic conjugate by gender, and many lines carry no other clue (no name, no "sir",
no pronoun). The fundamental frequency (F0) of adult speech separates most men (roughly
85-155 Hz) from most women (roughly 165-255 Hz). Each speaker's voiced frames are pooled over
their whole timeline and the median decides; a median in the overlap zone, or one too high for
an adult voice (children, shouting, singing), stays unknown.

The result is a voice estimate, not a fact: it is stored with `gender_source="voice"`, a
character's gender from the transcript (glossary, speaker matching) replaces it, and the
listener resolver trusts it a little less than a gender that came from the text.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
FRAME_SECONDS = 0.04
HOP_SECONDS = 0.01
# A frame counts as voiced when its normalised autocorrelation peak is at least this strong
VOICING_THRESHOLD = 0.45

MALE_BELOW_HZ = 155.0
FEMALE_ABOVE_HZ = 170.0
ADULT_MAX_HZ = 300.0
# Clearly inside one range: trusted more
MALE_CLEAR_HZ = 140.0
FEMALE_CLEAR_HZ = 190.0
MIN_VOICED_SECONDS = 0.4
CLEAR_VOICED_SECONDS = 1.0
# Analysing more than this per speaker adds time, not accuracy
MAX_SECONDS_PER_SPEAKER = 90.0

SOURCE_VOICE = "voice"


# Audio is analysed in pieces of this length, to keep memory small on a Raspberry Pi
CHUNK_SECONDS = 15.0


def voiced_f0(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """F0 in Hz of every voiced frame of `samples` (mono float), by autocorrelation."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    chunk = int(CHUNK_SECONDS * sample_rate)
    if x.size > chunk:
        return np.concatenate([_voiced_f0_chunk(x[i:i + chunk], sample_rate) for i in range(0, x.size, chunk)])
    return _voiced_f0_chunk(x, sample_rate)


def _voiced_f0_chunk(x: np.ndarray, sample_rate: int) -> np.ndarray:
    n = int(round(FRAME_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    if n <= 0 or hop <= 0 or x.size < n:
        return np.zeros(0, dtype=np.float32)

    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)

    # Skip near-silent frames: well below the loudness of this stretch of speech
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = rms > max(0.005, 0.1 * float(np.percentile(rms, 90)))
    frames = frames[loud]
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    frames = frames * np.hanning(n).astype(np.float32)[None, :]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(frames, size, axis=1)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, :n]
    energy = ac[:, :1] + 1e-9
    ac = ac / energy

    lo = max(1, int(sample_rate / F0_MAX_HZ))
    hi = min(n - 1, int(sample_rate / F0_MIN_HZ))
    if hi <= lo:
        return np.zeros(0, dtype=np.float32)
    window = ac[:, lo:hi]
    best = np.argmax(window, axis=1)
    peak = window[np.arange(window.shape[0]), best]
    lag = (best + lo).astype(np.float32)

    # Octave check: a strong peak at half the lag means the true period is the shorter one
    half = np.round(lag / 2).astype(int)
    can_halve = half >= lo
    half_peak = np.where(can_halve, ac[np.arange(ac.shape[0]), np.clip(half, 0, n - 1)], -1.0)
    use_half = can_halve & (half_peak >= 0.85 * peak)
    lag = np.where(use_half, half.astype(np.float32), lag)

    voiced = peak >= VOICING_THRESHOLD
    return (sample_rate / lag[voiced]).astype(np.float32)


def _speaker_audio(
    samples: np.ndarray,
    sample_rate: int,
    timeline: Sequence[Dict[str, Any]],
    speaker_id: str,
) -> np.ndarray:
    pieces = []
    total = 0
    budget = int(MAX_SECONDS_PER_SPEAKER * sample_rate)
    for turn in timeline:
        if str(turn.get("speaker")) != speaker_id:
            continue
        a = max(0, int(float(turn.get("start", 0.0)) * sample_rate))
        b = min(samples.size, int(float(turn.get("end", 0.0)) * sample_rate))
        if b - a <= 0:
            continue
        pieces.append(samples[a:b])
        total += b - a
        if total >= budget:
            break
    return np.concatenate(pieces)[:budget] if pieces else np.zeros(0, dtype=np.float32)


def classify_f0(median_f0: float, voiced_seconds: float) -> Dict[str, Any]:
    """`{"gender": male|female|unknown, "confidence": high|medium|low}` for a speaker's median F0."""
    if voiced_seconds < MIN_VOICED_SECONDS or not np.isfinite(median_f0):
        return {"gender": "unknown", "confidence": "low"}
    if median_f0 < MALE_BELOW_HZ:
        clear = median_f0 < MALE_CLEAR_HZ and voiced_seconds >= CLEAR_VOICED_SECONDS
        return {"gender": "male", "confidence": "high" if clear else "medium"}
    if FEMALE_ABOVE_HZ < median_f0 <= ADULT_MAX_HZ:
        clear = median_f0 > FEMALE_CLEAR_HZ and voiced_seconds >= CLEAR_VOICED_SECONDS
        return {"gender": "female", "confidence": "high" if clear else "medium"}
    return {"gender": "unknown", "confidence": "low"}


def estimate_speaker_genders(
    samples: np.ndarray,
    sample_rate: int,
    timeline: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per diarized speaker: `{"gender", "confidence", "median_f0", "voiced_seconds"}`."""
    out: Dict[str, Dict[str, Any]] = {}
    for speaker_id in dict.fromkeys(str(t.get("speaker")) for t in timeline if t.get("speaker")):
        f0 = voiced_f0(_speaker_audio(samples, sample_rate, timeline, speaker_id), sample_rate)
        voiced_seconds = float(f0.size) * HOP_SECONDS
        median = float(np.median(f0)) if f0.size else float("nan")
        result = classify_f0(median, voiced_seconds)
        result["median_f0"] = round(median, 1) if f0.size else None
        result["voiced_seconds"] = round(voiced_seconds, 2)
        out[speaker_id] = result
    return out


def apply_voice_genders(
    speakers: Dict[str, Dict[str, Any]],
    estimates: Dict[str, Dict[str, Any]],
) -> int:
    """Fill in the gender of speakers nothing else has named, from `estimates`. Returns how many."""
    filled = 0
    for speaker_id, est in estimates.items():
        profile = speakers.get(speaker_id)
        if not isinstance(profile, dict):
            continue
        if est.get("median_f0") is not None:
            profile["voice_f0"] = est["median_f0"]
        current = str(profile.get("gender") or "unknown").strip().lower()
        known_from_elsewhere = current in ("male", "female") and profile.get("gender_source") != SOURCE_VOICE
        if known_from_elsewhere or est.get("gender") not in ("male", "female"):
            continue
        profile["gender"] = est["gender"]
        profile["gender_source"] = SOURCE_VOICE
        profile["voice_confidence"] = est.get("confidence", "medium")
        filled += 1
    return filled


def label_speakers_by_voice(
    wav_path: Path,
    timeline: Sequence[Dict[str, Any]],
    speakers: Dict[str, Dict[str, Any]],
) -> int:
    """Read `wav_path` and fill in speaker genders from voice pitch (see `apply_voice_genders`)."""
    from app.pipeline.diarization import _read_wav_float32

    if not timeline or not speakers:
        return 0
    samples, sample_rate = _read_wav_float32(Path(wav_path))
    estimates = estimate_speaker_genders(samples, sample_rate, timeline)
    filled = apply_voice_genders(speakers, estimates)
    if estimates:
        summary = ", ".join(
            f"{sid}: {e.get('median_f0')} Hz -> {e['gender']}" for sid, e in estimates.items()
        )
        logger.info(f"Voice pitch per speaker ({filled} gender(s) set): {summary}")
    return filled


def is_voice_gender(profile: Optional[Dict[str, Any]]) -> bool:
    return isinstance(profile, dict) and profile.get("gender_source") == SOURCE_VOICE
