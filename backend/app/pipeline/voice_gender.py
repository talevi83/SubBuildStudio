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
# and its spectrum looks like clean harmonic speech rather than orchestral music or crowd noise
VOICING_THRESHOLD = 0.55
MAX_SPECTRAL_FLATNESS = 0.22
MIN_SPEECH_BAND_RATIO = 0.65
MAX_RELATIVE_IQR = 0.55
TURN_EDGE_TRIM_SECONDS = 0.12

MALE_BELOW_HZ = 150.0
FEMALE_ABOVE_HZ = 195.0
ADULT_MAX_HZ = 285.0
# Clearly inside one range: trusted more
MALE_CLEAR_HZ = 135.0
FEMALE_CLEAR_HZ = 215.0
MIN_VOICED_SECONDS = 0.6
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
    loud = rms > max(0.005, 0.12 * float(np.percentile(rms, 90)))
    frames = frames[loud]
    loud_rms = rms[loud]
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    # Exclude extreme shouting / orchestral crescendo bursts when normal speech frames exist
    if loud_rms.size >= 8:
        calm = loud_rms <= 2.6 * float(np.median(loud_rms))
        if int(np.sum(calm)) >= max(4, loud_rms.size // 2):
            frames = frames[calm]

    win = np.hanning(n).astype(np.float32)
    frames = frames * win[None, :]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(frames, size, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) + 1e-12
    freqs = np.fft.rfftfreq(size, d=1.0 / sample_rate)
    flat_band = (freqs >= 60.0) & (freqs <= min(4000.0, sample_rate * 0.5))
    if np.any(flat_band):
        geom = np.exp(np.mean(np.log(power[:, flat_band]), axis=1))
        arith = np.mean(power[:, flat_band], axis=1)
        flatness = geom / np.maximum(arith, 1e-12)
    else:
        flatness = np.zeros(frames.shape[0], dtype=np.float32)

    speech_band = (freqs >= 80.0) & (freqs <= min(3200.0, sample_rate * 0.5))
    total_band = freqs >= 20.0
    if np.any(speech_band) and np.any(total_band):
        speech_ratio = np.sum(power[:, speech_band], axis=1) / np.maximum(np.sum(power[:, total_band], axis=1), 1e-12)
    else:
        speech_ratio = np.ones(frames.shape[0], dtype=np.float32)

    # Boersma window-normalized autocorrelation: dividing by the Hanning window's own autocorrelation
    # removes the lag-taper bias so high-passed/shouted male voices with strong 2nd harmonics
    # (e.g. 111 Hz -> 222 Hz, 123 Hz -> 246 Hz) are not octave-doubled into the female range.
    ac = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, :n]
    energy = ac[:, :1] + 1e-9
    ac = ac / energy
    win_spec = np.fft.rfft(win, size)
    win_ac = np.fft.irfft(win_spec * np.conj(win_spec), size)[:n]
    win_ac = np.maximum(win_ac / max(float(win_ac[0]), 1e-9), 0.15).astype(np.float32)
    ac = np.clip(ac / win_ac[None, :], -1.0, 1.0)

    lo = max(1, int(sample_rate / F0_MAX_HZ))
    hi = min(n - 1, int(sample_rate / F0_MIN_HZ))
    if hi <= lo:
        return np.zeros(0, dtype=np.float32)
    window = ac[:, lo:hi]
    best = np.argmax(window, axis=1)
    peak = window[np.arange(window.shape[0]), best]
    lag = (best + lo).astype(np.float32)

    # Sub-harmonic check: only halve the lag when half-lag is virtually as strong as the full period
    half = np.round(lag / 2).astype(int)
    can_halve = half >= lo
    half_peak = np.where(can_halve, ac[np.arange(ac.shape[0]), np.clip(half, 0, n - 1)], -1.0)
    use_half = can_halve & (half_peak >= 0.96 * peak)
    lag = np.where(use_half, half.astype(np.float32), lag)

    voiced = (
        (peak >= VOICING_THRESHOLD)
        & (flatness <= MAX_SPECTRAL_FLATNESS)
        & (speech_ratio >= MIN_SPEECH_BAND_RATIO)
    )
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
    other_spans = [
        (float(t.get("start", 0.0)), float(t.get("end", 0.0)))
        for t in timeline
        if t.get("speaker") and str(t.get("speaker")) != speaker_id
    ]
    for turn in timeline:
        if str(turn.get("speaker")) != speaker_id:
            continue
        start_s = float(turn.get("start", 0.0))
        end_s = float(turn.get("end", 0.0))
        if end_s <= start_s:
            continue
        # Skip turns that directly overlap another speaker's turn
        if any(os < end_s - 0.02 and oe > start_s + 0.02 for os, oe in other_spans):
            continue
        # Trim boundary cross-talk when another speaker spoke right before or after
        if end_s - start_s >= 0.6:
            if any(0.0 <= start_s - oe <= 0.35 for _, oe in other_spans):
                start_s += TURN_EDGE_TRIM_SECONDS
            if any(0.0 <= os - end_s <= 0.35 for os, _ in other_spans):
                end_s -= TURN_EDGE_TRIM_SECONDS
        a = max(0, int(start_s * sample_rate))
        b = min(samples.size, int(end_s * sample_rate))
        if b - a <= 0:
            continue
        pieces.append(samples[a:b])
        total += b - a
        if total >= budget:
            break
    return np.concatenate(pieces)[:budget] if pieces else np.zeros(0, dtype=np.float32)


def classify_f0(median_f0: float, voiced_seconds: float, iqr_f0: Optional[float] = None) -> Dict[str, Any]:
    """`{"gender": male|female|unknown, "confidence": high|medium|low}` for a speaker's median F0."""
    if voiced_seconds < MIN_VOICED_SECONDS or not np.isfinite(median_f0) or median_f0 <= 0.0:
        return {"gender": "unknown", "confidence": "low"}
    if iqr_f0 is not None and np.isfinite(iqr_f0) and (iqr_f0 / median_f0) > MAX_RELATIVE_IQR:
        return {"gender": "unknown", "confidence": "low"}
    if median_f0 < MALE_BELOW_HZ:
        clear = median_f0 < MALE_CLEAR_HZ and voiced_seconds >= CLEAR_VOICED_SECONDS
        return {"gender": "male", "confidence": "high" if clear else "medium"}
    if FEMALE_ABOVE_HZ <= median_f0 <= ADULT_MAX_HZ:
        clear = median_f0 >= FEMALE_CLEAR_HZ and voiced_seconds >= CLEAR_VOICED_SECONDS
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
        iqr = float(np.percentile(f0, 75) - np.percentile(f0, 25)) if f0.size >= 4 else None
        result = classify_f0(median, voiced_seconds, iqr_f0=iqr)
        result["median_f0"] = round(median, 1) if f0.size else None
        result["voiced_seconds"] = round(voiced_seconds, 2)
        result["iqr_rel"] = round(iqr / median, 3) if (iqr is not None and f0.size and median > 0) else None
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
        iqr_rel = est.get("iqr_rel")
        # A pitch spread this wide means the cluster pools several people (men and women, or a
        # crowd): its id must not be trusted to say who is speaking or listening.
        if (
            iqr_rel is not None
            and iqr_rel > MAX_RELATIVE_IQR
            and float(est.get("voiced_seconds") or 0.0) >= CLEAR_VOICED_SECONDS
        ):
            profile["voice_mixed"] = True
        else:
            profile.pop("voice_mixed", None)
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
