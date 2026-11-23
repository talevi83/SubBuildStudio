"""Speaker gender from voice pitch: estimation, its limits, and how it ranks against the text."""
import numpy as np
import pytest

from app.pipeline.addressee import HIGH, resolve_addressees
from app.pipeline.voice_gender import (
    apply_voice_genders,
    classify_f0,
    estimate_speaker_genders,
    voiced_f0,
)

SR = 16000


def _voice(f0: float, seconds: float, seed: int = 0) -> np.ndarray:
    """A crude voiced sound: a harmonic series at `f0` with a little vibrato and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))) / SR
    wave = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 6))
    return (0.3 * wave + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.mark.parametrize("f0", [110.0, 125.0, 210.0, 240.0])
def test_f0_is_measured_without_octave_errors(f0):
    measured = np.median(voiced_f0(_voice(f0, 1.5), SR))
    assert abs(measured - f0) / f0 < 0.05


def test_silence_has_no_pitch():
    assert voiced_f0(np.zeros(SR, dtype=np.float32), SR).size == 0
    assert voiced_f0(np.zeros(100, dtype=np.float32), SR).size == 0


@pytest.mark.parametrize("median,seconds,expected", [
    (120.0, 2.0, ("male", "high")),
    (150.0, 2.0, ("male", "medium")),       # close to the overlap zone
    (165.0, 2.0, ("unknown", "low")),       # men and women both speak here
    (180.0, 2.0, ("female", "medium")),
    (230.0, 2.0, ("female", "high")),
    (230.0, 0.6, ("female", "medium")),     # little speech: less sure
    (230.0, 0.2, ("unknown", "low")),       # too little to judge
    (340.0, 3.0, ("unknown", "low")),       # a child's or a shouting voice
])
def test_classification_and_its_limits(median, seconds, expected):
    got = classify_f0(median, seconds)
    assert (got["gender"], got["confidence"]) == expected


def test_each_speaker_is_judged_on_their_own_turns():
    audio = np.concatenate([_voice(118, 2.0, 1), np.zeros(SR // 2, np.float32), _voice(235, 2.0, 2)])
    timeline = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}, {"start": 2.5, "end": 4.5, "speaker": "SPEAKER_02"}]
    est = estimate_speaker_genders(audio, SR, timeline)
    assert est["SPEAKER_01"]["gender"] == "male" and est["SPEAKER_02"]["gender"] == "female"


def test_voice_never_overrides_a_gender_from_the_text_but_can_be_refreshed():
    speakers = {
        "SPEAKER_01": {"label": "Dana", "gender": "female"},                       # named in the glossary
        "SPEAKER_02": {"label": "Speaker 2", "gender": "unknown"},
        "SPEAKER_03": {"label": "Speaker 3", "gender": "male", "gender_source": "voice"},
    }
    estimates = {
        "SPEAKER_01": {"gender": "male", "confidence": "high", "median_f0": 130.0},
        "SPEAKER_02": {"gender": "female", "confidence": "medium", "median_f0": 182.0},
        "SPEAKER_03": {"gender": "female", "confidence": "high", "median_f0": 220.0},
    }
    assert apply_voice_genders(speakers, estimates) == 2
    assert speakers["SPEAKER_01"]["gender"] == "female" and "gender_source" not in speakers["SPEAKER_01"]
    assert speakers["SPEAKER_01"]["voice_f0"] == 130.0
    assert speakers["SPEAKER_02"] == {"label": "Speaker 2", "gender": "female", "gender_source": "voice",
                                      "voice_confidence": "medium", "voice_f0": 182.0}
    assert speakers["SPEAKER_03"]["gender"] == "female"


def test_a_listener_known_only_by_voice_is_trusted_a_little_less():
    lines = [
        {"index": 1, "start_time": 0.0, "end_time": 1.5, "original_text": "Did you call?", "speaker": "SPEAKER_01"},
        {"index": 2, "start_time": 2.0, "end_time": 3.5, "original_text": "Yes, about your car.", "speaker": "SPEAKER_02"},
    ]
    from_text = {"SPEAKER_01": {"label": "Speaker 1", "gender": "male"},
                 "SPEAKER_02": {"label": "Speaker 2", "gender": "female"}}
    sure_voice = {sid: {**p, "gender_source": "voice", "voice_confidence": "high"} for sid, p in from_text.items()}
    unsure_voice = {sid: {**p, "gender_source": "voice", "voice_confidence": "medium"} for sid, p in from_text.items()}
    by_text = resolve_addressees(lines, from_text, [])
    by_sure = resolve_addressees(lines, sure_voice, [])
    by_unsure = resolve_addressees(lines, unsure_voice, [])
    assert by_text[1].form == by_sure[1].form == by_unsure[1].form == "f"
    assert by_text[1].confidence > by_sure[1].confidence >= HIGH        # still told as a fact
    assert by_unsure[1].confidence < HIGH                                # only a guess
