"""Tests for Smart Audio Map, per-stage thinking budgets, static prefix caching, Flex re-probe, and final-scene gender."""
import json
import types
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import app.core.model_catalog as mc
import app.providers.gemini_calls as gc
from app.core.config import settings
from app.core.user_settings import GeminiCallSettings, _defaults, load_settings, save_settings
from app.pipeline.addressee import resolve_addressees
from app.pipeline.context import PipelineContext
from app.pipeline.speaker_match import SpeakerMatchResponse, _call_ai_speaker_matcher
from app.pipeline.stages.transcribe_stage import TranscribeStage
from app.pipeline.stages.translate_stage import TranslateStage
from app.pipeline.stages.vocal_isolation_stage import VocalIsolationStage
from app.providers.llm.client import _anthropic_system_blocks
from app.providers.translation.base import TranslatableSegment
from app.providers.translation.gemini_provider import GeminiTranslator, default_batch_size_for_model
from app.services.smart_audio_map import (
    _read_mono_float32,
    analyze_audio_timeline,
    condition_audio_by_map,
    resolve_audio_modes,
)


def _write_mono_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(samples * 32767.0), -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# 1. Smart Audio Map: selective comms & whisper detection + return to OFF
# ---------------------------------------------------------------------------

def test_smart_audio_map_detects_comms_and_whispers_and_returns_to_off(tmp_path):
    rng = np.random.default_rng(42)
    sr = 16000
    dur = 16.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    env = (0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)).astype(np.float32)  # 4 Hz syllabic modulation

    audio = np.zeros_like(t)

    # 0.0 - 3.2s: Normal full-band male speech (-15 dBFS)
    s0, e0 = 0, int(3.2 * sr)
    audio[s0:e0] = 0.22 * env[s0:e0] * (
        0.55 * np.sin(2.0 * np.pi * 120.0 * t[s0:e0])
        + 0.30 * np.sin(2.0 * np.pi * 900.0 * t[s0:e0])
        + 0.25 * np.sin(2.0 * np.pi * 5200.0 * t[s0:e0])
    )

    # 3.6 - 5.2s: Muffled indoor/medieval speech (-22 dBFS, clean pure formants in 600-2200 Hz, NO static hiss)
    # Must NOT false-positive as radio comms even when detect_comms=True!
    sm, em = int(3.6 * sr), int(5.2 * sr)
    audio[sm:em] = 0.09 * env[sm:em] * (
        0.65 * np.sin(2.0 * np.pi * 650.0 * t[sm:em])
        + 0.35 * np.sin(2.0 * np.pi * 1650.0 * t[sm:em])
    )

    # 6.8 - 9.0s: Genuine static-laced narrowband walkie-talkie transmission (350-3200 Hz brick-wall + carrier hiss)
    s1, e1 = int(6.8 * sr), int(9.0 * sr)
    raw_radio = 0.09 * env[s1:e1] * (
        0.55 * np.sin(2.0 * np.pi * 850.0 * t[s1:e1])
        + 0.35 * np.sin(2.0 * np.pi * 1950.0 * t[s1:e1])
        + 0.55 * rng.standard_normal(e1 - s1).astype(np.float32)
    )
    n_fft = 1 << int(np.ceil(np.log2(e1 - s1)))
    rfreqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    rband = ((rfreqs >= 340.0) & (rfreqs <= 3250.0)).astype(np.float32)
    audio[s1:e1] = np.fft.irfft(np.fft.rfft(raw_radio, n=n_fft) * rband, n=n_fft)[: e1 - s1].astype(np.float32)

    # 9.8 - 11.8s: Faint whispered speech (-38 dBFS, >10 dB below normal dialogue)
    s2, e2 = int(9.8 * sr), int(11.8 * sr)
    audio[s2:e2] = 0.015 * env[s2:e2] * (
        0.5 * np.sin(2.0 * np.pi * 1500.0 * t[s2:e2])
        + 0.5 * np.sin(2.0 * np.pi * 4800.0 * t[s2:e2])
    )

    # 12.8 - 16.0s: Normal full-band speech again (must remain 100% OFF / untouched)
    s3, e3 = int(12.8 * sr), int(16.0 * sr)
    audio[s3:e3] = 0.20 * env[s3:e3] * (
        0.55 * np.sin(2.0 * np.pi * 130.0 * t[s3:e3])
        + 0.30 * np.sin(2.0 * np.pi * 1100.0 * t[s3:e3])
        + 0.25 * np.sin(2.0 * np.pi * 5100.0 * t[s3:e3])
    )

    in_wav = tmp_path / "movie.wav"
    out_wav = tmp_path / "conditioned.wav"
    _write_mono_wav(in_wav, audio, sr)

    fake_vad = types.ModuleType("faster_whisper.vad")
    fake_vad.VadOptions = lambda **kw: kw
    fake_vad.get_speech_timestamps = lambda samples, opts: [
        {"start": s0, "end": e0},
        {"start": sm, "end": em},
        {"start": s1, "end": e1},
        {"start": s2, "end": e2},
        {"start": s3, "end": e3},
    ]
    with patch.dict("sys.modules", {"faster_whisper.vad": fake_vad}):
        # Default `auto` scan (`detect_comms=False`): zero comms windows, only the genuine whisper is flagged
        auto_windows = analyze_audio_timeline(in_wav)
        assert {w.kind for w in auto_windows} == {"whisper"}

        # Explicit `detect_comms=True`: flags the static walkie-talkie [6.5-8.8s], NOT muffled medieval speech [3.8-5.8s]!
        windows = analyze_audio_timeline(in_wav, detect_whispers=True, detect_comms=True, detect_music_masked=False)

    kinds = {w.kind for w in windows}
    assert "comms" in kinds
    assert "whisper" in kinds
    assert all(w.start >= 6.0 and w.end <= 12.4 for w in windows)

    report = condition_audio_by_map(in_wav, out_wav, windows)
    assert out_wav.is_file()
    orig, _ = _read_mono_float32(in_wav)
    cond, _ = _read_mono_float32(out_wav)

    # Clean speech [0-3.2s], muffled medieval speech [3.6-5.2s], and ending speech [13.0-16.0s] stay 100% OFF
    assert np.allclose(orig[: int(5.5 * sr)], cond[: int(5.5 * sr)], atol=1e-4)
    assert np.allclose(orig[int(12.9 * sr) :], cond[int(12.9 * sr) :], atol=1e-4)
    # Inside the whisper window, pure linear gain lifted RMS energy
    orig_whisper_rms = float(np.sqrt(np.mean(orig[s2:e2] ** 2)))
    cond_whisper_rms = float(np.sqrt(np.mean(cond[s2:e2] ** 2)))
    assert cond_whisper_rms > orig_whisper_rms * 1.4
    assert "untouched (OFF)" in report.summary_line()


@pytest.mark.asyncio
async def test_auto_comms_mode_never_injects_military_hotwords_and_preserves_raw_diarization_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    raw_wav = tmp_path / "raw.wav"
    sr = 16000
    t = np.linspace(0.0, 5.0, 5 * sr, endpoint=False, dtype=np.float32)
    env = (0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)).astype(np.float32)
    sig = np.zeros_like(t)
    # 0.0 - 2.2s: normal reference dialogue (-16 dBFS)
    sig[: int(2.2 * sr)] = 0.20 * env[: int(2.2 * sr)] * np.sin(2.0 * np.pi * 900.0 * t[: int(2.2 * sr)])
    # 2.6 - 4.8s: faint whisper (-38 dBFS)
    sig[int(2.6 * sr) : int(4.8 * sr)] = 0.015 * env[int(2.6 * sr) : int(4.8 * sr)] * (
        0.5 * np.sin(2.0 * np.pi * 1600.0 * t[int(2.6 * sr) : int(4.8 * sr)])
        + 0.5 * np.sin(2.0 * np.pi * 4500.0 * t[int(2.6 * sr) : int(4.8 * sr)])
    )
    _write_mono_wav(raw_wav, sig, sr)

    fake_vad = types.ModuleType("faster_whisper.vad")
    fake_vad.VadOptions = lambda **kw: kw
    fake_vad.get_speech_timestamps = lambda samples, opts: [
        {"start": 0, "end": int(2.2 * sr)},
        {"start": int(2.6 * sr), "end": int(4.8 * sr)},
    ]

    ctx = PipelineContext(
        job_id="job_smart_auto",
        video_path=str(tmp_path / "film.mkv"),
        file_name="film.mkv",
        audio_wav_path=str(raw_wav),
        vocal_isolation=False,
        vocal_isolation_mode="off",
        comms_boost=False,
        comms_boost_mode="auto",
    )
    with patch.dict("sys.modules", {"faster_whisper.vad": fake_vad}):
        await VocalIsolationStage().execute(ctx)
    # Raw audio stays intact for WeSpeaker & F0 pitch; conditioned WAV feeds TranscribeStage; zero comms windows
    assert ctx.raw_audio_wav_path == str(raw_wav)
    assert ctx.audio_wav_path != str(raw_wav)
    assert "comms=" not in (ctx.smart_audio_summary or "")
    assert "whisper=" in (ctx.smart_audio_summary or "")
    assert Path(ctx.audio_wav_path).is_file()

    captured_kwargs = {}

    class _DummyTranscriber:
        supports_gap_recovery = False

        async def transcribe(self, audio_path, language=None, on_segment_callback=None, cancel_flag=None):
            captured_kwargs["audio_path"] = audio_path
            return []

    def _fake_create_transcriber(provider_name, **kwargs):
        captured_kwargs.update(kwargs)
        return _DummyTranscriber(), "whisperx", None

    with patch("app.providers.transcription.create_transcriber", side_effect=_fake_create_transcriber):
        await TranscribeStage().execute(ctx)

    assert captured_kwargs["comms_boost"] is False  # NO military hotwords injected in 'auto' mode!
    assert captured_kwargs["high_speech_sensitivity"] is True
    assert captured_kwargs["audio_path"] == ctx.audio_wav_path

    v_m, c_m = resolve_audio_modes(vocal_isolation=True, vocal_isolation_mode=None, comms_boost=False, comms_boost_mode="auto")
    assert (v_m, c_m) == ("on", "auto")


# ---------------------------------------------------------------------------
# 2. Per-Stage Thinking Defaults & Model-Aware Batch Sizing + Static Prefix Caching
# ---------------------------------------------------------------------------

def test_per_stage_thinking_defaults_and_non_google_validation():
    defs = _defaults()
    assert defs.translation.thinking in ("minimal", "low")
    assert defs.glossary.thinking == "default"
    assert defs.vision.thinking in ("minimal", "low")

    openai_call = GeminiCallSettings(provider="openai", model="gpt-5.4", thinking="minimal")
    assert openai_call.thinking == "minimal"
    with pytest.raises(ValueError, match="Unsupported thinking level"):
        GeminiCallSettings(provider="openai", model="gpt-5.4", thinking="ultra")


def test_dynamic_batch_sizing_by_model_capability():
    assert default_batch_size_for_model("google", "gemini-3.8-flash", 35) == 35
    assert default_batch_size_for_model("google", "gemini-4-carbon", 35) == 35
    assert default_batch_size_for_model("openai", "gpt-5.4", 35) == 35
    assert default_batch_size_for_model("openai", "gpt-6-terra", 35) == 35
    assert default_batch_size_for_model("anthropic", "claude-sonnet-5-5", 35) >= 35
    assert default_batch_size_for_model("anthropic", "claude-opus-5-5", 35) >= 35

    # Compact / small models stay at 18 lines
    assert default_batch_size_for_model("google", "gemini-3.5-flash-lite", 35) == 18
    assert default_batch_size_for_model("openai", "gpt-5-mini", 35) == 18
    assert default_batch_size_for_model("openai", "gpt-5.6-luna", 35) == 18
    assert default_batch_size_for_model("ollama", "qwen2.5:14b", 35) == 18


@pytest.mark.asyncio
async def test_static_system_prompt_prefix_across_batches_even_in_cost_saver_mode(tmp_path, monkeypatch):
    (settings.storage_dir / "settings.json").unlink(missing_ok=True)
    s = load_settings()
    s.cost_saver_mode = True
    save_settings(s)
    try:
        ctx = PipelineContext(job_id="j_cache", video_path="v.mp4", file_name="v.mp4", gemini_api_key_override="k")
        tr = TranslateStage()._make_translator(ctx)
        assert isinstance(tr, GeminiTranslator)
        # Cloud translator keeps prune_glossary=False so prefix caching works across all batches
        assert tr.prune_glossary is False

        seen_sys = []
        seen_payloads = []

        class _FakeModels:
            def generate_content(self, model, contents, config):
                seen_sys.append(config.system_instruction)
                payload = json.loads(contents)
                seen_payloads.append(payload)
                return types.SimpleNamespace(
                    text=json.dumps({
                        "translations": [
                            {"id": it["id"], "text": f"תרגום מלא של שורה מספר {it['id']}", "addressee_form": "m"}
                            for it in payload["target_batch"]
                        ]
                    })
                )

        tr.batch_size = 2
        tr.client = types.SimpleNamespace(models=_FakeModels())
        glossary = {
            "summary": "Medieval rebellion drama",
            "characters": [
                {"name": "Wat Tyler", "target_name": "וואט טיילר", "gender": "male", "notes": "leader"},
                {"name": "King Richard", "target_name": "המלך ריצ׳רד", "gender": "male", "notes": "king"},
            ],
            "terms": [{"source": "Domesday", "translation": "ספר יום הדין", "notes": ""}],
            "register_notes": "Elevated dramatic Hebrew",
        }
        segs = [
            TranslatableSegment(1, 0, 1, "Wat Tyler approaches."),
            TranslatableSegment(2, 1, 2, "Hold the line."),
            TranslatableSegment(3, 2, 3, "King Richard speaks."),
            TranslatableSegment(4, 3, 4, "Bring the Domesday book."),
        ]
        await tr.translate_segments(
            segs,
            "English",
            "Hebrew",
            context_notes="1381 England.",
            glossary=glossary,
            batch_instruction="Use formal register on these lines.",
        )

        assert len(seen_sys) == 2
        # Byte-for-byte identical system_instruction across batches!
        assert seen_sys[0] == seen_sys[1]
        # Per-batch instruction lives in the user payload, not mutating system_instruction
        assert seen_payloads[0]["batch_instruction"] == "Use formal register on these lines."

        # Verify Anthropic caches the ENTIRE static system + locked glossary inside single cached block[0]
        blocks = _anthropic_system_blocks(seen_sys[0], "\nSCHEMA")
        assert len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "וואט טיילר" in blocks[0]["text"]
    finally:
        (settings.storage_dir / "settings.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Smarter Flex Cooldown & Periodic Single-Attempt Re-Probe + Tier Accounting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flex_cooldown_reprobes_after_request_budget_and_recovers(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(gc.asyncio, "sleep", _no_sleep)
    busy = Exception("503 UNAVAILABLE. High demand.")
    state = gc.FlexState(limit=2, cooldown_requests=3, cooldown_seconds=600.0)
    usage = mc.UsageTracker()

    tiers_called = []
    # Script:
    # Req 1: 3x Flex busy -> 1x Standard fallback (consecutive=1)
    # Req 2: 3x Flex busy -> 1x Standard fallback (consecutive=2 -> enters 3-request cooldown)
    # Req 3, 4, 5: skipped straight to Standard (3 cooldown skips)
    # Req 6: single-attempt Flex re-probe -> busy -> immediate Standard fallback (no 3x retry!) -> new cooldown
    # Req 7, 8, 9: skipped straight to Standard
    # Req 10: single-attempt Flex re-probe -> SUCCEEDS!
    # Req 11: normal Flex -> SUCCEEDS!
    outcomes = (
        [busy, busy, busy, None]
        + [busy, busy, busy, None]
        + [None, None, None]
        + [busy, None]
        + [None, None, None]
        + [None]
        + [None]
    )

    class _Client:
        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config):
            tier = getattr(getattr(config, "service_tier", None), "value", getattr(config, "service_tier", None))
            tiers_called.append(tier)
            nxt = outcomes.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            meta = types.SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=200,
                thoughts_token_count=0,
                traffic_type="ON_DEMAND_FLEX" if tier == "flex" else "ON_DEMAND",
            )
            return types.SimpleNamespace(text="{}", usage_metadata=meta)

    client = _Client()
    from google.genai import types as gt

    cfg = gt.GenerateContentConfig(temperature=0.2)
    for _ in range(11):
        await gc.generate_content(
            client, model="gemini-3.8-flash", contents="hi", config=cfg, flex=True, usage=usage, stage="translation", flex_state=state
        )

    assert outcomes == []
    # Req 6 re-probe only made ONE flex attempt before falling back to Standard (None), then Req 10 & 11 ran on Flex!
    assert tiers_called[-3:] == [None, "flex", "flex"]
    st = usage.stages["translation"]
    assert st.requests == 11
    assert st.flex_requests == 2
    assert st.flex_fallbacks == 9
    assert st.standard_requests == 0


# ---------------------------------------------------------------------------
# 4. SpeakerMatchResponse Truncation Guard & Final-Scene Masculine Continuity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_speaker_matcher_reserves_16k_tokens_and_low_thinking(monkeypatch):
    captured = {}

    async def _fake_call_text_json(spec, **kwargs):
        captured["thinking"] = spec.thinking
        captured["max_output_tokens"] = kwargs.get("max_output_tokens")
        return types.SimpleNamespace(
            text=SpeakerMatchResponse(matches=[]).model_dump_json()
        )

    monkeypatch.setattr("app.providers.llm.call_text_json", _fake_call_text_json)
    matches = await _call_ai_speaker_matcher(
        "DOSSIER",
        target_language="Hebrew",
        api_key_override="fake-key",
    )
    assert matches == []
    assert captured["max_output_tokens"] == 16384
    assert captured["thinking"] in ("minimal", "low")


def test_final_confrontation_scene_preserves_masculine_addressee_across_bystander_and_pooled_lines():
    """Reproduces the exact final scene of job_32b559b5da (Ploughman confronting King Richard II)."""
    speakers = {
        "SPEAKER_10": {"label": "החורש", "character_name": "Ploughman", "gender": "male"},
        "SPEAKER_03": {"label": "המלך ריצ׳רד", "character_name": "King Richard II", "gender": "male"},
        # Minor cluster mis-tagged as female in noisy fallback
        "SPEAKER_05": {"label": "ג׳ואן", "character_name": "Joan", "gender": "female"},
    }
    segments = [
        {"index": 1168, "start_time": 7182.0, "end_time": 7184.0, "speaker": "SPEAKER_10", "original_text": "Have something of yours."},
        {"index": 1169, "start_time": 7184.5, "end_time": 7186.0, "speaker": "SPEAKER_03", "original_text": "What is that?"},
        {"index": 1170, "start_time": 7186.4, "end_time": 7189.0, "speaker": "SPEAKER_10", "original_text": "Look closer. It has your seal."},
        {"index": 1171, "start_time": 7189.5, "end_time": 7191.5, "speaker": "SPEAKER_03", "original_text": "Give it back to me."},
        {"index": 1180, "start_time": 7193.0, "end_time": 7194.8, "speaker": "SPEAKER_10", "original_text": "Kill the king."},
        {"index": 1181, "start_time": 7195.2, "end_time": 7197.0, "speaker": "SPEAKER_10", "original_text": "We will wait."},
        {"index": 1183, "start_time": 7197.4, "end_time": 7200.0, "speaker": "SPEAKER_10", "original_text": "Out in the forest, where you can't see us."},
        # Single 2-word bystander interjection mis-attributed to SPEAKER_05
        {"index": 1184, "start_time": 7202.2, "end_time": 7203.5, "speaker": "SPEAKER_05", "original_text": "No kings."},
        # Pooled/quarantined speaker (`speaker=None`) continuing to address the King
        {"index": 1186, "start_time": 7204.2, "end_time": 7207.5, "speaker": None, "original_text": "And every night, we will come for you in your dreams."},
        {"index": 1187, "start_time": 7208.0, "end_time": 7211.0, "speaker": None, "original_text": "And you will never have peace."},
    ]

    resolved = resolve_addressees(segments, speakers)
    # #1183 must stay addressed to King Richard (male), NOT hijacked by SPEAKER_05's 2-word "No kings."
    assert resolved[1183].gender == "male"
    assert resolved[1183].number == "singular"
    # #1186 and #1187 (quarantined speaker=None) inherit masculine singular from the confrontation scene
    assert 1186 in resolved and resolved[1186].gender == "male"
    assert 1187 in resolved and resolved[1187].gender == "male"
