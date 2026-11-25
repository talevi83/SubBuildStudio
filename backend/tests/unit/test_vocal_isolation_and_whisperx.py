"""Unit and integration tests for optional vocal isolation and WhisperX transcription provider."""
import contextlib
import sys
import types
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.hardware import get_hardware_info
from app.models.db_models import JobStatus
from app.pipeline.context import PipelineContext, RawSegment
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.stages.diarize_stage import DiarizeStage
from app.pipeline.stages.transcribe_stage import TranscribeStage
from app.pipeline.stages.vocal_isolation_stage import VocalIsolationStage
from app.providers.transcription import create_transcriber
from app.providers.transcription.faster_whisper import FasterWhisperTranscriber
from app.providers.transcription.whisperx_provider import WhisperXProvider, WhisperXUnavailableError
from app.services import audio_isolation


def _write_test_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(160, int(round(duration_sec * sample_rate)))
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
    wave_data = (0.35 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    pcm = (np.clip(wave_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class _FakeTensor:
    """Lightweight numpy-backed stand-in for PyTorch tensors used in audio isolation tests."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.asarray(arr, dtype=np.float32)

    @property
    def shape(self):
        return self.arr.shape

    def mean(self, dim=None, keepdim=False):
        val = np.mean(self.arr, axis=dim, keepdims=keepdim)
        return _FakeTensor(val) if isinstance(val, np.ndarray) else _FakeTensor(np.array(val, dtype=np.float32))

    def std(self):
        return float(np.std(self.arr))

    def __float__(self):
        return float(self.arr)

    def __sub__(self, other):
        val = other.arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.arr - val)

    def __add__(self, other):
        val = other.arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.arr + val)

    def __truediv__(self, other):
        val = other.arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.arr / val)

    def __mul__(self, other):
        val = other.arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.arr * val)

    def repeat(self, *reps):
        return _FakeTensor(np.tile(self.arr, reps))

    def unsqueeze(self, dim: int):
        return _FakeTensor(np.expand_dims(self.arr, axis=dim))

    def squeeze(self, dim: int):
        return _FakeTensor(np.squeeze(self.arr, axis=dim))

    def to(self, _device: str):
        return self

    def detach(self):
        return self

    def float(self):
        return _FakeTensor(self.arr.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def __getitem__(self, item):
        return _FakeTensor(self.arr[item])


def _install_fake_torch_and_demucs(monkeypatch, *, raise_in_apply: bool = False):
    calls = {"get_model": 0, "apply_model": 0, "segments": [], "empty_cache": 0, "ipc_collect": 0}

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.from_numpy = lambda arr: _FakeTensor(arr)
    fake_torch.no_grad = lambda: contextlib.nullcontext()

    def _interpolate(tensor, size, mode="linear", align_corners=False):
        arr = tensor.arr
        b, c, t = arr.shape
        x_old = np.linspace(0.0, 1.0, t, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
        out = np.zeros((b, c, int(size)), dtype=np.float32)
        for bi in range(b):
            for ci in range(c):
                out[bi, ci] = np.interp(x_new, x_old, arr[bi, ci])
        return _FakeTensor(out)

    fake_torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=_interpolate))
    fake_torch.amp = types.SimpleNamespace(autocast=lambda *_args, **_kwargs: contextlib.nullcontext())

    def _empty_cache():
        calls["empty_cache"] += 1

    def _ipc_collect():
        calls["ipc_collect"] += 1

    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=_empty_cache,
        ipc_collect=_ipc_collect,
    )

    class _FakeDemucsModel:
        sources = ["drums", "bass", "other", "vocals"]
        samplerate = 44100
        audio_channels = 2
        segment = 7.8

        def eval(self):
            return self

        def to(self, _dev):
            return self

    def _get_model(name):
        calls["get_model"] += 1
        return _FakeDemucsModel()

    def _apply_model(model, batch, shifts=0, split=True, overlap=0.25, progress=False, device="cpu", segment=7.5):
        calls["apply_model"] += 1
        calls["segments"].append(segment)
        if raise_in_apply:
            raise RuntimeError("Simulated Demucs failure")
        b, c, t = batch.shape
        out = np.zeros((b, len(model.sources), c, t), dtype=np.float32)
        # Keep the vocals stem at source index 3
        out[:, 3, :, :] = batch.arr * 0.5
        return _FakeTensor(out)

    fake_demucs = types.ModuleType("demucs")
    fake_demucs_pretrained = types.ModuleType("demucs.pretrained")
    fake_demucs_pretrained.get_model = _get_model
    fake_demucs_apply = types.ModuleType("demucs.apply")
    fake_demucs_apply.apply_model = _apply_model

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "demucs", fake_demucs)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", fake_demucs_pretrained)
    monkeypatch.setitem(sys.modules, "demucs.apply", fake_demucs_apply)
    monkeypatch.setattr(audio_isolation, "is_demucs_available", lambda: True)
    monkeypatch.setattr(audio_isolation, "is_audio_separator_available", lambda: False)
    return calls


# ---------------------------------------------------------------------------
# 1. Registry, Schema & API Validation
# ---------------------------------------------------------------------------

def test_provider_registry_and_fallback():
    fw, name, warn = create_transcriber("faster_whisper", model_size_or_path="tiny", device="cpu", compute_type="int8")
    assert isinstance(fw, FasterWhisperTranscriber)
    assert name == "faster_whisper"
    assert warn is None

    # When whisperx is missing and fallback is allowed, gracefully falls back
    with patch("app.providers.transcription.is_whisperx_available", return_value=False):
        fb, fb_name, fb_warn = create_transcriber("whisperx", model_size_or_path="tiny", allow_fallback=True)
        assert isinstance(fb, FasterWhisperTranscriber)
        assert fb_name == "faster_whisper"
        assert fb_warn and "WhisperX" in fb_warn

        with pytest.raises(WhisperXUnavailableError):
            create_transcriber("whisperx", model_size_or_path="tiny", allow_fallback=False)

    # When whisperx is available, instantiates WhisperXProvider cleanly
    with patch("app.providers.transcription.is_whisperx_available", return_value=True):
        wx, wx_name, wx_warn = create_transcriber("whisperx", model_size_or_path="medium", device="cpu", compute_type="int8")
        assert isinstance(wx, WhisperXProvider)
        assert wx_name == "whisperx"
        assert wx_warn is None

    with pytest.raises(ValueError, match="Unknown transcription provider"):
        create_transcriber("unknown_engine")


def test_api_accepts_whisperx_and_vocal_isolation_and_rejects_invalid_provider(client, tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"fake video bytes")

    bad_resp = client.post(
        "/api/jobs",
        json={
            "video_path": str(media),
            "whisper_model": "tiny",
            "transcription_provider": "invalid_engine",
        },
    )
    assert bad_resp.status_code == 400
    assert "Unknown transcription provider" in bad_resp.json()["detail"]

    with patch("app.api.routes_jobs.spawn_background", side_effect=lambda coro: coro.close()) as spawn_mock:
        ok_resp = client.post(
            "/api/jobs",
            json={
                "video_path": str(media),
                "whisper_model": "medium",
                "transcription_provider": "whisperx",
                "vocal_isolation": True,
                "enable_translation": False,
            },
        )
    assert ok_resp.status_code == 200
    data = ok_resp.json()
    assert data["transcription_provider"] == "whisperx"
    assert data["vocal_isolation"] is True
    assert spawn_mock.called

    sys_resp = client.get("/api/system/info")
    assert sys_resp.status_code == 200
    sys_data = sys_resp.json()
    assert "faster_whisper" in sys_data["transcription_providers"]
    assert "whisperx" in sys_data["transcription_providers"]
    assert isinstance(sys_data["whisperx_available"], bool)
    assert isinstance(sys_data["vocal_isolation_available"], bool)

    hw = get_hardware_info()
    assert "whisperx_available" in hw
    assert "vocal_isolation_available" in hw


def test_orchestrator_schedules_vocal_isolation_only_when_enabled():
    orch = PipelineOrchestrator("job_test_sched")
    ctx_default = PipelineContext(job_id="j1", video_path="v.mp4", file_name="v.mp4", vocal_isolation=False)
    statuses_default = [s for s, _ in orch._stages(ctx_default)]
    assert JobStatus.VOCAL_ISOLATION not in statuses_default

    ctx_iso = PipelineContext(job_id="j2", video_path="v.mp4", file_name="v.mp4", vocal_isolation=True)
    statuses_iso = [s for s, _ in orch._stages(ctx_iso)]
    assert statuses_iso.index(JobStatus.VOCAL_ISOLATION) == statuses_iso.index(JobStatus.EXTRACTING_AUDIO) + 1
    assert statuses_iso.index(JobStatus.TRANSCRIBING) == statuses_iso.index(JobStatus.VOCAL_ISOLATION) + 1


# ---------------------------------------------------------------------------
# 2. Vocal Isolation Service & Stage
# ---------------------------------------------------------------------------

def test_isolate_vocals_demucs_chunked_cache_and_vram_cleanup(monkeypatch, tmp_path):
    calls = _install_fake_torch_and_demucs(monkeypatch, raise_in_apply=False)
    in_wav = tmp_path / "mixed.wav"
    out_wav = tmp_path / "vocals" / "job_iso.vocals.wav"
    _write_test_wav(in_wav, duration_sec=22.0, sample_rate=16000)

    # Force 2 chunks by setting chunk_seconds=10.0 on a 22s WAV
    res1 = audio_isolation.isolate_vocals(
        str(in_wav),
        str(out_wav),
        model_name="htdemucs",
        device="cuda",
        chunk_seconds=10.0,
        overlap_seconds=1.0,
    )
    assert res1["cached"] is False
    assert res1["backend"] == "demucs"
    assert out_wav.is_file()
    assert calls["get_model"] == 1
    assert calls["apply_model"] >= 2
    assert all(seg <= 7.5 for seg in calls["segments"])
    # Strict VRAM cleanup ran in finally
    assert calls["empty_cache"] >= 1

    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0

    # Second invocation hits disk cache without loading or running Demucs
    res2 = audio_isolation.isolate_vocals(str(in_wav), str(out_wav), model_name="htdemucs")
    assert res2["cached"] is True
    assert res2["backend"] == "cache"
    assert calls["get_model"] == 1


def test_isolate_vocals_cleans_up_vram_on_error(monkeypatch, tmp_path):
    calls = _install_fake_torch_and_demucs(monkeypatch, raise_in_apply=True)
    in_wav = tmp_path / "mixed_err.wav"
    out_wav = tmp_path / "vocals" / "job_err.vocals.wav"
    _write_test_wav(in_wav, duration_sec=2.0, sample_rate=16000)

    with pytest.raises(RuntimeError, match="Simulated Demucs failure"):
        audio_isolation.isolate_vocals(str(in_wav), str(out_wav), device="cuda")

    assert calls["empty_cache"] >= 1
    assert not out_wav.exists()


@pytest.mark.asyncio
async def test_vocal_isolation_stage_wires_vocals_path_and_falls_back_gracefully(monkeypatch, tmp_path):
    in_wav = tmp_path / "raw_audio.wav"
    _write_test_wav(in_wav, duration_sec=2.0, sample_rate=16000)

    # Case A: Unavailable backend -> non-fatal warning, retains original extracted WAV
    ctx_missing = PipelineContext(
        job_id="job_iso_missing",
        video_path=str(tmp_path / "vid.mp4"),
        file_name="vid.mp4",
        audio_wav_path=str(in_wav),
        vocal_isolation=True,
    )
    with patch("app.services.audio_isolation.is_demucs_available", return_value=False), patch(
        "app.services.audio_isolation.is_audio_separator_available", return_value=False
    ):
        await VocalIsolationStage().execute(ctx_missing)

    assert ctx_missing.audio_wav_path == str(in_wav)
    assert ctx_missing.isolated_vocals_path is None
    assert any("Vocal isolation skipped" in w for w in ctx_missing.warnings)

    # Case B: Demucs available -> ctx.audio_wav_path switches to vocals.wav for TranscribeStage & DiarizeStage
    _install_fake_torch_and_demucs(monkeypatch, raise_in_apply=False)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    ctx_ok = PipelineContext(
        job_id="job_iso_ok",
        video_path=str(tmp_path / "vid.mp4"),
        file_name="vid.mp4",
        audio_wav_path=str(in_wav),
        vocal_isolation=True,
        enable_diarization=True,
    )
    await VocalIsolationStage().execute(ctx_ok)
    expected_vocals = str(tmp_path / "vocals" / "job_iso_ok.vocals.wav")
    assert ctx_ok.isolated_vocals_path == expected_vocals
    assert ctx_ok.audio_wav_path == expected_vocals
    assert Path(expected_vocals).is_file()

    # Verify DiarizeStage receives the clean original raw_audio.wav rather than Demucs vocals.wav
    # so separation artifacts never corrupt WeSpeaker voiceprints or F0 pitch estimation.
    seen_diarize_path = {}

    class _FakeEmbedder:
        def __init__(self, onnx_path):
            pass

    def _fake_diarize_spans(wav_path, spans, **_kwargs):
        seen_diarize_path["wav_path"] = str(wav_path)
        return (
            [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_01"}],
            {"SPEAKER_01": {"label": "Speaker 1", "gender": "unknown", "color": "#38BDF8"}},
        )

    ctx_ok.raw_segments = [RawSegment(index=1, start_time=0.0, end_time=1.5, text="Hello there.")]
    with patch("app.pipeline.stages.diarize_stage.ensure_diarization_model", return_value=tmp_path / "cam.onnx"), patch(
        "app.pipeline.stages.diarize_stage.OnnxSpeakerEmbedder", _FakeEmbedder
    ), patch("app.pipeline.stages.diarize_stage.diarize_audio_spans", side_effect=_fake_diarize_spans), patch(
        "app.pipeline.voice_gender.label_speakers_by_voice"
    ):
        await DiarizeStage().execute(ctx_ok)

    assert seen_diarize_path["wav_path"] == str(in_wav)


# ---------------------------------------------------------------------------
# 3. WhisperXProvider, Phoneme Alignment, No Gap Recovery & VRAM Cleanup
# ---------------------------------------------------------------------------

def _install_fake_whisperx(monkeypatch, *, with_diarizer: bool = False):
    trace = {
        "load_model": [],
        "asr_transcribe": [],
        "load_align": [],
        "align_calls": 0,
        "diarize_calls": 0,
        "empty_cache": 0,
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: trace.__setitem__("empty_cache", trace["empty_cache"] + 1),
        ipc_collect=lambda: None,
    )

    class _FakeASR:
        def transcribe(self, _audio, batch_size=8, language=None):
            trace["asr_transcribe"].append({"batch_size": batch_size, "language": language})
            return {
                "language": language or "en",
                "segments": [{"start": 0.2, "end": 2.4, "text": "Work fast, 12 Ploughman."}],
            }

    class _FakeDiarizer:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _audio):
            trace["diarize_calls"] += 1
            return [{"start": 0.2, "end": 2.4, "speaker": "SPEAKER_00"}]

    fake_wx = types.ModuleType("whisperx")
    fake_wx.load_model = lambda model_name, **kwargs: (trace["load_model"].append((model_name, kwargs)), _FakeASR())[1]
    fake_wx.load_audio = lambda path: np.zeros(16000, dtype=np.float32)
    fake_wx.load_align_model = lambda language_code, device: (
        trace["load_align"].append((language_code, device)),
        (object(), {"language": language_code}),
    )[1]

    def _align(segments, _model, _meta, _audio, _device, return_char_alignments=False):
        trace["align_calls"] += 1
        return {
            "segments": [
                {
                    "start": 0.21,
                    "end": 2.38,
                    "text": "Work fast, 12 Ploughman.",
                    "words": [
                        {"word": "Work", "start": 0.21, "end": 0.65, "score": 0.96},
                        {"word": "fast,", "start": 0.68, "end": 1.10, "score": 0.94},
                        # Deliberately unaligned digit token (missing start/end) as WhisperX returns
                        {"word": "12"},
                        {"word": "Ploughman.", "start": 1.60, "end": 2.38, "score": 0.91},
                    ],
                },
                {
                    # Watermark hallucination should still be filtered cleanly
                    "start": 5.0,
                    "end": 6.5,
                    "text": "© BF-WATCH TV 2021",
                    "words": [],
                },
            ]
        }

    def _assign_speakers(_diar, result):
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"
            for w in seg.get("words") or []:
                w["speaker"] = "SPEAKER_00"
        return result

    fake_wx.align = _align
    fake_wx.assign_word_speakers = _assign_speakers
    if with_diarizer:
        fake_wx.diarize = types.SimpleNamespace(DiarizationPipeline=_FakeDiarizer)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setattr("app.providers.transcription.whisperx_provider.is_whisperx_available", lambda: True)
    monkeypatch.setattr("app.providers.transcription.is_whisperx_available", lambda: True)
    return trace


@pytest.mark.asyncio
async def test_whisperx_provider_runs_3_steps_unloads_vram_and_skips_gap_recovery(monkeypatch, tmp_path):
    trace = _install_fake_whisperx(monkeypatch, with_diarizer=True)
    wav = tmp_path / "vocals.wav"
    _write_test_wav(wav, duration_sec=3.0)

    provider = WhisperXProvider(
        model_size_or_path="large-v3",
        device="cuda",
        compute_type="float16",
        batch_size=8,
        initial_prompt="Ploughman, Domesday",
        enable_diarization=True,
        hf_token="hf_test_token_123",
    )

    progress_events = []

    async def _on_seg(seg: RawSegment, pct: float):
        progress_events.append((seg.index, pct))

    segs = await provider.transcribe(str(wav), language="en", on_segment_callback=_on_seg)
    assert len(segs) == 1  # watermark filtered out
    assert segs[0].text == "Work fast, 12 Ploughman."
    assert segs[0].speaker == "SPEAKER_01"
    assert [w.word.strip() for w in segs[0].words] == ["Work", "fast,", "12", "Ploughman."]
    # Unaligned word "12" got valid interpolated timestamps
    assert 1.10 <= segs[0].words[2].start <= segs[0].words[2].end <= 1.60

    # Verify Step A, Step B, Step C invocations & options
    assert trace["load_model"][0][0] == "large-v3"
    assert trace["load_model"][0][1]["compute_type"] == "float16"
    assert "Ploughman, Domesday" in trace["load_model"][0][1]["asr_options"]["initial_prompt"]
    assert trace["asr_transcribe"][0]["batch_size"] == 8
    assert trace["align_calls"] == 1
    assert trace["diarize_calls"] == 1
    # VRAM empty_cache called after Step A, Step B, Step C, and final unload_model
    assert trace["empty_cache"] >= 4
    assert progress_events == [(1, 99.0)]

    # Verify TranscribeStage skips un-VAD'd gap recovery and word-start snap when using WhisperX
    ctx = PipelineContext(
        job_id="job_wx_stage",
        video_path=str(tmp_path / "movie.mp4"),
        file_name="movie.mp4",
        audio_wav_path=str(wav),
        transcription_provider="whisperx",
        whisper_model="medium",
        enable_diarization=False,
    )
    stage = TranscribeStage()
    await stage.execute(ctx)
    assert ctx.effective_transcription_provider == "whisperx"
    assert ctx.detected_language == "en"
    assert len(ctx.raw_segments) == 1
    # Must NOT fall back to Faster-Whisper or run gap recovery
    assert not any("fell back" in w.lower() for w in ctx.warnings)


@pytest.mark.asyncio
async def test_transcribe_stage_falls_back_gracefully_when_whisperx_missing(tmp_path):
    wav = tmp_path / "audio.wav"
    _write_test_wav(wav, duration_sec=1.5)

    class _FakeFW(FasterWhisperTranscriber):
        def __init__(self, **_kwargs):
            self.detected_language = "en"

        async def transcribe(self, audio_path, language=None, on_segment_callback=None, cancel_flag=None):
            seg = RawSegment(index=1, start_time=0.0, end_time=1.0, text="Fallback line.")
            if on_segment_callback:
                await on_segment_callback(seg, 100.0)
            return [seg]

        async def recover_missed_speech(self, *_args, **_kwargs):
            return []

        async def snap_starts_to_speech(self, *_args, **_kwargs):
            return None

    ctx = PipelineContext(
        job_id="job_wx_missing",
        video_path=str(tmp_path / "movie.mp4"),
        file_name="movie.mp4",
        audio_wav_path=str(wav),
        transcription_provider="whisperx",
        whisper_model="tiny",
    )
    with patch("app.providers.transcription.is_whisperx_available", return_value=False), patch(
        "app.providers.transcription.faster_whisper.FasterWhisperTranscriber", _FakeFW
    ):
        await TranscribeStage().execute(ctx)

    assert ctx.effective_transcription_provider == "faster_whisper"
    assert any("WhisperX is not installed" in w for w in ctx.warnings)
    assert len(ctx.raw_segments) == 1
    assert ctx.raw_segments[0].text == "Fallback line."


def test_ensure_torchaudio_compat_shims_missing_2_11_apis_and_soundfile_io(monkeypatch, tmp_path):
    from app.core.hardware import ensure_torchaudio_compat

    wav = tmp_path / "compat_sample.wav"
    _write_test_wav(wav, duration_sec=0.5, sample_rate=16000)

    fake_torchaudio = types.ModuleType("torchaudio")

    def _boom_load(*_args, **_kwargs):
        raise ImportError("TorchCodec is required for load_with_torchcodec.")

    fake_torchaudio.load = _boom_load

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = lambda arr: _FakeTensor(arr)

    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("app.core.hardware._has_package", lambda name: name == "torchaudio")

    ensure_torchaudio_compat()

    assert hasattr(fake_torchaudio, "AudioMetaData")
    assert fake_torchaudio.list_audio_backends() == ["soundfile"]
    assert fake_torchaudio.get_audio_backend() == "soundfile"
    fake_torchaudio.set_audio_backend("soundfile")

    meta = fake_torchaudio.info(str(wav))
    assert meta.sample_rate == 16000
    assert meta.num_channels == 1
    assert meta.num_frames == 8000

    waveform, sr = fake_torchaudio.load(str(wav), frame_offset=160, num_frames=800, backend="soundfile")
    assert sr == 16000
    assert waveform.shape == (1, 800)
