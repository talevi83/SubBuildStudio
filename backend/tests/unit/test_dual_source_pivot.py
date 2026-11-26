"""Unit tests for the feature-flagged Dual-Source (EN + AR) subtitle pivot pipeline."""
import json
import types
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus
from app.models.schemas import SubtitleStreamInfo
from app.pipeline.context import PipelineContext
from app.pipeline.embedded_tracks import pick_track_by_language
from app.pipeline.frame_sampling import frame_scale_filter
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.stages.dual_source_stage import DualSourceSyncStage
from app.providers.translation.base import TranslatableSegment
from app.providers.translation.gemini_provider import (
    GeminiTranslator,
    build_subtitle_system_prompt,
)
from app.services.dual_source_aligner import (
    TimedCue,
    align_dual_cues,
    align_en_ar_cues,
    evaluate_alignment_quality,
    parse_srt_cues,
    select_subtitle_pair,
)
from app.services.dual_source_fetcher import compute_opensubtitles_hash, parse_media_filename


# ---------------------------------------------------------------------------
# 1. Feature-flag gating on /api/config, /api/system/info & /api/jobs
# ---------------------------------------------------------------------------

def test_feature_flag_gating_on_config_and_job_creation(tmp_path, monkeypatch):
    video = tmp_path / "Movie.2026.1080p.WEB-DL.mkv"
    video.write_bytes(b"\x00" * 160_000)

    monkeypatch.setattr(PipelineOrchestrator, "run", lambda self: None)
    monkeypatch.setattr("app.api.routes_jobs.spawn_background", lambda coro: None)

    with TestClient(app) as client:
        # Default: strictly hidden
        monkeypatch.setattr(settings, "enable_dual_source_pivot", False)
        cfg_off = client.get("/api/config")
        assert cfg_off.status_code == 200
        assert cfg_off.json()["dual_source_pivot_enabled"] is False
        sys_off = client.get("/api/system/info")
        assert sys_off.status_code == 200
        assert sys_off.json()["dual_source_pivot_enabled"] is False

        # Job creation ignores enable_dual_source_pivot=True when server flag is off
        r_off = client.post(
            "/api/jobs",
            json={
                "video_path": str(video),
                "enable_dual_source_pivot": True,
                "dual_source_en_srt": "1\n00:00:01,000 --> 00:00:02,000\nHi\n",
                "dual_source_ar_srt": "1\n00:00:01,000 --> 00:00:02,000\nمرحبا\n",
            },
        )
        assert r_off.status_code == 200, r_off.text
        assert r_off.json()["enable_dual_source_pivot"] is False

        # Enabled locally: capability endpoint + job creation unlock Dual-Source
        monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
        cfg_on = client.get("/api/config")
        assert cfg_on.status_code == 200
        assert cfg_on.json()["dual_source_pivot_enabled"] is True
        sys_on = client.get("/api/system/info")
        assert sys_on.status_code == 200
        assert sys_on.json()["dual_source_pivot_enabled"] is True

        r_on = client.post(
            "/api/jobs",
            json={
                "video_path": str(video),
                "enable_dual_source_pivot": True,
            },
        )
        assert r_on.status_code == 200, r_on.text
        assert r_on.json()["enable_dual_source_pivot"] is True


# ---------------------------------------------------------------------------
# 2. Orchestrator routing & stage skipping / graceful fallback
# ---------------------------------------------------------------------------

def test_orchestrator_stages_gated_by_dual_source_flag(monkeypatch):
    orch = PipelineOrchestrator("job_ds_gate")
    try:
        ctx = PipelineContext(
            job_id="job_ds_gate",
            video_path="film.mkv",
            file_name="film.mkv",
            enable_dual_source_pivot=True,
            enable_diarization=True,
            detect_onscreen_text=True,
            enable_translation=True,
        )
        monkeypatch.setattr(settings, "enable_dual_source_pivot", False)
        stages_off = [st for st, _ in orch._stages(ctx)]
        assert JobStatus.DUAL_SOURCE_SYNC not in stages_off

        monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
        stages_on = [st for st, _ in orch._stages(ctx)]
        assert JobStatus.DUAL_SOURCE_SYNC in stages_on
        assert stages_on.index(JobStatus.EXTRACTING_AUDIO) + 1 == stages_on.index(JobStatus.DUAL_SOURCE_SYNC)
    finally:
        from app.pipeline.orchestrator import active_pipelines

        active_pipelines.pop("job_ds_gate", None)


@pytest.mark.asyncio
async def test_orchestrator_skips_whisper_diarize_vision_when_dual_source_active(tmp_path, monkeypatch):
    video = tmp_path / "The.Uprising.2026.WEB-DL.mkv"
    video.write_bytes(b"\x11" * 160_000)
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)

    en_srt = "1\n00:00:01,000 --> 00:00:03,000\nWhere are you going?\n"
    ar_srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nإلى أين تذهبين؟\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\nبعد عام واحد\n"
    )
    job_id = "job_ds_skip_01"
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id,
                video_path=str(video),
                file_name=video.name,
                status=JobStatus.IDLE,
                enable_dual_source_pivot=True,
                dual_source_en_srt=en_srt,
                dual_source_ar_srt=ar_srt,
                vocal_isolation=True,
                vocal_isolation_mode="on",
                enable_diarization=True,
                detect_onscreen_text=True,
                enable_translation=True,
                target_language="Hebrew",
            )
        )
        session.commit()

    called = []

    async def _ok_probe(self, ctx, on_progress=None):
        called.append("PROBE")
        ctx.duration = 10.0

    async def _ok_audio(self, ctx, on_progress=None):
        called.append("AUDIO")
        wav = tmp_path / "extracted.wav"
        wav.write_bytes(b"RIFF")
        ctx.audio_wav_path = str(wav)

    async def _fail_if_called(name):
        raise AssertionError(f"{name} should have been skipped when Dual-Source is active!")

    async def _ok_translate(self, ctx, on_progress=None):
        called.append("TRANSLATE")
        for seg in ctx.refined_segments:
            seg["translated_text"] = "טקסט מתורגם"

    with (
        patch("app.pipeline.stages.probe_stage.ProbeStage.execute", _ok_probe),
        patch("app.pipeline.stages.audio_stage.AudioExtractionStage.execute", _ok_audio),
        patch("app.services.dual_source_sync.sync_srt_to_audio", lambda ref, src, out, **kw: (out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8"), {"synced": True, "engine": "ffsubsync", "offset_seconds": -0.25, "scale_factor": None, "warning": None})[1]),
        patch("app.pipeline.stages.vocal_isolation_stage.VocalIsolationStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("VOCAL_ISOLATION")),
        patch("app.pipeline.stages.transcribe_stage.TranscribeStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("TRANSCRIBING")),
        patch("app.pipeline.stages.diarize_stage.DiarizeStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("DIARIZING")),
        patch("app.pipeline.stages.ocr_stage.OnScreenTextStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("SCANNING_ONSCREEN")),
        patch("app.pipeline.stages.translate_stage.TranslateStage.execute", _ok_translate),
    ):
        orch = PipelineOrchestrator(job_id)
        await orch.run()

    assert called == ["PROBE", "AUDIO", "TRANSLATE"]
    assert orch._ctx is not None
    assert orch._ctx.dual_source_active is True
    assert orch._ctx.dual_source_report["alignment"]["ar_only_count"] == 1
    kinds = {s.get("dual_source_kind") for s in orch._ctx.refined_segments}
    assert kinds == {"EN_AR", "AR_ONLY"}


@pytest.mark.asyncio
async def test_orchestrator_falls_back_to_whisper_when_dual_source_unavailable(tmp_path, monkeypatch):
    video = tmp_path / "NoSubs.2026.mkv"
    video.write_bytes(b"\x22" * 160_000)
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "opensubtitles_api_key", None)
    monkeypatch.setattr(settings, "subdl_api_key", None)

    job_id = "job_ds_fallback_01"
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id,
                video_path=str(video),
                file_name=video.name,
                status=JobStatus.IDLE,
                enable_dual_source_pivot=True,
                enable_diarization=False,
                detect_onscreen_text=True,
                enable_translation=False,
            )
        )
        session.commit()

    called = []

    async def _ok_probe(self, ctx, on_progress=None):
        called.append("PROBE")
        ctx.duration = 10.0

    async def _ok_audio(self, ctx, on_progress=None):
        called.append("AUDIO")

    async def _ok_transcribe(self, ctx, on_progress=None):
        called.append("TRANSCRIBE")

    async def _ok_ocr(self, ctx, on_progress=None):
        called.append("OCR")

    with (
        patch("app.pipeline.stages.probe_stage.ProbeStage.execute", _ok_probe),
        patch("app.pipeline.stages.audio_stage.AudioExtractionStage.execute", _ok_audio),
        patch("app.pipeline.stages.transcribe_stage.TranscribeStage.execute", _ok_transcribe),
        patch("app.pipeline.stages.ocr_stage.OnScreenTextStage.execute", _ok_ocr),
    ):
        orch = PipelineOrchestrator(job_id)
        await orch.run()

    assert called == ["PROBE", "AUDIO", "TRANSCRIBE", "OCR"]
    assert orch._ctx is not None and orch._ctx.dual_source_active is False


# ---------------------------------------------------------------------------
# 3. OpenSubtitles hashing & filename metadata parser
# ---------------------------------------------------------------------------

def test_opensubtitles_hash_and_filename_parser(tmp_path):
    sample = tmp_path / "sample.mkv"
    sample.write_bytes(b"ABCD1234" * 20_000)  # 160 KB
    h1, sz1 = compute_opensubtitles_hash(sample)
    h2, sz2 = compute_opensubtitles_hash(sample)
    assert h1 == h2
    assert len(h1) == 16 and int(h1, 16) >= 0
    assert sz1 == sz2 == 160_000

    ep = parse_media_filename("Severance.S02E05.Trojans.2160p.ATVP.WEB-DL.DDP5.1.H.265.mkv")
    assert ep["title"].lower() == "severance"
    assert ep["season"] == 2 and ep["episode"] == 5 and ep["type"] == "episode"

    mv = parse_media_filename("The.Uprising.2026.2160p.MA.WEB-DL.DDP5.1.mkv")
    assert mv["title"].lower() == "the uprising"
    assert mv["year"] == 2026 and mv["type"] == "movie"


# ---------------------------------------------------------------------------
# 4. Cross-language alignment (1:1, 1:many, +-1.5s window, EN-only, AR-only)
# ---------------------------------------------------------------------------

def test_align_en_ar_cues_covers_1_to_1_1_to_many_window_and_onscreen():
    en_cues = [
        TimedCue(1, 1.0, 3.0, "Hello, friend."),                      # 1:1 overlap
        TimedCue(2, 5.0, 9.0, "Wait right here until I return."),     # 1:many overlap (2 AR cues)
        TimedCue(3, 12.0, 14.0, "Who are you?"),                      # +-1.5s shifted midpoint match
        TimedCue(4, 20.0, 22.0, "Watch out!"),                        # EN_ONLY
    ]
    ar_cues = [
        TimedCue(1, 1.2, 2.9, "مرحباً يا صديقي."),
        TimedCue(2, 5.1, 6.9, "انتظر هنا"),
        TimedCue(3, 7.1, 8.8, "حتى أعود."),
        TimedCue(4, 14.3, 15.8, "من أنتِ؟"),                          # shifted: mid 15.05 - 13.0 = 2.05s? Wait test inside +-1.5s
        TimedCue(5, 30.0, 33.0, "إنجلترا، ١٣٨١"),                      # AR_ONLY on-screen
    ]
    # Shift cue #4 slightly closer so midpoint gap is 1.1s and edge gap is 0.1s (inside +-1.5s)
    ar_cues[3] = TimedCue(4, 14.1, 15.6, "من أنتِ؟")  # en mid=13.0, ar mid=14.85 -> 1.85s; make 13.8..15.2 -> mid=14.5 (1.5s edge=0.0..0)
    ar_cues[3] = TimedCue(4, 14.1, 14.8, "من أنتِ؟")  # en mid=13.0, ar mid=14.45 -> 1.45s, edge=0.1s

    aligned, stats = align_en_ar_cues(en_cues, ar_cues, window_sec=1.5)
    assert stats == {
        "en_total": 4,
        "ar_total": 5,
        "paired_count": 3,
        "en_only_count": 1,
        "ar_only_count": 1,
    }
    assert aligned[0].alignment_kind == "EN_AR"
    assert aligned[0].reference_text == "مرحباً يا صديقي."

    assert aligned[1].alignment_kind == "EN_AR"
    assert "انتظر هنا" in (aligned[1].reference_text or "")
    assert "حتى أعود." in (aligned[1].reference_text or "")

    assert aligned[2].alignment_kind == "EN_AR"
    assert aligned[2].reference_text == "من أنتِ؟"

    assert aligned[3].alignment_kind == "EN_ONLY"
    assert aligned[3].reference_text is None

    assert aligned[4].alignment_kind == "AR_ONLY"
    assert aligned[4].segment_type == "ON_SCREEN"
    assert aligned[4].reference_text == "إنجلترا، ١٣٨١"

    parsed = parse_srt_cues("1\n00:00:01,000 --> 00:00:02,500\n{\\an8}London, 1381\n")
    assert len(parsed) == 1 and parsed[0].top_positioned is True


# ---------------------------------------------------------------------------
# 5. Dual-source prompt & structured JSON batch translation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dual_source_prompt_and_batch_json_roundtrip(monkeypatch):
    prompt = build_subtitle_system_prompt(
        "English",
        "Hebrew",
        static_glossary=True,
        dual_source_mode=True,
    )
    assert "DUAL-SOURCE GENDER ANCHOR MODE" in prompt
    assert "ar_reference" in prompt
    assert "AR_ONLY" in prompt

    seen_payloads = []
    seen_sys = []

    class _FakeModels:
        def generate_content(self, model, contents, config):
            seen_sys.append(config.system_instruction)
            payload = json.loads(contents)
            seen_payloads.append(payload)
            return types.SimpleNamespace(
                text=json.dumps({
                    "translations": [
                        {
                            "id": item["id"],
                            "text": "[אנגליה, 1381]" if item.get("dual_source_kind") == "AR_ONLY" else "מתי חזרת?",
                            "addressee_form": "" if item.get("dual_source_kind") == "AR_ONLY" else "f",
                        }
                        for item in payload["target_batch"]
                    ]
                })
            )

    translator = GeminiTranslator(api_key="fake-key", batch_size=10, dual_source_mode=True, prune_glossary=False)
    translator.client = types.SimpleNamespace(models=_FakeModels())

    segs = [
        TranslatableSegment(
            index=1,
            start_time=1.0,
            end_time=3.0,
            original_text="When did you come back?",
            reference_text="متى عدتِ؟",
            dual_source_kind="EN_AR",
            segment_type="DIALOGUE",
        ),
        TranslatableSegment(
            index=2,
            start_time=5.0,
            end_time=8.0,
            original_text="إنجلترا، ١٣٨١",
            reference_text="إنجلترا، ١٣٨١",
            dual_source_kind="AR_ONLY",
            segment_type="ON_SCREEN",
        ),
    ]
    out = await translator.translate_segments(segs, "English", "Hebrew")
    assert len(seen_payloads) == 1
    tb = seen_payloads[0]["target_batch"]
    assert tb[0]["ar_reference"] == "متى عدتِ؟"
    assert tb[0]["dual_source_kind"] == "EN_AR"
    assert tb[1]["dual_source_kind"] == "AR_ONLY"
    assert out[0].translated_text == "מתי חזרת?"
    assert out[0].addressee_form == "f"
    assert out[1].translated_text == "[אנגליה, 1381]"


# ---------------------------------------------------------------------------
# 6. Embedded-first track detection before online subtitle search
# ---------------------------------------------------------------------------

def test_pick_track_by_language_finds_embedded_en_and_ar_and_skips_forced_commentary():
    streams = [
        SubtitleStreamInfo(index=0, codec_name="hdmv_pgs_subtitle", language="eng", title="PGS", text_based=False),
        SubtitleStreamInfo(index=1, codec_name="subrip", language="eng", title="Forced Signs", forced=True, text_based=True),
        SubtitleStreamInfo(index=2, codec_name="subrip", language="eng", title="Director Commentary", text_based=True),
        SubtitleStreamInfo(index=3, codec_name="subrip", language="eng", title="English SDH", text_based=True),
        SubtitleStreamInfo(index=4, codec_name="subrip", language="eng", title="English", default=True, text_based=True),
        SubtitleStreamInfo(index=5, codec_name="subrip", language="ara", title="Arabic", text_based=True),
    ]
    en_best = pick_track_by_language(streams, "en")
    ar_best = pick_track_by_language(streams, "ar")
    assert en_best is not None and en_best.index == 4
    assert ar_best is not None and ar_best.index == 5
    assert pick_track_by_language(streams, "fr") is None


@pytest.mark.asyncio
async def test_dual_source_prefers_embedded_en_and_ar_without_calling_online_api(tmp_path, monkeypatch):
    video = tmp_path / "Embedded.2026.mkv"
    video.write_bytes(b"\x00" * 160_000)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    en_srt = "1\n00:00:01,000 --> 00:00:03,000\nDid you see him?\n"
    ar_srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nهل رأيتِهِ؟\n\n"
        "2\n00:00:06,000 --> 00:00:08,000\nبعد عام\n"
    )

    ctx = PipelineContext(
        job_id="job_ds_emb_both",
        video_path=str(video),
        file_name=video.name,
        audio_wav_path=str(wav),
        enable_dual_source_pivot=True,
        subtitle_streams=[
            SubtitleStreamInfo(index=0, codec_name="subrip", language="eng", title="English", text_based=True),
            SubtitleStreamInfo(index=1, codec_name="subrip", language="ara", title="Arabic", text_based=True),
        ],
    )

    async def _fake_extract(vpath, stream):
        return en_srt if stream.index == 0 else ar_srt

    def _forbid_fetch(*args, **kwargs):
        raise AssertionError("fetch_dual_subtitles should not run when both EN and AR tracks are embedded!")

    def _forbid_sync(*args, **kwargs):
        raise AssertionError("sync_srt_to_audio should not run on embedded container tracks!")

    with (
        patch("app.pipeline.stages.dual_source_stage.extract_track_as_srt", _fake_extract),
        patch("app.pipeline.stages.dual_source_stage.fetch_dual_subtitles", _forbid_fetch),
        patch("app.pipeline.stages.dual_source_stage.sync_srt_to_audio", _forbid_sync),
    ):
        await DualSourceSyncStage().execute(ctx)

    assert ctx.dual_source_active is True
    assert ctx.dual_source_report["source"] == "embedded"
    assert ctx.dual_source_report["en_sync"]["engine"] == "embedded"
    assert ctx.dual_source_report["ar_sync"]["engine"] == "embedded"
    assert len(ctx.embedded_dialogue_segments) == 1
    assert ctx.embedded_dialogue_segments[0]["dual_source_kind"] == "EN_AR"
    assert len(ctx.onscreen_segments) == 1
    assert ctx.onscreen_segments[0]["dual_source_kind"] == "AR_ONLY"


@pytest.mark.asyncio
async def test_dual_source_uses_embedded_en_and_fetches_only_missing_ar_online(tmp_path, monkeypatch):
    video = tmp_path / "PartialEmbedded.2026.mkv"
    video.write_bytes(b"\x00" * 160_000)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    en_srt = "1\n00:00:01,000 --> 00:00:03,000\nHold on.\n"
    ar_srt = "1\n00:00:01,000 --> 00:00:03,000\nانتظري.\n"

    ctx = PipelineContext(
        job_id="job_ds_emb_partial",
        video_path=str(video),
        file_name=video.name,
        audio_wav_path=str(wav),
        enable_dual_source_pivot=True,
        subtitle_streams=[
            SubtitleStreamInfo(index=0, codec_name="subrip", language="eng", title="English", text_based=True),
        ],
    )

    fetch_langs_seen = []
    synced_files = []

    async def _fake_extract(vpath, stream):
        assert stream.index == 0
        return en_srt

    def _fake_fetch(*args, languages=(), **kwargs):
        fetch_langs_seen.append(tuple(languages))
        return {
            "en_srt": None,
            "ar_srt": ar_srt,
            "en_meta": None,
            "ar_meta": {"provider": "opensubtitles", "language": "ar"},
            "movie_hash": "0000000000000000",
            "parsed": {"title": "PartialEmbedded"},
        }

    def _fake_sync(ref, src, out, **kwargs):
        synced_files.append(src.name)
        out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return {"synced": True, "engine": "ffsubsync", "offset_seconds": 0.4, "scale_factor": None, "warning": None}

    with (
        patch("app.pipeline.stages.dual_source_stage.extract_track_as_srt", _fake_extract),
        patch("app.pipeline.stages.dual_source_stage.fetch_dual_subtitles", _fake_fetch),
        patch("app.pipeline.stages.dual_source_stage.sync_srt_to_audio", _fake_sync),
    ):
        await DualSourceSyncStage().execute(ctx)

    assert fetch_langs_seen == [("ar",)]
    assert synced_files == ["ar.raw.srt"]
    assert ctx.dual_source_active is True
    assert ctx.dual_source_report["source"] == "mixed"
    assert ctx.dual_source_report["en_sync"]["engine"] == "embedded"
    assert ctx.dual_source_report["ar_sync"]["engine"] == "ffsubsync"


# ---------------------------------------------------------------------------
# 7. Multi-language fallback chain, Smart ASR Fallback, and 4K OCR downscaling
# ---------------------------------------------------------------------------

def test_select_subtitle_pair_fallback_chain():
    # Best case: EN + AR
    assert select_subtitle_pair({"en": "a", "ar": "b"}) == {
        "pair_mode": "en_ar", "primary_lang": "en", "anchor_lang": "ar",
    }
    # Missing EN -> AR paired with es -> fr -> ru
    assert select_subtitle_pair({"ar": "a", "fr": "f", "ru": "r"}) == {
        "pair_mode": "ar_secondary", "primary_lang": "ar", "anchor_lang": "fr",
    }
    assert select_subtitle_pair({"ar": "a", "es": "e", "fr": "f"}) == {
        "pair_mode": "ar_secondary", "primary_lang": "ar", "anchor_lang": "es",
    }
    # Missing AR -> EN paired with gender-rich fallback ru -> pl -> es -> fr
    assert select_subtitle_pair({"en": "e", "es": "s", "ru": "r"}) == {
        "pair_mode": "en_gender_fallback", "primary_lang": "en", "anchor_lang": "ru",
    }
    assert select_subtitle_pair({"en": "e", "es": "s", "pl": "p"}) == {
        "pair_mode": "en_gender_fallback", "primary_lang": "en", "anchor_lang": "pl",
    }
    # Custom priority override
    assert select_subtitle_pair({"en": "e", "es": "s", "ru": "r"}, fallback_priority=["es", "ru"]) == {
        "pair_mode": "en_gender_fallback", "primary_lang": "en", "anchor_lang": "es",
    }
    # Single source & none
    assert select_subtitle_pair({"en": "e"}) == {
        "pair_mode": "single_source", "primary_lang": "en", "anchor_lang": None,
    }
    assert select_subtitle_pair({}) == {
        "pair_mode": "none", "primary_lang": None, "anchor_lang": None,
    }


@pytest.mark.asyncio
async def test_smart_asr_fallback_triggers_on_low_coverage_or_large_gaps(tmp_path, monkeypatch):
    video = tmp_path / "ShortSubs.2026.mkv"
    video.write_bytes(b"\x00" * 160_000)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    # Only 20s covered out of a 1200s film -> must trigger full Whisper ASR fallback
    short_en = "1\n00:00:01,000 --> 00:00:05,000\nLine one.\n\n2\n00:00:10,000 --> 00:00:20,000\nLine two.\n"
    short_ar = "1\n00:00:01,000 --> 00:00:05,000\nسطر ١.\n\n2\n00:00:10,000 --> 00:00:20,000\nسطر ٢.\n"
    ctx_short = PipelineContext(
        job_id="job_ds_lowcov",
        video_path=str(video),
        file_name=video.name,
        audio_wav_path=str(wav),
        duration=1200.0,
        enable_dual_source_pivot=True,
        dual_source_en_srt=short_en,
        dual_source_ar_srt=short_ar,
        dual_source_smart_asr_fallback=True,
    )
    with patch(
        "app.pipeline.stages.dual_source_stage.sync_srt_to_audio",
        lambda ref, src, out, **kw: (out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8"), {"synced": True, "engine": "ffsubsync", "offset_seconds": 0.0, "scale_factor": None, "warning": None})[1],
    ):
        await DualSourceSyncStage().execute(ctx_short)

    assert ctx_short.dual_source_active is False
    assert "Smart ASR Fallback" in (ctx_short.dual_source_report or {}).get("fallback_reason", "")

    # Good total span (1000s / 1100s) but 180s uncovered middle gap -> smart gap supplement enabled
    gap_en = (
        "1\n00:00:05,000 --> 00:00:08,000\nStart.\n\n"
        "2\n00:05:00,000 --> 00:05:05,000\nMiddle.\n\n"
        "3\n00:16:30,000 --> 00:16:40,000\nEnd.\n"
    )
    gap_ar = (
        "1\n00:00:05,000 --> 00:00:08,000\nالبداية.\n\n"
        "2\n00:05:00,000 --> 00:05:05,000\nالوسط.\n\n"
        "3\n00:16:30,000 --> 00:16:40,000\nالنهاية.\n"
    )
    ctx_gap = PipelineContext(
        job_id="job_ds_gap",
        video_path=str(video),
        file_name=video.name,
        audio_wav_path=str(wav),
        duration=1100.0,
        enable_dual_source_pivot=True,
        dual_source_en_srt=gap_en,
        dual_source_ar_srt=gap_ar,
        dual_source_smart_asr_fallback=True,
    )
    with patch(
        "app.pipeline.stages.dual_source_stage.sync_srt_to_audio",
        lambda ref, src, out, **kw: (out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8"), {"synced": True, "engine": "ffsubsync", "offset_seconds": 0.0, "scale_factor": None, "warning": None})[1],
    ):
        await DualSourceSyncStage().execute(ctx_gap)

    assert ctx_gap.dual_source_active is True
    assert ctx_gap.supplement_with_whisper is True
    assert ctx_gap.dual_source_report["gap_asr_supplement"] is True
    assert "DIARIZING" in ctx_gap.dual_source_report["skipped_stages"]
    assert "SCANNING_ONSCREEN" in ctx_gap.dual_source_report["skipped_stages"]
    assert "TRANSCRIBING" not in ctx_gap.dual_source_report["skipped_stages"]


def test_poor_pairing_ratio_degrades_to_single_source_and_generalized_prompts():
    en_cues = [TimedCue(i, float(i * 10), float(i * 10 + 2), f"Line {i}") for i in range(1, 9)]
    ar_cues = [TimedCue(1, 500.0, 502.0, "بعيد")]
    aligned, stats = align_dual_cues(en_cues, ar_cues, pair_mode="en_ar")
    q = evaluate_alignment_quality(aligned, 120.0, stats, pair_mode="en_ar", primary_synced=True)
    assert q["anchor_unreliable"] is True

    prompt_ru = build_subtitle_system_prompt(
        "English",
        "Hebrew",
        dual_source_mode=True,
        dual_source_primary_lang="English",
        dual_source_anchor_lang="Russian",
        dual_source_pair_mode="en_gender_fallback",
    )
    assert "Russian translation (`ar_reference`)" in prompt_ru

    prompt_ar_es = build_subtitle_system_prompt(
        "Arabic",
        "Hebrew",
        dual_source_mode=True,
        dual_source_primary_lang="Arabic",
        dual_source_anchor_lang="Spanish",
        dual_source_pair_mode="ar_secondary",
    )
    assert "primary Arabic text" in prompt_ar_es
    assert "Spanish reference" in prompt_ar_es

    assert frame_scale_filter(640) == "scale='min(640,iw)':-2:flags=fast_bilinear"
    assert frame_scale_filter(480) == "scale='min(480,iw)':-2:flags=fast_bilinear"


@pytest.mark.asyncio
async def test_audio_stage_does_not_skip_wav_extraction_when_dual_source_active(tmp_path, monkeypatch):
    from pathlib import Path
    from app.pipeline.stages.audio_stage import AudioExtractionStage
    from app.models.schemas import AudioStreamInfo

    video = tmp_path / "DualEmbeddedFirst.2026.mkv"
    video.write_bytes(b"\x00" * 160_000)
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    ctx = PipelineContext(
        job_id="job_audio_ds_wav",
        video_path=str(video),
        file_name=video.name,
        duration=600.0,
        enable_dual_source_pivot=True,
        use_embedded_subtitles=True,
        audio_streams=[AudioStreamInfo(index=0, codec_name="aac", channels=2, language="eng")],
        selected_audio_stream=0,
        subtitle_streams=[
            SubtitleStreamInfo(index=0, codec_name="subrip", language="eng", title="English", text_based=True),
        ],
    )

    extract_called = []

    async def _fake_extract_audio_track(video_path, output_wav_path, **kwargs):
        extract_called.append(output_wav_path)
        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_wav_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    with patch("app.pipeline.stages.audio_stage.extract_audio_track", _fake_extract_audio_track):
        await AudioExtractionStage().execute(ctx)

    assert len(extract_called) == 1
    assert ctx.audio_wav_path == extract_called[0]
    assert not ctx.embedded_dialogue_segments


# ---------------------------------------------------------------------------
# 8. Built-in cue_cadence/audio_envelope sync, SDH cleaning, Whisper skip & OCR trigger
# ---------------------------------------------------------------------------

def test_builtin_cue_cadence_and_audio_envelope_sync_fallback(tmp_path):
    import math
    import wave
    import numpy as np
    from app.services.dual_source_sync import sync_srt_to_audio, sync_srt_to_reference_cues

    def _srt_ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # Build reference timeline with 15 irregularly spaced cues and shift target by +7.65s
    ref_starts = [5.0, 9.2, 16.0, 21.5, 29.0, 34.4, 41.0, 48.5, 55.2, 62.0, 70.0, 76.5, 84.0, 91.2, 99.0]
    ref_blocks = []
    shifted_blocks = []
    for i, s in enumerate(ref_starts, start=1):
        ref_blocks.append(f"{i}\n{_srt_ts(s)} --> {_srt_ts(s + 2.2)}\nLine {i}")
        shifted_blocks.append(f"{i}\n{_srt_ts(s + 7.65)} --> {_srt_ts(s + 9.85)}\nWiersz {i}")

    ref_srt = tmp_path / "en.synced.srt"
    pl_raw = tmp_path / "pl.raw.srt"
    pl_out = tmp_path / "pl.synced.srt"
    ref_srt.write_text("\n\n".join(ref_blocks) + "\n", encoding="utf-8")
    pl_raw.write_text("\n\n".join(shifted_blocks) + "\n", encoding="utf-8")

    with (
        patch("app.services.dual_source_sync._ffsubsync_command", return_value=None),
        patch("app.services.dual_source_sync._alass_command", return_value=None),
    ):
        res_cadence = sync_srt_to_audio(tmp_path / "missing.wav", pl_raw, pl_out, reference_srt_path=ref_srt)

    assert res_cadence["synced"] is True
    assert res_cadence["engine"] == "cue_cadence"
    assert abs(res_cadence["offset_seconds"] + 7.65) <= 0.15
    synced_cues = parse_srt_cues(pl_out.read_text(encoding="utf-8"))
    assert abs(synced_cues[0].start - 5.0) <= 0.15

    # Now verify built-in WAV audio envelope sync when no reference SRT or CLI sync tool exists
    sr = 8000
    dur_s = 110
    samples = np.zeros(sr * dur_s, dtype=np.float32)
    t_grid = np.arange(len(samples), dtype=np.float32) / sr
    for s in ref_starts:
        mask = (t_grid >= s) & (t_grid <= s + 2.2)
        samples[mask] = 12000.0 * np.sin(2.0 * math.pi * 440.0 * t_grid[mask])
    wav_path = tmp_path / "speech.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())

    env_out = tmp_path / "pl.env.srt"
    with (
        patch("app.services.dual_source_sync._ffsubsync_command", return_value=None),
        patch("app.services.dual_source_sync._alass_command", return_value=None),
    ):
        res_env = sync_srt_to_audio(wav_path, pl_raw, env_out)

    assert res_env["synced"] is True
    assert res_env["engine"] == "audio_envelope"
    assert abs(res_env["offset_seconds"] + 7.65) <= 0.25


def test_parse_srt_cues_cleans_sdh_sfx_and_speaker_prefixes():
    raw_sdh = (
        "1\n00:00:05,020 --> 00:00:07,580\n(THUNDER RUMBLING)\n\n"
        "2\n00:00:17,820 --> 00:00:19,820\n-(DISHES CLATTER)\n-MAN: Bollocks!\n\n"
        "3\n00:00:34,099 --> 00:00:36,099\n(SCOFFS)\nI ain't doing that again.\n\n"
        "4\n00:01:05,420 --> 00:01:06,899\nWOMAN ON RADIO:\nExciting day today\n\n"
        "5\n00:01:20,260 --> 00:01:21,420\nBANK MANAGER: You need start-up money\n"
    )
    cleaned = parse_srt_cues(raw_sdh, clean_sdh=True, strip_speaker_labels=True)
    assert [c.text for c in cleaned] == [
        "Bollocks!",
        "I ain't doing that again.",
        "Exciting day today",
        "You need start-up money",
    ]


@pytest.mark.asyncio
async def test_dual_source_clears_stale_supplement_flag_and_runs_ocr_when_no_ar_only(tmp_path, monkeypatch):
    video = tmp_path / "Lazarus.S01E01.mkv"
    video.write_bytes(b"\x00" * 160_000)
    monkeypatch.setattr(settings, "enable_dual_source_pivot", True)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    en_srt = "1\n00:00:02,000 --> 00:00:04,500\nBig day today.\n"
    pl_srt = "1\n00:00:02,000 --> 00:00:04,500\nDzisiaj wielki dzień.\n"

    job_id = "job_ds_stale_supp_ocr"
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id,
                video_path=str(video),
                file_name=video.name,
                status=JobStatus.IDLE,
                enable_dual_source_pivot=True,
                supplement_with_whisper=True,
                detect_onscreen_text=True,
                enable_translation=False,
                dual_source_en_srt=en_srt,
                dual_source_ar_srt=None,
            )
        )
        session.commit()

    called = []

    async def _ok_probe(self, ctx, on_progress=None):
        ctx.duration = 10.0

    async def _ok_audio(self, ctx, on_progress=None):
        wav = tmp_path / "extracted.wav"
        wav.write_bytes(b"RIFF")
        ctx.audio_wav_path = str(wav)

    async def _fail_if_called(name):
        raise AssertionError(f"{name} should have been skipped when gap_supplement=False!")

    async def _ok_ocr(self, ctx, on_progress=None):
        called.append("OCR")
        ctx.onscreen_segments.append(
            {
                "start_time": 1.0,
                "end_time": 3.0,
                "original_text": "[Three months later]",
                "segment_type": "ON_SCREEN",
                "confidence": 1.0,
            }
        )

    def _fake_fetch(*args, languages=(), **kwargs):
        if "pl" in languages:
            return {
                "en_srt": None,
                "ar_srt": None,
                "pl_srt": pl_srt,
                "pl_meta": {"provider": "opensubtitles", "language": "pl"},
                "movie_hash": "0000000000000000",
                "parsed": {"title": "Lazarus"},
            }
        return {"en_srt": None, "ar_srt": None, "movie_hash": "0000000000000000", "parsed": {}}

    with (
        patch("app.pipeline.stages.probe_stage.ProbeStage.execute", _ok_probe),
        patch("app.pipeline.stages.audio_stage.AudioExtractionStage.execute", _ok_audio),
        patch("app.pipeline.stages.dual_source_stage.fetch_dual_subtitles", _fake_fetch),
        patch("app.pipeline.stages.dual_source_stage.sync_srt_to_audio", lambda ref, src, out, **kw: (out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8"), {"synced": True, "engine": "cue_cadence", "offset_seconds": 0.0, "scale_factor": None, "warning": None})[1]),
        patch("app.pipeline.stages.vocal_isolation_stage.VocalIsolationStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("VOCAL_ISOLATION")),
        patch("app.pipeline.stages.transcribe_stage.TranscribeStage.execute", lambda self, ctx, on_progress=None: _fail_if_called("TRANSCRIBING")),
        patch("app.pipeline.stages.ocr_stage.OnScreenTextStage.execute", _ok_ocr),
    ):
        orch = PipelineOrchestrator(job_id)
        await orch.run()

    assert called == ["OCR"]
    assert orch._ctx is not None
    assert orch._ctx.supplement_with_whisper is False
    assert "SCANNING_ONSCREEN" not in orch._ctx.dual_source_report["skipped_stages"]




