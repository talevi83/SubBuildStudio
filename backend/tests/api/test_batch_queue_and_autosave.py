"""Batch queue serialization, cumulative living series glossary, and auto-save SRT next to video."""
import asyncio
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import app.pipeline.orchestrator as orch
from app.core.database import _ADDED_COLUMNS, engine
from app.core.glossary_profiles import load_profile
from app.main import app
from app.models.db_models import JobStatus
from app.models.schemas import AudioStreamInfo
from app.pipeline.context import RawSegment
from app.pipeline.stages.translate_stage import TranslateStage
from app.providers.translation.base import BaseTranslator


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        body = res.json()
        if body["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach terminal state within {timeout}s")


def test_auto_save_srt_column_registered_for_migration():
    assert "auto_save_srt" in _ADDED_COLUMNS["jobs"]


def test_batch_queue_serializes_episodes_grows_glossary_and_autosaves(client: TestClient, tmp_path, monkeypatch):
    ep1 = tmp_path / "Show.S01E01.mkv"
    ep2 = tmp_path / "Show.S01E02.mkv"
    ep1.write_bytes(b"ep1")
    ep2.write_bytes(b"ep2")

    profile_name = f"LivingShow_{uuid.uuid4().hex[:6]}"
    execution_log = []
    active_runs = 0
    max_concurrent = 0
    known_seen = {}

    class FakeProbeStage:
        name = "PROBING"

        async def execute(self, ctx, on_progress=None):
            ctx.duration = 5.0
            ctx.audio_streams = [AudioStreamInfo(index=0, codec_name="aac", channels=2, language="en")]

    class FakeAudioStage:
        name = "EXTRACTING_AUDIO"

        async def execute(self, ctx, on_progress=None):
            ctx.audio_wav_path = None

    class FakeTranscribeStage:
        name = "TRANSCRIBING"

        async def execute(self, ctx, on_progress=None):
            nonlocal active_runs, max_concurrent
            active_runs += 1
            max_concurrent = max(max_concurrent, active_runs)
            execution_log.append(("start", ctx.file_name))
            await asyncio.sleep(0.15)
            ctx.detected_language = "en"
            ctx.raw_segments = [
                RawSegment(index=1, start_time=0.5, end_time=2.5, text=f"Hello from {ctx.file_name}?")
            ]
            execution_log.append(("end", ctx.file_name))
            active_runs -= 1

    class EpisodeGlossaryTranslator(BaseTranslator):
        async def build_glossary(self, segments, source_lang, target_lang, context_notes="", known_glossary=None):
            fname = segments[0].original_text
            known_seen[fname] = known_glossary
            if "S01E01" in fname:
                return {
                    "summary": "Sci-Fi drama",
                    "characters": [{"name": "Alice", "target_name": "אליס", "gender": "female"}],
                    "terms": [{"source": "Warp", "translation": "עיוות"}],
                    "register_notes": "",
                }
            return {
                "summary": "Ep2",
                "characters": [{"name": "Bob", "target_name": "בוב", "gender": "male"}],
                "terms": [{"source": "Shield", "translation": "מגן"}],
                "register_notes": "",
            }

        async def translate_segments(
            self,
            segments,
            source_lang,
            target_lang,
            context_notes="",
            apply_rtl=False,
            on_batch_complete=None,
            glossary=None,
            context_segments=None,
        ):
            for s in segments:
                s.translated_text = "שלום מהפרק?"
            if on_batch_complete:
                await on_batch_complete(segments, 100.0)
            return segments

    monkeypatch.setattr(orch, "ProbeStage", FakeProbeStage)
    monkeypatch.setattr(orch, "AudioExtractionStage", FakeAudioStage)
    monkeypatch.setattr(orch, "TranscribeStage", FakeTranscribeStage)
    monkeypatch.setattr(TranslateStage, "_make_translator", lambda self, ctx: EpisodeGlossaryTranslator())

    payload_base = {
        "enable_translation": True,
        "target_language": "Hebrew",
        "glossary_profile": profile_name,
        "auto_save_srt": True,
    }
    r1 = client.post("/api/jobs", json={**payload_base, "video_path": str(ep1), "file_name": ep1.name})
    r2 = client.post("/api/jobs", json={**payload_base, "video_path": str(ep2), "file_name": ep2.name})
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    j1 = r1.json()
    j2 = r2.json()
    assert j1["auto_save_srt"] is True
    assert j2["status"] == "QUEUED"

    done1 = _wait_terminal(client, j1["id"])
    done2 = _wait_terminal(client, j2["id"])
    assert done1["status"] == "COMPLETED"
    assert done2["status"] == "COMPLETED"

    # Serialized execution: episode 1 completed before episode 2 started
    assert max_concurrent == 1
    assert execution_log == [
        ("start", ep1.name),
        ("end", ep1.name),
        ("start", ep2.name),
        ("end", ep2.name),
    ]

    # Cumulative living glossary: ep1 started fresh and created the profile; ep2 loaded ep1's entries
    assert known_seen["Hello from Show.S01E01.mkv?"] is None
    assert [c["name"] for c in known_seen["Hello from Show.S01E02.mkv?"]["characters"]] == ["Alice"]

    saved_profile = load_profile(profile_name)
    assert saved_profile is not None
    assert [c["name"] for c in saved_profile["characters"]] == ["Alice", "Bob"]
    assert [t["source"] for t in saved_profile["terms"]] == ["Warp", "Shield"]

    # Auto-save SRT next to each episode with RTL directional marks
    srt1 = tmp_path / "Show.S01E01.he.srt"
    srt2 = tmp_path / "Show.S01E02.he.srt"
    assert srt1.is_file()
    assert srt2.is_file()
    text1 = srt1.read_text(encoding="utf-8-sig")
    assert "שלום מהפרק" in text1 and "\u200f" in text1


def test_auto_save_srt_without_translation_uses_source_language(client: TestClient, tmp_path, monkeypatch):
    ep = tmp_path / "Lecture.01.mkv"
    ep.write_bytes(b"vid")

    class FakeProbeStage:
        name = "PROBING"

        async def execute(self, ctx, on_progress=None):
            ctx.duration = 4.0
            ctx.audio_streams = [AudioStreamInfo(index=0, codec_name="aac", channels=2, language="en")]

    class FakeAudioStage:
        name = "EXTRACTING_AUDIO"

        async def execute(self, ctx, on_progress=None):
            ctx.audio_wav_path = None

    class FakeTranscribeStage:
        name = "TRANSCRIBING"

        async def execute(self, ctx, on_progress=None):
            ctx.detected_language = "en"
            ctx.raw_segments = [RawSegment(index=1, start_time=0.5, end_time=2.5, text="Welcome to lecture one.")]

    monkeypatch.setattr(orch, "ProbeStage", FakeProbeStage)
    monkeypatch.setattr(orch, "AudioExtractionStage", FakeAudioStage)
    monkeypatch.setattr(orch, "TranscribeStage", FakeTranscribeStage)

    res = client.post(
        "/api/jobs",
        json={"video_path": str(ep), "enable_translation": False, "auto_save_srt": True},
    )
    assert res.status_code == 200
    done = _wait_terminal(client, res.json()["id"])
    assert done["status"] == "COMPLETED"
    srt = tmp_path / "Lecture.01.en.srt"
    assert srt.is_file()
    assert "Welcome to lecture one." in srt.read_text(encoding="utf-8-sig")


def test_cancel_queued_job_immediately(client: TestClient, tmp_path, monkeypatch):
    ep1 = tmp_path / "Ep1.mkv"
    ep2 = tmp_path / "Ep2.mkv"
    ep1.write_bytes(b"1")
    ep2.write_bytes(b"2")

    gate_entered = asyncio.Event()
    release_gate = asyncio.Event()

    class BlockingProbe:
        name = "PROBING"

        async def execute(self, ctx, on_progress=None):
            if ctx.file_name == "Ep1.mkv":
                gate_entered.set()
                while not release_gate.is_set() and not ctx.is_cancelled:
                    await asyncio.sleep(0.02)
            ctx.duration = 2.0
            ctx.audio_streams = []

    class QuickAudio:
        name = "EXTRACTING_AUDIO"

        async def execute(self, ctx, on_progress=None):
            ctx.audio_wav_path = None

    class QuickTranscribe:
        name = "TRANSCRIBING"

        async def execute(self, ctx, on_progress=None):
            ctx.detected_language = "en"
            ctx.raw_segments = [RawSegment(index=1, start_time=0.2, end_time=1.2, text="Done")]

    monkeypatch.setattr(orch, "ProbeStage", BlockingProbe)
    monkeypatch.setattr(orch, "AudioExtractionStage", QuickAudio)
    monkeypatch.setattr(orch, "TranscribeStage", QuickTranscribe)

    r1 = client.post("/api/jobs", json={"video_path": str(ep1), "enable_translation": False})
    r2 = client.post("/api/jobs", json={"video_path": str(ep2), "enable_translation": False})
    assert r2.json()["status"] == "QUEUED"

    cancel_res = client.post(f"/api/jobs/{r2.json()['id']}/cancel")
    assert cancel_res.status_code == 200
    client.post(f"/api/jobs/{r1.json()['id']}/cancel")

    done2 = _wait_terminal(client, r2.json()["id"])
    done1 = _wait_terminal(client, r1.json()["id"])
    assert done2["status"] == "CANCELLED"
    assert done1["status"] == "CANCELLED"
