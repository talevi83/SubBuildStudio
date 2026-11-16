"""API / pipeline behaviour added in the improvement pass."""
import asyncio
import io
import os
import threading
import time
import uuid
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.context import PipelineCancelled, PipelineContext, RawSegment
from app.pipeline.stages.transcribe_stage import TranscribeStage
from app.pipeline.stages.translate_stage import TranslateStage
from app.api.routes_media import _parse_range, _display_name
from app.providers.transcription.base import BaseTranscriber
from app.providers.translation.base import BaseTranslator, TranslatableSegment
from app.utils.rtl_fixer import RLM


@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client


def _job_with_segments(texts: List[str], **job_kwargs) -> str:
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    with Session(engine) as session:
        session.add(Job(id=job_id, video_path="x.mkv", file_name="x.mkv", status=JobStatus.COMPLETED, **job_kwargs))
        session.commit()
        for i, t in enumerate(texts, start=1):
            session.add(SubtitleSegmentRecord(job_id=job_id, index=i, start_time=i * 2.0, end_time=i * 2.0 + 1.5,
                                              original_text=t))
        session.commit()
    return job_id


class FakeTranslator(BaseTranslator):
    def __init__(self, fail_indices=()):
        self.fail_indices = set(fail_indices)
        self.calls: List[List[int]] = []

    async def build_glossary(self, segments, source_lang, target_lang, context_notes=""):
        return {"summary": "test", "characters": [], "terms": []}

    async def translate_segments(self, segments, source_lang, target_lang, context_notes="", apply_rtl=False,
                                 on_batch_complete=None, glossary=None, context_segments=None):
        self.calls.append([s.index for s in segments])
        for s in segments:
            if s.index in self.fail_indices:
                s.translated_text, s.translation_failed = None, True
            else:
                s.translated_text, s.translation_failed = f"תרגום {s.index}?", False
        if on_batch_complete:
            await on_batch_complete(segments, 100.0)
        return segments


def _wait_for(client: TestClient, job_id: str, statuses=("COMPLETED", "FAILED", "CANCELLED"), timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job['status']}")


def test_untrusted_host_is_rejected(client: TestClient):
    res = client.get("/api/system/info", headers={"host": "evil.example.com"})
    assert res.status_code == 400


def test_system_info_lists_models(client: TestClient):
    data = client.get("/api/system/info").json()
    assert "large-v3-turbo" in data["whisper_models"]
    assert settings.gemini_model in data["gemini_models"]


def test_list_jobs_is_slim_but_has_counts(client: TestClient):
    job_id = _job_with_segments(["a", "b", "c"])
    jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
    assert jobs[job_id]["segments"] == []
    assert jobs[job_id]["segment_count"] == 3


def test_retranslate_flags_failures_then_retries_only_failed(client: TestClient, monkeypatch):
    fake = FakeTranslator(fail_indices={2})
    monkeypatch.setattr(TranslateStage, "_make_translator", lambda self, ctx: fake)
    job_id = _job_with_segments(["Hello?", "World.", "Bye."], target_language="Hebrew")

    assert client.post(f"/api/jobs/{job_id}/translate", json={}).status_code == 200
    job = _wait_for(client, job_id)
    segs = {s["index"]: s for s in job["segments"]}
    assert segs[1]["translated_text"] == "תרגום 1?"
    assert RLM not in segs[1]["translated_text"]  # stored clean, RTL marks only at export
    assert segs[2]["translation_failed"] is True and segs[2]["translated_text"] is None
    assert job["failed_translation_count"] == 1
    assert "could not be translated" in job["warning_message"]
    assert job["glossary"]["summary"] == "test"

    fake.fail_indices = set()
    client.post(f"/api/jobs/{job_id}/translate", json={"only_failed": True})
    job = _wait_for(client, job_id)
    assert fake.calls[-1] == [2]
    assert job["failed_translation_count"] == 0
    assert job["warning_message"] is None

    srt = client.get(f"/api/jobs/{job_id}/export/srt?use_translated=true").text
    assert f"{RLM}תרגום 1?{RLM}" in srt


def test_translation_without_key_keeps_transcription(client: TestClient):
    job_id = _job_with_segments(["Hello."])
    client.post(f"/api/jobs/{job_id}/translate", json={})
    job = _wait_for(client, job_id)
    assert job["status"] == "COMPLETED"
    assert "GEMINI_API_KEY" in job["warning_message"]
    assert job["segments"][0]["original_text"] == "Hello."


def test_bulk_update_and_validation(client: TestClient):
    job_id = _job_with_segments(["one", "two"])
    segs = client.get(f"/api/jobs/{job_id}").json()["segments"]
    res = client.patch(f"/api/jobs/{job_id}/segments", json=[
        {"id": segs[0]["id"], "translated_text": f"{RLM}אחד{RLM}"},
        {"id": segs[1]["id"], "original_text": "TWO"},
    ])
    assert res.status_code == 200
    assert res.json()[0]["translated_text"] == "אחד"
    assert res.json()[1]["original_text"] == "TWO"

    bad = client.patch(f"/api/jobs/{job_id}/segments/{segs[0]['id']}", json={"end_time": 0.5})
    assert bad.status_code == 422


def test_range_parsing():
    assert _parse_range("bytes=0-99", 1000) == (0, 99)
    assert _parse_range("bytes=900-", 1000) == (900, 999)
    assert _parse_range("bytes=900-5000", 1000) == (900, 999)  # clamped, not 416
    assert _parse_range("bytes=-100", 1000) == (900, 999)
    assert _parse_range("bytes=1000-", 1000) is None
    assert _parse_range("garbage", 1000) is None


def test_upload_name_and_dedupe(client: TestClient, monkeypatch):
    import hashlib
    from pathlib import Path

    async def fake_probe(path):
        from app.models.schemas import ProbeResponse
        p = Path(path)
        return ProbeResponse(file_path=str(p.resolve()), file_name=p.name, file_size=p.stat().st_size,
                             duration=1.0, format_name="matroska")

    monkeypatch.setattr("app.api.routes_media.probe_media_file", fake_probe)
    data_a = b"x" * 1234
    hash_a = hashlib.sha256(data_a).hexdigest()
    data_b = b"y" * 1234
    hash_b = hashlib.sha256(data_b).hexdigest()

    first = client.post("/api/media/upload", files={"file": ("My Show [1080p].mkv", io.BytesIO(data_a))}).json()
    assert first["file_name"] == "My Show [1080p].mkv"
    assert first["is_upload"] is True

    # Matching name + size + sha256 reuses the existing upload
    found = client.get(
        "/api/media/upload/existing",
        params={"name": "My Show [1080p].mkv", "size": 1234, "sha256": hash_a},
    ).json()
    assert found["file_path"] == first["file_path"]

    # Same name + size but different content hash (or omitted hash) must NOT reuse the older upload
    assert client.get(
        "/api/media/upload/existing",
        params={"name": "My Show [1080p].mkv", "size": 1234, "sha256": hash_b},
    ).json() is None
    assert client.get(
        "/api/media/upload/existing",
        params={"name": "My Show [1080p].mkv", "size": 1234},
    ).json() is None

    # Identical re-upload deduplicates to the first file
    second = client.post("/api/media/upload", files={"file": ("My Show [1080p].mkv", io.BytesIO(data_a))}).json()
    assert second["file_path"] == first["file_path"]

    # Different cut with the same name and byte length is preserved as a separate file
    different = client.post("/api/media/upload", files={"file": ("My Show [1080p].mkv", io.BytesIO(data_b))}).json()
    assert different["file_path"] != first["file_path"]
    assert Path(first["file_path"]).read_bytes() == data_a
    assert Path(different["file_path"]).read_bytes() == data_b
    assert _display_name("1a2b3c4d_Movie.mkv") == "Movie.mkv"


def test_upload_candidates_and_hash_cache(client: TestClient, monkeypatch):
    import hashlib
    from pathlib import Path
    import app.api.routes_media as media

    async def fake_probe(path):
        from app.models.schemas import ProbeResponse
        p = Path(path)
        return ProbeResponse(file_path=str(p.resolve()), file_name=p.name, file_size=p.stat().st_size,
                             duration=1.0, format_name="matroska")

    monkeypatch.setattr("app.api.routes_media.probe_media_file", fake_probe)
    name = f"cand_{uuid.uuid4().hex[:6]}.mkv"
    data = b"c" * 2048

    count = lambda size: client.get("/api/media/upload/candidates", params={"name": name, "size": size}).json()["count"]
    assert count(2048) == 0
    first = client.post("/api/media/upload", files={"file": (name, io.BytesIO(data))}).json()
    assert count(2048) == 1
    assert count(2049) == 0

    # The hash computed while saving is cached, so the lookup does not read the file again
    reads = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: (reads.append(str(f)), real_open(f, *a, **k))[1])
    found = client.get(
        "/api/media/upload/existing",
        params={"name": name, "size": 2048, "sha256": hashlib.sha256(data).hexdigest()},
    ).json()
    monkeypatch.setattr("builtins.open", real_open)
    assert found["file_path"] == first["file_path"]
    assert not any(Path(r).name.endswith(name) for r in reads)

    # A file changed in place is hashed again rather than trusted from the cache
    path = Path(first["file_path"])
    path.write_bytes(b"d" * 2048)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # coarse clocks: force a new mtime
    assert media._find_existing_upload(name, 2048, hashlib.sha256(data).hexdigest()) is None
    assert media._find_existing_upload(name, 2048, hashlib.sha256(b"d" * 2048).hexdigest()) == path


@pytest.mark.asyncio
async def test_ws_monitor_disconnect_during_broadcast_does_not_fail_job(client: TestClient, monkeypatch):
    from app.core.ws_manager import ws_manager
    from app.pipeline.orchestrator import PipelineOrchestrator

    fake = FakeTranslator()
    monkeypatch.setattr(TranslateStage, "_make_translator", lambda self, ctx: fake)
    job_id = _job_with_segments(["Hello world"], target_language="Hebrew")

    received_by_survivor = []

    class DisconnectingMonitor:
        async def accept(self):
            pass

        async def send_json(self, payload):
            # Disconnect during the send while broadcast() is iterating subscribers
            ws_manager.disconnect(job_id, self)
            raise RuntimeError("Client closed connection mid-send")

    class SurvivingMonitor:
        async def accept(self):
            pass

        async def send_json(self, payload):
            received_by_survivor.append(payload)

    m1 = DisconnectingMonitor()
    m2 = SurvivingMonitor()
    await ws_manager.connect(job_id, m1)
    await ws_manager.connect(job_id, m2)

    orchestrator = PipelineOrchestrator(job_id=job_id, mode="translate")
    await orchestrator.run()

    ws_manager.disconnect(job_id, m2)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "COMPLETED"
    assert any(evt.get("event") == "JOB_COMPLETED" for evt in received_by_survivor)


def test_per_job_settings_honor_explicit_false_over_global_true(client: TestClient, monkeypatch):
    from app.core.user_settings import load_settings, save_settings

    video = settings.storage_dir / "uploads" / f"{uuid.uuid4().hex[:8]}_settings_test.mkv"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr("app.api.routes_jobs.spawn_background", lambda coro: (coro.close(), None)[1])

    original = load_settings()
    try:
        updated = original.model_copy(
            update={"convert_to_metric": True, "remove_sdh_speakers": True, "skip_end_credits": True}
        )
        save_settings(updated)

        # Omitted fields inherit the saved global defaults
        omitted_res = client.post(
            "/api/jobs",
            json={"video_path": str(video), "enable_translation": False},
        )
        assert omitted_res.status_code == 200
        omitted_job = omitted_res.json()
        assert omitted_job["convert_to_metric"] is True
        assert omitted_job["remove_sdh_speakers"] is True
        assert omitted_job["skip_end_credits"] is True

        # Explicit False overrides the saved global True
        explicit_res = client.post(
            "/api/jobs",
            json={
                "video_path": str(video),
                "enable_translation": False,
                "convert_to_metric": False,
                "remove_sdh_speakers": False,
                "skip_end_credits": False,
            },
        )
        assert explicit_res.status_code == 200
        explicit_job = explicit_res.json()
        assert explicit_job["convert_to_metric"] is False
        assert explicit_job["remove_sdh_speakers"] is False
        assert explicit_job["skip_end_credits"] is False
    finally:
        save_settings(original)
        video.unlink(missing_ok=True)


def test_delete_job_removes_unused_upload(client: TestClient):
    upload = (settings.storage_dir / "uploads" / "deadbeef_clip.mkv")
    upload.write_bytes(b"123")
    res = client.post("/api/jobs", json={"video_path": str(upload), "enable_translation": False})
    job_id = res.json()["id"]
    assert res.json()["is_upload"] is True
    PipelineCancelledWait = _wait_for(client, job_id)  # probe fails on a fake file -> FAILED
    assert PipelineCancelledWait["status"] in ("FAILED", "CANCELLED", "COMPLETED")
    assert client.delete(f"/api/jobs/{job_id}").json()["upload_deleted"] is True
    assert not upload.exists()


def test_security_media_extension_api_key_and_ollama_validation(client: TestClient, tmp_path, monkeypatch):
    from app.core import config as cfg

    secret_env = tmp_path / "secret.env"
    secret_env.write_text("GEMINI_API_KEY=topsecret\n", encoding="utf-8")
    notes_txt = tmp_path / "notes.txt"
    notes_txt.write_text("hello", encoding="utf-8")
    db_file = tmp_path / "subbuild.db"
    db_file.write_bytes(b"SQLite format 3\x00")

    # Non-media local paths must be rejected by probe-path and job creation
    for bad in (secret_env, notes_txt, db_file):
        assert client.post("/api/media/probe-path", json={"file_path": str(bad)}).status_code == 400
        assert client.post("/api/jobs", json={"video_path": str(bad), "enable_translation": False}).status_code == 400

    # Non-media uploads must be rejected before saving
    assert client.post("/api/media/upload", files={"file": ("secret.env", io.BytesIO(b"KEY=1"))}).status_code == 400
    assert client.post("/api/media/upload", files={"file": ("notes.txt", io.BytesIO(b"hello"))}).status_code == 400

    # Streaming rejects non-media paths even if already present in DB
    bad_job_id = f"test_badstream_{uuid.uuid4().hex[:8]}"
    with Session(engine) as session:
        session.add(Job(id=bad_job_id, video_path=str(secret_env), file_name="secret.env", status=JobStatus.COMPLETED))
        session.commit()
    assert client.get(f"/api/media/stream/{bad_job_id}").status_code == 400

    # API key persistence strips newline/control characters so .env cannot be injected
    env_target = tmp_path / "written.env"
    orig_key = settings.gemini_api_key
    monkeypatch.setattr(cfg, "env_file_path", lambda: env_target)
    try:
        res_key = client.put("/api/settings/api-key", json={"api_key": "AIzaGoodKey\r\nHOST=0.0.0.0\n"})
        assert res_key.status_code == 200
        written = env_target.read_text(encoding="utf-8")
        assert written.strip() == "GEMINI_API_KEY=AIzaGoodKeyHOST=0.0.0.0"
        assert "\nHOST=0.0.0.0" not in written
    finally:
        settings.gemini_api_key = orig_key

    # Ollama setup rejects invalid/URL model specs
    assert client.post("/api/settings/ollama/setup", json={"model": "http://evil.example/model"}).status_code == 400
    assert client.post("/api/settings/ollama/setup", json={"model": "qwen2.5:7b; rm -rf /"}).status_code == 400


def test_update_segment_time_renumbers_indices(client: TestClient):
    job_id = _job_with_segments(["First cue", "Second cue"])
    segs = client.get(f"/api/jobs/{job_id}").json()["segments"]
    seg1, seg2 = segs[0], segs[1]
    assert seg1["index"] == 1 and seg2["index"] == 2

    # Move seg2 before seg1
    patched = client.patch(
        f"/api/jobs/{job_id}/segments/{seg2['id']}",
        json={"start_time": 0.5, "end_time": 1.5},
    )
    assert patched.status_code == 200
    assert patched.json()["index"] == 1

    updated = client.get(f"/api/jobs/{job_id}").json()["segments"]
    by_id = {s["id"]: s for s in updated}
    assert by_id[seg2["id"]]["index"] == 1
    assert by_id[seg1["id"]]["index"] == 2


class SlowTranscriber(BaseTranscriber):
    """Emits segments slowly from a worker thread and honours the cancel flag."""

    def __init__(self):
        self.stopped_early = False

    async def transcribe(self, audio_path, language=None, on_segment_callback=None, cancel_flag=None):
        for i in range(1, 200):
            if cancel_flag is not None and cancel_flag.is_set():
                self.stopped_early = True
                raise PipelineCancelled()
            await asyncio.sleep(0.01)
            if on_segment_callback:
                await on_segment_callback(RawSegment(index=i, start_time=i, end_time=i + 0.5, text="hi"), i / 2)
        return []


@pytest.mark.asyncio
async def test_cancel_stops_transcription_midway():
    transcriber = SlowTranscriber()
    ctx = PipelineContext(job_id="t", video_path="x", file_name="x", audio_wav_path="x.wav")

    async def cancel_soon():
        await asyncio.sleep(0.1)
        ctx.cancel_flag.set()

    asyncio.ensure_future(cancel_soon())
    with pytest.raises(PipelineCancelled):
        await TranscribeStage(transcriber).execute(ctx)
    assert transcriber.stopped_early
    assert len(ctx.raw_segments) < 50


def test_migration_adds_missing_columns(tmp_path):
    from sqlalchemy import create_engine, inspect, text
    import app.core.database as db

    old = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with old.begin() as conn:
        conn.execute(text("CREATE TABLE jobs (id VARCHAR PRIMARY KEY, video_path VARCHAR, file_name VARCHAR, gemini_api_key_override VARCHAR)"))
        conn.execute(text("CREATE TABLE subtitle_segments (id INTEGER PRIMARY KEY, job_id VARCHAR)"))
        conn.execute(text("INSERT INTO jobs VALUES ('j', 'x', 'x.mkv', 'secret')"))
        conn.execute(text(
            "INSERT INTO jobs VALUES ('u', 'C:\\Tools\\app\\storage\\uploads\\1a2b3c4d_Show [720p].mkv', '1a2b3c4d_Show [720p].mkv', NULL)"
        ))
    original = db.engine
    db.engine = old
    try:
        db._migrate()
    finally:
        db.engine = original
    cols = {c["name"] for c in inspect(old).get_columns("jobs")}
    assert {"is_upload", "warning_message", "detected_language", "glossary_json"} <= cols
    assert "translation_failed" in {c["name"] for c in inspect(old).get_columns("subtitle_segments")}
    with old.begin() as conn:
        assert conn.execute(text("SELECT gemini_api_key_override FROM jobs WHERE id='j'")).scalar() is None
        row = conn.execute(text("SELECT is_upload, file_name FROM jobs WHERE id='u'")).one()
        assert row == (1, "Show [720p].mkv")
        assert conn.execute(text("SELECT is_upload FROM jobs WHERE id='j'")).scalar() == 0


def test_cleanup_never_deletes_files_placed_by_hand(tmp_path):
    import os
    from app.core.housekeeping import cleanup_storage

    uploads = settings.storage_dir / "uploads"
    old = time.time() - 10 * 24 * 3600
    by_hand = uploads / "The Last Ship S01E01 Phase Six.mkv"
    app_copy = uploads / "0badc0de_abandoned.mkv"
    for f in (by_hand, app_copy):
        f.write_bytes(b"x")
        os.utime(f, (old, old))
    cleanup_storage()
    assert by_hand.exists()          # user's file: untouched, however old
    # The app's own abandoned copy goes only once it is old by both mtime and ctime;
    # ctime can't be faked in a test, so it survives here — the point is the prefix rule.
    by_hand.unlink()
    app_copy.unlink(missing_ok=True)


def test_burn_font_fallback_across_platforms(tmp_path, monkeypatch):
    from app.core import video_embed as ve

    assert ve._burn_font_name("win32") == "Arial"
    assert "FontName=Arial," in ve.burn_style("medium", platform="win32")

    missing_mac = tmp_path / "missing_arial.ttf"
    monkeypatch.setattr(ve, "_DARWIN_ARIAL_PATHS", (missing_mac,))
    assert ve._burn_font_name("darwin") == "Helvetica Neue"

    present_mac = tmp_path / "Arial.ttf"
    present_mac.write_bytes(b"font")
    monkeypatch.setattr(ve, "_DARWIN_ARIAL_PATHS", (present_mac,))
    assert ve._burn_font_name("darwin") == "Arial"

    empty_linux_fonts = tmp_path / "linux_empty"
    empty_linux_fonts.mkdir()
    monkeypatch.setattr(ve, "_LINUX_FONT_DIRS", (empty_linux_fonts,))
    assert ve._burn_font_name("linux") == "DejaVu Sans"

    noto_dir = tmp_path / "linux_noto"
    noto_dir.mkdir()
    (noto_dir / "NotoSans-Regular.ttf").write_bytes(b"font")
    monkeypatch.setattr(ve, "_LINUX_FONT_DIRS", (noto_dir,))
    assert ve._burn_font_name("linux") == "Noto Sans"
    assert "FontName=Noto Sans," in ve.burn_style("large", platform="linux")


def test_split_merge_and_shift_segments_and_history(client: TestClient):
    job_id = _job_with_segments(["Hello world foo bar.", "Second cue"], target_language="Hebrew")
    segs = client.get(f"/api/jobs/{job_id}").json()["segments"]
    seg1, seg2 = segs[0], segs[1]
    client.patch(
        f"/api/jobs/{job_id}/segments",
        json=[
            {"id": seg1["id"], "translated_text": "שלום עולם טוב מאוד."},
            {"id": seg2["id"], "translated_text": "שורה שנייה"},
        ],
    )

    # Split first cue at t=2.8
    split_res = client.post(f"/api/jobs/{job_id}/segments/{seg1['id']}/split", json={"split_time": 2.8})
    assert split_res.status_code == 200, split_res.text
    after_split = split_res.json()["segments"]
    assert len(after_split) == 3
    assert [s["index"] for s in after_split] == [1, 2, 3]
    assert after_split[0]["start_time"] == 2.0 and after_split[0]["end_time"] == 2.8
    assert after_split[1]["start_time"] == 2.8 and after_split[1]["end_time"] == 3.5

    # Undo -> back to 2 segments; Redo -> back to 3 segments
    undone = client.post(f"/api/jobs/{job_id}/history/undo").json()
    assert len(undone["segments"]) == 2
    redone = client.post(f"/api/jobs/{job_id}/history/redo").json()
    assert len(redone["segments"]) == 3

    # Merge the two split parts back into one
    merge_ids = [redone["segments"][0]["id"], redone["segments"][1]["id"]]
    merge_res = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": merge_ids})
    assert merge_res.status_code == 200, merge_res.text
    after_merge = merge_res.json()["segments"]
    assert len(after_merge) == 2
    assert after_merge[0]["start_time"] == 2.0 and after_merge[0]["end_time"] == 3.5
    assert "\n" in after_merge[0]["original_text"]
    assert "\n" in after_merge[0]["translated_text"]

    # Shift all forward +1.5s, then backward -10.0s (clamped at 0.0 preserving duration)
    shift_fwd = client.post(f"/api/jobs/{job_id}/segments/shift", json={"offset_seconds": 1.5})
    assert shift_fwd.status_code == 200
    assert shift_fwd.json()["segments"][0]["start_time"] == 3.5
    assert shift_fwd.json()["segments"][1]["start_time"] == 5.5

    shift_back = client.post(
        f"/api/jobs/{job_id}/segments/shift",
        json={"offset_seconds": -10.0, "segment_ids": [shift_fwd.json()["segments"][0]["id"]]},
    )
    assert shift_back.status_code == 200
    clamped = shift_back.json()["segments"][0]
    assert clamped["start_time"] == 0.0
    assert clamped["end_time"] == 1.5

    actions = [e["action"] for e in client.get(f"/api/jobs/{job_id}/history").json()["entries"]]
    assert "split" in actions and "merge" in actions and "shift_time" in actions


def test_bilingual_srt_and_vtt_export(client: TestClient):
    job_id = _job_with_segments(["Hello friend."], target_language="Hebrew")
    seg_id = client.get(f"/api/jobs/{job_id}").json()["segments"][0]["id"]
    client.patch(f"/api/jobs/{job_id}/segments/{seg_id}", json={"translated_text": "שלום חבר."})

    srt_res = client.get(f"/api/jobs/{job_id}/export/srt?use_translated=true&bilingual=true&apply_rtl=false")
    assert srt_res.status_code == 200
    assert ".he.dual.srt" in srt_res.headers["content-disposition"]
    assert "Hello friend.\nשלום חבר." in srt_res.text

    vtt_res = client.get(f"/api/jobs/{job_id}/export/vtt?use_translated=true&bilingual=true&apply_rtl=false")
    assert vtt_res.status_code == 200
    assert ".he.dual.vtt" in vtt_res.headers["content-disposition"]
    assert "WEBVTT" in vtt_res.text and "Hello friend.\nשלום חבר." in vtt_res.text


def test_extended_qa_flags_overlap_and_extreme_durations():
    from types import SimpleNamespace as NS
    from app.pipeline.qa import check_rows

    rows = [
        NS(id=1, index=1, start_time=0.0, end_time=0.3, original_text="Hi", translated_text="היי",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=2, index=2, start_time=1.0, end_time=4.0, original_text="One", translated_text="אחת",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=3, index=3, start_time=3.5, end_time=12.0, original_text="Two", translated_text="שתיים",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=4, index=4, start_time=1.0, end_time=3.0, original_text="[TITLE]", translated_text="[כותרת]",
           segment_type="ON_SCREEN", translation_failed=False),
    ]
    issues = check_rows(rows, glossary=None)
    by_pair = {(i.segment_id, i.kind) for i in issues}
    assert (1, "timing") in by_pair
    assert (3, "overlap") in by_pair
    assert (3, "timing") in by_pair
    assert (4, "overlap") not in by_pair  # ON_SCREEN lane is separate from DIALOGUE lane
