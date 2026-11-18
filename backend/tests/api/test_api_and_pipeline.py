import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.context import PipelineContext, RawSegment
from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines, recover_interrupted_jobs
from app.pipeline.stages.refine_stage import RefineStage
from app.pipeline.stages.translate_stage import TranslateStage
from app.providers.translation.base import TranslatableSegment
from app.providers.translation.noop import NoopTranslator
from tests.fixtures.builders import create_job_with_segments, insert_job
from tests.fixtures.fakes import FakeTranslator


@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client


def test_system_info(client: TestClient):
    response = client.get("/api/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "hardware" in data
    assert "is_gemini_configured" in data
    assert "whisper_default_model" in data
    assert isinstance(data.get("version"), str) and len(data["version"]) > 0


@pytest.mark.api
def test_system_update_endpoints_and_active_job_conflict(client: TestClient, monkeypatch):
    from app.api import routes_system

    fake_status = {
        "current_version": "1.5.0",
        "current_commit": "abc1234",
        "branch": "main",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "def5678",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": "https://github.com/talevi83/SubBuildStudio/releases/tag/v1.6.0",
        "release_notes": "Release 1.6.0",
        "error": None,
    }
    monkeypatch.setattr(routes_system, "check_for_updates", lambda: fake_status)
    monkeypatch.setattr(routes_system, "get_cached_update_status", lambda: fake_status)
    monkeypatch.setattr(
        routes_system,
        "apply_update",
        lambda rebuild_frontend=True: {
            "updated": True,
            "previous_version": "1.5.0",
            "new_version": "1.6.0",
            "new_commit": "def5678",
            "requirements_updated": False,
            "frontend_rebuilt": rebuild_frontend,
            "restart_required": True,
            "skipped_reason": None,
            "message": "Updated to v1.6.0 (def5678).",
        },
    )

    chk = client.post("/api/system/update/check")
    assert chk.status_code == 200
    assert chk.json()["latest_version"] == "1.6.0"
    assert chk.json()["update_available"] is True

    sys_info = client.get("/api/system/info").json()
    assert sys_info["update"]["update_available"] is True

    # While an active job exists in DB, update apply must fail with 409 Conflict
    active_job_id = create_job_with_segments(
        ["Line 1"],
        id=f"active_{uuid.uuid4().hex[:8]}",
        status=JobStatus.TRANSLATING,
    )
    blocked = client.post("/api/system/update/apply")
    assert blocked.status_code == 409

    # Once job finishes, apply succeeds cleanly
    with Session(engine) as s:
        row = s.get(Job, active_job_id)
        row.status = JobStatus.COMPLETED
        s.add(row)
        s.commit()

    applied = client.post("/api/system/update/apply")
    assert applied.status_code == 200
    assert applied.json()["updated"] is True
    assert applied.json()["new_version"] == "1.6.0"


@pytest.mark.asyncio
async def test_noop_translator():
    translator = NoopTranslator()
    segments = [
        TranslatableSegment(
            index=1,
            start_time=1.0,
            end_time=3.0,
            original_text="Hello world",
        )
    ]
    res = await translator.translate_segments(segments, "en", "he")
    assert len(res) == 1
    assert res[0].original_text == "Hello world"


@pytest.mark.asyncio
async def test_refine_stage_execution():
    stage = RefineStage()
    ctx = PipelineContext(
        job_id="test_job",
        video_path="test.mp4",
        file_name="test.mp4",
        raw_segments=[
            RawSegment(index=1, start_time=0.0, end_time=2.0, text="First subtitle block."),
            RawSegment(index=2, start_time=2.1, end_time=4.0, text="Second subtitle block."),
        ],
    )
    await stage.execute(ctx)
    assert len(ctx.refined_segments) == 2
    assert ctx.refined_segments[0]["index"] == 1
    assert ctx.refined_segments[1]["index"] == 2


import uuid


def test_export_endpoints(client: TestClient):
    # Insert test job and segments into db
    job_id = f"test_export_{uuid.uuid4().hex[:8]}"
    with Session(engine) as session:
        job = Job(
            id=job_id,
            video_path="test.mp4",
            file_name="test_export.mp4",
            status=JobStatus.COMPLETED,
        )
        session.add(job)
        session.commit()

        seg1 = SubtitleSegmentRecord(
            job_id=job_id,
            index=1,
            start_time=1.0,
            end_time=3.5,
            original_text="Hello friend",
            translated_text="שלום חבר",
        )
        session.add(seg1)
        session.commit()

    # Test SRT export
    res_srt = client.get(f"/api/jobs/{job_id}/export/srt?use_translated=true")
    assert res_srt.status_code == 200
    assert "00:00:01,000 --> 00:00:03,500" in res_srt.text
    assert "שלום חבר" in res_srt.text

    # Test VTT export
    res_vtt = client.get(f"/api/jobs/{job_id}/export/vtt?use_translated=false")
    assert res_vtt.status_code == 200
    assert "WEBVTT" in res_vtt.text
    assert "Hello friend" in res_vtt.text


def test_bulk_delete_and_clear_jobs(client: TestClient):
    ids = [f"test_bulk_{uuid.uuid4().hex[:8]}" for _ in range(3)]
    with Session(engine) as session:
        for jid in ids:
            session.add(Job(id=jid, video_path="test.mp4", file_name=f"{jid}.mp4", status=JobStatus.COMPLETED))
        session.commit()

    # Delete first two selected jobs
    res = client.post("/api/jobs/bulk-delete", json={"job_ids": ids[:2]})
    assert res.status_code == 200
    data = res.json()
    assert data["deleted_count"] == 2
    assert set(data["deleted_ids"]) == set(ids[:2])

    # Third job should still exist
    assert client.get(f"/api/jobs/{ids[2]}").status_code == 200
    assert client.get(f"/api/jobs/{ids[0]}").status_code == 404

    # Delete all remaining jobs
    res_all = client.post("/api/jobs/bulk-delete", json={"delete_all": True})
    assert res_all.status_code == 200
    assert ids[2] in res_all.json()["deleted_ids"]
    assert client.get("/api/jobs").json() == []


def test_system_info_lists_models(client: TestClient):
    data = client.get("/api/system/info").json()
    assert "large-v3-turbo" in data["whisper_models"]
    assert settings.gemini_model in data["gemini_models"]


def test_list_jobs_is_slim_but_has_counts(client: TestClient):
    job_id = create_job_with_segments(["a", "b", "c"])
    jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
    assert jobs[job_id]["segments"] == []
    assert jobs[job_id]["segment_count"] == 3


@pytest.mark.asyncio
async def test_ws_monitor_disconnect_during_broadcast_does_not_fail_job(client: TestClient, monkeypatch):
    from app.core.ws_manager import ws_manager

    fake = FakeTranslator()
    monkeypatch.setattr(TranslateStage, "_make_translator", lambda self, ctx: fake)
    job_id = create_job_with_segments(["Hello world"], target_language="Hebrew")

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


def test_bilingual_srt_and_vtt_export(client: TestClient):
    job_id = create_job_with_segments(["Hello friend."], target_language="Hebrew")
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


def test_get_job_returns_live_whisper_segments_while_transcribing(client: TestClient):
    job_id = insert_job(
        JobStatus.TRANSCRIBING,
        video_path="live.mkv",
        file_name="live.mkv",
        current_stage="TRANSCRIBING",
        progress_percent=25.0,
    )

    orchestrator = PipelineOrchestrator(job_id=job_id, mode="full")
    orchestrator._ctx = PipelineContext(job_id=job_id, video_path="live.mkv", file_name="live.mkv")
    orchestrator._ctx.raw_segments = [
        RawSegment(index=1, start_time=0.5, end_time=2.5, text="First live cue", confidence=0.95),
        RawSegment(index=2, start_time=3.0, end_time=5.0, text="Second live cue", confidence=0.91),
    ]
    try:
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert len(detail["segments"]) == 2
        assert [s["original_text"] for s in detail["segments"]] == ["First live cue", "Second live cue"]
        assert all(s["id"] is None for s in detail["segments"])

        listing = {j["id"]: j for j in client.get("/api/jobs").json()}
        assert listing[job_id]["segment_count"] == 2
    finally:
        active_pipelines.pop(job_id, None)


def test_recover_interrupted_jobs(client: TestClient):
    stuck = insert_job(JobStatus.TRANSCRIBING)
    done = insert_job(JobStatus.COMPLETED)

    assert recover_interrupted_jobs() >= 1

    with Session(engine) as session:
        stuck_job = session.get(Job, stuck)
        assert stuck_job.status == JobStatus.FAILED
        assert "TRANSCRIBING" in stuck_job.error_message
        assert session.get(Job, done).status == JobStatus.COMPLETED


def test_cancel_completed_job_is_rejected(client: TestClient):
    job_id = insert_job(JobStatus.COMPLETED)
    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.status_code == 409
    with Session(engine) as session:
        assert session.get(Job, job_id).status == JobStatus.COMPLETED


def test_cancel_orphaned_active_job(client: TestClient):
    job_id = insert_job(JobStatus.TRANSLATING)
    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.status_code == 200
    with Session(engine) as session:
        assert session.get(Job, job_id).status == JobStatus.CANCELLED


def test_export_filename_uses_language_and_supports_unicode(client: TestClient):
    job_id = insert_job(
        JobStatus.COMPLETED,
        file_name="סרט.mkv",
        target_language="Russian",
        source_language="en",
    )
    res = client.get(f"/api/jobs/{job_id}/export/srt?use_translated=true")
    assert res.status_code == 200
    cd = res.headers["content-disposition"]
    assert ".ru.srt" in cd
    assert "filename*=UTF-8''" in cd

    res = client.get(f"/api/jobs/{job_id}/export/vtt?use_translated=false")
    assert ".en.vtt" in res.headers["content-disposition"]

