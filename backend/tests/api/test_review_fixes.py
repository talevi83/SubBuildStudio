"""Regression tests for the security / bug review fixes (API level)."""
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core import edit_history as history
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from tests.fixtures.builders import create_job_with_segments, insert_job


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _segs(client, job_id):
    return client.get(f"/api/jobs/{job_id}").json()["segments"]


def _raw(client, method, url, body):
    """Send JSON text as-is: Python's json accepts NaN / Infinity, so a client can send them."""
    return client.request(method, url, content=body, headers={"Content-Type": "application/json"})


def _rows(job_id):
    with Session(engine) as s:
        rows = s.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()
        return sorted(rows, key=lambda r: r.start_time)


# ------------------------------------------------------------------ non-finite numbers

@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_segment_times_must_be_finite(client, value):
    job_id = create_job_with_segments(["a", "b", "c"])
    seg = _segs(client, job_id)[0]
    base = f"/api/jobs/{job_id}/segments"
    assert _raw(client, "PATCH", f"{base}/{seg['id']}", f'{{"start_time": {value}}}').status_code == 422
    assert _raw(client, "PATCH", f"{base}/{seg['id']}", f'{{"end_time": {value}}}').status_code == 422
    assert _raw(client, "PATCH", base, f'[{{"id": {seg["id"]}, "end_time": {value}}}]').status_code == 422
    assert _raw(client, "POST", base, f'{{"start_time": 50, "end_time": {value}, "original_text": "x"}}').status_code == 422
    assert _raw(client, "POST", f"{base}/shift", f'{{"offset_seconds": {value}}}').status_code == 422
    assert _raw(client, "POST", f"{base}/{seg['id']}/split", f'{{"split_time": {value}}}').status_code == 422
    assert _raw(client, "POST", f"/api/jobs/{job_id}/retime", f'{{"start": {value}}}').status_code == 422
    # Nothing was changed, and the project still exports
    assert [(s["start_time"], s["end_time"]) for s in _segs(client, job_id)] == [(2.0, 3.5), (4.0, 5.5), (6.0, 7.5)]
    assert client.get(f"/api/jobs/{job_id}/export/srt").status_code == 200


def test_estimate_and_storage_reject_non_finite_numbers(client):
    assert _raw(client, "POST", "/api/settings/estimate", '{"duration_seconds": Infinity}').status_code == 422
    assert _raw(client, "POST", "/api/settings/estimate", '{"duration_seconds": -5}').status_code == 422
    assert client.get("/api/storage", params={"older_than_days": "nan"}).status_code == 422


def test_sync_cues_must_be_finite(client, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    body = ('{"video_path": "%s", "sync_cues": [{"start": 0, "end": NaN, "text": "a"}, '
            '{"start": 1, "end": 2, "text": "b"}, {"start": 3, "end": 4, "text": "c"}]}') % video.as_posix()
    assert _raw(client, "POST", "/api/jobs", body).status_code == 422


# ------------------------------------------------------------------ edits during a run

def test_time_edits_and_moves_are_refused_while_translating(client):
    job_id = create_job_with_segments(["a", "b", "c"], status=JobStatus.TRANSLATING)
    first, second = _segs(client, job_id)[:2]
    url = f"/api/jobs/{job_id}/segments"
    assert client.patch(f"{url}/{first['id']}", json={"start_time": 100.0, "end_time": 101.0}).status_code == 409
    assert client.patch(url, json=[{"id": first["id"], "end_time": 3.9}]).status_code == 409
    assert client.post(f"{url}/move", json={"segment_id": first["id"], "to_position": 2}).status_code == 409
    assert [s["original_text"] for s in _segs(client, job_id)] == ["a", "b", "c"]
    # Fixing text while the translation runs stays possible
    assert client.patch(f"{url}/{second['id']}", json={"translated_text": "fixed"}).status_code == 200
    assert client.patch(url, json=[{"id": first["id"], "original_text": "A"}]).status_code == 200


def test_bulk_time_edit_renumbers_lines(client):
    job_id = create_job_with_segments(["a", "b", "c"])
    first = _segs(client, job_id)[0]
    r = client.patch(f"/api/jobs/{job_id}/segments", json=[{"id": first["id"], "start_time": 100.0, "end_time": 101.0}])
    assert r.status_code == 200
    assert [(row.original_text, row.index) for row in _rows(job_id)] == [("b", 1), ("c", 2), ("a", 3)]


# ------------------------------------------------------------------ merge

def test_merge_keeps_a_partly_untranslated_line_flagged(client):
    job_id = create_job_with_segments(["Hello", "World"])
    with Session(engine) as s:
        rows = sorted(s.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all(),
                      key=lambda r: r.index)
        rows[0].translated_text = "שלום"
        rows[1].translation_failed = True
        s.add_all(rows)
        s.commit()
        ids = [r.id for r in rows]
    merged = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": ids}).json()["segments"][0]
    assert merged["original_text"] == "Hello\nWorld"
    assert merged["translated_text"] == "שלום"
    assert merged["translation_failed"] is True


def test_merge_of_fully_translated_lines_is_not_flagged(client):
    job_id = create_job_with_segments(["Hello", "World"], translated=True)
    ids = [s["id"] for s in _segs(client, job_id)]
    merged = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": ids}).json()["segments"][0]
    assert merged["translation_failed"] is False


def test_merge_in_a_transcription_only_project_is_not_flagged(client):
    job_id = create_job_with_segments(["Hello", "World"])
    ids = [s["id"] for s in _segs(client, job_id)]
    merged = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": ids}).json()["segments"][0]
    assert merged["translated_text"] is None and merged["translation_failed"] is False


# ------------------------------------------------------------------ model names

def test_unknown_whisper_model_is_refused(client, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    for bad in ("attacker/evil-repo", "C:\\models\\x", "../x"):
        r = client.post("/api/jobs", json={"video_path": str(video), "whisper_model": bad})
        assert r.status_code == 400, bad
        assert "Whisper model" in r.json()["detail"]
    assert client.get("/api/jobs").json() == [] or all(j["video_path"] != str(video) for j in client.get("/api/jobs").json())


def test_gemini_model_must_be_a_plain_id(client, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    for bad in ("../../v1/files", "gemini?key=x", "a/b", " "):
        r = client.post("/api/jobs", json={"video_path": str(video), "gemini_model": bad})
        assert r.status_code == 400, bad
    job_id = create_job_with_segments(["a"], translated=True)
    assert client.post(f"/api/jobs/{job_id}/translate", json={"gemini_model": "../x"}).status_code == 400
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "COMPLETED"


def test_custom_gemini_model_id_is_still_allowed(client, tmp_path, monkeypatch):
    from app.pipeline import orchestrator as orch

    async def no_run(self):
        orch.active_pipelines.pop(self.job_id, None)

    monkeypatch.setattr(orch.PipelineOrchestrator, "run", no_run)
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    r = client.post("/api/jobs", json={"video_path": str(video), "gemini_model": "models/gemini-9-experimental"})
    assert r.status_code == 200
    assert r.json()["gemini_model"] == "models/gemini-9-experimental"


# ------------------------------------------------------------------ the event loop never waits on job_lock

def _unrelated_request_latency(client, start_blocked_request) -> float:
    held = threading.Event()
    release = threading.Event()

    def hold(job_id):
        with history.job_lock(job_id):  # e.g. a long /retime in the thread pool
            held.set()
            release.wait(5)

    job_id = start_blocked_request.job_id
    holder = threading.Thread(target=hold, args=(job_id,))
    holder.start()
    held.wait(2)
    waiter = threading.Thread(target=start_blocked_request)
    waiter.start()
    time.sleep(0.3)  # the blocked request is now waiting for the lock
    t0 = time.time()
    assert client.get("/api/glossary-profiles").status_code == 200
    latency = time.time() - t0
    release.set()
    waiter.join(5)
    holder.join(5)
    return latency


def test_retranslate_waits_for_the_job_lock_off_the_event_loop(client, monkeypatch):
    from app.pipeline import orchestrator as orch

    async def no_run(self):
        orch.active_pipelines.pop(self.job_id, None)

    monkeypatch.setattr(orch.PipelineOrchestrator, "run", no_run)
    job_id = create_job_with_segments(["a", "b"], translated=True)

    def request():
        client.post(f"/api/jobs/{job_id}/translate", json={})
    request.job_id = job_id
    assert _unrelated_request_latency(client, request) < 1.0


def test_embed_waits_for_the_job_lock_off_the_event_loop(client, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    job_id = create_job_with_segments(["a"], video_path=str(video))

    def request():
        # Target already exists -> answered "exists" without starting ffmpeg
        (tmp_path / "x.subs.mkv").write_bytes(b"\0")
        client.post(f"/api/jobs/{job_id}/export/embed", json={"mode": "soft", "use_translated": False})
    request.job_id = job_id
    assert _unrelated_request_latency(client, request) < 1.0


def test_two_concurrent_retranslations_start_one_run(client, monkeypatch):
    from app.pipeline import orchestrator as orch

    runs = []

    async def fake_run(self):
        runs.append(self.job_id)

    monkeypatch.setattr(orch.PipelineOrchestrator, "run", fake_run)
    job_id = create_job_with_segments(["a", "b"], translated=True)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(client.post(f"/api/jobs/{job_id}/translate", json={}).status_code))
               for _ in range(4)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    orch.active_pipelines.pop(job_id, None)
    assert sorted(codes) == [200, 409, 409, 409]
    assert len(runs) == 1


# ------------------------------------------------------------------ embed: one writer per output file

def test_embed_of_the_same_output_by_another_project_is_refused(client, tmp_path):
    from app.core import video_embed as ve

    video = tmp_path / "movie.mkv"
    video.write_bytes(b"\0" * 10)
    a = create_job_with_segments(["a"], video_path=str(video), file_name="movie.mkv")
    b = create_job_with_segments(["a"], video_path=str(video), file_name="movie.mkv")
    target = tmp_path / ve.output_name("movie.mkv", "en", "soft", ".mkv")
    busy = ve.EmbedTask(id="busy1", job_id=a, mode="soft", target=target, in_app_storage=False)
    ve._tasks[busy.id] = busy
    try:
        with Session(engine) as s:
            job = s.get(Job, b)
            job.source_language = "en"
            s.add(job)
            s.commit()
        r = client.post(f"/api/jobs/{b}/export/embed", json={"mode": "soft", "use_translated": False})
        assert r.status_code == 409
        assert "Another project" in r.json()["detail"]
    finally:
        ve._tasks.pop(busy.id, None)


# ------------------------------------------------------------------ streaming

def test_range_request_on_an_empty_file_is_416(client, tmp_path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")
    job_id = insert_job(video_path=str(video))
    r = client.get(f"/api/media/stream/{job_id}", headers={"Range": "bytes=-10"})
    assert r.status_code == 416
    assert r.headers["content-range"] == "bytes */0"


# ------------------------------------------------------------------ client-supplied names

def test_display_name_from_the_browser_is_reduced_to_a_safe_file_name(client, tmp_path, monkeypatch):
    from app.pipeline import orchestrator as orch

    async def no_run(self):
        orch.active_pipelines.pop(self.job_id, None)

    monkeypatch.setattr(orch.PipelineOrchestrator, "run", no_run)
    video = tmp_path / "v.mkv"
    video.write_bytes(b"\0")
    r = client.post("/api/jobs", json={"video_path": str(video), "file_name": "..\\..\\Windows\\CON.mkv"})
    assert r.status_code == 200
    assert r.json()["file_name"] == "CON_.mkv"


def test_export_name_never_uses_a_device_name(client):
    job_id = create_job_with_segments(["a"], file_name="nul.mkv")
    r = client.get(f"/api/jobs/{job_id}/export/srt")
    assert 'filename="nul_.srt"' in r.headers["content-disposition"]


def test_glossary_profile_cannot_be_a_device_name(client):
    job_id = create_job_with_segments(["a"])
    r = client.put(f"/api/jobs/{job_id}/glossary", json={"characters": [], "terms": [], "profile": "CON"})
    assert r.status_code == 200
    assert "CON_" in client.get("/api/glossary-profiles").json()
    assert "CON" not in client.get("/api/glossary-profiles").json()


# ------------------------------------------------------------------ deleting a running job

def test_pipeline_does_not_write_rows_for_a_deleted_job():
    from app.pipeline.context import PipelineCancelled
    from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines

    job_id = insert_job(JobStatus.REFINING)
    orch = PipelineOrchestrator(job_id=job_id)
    try:
        with Session(engine) as s:
            s.delete(s.get(Job, job_id))
            s.commit()
        with pytest.raises(PipelineCancelled):
            orch._save_segments([{"index": 1, "start_time": 0.0, "end_time": 1.0, "original_text": "late"}])
        assert orch.cancel_flag.is_set()
        assert _rows(job_id) == []
    finally:
        active_pipelines.pop(job_id, None)


# ------------------------------------------------------------------ translations land on the right rows

@pytest.mark.asyncio
async def test_translated_batch_is_saved_by_row_id_even_after_renumbering():
    from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines
    from app.providers.translation.base import TranslatableSegment

    job_id = create_job_with_segments(["one", "two", "three"], status=JobStatus.TRANSLATING)
    orch = PipelineOrchestrator(job_id=job_id, mode="translate")
    try:
        ctx = orch._build_context()
        assert [s["original_text"] for s in ctx.refined_segments] == ["one", "two", "three"]
        # Meanwhile the numbers change (an older client, a retime...): "one" is now #3, "three" #1
        with Session(engine) as s:
            rows = {r.original_text: r for r in s.exec(
                select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()}
            rows["one"].index, rows["three"].index = 3, 1
            s.add_all(rows.values())
            s.commit()
        snapshot = {s["index"]: s for s in ctx.refined_segments}
        batch = [TranslatableSegment(index=i, start_time=snapshot[i]["start_time"], end_time=snapshot[i]["end_time"],
                                     original_text=snapshot[i]["original_text"],
                                     translated_text=f"T({snapshot[i]['original_text']})") for i in (1, 3)]
        await orch._persist_translations(batch)
        by_text = {r.original_text: r.translated_text for r in _rows(job_id)}
        assert by_text == {"one": "T(one)", "two": None, "three": "T(three)"}
    finally:
        active_pipelines.pop(job_id, None)
