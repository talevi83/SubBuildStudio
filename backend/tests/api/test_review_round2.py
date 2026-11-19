"""Regression tests for the second code review: upload dedupe race, history locking, input bounds."""
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

import app.api.routes_media as media
from app.core import edit_history as history
from app.core.database import engine
from app.models.db_models import Job, JobStatus
from tests.fixtures.builders import create_job_with_segments


def test_two_identical_uploads_finishing_together_keep_exactly_one_copy(monkeypatch):
    folder = media.uploads_dir()
    data = b"same video" * 1000
    a = folder / "aaaaaaaa_Race.mkv"
    b = folder / "bbbbbbbb_Race.mkv"
    a.write_bytes(data)
    b.write_bytes(data)
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    real_find = media._find_existing_upload

    def slow_find(*args, **kwargs):  # widen the window in which both look for the other
        found = real_find(*args, **kwargs)
        time.sleep(0.2)
        return found

    monkeypatch.setattr(media, "_find_existing_upload", slow_find)
    results = {}
    threads = [threading.Thread(target=lambda p=p: results.__setitem__(p.name, media._keep_one_copy(p, "Race.mkv", digest)))
               for p in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    survivors = [p for p in (a, b) if p.exists()]
    try:
        assert len(survivors) == 1
        assert set(results.values()) == {survivors[0]}  # both uploads point at the copy that is left
    finally:
        for p in survivors:
            p.unlink()


def test_undo_waits_for_the_lock_and_then_sees_a_run_that_started_meanwhile(client: TestClient):
    job_id = create_job_with_segments(["a", "b"])
    seg = client.get(f"/api/jobs/{job_id}").json()["segments"][0]
    assert client.patch(f"/api/jobs/{job_id}/segments/{seg['id']}", json={"original_text": "A"}).status_code == 200

    answer = {}
    with history.job_lock(job_id):  # e.g. a retranslate request that is just starting its run
        worker = threading.Thread(target=lambda: answer.update(res=client.post(f"/api/jobs/{job_id}/history/undo")))
        worker.start()
        time.sleep(0.3)  # the undo request is now waiting for the lock
        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.TRANSLATING
            session.add(job)
            session.commit()
    worker.join(timeout=10)
    assert answer["res"].status_code == 409
    texts = [s["original_text"] for s in client.get(f"/api/jobs/{job_id}").json()["segments"]]
    assert texts == ["A", "b"]  # the running translation's rows were not swapped underneath it


def test_export_line_length_must_be_sensible(client: TestClient):
    job_id = create_job_with_segments(["hello"])
    assert client.get(f"/api/jobs/{job_id}/export/srt", params={"max_chars_per_line": -5}).status_code == 422
    assert client.get(f"/api/jobs/{job_id}/export/vtt", params={"max_chars_per_line": 5000}).status_code == 422
    assert client.get(f"/api/jobs/{job_id}/export/srt", params={"max_chars_per_line": 37}).status_code == 200


def test_an_unknown_on_screen_position_is_refused_when_creating_a_job(client: TestClient, tmp_path: Path):
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"\x1a\x45\xdf\xa3")
    res = client.post("/api/jobs", json={"video_path": str(video), "onscreen_position": "bottom_somewhere"})
    assert res.status_code == 400
    assert "onscreen_position" in res.json()["detail"]


def test_middle_center_is_a_valid_saved_setting():
    from app.core.user_settings import AppSettings, load_settings

    data = load_settings().model_dump()
    data["onscreen_position"] = "middle_center"
    assert AppSettings.model_validate(data).onscreen_position == "middle_center"


def test_editing_lines_of_a_running_job_is_refused_even_before_its_status_is_written(client: TestClient):
    from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines

    job_id = create_job_with_segments(["a", "b"])  # status COMPLETED, but a run was just registered
    PipelineOrchestrator(job_id=job_id, mode="translate")
    try:
        seg = client.get(f"/api/jobs/{job_id}").json()["segments"][0]
        assert client.delete(f"/api/jobs/{job_id}/segments/{seg['id']}").status_code == 409
    finally:
        active_pipelines.pop(job_id, None)
