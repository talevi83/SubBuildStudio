"""Storage report and cleanup never touch active jobs, user videos or the live server log."""
import os
import time
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine, init_db
from app.main import app
from app.models.db_models import Job, JobStatus


@pytest.fixture(name="store")
def store_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    for d in ("audio", "frames", "regions", "exports", "uploads", "logs/jobs"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


def _write(path, n=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)
    return path


def _job(status=JobStatus.COMPLETED, video="x.mkv", is_upload=False, updated=None):
    init_db()
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    with Session(engine) as s:
        s.add(Job(id=job_id, video_path=video, file_name="x.mkv", status=status, is_upload=is_upload,
                  updated_at=updated or datetime.now(timezone.utc)))
        s.commit()
    return job_id


def _report(client, days=30):
    return {c["id"]: c for c in client.get(f"/api/storage?older_than_days={days}").json()["categories"]}


def test_report_and_clean_file_categories(store):
    # Rows and files are created after startup, which itself recovers/cleans leftovers
    with TestClient(app) as client:
        active = _job(JobStatus.TRANSCRIBING)
        done = _job()
        _write(store / "audio" / f"{done}.wav", 300)
        _write(store / "audio" / f"{active}.wav", 500)          # running job: must stay
        _write(store / "frames" / done / "f1.jpg", 50)
        _write(store / "regions" / f"{done}.json", 20)
        _write(store / "logs" / "jobs" / f"{done}.log", 10)
        _write(store / "logs" / "server.log", 999)              # live log: must stay
        _write(store / "logs" / "server.log.1", 40)
        _write(store / "exports" / "a.srt", 5)
        r = _report(client)
        assert r["temp"]["bytes"] == 350 and r["temp"]["count"] == 2
        assert r["cache"]["bytes"] == 20
        assert r["logs"]["bytes"] == 50
        out = client.post("/api/storage/clean", json={"categories": ["temp", "cache", "logs", "exports"]}).json()
        assert out["freed_bytes"] == 350 + 20 + 50 + 5
    assert (store / "audio" / f"{active}.wav").exists()
    assert (store / "logs" / "server.log").exists()
    assert not (store / "logs" / "server.log.1").exists()
    assert not (store / "audio" / f"{done}.wav").exists()


def test_unreferenced_app_uploads_only(store):
    kept = _write(store / "uploads" / "1a2b3c4d_kept.mkv", 70)
    orphan = _write(store / "uploads" / "5e6f7a8b_orphan.mkv", 90)
    hour_ago = time.time() - 3600
    os.utime(orphan, (hour_ago, hour_ago))
    manual = _write(store / "uploads" / "my_own_file.mkv", 60)   # not made by the app
    _job(video=str(kept), is_upload=True)
    with TestClient(app) as client:
        assert _report(client)["uploads"]["bytes"] == 90
        client.post("/api/storage/clean", json={"categories": ["uploads"]})
    assert kept.exists() and manual.exists() and not orphan.exists()


def test_fresh_upload_without_a_job_yet_is_kept(store, monkeypatch):
    """Uploaded a minute ago, its job not started yet: cleaning must not pull it away."""
    from app.core import storage_cleanup as sc

    fresh = _write(store / "uploads" / "9a8b7c6d_just_dropped.mkv", 40)
    old = _write(store / "uploads" / "0a1b2c3d_forgotten.mkv", 30)
    monkeypatch.setattr(sc, "_arrived_at", lambda p: time.time() - (10 if p == fresh else 7200))
    with TestClient(app) as client:
        assert _report(client)["uploads"]["bytes"] == 30
        client.post("/api/storage/clean", json={"categories": ["uploads"]})
    assert fresh.exists() and not old.exists()


def test_old_jobs_removed_with_their_upload_but_not_user_videos(store):
    up = _write(store / "uploads" / "1a2b3c4d_old.mkv", 80)
    user_video = _write(store / "elsewhere" / "movie.mkv", 120)
    old = datetime(2001, 1, 1, tzinfo=timezone.utc)
    j1 = _job(video=str(up), is_upload=True, updated=old)
    j2 = _job(video=str(user_video), updated=old)
    j3 = _job(JobStatus.TRANSLATING, updated=old)              # running: must survive
    fresh = _job()
    with TestClient(app) as client:
        r = _report(client, days=365)
        assert r["jobs"]["count"] >= 2
        assert client.post("/api/storage/clean", json={"categories": ["jobs"]}).status_code == 400
        client.post("/api/storage/clean", json={"categories": ["jobs"], "older_than_days": 365})
        ids = {j["id"] for j in client.get("/api/jobs").json()}
    assert j1 not in ids and j2 not in ids and j3 in ids and fresh in ids
    assert not up.exists() and user_video.exists()


def test_unknown_category_rejected(store):
    with TestClient(app) as client:
        assert client.post("/api/storage/clean", json={"categories": ["nope"]}).status_code == 400


def test_embed_export_directories_reported_cleaned_and_deleted_with_job(store):
    with TestClient(app) as client:
        active = _job(JobStatus.TRANSLATING)
        done = _job()
        active_embed = _write(store / "exports" / f"embed_{active}" / "clip.he.subs.mkv", 250)
        done_embed = _write(store / "exports" / f"embed_{done}" / "clip.he.subs.mkv", 400)

        # Only finished job's embed directory is cleanable under 'exports'
        r = _report(client)
        assert r["exports"]["bytes"] == 400 and r["exports"]["count"] == 1

        out = client.post("/api/storage/clean", json={"categories": ["exports"]}).json()
        assert out["freed_bytes"] == 400
        assert not done_embed.parent.exists()
        assert active_embed.exists()

        # Deleting a job also removes its embed_{job_id} export folder
        another = _job()
        another_embed = _write(store / "exports" / f"embed_{another}" / "out.mp4", 150)
        assert client.delete(f"/api/jobs/{another}").status_code == 200
        assert not another_embed.parent.exists()


def test_cleanup_never_deletes_files_placed_by_hand(store):
    from app.core.housekeeping import cleanup_storage

    init_db()
    uploads = store / "uploads"
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
