"""Waveform peak extraction, caching, cleanup, and discrete timing drag history tests."""
import shutil
import struct
import subprocess
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.waveform import PEAKS_PER_SECOND, WAVEFORM_SAMPLE_RATE, cache_path, compute_peaks_from_pcm


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_job(video_path: str, duration: float = 3.0) -> str:
    job_id = f"wf_{uuid.uuid4().hex[:10]}"
    with Session(engine) as s:
        s.add(
            Job(
                id=job_id,
                video_path=video_path,
                file_name="clip.mkv",
                status=JobStatus.COMPLETED,
                video_duration=duration,
            )
        )
        s.add(
            SubtitleSegmentRecord(
                job_id=job_id,
                index=1,
                start_time=0.5,
                end_time=1.8,
                original_text="Test line",
            )
        )
        s.commit()
    return job_id


def test_compute_peaks_from_pcm_normalizes_silence_and_speech():
    # 0.5s silence + 0.5s 16000-amplitude burst at 8 kHz
    silence_samples = [0] * (WAVEFORM_SAMPLE_RATE // 2)
    burst_samples = [16000 if i % 2 == 0 else -16000 for i in range(WAVEFORM_SAMPLE_RATE // 2)]
    pcm = struct.pack("<" + "h" * len(silence_samples + burst_samples), *(silence_samples + burst_samples))

    peaks = compute_peaks_from_pcm(pcm)
    assert len(peaks) == PEAKS_PER_SECOND
    assert all(p == 0.0 for p in peaks[: PEAKS_PER_SECOND // 2])
    assert all(p == 1.0 for p in peaks[PEAKS_PER_SECOND // 2 :])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_waveform_endpoint_extracts_caches_and_cleans_up(client: TestClient, tmp_path, monkeypatch):
    clip = tmp_path / "tone.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=2",
            "-shortest",
            str(clip),
        ],
        check=True,
    )

    job_id = _make_job(str(clip), duration=2.0)
    res1 = client.get(f"/api/jobs/{job_id}/waveform")
    assert res1.status_code == 200, res1.text
    data = res1.json()
    assert data["job_id"] == job_id
    assert data["peaks_per_second"] == PEAKS_PER_SECOND
    assert len(data["peaks"]) >= 80
    assert max(data["peaks"]) > 0.5
    assert cache_path(job_id).is_file()

    # Second fetch must come purely from cache without invoking FFmpeg again
    import app.pipeline.waveform as wf_mod

    monkeypatch.setattr(wf_mod, "_extract_pcm_s16le", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("FFmpeg called")))
    res2 = client.get(f"/api/jobs/{job_id}/waveform")
    assert res2.status_code == 200
    assert res2.json() == data

    # Deleting the job must remove its waveform cache file
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert not cache_path(job_id).exists()


def test_storage_cleanup_removes_waveform_cache(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    (tmp_path / "regions").mkdir(parents=True, exist_ok=True)
    job_id = _make_job(str(tmp_path / "missing.mkv"))
    wf_file = cache_path(job_id)
    wf_file.write_text('{"job_id":"x","peaks_per_second":50,"duration":1,"peaks":[0.1]}', encoding="utf-8")

    clean_res = client.post("/api/storage/clean", json={"categories": ["cache"]})
    assert clean_res.status_code == 200
    assert not wf_file.exists()


def test_consecutive_timing_drags_create_discrete_undo_steps(client: TestClient, tmp_path):
    job_id = _make_job(str(tmp_path / "clip.mkv"))
    seg_id = client.get(f"/api/jobs/{job_id}").json()["segments"][0]["id"]

    r1 = client.patch(f"/api/jobs/{job_id}/segments/{seg_id}", json={"start_time": 0.7, "end_time": 2.0})
    assert r1.status_code == 200
    r2 = client.patch(f"/api/jobs/{job_id}/segments/{seg_id}", json={"start_time": 0.9, "end_time": 2.2})
    assert r2.status_code == 200

    hist = client.get(f"/api/jobs/{job_id}/history").json()
    assert [e["action"] for e in hist["entries"]] == ["original", "edit_timing", "edit_timing"]

    undo1 = client.post(f"/api/jobs/{job_id}/history/undo").json()
    assert undo1["segments"][0]["start_time"] == 0.7
    undo2 = client.post(f"/api/jobs/{job_id}/history/undo").json()
    assert undo2["segments"][0]["start_time"] == 0.5
