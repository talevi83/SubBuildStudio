"""Media upload deduplication, hash cache, byte-range parsing, and upload cleanup."""
import hashlib
import io
import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

import app.api.routes_media as media
from app.api.routes_media import _display_name, _parse_range
from app.core.config import settings
from app.models.schemas import ProbeResponse
from tests.fixtures.builders import wait_for_job


def test_range_parsing():
    assert _parse_range("bytes=0-99", 1000) == (0, 99)
    assert _parse_range("bytes=900-", 1000) == (900, 999)
    assert _parse_range("bytes=900-5000", 1000) == (900, 999)  # clamped, not 416
    assert _parse_range("bytes=-100", 1000) == (900, 999)
    assert _parse_range("bytes=1000-", 1000) is None
    assert _parse_range("garbage", 1000) is None


def test_upload_name_and_dedupe(client: TestClient, monkeypatch):
    async def fake_probe(path):
        p = Path(path)
        return ProbeResponse(
            file_path=str(p.resolve()),
            file_name=p.name,
            file_size=p.stat().st_size,
            duration=1.0,
            format_name="matroska",
        )

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
    assert (
        client.get(
            "/api/media/upload/existing",
            params={"name": "My Show [1080p].mkv", "size": 1234, "sha256": hash_b},
        ).json()
        is None
    )
    assert (
        client.get(
            "/api/media/upload/existing",
            params={"name": "My Show [1080p].mkv", "size": 1234},
        ).json()
        is None
    )

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
    async def fake_probe(path):
        p = Path(path)
        return ProbeResponse(
            file_path=str(p.resolve()),
            file_name=p.name,
            file_size=p.stat().st_size,
            duration=1.0,
            format_name="matroska",
        )

    monkeypatch.setattr("app.api.routes_media.probe_media_file", fake_probe)
    name = f"cand_{uuid.uuid4().hex[:6]}.mkv"
    data = b"c" * 2048

    def count(size: int) -> int:
        return client.get("/api/media/upload/candidates", params={"name": name, "size": size}).json()["count"]

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


def test_delete_job_removes_unused_upload(client: TestClient):
    upload = settings.storage_dir / "uploads" / "deadbeef_clip.mkv"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"123")
    res = client.post("/api/jobs", json={"video_path": str(upload), "enable_translation": False})
    job_id = res.json()["id"]
    assert res.json()["is_upload"] is True
    finished = wait_for_job(client, job_id)  # probe fails on a fake file -> FAILED
    assert finished["status"] in ("FAILED", "CANCELLED", "COMPLETED")
    assert client.delete(f"/api/jobs/{job_id}").json()["upload_deleted"] is True
    assert not upload.exists()
