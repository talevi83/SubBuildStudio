"""Oversized uploads are refused from Content-Length, before the body is spooled to disk."""
import io

from fastapi.testclient import TestClient

import app.core.upload_limits as limits
from app.main import app


def test_oversized_subtitle_file_is_refused_up_front():
    big = b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n" * (limits.SUBTITLE_MAX_BYTES // 30 + 10)
    with TestClient(app) as client:
        res = client.post("/api/media/parse-subtitles", files={"file": ("big.srt", io.BytesIO(big))})
    assert res.status_code == 413


def test_upload_is_refused_when_it_would_fill_the_disk(monkeypatch):
    monkeypatch.setattr(limits, "_free_bytes", lambda path: 10 * 1024 * 1024)  # 10 MB left
    with TestClient(app) as client:
        res = client.post("/api/media/upload", files={"file": ("clip.mkv", io.BytesIO(b"x" * 4096))})
    assert res.status_code == 507 and "free disk space" in res.json()["detail"]


def test_upload_without_content_length_is_refused():
    def chunks():
        yield b"--x\r\n"

    with TestClient(app) as client:
        res = client.post("/api/media/upload", content=chunks(),
                          headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert res.status_code == 411


def test_normal_subtitle_file_still_parses():
    srt = "".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\nLine {i}\n\n" for i in range(1, 5)).encode()
    with TestClient(app) as client:
        res = client.post("/api/media/parse-subtitles", files={"file": ("ok.srt", io.BytesIO(srt))})
    assert res.status_code == 200 and res.json()["cue_count"] == 4
