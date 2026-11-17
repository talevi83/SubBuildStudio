"""The per-launch session token (app/core/auth.py) and network-path refusal."""
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from starlette.websockets import WebSocketDisconnect

from app.core import config as cfg
from app.core.auth import COOKIE_NAME, SESSION_TOKEN, is_valid_token
from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus

NO_TOKEN = {"X-SubBuild-Token": ""}


def test_api_refuses_requests_without_the_token():
    with TestClient(app, headers=NO_TOKEN) as anonymous:
        assert anonymous.get("/api/jobs").status_code == 401
        assert anonymous.get("/api/system/info").status_code == 401
        assert anonymous.post("/api/media/probe-path", json={"file_path": "C:/x.mp4"}).status_code == 401
        assert anonymous.put("/api/settings/api-key", json={"api_key": "stolen"}).status_code == 401
        assert anonymous.post("/api/settings/ollama/setup").status_code == 401
        assert anonymous.get("/openapi.json").status_code == 401
        wrong = anonymous.get("/api/jobs", headers={"X-SubBuild-Token": "guess"})
        assert wrong.status_code == 401


def test_header_token_is_accepted():
    with TestClient(app) as client:  # conftest adds the test token
        assert client.get("/api/jobs").status_code == 200


def test_bootstrap_link_sets_an_httponly_cookie_and_cleans_the_url():
    with TestClient(app, headers=NO_TOKEN) as browser:
        res = browser.get(f"/?token={SESSION_TOKEN}", follow_redirects=False)
        assert res.status_code == 303 and res.headers["location"] == "/"
        cookie = res.headers["set-cookie"]
        assert f"{COOKIE_NAME}={SESSION_TOKEN}" in cookie
        assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
        # The browser now sends the cookie: the API and the page work without the header
        assert browser.get("/api/jobs").status_code == 200
        assert browser.get("/").status_code != 401


def test_dev_bootstrap_through_the_api_proxy():
    with TestClient(app, headers=NO_TOKEN) as browser:
        res = browser.get(f"/api/session?token={SESSION_TOKEN}", follow_redirects=False)
        assert res.status_code == 303
        assert browser.get("/api/jobs").status_code == 200


def test_wrong_bootstrap_token_sets_nothing():
    with TestClient(app, headers=NO_TOKEN) as browser:
        res = browser.get("/?token=nope", follow_redirects=False)
        assert res.status_code == 401 and "set-cookie" not in res.headers
        assert "SubBuild Studio" in res.text  # the explanation page, not the studio
        assert browser.get("/api/jobs").status_code == 401


def test_websocket_needs_the_token():
    with TestClient(app, headers=NO_TOKEN) as anonymous:
        with pytest.raises(WebSocketDisconnect) as closed:
            with anonymous.websocket_connect("/ws/jobs/job_x"):
                pass
        assert closed.value.code == 1008
    with TestClient(app) as client:
        with client.websocket_connect("/ws/jobs/job_x") as ws:
            ws.send_text("ping")


def test_token_comparison():
    assert is_valid_token(SESSION_TOKEN)
    assert not is_valid_token("") and not is_valid_token(None) and not is_valid_token(SESSION_TOKEN + "x")


@pytest.mark.parametrize("path", [
    r"\\attacker\share\movie.mp4",
    "//attacker/share/movie.mp4",
    r"\\?\UNC\attacker\share\movie.mp4",
    r"\/attacker\share\movie.mp4",
])
def test_network_paths_are_refused_before_touching_them(path, monkeypatch):
    from pathlib import Path

    def boom(self):
        raise AssertionError("a network path must not be checked on disk")

    monkeypatch.setattr(Path, "is_file", boom)
    with TestClient(app) as client:
        res = client.post("/api/media/probe-path", json={"file_path": path})
        assert res.status_code == 400 and "Network paths" in res.json()["detail"]
        res = client.post("/api/jobs", json={"video_path": path})
        assert res.status_code == 400 and "Network paths" in res.json()["detail"]


def test_long_local_paths_are_still_allowed():
    from app.api.routes_media import reject_network_path

    reject_network_path(r"\\?\C:\Videos\movie.mp4")
    reject_network_path(r"C:\Videos\movie.mp4")
    reject_network_path("/home/user/movie.mp4")


def test_non_media_path_is_refused_without_revealing_whether_it_exists(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    with TestClient(app) as client:
        existing = client.post("/api/media/probe-path", json={"file_path": str(secret)})
        missing = client.post("/api/media/probe-path", json={"file_path": str(tmp_path / "nope.txt")})
    assert existing.status_code == missing.status_code == 400
    assert "Unsupported" in existing.json()["detail"] and "Unsupported" in missing.json()["detail"]


def test_security_media_extension_api_key_and_ollama_validation(client: TestClient, tmp_path, monkeypatch):
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
