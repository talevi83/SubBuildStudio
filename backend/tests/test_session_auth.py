"""The per-launch session token (app/core/auth.py) and network-path refusal."""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.auth import COOKIE_NAME, SESSION_TOKEN, is_valid_token
from app.main import app

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
