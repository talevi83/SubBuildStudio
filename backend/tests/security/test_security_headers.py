"""Anti-framing and related headers on every response (app/core/security_headers.py)."""
from fastapi.testclient import TestClient

from app.core.security_headers import SECURITY_HEADERS
from app.main import app


def _assert_headers(res):
    for name, value in SECURITY_HEADERS.items():
        assert res.headers.get(name) == value, name
    assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]


def test_api_responses_carry_security_headers():
    with TestClient(app) as client:
        _assert_headers(client.get("/api/jobs"))


def test_refusals_carry_security_headers_too():
    with TestClient(app, headers={"X-SubBuild-Token": ""}) as anonymous:
        _assert_headers(anonymous.get("/api/jobs"))          # 401 from the session guard
        _assert_headers(anonymous.get("/"))                  # the "how to open the studio" page
    with TestClient(app, headers={"Origin": "https://evil.example"}) as cross_site:
        _assert_headers(cross_site.post("/api/jobs/bulk-delete", json={"delete_all": True}))  # 403 CSRF
