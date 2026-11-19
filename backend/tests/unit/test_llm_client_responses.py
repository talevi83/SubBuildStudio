"""Provider answers that aren't JSON give a readable error instead of a bare JSONDecodeError."""
import httpx
import pytest

import app.providers.llm.client as client


def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(client.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_an_html_page_answering_200_is_reported_clearly(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(200, text="<html>Sign in to the Wi-Fi</html>",
                                                             headers={"content-type": "text/html"}))
    with pytest.raises(RuntimeError, match=r"OpenAI returned a response that is not JSON .*text/html"):
        await client._post_json_with_retry("https://api.example/v1", {}, {}, timeout=5, provider_label="OpenAI")


@pytest.mark.asyncio
async def test_json_that_is_not_an_object_is_refused(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(200, json=[1, 2]))
    with pytest.raises(RuntimeError, match="unexpected JSON"):
        await client._post_json_with_retry("https://api.example/v1", {}, {}, timeout=5, provider_label="Ollama")


@pytest.mark.asyncio
async def test_a_normal_answer_is_returned(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(200, json={"ok": True}))
    assert await client._post_json_with_retry("https://api.example/v1", {}, {}, timeout=5,
                                              provider_label="OpenAI") == {"ok": True}
