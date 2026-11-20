"""Friendly, actionable messages for network-level model-discovery failures."""
import asyncio

import httpx
import pytest

from app.core.model_discovery import describe_discovery_error

pytestmark = pytest.mark.unit


def _wrapped(inner: BaseException, outer: BaseException) -> BaseException:
    """Mimic httpx: the OSError is the __cause__ of a ConnectError."""
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except BaseException as final:
        return final


def _winerror(code: int, msg: str) -> OSError:
    err = OSError(msg)
    err.winerror = code  # set explicitly so the test also runs off Windows
    return err


def test_blocked_socket_10013_is_explained():
    inner = _winerror(10013, "An attempt was made to access a socket in a way forbidden by its access permissions")
    exc = _wrapped(inner, httpx.ConnectError("[WinError 10013] An attempt was made to access a socket"))
    msg = describe_discovery_error("google", exc)
    assert "Google Gemini" in msg
    assert "firewall" in msg.lower()
    assert "10013" in msg


def test_blocked_detected_from_message_text_only():
    msg = describe_discovery_error("openai", httpx.ConnectError("[WinError 10013] forbidden"))
    assert "firewall" in msg.lower()
    assert "OpenAI" in msg


def test_connection_failure_is_explained():
    exc = _wrapped(_winerror(11001, "getaddrinfo failed"), httpx.ConnectError("getaddrinfo failed"))
    msg = describe_discovery_error("anthropic", exc)
    assert msg.startswith("Could not connect to Anthropic")
    assert "internet connection" in msg


def test_timeout_is_explained():
    msg = describe_discovery_error("google", asyncio.TimeoutError())
    assert "did not respond in time" in msg


def test_non_network_errors_keep_original_text():
    msg = describe_discovery_error("openai", RuntimeError("OpenAI HTTP 401: invalid api key"))
    assert msg == "OpenAI HTTP 401: invalid api key"
