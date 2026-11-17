"""Per-launch session token for the local API.

The API can read any media file on this PC, spend the user's Gemini quota and write subtitle
files next to their videos, so being reachable on 127.0.0.1 must not be enough to use it: any
other program, any other Windows account on the same machine (loopback is shared) and, with
HOST=0.0.0.0, anyone on the network could otherwise drive it.

Every launch gets a random token (``SUBBUILD_TOKEN``, set by the desktop launcher / run scripts,
or generated here). The studio is opened once at ``/?token=<token>``; that request turns the
token into an HttpOnly cookie and redirects to ``/`` so the token leaves the address bar. From
then on the browser sends the cookie with every fetch, ``<video src>``, download and WebSocket,
so the frontend needs no changes. Tools (the launcher, tests, curl) can send the token in the
``X-SubBuild-Token`` header instead.

Protected: everything under /api/ and /ws/, the API docs, and the studio page itself (without a
valid cookie "/" explains how to open the studio instead of loading a page whose calls would all
fail). Static assets (the built JS/CSS) hold no data and stay public.
"""
import hmac
import os
import secrets
from http.cookies import CookieError, SimpleCookie
from typing import Optional

from starlette.datastructures import Headers, QueryParams
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

COOKIE_NAME = "subbuild_session"
HEADER_NAME = "x-subbuild-token"

_env_token = (os.environ.get("SUBBUILD_TOKEN") or "").strip()
# True when nobody handed us a token: main.py then prints the link to open the studio
TOKEN_GENERATED = not _env_token
SESSION_TOKEN = _env_token or secrets.token_urlsafe(32)

_PROTECTED_PREFIXES = ("/api/", "/ws/", "/docs", "/redoc", "/openapi.json")
_PAGE_PATHS = {"/", "/index.html"}
# "/api/session" works through the Vite dev server's /api proxy as well
_BOOTSTRAP_PATHS = {"/", "/api/session"}

_SIGN_IN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SubBuild Studio</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0b0b10;color:#e4e4e7;font-family:Segoe UI,system-ui,sans-serif}
.box{max-width:520px;padding:24px;text-align:center}h1{font-size:18px}p{font-size:13px;color:#a1a1aa;line-height:1.6}</style>
</head><body><div class="box"><h1>SubBuild Studio</h1>
<p>This page needs the link the studio was started with. Open SubBuild Studio from its desktop
shortcut or tray icon, or use the link printed in the console window that started the server.</p>
<p dir="rtl">&#1497;&#1513; &#1500;&#1508;&#1514;&#1493;&#1495; &#1488;&#1514; SubBuild Studio
&#1502;&#1492;&#1511;&#1497;&#1510;&#1493;&#1512; &#1488;&#1493; &#1502;&#1488;&#1497;&#1497;&#1511;&#1493;&#1503;
&#1492;&#1502;&#1490;&#1513;, &#1488;&#1493; &#1489;&#1511;&#1497;&#1513;&#1493;&#1512; &#1513;&#1492;&#1493;&#1491;&#1508;&#1505;
&#1489;&#1495;&#1500;&#1493;&#1503; &#1492;&#1508;&#1511;&#1493;&#1491;&#1493;&#1514;.</p></div></body></html>"""


def is_valid_token(value: Optional[str]) -> bool:
    return bool(value) and hmac.compare_digest(str(value).encode(), SESSION_TOKEN.encode())


def bootstrap_path(token: Optional[str] = None) -> str:
    """The path that opens the studio and signs this browser in: "/?token=..."."""
    return f"/?token={token or SESSION_TOKEN}"


def _cookie_token(headers: Headers) -> Optional[str]:
    raw = headers.get("cookie")
    if not raw:
        return None
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except CookieError:
        return None
    morsel = jar.get(COOKIE_NAME)
    return morsel.value if morsel is not None else None


class SessionTokenGuard:
    """Pure ASGI middleware: no valid session token, no API."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or "/"
        method = scope.get("method", "GET")
        headers = Headers(scope=scope)

        if scope_type == "http" and method == "GET" and path in _BOOTSTRAP_PATHS:
            query = QueryParams(scope.get("query_string", b"").decode("latin-1"))
            if is_valid_token(query.get("token")):
                response = RedirectResponse("/", status_code=303)
                # Lax (not Strict): the first navigation comes from the launcher / a typed link,
                # which some browsers treat as cross-site. Cross-site POSTs and WebSockets are
                # still refused by CrossSiteRequestGuard.
                response.set_cookie(COOKIE_NAME, SESSION_TOKEN, httponly=True, samesite="lax", path="/")
                response.headers["Cache-Control"] = "no-store"
                await response(scope, receive, send)
                return

        protected = path.startswith(_PROTECTED_PREFIXES)
        page = scope_type == "http" and path in _PAGE_PATHS
        if not (protected or page):
            await self.app(scope, receive, send)  # static assets
            return
        if scope_type == "http" and method == "OPTIONS":
            await self.app(scope, receive, send)  # CORS preflights carry no credentials
            return

        if is_valid_token(_cookie_token(headers)) or is_valid_token(headers.get(HEADER_NAME)):
            await self.app(scope, receive, send)
            return

        if scope_type == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
        elif page:
            await HTMLResponse(_SIGN_IN_PAGE, status_code=401, headers={"Cache-Control": "no-store"})(scope, receive, send)
        else:
            await JSONResponse({"detail": "Not signed in to this SubBuild Studio session"}, status_code=401)(scope, receive, send)
