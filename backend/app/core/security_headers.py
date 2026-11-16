"""Security headers on every HTTP response.

The studio's own page must never be shown inside another website's frame: a page the user visits
while the studio runs in a browser could otherwise lay an invisible iframe of
http://localhost:8000 under its own buttons and turn the user's clicks into "Clear all" or
"Install Ollama" (clickjacking). Requests from inside such a frame come from the studio's own
origin, so the CSRF guard cannot tell them apart; only the browser can refuse the frame.

The Content-Security-Policy is deliberately narrow (framing, <base>, plugins, form targets): a
script/style policy would also have to allow the Google Fonts stylesheet and whatever pywebview
injects into the desktop window, and needs testing in WebView2 before it is switched on.

Cross-Origin-Resource-Policy keeps other websites from even embedding the studio's responses
(a <video src> pointing at /api/media/stream/... from another site is refused by the browser).
"""
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware (works with streaming and file responses)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    if name not in headers:
                        headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)
