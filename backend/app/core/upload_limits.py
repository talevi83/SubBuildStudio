"""Refuse oversized uploads before their body is read.

Starlette spools a multipart body to a temporary file *before* the route handler runs, so a
check inside the handler comes too late: one request could already have filled the disk.
This middleware looks at Content-Length first (browsers always send it for FormData uploads):

* /api/media/parse-subtitles: subtitle files are small; the parser accepts up to 20 MB.
* /api/media/upload: the body must leave room on both the temp folder (where the body is
  spooled) and the uploads folder (where it is copied), plus a safety margin.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SUBTITLE_MAX_BYTES = 21 * 1024 * 1024        # 20 MB file + multipart overhead
FREE_SPACE_MARGIN = 1024 * 1024 * 1024        # keep 1 GB free after an upload


def _free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


class UploadLimitGuard:
    def __init__(self, app: ASGIApp, uploads_dir: Callable[[], Path]) -> None:
        self.app = app
        self.uploads_dir = uploads_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path not in ("/api/media/upload", "/api/media/parse-subtitles"):
            await self.app(scope, receive, send)
            return

        raw = Headers(scope=scope).get("content-length")
        try:
            length = int(raw) if raw is not None else None
        except ValueError:
            length = None
        if length is None or length < 0:
            await JSONResponse({"detail": "Content-Length is required for uploads"}, status_code=411)(scope, receive, send)
            return

        if path == "/api/media/parse-subtitles":
            if length > SUBTITLE_MAX_BYTES:
                await JSONResponse({"detail": "The subtitle file is too large (over 20 MB)."}, status_code=413)(scope, receive, send)
                return
        else:
            for folder in (Path(tempfile.gettempdir()), self.uploads_dir()):
                free = _free_bytes(folder)
                if free is not None and free < length + FREE_SPACE_MARGIN:
                    await JSONResponse(
                        {"detail": f"Not enough free disk space for this upload in {folder}"}, status_code=507,
                    )(scope, receive, send)
                    return
        await self.app(scope, receive, send)
