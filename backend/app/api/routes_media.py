import asyncio
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from sqlmodel import Session

from app.core.database import get_session
from app.core.housekeeping import uploads_dir
from app.models.db_models import Job
from app.models.schemas import ParsedSubtitlesResponse, ProbeResponse, ProbePathRequest
from app.pipeline.stages.probe_stage import probe_media_file

router = APIRouter(prefix="/api/media", tags=["Media"])

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}_")
_HASH_CHUNK_BYTES = 4 * 1024 * 1024

# Browsers sniff the real container, so MKV/AVI are served as video/mp4 (Chrome plays
# H.264/AAC in Matroska this way). Only WebM needs its own type.
_MEDIA_TYPES = {".webm": "video/webm", ".ogv": "video/ogg"}


def _sanitize_filename(raw: Optional[str]) -> str:
    """Reduce a client-supplied filename to a safe basename (no directories, no reserved chars)."""
    # Normalize both separator styles before taking the basename
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    return name[:200] or "upload.bin"


def _display_name(stored_name: str) -> str:
    """Strip the random prefix we add to uploaded files ("1a2b3c4d_Movie.mkv" -> "Movie.mkv")."""
    return _UPLOAD_PREFIX.sub("", stored_name, count=1)


# Hashes of uploaded files, keyed by path and checked against size + mtime, so a candidate is
# read from disk once rather than on every drop of a file with the same name and size.
_hash_cache: Dict[str, Tuple[int, int, str]] = {}


def _file_sha256(path: Path) -> str:
    """Lowercase hex SHA-256 of a file on disk (cached while the file is unchanged)."""
    st = path.stat()
    key = str(path.resolve())
    cached = _hash_cache.get(key)
    if cached is not None and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
        return cached[2]
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    _hash_cache[key] = (st.st_size, st.st_mtime_ns, value)
    return value


def _remember_sha256(path: Path, sha256: str) -> None:
    """Record the hash computed while a file was being written, so it is never read again for it."""
    try:
        st = path.stat()
        _hash_cache[str(path.resolve())] = (st.st_size, st.st_mtime_ns, sha256)
    except OSError:
        pass


def _upload_candidates(name: str, size: int, exclude: Optional[Path] = None) -> List[Path]:
    """Earlier uploads with the same name and size, oldest first. Name and size only find
    candidates: different videos can share both, so reuse also needs the content hash."""
    matches = []
    # Plain iteration (not glob): names like "Movie [1080p].mkv" contain glob syntax
    for f in uploads_dir().iterdir():
        try:
            if exclude is not None and f.resolve() == exclude:
                continue
            if f.is_file() and _display_name(f.name) == name and f.stat().st_size == size:
                matches.append((f.stat().st_mtime, f))
        except OSError:
            continue
    return [f for _, f in sorted(matches, key=lambda item: item[0])]


def _find_existing_upload(
    name: str,
    size: int,
    sha256: Optional[str] = None,
    exclude: Optional[Path] = None,
) -> Optional[Path]:
    """The oldest earlier upload of the same file (same name, size, and SHA-256 content hash), if still on disk."""
    if not sha256:
        return None
    expected_hash = sha256.strip().lower()
    for candidate in _upload_candidates(name, size, exclude=exclude):
        try:
            if _file_sha256(candidate) == expected_hash:
                return candidate
        except OSError:
            continue
    return None


async def _probe_upload(path: Path) -> ProbeResponse:
    res = await probe_media_file(str(path))
    res.file_name = _display_name(path.name)
    res.is_upload = True
    return res


@router.post("/probe-path", response_model=ProbeResponse)
async def probe_local_path(req: ProbePathRequest):
    """Inspect and probe media streams of a file directly from the local filesystem."""
    file_path = req.file_path.strip().strip('"').strip("'")
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found on local disk: {file_path}")

    try:
        res = await probe_media_file(str(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to probe media file: {exc}")
    if path.resolve().parent == uploads_dir():
        res.file_name = _display_name(path.name)
        res.is_upload = True
    return res


@router.get("/upload/candidates")
async def count_upload_candidates(name: str = Query(...), size: int = Query(...)):
    """How many earlier uploads share this name and size. The browser hashes a dropped file
    (slow for big videos) only when there is something to compare it with."""
    found = await asyncio.to_thread(_upload_candidates, _sanitize_filename(name), size)
    return {"count": len(found)}


@router.get("/upload/existing", response_model=Optional[ProbeResponse])
async def find_existing_upload(
    name: str = Query(...),
    size: int = Query(...),
    sha256: Optional[str] = Query(None),
):
    """Return the probe of an identical earlier upload so the browser can skip re-uploading."""
    if not sha256:
        return None
    existing = await asyncio.to_thread(_find_existing_upload, _sanitize_filename(name), size, sha256)
    if existing is None:
        return None
    try:
        return await _probe_upload(existing)
    except Exception:
        return None


@router.post("/upload", response_model=ProbeResponse)
async def upload_media_file(file: UploadFile = File(...)):
    """Upload a video file from browser dropzone and probe its streams."""
    clean_name = _sanitize_filename(file.filename)
    safe_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"
    target_dir = uploads_dir()
    upload_dest = (target_dir / safe_name).resolve()
    if upload_dest.parent != target_dir:
        raise HTTPException(status_code=400, detail="Invalid file name")
    upload_dest.parent.mkdir(parents=True, exist_ok=True)

    def _save() -> str:
        digest = hashlib.sha256()
        with open(upload_dest, "wb") as buffer:
            while True:
                chunk = file.file.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                buffer.write(chunk)
        return digest.hexdigest()

    try:
        sha256 = await asyncio.to_thread(_save)
    except Exception as exc:
        upload_dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")

    # Same file uploaded before? Keep one copy only when the content hash also matches.
    size = upload_dest.stat().st_size
    existing = await asyncio.to_thread(_find_existing_upload, clean_name, size, sha256, exclude=upload_dest)
    if existing is not None:
        upload_dest.unlink(missing_ok=True)
        upload_dest = existing
    else:
        _remember_sha256(upload_dest, sha256)

    try:
        return await _probe_upload(upload_dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Probing uploaded file failed: {exc}")


@router.post("/parse-subtitles", response_model=ParsedSubtitlesResponse)
async def parse_subtitles(
    file: UploadFile = File(...),
    encoding: Optional[str] = Query(None, description="Optional explicit character encoding to decode the file with"),
):
    """Read a subtitle file (SRT / WebVTT / ASS) for "sync existing subtitles". Nothing is stored."""
    from app.utils.subtitle_parser import SubtitleParseError, parse_subtitle_file

    data = await file.read(20 * 1024 * 1024 + 1)
    try:
        return parse_subtitle_file(data, _sanitize_filename(file.filename), encoding=encoding)
    except SubtitleParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _stream_file_range(file_path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    """Generator to yield chunks for HTTP Range requests."""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def _parse_range(range_header: str, file_size: int) -> Optional[tuple]:
    """Parse a single byte range ("bytes=a-b", "bytes=a-", "bytes=-n"); None if unsatisfiable."""
    match = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*(?:,.*)?", range_header)
    if not match or (match.group(1) == "" and match.group(2) == ""):
        return None
    start_s, end_s = match.group(1), match.group(2)
    if start_s == "":
        # Suffix range: the last N bytes
        length = int(end_s)
        if length == 0:
            return None
        return max(0, file_size - length), file_size - 1
    start = int(start_s)
    end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)  # RFC 7233: clamp an end beyond the file
    if start >= file_size or start > end:
        return None
    return start, end


@router.get("/stream/{job_id}")
async def stream_video(
    job_id: str,
    range_header: Optional[str] = Header(None, alias="Range"),
    session: Session = Depends(get_session),
):
    """HTTP Range streaming endpoint for smooth seeking in the HTML5 video player."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    video_path = Path(job.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file missing on disk: {job.video_path}")

    file_size = os.path.getsize(video_path)
    media_type = _MEDIA_TYPES.get(video_path.suffix.lower(), "video/mp4")

    if not range_header:
        return FileResponse(str(video_path), media_type=media_type, headers={"Accept-Ranges": "bytes"})

    byte_range = _parse_range(range_header, file_size)
    if byte_range is None:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    start, end = byte_range

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": media_type,
    }
    return StreamingResponse(_stream_file_range(video_path, start, end), status_code=206, headers=headers)
