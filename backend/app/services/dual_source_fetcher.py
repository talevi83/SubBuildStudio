"""Media identification, OpenSubtitles 64-bit hashing, and EN/AR subtitle fetching."""
import io
import logging
import os
import re
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

OPENSUBTITLES_API_URL = "https://api.opensubtitles.com/api/v1"
SUBDL_API_URL = "https://api.subdl.com/api/v1/subtitles"
SUBDL_DOWNLOAD_BASE = "https://dl.subdl.com"
_HASH_CHUNK_BYTES = 65536
_USER_AGENT = "SubBuildStudio v1.5"

_EPISODE_RE = re.compile(r"\bS(?P<season>\d{1,2})[\s._-]*E(?P<episode>\d{1,3})\b", re.IGNORECASE)
_ALT_EPISODE_RE = re.compile(r"\b(?P<season>\d{1,2})x(?P<episode>\d{1,3})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?:^|[.\s(_-])((?:19|20)\d{2})(?:$|[.\s)_-])")
_RELEASE_STOP_RE = re.compile(
    r"\b(?:2160p|1080p|720p|480p|4k|uhd|hdr|dv|web[-.\s]?dl|webrip|bluray|bdrip|brrip|"
    r"h\.?264|h\.?265|x264|x265|hevc|avc|ddp?\d(?:\.\d)?|atmos|aac|ac3|ma|dts|remux|proper|repack)\b",
    re.IGNORECASE,
)


def compute_opensubtitles_hash(video_path: Path) -> Tuple[str, int]:
    """Compute the standard OpenSubtitles 64-bit movie hash and file size in bytes."""
    path = Path(video_path)
    file_size = path.stat().st_size
    if file_size < _HASH_CHUNK_BYTES * 2:
        raise ValueError(f"Video file is too small ({file_size} bytes) for 64-bit OpenSubtitles hash")

    total = file_size
    with path.open("rb") as fh:
        for chunk in (fh.read(_HASH_CHUNK_BYTES),):
            for (val,) in struct.iter_unpack("<Q", chunk):
                total = (total + val) & 0xFFFFFFFFFFFFFFFF
        fh.seek(max(0, file_size - _HASH_CHUNK_BYTES), os.SEEK_SET)
        tail = fh.read(_HASH_CHUNK_BYTES)
        for (val,) in struct.iter_unpack("<Q", tail):
            total = (total + val) & 0xFFFFFFFFFFFFFFFF
    return f"{total:016x}", file_size


def parse_media_filename(file_name: str) -> Dict[str, Any]:
    """Extract media metadata (title, year, season, episode, type) from a filename."""
    stem = Path(file_name).stem
    release_tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", stem)
        if len(tok) >= 2
    }

    try:
        import guessit  # type: ignore

        guess = guessit.guessit(file_name)
        title = str(guess.get("title") or "").strip()
        if title:
            return {
                "title": title,
                "year": guess.get("year"),
                "season": guess.get("season"),
                "episode": guess.get("episode"),
                "type": "episode" if guess.get("type") == "episode" else "movie",
                "release_tokens": sorted(release_tokens),
            }
    except Exception:
        pass

    ep_match = _EPISODE_RE.search(stem) or _ALT_EPISODE_RE.search(stem)
    season = int(ep_match.group("season")) if ep_match else None
    episode = int(ep_match.group("episode")) if ep_match else None

    cutoff = len(stem)
    if ep_match:
        cutoff = min(cutoff, ep_match.start())

    year: Optional[int] = None
    year_match = _YEAR_RE.search(stem[:cutoff] if ep_match else stem)
    if year_match:
        year = int(year_match.group(1))
        cutoff = min(cutoff, year_match.start())

    stop_match = _RELEASE_STOP_RE.search(stem[:cutoff])
    if stop_match:
        cutoff = min(cutoff, stop_match.start())

    raw_title = re.sub(r"[._]+", " ", stem[:cutoff]).strip(" -_()[].")
    if not raw_title:
        raw_title = re.sub(r"[._]+", " ", stem).strip()

    return {
        "title": raw_title,
        "year": year,
        "season": season,
        "episode": episode,
        "type": "episode" if season is not None and episode is not None else "movie",
        "release_tokens": sorted(release_tokens),
    }


def _score_opensubtitles_candidate(item: Dict[str, Any], release_tokens: Set[str]) -> Tuple[int, int, int, int]:
    """Score an OpenSubtitles result: moviehash > WEB-DL/release match > non-HI > download count."""
    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else item
    hash_match = 1 if attrs.get("moviehash_match") else 0
    release_text = str(attrs.get("release") or "").lower()
    release_words = set(re.findall(r"[a-z0-9]+", release_text))
    overlap = len(release_tokens & release_words)
    web_bonus = 2 if ("web" in release_text or attrs.get("from_trusted")) else 0
    non_hi = 0 if attrs.get("hearing_impaired") else 1
    downloads = int(attrs.get("download_count") or 0)
    return (hash_match, web_bonus + overlap, non_hi, downloads)


def _extract_srt_from_bytes(raw: bytes, source_url: str = "") -> Optional[str]:
    """Decode raw subtitle payload (plain SRT bytes or a ZIP archive containing an .srt)."""
    if not raw:
        return None
    if source_url.lower().endswith(".zip") or raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                srt_names = [n for n in zf.namelist() if n.lower().endswith(".srt") and not n.startswith("__MACOSX")]
                if srt_names:
                    raw = zf.read(srt_names[0])
                else:
                    return None
        except Exception as exc:
            logger.warning(f"Failed to unpack zipped subtitle from {source_url}: {exc}")
            return None
    from app.utils.subtitle_parser import decode_subtitle_bytes

    text, _ = decode_subtitle_bytes(raw)
    return text.strip() or None


def fetch_opensubtitles_srt(
    video_path: Path,
    file_name: str,
    language: str,
    api_key: str,
    *,
    timeout: float = 20.0,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Search and download the best matching SRT subtitle for `language` from OpenSubtitles REST API."""
    clean_key = (api_key or "").strip()
    if not clean_key:
        return None

    parsed = parse_media_filename(file_name)
    params: Dict[str, Any] = {
        "languages": language.strip().lower(),
        "query": parsed["title"],
    }
    if parsed.get("year") and parsed.get("type") == "movie":
        params["year"] = parsed["year"]
    if parsed.get("season") is not None:
        params["season_number"] = parsed["season"]
    if parsed.get("episode") is not None:
        params["episode_number"] = parsed["episode"]

    try:
        movie_hash, _ = compute_opensubtitles_hash(video_path)
        params["moviehash"] = movie_hash
    except Exception:
        movie_hash = ""

    headers = {
        "Api-Key": clean_key,
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(f"{OPENSUBTITLES_API_URL}/subtitles", params=params, headers=headers)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
        if not data:
            return None

        release_tokens = set(parsed.get("release_tokens") or ())
        ranked = sorted(data, key=lambda item: _score_opensubtitles_candidate(item, release_tokens), reverse=True)
        for cand in ranked[:3]:
            attrs = cand.get("attributes") if isinstance(cand.get("attributes"), dict) else cand
            files = attrs.get("files") or []
            if not files:
                continue
            file_id = files[0].get("file_id")
            if not file_id:
                continue
            dl_resp = client.post(
                f"{OPENSUBTITLES_API_URL}/download",
                json={"file_id": file_id},
                headers=headers,
            )
            dl_resp.raise_for_status()
            link = (dl_resp.json() or {}).get("link")
            if not link:
                continue
            raw_resp = client.get(link)
            raw_resp.raise_for_status()
            text = _extract_srt_from_bytes(raw_resp.content, link)
            if text:
                return text, {
                    "provider": "opensubtitles",
                    "language": language,
                    "release": attrs.get("release"),
                    "moviehash_match": bool(attrs.get("moviehash_match")),
                    "download_count": attrs.get("download_count"),
                }
    return None


def fetch_subdl_srt(
    file_name: str,
    language: str,
    api_key: str,
    *,
    timeout: float = 20.0,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Search and download the best matching SRT subtitle for `language` from Subdl API."""
    clean_key = (api_key or "").strip()
    if not clean_key:
        return None

    parsed = parse_media_filename(file_name)
    params: Dict[str, Any] = {
        "api_key": clean_key,
        "film_name": parsed["title"],
        "languages": language.strip().upper(),
        "subs_per_page": 10,
    }
    if parsed.get("year"):
        params["year"] = parsed["year"]
    if parsed.get("season") is not None:
        params["season_number"] = parsed["season"]
        params["type"] = "tv"
    if parsed.get("episode") is not None:
        params["episode_number"] = parsed["episode"]

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(SUBDL_API_URL, params=params)
        resp.raise_for_status()
        subtitles = (resp.json() or {}).get("subtitles") or []
        for item in subtitles[:3]:
            rel_url = str(item.get("url") or "").strip()
            if not rel_url:
                continue
            dl_url = rel_url if rel_url.startswith("http") else f"{SUBDL_DOWNLOAD_BASE}{rel_url}"
            raw_resp = client.get(dl_url)
            raw_resp.raise_for_status()
            text = _extract_srt_from_bytes(raw_resp.content, dl_url)
            if text:
                return text, {
                    "provider": "subdl",
                    "language": language,
                    "release": item.get("release_name"),
                }
    return None


def fetch_dual_subtitles(
    video_path: Path,
    file_name: str,
    *,
    opensubtitles_api_key: Optional[str] = None,
    subdl_api_key: Optional[str] = None,
    languages: Sequence[str] = ("en", "ar"),
) -> Dict[str, Any]:
    """Fetch paired subtitles via OpenSubtitles with Subdl fallback for requested `languages`."""
    parsed = parse_media_filename(file_name)
    try:
        movie_hash, _ = compute_opensubtitles_hash(video_path)
    except Exception:
        movie_hash = None

    result: Dict[str, Any] = {
        "en_srt": None,
        "ar_srt": None,
        "en_meta": None,
        "ar_meta": None,
        "movie_hash": movie_hash,
        "parsed": parsed,
    }

    for lang in languages:
        fetched: Optional[Tuple[str, Dict[str, Any]]] = None
        if opensubtitles_api_key and opensubtitles_api_key.strip():
            try:
                fetched = fetch_opensubtitles_srt(video_path, file_name, lang, opensubtitles_api_key)
            except Exception as exc:
                logger.warning(f"OpenSubtitles fetch failed for '{file_name}' ({lang}): {exc}")
        if fetched is None and subdl_api_key and subdl_api_key.strip():
            try:
                fetched = fetch_subdl_srt(file_name, lang, subdl_api_key)
            except Exception as exc:
                logger.warning(f"Subdl fetch failed for '{file_name}' ({lang}): {exc}")
        if fetched is not None:
            result[f"{lang}_srt"] = fetched[0]
            result[f"{lang}_meta"] = fetched[1]

    return result
