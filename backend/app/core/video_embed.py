"""Put a job's subtitles into a copy of its video.

Two ways:
  soft  - add a subtitle track (MKV: SubRip, MP4/MOV: mov_text). Every other stream is copied
          untouched, so it takes about as long as copying the file and quality is unchanged.
          Players can switch the track on and off.
  burn  - draw the subtitles onto the picture (libass) and re-encode to an H.264 MP4 that plays
          anywhere (TVs, phones, WhatsApp). Slow; tries NVENC first when ffmpeg has it and falls
          back to libx264.

The work runs as a background asyncio task with ffmpeg's `-progress` output turned into a percent.
The result is written to a hidden temp name first and renamed at the end, so a cancelled or failed
run never leaves a half-written video behind (or replaces a good one).
"""
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.pipeline.stages.audio_stage import get_ffmpeg_bin
from app.pipeline.stages.probe_stage import TEXT_SUBTITLE_CODECS, get_ffprobe_bin
from app.core.ffmpeg_safety import safe_input, safe_probe_input
from app.utils.filenames import safe_basename

logger = logging.getLogger(__name__)

MP4_FAMILY = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov"}

# Font size on libass's 288-line canvas for SRT input (scales with the video's height)
BURN_FONT_SIZES = {"small": 16, "medium": 19, "large": 23}
MIN_BURN_FONT_PX = 12
MAX_BURN_FONT_PX = 36

ALLOWED_BURN_FONTS: Dict[str, str] = {
    name.lower(): name
    for name in (
        "Heebo",
        "Rubik",
        "Assistant",
        "Secular One",
        "Alef",
        "Varela Round",
        "Frank Ruhl Libre",
        "Inter",
        "Arial",
        "Courier New",
    )
}
_HEX_COLOR_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")

# x264 settings per quality choice: (preset, crf). NVENC: (preset, cq)
X264_QUALITY = {"standard": ("veryfast", 20), "high": ("slow", 18)}
NVENC_QUALITY = {"standard": ("p4", 23), "high": ("p6", 19)}

# Audio codecs that play in an MP4 on practically every device; anything else becomes AAC
PORTABLE_AUDIO = {"aac", "mp3"}


@dataclass
class EmbedOptions:
    mode: str = "soft"                 # soft | burn
    language3: Optional[str] = None    # ISO 639-2 tag for the new track ("heb")
    track_title: Optional[str] = None  # shown in the player's subtitle menu
    set_default: bool = True           # soft: make the new track the default one
    font_size: str = "medium"          # burn
    font_family: Optional[str] = None  # burn
    font_size_px: Optional[int] = None # burn: preview pixel size (12..36)
    text_color: Optional[str] = None   # burn: #RRGGBB
    outline_color: Optional[str] = None       # burn: #RRGGBB
    outline_width: Optional[float] = None     # burn: 0.0..6.0
    background_color: Optional[str] = None    # burn: #RRGGBB
    background_opacity: Optional[int] = None  # burn: 0..100
    margin_v: Optional[int] = None            # burn: 8..120
    bold: bool = False                        # burn
    color_speakers: bool = False              # burn: per-speaker colors in ASS
    highlight_mode: str = "none"              # burn: none | word | karaoke
    highlight_color: Optional[str] = None     # burn: #RRGGBB
    subtitle_format: str = "srt"              # srt | ass
    quality: str = "standard"          # burn
    audio_stream: int = 0              # burn: which audio track goes into the MP4 (relative index)


@dataclass
class EmbedTask:
    id: str
    job_id: str
    mode: str
    target: Path
    in_app_storage: bool               # True: the file is offered as a download, not saved next to the video
    state: str = "running"             # running | done | error | cancelled
    percent: float = 0.0
    error: Optional[str] = None
    encoder: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    runner: Optional[asyncio.Task] = field(default=None, repr=False)


_tasks: Dict[str, EmbedTask] = {}
_nvenc_available: Optional[bool] = None


class EmbedCancelled(Exception):
    pass


# ---------------------------------------------------------------- planning

def output_extension(video_path: str, mode: str) -> str:
    """Soft keeps MP4/MOV (mov_text track) and turns everything else into MKV; burn is always MP4."""
    if mode == "burn":
        return ".mp4"
    ext = Path(video_path).suffix.lower()
    return ext if ext in MP4_FAMILY else ".mkv"


def output_name(video_name: str, lang_code: Optional[str], mode: str, ext: str) -> str:
    """Movie.mkv -> Movie.he.subs.mkv (soft) / Movie.he.burned.mp4 (burn)."""
    stem = safe_basename(Path(video_name).stem, fallback="video")
    tag = "subs" if mode == "soft" else "burned"
    return f"{stem}.{lang_code}.{tag}{ext}" if lang_code else f"{stem}.{tag}{ext}"


def temp_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.subbuild-tmp")


async def probe_streams(video_path: str) -> dict:
    """ffprobe streams + duration: {"duration": float, "streams": [...]}."""
    proc = await asyncio.create_subprocess_exec(
        get_ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
        *safe_probe_input(str(Path(video_path).resolve())),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe could not read the video: {err.decode(errors='ignore')[-500:]}")
    data = json.loads(out.decode(errors="ignore") or "{}")
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration": duration, "streams": data.get("streams") or []}


def build_soft_command(video_path: str, subs_path: str, out_path: str, probe: dict, opts: EmbedOptions) -> List[str]:
    """Copy every stream and add the subtitle file as one more subtitle track."""
    ext = output_extension(video_path, "soft")
    is_mp4 = ext in MP4_FAMILY
    streams = probe.get("streams") or []
    cmd = [get_ffmpeg_bin(), "-y", "-nostdin", "-v", "error", "-nostats", "-progress", "pipe:1",
           *safe_input(str(Path(video_path).resolve())), *safe_input(str(subs_path))]

    if is_mp4:
        # MP4 holds only text subtitles as mov_text, and no unknown data tracks
        cmd += ["-map", "0:v?", "-map", "0:a?"]
        kept_subs = [s for s in streams if s.get("codec_type") == "subtitle"
                     and (s.get("codec_name") or "").lower() in TEXT_SUBTITLE_CODECS]
        for s in kept_subs:
            cmd += ["-map", f"0:{s['index']}"]
    else:
        # Matroska takes video/audio/subtitles/attachments (fonts); data tracks (e.g. from .ts) would fail
        cmd += ["-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-map", "0:t?"]
        kept_subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    cmd += ["-map", "1:0", "-c", "copy"]
    if is_mp4:
        cmd += ["-c:s", "mov_text"]

    new = len(kept_subs)  # index of the new track among the output's subtitle tracks
    if opts.language3:
        cmd += [f"-metadata:s:s:{new}", f"language={opts.language3}"]
    if opts.track_title:
        cmd += [f"-metadata:s:s:{new}", f"title={opts.track_title}"]
    if opts.set_default:
        cmd += ["-disposition:s", "0", f"-disposition:s:{new}", "default"]
    cmd += ["-max_muxing_queue_size", "4096"]
    if is_mp4:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-f", MP4_FAMILY.get(ext, "matroska"), str(out_path)]
    return cmd


_DARWIN_ARIAL_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)
_LINUX_FONT_DIRS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".local" / "share" / "fonts",
)
_LINUX_FONT_CANDIDATES = (
    ("arial.ttf", "Arial"),
    ("notosans-regular.ttf", "Noto Sans"),
    ("dejavusans.ttf", "DejaVu Sans"),
    ("liberationsans-regular.ttf", "Liberation Sans"),
)


def _burn_font_name(platform: Optional[str] = None) -> str:
    """Pick a widely supported subtitle burn-in font for the host platform."""
    plat = platform or sys.platform
    if plat == "win32":
        return "Arial"
    if plat == "darwin":
        for p in _DARWIN_ARIAL_PATHS:
            if p.is_file():
                return "Arial"
        return "Helvetica Neue"

    found_files: Set[str] = set()
    for root in _LINUX_FONT_DIRS:
        if not root.is_dir():
            continue
        try:
            for font_file in root.rglob("*.ttf"):
                found_files.add(font_file.name.lower())
        except OSError:
            continue
    for filename, family in _LINUX_FONT_CANDIDATES:
        if filename in found_files:
            return family
    return "DejaVu Sans"


def _resolve_burn_font(font_family: Optional[str], platform: Optional[str] = None) -> str:
    """Resolve an allowed subtitle font family or fall back to the host platform default."""
    if font_family:
        canonical = ALLOWED_BURN_FONTS.get(font_family.strip().lower())
        if canonical:
            return canonical
    return _burn_font_name(platform)


def hex_to_ass_color(hex_color: str, opacity_percent: int = 100) -> str:
    """Convert #RRGGBB and opacity (0..100%) to ASS &HAABBGGRR (AA: 00 opaque .. FF transparent)."""
    match = _HEX_COLOR_RE.match((hex_color or "").strip())
    raw = match.group(1) if match else "FFFFFF"
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    clamped_opacity = max(0, min(100, int(opacity_percent)))
    alpha = round((100 - clamped_opacity) * 255 / 100)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


_GOOGLE_FONT_SOURCES: Dict[str, tuple[str, str]] = {
    "heebo": ("Heebo.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/heebo/Heebo%5Bwght%5D.ttf"),
    "rubik": ("Rubik.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/rubik/Rubik%5Bwght%5D.ttf"),
    "assistant": ("Assistant.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/assistant/Assistant%5Bwght%5D.ttf"),
    "secular one": ("SecularOne-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/secularone/SecularOne-Regular.ttf"),
    "alef": ("Alef-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/alef/Alef-Regular.ttf"),
    "varela round": ("VarelaRound-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/varelaround/VarelaRound-Regular.ttf"),
    "frank ruhl libre": ("FrankRuhlLibre.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf"),
    "inter": ("Inter.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf"),
}


def _dropin_font_files() -> List[Path]:
    """Optional .ttf/.otf font files placed under app/assets/fonts, SUBBUILD_HOME/fonts, or storage/fonts."""
    from app.core.config import settings

    dirs = [
        Path(__file__).resolve().parents[1] / "assets" / "fonts",
        settings.storage_dir / "fonts",
    ]
    home = os.environ.get("SUBBUILD_HOME")
    if home:
        dirs.append(Path(home) / "fonts")
    fonts: List[Path] = []
    seen_names: Set[str] = set()
    for folder in dirs:
        if not folder.is_dir():
            continue
        try:
            for item in folder.iterdir():
                if item.is_file() and item.suffix.lower() in (".ttf", ".otf"):
                    key = item.name.lower()
                    if key not in seen_names:
                        seen_names.add(key)
                        fonts.append(item)
        except OSError:
            continue
    return fonts


def ensure_burn_font(font_family: Optional[str]) -> Optional[Path]:
    """Download an open-source Google Font TTF into storage/fonts on first burn-in if missing."""
    if not font_family or os.environ.get("SUBBUILD_OFFLINE") == "1":
        return None
    spec = _GOOGLE_FONT_SOURCES.get(font_family.strip().lower())
    if not spec:
        return None
    local_name, url = spec
    stem_norm = re.sub(r"[^a-z0-9]", "", font_family.lower())
    for existing in _dropin_font_files():
        ex_norm = re.sub(r"[^a-z0-9]", "", existing.stem.lower())
        if existing.name.lower() == local_name.lower() or (stem_norm and stem_norm in ex_norm):
            return existing

    from app.core.config import settings
    import httpx

    fonts_dir = settings.storage_dir / "fonts"
    target = fonts_dir / local_name
    tmp = fonts_dir / f".{local_name}.tmp"
    try:
        fonts_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200 or len(resp.content) < 4096:
                return None
            tmp.write_bytes(resp.content)
            os.replace(tmp, target)
        logger.info("Downloaded burn-in font %s to %s", font_family, target)
        return target
    except Exception as exc:
        logger.debug("Could not fetch optional font %s (%s); using host fallback", font_family, exc)
        tmp.unlink(missing_ok=True)
        return None


def burn_style(
    font_size: str = "medium",
    platform: Optional[str] = None,
    *,
    font_family: Optional[str] = None,
    font_size_px: Optional[int] = None,
    text_color: Optional[str] = None,
    outline_color: Optional[str] = None,
    outline_width: Optional[float] = None,
    background_color: Optional[str] = None,
    background_opacity: Optional[int] = None,
    margin_v: Optional[int] = None,
    bold: bool = False,
) -> str:
    if font_size_px is not None:
        clamped_px = max(MIN_BURN_FONT_PX, min(MAX_BURN_FONT_PX, int(font_size_px)))
        size = max(10, round(clamped_px * 19 / 18))
    else:
        size = BURN_FONT_SIZES.get(font_size, BURN_FONT_SIZES["medium"])
    font = _resolve_burn_font(font_family, platform)
    primary = hex_to_ass_color(text_color or "#FFFFFF", 100)
    mv = max(8, min(120, int(margin_v))) if margin_v is not None else 16
    bold_part = ",Bold=-1" if bold else ""
    if background_opacity is not None and background_opacity > 0:
        box_col = hex_to_ass_color(background_color or "#000000", background_opacity)
        return (
            f"FontName={font},FontSize={size}{bold_part},PrimaryColour={primary},"
            f"OutlineColour={box_col},BackColour={box_col},BorderStyle=3,Outline=3,Shadow=0,MarginV={mv}"
        )
    out_col = hex_to_ass_color(outline_color or "#000000", 100)
    out_w = max(0.0, min(6.0, float(outline_width))) if outline_width is not None else 1.6
    return (
        f"FontName={font},FontSize={size}{bold_part},PrimaryColour={primary},OutlineColour={out_col},"
        f"BackColour=&H80000000,BorderStyle=1,Outline={out_w},Shadow=0.6,MarginV={mv}"
    )


def build_burn_command(video_path: str, subs_name: str, out_path: str, probe: dict, opts: EmbedOptions,
                       encoder: str) -> List[str]:
    """Re-encode the first video stream with the subtitles drawn on it, plus one audio track.

    `subs_name` is a bare file name: ffmpeg runs with the subtitle file's folder as its working
    directory, which avoids the filter-graph escaping of Windows paths (C\\:/...).
    """
    streams = probe.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    a_idx = opts.audio_stream if 0 <= opts.audio_stream < len(audio) else 0

    fonts_arg = ":fontsdir=." if _dropin_font_files() else ""
    if subs_name.lower().endswith(".ass"):
        vf = f"subtitles={subs_name}{fonts_arg}:charenc=UTF-8,format=yuv420p"
    else:
        style = burn_style(
            opts.font_size,
            font_family=opts.font_family,
            font_size_px=opts.font_size_px,
            text_color=opts.text_color,
            outline_color=opts.outline_color,
            outline_width=opts.outline_width,
            background_color=opts.background_color,
            background_opacity=opts.background_opacity,
            margin_v=opts.margin_v,
            bold=opts.bold,
        )
        vf = f"subtitles={subs_name}{fonts_arg}:charenc=UTF-8:force_style='{style}',format=yuv420p"
    cmd = [get_ffmpeg_bin(), "-y", "-nostdin", "-v", "error", "-nostats", "-progress", "pipe:1",
           *safe_input(str(Path(video_path).resolve())), "-map", "0:v:0"]
    if audio:
        cmd += ["-map", f"0:a:{a_idx}"]
    cmd += ["-vf", vf]
    if encoder == "h264_nvenc":
        preset, cq = NVENC_QUALITY.get(opts.quality, NVENC_QUALITY["standard"])
        cmd += ["-c:v", "h264_nvenc", "-preset", preset, "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    else:
        preset, crf = X264_QUALITY.get(opts.quality, X264_QUALITY["standard"])
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    if audio:
        codec = (audio[a_idx].get("codec_name") or "").lower()
        if codec in PORTABLE_AUDIO:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    cmd += ["-sn", "-dn", "-map_metadata", "0", "-map_chapters", "0",
            "-max_muxing_queue_size", "4096", "-movflags", "+faststart", "-f", "mp4", str(out_path)]
    return cmd


async def nvenc_available() -> bool:
    """Whether this ffmpeg build lists h264_nvenc (it may still fail without an NVIDIA GPU)."""
    global _nvenc_available
    if _nvenc_available is None:
        try:
            proc = await asyncio.create_subprocess_exec(
                get_ffmpeg_bin(), "-hide_banner", "-encoders",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            _nvenc_available = b"h264_nvenc" in out
        except OSError:
            _nvenc_available = False
    return _nvenc_available


# ---------------------------------------------------------------- running

async def _run(task: EmbedTask, cmd: List[str], duration: float, cwd: Optional[str] = None,
               start: float = 0.0, span: float = 100.0) -> None:
    """Run ffmpeg, mapping its progress onto task.percent in [start, start+span]."""
    logger.info("Embed %s (%s): %s", task.id, task.mode, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    task.process = proc
    stderr_task = asyncio.ensure_future(proc.stderr.read())
    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            key, _, value = line.decode(errors="ignore").strip().partition("=")
            if key in ("out_time_us", "out_time_ms") and duration > 0:
                try:
                    done = int(value) / 1_000_000 / duration
                except ValueError:
                    continue
                task.percent = round(start + span * max(0.0, min(0.99, done)), 1)
        await proc.wait()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        task.process = None
    err = (await stderr_task).decode(errors="ignore").strip()
    if task.cancel_requested:
        raise EmbedCancelled()
    if proc.returncode != 0:
        raise RuntimeError(err[-1500:] or f"ffmpeg exited with code {proc.returncode}")


async def _execute(task: EmbedTask, video_path: str, subtitle_text: str, opts: EmbedOptions) -> None:
    tmp_out = temp_path(task.target)
    workdir = Path(tempfile.mkdtemp(prefix="subbuild_embed_"))
    try:
        sub_ext = "ass" if (opts.mode == "burn" and opts.subtitle_format == "ass") else "srt"
        subs = workdir / f"subs.{sub_ext}"
        subs.write_text(subtitle_text, encoding="utf-8")
        probe = await probe_streams(video_path)
        duration = probe["duration"]
        if task.cancel_requested:
            raise EmbedCancelled()

        if opts.mode == "soft":
            task.encoder = "copy"
            await _run(task, build_soft_command(video_path, str(subs), str(tmp_out), probe, opts), duration)
        else:
            await asyncio.to_thread(ensure_burn_font, opts.font_family)
            for font_file in _dropin_font_files():
                shutil.copy2(font_file, workdir / font_file.name)
            encoders = (["h264_nvenc"] if await nvenc_available() else []) + ["libx264"]
            for i, encoder in enumerate(encoders):
                task.encoder = encoder
                task.percent = 0.0
                try:
                    await _run(task, build_burn_command(video_path, subs.name, str(tmp_out), probe, opts, encoder),
                               duration, cwd=str(workdir))
                    break
                except RuntimeError as exc:
                    last = i == len(encoders) - 1
                    if last:
                        raise
                    logger.warning("Embed %s: %s failed, retrying with %s: %s",
                                   task.id, encoder, encoders[i + 1], str(exc)[-300:])
                    tmp_out.unlink(missing_ok=True)

        if task.cancel_requested:
            raise EmbedCancelled()
        os.replace(tmp_out, task.target)
        task.percent = 100.0
        task.state = "done"
        logger.info("Embed %s done: %s", task.id, task.target)
    except EmbedCancelled:
        task.state = "cancelled"
    except asyncio.CancelledError:
        task.state = "cancelled"
        raise
    except Exception as exc:  # ffmpeg errors, disk full, ...
        task.state = "error"
        task.error = str(exc) or exc.__class__.__name__
        logger.error("Embed %s failed: %s", task.id, task.error)
    finally:
        tmp_out.unlink(missing_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)
        task.finished_at = time.time()


# Finished task records are kept this long (the dialog polls them), then forgotten
_FINISHED_TTL_SECONDS = 6 * 3600


def _prune_finished(now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    for task_id, t in list(_tasks.items()):
        if t.state != "running" and t.finished_at is not None and now - t.finished_at > _FINISHED_TTL_SECONDS:
            _tasks.pop(task_id, None)


def start_embed(job_id: str, video_path: str, target: Path, subtitle_text: str, opts: EmbedOptions,
                in_app_storage: bool) -> EmbedTask:
    """Start the background run (needs a running event loop) and return its task record."""
    _prune_finished()
    task = EmbedTask(id=uuid.uuid4().hex[:12], job_id=job_id, mode=opts.mode, target=target,
                     in_app_storage=in_app_storage)
    _tasks[task.id] = task
    task.runner = asyncio.get_running_loop().create_task(_execute(task, video_path, subtitle_text, opts))
    return task


def get_task(task_id: str) -> Optional[EmbedTask]:
    return _tasks.get(task_id)


def latest_task(job_id: str) -> Optional[EmbedTask]:
    tasks = [t for t in _tasks.values() if t.job_id == job_id]
    return max(tasks, key=lambda t: t.started_at) if tasks else None


def running_task(job_id: str) -> Optional[EmbedTask]:
    return next((t for t in _tasks.values() if t.job_id == job_id and t.state == "running"), None)


def running_task_for_target(target: Path) -> Optional[EmbedTask]:
    """A running task (of any project) that writes this same file: both would use one temp file."""
    key = _path_key(target)
    return next((t for t in _tasks.values() if t.state == "running" and _path_key(t.target) == key), None)


def _path_key(path: Path) -> str:
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = str(path)
    return os.path.normcase(resolved)


def cancel_task(task: EmbedTask) -> None:
    if task.state != "running":
        return
    task.cancel_requested = True
    if task.process is not None and task.process.returncode is None:
        try:
            task.process.kill()
        except ProcessLookupError:
            pass


async def shutdown_embeds() -> None:
    """Stop every running ffmpeg on app shutdown so no orphan keeps writing."""
    running = [t for t in _tasks.values() if t.state == "running"]
    for t in running:
        cancel_task(t)
    runners = [t.runner for t in running if t.runner is not None]
    if runners:
        await asyncio.wait(runners, timeout=5)
