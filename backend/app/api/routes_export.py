import asyncio
import json
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from fastapi import APIRouter, HTTPException, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.core import edit_history as history
from app.core.config import settings
from app.core.database import get_session
from app.models.db_models import Job
from app.models.schemas import EmbedVideoRequest, EmbedVideoStatus, SaveNextToVideoRequest, SaveNextToVideoResponse
from app.utils.ass_formatter import AssStyleConfig, serialize_to_ass
from app.utils.filenames import safe_basename
from app.utils.languages import language_code, language_code3, language_name
from app.utils.srt_formatter import serialize_to_srt, serialize_to_vtt

router = APIRouter(prefix="/api/jobs", tags=["Export"])


def _lang_code(language: Optional[str]) -> Optional[str]:
    """Short language code for a language name/code (Movie.he.srt), or None if unknown/auto."""
    return language_code(language)


def _build_filename(job: Job, use_translated: bool, ext: str, bilingual: bool = False) -> str:
    code = _lang_code(job.target_language if use_translated else job.effective_source_language)
    if bilingual:
        suffix = f".{code}.dual.{ext}" if code else f".dual.{ext}"
    else:
        suffix = f".{code}.{ext}" if code else f".{ext}"
    return f"{safe_basename(Path(job.file_name).stem, fallback='subtitles')}{suffix}"


def _attachment_headers(filename: str) -> dict:
    """Content-Disposition that works for non-ASCII names (RFC 6266 / RFC 5987)."""
    ascii_fallback = filename.encode("ascii", "replace").decode("ascii").replace("?", "_").replace('"', "_")
    return {
        "Content-Disposition": f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(filename)}",
        "Cache-Control": "no-cache",
    }


def _job_speakers(job: Job) -> Optional[dict]:
    raw = getattr(job, "speakers_json", None)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _render(
    job: Job,
    use_translated: bool,
    apply_rtl: bool,
    max_chars_per_line: Optional[int],
    fmt: str,
    bilingual: bool = False,
    ass_style: Optional[AssStyleConfig] = None,
) -> bytes:
    """The subtitle file's bytes, exactly as downloaded (SRT with a UTF-8 BOM)."""
    onscreen_pos = getattr(job, "onscreen_position", "top_center") or "top_center"
    max_chars = max_chars_per_line or settings.max_chars_per_line
    if fmt == "ass":
        content = serialize_to_ass(
            job.segments,
            use_translated=use_translated,
            apply_rtl=apply_rtl,
            max_chars_per_line=max_chars,
            onscreen_position=onscreen_pos,
            bilingual=bilingual,
            style=ass_style,
            speakers=_job_speakers(job),
        )
        return ("\ufeff" + content).encode("utf-8")

    serializer = serialize_to_srt if fmt == "srt" else serialize_to_vtt
    content = serializer(
        job.segments,
        use_translated=use_translated,
        apply_rtl=apply_rtl,
        max_chars_per_line=max_chars,
        onscreen_position=onscreen_pos,
        bilingual=bilingual,
    )
    # UTF-8 BOM: some players (and Windows Notepad) otherwise mis-detect Hebrew as ANSI.
    body = ("\ufeff" + content) if fmt == "srt" else content
    return body.encode("utf-8")


def _export(
    job_id: str,
    session: Session,
    use_translated: bool,
    apply_rtl: bool,
    max_chars_per_line: Optional[int],
    fmt: str,
    bilingual: bool = False,
    ass_style: Optional[AssStyleConfig] = None,
) -> Response:
    with history.job_lock(job_id):
        session.expire_all()
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if fmt == "srt":
            media_type = "application/x-subrip; charset=utf-8"
        elif fmt == "ass":
            media_type = "text/x-ssa; charset=utf-8"
        else:
            media_type = "text/vtt; charset=utf-8"
        content = _render(
            job, use_translated, apply_rtl, max_chars_per_line, fmt,
            bilingual=bilingual, ass_style=ass_style,
        )
        headers = _attachment_headers(_build_filename(job, use_translated, fmt, bilingual=bilingual))
    return Response(
        content=content,
        media_type=media_type,
        headers=headers,
    )


@router.get("/{job_id}/export/srt")
def export_srt(
    job_id: str,
    use_translated: bool = Query(False, description="Export translated subtitle text if True"),
    apply_rtl: bool = Query(True, description="Apply BiDi RTL marks for Hebrew/Arabic"),
    max_chars_per_line: Optional[int] = Query(None, ge=10, le=200, description="Line length limit"),
    bilingual: bool = Query(False, description="Export both original and translated text in each cue"),
    session: Session = Depends(get_session),
):
    """Export and download subtitles as standard .srt format."""
    return _export(job_id, session, use_translated, apply_rtl, max_chars_per_line, "srt", bilingual=bilingual)


@router.get("/{job_id}/export/vtt")
def export_vtt(
    job_id: str,
    use_translated: bool = Query(False, description="Export translated subtitle text if True"),
    apply_rtl: bool = Query(True, description="Apply BiDi RTL marks for Hebrew/Arabic"),
    max_chars_per_line: Optional[int] = Query(None, ge=10, le=200, description="Line length limit"),
    bilingual: bool = Query(False, description="Export both original and translated text in each cue"),
    session: Session = Depends(get_session),
):
    """Export and download subtitles as WebVTT (.vtt) format."""
    return _export(job_id, session, use_translated, apply_rtl, max_chars_per_line, "vtt", bilingual=bilingual)


@router.get("/{job_id}/export/ass")
def export_ass(
    job_id: str,
    use_translated: bool = Query(False, description="Export translated subtitle text if True"),
    apply_rtl: bool = Query(True, description="Apply BiDi RTL marks for Hebrew/Arabic"),
    max_chars_per_line: Optional[int] = Query(None, ge=10, le=200, description="Line length limit"),
    bilingual: bool = Query(False, description="Export both original and translated text in each cue"),
    font_family: Optional[str] = Query(None, max_length=64),
    font_size_px: int = Query(18, ge=12, le=36),
    text_color: str = Query("#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$"),
    outline_color: str = Query("#000000", pattern=r"^#[0-9A-Fa-f]{6}$"),
    outline_width: float = Query(1.6, ge=0.0, le=6.0),
    background_color: str = Query("#000000", pattern=r"^#[0-9A-Fa-f]{6}$"),
    background_opacity: int = Query(0, ge=0, le=100),
    margin_v: int = Query(16, ge=8, le=120),
    bold: bool = Query(False),
    color_speakers: bool = Query(False),
    highlight_mode: str = Query("none", pattern=r"^(none|word|karaoke)$"),
    highlight_color: str = Query("#FACC15", pattern=r"^#[0-9A-Fa-f]{6}$"),
    session: Session = Depends(get_session),
):
    """Export and download styled subtitles in Advanced SubStation Alpha (.ass) format."""
    style = AssStyleConfig(
        font_family=font_family or "Heebo",
        font_size_px=font_size_px,
        text_color=text_color,
        outline_color=outline_color,
        outline_width=outline_width,
        background_color=background_color,
        background_opacity=background_opacity,
        margin_v=margin_v,
        bold=bold,
        color_speakers=color_speakers,
        highlight_mode=highlight_mode,
        highlight_color=highlight_color,
    )
    return _export(
        job_id, session, use_translated, apply_rtl, max_chars_per_line, "ass",
        bilingual=bilingual, ass_style=style,
    )


@router.post("/{job_id}/export/save-next-to-video", response_model=SaveNextToVideoResponse)
def save_next_to_video(job_id: str, req: SaveNextToVideoRequest, session: Session = Depends(get_session)):
    """Write the subtitles into the video's own folder, named for media servers (Movie.he.srt).

    Only for videos opened by path: an uploaded copy lives in the app's storage folder. An existing
    file is never replaced unless `overwrite` is set; the answer says it exists so the UI can ask.
    """
    import os

    with history.job_lock(job_id):
        session.expire_all()
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job.is_upload:
            raise HTTPException(
                status_code=400,
                detail="This video was uploaded through the browser, so it has no folder of its own. Download the file instead.",
            )
        if not any(seg.translated_text for seg in job.segments) and req.use_translated:
            raise HTTPException(status_code=400, detail="This project has no translation to save yet.")
        folder = Path(job.video_path).parent
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"The video's folder no longer exists: {folder}")

        target = folder / _build_filename(job, req.use_translated, req.format)
        exists = target.exists()
        if exists and not req.overwrite:
            return SaveNextToVideoResponse(saved=False, path=str(target), exists=True)

        data = _render(job, req.use_translated, req.apply_rtl, None, req.format)
    tmp = target.with_name(f".{target.name}.subbuild-tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)  # a half-written file never replaces a good one
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not write {target}: {exc}")
    return SaveNextToVideoResponse(saved=True, path=str(target), exists=exists)


# ---------------------------------------------------------------- subtitles inside the video

def _embed_status(job_id: str, task) -> EmbedVideoStatus:
    return EmbedVideoStatus(
        task_id=task.id,
        state=task.state,
        mode=task.mode,
        percent=task.percent,
        path=task.target.name if task.in_app_storage else str(task.target),
        file_name=task.target.name,
        next_to_video=not task.in_app_storage,
        download_url=(f"/api/jobs/{job_id}/export/embed/{task.id}/download"
                      if task.in_app_storage and task.state == "done" else None),
        error=task.error,
        encoder=task.encoder,
    )


@router.post("/{job_id}/export/embed", response_model=EmbedVideoStatus)
async def start_embed_video(job_id: str, req: EmbedVideoRequest, session: Session = Depends(get_session)):
    """Start making a copy of the video with the subtitles inside; poll the status for progress.

    Videos opened by path get the copy next to them (Movie.he.subs.mkv / Movie.he.burned.mp4); an
    existing file is never replaced unless `overwrite` is set. Uploaded videos have no folder of their
    own, so their copy goes to the app's exports folder and is offered as a download.
    """
    from app.core import video_embed as ve

    def prepare():
        # In a worker thread: job_lock is a threading.Lock (a long /retime may hold it), and
        # waiting for it on the event loop would freeze the whole server.
        with history.job_lock(job_id):
            session.expire_all()
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            if not any(seg.id for seg in job.segments):
                raise HTTPException(status_code=400, detail="This project has no subtitles yet.")
            if req.use_translated and not any(seg.translated_text for seg in job.segments):
                raise HTTPException(status_code=400, detail="This project has no translation to embed yet.")
            video = Path(job.video_path)
            if not video.is_file():
                raise HTTPException(status_code=404, detail=f"The video file no longer exists: {video}")
            if ve.running_task(job_id):
                raise HTTPException(status_code=409, detail="A video with subtitles is already being made for this project.")

            language = job.target_language if req.use_translated else job.effective_source_language
            code = _lang_code(language)
            ext = ve.output_extension(job.video_path, req.mode)
            name = ve.output_name(job.file_name or video.name, code, req.mode, ext)
            if job.is_upload:
                folder = settings.storage_dir.resolve() / "exports" / f"embed_{job.id}"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / name
            else:
                target = video.parent / name
                if target.exists() and not req.overwrite:
                    return EmbedVideoStatus(state="exists", mode=req.mode, path=str(target), file_name=name,
                                            next_to_video=True)

            if req.mode == "soft":
                # The copy is about as big as the original; fail now rather than at 95%
                import shutil
                try:
                    free = shutil.disk_usage(target.parent).free
                except OSError:
                    free = None
                if free is not None and free < video.stat().st_size * 1.02 + 50_000_000:
                    raise HTTPException(status_code=507, detail=f"Not enough free disk space in {target.parent}")

            if req.mode == "burn":
                tier_px = {"small": 15, "medium": 18, "large": 23}.get(req.font_size, 18)
                ass_style = AssStyleConfig(
                    font_family=req.font_family or ve._burn_font_name(),
                    font_size_px=req.font_size_px if req.font_size_px is not None else tier_px,
                    text_color=req.text_color or "#FFFFFF",
                    outline_color=req.outline_color or "#000000",
                    outline_width=req.outline_width if req.outline_width is not None else 1.6,
                    background_color=req.background_color or "#000000",
                    background_opacity=req.background_opacity if req.background_opacity is not None else 0,
                    margin_v=req.margin_v if req.margin_v is not None else 16,
                    bold=req.bold,
                    color_speakers=req.color_speakers,
                    highlight_mode=req.highlight_mode,
                    highlight_color=req.highlight_color or "#FACC15",
                )
                text = _render(
                    job, req.use_translated, req.apply_rtl, None, "ass",
                    bilingual=req.bilingual, ass_style=ass_style,
                ).decode("utf-8").lstrip("\ufeff")
                sub_fmt = "ass"
            else:
                text = _render(job, req.use_translated, req.apply_rtl, None, "srt").decode("utf-8").lstrip("\ufeff")
                sub_fmt = "srt"
            opts = ve.EmbedOptions(
                mode=req.mode,
                language3=language_code3(language),
                track_title=f"{language_name(language, 'Subtitles')} (SubBuild Studio)",
                set_default=req.set_default,
                font_size=req.font_size,
                font_family=req.font_family,
                font_size_px=req.font_size_px,
                text_color=req.text_color,
                outline_color=req.outline_color,
                outline_width=req.outline_width,
                background_color=req.background_color,
                background_opacity=req.background_opacity,
                margin_v=req.margin_v,
                bold=req.bold,
                color_speakers=req.color_speakers,
                highlight_mode=req.highlight_mode,
                highlight_color=req.highlight_color,
                subtitle_format=sub_fmt,
                quality=req.quality,
                audio_stream=job.selected_audio_stream or 0,
            )
            return video, target, text, opts, job.is_upload

    prepared = await asyncio.to_thread(prepare)
    if isinstance(prepared, EmbedVideoStatus):
        return prepared
    video, target, text, opts, job_is_upload = prepared
    # Checked again on the event loop (where tasks are registered), so two requests can't both start;
    # another project on the same video writing the same file is refused too.
    if ve.running_task(job_id):
        raise HTTPException(status_code=409, detail="A video with subtitles is already being made for this project.")
    if ve.running_task_for_target(target):
        raise HTTPException(status_code=409, detail=f"Another project is already writing {target.name}.")
    task = ve.start_embed(job_id, str(video.resolve()), target, text, opts, in_app_storage=job_is_upload)
    return _embed_status(job_id, task)


@router.get("/{job_id}/preview-frame")
async def get_preview_frame(
    job_id: str,
    t: Optional[float] = Query(None, ge=0.0, description="Timestamp in seconds"),
    session: Session = Depends(get_session),
):
    """Return a single JPEG video frame for live burn-in subtitle preview."""
    from app.pipeline import speaker_detect as sd
    from app.pipeline.stages.audio_stage import get_ffmpeg_bin

    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    video = Path(job.video_path)
    if not video.is_file():
        raise HTTPException(status_code=404, detail=f"The video file no longer exists: {video}")

    if t is not None:
        ts = float(t)
        if job.video_duration and job.video_duration > 0.2:
            ts = min(ts, job.video_duration - 0.1)
    else:
        ordered = sorted(job.segments, key=lambda s: (s.start_time, s.index))
        sample_seg = next(
            (
                s for s in ordered
                if (s.segment_type or "DIALOGUE") != "ON_SCREEN"
                and ((s.translated_text or s.original_text or "").strip())
            ),
            None,
        )
        if sample_seg is not None:
            ts = max(0.0, (float(sample_seg.start_time) + float(sample_seg.end_time)) / 2.0)
        else:
            dur = float(job.video_duration or 10.0)
            ts = min(5.0, max(0.0, dur / 2.0))

    jpeg = await asyncio.to_thread(sd.grab_frame, str(video.resolve()), ts, get_ffmpeg_bin())
    if not jpeg:
        raise HTTPException(status_code=502, detail="Could not extract preview frame from video")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=30"},
    )


@router.get("/{job_id}/export/embed", response_model=Optional[EmbedVideoStatus])
def latest_embed_video(job_id: str):
    """The project's most recent embed run (so a reopened dialog picks up a running one), or null."""
    from app.core import video_embed as ve

    task = ve.latest_task(job_id)
    return _embed_status(job_id, task) if task else None


def _get_embed_task(job_id: str, task_id: str):
    from app.core import video_embed as ve

    task = ve.get_task(task_id)
    if task is None or task.job_id != job_id:
        raise HTTPException(status_code=404, detail="Unknown embed task (the app may have restarted).")
    return task


@router.get("/{job_id}/export/embed/{task_id}", response_model=EmbedVideoStatus)
def embed_video_status(job_id: str, task_id: str):
    return _embed_status(job_id, _get_embed_task(job_id, task_id))


@router.post("/{job_id}/export/embed/{task_id}/cancel", response_model=EmbedVideoStatus)
def cancel_embed_video(job_id: str, task_id: str):
    from app.core import video_embed as ve

    task = _get_embed_task(job_id, task_id)
    ve.cancel_task(task)
    return _embed_status(job_id, task)


@router.get("/{job_id}/export/embed/{task_id}/download")
def download_embed_video(job_id: str, task_id: str):
    """The finished copy of an uploaded video (copies of videos opened by path are next to them)."""
    task = _get_embed_task(job_id, task_id)
    if not task.in_app_storage:
        raise HTTPException(status_code=400, detail=f"This video was saved next to the original: {task.target}")
    if task.state != "done" or not task.target.is_file():
        raise HTTPException(status_code=404, detail="The video is not ready.")
    media_type = "video/mp4" if task.target.suffix.lower() in (".mp4", ".m4v") else (
        "video/quicktime" if task.target.suffix.lower() == ".mov" else "video/x-matroska")
    return FileResponse(task.target, media_type=media_type, filename=task.target.name)


def _job_log_header(job: Job) -> str:
    """A readable summary of the job to put on top of its log."""
    import json
    from datetime import datetime

    from app.core.user_settings import load_settings

    lines = [
        "SubBuild Studio - job log",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Job:        {job.id}",
        f"File:       {job.file_name}  ({job.video_duration or 0:.1f}s)",
        f"Status:     {job.status.value}  (stage {job.current_stage}, {job.progress_percent:.1f}%)",
        f"Created:    {job.created_at}   Updated: {job.updated_at}",
        f"Whisper:    {job.whisper_model}, source {job.source_language} (detected {job.detected_language}), "
        f"VAD {job.vad_filter}, comms boost {job.comms_boost}, on-screen text {job.detect_onscreen_text}",
        f"Translation:{' on' if job.enable_translation else ' off'} -> {job.target_language}, model {job.gemini_model}",
    ]
    if job.warning_message:
        lines.append(f"Warning:    {job.warning_message}")
    if job.error_message:
        lines.append(f"Error:      {job.error_message}")

    counts: dict = {}
    for seg in job.segments:
        kind = seg.segment_type or "DIALOGUE"
        counts[kind] = counts.get(kind, 0) + 1
    untranslated = [seg for seg in job.segments if not seg.translated_text]
    failed = [seg for seg in job.segments if seg.translation_failed]
    lines.append(f"Subtitles:  {len(job.segments)} {counts}; untranslated {len(untranslated)}, "
                 f"flagged failed {len(failed)}")
    if job.enable_translation and untranslated:
        sample = ", ".join(f"#{s.index} {s.start_time:.1f}s" for s in sorted(untranslated, key=lambda s: s.start_time)[:20])
        lines.append(f"            untranslated: {sample}{' ...' if len(untranslated) > 20 else ''}")

    if job.usage_json:
        try:
            usage = json.loads(job.usage_json)
            lines.append(f"AI cost (all runs): {usage.get('total_cost_usd')}")
            for name, st in (usage.get("stages") or {}).items():
                lines.append(
                    f"  {name}: {st.get('model')}, {st.get('requests')} request(s), in {st.get('input_tokens')}, "
                    f"out {st.get('output_tokens')}, thinking {st.get('thinking_tokens')}, cost {st.get('cost_usd')}, "
                    f"Flex {st.get('flex_requests', 0)}, Flex->Standard {st.get('flex_fallbacks', 0)}"
                )
            loc = usage.get("local")
            if isinstance(loc, dict) and (loc.get("requests") or loc.get("saved_tokens") or loc.get("frames_filtered")):
                lines.append(
                    f"  Local AI & Cost-Saver: model {loc.get('model') or 'local'}, "
                    f"{loc.get('requests', 0)} local query(s), saved {loc.get('saved_tokens', 0)} tokens "
                    f"(~${loc.get('saved_cost_usd', 0.0):.4f})"
                )
        except ValueError:
            pass
    try:
        lines.append(f"Current AI settings: {load_settings().model_dump_json()}")
    except Exception:
        pass
    return "\n".join(lines) + "\n"


@router.get("/{job_id}/export/log")
def export_job_log(job_id: str, session: Session = Depends(get_session)):
    """Everything logged while this job ran (all runs, incl. retranslations), with a summary on top."""
    from app.core.job_logging import job_log_path

    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    path = job_log_path(job.id)
    if path is not None and path.exists():
        body = path.read_text(encoding="utf-8", errors="replace")
    else:
        body = "(No log was recorded for this job - it ran before detailed logging was added.)\n"
    content = _job_log_header(job) + "\n" + "=" * 30 + " log " + "=" * 30 + "\n" + body
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers=_attachment_headers(f"{Path(job.file_name).stem}.{job.id}.log"),
    )
