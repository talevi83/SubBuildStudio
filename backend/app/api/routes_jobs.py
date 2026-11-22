import asyncio
import functools
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import func
from sqlmodel import Session, select

from app.core import edit_history as history
from app.core.config import settings
from app.core.database import get_session
from app.core.glossary_profiles import clean_name, save_profile
from app.core.housekeeping import delete_upload_if_unused, is_in_uploads
from app.core.user_settings import load_settings
from app.api.routes_media import reject_network_path, validate_media_path
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.models.schemas import (
    AudioStreamInfo,
    BulkDeleteJobsRequest,
    BulkDeleteJobsResponse,
    BulkSegmentUpdate,
    GlossaryUpdate,
    JobCreate,
    JobResponse,
    JobSettingsUpdate,
    DetectSpeakersRequest,
    DetectSpeakersResponse,
    MoveSegmentRequest,
    QAIssueSchema,
    RetimeRequest,
    RetranslateRequest,
    SegmentCreate,
    SegmentUpdate,
    SegmentsChangeResponse,
    ShiftSegmentsRequest,
    MergeSegmentsRequest,
    SplitSegmentRequest,
    SubtitleSegmentSchema,
)
from app.pipeline.orchestrator import PipelineOrchestrator, spawn_background, ACTIVE_STATUSES, _now
from app.utils.filenames import safe_basename
from app.utils.rtl_fixer import strip_directional_marks
from app.utils.srt_formatter import calculate_cps

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])
logger = logging.getLogger(__name__)


def _history_locked(fn):
    """One history-changing request per job at a time (overlapping autosaves would interleave)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with history.job_lock(kwargs["job_id"]):
            return fn(*args, **kwargs)
    return wrapper


def _history_begin(session: Session, job: Job) -> bool:
    """Start recording an edit. Not while a run is changing the subtitles (it is caught up later)."""
    if job.status in ACTIVE_STATUSES:
        return False
    history.begin(session, job)
    return True


def _history_record(session: Session, job: Job, recording: bool, action: str,
                    details: Optional[dict] = None, coalesce_key: Optional[str] = None) -> None:
    if not recording:
        return
    try:
        history.record(session, job, action, details, coalesce_key)
    except Exception as e:  # the edit itself is saved; never fail the request over its history
        logger.warning(f"Could not record edit history for {job.id}: {e}")


def _segment_to_schema(s: SubtitleSegmentRecord) -> SubtitleSegmentSchema:
    return SubtitleSegmentSchema(
        id=s.id,
        job_id=s.job_id,
        index=s.index,
        start_time=s.start_time,
        end_time=s.end_time,
        original_text=s.original_text,
        # Older jobs stored RTL marks in the text; the editor should never see them.
        translated_text=strip_directional_marks(s.translated_text) if s.translated_text else s.translated_text,
        translation_failed=bool(s.translation_failed),
        segment_type=s.segment_type or "DIALOGUE",
        confidence=s.confidence,
        cps=s.cps,
    )


def _job_to_response(
    job: Job,
    include_segments: bool = True,
    counts: Optional[Tuple[int, int]] = None,
) -> JobResponse:
    """Convert a Job ORM entity into a JobResponse (optionally without segments)."""
    streams: List[AudioStreamInfo] = []
    if job.audio_streams_json:
        try:
            streams = [AudioStreamInfo(**s) for s in json.loads(job.audio_streams_json)]
        except Exception:
            pass
    glossary = None
    if job.glossary_json:
        try:
            glossary = json.loads(job.glossary_json)
        except Exception:
            pass
    sync_report = None
    if job.sync_report_json:
        try:
            sync_report = json.loads(job.sync_report_json)
        except Exception:
            pass
    usage = None
    if job.usage_json:
        try:
            raw_usage = json.loads(job.usage_json)
            if isinstance(raw_usage, dict):
                if "models" not in raw_usage or "local" not in raw_usage:
                    from app.core.model_catalog import UsageTracker

                    tracker = UsageTracker()
                    tracker.merge_dict(raw_usage)
                    usage = tracker.to_dict()
                else:
                    usage = raw_usage
        except Exception:
            pass

    segments: List[SubtitleSegmentSchema] = []
    if include_segments:
        if job.segments:
            segments = [_segment_to_schema(s) for s in job.segments]
        else:
            live = PipelineOrchestrator.get_live_segments(job.id)
            segments = [SubtitleSegmentSchema(**s) for s in live]
        segment_count = len(segments)
        failed_count = sum(1 for s in segments if s.translation_failed)
    else:
        segment_count, failed_count = counts or (0, 0)
        if segment_count == 0:
            segment_count = len(PipelineOrchestrator.get_live_segments(job.id))

    return JobResponse(
        id=job.id,
        video_path=job.video_path,
        file_name=job.file_name,
        file_size=job.file_size,
        is_upload=job.is_upload,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent,
        speed_factor=job.speed_factor,
        eta_seconds=job.eta_seconds,
        error_message=job.error_message,
        warning_message=job.warning_message,
        video_duration=job.video_duration,
        audio_streams=streams,
        selected_audio_stream=job.selected_audio_stream,
        whisper_model=job.whisper_model,
        source_language=job.source_language,
        detected_language=job.detected_language,
        vad_filter=job.vad_filter,
        comms_boost=job.comms_boost,
        high_speech_sensitivity=getattr(job, "high_speech_sensitivity", False),
        use_embedded_subtitles=getattr(job, "use_embedded_subtitles", True),
        selected_subtitle_stream=getattr(job, "selected_subtitle_stream", None),
        supplement_with_whisper=getattr(job, "supplement_with_whisper", False),
        detect_onscreen_text=job.detect_onscreen_text,
        onscreen_sensitivity=getattr(job, "onscreen_sensitivity", 2),
        allow_onscreen_overlap=getattr(job, "allow_onscreen_overlap", True),
        onscreen_position=getattr(job, "onscreen_position", "top_center") or "top_center",
        convert_to_metric=getattr(job, "convert_to_metric", False),
        remove_sdh_speakers=getattr(job, "remove_sdh_speakers", False),
        skip_end_credits=getattr(job, "skip_end_credits", True),
        enable_translation=job.enable_translation,
        target_language=job.target_language,
        gemini_model=job.gemini_model,
        context_notes=job.context_notes,
        glossary_profile=job.glossary_profile,
        glossary=glossary,
        usage=usage,
        segment_count=segment_count,
        failed_translation_count=failed_count,
        job_kind=job.job_kind or "transcribe",
        sync_file_name=job.sync_file_name,
        sync_report=sync_report,
        segments=segments,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _require_not_running(job: Job, what: str) -> None:
    """Edits while a run changes the subtitles would be overwritten (or shift the rows it writes to)."""
    if job.status in ACTIVE_STATUSES or PipelineOrchestrator.is_running(job.id):
        raise HTTPException(status_code=409, detail=f"The job is still running; {what} when it has finished")


def _renumber(session: Session, job_id: str) -> None:
    rows = session.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()
    rows.sort(key=lambda r: (r.start_time, r.end_time, r.id or 0))
    for position, row in enumerate(rows, start=1):
        if row.index != position:
            row.index = position
            session.add(row)


def _get_job_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


_GEMINI_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")


def _validate_whisper_model(name: str) -> None:
    """Only the offered Whisper models: faster-whisper would otherwise download any HuggingFace
    repo, or load any local folder, named here."""
    if name not in settings.whisper_models and name != settings.whisper_default_model:
        raise HTTPException(status_code=400, detail=f"Unknown Whisper model: {name}")


def _validate_gemini_model(name: str) -> None:
    """Custom model IDs are allowed (power users), but only as a plain model ID without path traversal."""
    cleaned = name.replace("models/", "", 1)
    if ".." in cleaned or not _GEMINI_MODEL_ID.match(cleaned):
        raise HTTPException(status_code=400, detail=f"Invalid model name: {name}")


@router.post("", response_model=JobResponse)
async def create_job(req: JobCreate, session: Session = Depends(get_session)):
    """Create a new subtitling job and start the processing pipeline in the background."""
    _validate_whisper_model(req.whisper_model)
    if req.gemini_model:
        _validate_gemini_model(req.gemini_model)
    raw_path = req.video_path.strip().strip('"').strip("'")
    reject_network_path(raw_path)
    path = Path(raw_path)
    validate_media_path(path)  # before is_file(): a non-media path is refused without revealing whether it exists
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Target video file not found: {req.video_path}")

    if (req.onscreen_position or "").strip() and req.onscreen_position.strip() not in _VALID_ONSCREEN_POSITIONS:
        raise HTTPException(status_code=400, detail=f"Unknown onscreen_position: {req.onscreen_position}")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    resolved = path.resolve()
    syncing = req.sync_cues is not None
    if syncing and len(req.sync_cues) < 3:
        raise HTTPException(status_code=400, detail="The subtitle file needs at least 3 lines to sync")

    saved_settings = load_settings()
    new_job = Job(
        id=job_id,
        video_path=str(resolved),
        # Display name from the browser; it also names exported files, so keep it a plain file name
        file_name=safe_basename(req.file_name, fallback=path.name) if req.file_name else path.name,
        file_size=path.stat().st_size,
        is_upload=is_in_uploads(str(resolved)),
        status=JobStatus.IDLE,
        selected_audio_stream=req.selected_audio_stream,
        whisper_model=req.whisper_model,
        source_language=req.source_language,
        vad_filter=req.vad_filter,
        comms_boost=req.comms_boost,
        high_speech_sensitivity=req.high_speech_sensitivity,
        use_embedded_subtitles=req.use_embedded_subtitles,
        selected_subtitle_stream=req.selected_subtitle_stream,
        supplement_with_whisper=getattr(req, "supplement_with_whisper", False),
        detect_onscreen_text=req.detect_onscreen_text,
        onscreen_sensitivity=req.onscreen_sensitivity,
        allow_onscreen_overlap=req.allow_onscreen_overlap,
        onscreen_position=(req.onscreen_position or "").strip() or getattr(saved_settings, "onscreen_position", "top_center"),
        convert_to_metric=(
            req.convert_to_metric
            if req.convert_to_metric is not None
            else getattr(saved_settings, "convert_to_metric", False)
        ),
        remove_sdh_speakers=(
            req.remove_sdh_speakers
            if req.remove_sdh_speakers is not None
            else getattr(saved_settings, "remove_sdh_speakers", False)
        ),
        skip_end_credits=(
            req.skip_end_credits
            if req.skip_end_credits is not None
            else getattr(saved_settings, "skip_end_credits", True)
        ),
        enable_translation=req.enable_translation,
        target_language=req.target_language,
        gemini_model=req.gemini_model or saved_settings.translation.model,
        context_notes=req.context_notes,
        glossary_profile=clean_name(req.glossary_profile) or None,
    )
    if syncing:
        new_job.job_kind = "sync"
        new_job.enable_translation = False
        new_job.sync_file_name = req.sync_file_name or "subtitles.srt"
        new_job.sync_source_json = json.dumps({
            "cues": [c.model_dump() for c in req.sync_cues],
            "language": req.sync_language,
            "use_gemini": req.sync_use_gemini,
        }, ensure_ascii=False)

    session.add(new_job)
    session.commit()
    session.refresh(new_job)

    # The API key override (if any) is handed to the pipeline in memory only.
    orchestrator = PipelineOrchestrator(
        job_id=job_id, mode="sync" if syncing else "full", api_key=req.gemini_api_key_override)
    spawn_background(orchestrator.run())

    return _job_to_response(new_job)


@router.get("", response_model=List[JobResponse])
def list_jobs(session: Session = Depends(get_session)):
    """List all jobs (newest first) without their segments — use GET /{id} for those."""
    jobs = session.exec(select(Job).order_by(Job.created_at.desc())).all()
    rows = session.exec(
        select(
            SubtitleSegmentRecord.job_id,
            func.count(SubtitleSegmentRecord.id),
            func.sum(SubtitleSegmentRecord.translation_failed),
        ).group_by(SubtitleSegmentRecord.job_id)
    ).all()
    counts: Dict[str, Tuple[int, int]] = {r[0]: (int(r[1] or 0), int(r[2] or 0)) for r in rows}
    return [_job_to_response(j, include_segments=False, counts=counts.get(j.id, (0, 0))) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)):
    """Retrieve full job details and all subtitle segments."""
    return _job_to_response(_get_job_or_404(session, job_id))


_VALID_ONSCREEN_POSITIONS = {"top_center", "middle_right", "middle_left", "top_right", "top_left", "middle_center"}


@router.patch("/{job_id}/settings", response_model=JobResponse)
def update_job_settings(job_id: str, req: JobSettingsUpdate, session: Session = Depends(get_session)):
    """Update post-processing display/export settings (such as onscreen_position) on an existing job."""
    job = _get_job_or_404(session, job_id)
    if req.onscreen_position is not None:
        pos = req.onscreen_position.strip()
        if pos not in _VALID_ONSCREEN_POSITIONS:
            raise HTTPException(status_code=400, detail=f"Unknown onscreen_position: {pos}")
        job.onscreen_position = pos
    job.updated_at = _now()
    session.add(job)
    session.commit()
    session.refresh(job)
    return _job_to_response(job)


@router.delete("/{job_id}/segments", response_model=SegmentsChangeResponse)
@_history_locked
def delete_segments_bulk(job_id: str, segment_ids: List[int], session: Session = Depends(get_session)):
    """Delete multiple subtitle lines by ID and renumber remaining lines."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "delete lines")
    if not segment_ids:
        rows = _ordered_rows(session, job_id)
        return SegmentsChangeResponse(changed=0, segments=[_segment_to_schema(r) for r in rows])
    recording = _history_begin(session, job)
    to_delete = session.exec(
        select(SubtitleSegmentRecord).where(
            SubtitleSegmentRecord.job_id == job_id,
            SubtitleSegmentRecord.id.in_(segment_ids),
        )
    ).all()
    for seg in to_delete:
        session.delete(seg)
    session.flush()
    rows = _ordered_rows(session, job_id)
    rows.sort(key=lambda r: (r.start_time, r.end_time))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    _history_record(session, job, recording, "delete_many", {"count": len(to_delete)})
    return SegmentsChangeResponse(changed=len(to_delete), segments=[_segment_to_schema(r) for r in rows])


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, session: Session = Depends(get_session)):
    """Cancel an actively executing job pipeline."""
    job = _get_job_or_404(session, job_id)

    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already {job.status.value} and cannot be cancelled",
        )

    cancelled = PipelineOrchestrator.cancel_job(job_id)
    if not cancelled:
        # No live pipeline (e.g. orphaned record) — mark it cancelled directly
        job.status = JobStatus.CANCELLED
        job.updated_at = _now()
        session.add(job)
        session.commit()

    return {"status": "CANCELLED", "job_id": job_id}


_GENDER_WORDS = {"male": "a man (masculine)", "female": "a woman (feminine)", "group": "a group (plural)"}


def build_row_hint(req: RetranslateRequest) -> Optional[str]:
    """Turn the speaker/addressee choices and the free text into one instruction for the translator."""
    parts: List[str] = []
    if req.addressee_gender in _GENDER_WORDS:
        parts.append(
            f"The speaker is talking to {_GENDER_WORDS[req.addressee_gender]}: conjugate every second-person form "
            f"('you', 'your', imperatives, adjectives) to match."
        )
    if req.speaker_gender in _GENDER_WORDS:
        parts.append(
            f"The speaker is {_GENDER_WORDS[req.speaker_gender]}: conjugate the speaker's own first-person "
            f"verbs and adjectives to match."
        )
    if req.hint and req.hint.strip():
        parts.append(req.hint.strip())
    return " ".join(parts) or None


@router.post("/{job_id}/translate", response_model=JobResponse)
async def retranslate_job(job_id: str, req: RetranslateRequest, session: Session = Depends(get_session)):
    """(Re)translate an existing job's subtitles — all of them, or only the failed ones."""
    thinking = _valid_thinking(req.thinking)  # reject a bad value before touching the job
    row_hint = build_row_hint(req)

    def prepare() -> Tuple[Job, Optional[List[int]], bool]:
        # Runs in a worker thread: job_lock is a threading.Lock that a long /retime may hold, and
        # waiting for it on the event loop would freeze every other request and WebSocket.
        # The checks run under the same lock, so two concurrent requests can't both start a run.
        with history.job_lock(job_id):
            job = _get_job_or_404(session, job_id)
            if PipelineOrchestrator.is_running(job_id) or job.status in ACTIVE_STATUSES:
                raise HTTPException(status_code=409, detail="This job is still running")
            if not job.segments:
                raise HTTPException(status_code=400, detail="This job has no subtitles to translate")
            if req.gemini_model:
                _validate_gemini_model(req.gemini_model)

            only_indices: Optional[List[int]] = None
            if req.segment_ids:
                wanted = set(req.segment_ids)
                only_indices = sorted(s.index for s in job.segments if s.id in wanted)
                if not only_indices:
                    raise HTTPException(status_code=404, detail="None of the selected subtitles were found")
            if row_hint and req.remember_hint and only_indices:
                note = f"[#{only_indices[0]}-#{only_indices[-1]}] {row_hint}"
                job.context_notes = f"{job.context_notes.strip()}\n{note}" if job.context_notes else note

            target_changed = bool(req.target_language and req.target_language != job.target_language)
            if req.target_language:
                job.target_language = req.target_language
            # A row-scoped run keeps the project's model unless the user picked another one
            default_model = job.gemini_model if only_indices and job.gemini_model else load_settings().translation.model
            job.gemini_model = req.gemini_model or default_model
            if req.context_notes is not None:
                job.context_notes = req.context_notes
            history.begin(session, job)  # the run records its result as one step when it ends
            job.enable_translation = True
            job.status = JobStatus.TRANSLATING
            job.current_stage = "TRANSLATING"
            job.progress_percent = 0.0
            job.error_message = None
            job.warning_message = None
            job.updated_at = _now()
            session.add(job)
            session.commit()
            session.refresh(job)
            return job, only_indices, target_changed

    job, only_indices, target_changed = await asyncio.to_thread(prepare)

    orchestrator = PipelineOrchestrator(
        job_id=job_id,
        mode="translate",
        api_key=req.gemini_api_key_override,
        only_failed=req.only_failed and not target_changed,
        rebuild_glossary=req.rebuild_glossary or target_changed,
        only_indices=only_indices,
        hint=row_hint,
        thinking=thinking,
    )
    spawn_background(orchestrator.run())
    return _job_to_response(job)


_THINKING_LEVELS = {"default", "off", "minimal", "low", "medium", "high"}


def _valid_thinking(value: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return None
    if value not in _THINKING_LEVELS:
        raise HTTPException(status_code=400, detail=f"Unknown thinking level: {value}")
    return value


@router.put("/{job_id}/glossary")
def update_glossary(job_id: str, body: GlossaryUpdate, session: Session = Depends(get_session)):
    """Replace the job's glossary with the user's edited version (used by the next translation)."""
    job = _get_job_or_404(session, job_id)
    if PipelineOrchestrator.is_running(job_id) or job.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="This job is still running")
    glossary = {
        "summary": body.summary.strip(),
        "characters": [
            {**c.model_dump(), "name": c.name.strip(), "target_name": c.target_name.strip()}
            for c in body.characters if c.name.strip()
        ],
        "terms": [t.model_dump() for t in body.terms if t.source.strip()],
        "register_notes": body.register_notes.strip(),
    }
    job.glossary_json = json.dumps(glossary, ensure_ascii=False)
    if body.profile is not None:
        name = clean_name(body.profile)
        if name:
            save_profile(name, glossary)
        job.glossary_profile = name or None
    job.updated_at = _now()
    session.add(job)
    session.commit()
    return glossary


@router.post("/{job_id}/detect-speakers", response_model=DetectSpeakersResponse)
async def detect_speakers_from_video(job_id: str, req: DetectSpeakersRequest, session: Session = Depends(get_session)):
    """Ask the configured vision-capable LLM, from a few video frames plus dialogue, who speaks to whom."""
    from app.core.provider_catalog import infer_provider_from_model
    from app.core.provider_keys import resolve_secret_key
    from app.pipeline import speaker_detect as sd
    from app.pipeline.stages.audio_stage import get_ffmpeg_bin

    job = _get_job_or_404(session, job_id)
    _require_finished(job)
    wanted = set(req.segment_ids)
    rows = [r for r in _ordered_rows(session, job_id) if r.id in wanted]
    if not rows:
        raise HTTPException(status_code=400, detail="No matching rows")
    if req.gemini_model:
        _validate_gemini_model(req.gemini_model)

    app_settings = load_settings()
    chosen_model = req.gemini_model or app_settings.translation.model
    matches_settings_model = (chosen_model == app_settings.translation.model)
    inferred = None if matches_settings_model else (infer_provider_from_model(req.gemini_model) if req.gemini_model else None)
    chosen_provider = inferred or app_settings.translation.provider
    key_id = app_settings.translation.key_id if (not req.gemini_model or matches_settings_model) else None

    genai_client = None
    if chosen_provider == "google":
        vault = resolve_secret_key("google", key_id=key_id)
        key = (req.gemini_api_key_override or "").strip() or (vault.api_key if vault else "") or settings.gemini_api_key
        if not key or key == "your_gemini_api_key_here":
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY is missing")
        if key_id is None:
            from google import genai
            genai_client = genai.Client(api_key=key)
    else:
        if resolve_secret_key(chosen_provider, key_id=key_id) is None:
            raise HTTPException(status_code=400, detail=f"No API key configured for provider '{chosen_provider}'")

    times = sd.frame_times(rows[0].start_time, rows[-1].end_time)
    ffmpeg = get_ffmpeg_bin()
    grabbed = await asyncio.gather(*[asyncio.to_thread(sd.grab_frame, job.video_path, t, ffmpeg) for t in times])
    frames = [(t, data) for t, data in zip(times, grabbed) if data]
    if not frames:
        raise HTTPException(status_code=502, detail="Could not read frames from the video")
    lines = [(r.start_time, r.original_text) for r in rows[:12]]
    try:
        guess = await sd.detect_speakers(
            genai_client, chosen_model, frames, lines,
            thinking=_valid_thinking(req.thinking) or app_settings.translation.thinking,
            provider=chosen_provider,
            key_id=key_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Detection failed: {exc}")
    pick = lambda g: None if g == "unknown" else g  # noqa: E731
    return DetectSpeakersResponse(
        speaker_gender=pick(guess.speaker_gender),
        addressee_gender=pick(guess.addressee_gender),
        explanation=guess.explanation,
    )


@router.get("/{job_id}/qa", response_model=List[QAIssueSchema])
def quality_check(job_id: str, session: Session = Depends(get_session)):
    """Cheap, model-free checks on the translated lines (untranslated words, glossary mismatches, agreement)."""
    from app.pipeline.qa import check_rows

    job = _get_job_or_404(session, job_id)
    glossary = None
    if job.glossary_json:
        try:
            glossary = json.loads(job.glossary_json)
        except Exception:
            glossary = None
    rows = _ordered_rows(session, job_id)
    translation_enabled = bool(job.enable_translation or any(r.translated_text for r in rows))
    issues = check_rows(rows, glossary, translation_enabled=translation_enabled)
    return [QAIssueSchema(segment_id=i.segment_id, index=i.index, kind=i.kind, message=i.message) for i in issues]


def _delete_single_job(session: Session, job: Job) -> Tuple[str, bool]:
    """Delete a job, its cached audio/frames/logs, and (for browser uploads) the uploaded copy."""
    import shutil
    from app.core import video_embed as ve
    from app.core.job_logging import delete_job_log

    job_id = job.id
    PipelineOrchestrator.cancel_job(job_id)
    running_embed = ve.running_task(job_id)
    if running_embed is not None:
        ve.cancel_task(running_embed)
    video_path, is_upload = job.video_path, job.is_upload

    try:
        audio_path = settings.storage_dir / "audio" / f"{job_id}.wav"
        audio_path.unlink(missing_ok=True)
    except OSError:
        pass

    session.delete(job)
    session.commit()

    delete_job_log(job_id)
    try:
        (settings.storage_dir / "regions" / f"{job_id}.json").unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(settings.storage_dir / "frames" / job_id, ignore_errors=True)
    except OSError:
        pass
    try:
        shutil.rmtree(settings.storage_dir / "exports" / f"embed_{job_id}", ignore_errors=True)
    except OSError:
        pass

    upload_deleted = False
    if is_upload or is_in_uploads(video_path):
        upload_deleted = delete_upload_if_unused(video_path)
    return job_id, upload_deleted


@router.post("/bulk-delete", response_model=BulkDeleteJobsResponse)
def bulk_delete_jobs(req: BulkDeleteJobsRequest, session: Session = Depends(get_session)):
    """Delete multiple jobs by ID, or all existing jobs when delete_all=True."""
    if req.delete_all:
        jobs = session.exec(select(Job)).all()
    elif req.job_ids:
        jobs = session.exec(select(Job).where(Job.id.in_(req.job_ids))).all()
    else:
        return BulkDeleteJobsResponse(deleted_count=0, deleted_ids=[])

    deleted_ids: List[str] = []
    for job in jobs:
        jid, _ = _delete_single_job(session, job)
        deleted_ids.append(jid)
    return BulkDeleteJobsResponse(deleted_count=len(deleted_ids), deleted_ids=deleted_ids)


@router.delete("")
def delete_all_jobs(session: Session = Depends(get_session)):
    """Delete all existing jobs and their associated cache/upload files."""
    jobs = session.exec(select(Job)).all()
    deleted_ids: List[str] = []
    for job in jobs:
        jid, _ = _delete_single_job(session, job)
        deleted_ids.append(jid)
    return {"deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}


@router.delete("/{job_id}")
def delete_job(job_id: str, session: Session = Depends(get_session)):
    """Delete a job, its cached audio and (for browser uploads) the uploaded copy."""
    job = _get_job_or_404(session, job_id)
    _, upload_deleted = _delete_single_job(session, job)
    return {"deleted": True, "job_id": job_id, "upload_deleted": upload_deleted}


def _apply_segment_patch(seg: SubtitleSegmentRecord, patch: SegmentUpdate) -> None:
    start = round(patch.start_time, 3) if patch.start_time is not None else seg.start_time
    end = round(patch.end_time, 3) if patch.end_time is not None else seg.end_time
    if start < 0 or end <= start:
        raise HTTPException(status_code=422, detail=f"Segment #{seg.index}: end time must be after start time")
    seg.start_time, seg.end_time = start, end
    if patch.original_text is not None:
        seg.original_text = patch.original_text
    if patch.translated_text is not None:
        seg.translated_text = strip_directional_marks(patch.translated_text)
        if seg.translated_text.strip():
            seg.translation_failed = False

    text_for_cps = seg.translated_text if seg.translated_text else seg.original_text
    seg.cps = calculate_cps(text_for_cps, max(0.1, seg.end_time - seg.start_time))


@router.patch("/{job_id}/segments/{segment_id}", response_model=SubtitleSegmentSchema)
@_history_locked
def update_segment(
    job_id: str,
    segment_id: int,
    patch: SegmentUpdate,
    session: Session = Depends(get_session),
):
    """Update subtitle segment timestamps or text (Original or Translated)."""
    seg = session.get(SubtitleSegmentRecord, segment_id)
    if not seg or seg.job_id != job_id:
        raise HTTPException(status_code=404, detail="Subtitle segment not found")
    job = _get_job_or_404(session, job_id)
    if patch.start_time is not None or patch.end_time is not None:
        # Text can be fixed while a translation runs; times can't (they reorder the lines)
        _require_not_running(job, "change times")
    recording = _history_begin(session, job)

    _apply_segment_patch(seg, patch)
    session.add(seg)
    if patch.start_time is not None or patch.end_time is not None:
        session.flush()
        rows = session.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()
        rows.sort(key=lambda r: (r.start_time, r.end_time, r.id or 0))
        for position, row in enumerate(rows, start=1):
            if row.index != position:
                row.index = position
                session.add(row)
    session.commit()
    session.refresh(seg)
    fields = sorted(patch.model_dump(exclude_unset=True, exclude_none=True))
    timing_only = bool(fields) and set(fields) <= {"start_time", "end_time"}
    _history_record(
        session, job, recording, "edit_timing" if timing_only else "edit",
        {"index": seg.index, "segment_id": seg.id, "fields": fields},
        coalesce_key=f"{'timing' if timing_only else 'text'}:{seg.id}:{','.join(fields)}",
    )
    session.refresh(seg)
    return _segment_to_schema(seg)


@router.patch("/{job_id}/segments", response_model=List[SubtitleSegmentSchema])
@_history_locked
def update_segments_bulk(
    job_id: str,
    patches: List[BulkSegmentUpdate],
    session: Session = Depends(get_session),
):
    """Apply many segment edits in one transaction (used by Replace All)."""
    ids = [p.id for p in patches]
    rows = {
        s.id: s
        for s in session.exec(
            select(SubtitleSegmentRecord).where(
                SubtitleSegmentRecord.job_id == job_id,
                SubtitleSegmentRecord.id.in_(ids),
            )
        ).all()
    }
    missing = [i for i in ids if i not in rows]
    if missing:
        raise HTTPException(status_code=404, detail=f"Subtitle segments not found: {missing}")
    job = _get_job_or_404(session, job_id)
    times_changed = any(p.start_time is not None or p.end_time is not None for p in patches)
    if times_changed:
        _require_not_running(job, "change times")
    recording = _history_begin(session, job)
    for p in patches:
        _apply_segment_patch(rows[p.id], p)
        session.add(rows[p.id])
    if times_changed:
        # Times changed: keep the line numbers in time order, like a single edit does
        session.flush()
        _renumber(session, job_id)
    session.commit()
    _history_record(session, job, recording, "bulk_edit", {"count": len(patches)})
    return [_segment_to_schema(session.get(SubtitleSegmentRecord, i)) for i in ids]


_SEGMENT_TYPES = {"DIALOGUE", "COMMS", "ON_SCREEN", "FOREIGN"}
_MIN_NEW_DURATION = 0.1


def _timecode(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


@router.post("/{job_id}/segments", response_model=SegmentsChangeResponse)
@_history_locked
def create_segment(job_id: str, req: SegmentCreate, session: Session = Depends(get_session)):
    """Add a subtitle line by hand. It must not overlap an existing line; lines are renumbered."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "add lines")
    start, end = round(req.start_time, 3), round(req.end_time, 3)
    if start < 0:
        raise HTTPException(status_code=422, detail="The start time cannot be negative")
    if end - start < _MIN_NEW_DURATION:
        raise HTTPException(status_code=422, detail="The end time must be after the start time")
    if job.video_duration and start >= job.video_duration:
        raise HTTPException(status_code=422,
                            detail=f"The video ends at {_timecode(job.video_duration)}; the line starts after that")
    if job.video_duration and end > job.video_duration + 0.5:
        end = round(job.video_duration, 3)
    original = (req.original_text or "").strip()
    translated = strip_directional_marks(req.translated_text).strip() if req.translated_text else ""
    if not original and not translated:
        raise HTTPException(status_code=422, detail="Write the text of the new line")
    kind = (req.segment_type or "DIALOGUE").upper()
    if kind not in _SEGMENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unknown line type: {req.segment_type}")

    rows = _ordered_rows(session, job_id)
    if kind == "ON_SCREEN":
        clash = next(
            (r for r in rows if r.segment_type == "ON_SCREEN" and start < r.end_time - 0.0005 and end > r.start_time + 0.0005),
            None,
        )
    else:
        clash = next(
            (r for r in rows if r.segment_type != "ON_SCREEN" and start < r.end_time - 0.0005 and end > r.start_time + 0.0005),
            None,
        )
    if clash is not None:
        position = rows.index(clash) + 1
        raise HTTPException(status_code=409, detail=(
            f"Overlaps line #{position} ({_timecode(clash.start_time)} - {_timecode(clash.end_time)})"))

    recording = _history_begin(session, job)
    new = SubtitleSegmentRecord(
        job_id=job_id, index=0, start_time=start, end_time=end,
        original_text=original, translated_text=translated or None, translation_failed=False,
        segment_type=kind, confidence=1.0,
    )
    _refresh_cps(new)
    session.add(new)
    rows.append(new)
    rows.sort(key=lambda r: (r.start_time, r.end_time))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    session.refresh(new)
    _history_record(session, job, recording, "insert", {"index": new.index})
    rows = _ordered_rows(session, job_id)
    return SegmentsChangeResponse(changed=1, created_id=new.id, segments=[_segment_to_schema(r) for r in rows])


@router.delete("/{job_id}/segments/{segment_id}", response_model=SegmentsChangeResponse)
@_history_locked
def delete_segment(job_id: str, segment_id: int, session: Session = Depends(get_session)):
    """Delete a subtitle line by user request and renumber subsequent lines."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "delete lines")
    seg = session.get(SubtitleSegmentRecord, segment_id)
    if not seg or seg.job_id != job_id:
        raise HTTPException(status_code=404, detail="Subtitle segment not found")

    recording = _history_begin(session, job)
    deleted_index = seg.index
    session.delete(seg)
    session.flush()

    rows = _ordered_rows(session, job_id)
    rows.sort(key=lambda r: (r.start_time, r.end_time))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    _history_record(session, job, recording, "delete", {"index": deleted_index})

    return SegmentsChangeResponse(changed=1, segments=[_segment_to_schema(r) for r in _ordered_rows(session, job_id)])


def _split_cue_text(text: str) -> Tuple[str, str]:
    """Split a subtitle cue cleanly across line break or nearest balanced word boundary."""
    s = (text or "").strip()
    if not s:
        return "", ""
    if "\n" in s:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[0], "\n".join(lines[1:])
    words = s.split()
    if len(words) >= 2:
        best_i, best_score = 1, float("inf")
        for i in range(1, len(words)):
            left = " ".join(words[:i])
            right = " ".join(words[i:])
            score = abs(len(left) - len(right)) - (8 if words[i - 1][-1:] in ".!?;:," else 0)
            if score < best_score:
                best_i, best_score = i, score
        return " ".join(words[:best_i]), " ".join(words[best_i:])
    if len(s) >= 2:
        mid = len(s) // 2
        return s[:mid], s[mid:]
    return s, ""


@router.post("/{job_id}/segments/{segment_id}/split", response_model=SegmentsChangeResponse)
@_history_locked
def split_segment(
    job_id: str,
    segment_id: int,
    req: Optional[SplitSegmentRequest] = None,
    session: Session = Depends(get_session),
):
    """Split one subtitle segment into two consecutive segments and renumber."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "split lines")
    seg = session.get(SubtitleSegmentRecord, segment_id)
    if not seg or seg.job_id != job_id:
        raise HTTPException(status_code=404, detail="Subtitle segment not found")

    dur = seg.end_time - seg.start_time
    if dur < 0.4:
        raise HTTPException(status_code=422, detail="Line is too short to split")

    orig1, orig2 = _split_cue_text(seg.original_text or "")
    has_tr = bool((seg.translated_text or "").strip())
    tr1, tr2 = _split_cue_text(seg.translated_text or "") if has_tr else ("", "")
    if not orig2 and not tr2:
        raise HTTPException(status_code=422, detail="Line text cannot be split into two parts")

    req_time = req.split_time if req is not None else None
    if req_time is not None and seg.start_time + 0.15 <= req_time <= seg.end_time - 0.15:
        split_at = round(req_time, 3)
    else:
        p1, p2 = (tr1, tr2) if (tr1 and tr2) else (orig1, orig2)
        ratio = max(0.25, min(0.75, len(p1) / max(1, len(p1) + len(p2))))
        split_at = round(seg.start_time + dur * ratio, 3)

    recording = _history_begin(session, job)
    old_end = seg.end_time
    old_index = seg.index

    seg.end_time = split_at
    seg.original_text = orig1 or seg.original_text
    seg.translated_text = (tr1 or seg.translated_text) if has_tr else None
    _refresh_cps(seg)
    session.add(seg)

    second = SubtitleSegmentRecord(
        job_id=job_id,
        index=0,
        start_time=split_at,
        end_time=old_end,
        original_text=orig2 or orig1,
        translated_text=(tr2 or tr1) if has_tr else None,
        translation_failed=seg.translation_failed,
        segment_type=seg.segment_type,
        confidence=seg.confidence,
    )
    _refresh_cps(second)
    session.add(second)
    session.flush()

    rows = _ordered_rows(session, job_id)
    rows.sort(key=lambda r: (r.start_time, r.end_time, r.id or 0))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    session.refresh(second)
    _history_record(session, job, recording, "split", {"index": old_index})
    rows = _ordered_rows(session, job_id)
    return SegmentsChangeResponse(changed=2, created_id=second.id, segments=[_segment_to_schema(r) for r in rows])


def _lane(row: SubtitleSegmentRecord) -> str:
    """On-screen text is shown in its own place and may overlap dialogue; everything else shares one lane."""
    return "ON_SCREEN" if row.segment_type == "ON_SCREEN" else "MAIN"


def _are_neighbours(rows: List[SubtitleSegmentRecord], targets: List[SubtitleSegmentRecord]) -> bool:
    """True when `targets` are one lane's consecutive lines (in time order)."""
    lanes = {_lane(t) for t in targets}
    if len(lanes) != 1:
        return False
    lane = lanes.pop()
    order = [r.id for r in rows if _lane(r) == lane]
    positions = sorted(order.index(t.id) for t in targets)
    return positions == list(range(positions[0], positions[0] + len(positions)))


@router.post("/{job_id}/segments/merge", response_model=SegmentsChangeResponse)
@_history_locked
def merge_segments(job_id: str, req: MergeSegmentsRequest, session: Session = Depends(get_session)):
    """Merge 2 or more subtitle segments into one combined cue and renumber."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "merge lines")

    targets = session.exec(
        select(SubtitleSegmentRecord).where(
            SubtitleSegmentRecord.job_id == job_id,
            SubtitleSegmentRecord.id.in_(req.segment_ids),
        )
    ).all()
    targets.sort(key=lambda r: (r.start_time, r.end_time, r.index))
    if len(targets) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 lines to merge")
    if not _are_neighbours(_ordered_rows(session, job_id), targets):
        # Merging #1 and #4 would make one line that covers #2 and #3 (overlapping subtitles)
        raise HTTPException(
            status_code=400,
            detail="Only neighbouring lines can be merged (on-screen text and dialogue are merged separately)",
        )

    recording = _history_begin(session, job)
    first = targets[0]
    first.start_time = min(t.start_time for t in targets)
    first.end_time = max(t.end_time for t in targets)
    first.original_text = "\n".join(t.original_text.strip() for t in targets if (t.original_text or "").strip())
    tr_parts = [t.translated_text.strip() for t in targets if (t.translated_text or "").strip()]
    # A line with text but no translation would vanish from the merged translation: keep the merged
    # line flagged so "Retranslate failed" picks it up instead of it looking fully translated.
    untranslated_part = any(
        (t.original_text or "").strip() and not (t.translated_text or "").strip() for t in targets
    )
    first.translated_text = "\n".join(tr_parts) if tr_parts else None
    if tr_parts:
        first.translation_failed = untranslated_part
    else:
        first.translation_failed = any(t.translation_failed for t in targets)
    _refresh_cps(first)
    session.add(first)

    for extra in targets[1:]:
        session.delete(extra)
    session.flush()

    rows = _ordered_rows(session, job_id)
    rows.sort(key=lambda r: (r.start_time, r.end_time, r.id or 0))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    session.refresh(first)
    _history_record(session, job, recording, "merge", {"index": first.index, "count": len(targets)})
    rows = _ordered_rows(session, job_id)
    return SegmentsChangeResponse(changed=len(targets), created_id=first.id, segments=[_segment_to_schema(r) for r in rows])


@router.post("/{job_id}/segments/shift", response_model=SegmentsChangeResponse)
@_history_locked
def shift_segments(job_id: str, req: ShiftSegmentsRequest, session: Session = Depends(get_session)):
    """Shift timestamps of all or selected subtitle segments by offset_seconds (+/-)."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "shift lines")
    if abs(req.offset_seconds) < 0.001 or abs(req.offset_seconds) > 86400:
        raise HTTPException(status_code=422, detail="Offset must be between 0.001s and 86400s in magnitude")

    rows = _ordered_rows(session, job_id)
    wanted = set(req.segment_ids) if req.segment_ids else None
    targets = [r for r in rows if wanted is None or r.id in wanted]
    if not targets:
        return SegmentsChangeResponse(changed=0, segments=[_segment_to_schema(r) for r in rows])

    recording = _history_begin(session, job)
    for seg in targets:
        dur = max(0.1, round(seg.end_time - seg.start_time, 3))
        shifted_start = seg.start_time + req.offset_seconds
        new_start = max(0.0, round(shifted_start, 3))
        if shifted_start < 0:
            new_end = round(new_start + dur, 3)
        else:
            new_end = max(round(new_start + 0.1, 3), round(seg.end_time + req.offset_seconds, 3))
        seg.start_time = new_start
        seg.end_time = new_end
        _refresh_cps(seg)
        session.add(seg)
    session.flush()

    rows = _ordered_rows(session, job_id)
    rows.sort(key=lambda r: (r.start_time, r.end_time, r.id or 0))
    for position, row in enumerate(rows, start=1):
        row.index = position
        session.add(row)
    session.commit()
    _history_record(session, job, recording, "shift_time", {"count": len(targets)})
    return SegmentsChangeResponse(changed=len(targets), segments=[_segment_to_schema(r) for r in rows])


# --- Timing-only tools -------------------------------------------------------------------------

_MOVE_FIELDS = ("original_text", "translated_text", "translation_failed", "segment_type", "confidence")
_WINDOW_PAD = 3.0


def _job_audio_params(job: Job) -> Tuple[int, int]:
    """(selected audio stream, its channel count) for re-extracting audio."""
    channels = 2
    if job.audio_streams_json:
        try:
            for st in json.loads(job.audio_streams_json):
                if st.get("index") == job.selected_audio_stream:
                    channels = int(st.get("channels", 2))
        except (ValueError, TypeError):
            pass
    return job.selected_audio_stream, channels


def _ordered_rows(session: Session, job_id: str) -> List[SubtitleSegmentRecord]:
    rows = session.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()
    return sorted(rows, key=lambda r: (r.start_time, r.index))


def _refresh_cps(seg: SubtitleSegmentRecord) -> None:
    text = seg.translated_text if seg.translated_text else seg.original_text
    seg.cps = calculate_cps(text, max(0.1, seg.end_time - seg.start_time))


def _retime_rows(job: Job, rows: List[SubtitleSegmentRecord], scope: List[int], allow_silent_shift: bool) -> int:
    """Run the timing-only check on `scope` positions of the time-ordered `rows`; returns cues changed."""
    from app.pipeline.retime import retime_cues
    from app.pipeline.speech_regions import speech_regions

    if not scope:
        return 0
    stream, channels = _job_audio_params(job)
    if len(scope) == len(rows):
        regions = speech_regions(job.id, job.video_path, stream, channels, job.comms_boost)
    else:
        lo = max(0.0, rows[min(scope)].start_time - _WINDOW_PAD)
        hi = rows[max(scope)].end_time + _WINDOW_PAD
        regions = speech_regions(job.id, job.video_path, stream, channels, job.comms_boost, lo, hi)
    stats = retime_cues(rows, regions, scope=scope, allow_silent_shift=allow_silent_shift)
    return stats.changed


def _require_finished(job: Job) -> None:
    if job.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="The job is still running")
    if not job.video_path or not Path(job.video_path).exists():
        raise HTTPException(status_code=409, detail="The source video is no longer available")


@router.post("/{job_id}/retime", response_model=SegmentsChangeResponse)
@_history_locked
def retime_job(job_id: str, req: RetimeRequest, session: Session = Depends(get_session)):
    """Timing-only check against the audio: whole video, or only the cues inside [start, end]."""
    job = _get_job_or_404(session, job_id)
    _require_finished(job)
    rows = _ordered_rows(session, job_id)
    if req.start is None and req.end is None:
        scope = list(range(len(rows)))
    else:
        lo = req.start if req.start is not None else 0.0
        hi = req.end if req.end is not None else float("inf")
        scope = [i for i, r in enumerate(rows) if r.end_time >= lo and r.start_time <= hi]
    recording = _history_begin(session, job)
    try:
        changed = _retime_rows(job, rows, scope, allow_silent_shift=False)
    except Exception as e:  # ffmpeg / VAD failures must not take the job down
        raise HTTPException(status_code=500, detail=f"Timing check failed: {e}")
    for r in rows:
        _refresh_cps(r)
        session.add(r)
    session.commit()
    _history_record(session, job, recording, "retime", {"count": changed})
    return SegmentsChangeResponse(changed=changed, segments=[_segment_to_schema(r) for r in _ordered_rows(session, job_id)])


@router.post("/{job_id}/segments/move", response_model=SegmentsChangeResponse)
@_history_locked
def move_segment(job_id: str, req: MoveSegmentRequest, session: Session = Depends(get_session)):
    """
    Move one row to another position: the *content* travels, the time slots stay where they are,
    then only the affected section is re-checked for timing against the audio.
    """
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "move lines")
    rows = _ordered_rows(session, job_id)
    old = next((i for i, r in enumerate(rows) if r.id == req.segment_id), None)
    if old is None:
        raise HTTPException(status_code=404, detail="Subtitle segment not found")
    new = max(0, min(len(rows) - 1, req.to_position))
    if new == old:
        return SegmentsChangeResponse(changed=0, segments=[_segment_to_schema(r) for r in rows])

    recording = _history_begin(session, job)
    lo, hi = min(old, new), max(old, new)
    contents = [{f: getattr(rows[i], f) for f in _MOVE_FIELDS} for i in range(lo, hi + 1)]
    moved = contents.pop(old - lo)
    contents.insert(new - lo, moved)
    for offset, content in enumerate(contents):
        row = rows[lo + offset]
        for f, v in content.items():
            setattr(row, f, v)

    retimed, message, changed = True, None, 0
    try:
        if _can_retime(job):
            changed = _retime_rows(job, rows, list(range(lo, hi + 1)), allow_silent_shift=True)
        else:
            retimed, message = False, "Source video unavailable; times were kept as they were"
    except Exception as e:
        retimed, message = False, f"Rows moved, but the timing check failed: {e}"
    for r in rows[lo:hi + 1]:
        _refresh_cps(r)
        session.add(r)
    session.commit()
    _history_record(session, job, recording, "move", {"from": old + 1, "to": new + 1})
    rows = _ordered_rows(session, job_id)
    return SegmentsChangeResponse(
        changed=changed, retimed=retimed, message=message, segments=[_segment_to_schema(r) for r in rows]
    )


def _can_retime(job: Job) -> bool:
    return job.status not in ACTIVE_STATUSES and bool(job.video_path) and Path(job.video_path).exists()
