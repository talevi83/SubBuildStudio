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
    AddresseeSchema,
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
    DiarizeJobRequest,
    MatchSpeakersRequest,
    MergeSpeakersRequest,
    MoveSegmentRequest,
    QAIssueSchema,
    RetimeRequest,
    RetranslateRequest,
    SegmentCreate,
    SegmentUpdate,
    SegmentsChangeResponse,
    ShiftSegmentsRequest,
    SpeakerProfileSchema,
    SpeakersUpdateRequest,
    MergeSegmentsRequest,
    SplitSegmentRequest,
    SubtitleSegmentSchema,
    WaveformResponse,
    WordTimingSchema,
)
from app.pipeline.orchestrator import (
    ACTIVE_STATUSES,
    PipelineOrchestrator,
    _get_pipeline_semaphore,
    _now,
    active_pipelines,
    spawn_background,
)
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


def _parse_words_raw(raw: Optional[str]) -> Optional[List[dict]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return None
        return [item for item in parsed if isinstance(item, dict) and "w" in item and "s" in item and "e" in item]
    except (ValueError, TypeError):
        return None


def _parse_words_json(raw: Optional[str]) -> Optional[List[WordTimingSchema]]:
    parsed = _parse_words_raw(raw)
    if not parsed:
        return None
    out: List[WordTimingSchema] = []
    for item in parsed:
        try:
            out.append(WordTimingSchema(w=str(item["w"]), s=float(item["s"]), e=float(item["e"])))
        except (ValueError, TypeError):
            continue
    return out or None


def _parse_addressee_json(raw: Optional[str]) -> Optional[AddresseeSchema]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return AddresseeSchema(**parsed) if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _manual_addressee_json(choice: Optional[str]) -> Optional[str]:
    """The stored form of a listener picked by hand (male / female / group); None for no choice."""
    from app.pipeline.addressee import manual_record

    record = manual_record(choice)
    return json.dumps(record.to_dict(), ensure_ascii=False) if record is not None else None


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
        words=_parse_words_json(getattr(s, "words_json", None)),
        speaker=getattr(s, "speaker", None),
        addressee=_parse_addressee_json(getattr(s, "addressee_json", None)),
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
    speakers = None
    if getattr(job, "speakers_json", None):
        try:
            raw_speakers = json.loads(job.speakers_json)
            if isinstance(raw_speakers, dict):
                speakers = {
                    str(k): SpeakerProfileSchema(**v)
                    for k, v in raw_speakers.items()
                    if isinstance(v, dict)
                } or None
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
        stage_progress_percent=PipelineOrchestrator.get_stage_progress(job.id),
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
        enable_diarization=bool(getattr(job, "enable_diarization", False)),
        speakers=speakers,
        auto_save_srt=bool(getattr(job, "auto_save_srt", False)),
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
    is_upload_file = is_in_uploads(str(resolved))
    limit = max(1, settings.max_concurrent_transcriptions)
    running_slots = sum(1 for p in active_pipelines.values() if p.mode in ("full", "sync"))
    is_queued = _get_pipeline_semaphore().locked() or running_slots >= limit

    new_job = Job(
        id=job_id,
        video_path=str(resolved),
        # Display name from the browser; it also names exported files, so keep it a plain file name
        file_name=safe_basename(req.file_name, fallback=path.name) if req.file_name else path.name,
        file_size=path.stat().st_size,
        is_upload=is_upload_file,
        status=JobStatus.QUEUED if is_queued else JobStatus.IDLE,
        current_stage="QUEUED" if is_queued else None,
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
        enable_diarization=bool(getattr(req, "enable_diarization", False)),
        auto_save_srt=bool(getattr(req, "auto_save_srt", False)) and not is_upload_file,
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


@router.get("/{job_id}/waveform", response_model=WaveformResponse)
async def get_job_waveform(job_id: str, session: Session = Depends(get_session)):
    """Return cached or freshly extracted audio amplitude peaks (50 peaks/s) for the Studio timeline."""
    from app.pipeline.waveform import cache_path, get_or_create_waveform

    job = _get_job_or_404(session, job_id)
    cached_file = cache_path(job.id)
    wav_file = settings.storage_dir / "audio" / f"{job.id}.wav"
    if not cached_file.is_file() and not wav_file.is_file():
        if not job.video_path or not Path(job.video_path).is_file():
            raise HTTPException(status_code=404, detail="Source video is unavailable for waveform generation")

    stream, channels = _job_audio_params(job)
    try:
        data = await get_or_create_waveform(
            job_id=job.id,
            video_path=job.video_path,
            audio_stream=stream,
            channels=channels,
            comms_boost=job.comms_boost,
            duration=job.video_duration or 0.0,
        )
        return WaveformResponse(**data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Waveform extraction failed for job {job_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not extract audio waveform: {exc}")


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
            manual_listener = _manual_addressee_json(req.addressee_gender) if only_indices else None
            if manual_listener:
                # Remember who these rows are spoken to: the tag is shown on the rows, survives later
                # runs, and tells the resolver about the lines around them
                selected = set(req.segment_ids or ())
                for seg in job.segments:
                    if seg.id in selected:
                        seg.addressee_json = manual_listener
                        session.add(seg)
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


@router.put("/{job_id}/speakers", response_model=JobResponse)
def update_speakers(job_id: str, body: SpeakersUpdateRequest, session: Session = Depends(get_session)):
    """Save edited speaker profiles (names, genders, colors) on a job."""
    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "edit speaker profiles")
    serialized = {k: v.model_dump() for k, v in body.speakers.items()}
    job.speakers_json = json.dumps(serialized, ensure_ascii=False)
    job.updated_at = _now()
    session.add(job)
    session.commit()
    session.refresh(job)
    return _job_to_response(job)


@router.post("/{job_id}/diarize", response_model=JobResponse)
async def diarize_job(
    job_id: str,
    req: Optional[DiarizeJobRequest] = None,
    session: Session = Depends(get_session),
):
    """Run local ONNX acoustic speaker diarization retroactively on an existing job."""
    from app.pipeline import diarization, speech_regions
    from app.pipeline.speaker_match import match_speakers_to_characters

    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "detect speakers from audio")
    if not job.video_path or not Path(job.video_path).is_file():
        raise HTTPException(status_code=404, detail="Source video file not found")

    rows = _ordered_rows(session, job_id)
    if not rows:
        return _job_to_response(job)

    spans = [(r.start_time, r.end_time) for r in rows if (r.segment_type or "DIALOGUE") != "ON_SCREEN"]
    if not spans:
        return _job_to_response(job)

    stream, channels = _job_audio_params(job)
    wav_path = settings.storage_dir / "audio" / f"{job_id}_diarize.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = req.threshold if (req and req.threshold is not None) else 0.66
    min_sp = req.min_speakers if req else None
    max_sp = req.max_speakers if req else None
    glossary = json.loads(job.glossary_json) if job.glossary_json else None
    target_language = job.target_language

    def _run_diarization():
        speech_regions._extract(job.video_path, str(wav_path), stream, channels, job.comms_boost, 0.0, None)
        onnx_path = diarization.ensure_diarization_model()
        embedder = diarization.OnnxSpeakerEmbedder(onnx_path=onnx_path)
        timeline, speakers = diarization.diarize_audio_spans(
            wav_path,
            spans,
            embedder=embedder,
            threshold=threshold,
            min_speakers=min_sp,
            max_speakers=max_sp,
        )
        if timeline and speakers:
            from app.pipeline.voice_gender import label_speakers_by_voice

            try:
                label_speakers_by_voice(wav_path, timeline, speakers)
            except Exception as exc:  # a gender guess is a bonus, never a reason to fail
                logger.info(f"Voice gender estimate skipped for {job_id}: {exc}")
        return timeline, speakers

    try:
        timeline, speakers = await asyncio.to_thread(_run_diarization)
    finally:
        wav_path.unlink(missing_ok=True)

    match_segs = [
        {
            "id": r.id,
            "index": r.index,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "original_text": r.original_text,
            "segment_type": r.segment_type,
            "speaker": None,
        }
        for r in rows
    ]
    diarization.assign_speakers_to_segments(match_segs, timeline)
    speakers = diarization.prune_unused_speakers(speakers, match_segs)
    if req is None or req.auto_match:
        try:
            await match_speakers_to_characters(
                match_segs,
                speakers,
                glossary,
                target_language=target_language,
                use_ai=True,
                overwrite_manual=False,
                auto_merge=True,
            )
            speakers = diarization.prune_unused_speakers(speakers, match_segs)
        except Exception as exc:
            logger.info(f"Auto-matching after diarization skipped for {job_id}: {exc}")

    with history.job_lock(job_id):
        session.expire_all()
        job = _get_job_or_404(session, job_id)
        _require_not_running(job, "detect speakers from audio")
        rows = _ordered_rows(session, job_id)
        recording = _history_begin(session, job)
        by_id = {s["id"]: s["speaker"] for s in match_segs if s.get("id") is not None}
        by_idx = {s["index"]: s["speaker"] for s in match_segs}
        for r in rows:
            r.speaker = by_id[r.id] if r.id in by_id else by_idx.get(r.index)
            session.add(r)
        speakers = diarization.prune_unused_speakers(speakers, rows)
        job.speakers_json = json.dumps(speakers, ensure_ascii=False)
        if glossary is not None:
            job.glossary_json = json.dumps(glossary, ensure_ascii=False)
        job.enable_diarization = True
        job.updated_at = _now()
        session.add(job)
        session.commit()
        _history_record(session, job, recording, "diarize", {"count": len(speakers)})
        session.refresh(job)
        return _job_to_response(job)


@router.post("/{job_id}/match-speakers", response_model=JobResponse)
async def match_speakers_endpoint(
    job_id: str,
    req: Optional[MatchSpeakersRequest] = None,
    session: Session = Depends(get_session),
):
    """Cross-reference acoustic speakers against dialogue cues and glossary characters."""
    from app.pipeline import diarization
    from app.pipeline.speaker_match import match_speakers_to_characters

    job = _get_job_or_404(session, job_id)
    _require_not_running(job, "match speakers to characters")
    rows = _ordered_rows(session, job_id)
    speakers = json.loads(job.speakers_json or "{}")
    if not speakers or not rows:
        return _job_to_response(job)
    glossary = json.loads(job.glossary_json) if job.glossary_json else None

    match_segs = [
        {
            "id": r.id,
            "index": r.index,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "original_text": r.original_text,
            "segment_type": r.segment_type,
            "speaker": r.speaker,
        }
        for r in rows
    ]
    await match_speakers_to_characters(
        match_segs,
        speakers,
        glossary,
        target_language=job.target_language,
        use_ai=True,
        overwrite_manual=bool(req.overwrite_manual) if req else False,
        auto_merge=bool(req.auto_merge) if req else True,
    )

    with history.job_lock(job_id):
        session.expire_all()
        job = _get_job_or_404(session, job_id)
        _require_not_running(job, "match speakers to characters")
        rows = _ordered_rows(session, job_id)
        by_id = {s["id"]: s["speaker"] for s in match_segs if s.get("id") is not None}
        changed_rows = any(r.id in by_id and r.speaker != by_id[r.id] for r in rows)
        recording = _history_begin(session, job) if changed_rows else None
        if changed_rows:
            for r in rows:
                if r.id in by_id and r.speaker != by_id[r.id]:
                    r.speaker = by_id[r.id]
                    session.add(r)
        speakers = diarization.prune_unused_speakers(speakers, rows)
        job.speakers_json = json.dumps(speakers, ensure_ascii=False)
        if glossary is not None:
            job.glossary_json = json.dumps(glossary, ensure_ascii=False)
        job.updated_at = _now()
        session.add(job)
        session.commit()
        if recording is not None:
            _history_record(session, job, recording, "diarize", {"count": len(speakers)})
        session.refresh(job)
        return _job_to_response(job)


@router.post("/{job_id}/speakers/merge", response_model=JobResponse)
def merge_speakers_endpoint(
    job_id: str,
    body: MergeSpeakersRequest,
    session: Session = Depends(get_session),
):
    """Merge or unassign acoustic speaker clusters and clean up glossary references."""
    from app.pipeline import diarization

    source_set = {s.strip() for s in body.source_speaker_ids if s and s.strip()}
    target_id = (body.target_speaker_id or "").strip() or None
    if target_id and target_id in source_set:
        source_set.discard(target_id)
    if not source_set:
        raise HTTPException(status_code=400, detail="No source speakers selected")

    with history.job_lock(job_id):
        job = _get_job_or_404(session, job_id)
        _require_not_running(job, "merge speakers")
        rows = _ordered_rows(session, job_id)
        speakers = json.loads(job.speakers_json or "{}")
        glossary = json.loads(job.glossary_json) if job.glossary_json else None

        changed_rows = any((r.speaker or "") in source_set for r in rows)
        recording = _history_begin(session, job) if changed_rows else None
        for r in rows:
            if (r.speaker or "") in source_set:
                r.speaker = target_id
                session.add(r)

        if target_id and changed_rows and target_id not in speakers:
            idx = len(speakers)
            speakers[target_id] = {
                "label": target_id,
                "gender": "unknown",
                "color": diarization.SPEAKER_PALETTE[idx % len(diarization.SPEAKER_PALETTE)],
            }
        for src in source_set:
            speakers.pop(src, None)

        if glossary and isinstance(glossary.get("characters"), list):
            for ch in glossary["characters"]:
                if isinstance(ch, dict) and str(ch.get("speaker_id") or "").strip() in source_set:
                    ch["speaker_id"] = target_id or ""

        job.speakers_json = json.dumps(speakers, ensure_ascii=False)
        if glossary is not None:
            job.glossary_json = json.dumps(glossary, ensure_ascii=False)
        job.updated_at = _now()
        session.add(job)
        session.commit()
        if recording is not None:
            _history_record(session, job, recording, "diarize", {"merged": len(source_set), "target": target_id})
        session.refresh(job)
        return _job_to_response(job)


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
        (settings.storage_dir / "regions" / f"{job_id}.waveform.json").unlink(missing_ok=True)
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
        if patch.original_text != seg.original_text:
            existing_words = _parse_words_raw(getattr(seg, "words_json", None))
            if existing_words is not None and len(patch.original_text.split()) != len(existing_words):
                seg.words_json = None
        seg.original_text = patch.original_text
    if patch.translated_text is not None:
        new_translation = strip_directional_marks(patch.translated_text)
        if new_translation != (seg.translated_text or "") and getattr(seg, "addressee_json", None):
            # The form the translator reported belongs to the text it wrote, not to the user's
            try:
                listener = json.loads(seg.addressee_json)
                if isinstance(listener, dict) and listener.pop("used", None) is not None:
                    seg.addressee_json = json.dumps(listener, ensure_ascii=False)
            except ValueError:
                pass
        seg.translated_text = new_translation
        if seg.translated_text.strip():
            seg.translation_failed = False
    if "speaker" in patch.model_fields_set:
        seg.speaker = (patch.speaker or "").strip() or None
    if "addressee_gender" in patch.model_fields_set:
        choice = (patch.addressee_gender or "").strip()
        if not choice:
            seg.addressee_json = None
        else:
            manual = _manual_addressee_json(choice)
            if manual is None:
                raise HTTPException(status_code=422, detail=f"Unknown addressee: {patch.addressee_gender}")
            seg.addressee_json = manual

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
        coalesce_key=None if timing_only else f"text:{seg.id}:{','.join(fields)}",
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
        segment_type=kind, confidence=1.0, speaker=(req.speaker or "").strip() or None,
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
    raw_words = _parse_words_raw(getattr(seg, "words_json", None))
    left_words = [w for w in raw_words if float(w["s"]) < split_at] if raw_words else []
    right_words = [w for w in raw_words if float(w["s"]) >= split_at] if raw_words else []

    seg.end_time = split_at
    seg.original_text = orig1 or seg.original_text
    seg.translated_text = (tr1 or seg.translated_text) if has_tr else None
    seg.words_json = json.dumps(left_words, ensure_ascii=False) if left_words else None
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
        words_json=json.dumps(right_words, ensure_ascii=False) if right_words else None,
        speaker=getattr(seg, "speaker", None),
        addressee_json=getattr(seg, "addressee_json", None),
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
    word_chunks = [_parse_words_raw(getattr(t, "words_json", None)) for t in targets if (t.original_text or "").strip()]
    if word_chunks and all(word_chunks):
        first.words_json = json.dumps([w for chunk in word_chunks for w in (chunk or [])], ensure_ascii=False)
    else:
        first.words_json = None
    first.speaker = next((getattr(t, "speaker", None) for t in targets if getattr(t, "speaker", None)), None)
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
        old_start = seg.start_time
        dur = max(0.1, round(seg.end_time - seg.start_time, 3))
        shifted_start = seg.start_time + req.offset_seconds
        new_start = max(0.0, round(shifted_start, 3))
        if shifted_start < 0:
            new_end = round(new_start + dur, 3)
        else:
            new_end = max(round(new_start + 0.1, 3), round(seg.end_time + req.offset_seconds, 3))
        seg.start_time = new_start
        seg.end_time = new_end
        parsed_words = _parse_words_raw(getattr(seg, "words_json", None))
        if parsed_words:
            delta = round(new_start - old_start, 3)
            shifted_words = []
            for w in parsed_words:
                ws = round(max(0.0, float(w["s"]) + delta), 3)
                we = round(max(ws, float(w["e"]) + delta), 3)
                shifted_words.append({"w": w["w"], "s": ws, "e": we})
            seg.words_json = json.dumps(shifted_words, ensure_ascii=False)
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

_MOVE_FIELDS = (
    "original_text", "translated_text", "translation_failed", "segment_type", "confidence", "speaker", "addressee_json",
)
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
        r.words_json = None
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
