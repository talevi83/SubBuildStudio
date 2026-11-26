from datetime import datetime
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict, Field
from app.models.db_models import JobStatus


class RequestModel(BaseModel):
    """Base for request bodies: NaN / Infinity are refused (JSON from Python clients can carry them,
    and every time comparison is False for NaN, so it would slip past the range checks)."""
    model_config = ConfigDict(allow_inf_nan=False)


class AudioStreamInfo(BaseModel):
    index: int = Field(description="Stream index within audio streams (0-based)")
    codec_name: str
    channels: int = 2
    channel_layout: Optional[str] = None
    sample_rate: Optional[str] = None
    language: Optional[str] = "und"
    title: Optional[str] = None


class SubtitleStreamInfo(BaseModel):
    index: int = Field(description="Stream index within subtitle streams (0-based, for -map 0:s:N)")
    codec_name: str
    language: Optional[str] = "und"
    title: Optional[str] = None
    forced: bool = False
    default: bool = False
    text_based: bool = True  # False for image subtitles (PGS/VobSub) which can't be parsed as text

    @property
    def looks_forced(self) -> bool:
        title = (self.title or "").lower()
        return self.forced or "forced" in title or "signs" in title

    @property
    def looks_sdh(self) -> bool:
        title = (self.title or "").lower()
        return any(k in title for k in ("sdh", "hearing", "deaf", "(cc)", "[cc]"))


class ProbePathRequest(RequestModel):
    file_path: str = Field(description="Direct absolute path on the local filesystem")


class ProbeResponse(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    duration: float
    format_name: str
    is_upload: bool = False
    video_streams: List[Dict[str, Any]] = []
    audio_streams: List[AudioStreamInfo] = []
    subtitle_streams: List[SubtitleStreamInfo] = []
    chapters: List[Dict[str, Any]] = []


class JobConfig(RequestModel):
    selected_audio_stream: int = 0
    transcription_provider: str = "faster_whisper"
    whisper_model: str = "medium"
    vocal_isolation: bool = False
    vocal_isolation_mode: Optional[str] = None
    source_language: Optional[str] = "auto"
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    comms_boost: bool = False
    comms_boost_mode: Optional[str] = None
    high_speech_sensitivity: bool = False
    use_embedded_subtitles: bool = True
    selected_subtitle_stream: Optional[int] = None
    supplement_with_whisper: bool = False
    detect_onscreen_text: bool = False
    onscreen_sensitivity: int = Field(default=2, ge=1, le=3)
    allow_onscreen_overlap: bool = True
    onscreen_position: Optional[str] = None
    convert_to_metric: Optional[bool] = None
    remove_sdh_speakers: Optional[bool] = None
    skip_end_credits: Optional[bool] = None
    enable_diarization: bool = False
    auto_save_srt: bool = False
    enable_translation: bool = False
    target_language: str = "Hebrew"
    # None = use the translation model chosen on the Settings screen
    gemini_model: Optional[str] = None
    # Used for this run only; never written to the database.
    gemini_api_key_override: Optional[str] = None
    context_notes: Optional[str] = None
    glossary_profile: Optional[str] = None
    # Feature-flagged Dual-Source (EN + AR) pivot mode
    enable_dual_source_pivot: bool = False
    dual_source_fallback_languages: Optional[List[str]] = None
    dual_source_smart_asr_fallback: bool = True


class SyncCue(RequestModel):
    start: float
    end: float
    text: str
    position: str = "bottom"   # "top" = placed at the top of the picture (signs, titles)
    kind: str = "speech"       # "sound" = hearing-impaired description, nothing is spoken


class ParsedSubtitlesResponse(BaseModel):
    file_name: str
    format: str
    encoding: str
    language: str
    cue_count: int
    first_start: float
    last_end: float
    top_count: int = 0
    sound_count: int = 0
    cues: List[SyncCue]


class JobCreate(JobConfig):
    video_path: str
    file_name: Optional[str] = None
    # "Sync existing subtitles": the parsed file (POST /api/media/parse-subtitles) instead of transcribing
    sync_cues: Optional[List[SyncCue]] = None
    sync_file_name: Optional[str] = None
    sync_language: Optional[str] = None
    sync_use_gemini: bool = True
    # Optional manual .en.srt / .ar.srt overrides for Dual-Source Pivot testing
    dual_source_en_srt: Optional[str] = None
    dual_source_ar_srt: Optional[str] = None


class RetranslateRequest(RequestModel):
    only_failed: bool = False
    target_language: Optional[str] = None
    gemini_model: Optional[str] = None
    context_notes: Optional[str] = None
    gemini_api_key_override: Optional[str] = None
    rebuild_glossary: bool = False
    # Translate only these rows (segment ids) - used with a hint about who is speaking to whom
    segment_ids: Optional[List[int]] = None
    speaker_gender: Optional[str] = Field(default=None, description="male, female or group")
    addressee_gender: Optional[str] = Field(default=None, description="male, female or group")
    hint: Optional[str] = Field(default=None, description="Free-text note for the selected rows only")
    remember_hint: bool = Field(default=False, description="Also keep the hint in the job's context notes")
    thinking: Optional[str] = Field(default=None, description="Thinking level for this run (default, off, minimal, low, medium, high)")


class GlossaryCharacterEdit(RequestModel):
    name: str
    target_name: str = ""
    gender: str = "unknown"
    notes: str = ""
    speaker_id: str = ""


class GlossaryTermEdit(RequestModel):
    source: str
    translation: str = ""
    notes: str = ""


class GlossaryUpdate(RequestModel):
    summary: str = ""
    characters: List[GlossaryCharacterEdit] = []
    terms: List[GlossaryTermEdit] = []
    register_notes: str = ""
    # Name of a series profile to save this glossary to and use from now on; "" detaches the job
    profile: Optional[str] = None


class JobSettingsUpdate(RequestModel):
    onscreen_position: Optional[str] = Field(
        default=None,
        description="Position for on-screen/bracketed subtitles: top_center, middle_right, middle_left, top_right, top_left",
    )


class WordTimingSchema(BaseModel):
    w: str
    s: float
    e: float


class SpeakerProfileSchema(BaseModel):
    label: str
    gender: Literal["male", "female", "unknown"] = "unknown"
    color: str = "#38BDF8"
    character_name: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    evidence: Optional[str] = None


class SpeakersUpdateRequest(RequestModel):
    speakers: Dict[str, SpeakerProfileSchema]


class DiarizeJobRequest(RequestModel):
    min_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    max_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    auto_match: bool = True


class MatchSpeakersRequest(RequestModel):
    overwrite_manual: bool = False
    auto_merge: bool = True


class MergeSpeakersRequest(RequestModel):
    source_speaker_ids: List[str] = Field(min_length=1)
    target_speaker_id: Optional[str] = None


class AddresseeSchema(BaseModel):
    """Who a dialogue line is spoken to, and how sure the app is (see app/pipeline/addressee.py)."""
    gender: str = "unknown"       # male | female | mixed | unknown
    number: str = "unknown"       # singular | plural | unknown
    target: Optional[str] = None  # SPEAKER_XX of the listener
    label: Optional[str] = None
    confidence: float = 0.0
    source: str = ""
    locked: bool = False          # set by the user
    used: Optional[str] = None    # 2nd-person form the translation was written with: m | f | pl


class SubtitleSegmentSchema(BaseModel):
    id: Optional[int] = None
    job_id: str
    index: int
    start_time: float
    end_time: float
    original_text: str
    translated_text: Optional[str] = None
    translation_failed: bool = False
    segment_type: str = "DIALOGUE"
    confidence: Optional[float] = 1.0
    cps: Optional[float] = 0.0
    words: Optional[List[WordTimingSchema]] = None
    speaker: Optional[str] = None
    addressee: Optional[AddresseeSchema] = None


class SegmentUpdate(RequestModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    original_text: Optional[str] = None
    translated_text: Optional[str] = None
    speaker: Optional[str] = None
    # Who the line is spoken to, set by hand: male, female or group. null clears the tag, and the
    # next translation of the line works it out again.
    addressee_gender: Optional[str] = None


class BulkSegmentUpdate(SegmentUpdate):
    id: int


class RetimeRequest(RequestModel):
    """Optional window (seconds); omitted = the whole video."""
    start: Optional[float] = None
    end: Optional[float] = None


class SegmentCreate(RequestModel):
    """A subtitle line the user adds by hand (between existing lines)."""
    start_time: float
    end_time: float
    original_text: str = ""
    translated_text: Optional[str] = None
    segment_type: str = "DIALOGUE"
    speaker: Optional[str] = None


class MoveSegmentRequest(RequestModel):
    segment_id: int
    to_position: int = Field(description="0-based position in time order to drop the row at")


class SplitSegmentRequest(RequestModel):
    split_time: Optional[float] = Field(default=None, description="Optional split timestamp in seconds")


class MergeSegmentsRequest(RequestModel):
    segment_ids: List[int] = Field(min_length=2, description="IDs of 2 or more segments to merge")


class ShiftSegmentsRequest(RequestModel):
    offset_seconds: float = Field(description="Seconds to shift timestamps by (+/-)")
    segment_ids: Optional[List[int]] = Field(default=None, description="Optional segment IDs to shift; None shifts all")


class SegmentsChangeResponse(BaseModel):
    changed: int = 0
    retimed: bool = True
    message: Optional[str] = None
    created_id: Optional[int] = None  # the line POST /segments added
    segments: List["SubtitleSegmentSchema"] = []


class BulkDeleteJobsRequest(RequestModel):
    job_ids: Optional[List[str]] = Field(default=None, description="Job IDs to delete; None or empty with delete_all=True deletes all jobs")
    delete_all: bool = False


class BulkDeleteJobsResponse(BaseModel):
    deleted_count: int
    deleted_ids: List[str]


class DetectSpeakersRequest(RequestModel):
    segment_ids: List[int]
    gemini_model: Optional[str] = None
    thinking: Optional[str] = None
    gemini_api_key_override: Optional[str] = None


class DetectSpeakersResponse(BaseModel):
    speaker_gender: Optional[str] = None
    addressee_gender: Optional[str] = None
    explanation: str = ""


class QAIssueSchema(BaseModel):
    segment_id: int
    index: int
    kind: str
    message: str


class JobResponse(BaseModel):
    id: str
    video_path: str
    file_name: str
    file_size: Optional[int] = None
    is_upload: bool = False
    status: JobStatus
    current_stage: Optional[str] = None
    progress_percent: float = 0.0
    stage_progress_percent: Optional[float] = None
    speed_factor: Optional[float] = None
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    video_duration: Optional[float] = None
    audio_streams: List[AudioStreamInfo] = []
    selected_audio_stream: int = 0
    transcription_provider: str = "faster_whisper"
    whisper_model: str = "medium"
    vocal_isolation: bool = False
    vocal_isolation_mode: str = "off"
    source_language: Optional[str] = "auto"
    detected_language: Optional[str] = None
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    comms_boost: bool = False
    comms_boost_mode: str = "off"
    high_speech_sensitivity: bool = False
    use_embedded_subtitles: bool = True
    selected_subtitle_stream: Optional[int] = None
    supplement_with_whisper: bool = False
    detect_onscreen_text: bool = False
    onscreen_sensitivity: int = Field(default=2, ge=1, le=3)
    allow_onscreen_overlap: bool = True
    onscreen_position: Optional[str] = "top_center"
    convert_to_metric: bool = False
    remove_sdh_speakers: bool = False
    skip_end_credits: bool = True
    enable_diarization: bool = False
    speakers: Optional[Dict[str, SpeakerProfileSchema]] = None
    auto_save_srt: bool = False
    enable_translation: bool = False
    target_language: str = "Hebrew"
    gemini_model: str = "gemini-3.8-flash"
    context_notes: Optional[str] = None
    glossary_profile: Optional[str] = None
    glossary: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    segment_count: int = 0
    failed_translation_count: int = 0
    job_kind: str = "transcribe"
    sync_file_name: Optional[str] = None
    sync_report: Optional[Dict[str, Any]] = None
    enable_dual_source_pivot: bool = False
    dual_source_report: Optional[Dict[str, Any]] = None
    segments: List[SubtitleSegmentSchema] = []
    created_at: datetime
    updated_at: datetime


class WebSocketEventPayload(BaseModel):
    # "STAGE_PROGRESS" | "SEGMENT_CREATED" | "SEGMENTS_REFINED" | "BATCH_TRANSLATED"
    # | "JOB_COMPLETED" | "JOB_FAILED" | "JOB_CANCELLED"
    event: str
    job_id: str
    stage: Optional[str] = None
    progress_percent: float = 0.0
    stage_progress_percent: Optional[float] = None
    speed_factor: Optional[float] = None
    eta_seconds: Optional[float] = None
    segment: Optional[SubtitleSegmentSchema] = None
    segments_batch: Optional[List[SubtitleSegmentSchema]] = None
    message: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class UpdateStatusSchema(BaseModel):
    current_version: str = "1.5.0"
    current_commit: str = ""
    branch: str = ""
    update_available: bool = False
    update_kind: Optional[str] = None
    latest_version: str = "1.5.0"
    latest_commit: str = ""
    commits_behind: int = 0
    target_ref: Optional[str] = None
    release_url: Optional[str] = None
    release_notes: Optional[str] = None
    # "windows" / "macos" / "linux" when the copy came from an installer / app / AppImage, which
    # updates by a new download
    installer: Optional[str] = None
    # Direct link to that installer in the release, once the release workflow has attached it
    download_url: Optional[str] = None
    error: Optional[str] = None


class GpuLibsStatusSchema(BaseModel):
    """The Linux AppImage's optional NVIDIA libraries (app.core.gpu_runtime)."""
    supported: bool = False   # only the AppImage downloads them itself
    installed: bool = False
    installing: bool = False
    error: Optional[str] = None
    download_size_mb: int = 0


class UpdateApplyResponse(BaseModel):
    updated: bool
    previous_version: str
    new_version: str
    new_commit: str = ""
    requirements_updated: bool = False
    frontend_rebuilt: bool = False
    restart_required: bool = False
    skipped_reason: Optional[str] = None
    message: str = ""


class SystemInfoResponse(BaseModel):
    is_gemini_configured: bool
    is_ai_configured: bool = False
    masked_api_key: Optional[str] = None
    gemini_model: str
    gemini_models: List[str] = []
    providers: Dict[str, Any] = {}
    transcription_providers: List[str] = ["faster_whisper", "whisperx"]
    whisperx_available: bool = False
    vocal_isolation_available: bool = False
    whisper_default_model: str
    whisper_models: List[str] = []
    max_cps_warning: float = 20.0
    hardware: Dict[str, Any]
    version: str = "1.5.0"
    update: Optional[UpdateStatusSchema] = None
    dual_source_pivot_enabled: bool = False


class AppConfigResponse(BaseModel):
    """Runtime feature-flag and local tool availability (never exposes API keys)."""
    dual_source_pivot_enabled: bool = False
    opensubtitles_configured: bool = False
    subdl_configured: bool = False
    ffsubsync_available: bool = False
    alass_available: bool = False


class SaveNextToVideoRequest(RequestModel):
    """Write the subtitles into the folder of the job's video (Movie.he.srt next to Movie.mkv)."""
    format: Literal["srt", "vtt", "ass"] = "srt"
    use_translated: bool = True
    apply_rtl: bool = True
    # A file with the same name already next to the video is only replaced when this is true
    overwrite: bool = False


class SaveNextToVideoResponse(BaseModel):
    saved: bool
    path: str
    # True when the file already existed (with overwrite=false nothing was written)
    exists: bool


class EmbedVideoRequest(RequestModel):
    """Make a copy of the job's video with its subtitles inside (a track, or burned into the picture)."""
    # soft: extra subtitle track, no re-encode (MKV/MP4). burn: drawn on the picture, H.264 MP4
    mode: Literal["soft", "burn"] = "soft"
    use_translated: bool = True
    apply_rtl: bool = True
    # soft: make the new track the one players show by default
    set_default: bool = True
    # burn: text size and encode quality (standard = fast)
    font_size: Literal["small", "medium", "large"] = "medium"
    font_family: Optional[str] = Field(default=None, max_length=64)
    font_size_px: Optional[int] = Field(default=None, ge=12, le=36)
    text_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_width: Optional[float] = Field(default=None, ge=0.0, le=6.0)
    background_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    background_opacity: Optional[int] = Field(default=None, ge=0, le=100)
    margin_v: Optional[int] = Field(default=None, ge=8, le=120)
    bold: bool = False
    color_speakers: bool = False
    highlight_mode: Literal["none", "word", "karaoke"] = "none"
    highlight_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    bilingual: bool = False
    quality: Literal["standard", "high"] = "standard"
    # A video with the same name already next to the original is only replaced when this is true
    overwrite: bool = False


class EmbedVideoStatus(BaseModel):
    task_id: Optional[str] = None
    # exists: nothing started because the output file is already there (overwrite=false)
    state: Literal["running", "done", "error", "cancelled", "exists"]
    mode: Literal["soft", "burn"]
    percent: float = 0.0
    # Full path next to the video, or just the file name when it is offered as a download
    path: str
    file_name: str
    # True: written next to the original video. False: kept in app storage, fetch via download_url
    next_to_video: bool
    download_url: Optional[str] = None
    error: Optional[str] = None
    # copy | h264_nvenc | libx264
    encoder: Optional[str] = None


class HistoryEntrySchema(BaseModel):
    id: int
    # original | edit | edit_timing | bulk_edit | insert | delete | delete_many | move | retime | translate | external
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    segment_count: int


class HistoryResponse(BaseModel):
    """The job's edit history, oldest first; `cursor` is the step the subtitles are at now."""
    cursor: Optional[int] = None
    can_undo: bool = False
    can_redo: bool = False
    entries: List[HistoryEntrySchema] = Field(default_factory=list)


class HistoryRestoreResponse(BaseModel):
    segments: List[SubtitleSegmentSchema]
    history: HistoryResponse


class WaveformResponse(BaseModel):
    job_id: str
    peaks_per_second: int = 50
    duration: float
    peaks: List[float]
