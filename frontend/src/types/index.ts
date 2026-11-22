export type JobStatus =
  | 'IDLE'
  | 'QUEUED'
  | 'PROBING'
  | 'EXTRACTING_AUDIO'
  | 'TRANSCRIBING'
  | 'SCANNING_ONSCREEN'
  | 'REFINING'
  | 'TRANSLATING'
  | 'SYNCING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED';

export const ACTIVE_STATUSES: JobStatus[] = [
  'IDLE',
  'QUEUED',
  'PROBING',
  'EXTRACTING_AUDIO',
  'TRANSCRIBING',
  'SCANNING_ONSCREEN',
  'REFINING',
  'TRANSLATING',
  'SYNCING',
];

export const isActiveStatus = (status?: JobStatus | null) =>
  !!status && ACTIVE_STATUSES.includes(status);

export interface AudioStreamInfo {
  index: number;
  codec_name: string;
  channels: number;
  channel_layout?: string;
  sample_rate?: string;
  language?: string;
  title?: string;
}

export interface SubtitleStreamInfo {
  index: number;
  codec_name: string;
  language?: string;
  title?: string | null;
  forced: boolean;
  default: boolean;
  text_based: boolean;
}

export type SegmentType = 'DIALOGUE' | 'COMMS' | 'ON_SCREEN' | 'FOREIGN';
export type OnscreenPosition = 'top_center' | 'top_left' | 'top_right' | 'middle_left' | 'middle_right';
export type ProblemFilter = 'NONE' | 'ALL_PROBLEMS' | 'UNTRANSLATED' | 'LATIN_LEFTOVER' | 'HIGH_CPS';

export interface VideoStreamInfo {
  codec_name: string;
  width?: number;
  height?: number;
  fps?: string;
  duration?: string;
}

export interface ProbeResponse {
  file_path: string;
  file_name: string;
  file_size: number;
  duration: number;
  format_name: string;
  is_upload?: boolean;
  video_streams: VideoStreamInfo[];
  audio_streams: AudioStreamInfo[];
  subtitle_streams?: SubtitleStreamInfo[];
}

export interface SubtitleSegment {
  id?: number;
  job_id: string;
  index: number;
  start_time: number;
  end_time: number;
  original_text: string;
  translated_text?: string | null;
  translation_failed?: boolean;
  segment_type?: SegmentType;
  confidence?: number;
  cps?: number;
}

export interface GlossaryCharacter {
  name: string;
  target_name?: string;
  gender: string;
  notes?: string;
}

export interface Glossary {
  summary?: string;
  characters?: GlossaryCharacter[];
  terms?: { source: string; translation: string; notes?: string }[];
  register_notes?: string;
}

export interface Job {
  id: string;
  video_path: string;
  file_name: string;
  file_size?: number;
  is_upload?: boolean;
  status: JobStatus;
  current_stage?: string | null;
  progress_percent: number;
  speed_factor?: number | null;
  eta_seconds?: number | null;
  error_message?: string | null;
  warning_message?: string | null;
  video_duration?: number | null;
  audio_streams: AudioStreamInfo[];
  selected_audio_stream: number;
  whisper_model: string;
  source_language?: string | null;
  detected_language?: string | null;
  vad_filter: boolean;
  comms_boost: boolean;
  high_speech_sensitivity?: boolean;
  use_embedded_subtitles?: boolean;
  selected_subtitle_stream?: number | null;
  supplement_with_whisper?: boolean;
  detect_onscreen_text: boolean;
  onscreen_sensitivity?: number;
  allow_onscreen_overlap?: boolean;
  onscreen_position?: OnscreenPosition;
  convert_to_metric?: boolean;
  remove_sdh_speakers?: boolean;
  skip_end_credits?: boolean;
  auto_save_srt?: boolean;
  enable_translation: boolean;
  target_language: string;
  gemini_model: string;
  context_notes?: string | null;
  glossary_profile?: string | null;
  glossary?: Glossary | null;
  usage?: JobUsage | null;
  segment_count: number;
  failed_translation_count: number;
  /** 'sync' = the user's own subtitle file re-timed to the video */
  job_kind?: 'transcribe' | 'sync';
  sync_file_name?: string | null;
  sync_report?: SyncReport | null;
  segments: SubtitleSegment[];
  created_at: string;
  updated_at: string;
}

export interface WebSocketEventPayload {
  event:
    | 'STAGE_PROGRESS'
    | 'SEGMENT_CREATED'
    | 'SEGMENTS_REFINED'
    | 'BATCH_TRANSLATED'
    | 'JOB_COMPLETED'
    | 'JOB_FAILED'
    | 'JOB_CANCELLED';
  job_id: string;
  stage?: string;
  progress_percent: number;
  speed_factor?: number | null;
  eta_seconds?: number | null;
  segment?: SubtitleSegment | null;
  segments_batch?: SubtitleSegment[] | null;
  message?: string | null;
  usage?: JobUsage | null;
}

export interface HardwareInfo {
  cuda_available: boolean;
  device_count: number;
  gpu_name?: string | null;
  vram_total_mb?: number | null;
  vram_free_mb?: number | null;
  cpu_count: number;
  recommended_device: string;
  recommended_compute_type: string;
  /** Linux AppImage: an NVIDIA card is there, but its CUDA libraries are not downloaded yet */
  gpu_libs_missing?: boolean;
}

/** The Linux AppImage's optional NVIDIA libraries (GET /api/system/gpu-libs) */
export interface GpuLibsStatus {
  supported: boolean;
  installed: boolean;
  installing: boolean;
  error?: string | null;
  download_size_mb: number;
}

export interface UpdateStatus {
  current_version: string;
  current_commit: string;
  branch: string;
  update_available: boolean;
  update_kind?: 'release' | 'commit' | null;
  latest_version: string;
  latest_commit: string;
  commits_behind: number;
  target_ref?: string | null;
  release_url?: string | null;
  release_notes?: string | null;
  /** 'windows' / 'macos' / 'linux' for a copy installed with the installer / app / AppImage: it updates by a new download */
  installer?: 'windows' | 'macos' | 'linux' | null;
  /** Direct link to that installer in the release (missing until the release build attached it) */
  download_url?: string | null;
  error?: string | null;
}

export interface UpdateApplyResponse {
  updated: boolean;
  previous_version: string;
  new_version: string;
  new_commit: string;
  requirements_updated: boolean;
  frontend_rebuilt: boolean;
  restart_required: boolean;
  skipped_reason?: string | null;
  message: string;
}

export interface SystemInfo {
  is_gemini_configured: boolean;
  is_ai_configured?: boolean;
  masked_api_key?: string | null;
  gemini_model: string;
  gemini_models: string[];
  providers?: Record<string, ProviderSummary>;
  whisper_default_model: string;
  whisper_models: string[];
  max_cps_warning: number;
  hardware: HardwareInfo;
  version?: string;
  update?: UpdateStatus | null;
}

export interface ApiKeyStatus {
  is_gemini_configured: boolean;
  masked_api_key?: string | null;
}

/** One line of a subtitle file the user wants synced (as parsed by the server). */
export interface SyncCue {
  start: number;
  end: number;
  text: string;
  /** 'top' = placed at the top of the picture in the file (signs, titles) */
  position: 'top' | 'bottom';
  /** 'sound' = a hearing-impaired description such as [door slams] */
  kind: 'speech' | 'sound';
}

export interface ParsedSubtitles {
  file_name: string;
  format: string;
  encoding: string;
  language: string;
  cue_count: number;
  first_start: number;
  last_end: number;
  top_count: number;
  sound_count: number;
  cues: SyncCue[];
}

export type SyncMethod = 'text' | 'gemini' | 'screen' | 'burned_in' | 'embedded' | 'speech';

export interface SyncReport {
  file_name?: string;
  language?: string;
  spoken_language?: string;
  lines?: number;
  matching?: string;
  frame_rate?: string;
  offset?: number;
  jump_count?: number;
  reference?: { source: 'embedded' | 'burned_in' | null; detail?: string };
  methods?: Partial<Record<SyncMethod | 'low_confidence', number>>;
}

export type IngestMode = 'transcribe' | 'sync';

export interface JobConfig {
  selected_audio_stream: number;
  whisper_model: string;
  source_language: string;
  vad_filter: boolean;
  comms_boost: boolean;
  high_speech_sensitivity?: boolean;
  use_embedded_subtitles?: boolean;
  selected_subtitle_stream?: number | null;
  supplement_with_whisper?: boolean;
  detect_onscreen_text: boolean;
  onscreen_sensitivity?: number;
  allow_onscreen_overlap?: boolean;
  onscreen_position?: OnscreenPosition;
  convert_to_metric?: boolean;
  remove_sdh_speakers?: boolean;
  skip_end_credits?: boolean;
  auto_save_srt?: boolean;
  enable_translation: boolean;
  target_language: string;
  gemini_api_key_override?: string;
  context_notes?: string;
  /** Series glossary shared between episodes (saved by name) */
  glossary_profile?: string;
}

export interface RetranslateOptions {
  only_failed?: boolean;
  target_language?: string;
  gemini_model?: string;
  context_notes?: string;
  gemini_api_key_override?: string;
  rebuild_glossary?: boolean;
  /** Translate only these rows (segment ids), optionally with a note about who speaks to whom */
  segment_ids?: number[];
  thinking?: ThinkingLevel;
  speaker_gender?: Gender | null;
  addressee_gender?: Gender | null;
  hint?: string;
  remember_hint?: boolean;
}

export type Gender = 'male' | 'female' | 'group';

export type SegmentPatch = Partial<
  Pick<SubtitleSegment, 'start_time' | 'end_time' | 'original_text' | 'translated_text'>
>;

export type StorageCategoryId = 'temp' | 'cache' | 'logs' | 'exports' | 'uploads' | 'jobs';

export interface StorageReport {
  categories: { id: StorageCategoryId; bytes: number; count: number }[];
  database_bytes: number;
}

export interface StorageCleanResult {
  freed_bytes: number;
  removed: Record<string, { bytes: number; count: number }>;
}

export interface SpeakerGuess {
  speaker_gender: Gender | null;
  addressee_gender: Gender | null;
  explanation: string;
}

export interface QAIssue {
  segment_id: number;
  index: number;
  kind: 'latin' | 'name' | 'term' | 'agreement' | 'missing' | 'overlap' | 'timing';
  message: string;
}

export interface SegmentsChange {
  changed: number;
  retimed: boolean;
  message?: string | null;
  /** The line POST /segments added */
  created_id?: number | null;
  segments: SubtitleSegment[];
}

/** A subtitle line the user adds by hand. */
export interface SegmentCreate {
  start_time: number;
  end_time: number;
  original_text: string;
  translated_text?: string | null;
  segment_type?: SegmentType;
}

/** Where the "add a line" dialog opens: at a time (the playhead), or right after a line. */
export interface InsertDraft {
  time: number;
  afterId?: number;
}

export const RTL_LANGUAGES = ['hebrew', 'arabic', 'persian', 'farsi', 'urdu', 'yiddish', 'he', 'ar', 'fa', 'ur', 'yi'];
export const isRtlLanguage = (lang?: string | null) =>
  !!lang && RTL_LANGUAGES.includes(lang.trim().toLowerCase());

// ---------------------------------------------------------------------------
// Settings, model catalog, pricing
// ---------------------------------------------------------------------------

export type AIProvider = 'google' | 'openai' | 'anthropic' | 'ollama';
export type ThinkingLevel = 'default' | 'off' | 'minimal' | 'low' | 'medium' | 'high';
export type MediaResolution = 'low' | 'medium' | 'high';

export interface ProviderKeyRecord {
  id: string;
  provider: AIProvider;
  label: string;
  masked_key?: string | null;
  has_key: boolean;
  base_url?: string | null;
  is_default: boolean;
  source: 'vault' | 'env' | string;
  created_at?: string | null;
  verified_at?: string | null;
  last_error?: string | null;
  discovered_count?: number | null;
}

export interface ProviderSummary {
  provider: AIProvider;
  configured: boolean;
  key_count: number;
  default_key_id?: string | null;
  masked_default_key?: string | null;
  default_base_url?: string | null;
  discovered?: boolean;
}

export interface GeminiCallSettings {
  provider?: AIProvider;
  model: string;
  key_id?: string | null;
  thinking: ThinkingLevel;
  /** Flex service tier: half price, slower, falls back to Standard when Google is busy */
  flex?: boolean;
}

export interface AppSettings {
  translation: GeminiCallSettings;
  glossary: GeminiCallSettings;
  vision: GeminiCallSettings;
  vision_media_resolution: MediaResolution;
  onscreen_position?: OnscreenPosition;
  convert_to_metric?: boolean;
  remove_sdh_speakers?: boolean;
  skip_end_credits?: boolean;
  cost_saver_mode?: boolean;
}

export interface CatalogModel {
  id: string;
  provider?: AIProvider;
  label: string;
  family: string;
  preview: boolean;
  available?: boolean;
  key_ids?: string[];
  is_local?: boolean;
  param_billions?: number | null;
  thinking: ThinkingLevel[];
  default_level: string;
  input_per_m: number | null;
  output_per_m: number | null;
  input_long_per_m?: number | null;
  output_long_per_m?: number | null;
  price_change?: { after: string; input_per_m: number; output_per_m: number } | null;
  episode_cost_usd?: number | null;
  movie_cost_usd?: number | null;
  episode_vision_cost_usd?: number | null;
  movie_vision_cost_usd?: number | null;
  supports_media_resolution: boolean;
  /** Pipeline steps this model is offered for */
  tasks: PipelineTask[];
  recommended_for?: PipelineTask[];
  warnings_by_task?: Partial<Record<PipelineTask, string>>;
  suitability_by_task?: Partial<Record<PipelineTask, 'recommended' | 'suitable' | 'warning' | 'unsuitable'>>;
  supports_flex?: boolean;
  /** Share of the Standard price billed on Flex (0.5), or null when Flex isn't offered */
  flex_price_factor?: number | null;
  rank?: number;
  note?: string;
}

export type PipelineTask = 'translation' | 'glossary' | 'vision';

export interface ModelCheckResult {
  ok: boolean;
  error: string | null;
}

export interface ModelCheckResponse {
  /** Keyed by model id */
  results: Record<string, ModelCheckResult>;
  /** Keyed by pipeline step (the vision probe also sends an image) */
  steps?: Partial<Record<PipelineTask, ModelCheckResult & { model: string }>>;
}

export interface SettingsResponse {
  settings: AppSettings;
  catalog: CatalogModel[];
  providers?: Record<string, ProviderSummary>;
  keys?: ProviderKeyRecord[];
  pricing_source: string;
  pricing_checked: string;
}

export interface ProviderKeysResponse {
  keys: ProviderKeyRecord[];
  providers: Record<string, ProviderSummary>;
}

export interface ProviderKeyMutationResponse {
  key?: ProviderKeyRecord;
  deleted?: boolean;
  keys: ProviderKeyRecord[];
  providers: Record<string, ProviderSummary>;
  catalog: CatalogModel[];
  settings: AppSettings;
  discovery_error?: string | null;
}

export interface RefreshProvidersResponse {
  settings: AppSettings;
  catalog: CatalogModel[];
  providers: Record<string, ProviderSummary>;
  keys: ProviderKeyRecord[];
  errors: Record<string, string>;
  discovered_counts: Record<string, number>;
}

export interface OllamaStatus {
  installed: boolean;
  running: boolean;
  models: string[];
  selected_model: string | null;
  recommended_model: string;
  download_url: string;
  installing: boolean;
  step: string | null;
  error: string | null;
}

export type GeminiStage = 'glossary' | 'translation' | 'onscreen' | 'foreign';

export interface StageEstimate {
  model: string;
  thinking: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  thinking_tokens: number;
  cost_usd: number | null;
  flex?: boolean;
  frames?: number;
}

export interface CostEstimate {
  duration_seconds: number;
  segments: number;
  stages: Partial<Record<GeminiStage, StageEstimate>>;
  total_cost_usd: number | null;
}

export interface StageUsage {
  model: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  thinking_tokens: number;
  cost_usd: number | null;
  flex_requests?: number;
  flex_fallbacks?: number;
}

export interface ModelUsage extends StageUsage {
  total_tokens: number;
}

export interface LocalUsage {
  model: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens?: number;
  segments_tagged: number;
  frames_filtered: number;
  saved_tokens: number;
  saved_cost_usd: number;
}

export interface JobUsage {
  stages: Partial<Record<GeminiStage | 'sync', StageUsage>>;
  models?: Record<string, ModelUsage>;
  local?: LocalUsage | null;
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_thinking_tokens?: number;
  total_tokens?: number;
  total_cost_usd: number | null;
}

/** POST /api/jobs/{id}/export/save-next-to-video */
export type EmbedMode = 'soft' | 'burn';
export type EmbedFontSize = 'small' | 'medium' | 'large';
export type EmbedQuality = 'standard' | 'high';

export interface EmbedVideoOptions {
  /** soft: extra subtitle track (no re-encode). burn: drawn on the picture, H.264 MP4 */
  mode: EmbedMode;
  use_translated?: boolean;
  apply_rtl?: boolean;
  set_default?: boolean;
  font_size?: EmbedFontSize;
  quality?: EmbedQuality;
  overwrite?: boolean;
}

export interface EmbedVideoStatus {
  task_id: string | null;
  /** exists: nothing started, the output file is already there (send overwrite to replace it) */
  state: 'running' | 'done' | 'error' | 'cancelled' | 'exists';
  mode: EmbedMode;
  percent: number;
  /** Full path next to the video, or just the file name for an uploaded video's copy */
  path: string;
  file_name: string;
  next_to_video: boolean;
  /** Set when an uploaded video's copy is ready to download */
  download_url: string | null;
  error: string | null;
  /** copy | h264_nvenc | libx264 */
  encoder: string | null;
}

export interface SaveNextToVideoResult {
  saved: boolean;
  path: string;
  /** The file already existed; with overwrite=false nothing was written */
  exists: boolean;
}

export type HistoryAction =
  | 'original'
  | 'edit'
  | 'edit_timing'
  | 'bulk_edit'
  | 'insert'
  | 'delete'
  | 'delete_many'
  | 'split'
  | 'merge'
  | 'shift_time'
  | 'move'
  | 'retime'
  | 'translate'
  | 'external';

/** One step of a project's edit history (the subtitles as they were after it) */
export interface HistoryEntry {
  id: number;
  action: HistoryAction | string;
  details: { index?: number; count?: number; from?: number; to?: number; scope?: string; fields?: string[] };
  created_at: string;
  segment_count: number;
}

/** GET /api/jobs/{id}/history — oldest first; `cursor` is the step the subtitles are at now */
export interface EditHistory {
  cursor: number | null;
  can_undo: boolean;
  can_redo: boolean;
  entries: HistoryEntry[];
}

export interface HistoryRestoreResult {
  segments: SubtitleSegment[];
  history: EditHistory;
}

export interface WaveformData {
  job_id: string;
  peaks_per_second: number;
  duration: number;
  peaks: number[];
}

