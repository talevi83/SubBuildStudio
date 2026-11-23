import { create } from 'zustand';
import { api } from '../services/api';
import { isActiveStatus, isRtlLanguage } from '../types';
import type {
  Job,
  JobStatus,
  SystemInfo,
  ProbeResponse,
  JobConfig,
  SubtitleSegment,
  SegmentPatch,
  Glossary,
  RetranslateOptions,
  WebSocketEventPayload,
  IngestMode,
  ParsedSubtitles,
  InsertDraft,
  SegmentCreate,
  ProblemFilter,
  EditHistory,
  HistoryRestoreResult,
  SpeakerProfile,
} from '../types';
import { getItemWithLegacy } from '../utils/legacyStorage';
import { naturalSortBy } from '../utils/desktop';
import { manualAddressee } from '../components/studio/table/addressee';
import type { MessageKey } from '../i18n';
import { buildSearchPattern, DEFAULT_SEARCH_OPTIONS, planReplacements, type SearchOptions } from '../utils/textSearch';

const CONFIG_STORAGE_KEY = 'subbuild.pipelineConfig';

const DEFAULT_CONFIG: JobConfig = {
  selected_audio_stream: 0,
  whisper_model: 'medium',
  source_language: 'auto',
  vad_filter: true,
  comms_boost: false,
  high_speech_sensitivity: false,
  use_embedded_subtitles: true,
  selected_subtitle_stream: null,
  supplement_with_whisper: false,
  detect_onscreen_text: false,
  onscreen_sensitivity: 2,
  allow_onscreen_overlap: true,
  onscreen_position: 'top_center',
  convert_to_metric: false,
  remove_sdh_speakers: false,
  skip_end_credits: true,
  enable_diarization: false,
  auto_save_srt: false,
  enable_translation: true,
  target_language: 'Hebrew',
  gemini_api_key_override: '',
  context_notes: '',
};

/** Remembered pipeline settings (never the API key). Storage can be unavailable — fail soft. */
function loadSavedConfig(): Partial<JobConfig> {
  try {
    const raw = getItemWithLegacy(CONFIG_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    // Older versions stored the Gemini model here; it now lives in Settings
    delete parsed.gemini_model;
    // The series belongs to one video, never a default for the next one (older versions saved it)
    delete parsed.glossary_profile;
    return parsed;
  } catch {
    return {};
  }
}

function saveConfig(config: JobConfig) {
  try {
    const {
      gemini_api_key_override: _key,
      selected_audio_stream: _stream,
      selected_subtitle_stream: _subStream,
      context_notes: _notes,
      glossary_profile: _series,
      ...rest
    } = config;
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(rest));
  } catch {
    /* ignore */
  }
}

const savedConfig = loadSavedConfig();

const SYNC_PREFS_KEY = 'subbuild.syncPrefs';

interface SyncPrefs {
  ingestMode: IngestMode;
  syncUseGemini: boolean;
}

function loadSyncPrefs(): SyncPrefs {
  try {
    const parsed = JSON.parse(getItemWithLegacy(SYNC_PREFS_KEY) || '{}');
    return {
      ingestMode: parsed.ingestMode === 'sync' ? 'sync' : 'transcribe',
      syncUseGemini: parsed.syncUseGemini !== false,
    };
  } catch {
    return { ingestMode: 'transcribe', syncUseGemini: true };
  }
}

function saveSyncPrefs(prefs: SyncPrefs) {
  try {
    localStorage.setItem(SYNC_PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* ignore */
  }
}

const savedSyncPrefs = loadSyncPrefs();

interface SeekRequest {
  time: number;
  nonce: number;
  play: boolean;
}

interface JobState {
  systemInfo: SystemInfo | null;
  jobs: Job[];
  activeJob: Job | null;
  probeResult: ProbeResponse | null;
  batchProbes: ProbeResponse[];
  pipelineConfig: JobConfig;
  wsConnected: boolean;
  stageMessage: string | null;
  videoCurrentTime: number;
  selectedSegmentIndex: number | null;
  problemFilter: ProblemFilter;
  setProblemFilter: (filter: ProblemFilter) => void;
  speakerFilter: string;
  setSpeakerFilter: (speaker: string) => void;
  seekRequest: SeekRequest | null;
  subtitleOverlayMode: 'original' | 'translated' | 'dual';
  applyRtlExport: boolean;
  isLoading: boolean;
  error: string | null;
  timingBusy: boolean;
  timingNote: string | null;
  savingCount: number;
  lastSavedAt: number | null;
  /** Edit history of the active job (undo / redo / jump to a step), see GET /jobs/{id}/history */
  history: (EditHistory & { jobId: string }) | null;
  historyBusy: boolean;
  refreshHistory: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  /** Jump to any step; later steps stay available until the next edit */
  restoreHistory: (entryId: number) => Promise<void>;
  /** The "add a subtitle line" dialog, when open */
  insertDraft: InsertDraft | null;
  openInsert: (draft: InsertDraft) => void;
  closeInsert: () => void;
  /** Add a line; resolves to an error message, or null when it was added */
  addSegment: (body: SegmentCreate) => Promise<string | null>;
  /** Delete a line; resolves to an error message, or null when it was deleted */
  deleteSegment: (segmentId: number) => Promise<string | null>;
  /** Delete multiple lines; resolves to an error message, or null when they were deleted */
  deleteSegments: (segmentIds: number[]) => Promise<string | null>;
  /** Split one line into two consecutive lines */
  splitSegment: (segmentId: number, splitTime?: number) => Promise<string | null>;
  /** Merge 2+ lines into one combined line */
  mergeSegments: (segmentIds: number[]) => Promise<string | null>;
  /** A short confirmation shown at the bottom (e.g. "Line #4 deleted"), optionally with Undo */
  notice: Notice | null;
  clearNotice: () => void;
  /** Shift all or selected lines by offsetSeconds */
  shiftSegments: (offsetSeconds: number, segmentIds?: number[]) => Promise<string | null>;
  /** Update onscreen_position on the active job and persist it */
  updateJobOnscreenPosition: (pos: import('../types').OnscreenPosition) => Promise<void>;
  /** New project: make subtitles from the audio, or sync a subtitle file the user already has */
  ingestMode: IngestMode;
  syncFile: ParsedSubtitles | null;
  /** The raw uploaded File for re-decoding when the user changes the subtitle encoding */
  syncRawFile: File | null;
  /** Language of the subtitle file (detected; the user may correct it) */
  syncLanguage: string | null;
  syncUseGemini: boolean;
  setIngestMode: (mode: IngestMode) => void;
  setSyncFile: (file: ParsedSubtitles | null, rawFile?: File | null) => void;
  setSyncLanguage: (language: string) => void;
  setSyncUseGemini: (value: boolean) => void;
  /** Flush any debounced inline text edits and wait for all in-flight segment saves to finish */
  flushPendingEdits: () => Promise<void>;
  /** Apply global defaults loaded from or saved in Settings */
  applySettingsDefaults: (
    defaults: Pick<
      import('../types').AppSettings,
      'convert_to_metric' | 'remove_sdh_speakers' | 'skip_end_credits' | 'onscreen_position'
    >,
    overwriteSaved?: boolean
  ) => void;

  fetchSystemInfo: () => Promise<void>;
  fetchJobs: () => Promise<void>;
  selectJob: (jobId: string) => Promise<void>;
  refreshActiveJob: () => Promise<void>;
  clearActiveJob: () => void;
  setProbeResult: (res: ProbeResponse | null) => void;
  setBatchProbes: (probes: ProbeResponse[]) => void;
  addBatchProbes: (probes: ProbeResponse[]) => void;
  removeBatchProbe: (filePath: string) => void;
  updatePipelineConfig: (config: Partial<JobConfig>) => void;
  createAndStartJob: () => Promise<Job | null>;
  cancelActiveJob: () => Promise<void>;
  cancelJobById: (jobId: string) => Promise<void>;
  cancelAllActiveJobs: () => Promise<void>;
  retranslateActiveJob: (options: RetranslateOptions) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;
  deleteJobs: (jobIds: string[]) => Promise<void>;
  deleteAllJobs: () => Promise<void>;
  previewSegmentTiming: (segmentId: number, patch: { start_time: number; end_time: number }) => void;
  updateSegment: (segmentId: number, patch: SegmentPatch) => Promise<void>;
  /** Replace in every line; `options` is the search options, or just "use regex" (older callers) */
  replaceAll: (
    search: string,
    replacement: string,
    column: 'original' | 'translated',
    options?: boolean | Partial<SearchOptions>
  ) => Promise<number>;
  saveGlossary: (glossary: Glossary, profile?: string) => Promise<boolean>;
  saveSpeakers: (speakers: Record<string, SpeakerProfile>) => Promise<boolean>;
  diarizeActiveJob: (opts?: { min_speakers?: number; max_speakers?: number; threshold?: number; auto_match?: boolean }) => Promise<boolean>;
  matchSpeakersActiveJob: (opts?: { overwrite_manual?: boolean; auto_merge?: boolean }) => Promise<boolean>;
  mergeSpeakersActiveJob: (sourceSpeakerIds: string[], targetSpeakerId: string | null) => Promise<boolean>;
  bulkUpdateSpeaker: (segmentIds: number[], speaker: string | null) => Promise<void>;
  retimeActiveJob: () => Promise<void>;
  moveSegment: (segmentId: number, toPosition: number) => Promise<void>;
  setVideoCurrentTime: (time: number) => void;
  setSelectedSegmentIndex: (index: number | null) => void;
  seekTo: (time: number, play?: boolean) => void;
  setSubtitleOverlayMode: (mode: 'original' | 'translated' | 'dual') => void;
  setApplyRtlExport: (val: boolean) => void;
  clearError: () => void;
  connectWebSocket: (jobId: string, isReconnect?: boolean) => void;
  disconnectWebSocket: () => void;
}

export interface Notice {
  id: number;
  key: MessageKey;
  vars?: Record<string, string | number>;
  /** Offer "Undo" (the change is one step of the edit history) */
  undo?: boolean;
}

let activeSocket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
/** Segment saves still on their way: undo and export wait for them, so a late autosave can't race */
const inflightSaves = new Set<Promise<unknown>>();
const pendingFlushCallbacks = new Set<() => void>();
const pendingDirtyChecks = new Set<() => boolean>();

let historyRefreshTimer: ReturnType<typeof setTimeout> | null = null;

/** Register a synchronous flush callback from an EditableText cell so Export / Undo can flush before saving. */
export const registerPendingFlush = (fn: () => void, isDirty?: () => boolean): (() => void) => {
  pendingFlushCallbacks.add(fn);
  if (isDirty) pendingDirtyChecks.add(isDirty);
  return () => {
    pendingFlushCallbacks.delete(fn);
    if (isDirty) pendingDirtyChecks.delete(isDirty);
  };
};

/** Flush pending inline edits right away (no waiting), e.g. when the window is being closed. */
export const flushPendingEditsNow = (): void => {
  for (const flush of [...pendingFlushCallbacks]) flush();
};

const flushAllPendingEdits = async () => {
  for (const flush of [...pendingFlushCallbacks]) {
    flush();
  }
  if (inflightSaves.size > 0) {
    await Promise.allSettled([...inflightSaves]);
  }
};

/** Apply updates from the server: by row id when both sides have one (line numbers can change), else by number. */
export const mergeSegments = (segments: SubtitleSegment[], updates: SubtitleSegment[]): SubtitleSegment[] => {
  const byId = new Map(updates.filter((u) => u.id != null).map((u) => [u.id!, u]));
  const byIndex = new Map(updates.filter((u) => u.id == null).map((u) => [u.index, u]));
  return segments.map((s) => {
    const update = (s.id != null ? byId.get(s.id) : undefined) ?? byIndex.get(s.index);
    // The row keeps its own number and times; a batch only carries the translation result
    return update ? { ...s, ...update, index: s.index, start_time: s.start_time, end_time: s.end_time } : s;
  });
};

/** True while an inline edit is waiting to be saved or a save is on its way. */
export const hasUnsavedEdits = (): boolean => inflightSaves.size > 0 || [...pendingDirtyChecks].some((isDirty) => isDirty());

export const useJobStore = create<JobState>((set, get) => {
  /** Apply a change to the active job and mirror status/progress into the sidebar list. */
  const patchActiveJob = (patch: Partial<Job>) => {
    const { activeJob, jobs } = get();
    if (!activeJob) return;
    const updated = { ...activeJob, ...patch };
    set({
      activeJob: updated,
      jobs: jobs.map((j) =>
        j.id === updated.id
          ? {
              ...j,
              status: updated.status,
              progress_percent: updated.progress_percent,
              segment_count: updated.segments.length || j.segment_count,
            }
          : j
      ),
    });
  };

  /** Refresh the history list shortly after an edit (several quick edits -> one request). */
  const scheduleHistoryRefresh = (delay = 250) => {
    if (historyRefreshTimer) clearTimeout(historyRefreshTimer);
    historyRefreshTimer = setTimeout(() => {
      historyRefreshTimer = null;
      void get().refreshHistory();
    }, delay);
  };

  const runHistoryAction = async (action: (jobId: string) => Promise<HistoryRestoreResult>) => {
    const { activeJob, historyBusy, timingBusy } = get();
    if (!activeJob || historyBusy || timingBusy || isActiveStatus(activeJob.status)) return;
    const jobId = activeJob.id;
    set({ historyBusy: true });
    try {
      await flushAllPendingEdits();
      const res = await action(jobId);
      if (get().activeJob?.id !== jobId) return;
      patchActiveJob({ segments: res.segments });
      set({ history: { ...res.history, jobId }, timingNote: null });
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : String(e) });
      void get().refreshHistory();
    } finally {
      set({ historyBusy: false });
    }
  };

  const handleMessage = (data: WebSocketEventPayload) => {
    const { activeJob } = get();
    if (!activeJob || activeJob.id !== data.job_id) return;

    switch (data.event) {
      case 'STAGE_PROGRESS': {
        const nextStatus = (data.stage as JobStatus) || activeJob.status;
        const stageChanged = nextStatus !== activeJob.status;
        const nextStagePct =
          nextStatus === 'QUEUED' || nextStatus === 'IDLE'
            ? null
            : typeof data.stage_progress_percent === 'number'
              ? data.stage_progress_percent
              : stageChanged
                ? 0
                : activeJob.stage_progress_percent;
        patchActiveJob({
          status: nextStatus,
          progress_percent: data.progress_percent ?? activeJob.progress_percent,
          stage_progress_percent: nextStagePct,
          speed_factor: data.speed_factor ?? activeJob.speed_factor,
          eta_seconds: data.eta_seconds ?? activeJob.eta_seconds,
          ...(data.usage ? { usage: data.usage } : {}),
        });
        if (data.message) {
          set({ stageMessage: data.message });
        } else if (stageChanged) {
          set({ stageMessage: null });
        }
        break;
      }
      case 'SEGMENT_CREATED': {
        if (!data.segment) break;
        const seg = data.segment;
        const exists = activeJob.segments.some((s) => s.index === seg.index);
        const segments = exists
          ? activeJob.segments.map((s) => (s.index === seg.index ? { ...s, ...seg } : s))
          : [...activeJob.segments, seg];
        patchActiveJob({
          status: 'TRANSCRIBING',
          progress_percent: data.progress_percent || activeJob.progress_percent,
          stage_progress_percent:
            typeof data.stage_progress_percent === 'number'
              ? data.stage_progress_percent
              : activeJob.stage_progress_percent,
          speed_factor: data.speed_factor ?? activeJob.speed_factor,
          eta_seconds: data.eta_seconds ?? activeJob.eta_seconds,
          segments,
          ...(data.usage ? { usage: data.usage } : {}),
        });
        break;
      }
      case 'SEGMENTS_REFINED': {
        // Raw Whisper segments are replaced by the final, editable (id-bearing) subtitles
        patchActiveJob({
          segments: data.segments_batch || [],
          progress_percent: data.progress_percent || activeJob.progress_percent,
          ...(data.usage ? { usage: data.usage } : {}),
        });
        break;
      }
      case 'BATCH_TRANSLATED': {
        if (!data.segments_batch) break;
        patchActiveJob({
          segments: mergeSegments(activeJob.segments, data.segments_batch),
          ...(data.usage ? { usage: data.usage } : {}),
        });
        break;
      }
      case 'JOB_COMPLETED':
      case 'JOB_FAILED':
      case 'JOB_CANCELLED': {
        patchActiveJob({
          stage_progress_percent: null,
          ...(data.usage ? { usage: data.usage } : {}),
        });
        set({ stageMessage: null });
        get().refreshActiveJob();
        get().fetchJobs();
        // A retranslation becomes a history step as the run ends (a failed run records it just after)
        scheduleHistoryRefresh(800);
        break;
      }
    }
  };

  return {
    systemInfo: null,
    jobs: [],
    activeJob: null,
    probeResult: null,
    batchProbes: [],
    pipelineConfig: { ...DEFAULT_CONFIG, ...savedConfig },
    wsConnected: false,
    stageMessage: null,
    videoCurrentTime: 0,
    selectedSegmentIndex: null,
    problemFilter: 'NONE',
    setProblemFilter: (problemFilter) => set({ problemFilter }),
    speakerFilter: 'ALL',
    setSpeakerFilter: (speakerFilter) => set({ speakerFilter }),
    seekRequest: null,
    subtitleOverlayMode: 'dual',
    applyRtlExport: true,
    isLoading: false,
    error: null,
    timingBusy: false,
    timingNote: null,
    savingCount: 0,
    lastSavedAt: null,
    history: null,
    historyBusy: false,
    refreshHistory: async () => {
      const id = get().activeJob?.id;
      if (!id) return;
      try {
        const h = await api.getHistory(id);
        if (get().activeJob?.id === id) set({ history: { ...h, jobId: id } });
      } catch {
        /* the history list is a convenience; editing keeps working */
      }
    },
    undo: () => runHistoryAction((id) => api.undo(id)),
    redo: () => runHistoryAction((id) => api.redo(id)),
    restoreHistory: (entryId) => runHistoryAction((id) => api.restoreHistory(id, entryId)),
    notice: null,
    clearNotice: () => set({ notice: null }),
    insertDraft: null,
    openInsert: (draft) => {
      const job = get().activeJob;
      // Only on a finished job whose lines are all saved (not the live preview while it runs)
      if (!job || isActiveStatus(job.status) || job.segments.some((s) => !s.id)) return;
      set({ insertDraft: draft });
    },
    closeInsert: () => set({ insertDraft: null }),
    addSegment: async (body) => {
      const job = get().activeJob;
      if (!job) return 'No project is open';
      try {
        const res = await api.createSegment(job.id, body);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        const created = res.segments.find((s) => s.id === res.created_id);
        set({ insertDraft: null, selectedSegmentIndex: created?.index ?? null });
        scheduleHistoryRefresh();
        if (created) get().seekTo(Math.max(0, created.start_time - 0.5));
        return null;
      } catch (e: unknown) {
        return e instanceof Error ? e.message : String(e);
      }
    },
    deleteSegment: async (segmentId) => {
      const job = get().activeJob;
      if (!job) return 'No project is open';
      const deletedIndex = job.segments.find((s) => s.id === segmentId)?.index;
      try {
        await flushAllPendingEdits();
        const res = await api.deleteSegment(job.id, segmentId);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        set({
          selectedSegmentIndex: null,
          notice: { id: Date.now(), key: 'ux.rowDeleted', vars: { index: deletedIndex ?? '' }, undo: true },
        });
        scheduleHistoryRefresh();
        return null;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        set({ error: msg });
        return msg;
      }
    },
    deleteSegments: async (segmentIds) => {
      const job = get().activeJob;
      if (!job || segmentIds.length === 0) return 'No project is open';
      try {
        await flushAllPendingEdits();
        const res = await api.deleteSegments(job.id, segmentIds);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        set({
          selectedSegmentIndex: null,
          notice: { id: Date.now(), key: 'ux.rowsDeleted', vars: { count: res.changed }, undo: true },
        });
        scheduleHistoryRefresh();
        return null;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        set({ error: msg });
        return msg;
      }
    },
    splitSegment: async (segmentId, splitTime) => {
      const job = get().activeJob;
      if (!job) return 'No project is open';
      try {
        await flushAllPendingEdits();
        const res = await api.splitSegment(job.id, segmentId, splitTime);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        const created = res.segments.find((s) => s.id === res.created_id);
        if (created) set({ selectedSegmentIndex: created.index });
        scheduleHistoryRefresh();
        return null;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        set({ error: msg });
        return msg;
      }
    },
    mergeSegments: async (segmentIds) => {
      const job = get().activeJob;
      if (!job || segmentIds.length < 2) return 'Select at least 2 lines to merge';
      try {
        await flushAllPendingEdits();
        const res = await api.mergeSegments(job.id, segmentIds);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        const merged = res.segments.find((s) => s.id === res.created_id);
        set({ selectedSegmentIndex: merged?.index ?? null });
        scheduleHistoryRefresh();
        return null;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        set({ error: msg });
        return msg;
      }
    },
    shiftSegments: async (offsetSeconds, segmentIds) => {
      const job = get().activeJob;
      if (!job) return 'No project is open';
      try {
        await flushAllPendingEdits();
        const res = await api.shiftSegments(job.id, offsetSeconds, segmentIds);
        if (get().activeJob?.id !== job.id) return null;
        patchActiveJob({ segments: res.segments });
        scheduleHistoryRefresh();
        return null;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        set({ error: msg });
        return msg;
      }
    },
    updateJobOnscreenPosition: async (pos) => {
      const job = get().activeJob;
      get().updatePipelineConfig({ onscreen_position: pos });
      if (!job) return;
      patchActiveJob({ onscreen_position: pos });
      try {
        await api.updateJobSettings(job.id, { onscreen_position: pos });
      } catch (e: unknown) {
        set({ error: e instanceof Error ? e.message : String(e) });
      }
    },
    ingestMode: savedSyncPrefs.ingestMode,
    syncFile: null,
    syncRawFile: null,
    syncLanguage: null,
    syncUseGemini: savedSyncPrefs.syncUseGemini,

    setIngestMode: (ingestMode) => {
      saveSyncPrefs({ ingestMode, syncUseGemini: get().syncUseGemini });
      set({ ingestMode });
    },
    setSyncFile: (syncFile, rawFile) =>
      set((state) => ({
        syncFile,
        syncRawFile: syncFile === null ? null : rawFile !== undefined ? rawFile : state.syncRawFile,
        syncLanguage: syncFile?.language || null,
      })),
    setSyncLanguage: (syncLanguage) => set({ syncLanguage }),
    setSyncUseGemini: (syncUseGemini) => {
      saveSyncPrefs({ ingestMode: get().ingestMode, syncUseGemini });
      set({ syncUseGemini });
    },
    flushPendingEdits: () => flushAllPendingEdits(),
    applySettingsDefaults: (defaults, overwriteSaved = false) => {
      const cfg = get().pipelineConfig;
      const next: JobConfig = {
        ...cfg,
        onscreen_position:
          overwriteSaved || savedConfig.onscreen_position === undefined
            ? (defaults.onscreen_position ?? cfg.onscreen_position)
            : cfg.onscreen_position,
        convert_to_metric:
          overwriteSaved || savedConfig.convert_to_metric === undefined
            ? (defaults.convert_to_metric ?? cfg.convert_to_metric)
            : cfg.convert_to_metric,
        remove_sdh_speakers:
          overwriteSaved || savedConfig.remove_sdh_speakers === undefined
            ? (defaults.remove_sdh_speakers ?? cfg.remove_sdh_speakers)
            : cfg.remove_sdh_speakers,
        skip_end_credits:
          overwriteSaved || savedConfig.skip_end_credits === undefined
            ? (defaults.skip_end_credits ?? cfg.skip_end_credits)
            : cfg.skip_end_credits,
      };
      if (overwriteSaved) saveConfig(next);
      set({ pipelineConfig: next });
    },

    fetchSystemInfo: async () => {
      try {
        const info = await api.getSystemInfo();
        const cfg = get().pipelineConfig;
        set({
          systemInfo: info,
          pipelineConfig: {
            ...cfg,
            // Saved choice wins; otherwise use the server default
            whisper_model: savedConfig.whisper_model || info.whisper_default_model || cfg.whisper_model,
          },
        });
      } catch (e) {
        console.error('Failed to fetch system info:', e);
      }
    },

    fetchJobs: async () => {
      try {
        const jobs = await api.listJobs();
        set({ jobs });
      } catch (e) {
        console.error('Failed to list jobs:', e);
      }
    },

    selectJob: async (jobId: string) => {
      if (get().activeJob?.id === jobId) return;
      try {
        set({ isLoading: true, error: null });
        const job = await api.getJob(jobId);
        const hasTranslation = job.segments.some((s) => Boolean(s.translated_text));
        const isRtlSource = isRtlLanguage(job.detected_language || job.source_language);
        const untranslatedRtl = !hasTranslation && isRtlSource;
        set({
          activeJob: job,
          history: null,
          isLoading: false,
          notice: null,
          insertDraft: null,
          selectedSegmentIndex: null,
          problemFilter: 'NONE',
          speakerFilter: 'ALL',
          videoCurrentTime: 0,
          seekRequest: null,
          stageMessage: null,
          savingCount: 0,
          lastSavedAt: null,
          ...(untranslatedRtl
            ? {
                applyRtlExport: true,
                ...(get().subtitleOverlayMode === 'translated' ? { subtitleOverlayMode: 'original' as const } : {}),
              }
            : {}),
        });
        get().connectWebSocket(jobId);
        void get().refreshHistory();
      } catch (e: any) {
        set({ error: e.message, isLoading: false });
      }
    },

    refreshActiveJob: async () => {
      const id = get().activeJob?.id;
      if (!id) return;
      try {
        const fresh = await api.getJob(id);
        const current = get().activeJob;
        if (!current || current.id !== id) return;
        let nextSegments = fresh.segments;
        if (
          isActiveStatus(fresh.status) &&
          current.segments.some((s) => !s.id) &&
          fresh.segments.every((s) => !s.id) &&
          fresh.segments.length < current.segments.length
        ) {
          const merged = new Map(current.segments.map((s) => [s.index, s]));
          for (const seg of fresh.segments) merged.set(seg.index, seg);
          nextSegments = [...merged.values()].sort((a, b) => a.index - b.index);
        }
        const hasTranslation = nextSegments.some((s) => Boolean(s.translated_text));
        const isRtlSource = isRtlLanguage(fresh.detected_language || fresh.source_language);
        const untranslatedRtl = !isActiveStatus(fresh.status) && nextSegments.length > 0 && !hasTranslation && isRtlSource;
        const stage_progress_percent =
          typeof fresh.stage_progress_percent === 'number'
            ? fresh.stage_progress_percent
            : current.status === fresh.status &&
                isActiveStatus(fresh.status) &&
                fresh.status !== 'QUEUED' &&
                fresh.status !== 'IDLE' &&
                typeof current.stage_progress_percent === 'number'
              ? current.stage_progress_percent
              : null;
        set({
          activeJob: { ...fresh, stage_progress_percent, segments: nextSegments },
          ...(untranslatedRtl
            ? {
                applyRtlExport: true,
                ...(get().subtitleOverlayMode === 'translated' ? { subtitleOverlayMode: 'original' as const } : {}),
              }
            : {}),
        });
      } catch (e) {
        console.error('Failed to refresh job:', e);
      }
    },

    clearActiveJob: () => {
      get().disconnectWebSocket();
      set({
        activeJob: null,
        history: null,
        notice: null,
        probeResult: null,
        batchProbes: [],
        insertDraft: null,
        syncFile: null,
        syncRawFile: null,
        syncLanguage: null,
        selectedSegmentIndex: null,
        problemFilter: 'NONE',
        speakerFilter: 'ALL',
        seekRequest: null,
        stageMessage: null,
        savingCount: 0,
        lastSavedAt: null,
      });
    },

    setProbeResult: (probeResult) => {
      set((state) => ({
        probeResult,
        batchProbes: probeResult ? [probeResult] : [],
        pipelineConfig: {
          ...state.pipelineConfig,
          selected_audio_stream: 0,
          selected_subtitle_stream: null,
          // A new video starts without a series glossary until the user picks one
          glossary_profile: '',
        },
      }));
    },

    setBatchProbes: (probes) => {
      const sorted = probes.length > 1 ? naturalSortBy(probes, (p) => p.file_name) : [...probes];
      set((state) => ({
        batchProbes: sorted,
        probeResult: sorted[0] ?? null,
        pipelineConfig: {
          ...state.pipelineConfig,
          selected_audio_stream: 0,
          selected_subtitle_stream: null,
          glossary_profile: '',
        },
      }));
    },

    addBatchProbes: (probes) => {
      if (probes.length === 0) return;
      set((state) => {
        const base = state.batchProbes.length > 0 ? state.batchProbes : state.probeResult ? [state.probeResult] : [];
        const byPath = new Map(base.map((p) => [p.file_path, p]));
        for (const p of probes) byPath.set(p.file_path, p);
        const merged = Array.from(byPath.values());
        const sorted = merged.length > 1 ? naturalSortBy(merged, (p) => p.file_name) : merged;
        return {
          batchProbes: sorted,
          probeResult: sorted[0] ?? null,
        };
      });
    },

    removeBatchProbe: (filePath) => {
      set((state) => {
        const remaining = state.batchProbes.filter((p) => p.file_path !== filePath);
        return {
          batchProbes: remaining,
          probeResult: remaining[0] ?? null,
        };
      });
    },

    updatePipelineConfig: (config) => {
      const next = { ...get().pipelineConfig, ...config };
      saveConfig(next);
      set({ pipelineConfig: next });
    },

    createAndStartJob: async () => {
      const { batchProbes, probeResult, pipelineConfig, ingestMode, syncFile, syncLanguage, syncUseGemini } = get();
      const targets = batchProbes.length > 0 ? batchProbes : probeResult ? [probeResult] : [];
      if (targets.length === 0) return null;
      const syncing = ingestMode === 'sync';
      if (syncing && !syncFile) return null;

      if (targets.length > 1 && !syncing) {
        const createdJobs: Job[] = [];
        try {
          set({ isLoading: true, error: null });
          for (const target of targets) {
            const job = await api.createJob({
              ...pipelineConfig,
              gemini_api_key_override: pipelineConfig.gemini_api_key_override || undefined,
              auto_save_srt: !target.is_upload && Boolean(pipelineConfig.auto_save_srt),
              video_path: target.file_path,
              file_name: target.file_name,
            });
            createdJobs.push(job);
          }
          const firstJob = createdJobs[0];
          set((state) => ({
            activeJob: firstJob,
            jobs: [...[...createdJobs].reverse(), ...state.jobs],
            batchProbes: [],
            probeResult: null,
            isLoading: false,
            selectedSegmentIndex: null,
            seekRequest: null,
            videoCurrentTime: 0,
          }));
          get().connectWebSocket(firstJob.id);
          return firstJob;
        } catch (e: any) {
          if (createdJobs.length > 0) {
            const firstJob = createdJobs[0];
            set((state) => ({
              activeJob: firstJob,
              jobs: [...[...createdJobs].reverse(), ...state.jobs],
              batchProbes: [],
              probeResult: null,
            }));
            get().connectWebSocket(firstJob.id);
          }
          set({ error: e.message, isLoading: false });
          return createdJobs[0] ?? null;
        }
      }

      const target = targets[0];
      try {
        set({ isLoading: true, error: null });
        const newJob = await api.createJob({
          ...pipelineConfig,
          gemini_api_key_override: pipelineConfig.gemini_api_key_override || undefined,
          auto_save_srt: !target.is_upload && Boolean(pipelineConfig.auto_save_srt),
          video_path: target.file_path,
          file_name: target.file_name,
          ...(syncing && syncFile
            ? {
                enable_translation: false,
                sync_cues: syncFile.cues,
                sync_file_name: syncFile.file_name,
                sync_language: syncLanguage || syncFile.language,
                sync_use_gemini: syncUseGemini,
              }
            : {}),
        });

        set((state) => ({
          activeJob: newJob,
          jobs: [newJob, ...state.jobs],
          batchProbes: [],
          isLoading: false,
          selectedSegmentIndex: null,
          seekRequest: null,
          videoCurrentTime: 0,
        }));

        get().connectWebSocket(newJob.id);
        return newJob;
      } catch (e: any) {
        set({ error: e.message, isLoading: false });
        return null;
      }
    },

    cancelActiveJob: async () => {
      const { activeJob } = get();
      if (!activeJob) return;
      try {
        await api.cancelJob(activeJob.id);
        set({ stageMessage: 'Cancelling...' });
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    cancelJobById: async (jobId: string) => {
      try {
        await api.cancelJob(jobId);
        set((state) => ({
          jobs: state.jobs.map((j) => (j.id === jobId ? { ...j, status: 'CANCELLED' as JobStatus } : j)),
          activeJob:
            state.activeJob?.id === jobId
              ? { ...state.activeJob, status: 'CANCELLED' as JobStatus }
              : state.activeJob,
          ...(state.activeJob?.id === jobId ? { stageMessage: 'Cancelling...' } : {}),
        }));
        void get().fetchJobs();
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    cancelAllActiveJobs: async () => {
      const active = get().jobs.filter((j) => isActiveStatus(j.status));
      if (active.length === 0) return;
      const ordered = [
        ...active.filter((j) => j.status === 'QUEUED' || j.status === 'IDLE'),
        ...active.filter((j) => j.status !== 'QUEUED' && j.status !== 'IDLE'),
      ];
      try {
        await Promise.allSettled(ordered.map((j) => api.cancelJob(j.id)));
        await get().fetchJobs();
        if (get().activeJob && isActiveStatus(get().activeJob?.status)) {
          await get().refreshActiveJob();
        }
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    retranslateActiveJob: async (options) => {
      const { activeJob, pipelineConfig } = get();
      if (!activeJob) return;
      try {
        set({ error: null });
        const job = await api.retranslate(activeJob.id, {
          gemini_api_key_override: pipelineConfig.gemini_api_key_override || undefined,
          ...options,
        });
        // Keep the segments we already have; the job response carries the new status
        patchActiveJob({ ...job, segments: options.only_failed || options.segment_ids ? activeJob.segments : job.segments });
        get().connectWebSocket(activeJob.id);
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    deleteJob: async (jobId: string) => {
      try {
        await api.deleteJob(jobId);
        if (get().activeJob?.id === jobId) get().clearActiveJob();
        set((state) => ({ jobs: state.jobs.filter((j) => j.id !== jobId) }));
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    deleteJobs: async (jobIds: string[]) => {
      if (jobIds.length === 0) return;
      try {
        await api.deleteJobs(jobIds);
        const idSet = new Set(jobIds);
        const activeId = get().activeJob?.id;
        if (activeId && idSet.has(activeId)) get().clearActiveJob();
        set((state) => ({ jobs: state.jobs.filter((j) => !idSet.has(j.id)) }));
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    deleteAllJobs: async () => {
      try {
        await api.deleteAllJobs();
        if (get().activeJob) get().clearActiveJob();
        set({ jobs: [] });
      } catch (e: any) {
        set({ error: e.message });
      }
    },

    previewSegmentTiming: (segmentId, patch) => {
      const { activeJob } = get();
      if (!activeJob) return;
      patchActiveJob({
        segments: activeJob.segments.map((s) => {
          if (s.id !== segmentId) return s;
          const dur = Math.max(0.1, patch.end_time - patch.start_time);
          const text = (s.translated_text || s.original_text || '').replace(/\n/g, '').trim();
          return {
            ...s,
            start_time: patch.start_time,
            end_time: patch.end_time,
            cps: Math.round((text.length / dur) * 10) / 10,
          };
        }),
      });
    },

    updateSegment: async (segmentId, patch) => {
      const { activeJob } = get();
      if (!activeJob) return;
      const jobId = activeJob.id;
      const timingChanged = patch.start_time !== undefined || patch.end_time !== undefined;

      const normalizeSegments = (segs: SubtitleSegment[]) => {
        if (!timingChanged) return segs;
        return [...segs]
          .sort((a, b) => a.start_time - b.start_time || a.end_time - b.end_time)
          .map((s, idx) => ({ ...s, index: idx + 1 }));
      };

      // Optimistic: show the change immediately, the server only confirms derived fields
      const { addressee_gender: pickedListener, ...fields } = patch;
      const shown: Partial<SubtitleSegment> =
        pickedListener !== undefined ? { ...fields, addressee: manualAddressee(pickedListener) } : fields;
      patchActiveJob({
        segments: normalizeSegments(
          activeJob.segments.map((s) => (s.id === segmentId ? { ...s, ...shown } : s))
        ),
      });

      set((s) => ({ savingCount: s.savingCount + 1 }));
      try {
        const request = api.updateSegment(jobId, segmentId, patch);
        inflightSaves.add(request);
        const saved = await request.finally(() => inflightSaves.delete(request));
        scheduleHistoryRefresh();
        const current = get().activeJob;
        if (!current || current.id !== jobId) {
          set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), lastSavedAt: Date.now() }));
          return;
        }
        set((s) => ({
          savingCount: Math.max(0, s.savingCount - 1),
          lastSavedAt: Date.now(),
          activeJob: {
            ...current,
            segments: normalizeSegments(
              current.segments.map((s) =>
                s.id === segmentId
                  ? {
                      ...s,
                      // Text may have changed again while the request was in flight — keep local text
                      cps: saved.cps,
                      translation_failed: saved.translation_failed,
                      start_time: patch.start_time !== undefined ? saved.start_time : s.start_time,
                      end_time: patch.end_time !== undefined ? saved.end_time : s.end_time,
                      addressee: pickedListener !== undefined ? saved.addressee : s.addressee,
                    }
                  : s
              )
            ),
          },
        }));
      } catch (e: any) {
        set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), error: `Could not save subtitle: ${e.message}` }));
        get().refreshActiveJob();
      }
    },

    saveGlossary: async (glossary, profile) => {
      const { activeJob } = get();
      if (!activeJob) return false;
      try {
        const saved = await api.saveGlossary(activeJob.id, glossary, profile);
        patchActiveJob({
          glossary: saved,
          ...(profile !== undefined ? { glossary_profile: profile.trim() || null } : {}),
        });
        return true;
      } catch (e: any) {
        set({ error: e.message });
        return false;
      }
    },

    saveSpeakers: async (speakers) => {
      const { activeJob } = get();
      if (!activeJob) return false;
      try {
        const updated = await api.updateSpeakers(activeJob.id, speakers);
        patchActiveJob({ speakers: updated.speakers ?? speakers });
        return true;
      } catch (e: any) {
        set({ error: e.message });
        return false;
      }
    },

    diarizeActiveJob: async (opts) => {
      const { activeJob } = get();
      if (!activeJob) return false;
      const jobId = activeJob.id;
      try {
        const updated = await api.diarizeJob(jobId, opts);
        if (get().activeJob?.id !== jobId) return false;
        patchActiveJob({
          speakers: updated.speakers,
          segments: updated.segments,
          glossary: updated.glossary,
          enable_diarization: updated.enable_diarization,
        });
        scheduleHistoryRefresh();
        return true;
      } catch (e: any) {
        set({ error: e.message });
        return false;
      }
    },

    matchSpeakersActiveJob: async (opts) => {
      const { activeJob } = get();
      if (!activeJob) return false;
      const jobId = activeJob.id;
      try {
        const updated = await api.matchSpeakers(jobId, opts);
        if (get().activeJob?.id !== jobId) return false;
        patchActiveJob({
          speakers: updated.speakers,
          segments: updated.segments,
          glossary: updated.glossary,
        });
        scheduleHistoryRefresh();
        return true;
      } catch (e: any) {
        set({ error: e.message });
        return false;
      }
    },

    mergeSpeakersActiveJob: async (sourceSpeakerIds, targetSpeakerId) => {
      const { activeJob } = get();
      if (!activeJob || sourceSpeakerIds.length === 0) return false;
      const jobId = activeJob.id;
      try {
        const updated = await api.mergeSpeakers(jobId, sourceSpeakerIds, targetSpeakerId);
        if (get().activeJob?.id !== jobId) return false;
        patchActiveJob({
          speakers: updated.speakers,
          segments: updated.segments,
          glossary: updated.glossary,
        });
        scheduleHistoryRefresh();
        return true;
      } catch (e: any) {
        set({ error: e.message });
        return false;
      }
    },

    bulkUpdateSpeaker: async (segmentIds, speaker) => {
      const { activeJob } = get();
      if (!activeJob || segmentIds.length === 0) return;
      const jobId = activeJob.id;
      const idSet = new Set(segmentIds);
      patchActiveJob({
        segments: activeJob.segments.map((s) => (s.id && idSet.has(s.id) ? { ...s, speaker } : s)),
      });
      set((s) => ({ savingCount: s.savingCount + 1 }));
      try {
        const saved = await api.updateSegments(
          jobId,
          segmentIds.map((id) => ({ id, speaker }))
        );
        const byId = new Map(saved.map((s) => [s.id, s]));
        const cur = get().activeJob;
        if (cur && cur.id === jobId) {
          patchActiveJob({
            segments: cur.segments.map((s) => (s.id && byId.has(s.id) ? { ...s, ...byId.get(s.id)! } : s)),
          });
        }
        set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), lastSavedAt: Date.now() }));
        scheduleHistoryRefresh();
      } catch (e: any) {
        set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), error: e.message }));
        void get().refreshActiveJob();
      }
    },

    retimeActiveJob: async () => {
      const { activeJob, timingBusy } = get();
      if (!activeJob || timingBusy) return;
      const jobId = activeJob.id;
      set({ timingBusy: true, timingNote: null });
      try {
        const res = await api.retimeJob(jobId);
        if (get().activeJob?.id !== jobId) return;
        patchActiveJob({ segments: res.segments });
        set({ timingNote: String(res.changed) });
        scheduleHistoryRefresh();
      } catch (e: any) {
        set({ error: e.message, timingNote: null });
      } finally {
        set({ timingBusy: false });
      }
    },

    moveSegment: async (segmentId, toPosition) => {
      const { activeJob, timingBusy } = get();
      if (!activeJob || timingBusy) return;
      const jobId = activeJob.id;
      set({ timingBusy: true, timingNote: null });
      try {
        const res = await api.moveSegment(jobId, segmentId, toPosition);
        if (get().activeJob?.id !== jobId) return;
        patchActiveJob({ segments: res.segments });
        scheduleHistoryRefresh();
        if (res.message) set({ error: res.message });
      } catch (e: any) {
        set({ error: e.message });
        get().refreshActiveJob();
      } finally {
        set({ timingBusy: false });
      }
    },

    replaceAll: async (search, replacement, column, options) => {
      await flushAllPendingEdits();
      const { activeJob } = get();
      if (!activeJob || !search) return 0;
      const opts = { ...DEFAULT_SEARCH_OPTIONS, ...(typeof options === 'boolean' ? { useRegex: options } : options) };
      const { regex, error } = buildSearchPattern(search, opts);
      if (!regex) {
        if (error) set({ error });
        return 0;
      }
      const field = column === 'translated' ? 'translated_text' : 'original_text';
      const patches = planReplacements(activeJob.segments, field, regex, replacement ?? '', opts.useRegex).map(
        ({ id, after }) => ({ id, [field]: after })
      );
      if (patches.length === 0) return 0;
      set((s) => ({ savingCount: s.savingCount + 1 }));
      try {
        const saved = await api.updateSegments(activeJob.id, patches);
        const byId = new Map(saved.map((s) => [s.id, s]));
        patchActiveJob({
          segments: activeJob.segments.map((s) => (s.id && byId.has(s.id) ? { ...s, ...byId.get(s.id)! } : s)),
        });
        set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), lastSavedAt: Date.now() }));
        scheduleHistoryRefresh();
        return patches.length;
      } catch (e: any) {
        set((s) => ({ savingCount: Math.max(0, s.savingCount - 1), error: e.message }));
        return 0;
      }
    },

    setVideoCurrentTime: (time) => set({ videoCurrentTime: time }),

    setSelectedSegmentIndex: (index) => set({ selectedSegmentIndex: index }),

    seekTo: (time, play = false) =>
      set({ seekRequest: { time: Math.max(0, time), nonce: Date.now() + Math.random(), play } }),

    setSubtitleOverlayMode: (mode) => set({ subtitleOverlayMode: mode }),

    setApplyRtlExport: (val) => set({ applyRtlExport: val }),

    clearError: () => set({ error: null }),

    connectWebSocket: (jobId: string, isReconnect = false) => {
      const attempts = reconnectAttempts;
      get().disconnectWebSocket();
      reconnectAttempts = isReconnect ? attempts : 0;

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${protocol}//${window.location.host}/ws/jobs/${jobId}`);
      activeSocket = socket;

      socket.onopen = () => {
        reconnectAttempts = 0;
        set({ wsConnected: true });
        // Catch up on anything missed while disconnected or while switching back to a live job
        const current = get().activeJob;
        if (isReconnect || (current && current.id === jobId && isActiveStatus(current.status))) {
          get().refreshActiveJob();
        }
      };

      socket.onmessage = (event) => {
        try {
          handleMessage(JSON.parse(event.data));
        } catch (err) {
          console.error('Failed to handle WS message:', err);
        }
      };

      socket.onclose = () => {
        if (activeSocket !== socket) return; // replaced or closed on purpose
        activeSocket = null;
        set({ wsConnected: false });
        const job = get().activeJob;
        if (job && job.id === jobId && isActiveStatus(job.status)) {
          const delay = Math.min(10000, 1000 * 2 ** reconnectAttempts++);
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            if (get().activeJob?.id === jobId) get().connectWebSocket(jobId, true);
          }, delay);
        }
      };
    },

    disconnectWebSocket: () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      reconnectAttempts = 0;
      if (activeSocket) {
        const socket = activeSocket;
        activeSocket = null;
        socket.close();
      }
      set({ wsConnected: false });
    },
  };
});
