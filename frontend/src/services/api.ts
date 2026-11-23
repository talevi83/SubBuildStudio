import type { ParsedSubtitles, SegmentCreate, SyncCue, SystemInfo, UpdateStatus, UpdateApplyResponse, GpuLibsStatus, ProbeResponse, Job, JobConfig, SubtitleSegment, SegmentPatch, SegmentsChange, QAIssue, SpeakerGuess, StorageReport, StorageCleanResult, SaveNextToVideoResult, EmbedVideoOptions, EmbedVideoStatus, EditHistory, HistoryRestoreResult, Glossary, RetranslateOptions, AppSettings, SettingsResponse, CostEstimate, ModelCheckResponse, WaveformData } from '../types';

import { hashBlobSha256 } from '../utils/sha256';

const API_BASE = '/api';

const NOT_SIGNED_IN =
  'This window is not signed in to the running SubBuild Studio. Open the studio from its desktop shortcut or tray icon, or use the link printed in the console that started it.';

/**
 * The message to show for a failed request. FastAPI sends `detail` as a string for our own errors,
 * but as a list of `{loc, msg}` objects when the request didn't pass validation (HTTP 422).
 */
export function errorDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (typeof d === 'string') return d;
        const item = d as { loc?: unknown[]; msg?: unknown };
        if (typeof item?.msg !== 'string') return null;
        const field = Array.isArray(item.loc) ? item.loc.filter((p) => p !== 'body' && p !== 'query').join('.') : '';
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .filter((p): p is string => !!p);
    if (parts.length) return parts.join('; ');
  }
  return fallback;
}

async function request<T>(url: string, init?: RequestInit, fallbackError = 'Request failed'): Promise<T> {
  const res = await fetch(url, init);
  if (res.status === 401) throw new Error(NOT_SIGNED_IN);
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(errorDetail(err, fallbackError));
  }
  return res.json();
}

/** Thrown when the user cancels an upload (or the duplicate check before it). */
export class UploadCancelled extends Error {
  constructor() {
    super('Upload cancelled');
    this.name = 'UploadCancelled';
  }
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  getSystemInfo(): Promise<SystemInfo> {
    return request(`${API_BASE}/system/info`, undefined, 'Failed to load system info');
  },

  /** The last check's result, or a fresh check when there is none yet (used at startup). */
  getUpdateStatus(): Promise<UpdateStatus> {
    return request(`${API_BASE}/system/update/status`, undefined, 'Failed to check for updates');
  },

  checkForUpdates(): Promise<UpdateStatus> {
    return request(`${API_BASE}/system/update/check`, json('POST', {}), 'Failed to check for updates');
  },

  applyUpdate(): Promise<UpdateApplyResponse> {
    return request(`${API_BASE}/system/update/apply`, json('POST', {}), 'Failed to apply update');
  },

  /** Linux AppImage: the optional NVIDIA libraries for GPU transcription */
  getGpuLibsStatus(): Promise<GpuLibsStatus> {
    return request(`${API_BASE}/system/gpu-libs`, undefined, 'Failed to check the GPU libraries');
  },

  installGpuLibs(): Promise<GpuLibsStatus> {
    return request(`${API_BASE}/system/gpu-libs/install`, json('POST', {}), 'Failed to start the GPU download');
  },

  probePath(filePath: string): Promise<ProbeResponse> {
    return request(`${API_BASE}/media/probe-path`, json('POST', { file_path: filePath }), 'Failed to inspect file path');
  },

  /** Returns the probe of an identical earlier upload (same name + size + SHA-256 content hash), or null. */
  async findExistingUpload(
    file: File,
    options: { signal?: AbortSignal; onProgress?: (fraction: number) => void } = {}
  ): Promise<ProbeResponse | null> {
    // Hashing reads the whole file, which takes as long as uploading it. Only do it when an
    // earlier upload with the same name and size exists (if the check fails, hash anyway).
    const candidates = await request<{ count: number }>(
      `${API_BASE}/media/upload/candidates?${new URLSearchParams({ name: file.name, size: String(file.size) })}`
    ).catch(() => null);
    if (candidates && candidates.count === 0) return null;
    const sha256 = await hashBlobSha256(file, options);
    if (options.signal?.aborted) throw new UploadCancelled();
    if (!sha256) return null;
    const params = new URLSearchParams({ name: file.name, size: String(file.size), sha256 });
    return request<ProbeResponse | null>(`${API_BASE}/media/upload/existing?${params}`).catch(() => null);
  },

  /** Upload with progress reporting (fetch() can't report upload progress). */
  uploadFile(file: File, onProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<ProbeResponse> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(new UploadCancelled());
        return;
      }
      const xhr = new XMLHttpRequest();
      signal?.addEventListener('abort', () => xhr.abort(), { once: true });
      xhr.onabort = () => reject(new UploadCancelled());
      xhr.open('POST', `${API_BASE}/media/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body: any = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else if (xhr.status === 401) reject(new Error(NOT_SIGNED_IN));
        else reject(new Error(errorDetail(body, 'Upload failed')));
      };
      xhr.onerror = () => reject(new Error('Upload failed (network error)'));
      const formData = new FormData();
      formData.append('file', file);
      xhr.send(formData);
    });
  },

  /** Parse a subtitle file the user wants synced (SRT / WebVTT / ASS; any encoding). Nothing is stored. */
  parseSubtitles(file: File, encoding?: string): Promise<ParsedSubtitles> {
    const formData = new FormData();
    formData.append('file', file);
    const query = encoding ? `?${new URLSearchParams({ encoding })}` : '';
    return request(
      `${API_BASE}/media/parse-subtitles${query}`,
      { method: 'POST', body: formData },
      'Could not read the subtitle file'
    );
  },

  createJob(
    params: JobConfig & {
      video_path: string;
      file_name?: string;
      sync_cues?: SyncCue[];
      sync_file_name?: string;
      sync_language?: string;
      sync_use_gemini?: boolean;
    }
  ): Promise<Job> {
    return request(`${API_BASE}/jobs`, json('POST', params), 'Failed to create job');
  },

  listJobs(): Promise<Job[]> {
    return request(`${API_BASE}/jobs`, undefined, 'Failed to list jobs');
  },

  getJob(jobId: string): Promise<Job> {
    return request(`${API_BASE}/jobs/${jobId}`, undefined, `Failed to load job ${jobId}`);
  },

  getWaveform(jobId: string): Promise<WaveformData> {
    return request(`${API_BASE}/jobs/${jobId}/waveform`, undefined, 'Failed to load audio waveform');
  },

  updateJobSettings(jobId: string, patch: { onscreen_position?: import('../types').OnscreenPosition }): Promise<Job> {
    return request(`${API_BASE}/jobs/${jobId}/settings`, json('PATCH', patch), 'Failed to update job settings');
  },

  cancelJob(jobId: string): Promise<unknown> {
    return request(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' }, 'Failed to cancel job');
  },

  retranslate(jobId: string, options: RetranslateOptions): Promise<Job> {
    return request(`${API_BASE}/jobs/${jobId}/translate`, json('POST', options), 'Failed to start translation');
  },

  deleteJob(jobId: string): Promise<unknown> {
    return request(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' }, 'Failed to delete job');
  },

  deleteJobs(jobIds: string[]): Promise<{ deleted_count: number; deleted_ids: string[] }> {
    return request(`${API_BASE}/jobs/bulk-delete`, json('POST', { job_ids: jobIds }), 'Failed to delete selected jobs');
  },

  deleteAllJobs(): Promise<{ deleted_count: number; deleted_ids: string[] }> {
    return request(`${API_BASE}/jobs/bulk-delete`, json('POST', { delete_all: true }), 'Failed to clear jobs');
  },

  updateSegment(jobId: string, segmentId: number, patch: SegmentPatch): Promise<SubtitleSegment> {
    return request(
      `${API_BASE}/jobs/${jobId}/segments/${segmentId}`,
      // keepalive: an edit saved as the window closes still reaches the server
      { ...json('PATCH', patch), keepalive: true },
      'Failed to update subtitle segment'
    );
  },

  updateSegments(jobId: string, patches: (SegmentPatch & { id: number })[]): Promise<SubtitleSegment[]> {
    return request(`${API_BASE}/jobs/${jobId}/segments`, json('PATCH', patches), 'Failed to update subtitles');
  },

  saveGlossary(jobId: string, glossary: Glossary, profile?: string): Promise<Glossary> {
    return request(
      `${API_BASE}/jobs/${jobId}/glossary`,
      json('PUT', profile === undefined ? glossary : { ...glossary, profile }),
      'Could not save the glossary'
    );
  },

  updateSpeakers(
    jobId: string,
    speakers: Record<string, import('../types').SpeakerProfile>
  ): Promise<Job> {
    return request(
      `${API_BASE}/jobs/${jobId}/speakers`,
      json('PUT', { speakers }),
      'Could not save speaker profiles'
    );
  },

  diarizeJob(
    jobId: string,
    opts?: { min_speakers?: number; max_speakers?: number; threshold?: number; auto_match?: boolean }
  ): Promise<Job> {
    return request(
      `${API_BASE}/jobs/${jobId}/diarize`,
      json('POST', opts || {}),
      'Could not detect speakers from audio'
    );
  },

  matchSpeakers(
    jobId: string,
    opts?: { overwrite_manual?: boolean; auto_merge?: boolean }
  ): Promise<Job> {
    return request(
      `${API_BASE}/jobs/${jobId}/match-speakers`,
      json('POST', opts || {}),
      'Could not match speakers to characters'
    );
  },

  mergeSpeakers(
    jobId: string,
    sourceSpeakerIds: string[],
    targetSpeakerId: string | null
  ): Promise<Job> {
    return request(
      `${API_BASE}/jobs/${jobId}/speakers/merge`,
      json('POST', { source_speaker_ids: sourceSpeakerIds, target_speaker_id: targetSpeakerId }),
      'Could not merge speaker profiles'
    );
  },

  listGlossaryProfiles(): Promise<string[]> {
    return request(`${API_BASE}/glossary-profiles`, undefined, 'Could not load the series list');
  },

  storageReport(olderThanDays: number): Promise<StorageReport> {
    return request(`${API_BASE}/storage?older_than_days=${olderThanDays}`, undefined, 'Could not read storage usage');
  },

  storageClean(categories: string[], olderThanDays?: number): Promise<StorageCleanResult> {
    return request(
      `${API_BASE}/storage/clean`,
      json('POST', { categories, older_than_days: olderThanDays }),
      'Cleanup failed',
    );
  },

  detectSpeakers(
    jobId: string,
    segmentIds: number[],
    apiKey?: string,
    model?: string,
    thinking?: string,
  ): Promise<SpeakerGuess> {
    return request(
      `${API_BASE}/jobs/${jobId}/detect-speakers`,
      json('POST', {
        segment_ids: segmentIds,
        gemini_api_key_override: apiKey || undefined,
        gemini_model: model || undefined,
        thinking: thinking || undefined,
      }),
      'Could not detect speakers from the video',
    );
  },

  qualityCheck(jobId: string): Promise<QAIssue[]> {
    return request(`${API_BASE}/jobs/${jobId}/qa`, undefined, 'Quality check failed');
  },

  retimeJob(jobId: string): Promise<SegmentsChange> {
    return request(`${API_BASE}/jobs/${jobId}/retime`, json('POST', {}), 'Timing check failed');
  },

  /** Add a subtitle line by hand (it must not overlap an existing line). */
  createSegment(jobId: string, body: SegmentCreate): Promise<SegmentsChange> {
    return request(`${API_BASE}/jobs/${jobId}/segments`, json('POST', body), 'Could not add the line');
  },

  /** Delete a subtitle line by user request. */
  deleteSegment(jobId: string, segmentId: number): Promise<SegmentsChange> {
    return request(`${API_BASE}/jobs/${jobId}/segments/${segmentId}`, { method: 'DELETE' }, 'Could not delete the line');
  },

  /** Delete multiple subtitle lines by user request. */
  deleteSegments(jobId: string, segmentIds: number[]): Promise<SegmentsChange> {
    return request(`${API_BASE}/jobs/${jobId}/segments`, json('DELETE', segmentIds), 'Could not delete selected lines');
  },

  moveSegment(jobId: string, segmentId: number, toPosition: number): Promise<SegmentsChange> {
    return request(
      `${API_BASE}/jobs/${jobId}/segments/move`,
      json('POST', { segment_id: segmentId, to_position: toPosition }),
      'Could not move subtitle'
    );
  },

  splitSegment(jobId: string, segmentId: number, splitTime?: number): Promise<SegmentsChange> {
    return request(
      `${API_BASE}/jobs/${jobId}/segments/${segmentId}/split`,
      json('POST', splitTime !== undefined ? { split_time: splitTime } : {}),
      'Could not split subtitle line'
    );
  },

  mergeSegments(jobId: string, segmentIds: number[]): Promise<SegmentsChange> {
    return request(
      `${API_BASE}/jobs/${jobId}/segments/merge`,
      json('POST', { segment_ids: segmentIds }),
      'Could not merge selected lines'
    );
  },

  shiftSegments(jobId: string, offsetSeconds: number, segmentIds?: number[]): Promise<SegmentsChange> {
    return request(
      `${API_BASE}/jobs/${jobId}/segments/shift`,
      json('POST', {
        offset_seconds: offsetSeconds,
        ...(segmentIds && segmentIds.length > 0 ? { segment_ids: segmentIds } : {}),
      }),
      'Could not shift subtitle timing'
    );
  },

  getSettings(): Promise<SettingsResponse> {
    return request(`${API_BASE}/settings`, undefined, 'Could not load settings');
  },

  saveSettings(settings: AppSettings): Promise<SettingsResponse> {
    return request(`${API_BASE}/settings`, json('PUT', settings), 'Could not save settings');
  },

  getApiKeyStatus(): Promise<import('../types').ApiKeyStatus> {
    return request(`${API_BASE}/settings/api-key`, undefined, 'Could not check API key status');
  },

  saveApiKey(apiKey: string): Promise<import('../types').ApiKeyStatus> {
    return request(`${API_BASE}/settings/api-key`, json('PUT', { api_key: apiKey }), 'Could not save API key');
  },

  getProviderKeys(provider?: string): Promise<import('../types').ProviderKeysResponse> {
    const q = provider ? `?provider=${encodeURIComponent(provider)}` : '';
    return request(`${API_BASE}/settings/keys${q}`, undefined, 'Could not load AI provider keys');
  },

  addProviderKey(body: {
    provider: import('../types').AIProvider;
    api_key?: string;
    label?: string;
    base_url?: string;
    is_default?: boolean;
    verify?: boolean;
  }): Promise<import('../types').ProviderKeyMutationResponse> {
    return request(`${API_BASE}/settings/keys`, json('POST', body), 'Could not add API key');
  },

  updateProviderKey(
    keyId: string,
    body: { label?: string; api_key?: string; base_url?: string; is_default?: boolean }
  ): Promise<import('../types').ProviderKeyMutationResponse> {
    return request(`${API_BASE}/settings/keys/${keyId}`, json('PUT', body), 'Could not update API key');
  },

  deleteProviderKey(keyId: string): Promise<import('../types').ProviderKeyMutationResponse> {
    return request(`${API_BASE}/settings/keys/${keyId}`, { method: 'DELETE' }, 'Could not delete API key');
  },

  refreshProviderModels(): Promise<import('../types').RefreshProvidersResponse> {
    return request(`${API_BASE}/settings/providers/refresh`, json('POST', {}), 'Could not refresh provider models');
  },

  estimateCost(params: {
    duration_seconds: number;
    enable_translation: boolean;
    detect_onscreen_text: boolean;
    has_forced_track?: boolean;
    sync_lines?: number;
    cost_saver_mode?: boolean;
    settings?: AppSettings;
  }): Promise<CostEstimate> {
    return request(`${API_BASE}/settings/estimate`, json('POST', params), 'Could not estimate cost');
  },

  checkModels(settings: AppSettings, apiKey?: string): Promise<ModelCheckResponse> {
    return request(
      `${API_BASE}/settings/check`,
      json('POST', { settings, api_key: apiKey || undefined }),
      'Could not test the models'
    );
  },

  getOllamaStatus(): Promise<import('../types').OllamaStatus> {
    return request(`${API_BASE}/settings/ollama/status`, undefined, 'Could not check local Ollama status');
  },

  setupOllama(model?: string): Promise<import('../types').OllamaStatus> {
    return request(
      `${API_BASE}/settings/ollama/setup`,
      json('POST', model ? { model } : {}),
      'Could not install or start Ollama'
    );
  },

  getVideoStreamUrl(jobId: string): string {
    return `${API_BASE}/media/stream/${jobId}`;
  },

  getExportUrl(
    jobId: string,
    format: 'srt' | 'vtt',
    useTranslated: boolean,
    applyRtl = true,
    bilingual = false
  ): string {
    return `${API_BASE}/jobs/${jobId}/export/${format}?use_translated=${useTranslated}&apply_rtl=${applyRtl}${
      bilingual ? '&bilingual=true' : ''
    }`;
  },

  getAssExportUrl(
    jobId: string,
    opts: {
      useTranslated: boolean;
      applyRtl?: boolean;
      bilingual?: boolean;
      style?: import('../types').SubtitleStyleSettings;
      colorSpeakers?: boolean;
    }
  ): string {
    const params = new URLSearchParams({
      use_translated: String(opts.useTranslated),
      apply_rtl: String(opts.applyRtl ?? true),
    });
    if (opts.bilingual) params.set('bilingual', 'true');
    if (opts.colorSpeakers) params.set('color_speakers', 'true');
    if (opts.style) {
      params.set('font_family', opts.style.fontFamily);
      params.set('font_size_px', String(opts.style.fontSizePx));
      params.set('text_color', opts.style.textColor);
      params.set('outline_color', opts.style.outlineColor);
      params.set('outline_width', String(opts.style.outlineWidth));
      params.set('background_color', opts.style.backgroundColor);
      params.set('background_opacity', String(opts.style.backgroundOpacity));
      params.set('margin_v', String(opts.style.marginV));
      params.set('bold', String(opts.style.bold));
      params.set('highlight_mode', opts.style.highlightMode);
      params.set('highlight_color', opts.style.highlightColor);
    }
    return `${API_BASE}/jobs/${jobId}/export/ass?${params.toString()}`;
  },

  getPreviewFrameUrl(jobId: string, t?: number): string {
    return t !== undefined ? `${API_BASE}/jobs/${jobId}/preview-frame?t=${t.toFixed(2)}` : `${API_BASE}/jobs/${jobId}/preview-frame`;
  },

  getFcpxmlExportUrl(jobId: string, useTranslated: boolean, applyRtl = true, fps = 25): string {
    const params = new URLSearchParams({
      use_translated: String(useTranslated),
      apply_rtl: String(applyRtl),
      fps: String(fps),
    });
    return `${API_BASE}/jobs/${jobId}/export/fcpxml?${params.toString()}`;
  },

  getMarkdownExportUrl(jobId: string, useTranslated: boolean, bilingual = false): string {
    const params = new URLSearchParams({
      use_translated: String(useTranslated),
      bilingual: String(bilingual),
    });
    return `${API_BASE}/jobs/${jobId}/export/markdown?${params.toString()}`;
  },

  getDocxExportUrl(jobId: string, useTranslated: boolean, bilingual = false): string {
    const params = new URLSearchParams({
      use_translated: String(useTranslated),
      bilingual: String(bilingual),
    });
    return `${API_BASE}/jobs/${jobId}/export/docx?${params.toString()}`;
  },

  /** The project's edit history (steps oldest first, and the current one). */
  getHistory(jobId: string): Promise<EditHistory> {
    return request(`${API_BASE}/jobs/${jobId}/history`, undefined, 'Could not load the edit history');
  },

  undo(jobId: string): Promise<HistoryRestoreResult> {
    return request(`${API_BASE}/jobs/${jobId}/history/undo`, { method: 'POST' }, 'Undo failed');
  },

  redo(jobId: string): Promise<HistoryRestoreResult> {
    return request(`${API_BASE}/jobs/${jobId}/history/redo`, { method: 'POST' }, 'Redo failed');
  },

  /** Jump to any step of the history. */
  restoreHistory(jobId: string, entryId: number): Promise<HistoryRestoreResult> {
    return request(`${API_BASE}/jobs/${jobId}/history/${entryId}/restore`, { method: 'POST' }, 'Could not go back to that step');
  },

  /** Write the subtitles into the video's own folder (Movie.he.srt next to Movie.mkv). */
  saveNextToVideo(
    jobId: string,
    options: { format?: 'srt' | 'vtt' | 'ass'; use_translated?: boolean; apply_rtl?: boolean; overwrite?: boolean }
  ): Promise<SaveNextToVideoResult> {
    return request(
      `${API_BASE}/jobs/${jobId}/export/save-next-to-video`,
      json('POST', options),
      'Could not save the subtitles next to the video'
    );
  },

  /** Start making a copy of the video with the subtitles inside (poll getEmbedVideoStatus for progress). */
  startEmbedVideo(jobId: string, options: EmbedVideoOptions): Promise<EmbedVideoStatus> {
    return request(
      `${API_BASE}/jobs/${jobId}/export/embed`,
      json('POST', options),
      'Could not start making the video with subtitles'
    );
  },

  /** The project's most recent embed run (null when there was none since the app started). */
  getLatestEmbedVideo(jobId: string): Promise<EmbedVideoStatus | null> {
    return request(`${API_BASE}/jobs/${jobId}/export/embed`);
  },

  getEmbedVideoStatus(jobId: string, taskId: string): Promise<EmbedVideoStatus> {
    return request(`${API_BASE}/jobs/${jobId}/export/embed/${taskId}`, undefined, 'Lost track of the video being made');
  },

  cancelEmbedVideo(jobId: string, taskId: string): Promise<EmbedVideoStatus> {
    return request(`${API_BASE}/jobs/${jobId}/export/embed/${taskId}/cancel`, { method: 'POST' }, 'Could not stop it');
  },

  /** Detailed log of everything that happened while the job ran (summary + log lines). */
  getJobLogUrl(jobId: string): string {
    return `${API_BASE}/jobs/${jobId}/export/log`;
  },

  /** The backend's own log file. */
  getServerLogUrl(): string {
    return `${API_BASE}/system/log`;
  },
};
