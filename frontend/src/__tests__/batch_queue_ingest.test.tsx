// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { MediaDropzone } from '../components/ingest/MediaDropzone';
import { PipelineConfig } from '../components/ingest/PipelineConfig';
import { Sidebar } from '../components/layout/Sidebar';
import { useJobStore } from '../store/useJobStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { api } from '../services/api';
import { naturalSortBy } from '../utils/desktop';
import type { Job, ProbeResponse } from '../types';

const makeProbe = (filePath: string, fileName: string, isUpload = false): ProbeResponse => ({
  file_path: filePath,
  file_name: fileName,
  file_size: 100 * 1024 * 1024,
  duration: 1200,
  format_name: 'matroska',
  is_upload: isUpload,
  video_streams: [{ codec_name: 'h264', width: 1920, height: 1080 }],
  audio_streams: [{ index: 0, codec_name: 'aac', channels: 2, language: 'en' }],
});

const makeJob = (id: string, fileName: string, status: Job['status'], progress = 0): Job => ({
  id,
  video_path: `C:\\Series\\${fileName}`,
  file_name: fileName,
  file_size: 1024000,
  is_upload: false,
  status,
  current_stage: status,
  progress_percent: progress,
  video_duration: 1200,
  audio_streams: [],
  selected_audio_stream: 0,
  whisper_model: 'medium',
  source_language: 'en',
  vad_filter: true,
  comms_boost: false,
  detect_onscreen_text: false,
  auto_save_srt: true,
  enable_translation: true,
  target_language: 'Hebrew',
  gemini_model: 'gemini-3.8-flash',
  segment_count: 0,
  failed_translation_count: 0,
  created_at: '2026-11-22T12:00:00Z',
  updated_at: '2026-11-22T12:00:00Z',
  segments: [],
});

describe('Season Batch Queue & Multi-Episode Ingest', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    (window as unknown as { pywebview?: unknown }).pywebview = undefined;
    useJobStore.setState({
      activeJob: null,
      jobs: [],
      probeResult: null,
      batchProbes: [],
      ingestMode: 'transcribe',
      syncFile: null,
      syncRawFile: null,
      syncLanguage: null,
      error: null,
      isLoading: false,
    });
    useJobStore.getState().updatePipelineConfig({
      auto_save_srt: false,
      glossary_profile: '',
    });
  });

  it('naturalSortBy sorts S01E01, S01E02, ..., S01E10 correctly', () => {
    const unsorted = ['Show.S01E10.mkv', 'Show.S01E02.mkv', 'Show.S01E01.mkv', 'Show.S01E09.mkv'];
    expect(naturalSortBy(unsorted, (s) => s)).toEqual([
      'Show.S01E01.mkv',
      'Show.S01E02.mkv',
      'Show.S01E09.mkv',
      'Show.S01E10.mkv',
    ]);
  });

  it('picks multiple local desktop files, natural-sorts them, allows removing an item, and launches batch queue', async () => {
    const paths = [
      'C:\\Series\\Show.S01E10.mkv',
      'C:\\Series\\Show.S01E02.mkv',
      'C:\\Series\\Show.S01E01.mkv',
    ];
    (window as unknown as { pywebview?: unknown }).pywebview = {
      api: {
        pick_media_files: vi.fn().mockResolvedValue(paths),
      },
    };

    vi.spyOn(api, 'probePath').mockImplementation(async (filePath: string) => {
      const name = filePath.split('\\').pop() || filePath;
      return makeProbe(filePath, name, false);
    });
    vi.spyOn(api, 'listGlossaryProfiles').mockResolvedValue(['MySeries']);
    vi.spyOn(api, 'estimateCost').mockResolvedValue({
      duration_seconds: 2400,
      segments: 100,
      stages: {},
      total_cost_usd: 0.02,
    });
    const createdPayloads: Record<string, unknown>[] = [];
    vi.spyOn(api, 'createJob').mockImplementation(async (params) => {
      createdPayloads.push({ ...params });
      return makeJob(`job_${createdPayloads.length}`, params.file_name || 'ep.mkv', createdPayloads.length === 1 ? 'IDLE' : 'QUEUED');
    });
    vi.spyOn(useJobStore.getState(), 'connectWebSocket').mockImplementation(() => {});

    useSettingsStore.setState({
      loaded: true,
      catalog: [],
      settings: {
        translation: { model: 'gemini-3.8-flash', thinking: 'low', flex: false },
        glossary: { model: 'gemini-3.8-flash', thinking: 'low', flex: false },
        vision: { model: 'gemini-3.8-flash', thinking: 'minimal', flex: false },
        vision_media_resolution: 'medium',
        cost_saver_mode: false,
      },
    });

    const dropView = render(<MediaDropzone />);
    expect(dropView.getByTestId('batch-feature-badge')).toBeTruthy();
    expect(dropView.getByTestId('batch-callout')).toBeTruthy();
    fireEvent.click(dropView.getByTestId('media-dropzone-area'));

    await waitFor(() => {
      expect(dropView.getByTestId('batch-ingest-list')).toBeTruthy();
    });

    // Verify natural order in batch list
    const batch = useJobStore.getState().batchProbes;
    expect(batch.map((p) => p.file_name)).toEqual([
      'Show.S01E01.mkv',
      'Show.S01E02.mkv',
      'Show.S01E10.mkv',
    ]);

    // Remove the middle episode (S01E02)
    fireEvent.click(dropView.getByTestId('batch-remove-1'));
    expect(useJobStore.getState().batchProbes.map((p) => p.file_name)).toEqual([
      'Show.S01E01.mkv',
      'Show.S01E10.mkv',
    ]);

    const cfgView = render(<PipelineConfig />);
    expect(cfgView.getByTestId('pipeline-batch-banner')).toBeTruthy();
    const autoSaveWrapper = cfgView.getByTestId('auto-save-srt-toggle-wrapper');
    const autoSaveSwitch = autoSaveWrapper.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(autoSaveSwitch.checked).toBe(false);
    fireEvent.click(autoSaveSwitch);
    expect(autoSaveSwitch.checked).toBe(true);

    fireEvent.change(cfgView.getByTestId('series-profile'), { target: { value: 'MySeries' } });
    fireEvent.click(cfgView.getByTestId('launch'));

    await waitFor(() => {
      expect(createdPayloads).toHaveLength(2);
    });
    expect(createdPayloads[0].file_name).toBe('Show.S01E01.mkv');
    expect(createdPayloads[1].file_name).toBe('Show.S01E10.mkv');
    expect(createdPayloads[0].auto_save_srt).toBe(true);
    expect(createdPayloads[1].auto_save_srt).toBe(true);
    expect(createdPayloads[0].glossary_profile).toBe('MySeries');
    expect(createdPayloads[1].glossary_profile).toBe('MySeries');
  });

  it('hides auto_save_srt toggle for browser uploads', () => {
    vi.spyOn(api, 'listGlossaryProfiles').mockResolvedValue([]);
    useJobStore.setState({
      probeResult: makeProbe('/storage/uploads/abc_Ep1.mkv', 'Ep1.mkv', true),
      batchProbes: [makeProbe('/storage/uploads/abc_Ep1.mkv', 'Ep1.mkv', true)],
    });
    const cfgView = render(<PipelineConfig />);
    expect(cfgView.queryByTestId('auto-save-srt-toggle-wrapper')).toBeNull();
  });

  it('shows batch queue banner in Sidebar and cancels queued jobs before running jobs', async () => {
    const running = makeJob('job_run', 'Show.S01E01.mkv', 'TRANSCRIBING', 40);
    const queued = makeJob('job_q1', 'Show.S01E02.mkv', 'QUEUED', 0);

    vi.spyOn(api, 'listJobs').mockResolvedValue([queued, running]);
    vi.spyOn(api, 'getJob').mockResolvedValue({ ...running, status: 'CANCELLED' });
    vi.spyOn(api, 'getSystemInfo').mockResolvedValue({
      is_gemini_configured: true,
      gemini_model: 'gemini-3.8-flash',
      gemini_models: [],
      whisper_default_model: 'medium',
      whisper_models: ['medium'],
      max_cps_warning: 20,
      hardware: {
        cuda_available: false,
        device_count: 0,
        cpu_count: 8,
        recommended_device: 'cpu',
        recommended_compute_type: 'int8',
      },
    });
    const cancelledIds: string[] = [];
    vi.spyOn(api, 'cancelJob').mockImplementation(async (jobId: string) => {
      cancelledIds.push(jobId);
      return { status: 'CANCELLED', job_id: jobId };
    });

    useJobStore.setState({ jobs: [queued, running], activeJob: running });
    const view = render(<Sidebar />);

    expect(view.getByTestId('sidebar-batch-queue-bar')).toBeTruthy();
    expect(view.getByTestId('sidebar-cancel-job-job_q1')).toBeTruthy();

    fireEvent.click(view.getByTestId('sidebar-cancel-queue-btn'));
    await waitFor(() => {
      expect(cancelledIds).toEqual(['job_q1', 'job_run']);
    });
  });

  it('allows building a batch incrementally from a single video using Add episodes and Add to batch', async () => {
    vi.spyOn(api, 'probePath').mockImplementation(async (filePath: string) => {
      const name = filePath.split('\\').pop() || filePath;
      return makeProbe(filePath, name, false);
    });
    const pickSpy = vi.fn().mockResolvedValue(['C:\\Series\\Show.S01E03.mkv']);
    (window as unknown as { pywebview?: unknown }).pywebview = {
      api: {
        pick_media_files: pickSpy,
      },
    };

    const view = render(<MediaDropzone />);
    fireEvent.change(view.getByTestId('local-path-input'), {
      target: { value: 'C:\\Series\\Show.S01E01.mkv' },
    });
    fireEvent.click(view.getByText(/Inspect Media/i));

    await waitFor(() => {
      expect(view.getByTestId('batch-add-more-btn')).toBeTruthy();
      expect(view.getByTestId('path-add-to-batch-btn')).toBeTruthy();
    });

    // Append Episode 3 via the Add episodes button -> transitions seamlessly to batch list
    fireEvent.click(view.getByTestId('batch-add-more-btn'));
    await waitFor(() => {
      expect(view.getByTestId('batch-ingest-list')).toBeTruthy();
    });

    // Append Episode 2 via local path input + "Add to batch" -> natural-sorted in between Ep1 and Ep3
    fireEvent.change(view.getByTestId('local-path-input'), {
      target: { value: 'C:\\Series\\Show.S01E02.mkv' },
    });
    fireEvent.click(view.getByTestId('path-add-to-batch-btn'));

    await waitFor(() => {
      expect(useJobStore.getState().batchProbes.map((p) => p.file_name)).toEqual([
        'Show.S01E01.mkv',
        'Show.S01E02.mkv',
        'Show.S01E03.mkv',
      ]);
    });
  });
});
