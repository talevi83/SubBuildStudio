// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { PipelineConfig } from '../components/ingest/PipelineConfig';
import { ExportToolbar } from '../components/studio/ExportToolbar';
import { VideoSyncPlayer } from '../components/studio/VideoSyncPlayer';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { Job, ProbeResponse, SystemInfo } from '../types';

function makeHebrewJob(): Job {
  return {
    id: 'job_he_1',
    video_path: 'C:\\Videos\\Episode.mp4',
    file_name: 'Episode.mp4',
    file_size: 1024,
    is_upload: false,
    status: 'COMPLETED',
    current_stage: 'COMPLETED',
    progress_percent: 100,
    video_duration: 60,
    audio_streams: [],
    selected_audio_stream: 0,
    whisper_model: 'ivrit-ai/whisper-large-v3-turbo-ct2',
    source_language: 'he',
    detected_language: 'he',
    vad_filter: true,
    comms_boost: false,
    detect_onscreen_text: false,
    enable_translation: false,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3-flash-preview',
    segment_count: 1,
    failed_translation_count: 0,
    created_at: '2026-11-22T10:00:00Z',
    updated_at: '2026-11-22T10:00:00Z',
    segments: [
      {
        id: 10,
        job_id: 'job_he_1',
        index: 1,
        start_time: 0,
        end_time: 4,
        original_text: 'שלום לכולם!',
        translated_text: null,
        translation_failed: false,
        segment_type: 'DIALOGUE',
        confidence: 1,
        cps: 4,
      },
    ],
  };
}

const probe: ProbeResponse = {
  file_path: 'C:\\Videos\\Episode.mp4',
  file_name: 'Episode.mp4',
  file_size: 1024,
  duration: 120,
  format_name: 'mp4',
  video_streams: [{ codec_name: 'h264', width: 1920, height: 1080 }],
  audio_streams: [{ index: 0, codec_name: 'aac', channels: 2 }],
  subtitle_streams: [],
};

const sysInfo: SystemInfo = {
  is_gemini_configured: true,
  is_ai_configured: true,
  gemini_model: 'gemini-3-flash-preview',
  gemini_models: ['gemini-3-flash-preview'],
  whisper_models: ['tiny', 'base', 'small', 'medium', 'large-v3-turbo', 'ivrit-ai/whisper-large-v3-turbo-ct2', 'large-v3'],
  whisper_default_model: 'medium',
  max_cps_warning: 20,
  hardware: {
    cuda_available: false,
    device_count: 0,
    cpu_count: 8,
    recommended_device: 'cpu',
    recommended_compute_type: 'int8',
  },
};

describe('Hebrew-to-Hebrew creator & original RTL support', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.clear();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    useJobStore.setState({
      systemInfo: sysInfo,
      probeResult: probe,
      ingestMode: 'transcribe',
      activeJob: null,
      applyRtlExport: true,
      subtitleOverlayMode: 'dual',
    });
  });

  it('shows ivrit-ai recommendation for Hebrew source and same-language AI polish banner when target matches', () => {
    useJobStore.setState((s) => ({
      pipelineConfig: {
        ...s.pipelineConfig,
        source_language: 'he',
        target_language: 'Hebrew',
        whisper_model: 'medium',
        enable_translation: true,
      },
    }));

    const { getByTestId, queryByTestId } = render(<PipelineConfig />);
    expect(getByTestId('hebrew-model-banner')).toBeTruthy();
    expect(getByTestId('same-lang-polish-banner')).toBeTruthy();

    fireEvent.click(getByTestId('use-ivrit-model'));
    expect(useJobStore.getState().pipelineConfig.whisper_model).toBe('ivrit-ai/whisper-large-v3-turbo-ct2');
    expect(queryByTestId('hebrew-model-banner')).toBeNull();

    fireEvent.click(getByTestId('disable-translation-btn'));
    expect(useJobStore.getState().pipelineConfig.enable_translation).toBe(false);
  });

  it('does NOT show same-language polish banner when source_language is auto', () => {
    useJobStore.setState((s) => ({
      pipelineConfig: {
        ...s.pipelineConfig,
        source_language: 'auto',
        target_language: 'English',
        enable_translation: true,
      },
    }));

    const { queryByTestId } = render(<PipelineConfig />);
    expect(queryByTestId('same-lang-polish-banner')).toBeNull();
  });

  it('supports save-next-to-video and RTL export URLs on untranslated Hebrew job', async () => {
    useJobStore.setState({ activeJob: makeHebrewJob(), applyRtlExport: true });

    const createdUrls: string[] = [];
    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag);
      if (tag === 'a') {
        vi.spyOn(el, 'click').mockImplementation(() => {
          createdUrls.push((el as HTMLAnchorElement).getAttribute('href') || '');
        });
      }
      return el;
    });

    const { getByTestId } = render(<ExportToolbar />);
    expect(getByTestId('save-next-to-video')).toBeTruthy();

    fireEvent.click(getByTestId('export-original-srt'));
    fireEvent.click(getByTestId('export-vtt'));

    await waitFor(() => expect(createdUrls.length).toBe(2));
    expect(createdUrls[0]).toContain('/export/srt?');
    expect(createdUrls[0]).toContain('use_translated=false');
    expect(createdUrls[0]).toContain('apply_rtl=true');
    expect(createdUrls[1]).toContain('/export/vtt?');
    expect(createdUrls[1]).toContain('use_translated=false');
    expect(createdUrls[1]).toContain('apply_rtl=true');
  });

  it('renders original Hebrew subtitles with dir="rtl" in player and font-rtl-hebrew in table', () => {
    useJobStore.setState({ activeJob: makeHebrewJob(), subtitleOverlayMode: 'original' });

    const { getByTestId } = render(<VideoSyncPlayer />);
    const box = getByTestId('player-bottom-subtitle-box');
    const p = box.querySelector('p');
    expect(p?.getAttribute('dir')).toBe('rtl');
    expect(p?.textContent).toBe('שלום לכולם!');

    cleanup();
    const { container } = render(<SubtitleTable />);
    const firstTextarea = container.querySelector('textarea');
    expect(firstTextarea?.className).toContain('font-rtl-hebrew');
  });
});
