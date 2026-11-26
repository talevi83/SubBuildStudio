// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { PipelineConfig } from '../components/ingest/PipelineConfig';
import { LivePipelineBar } from '../components/monitor/LivePipelineBar';
import { useJobStore } from '../store/useJobStore';
import { api } from '../services/api';
import type { Job, ProbeResponse } from '../types';

const sampleProbe: ProbeResponse = {
  file_path: 'C:\\Movies\\The_Uprising_2026.mkv',
  file_name: 'The_Uprising_2026.mkv',
  file_size: 200 * 1024 * 1024,
  duration: 5400,
  format_name: 'matroska',
  is_upload: false,
  video_streams: [{ codec_name: 'h264', width: 1920, height: 1080 }],
  audio_streams: [{ index: 0, codec_name: 'aac', channels: 2, language: 'en' }],
};

describe('Transcription Context / Glossary / Names UI', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.spyOn(api, 'listGlossaryProfiles').mockResolvedValue([]);
    vi.spyOn(api, 'estimateCost').mockResolvedValue({
      duration_seconds: 5400,
      segments: 1260,
      stages: {},
      total_cost_usd: 0.05,
    });
    useJobStore.setState({
      activeJob: null,
      jobs: [],
      probeResult: sampleProbe,
      batchProbes: [sampleProbe],
      ingestMode: 'transcribe',
      syncFile: null,
      error: null,
      isLoading: false,
    });
    useJobStore.getState().updatePipelineConfig({
      enable_translation: false,
      initial_prompt: '',
      context_notes: '',
    });
  });

  it('exposes Context / Glossary / Names in Transcription settings and feeds both initial_prompt and context_notes', async () => {
    const fakeJob: Job = {
      id: 'job_uprising1',
      video_path: sampleProbe.file_path,
      file_name: sampleProbe.file_name,
      status: 'IDLE',
      progress_percent: 0,
      audio_streams: sampleProbe.audio_streams,
      selected_audio_stream: 0,
      whisper_model: 'large-v3',
      source_language: 'en',
      vad_filter: true,
      comms_boost: false,
      detect_onscreen_text: false,
      enable_translation: false,
      target_language: 'Hebrew',
      gemini_model: 'gemini-3.8-flash',
      segment_count: 0,
      failed_translation_count: 0,
      created_at: '2026-11-24T22:00:00Z',
      updated_at: '2026-11-24T22:00:00Z',
      segments: [],
    };
    const createSpy = vi.spyOn(api, 'createJob').mockResolvedValue(fakeJob);
    vi.spyOn(useJobStore.getState(), 'connectWebSocket').mockImplementation(() => {});

    const view = render(<PipelineConfig />);
    const textarea = view.getByTestId('transcription-context-glossary') as HTMLTextAreaElement;
    expect(textarea).toBeTruthy();

    const glossaryValue = 'Wat Tyler, John Ball = ג׳ון בול, Ploughman, Domesday, Corpus Christi, Beauchamp';
    fireEvent.change(textarea, { target: { value: glossaryValue } });

    expect(useJobStore.getState().pipelineConfig.initial_prompt).toBe(glossaryValue);
    expect(useJobStore.getState().pipelineConfig.context_notes).toBe(glossaryValue);

    fireEvent.click(view.getByTestId('launch'));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy.mock.calls[0][0]).toMatchObject({
      initial_prompt: glossaryValue,
      context_notes: glossaryValue,
    });
  });

  it('exposes Transcription Engine selector (WhisperX) and Vocal Isolation toggle in PipelineConfig and LivePipelineBar', async () => {
    const fakeJob: Job = {
      id: 'job_wx_iso_1',
      video_path: sampleProbe.file_path,
      file_name: sampleProbe.file_name,
      status: 'VOCAL_ISOLATION',
      current_stage: 'VOCAL_ISOLATION',
      progress_percent: 25,
      audio_streams: sampleProbe.audio_streams,
      selected_audio_stream: 0,
      transcription_provider: 'whisperx',
      whisper_model: 'large-v3',
      vocal_isolation: true,
      source_language: 'en',
      vad_filter: true,
      comms_boost: false,
      detect_onscreen_text: false,
      enable_translation: false,
      target_language: 'Hebrew',
      gemini_model: 'gemini-3.8-flash',
      segment_count: 0,
      failed_translation_count: 0,
      created_at: '2026-11-24T22:00:00Z',
      updated_at: '2026-11-24T22:00:00Z',
      segments: [],
    };
    const createSpy = vi.spyOn(api, 'createJob').mockResolvedValue(fakeJob);
    vi.spyOn(useJobStore.getState(), 'connectWebSocket').mockImplementation(() => {});

    useJobStore.getState().updatePipelineConfig({
      transcription_provider: 'faster_whisper',
      vocal_isolation: false,
    });

    const view = render(<PipelineConfig />);
    const engineSelect = view.getByTestId('transcription-provider-select') as HTMLSelectElement;
    // Advanced audio controls stay collapsed until opened
    expect(view.queryByTestId('vocal-isolation-toggle')).toBeNull();
    expect(view.queryByTestId('high-speech-sensitivity-checkbox')).toBeNull();

    fireEvent.click(view.getByTestId('advanced-audio-toggle'));
    const vocalToggle = view.getByTestId('vocal-isolation-toggle') as HTMLInputElement;

    fireEvent.change(engineSelect, { target: { value: 'whisperx' } });
    fireEvent.click(vocalToggle);

    expect(useJobStore.getState().pipelineConfig.transcription_provider).toBe('whisperx');
    expect(useJobStore.getState().pipelineConfig.vocal_isolation).toBe(true);

    fireEvent.click(view.getByTestId('launch'));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy.mock.calls[0][0]).toMatchObject({
      transcription_provider: 'whisperx',
      vocal_isolation: true,
    });

    cleanup();
    useJobStore.setState({ activeJob: fakeJob });
    const bar = render(<LivePipelineBar />);
    expect(bar.getByTestId('pipeline-stage-VOCAL_ISOLATION').getAttribute('data-stage-state')).toBe('active');
    expect(bar.getByText('WhisperX')).toBeTruthy();
  });

  it('supports Smart Audio Map auto modes without enabling global comms_boost in Documentary preset', async () => {
    const view = render(<PipelineConfig />);
    const docuBtn = view.getByTestId('genre-preset-docu');
    expect(docuBtn).toBeTruthy();
    fireEvent.click(docuBtn);

    const cfg = useJobStore.getState().pipelineConfig;
    expect(cfg.comms_boost).toBe(false);
    expect(cfg.comms_boost_mode).toBe('auto');
    expect(cfg.vocal_isolation).toBe(false);
    expect(cfg.vocal_isolation_mode).toBe('off');
    expect((view.getByTestId('smart-audio-toggle') as HTMLInputElement).checked).toBe(true);
    expect((view.getByTestId('radio-comms-toggle') as HTMLInputElement).checked).toBe(false);

    fireEvent.click(view.getByTestId('advanced-audio-toggle'));
    const vocalModeSelect = view.getByTestId('vocal-isolation-mode-select') as HTMLSelectElement;
    fireEvent.change(vocalModeSelect, { target: { value: 'auto' } });
    expect(useJobStore.getState().pipelineConfig.vocal_isolation_mode).toBe('auto');
    expect(useJobStore.getState().pipelineConfig.vocal_isolation).toBe(false);
  });

  it('hides Dual-Source Sync toggle when disabled and exposes it with optional SRT pickers when enabled', () => {
    useJobStore.setState({
      systemInfo: {
        is_gemini_configured: true,
        gemini_model: 'gemini-3.8-flash',
        gemini_models: ['gemini-3.8-flash'],
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
        dual_source_pivot_enabled: false,
      },
    });
    const viewOff = render(<PipelineConfig />);
    expect(viewOff.queryByTestId('dual-source-pivot-toggle')).toBeNull();

    cleanup();
    useJobStore.setState((s) => ({
      systemInfo: s.systemInfo ? { ...s.systemInfo, dual_source_pivot_enabled: true } : null,
    }));
    const viewOn = render(<PipelineConfig />);
    const toggle = viewOn.getByTestId('dual-source-pivot-toggle') as HTMLButtonElement;
    expect(toggle).toBeTruthy();
    fireEvent.click(toggle);
    expect(useJobStore.getState().pipelineConfig.enable_dual_source_pivot).toBe(true);
    expect(viewOn.getByTestId('dual-source-fallback-order-select')).toBeTruthy();
    expect(viewOn.getByTestId('dual-source-smart-asr-toggle')).toBeTruthy();
    expect(viewOn.getByTestId('dual-source-en-input')).toBeTruthy();
    expect(viewOn.getByTestId('dual-source-ar-input')).toBeTruthy();
    expect(viewOn.getByTestId('pipeline-section-tabs')).toBeTruthy();
    fireEvent.click(viewOn.getByTestId('dual-source-direct-asr-toggle'));
    expect(useJobStore.getState().pipelineConfig.enable_dual_source_pivot).toBe(false);
    expect(useJobStore.getState().pipelineConfig.use_embedded_subtitles).toBe(false);
  });

  it('hides duplicate embedded-subtitles toggle when Smart Subtitles First is active and prioritizes embedded tracks automatically', () => {
    const probeWithSubs: ProbeResponse = {
      ...sampleProbe,
      subtitle_streams: [
        {
          index: 2,
          codec_name: 'subrip',
          language: 'eng',
          title: 'English',
          default: true,
          forced: false,
          text_based: true,
        },
      ],
    };
    useJobStore.setState((s) => ({
      probeResult: probeWithSubs,
      batchProbes: [probeWithSubs],
      systemInfo: s.systemInfo
        ? { ...s.systemInfo, dual_source_pivot_enabled: true }
        : {
            is_gemini_configured: true,
            gemini_model: 'gemini-3.8-flash',
            gemini_models: ['gemini-3.8-flash'],
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
            dual_source_pivot_enabled: true,
          },
    }));
    useJobStore.getState().updatePipelineConfig({
      enable_dual_source_pivot: true,
      use_embedded_subtitles: false,
    });

    const view = render(<PipelineConfig />);
    expect(view.queryByTestId('use-embedded-subtitles-toggle')).toBeNull();
    expect(view.getByTestId('dual-source-embedded-detected')).toBeTruthy();
    expect(view.getByTestId('whisper-fallback-mode-banner')).toBeTruthy();

    fireEvent.click(view.getByTestId('dual-source-direct-asr-toggle'));
    expect(useJobStore.getState().pipelineConfig.enable_dual_source_pivot).toBe(false);
    expect(useJobStore.getState().pipelineConfig.use_embedded_subtitles).toBe(false);
    expect(view.getByTestId('use-embedded-subtitles-toggle')).toBeTruthy();

    fireEvent.click(view.getByTestId('dual-source-pivot-toggle'));
    expect(useJobStore.getState().pipelineConfig.enable_dual_source_pivot).toBe(true);
    expect(useJobStore.getState().pipelineConfig.use_embedded_subtitles).toBe(true);
    expect(view.queryByTestId('use-embedded-subtitles-toggle')).toBeNull();
  });

  it('switches genre presets between Creator presets (flag off) and Series/Cinema presets (flag on)', () => {
    useJobStore.setState({
      systemInfo: {
        is_gemini_configured: true,
        gemini_model: 'gemini-3.8-flash',
        gemini_models: ['gemini-3.8-flash'],
        whisper_default_model: 'medium',
        whisper_models: ['medium', 'large-v3-turbo', 'large-v3'],
        max_cps_warning: 20,
        hardware: {
          cuda_available: false,
          device_count: 0,
          cpu_count: 8,
          recommended_device: 'cpu',
          recommended_compute_type: 'int8',
        },
        dual_source_pivot_enabled: false,
      },
    });
    const viewOff = render(<PipelineConfig />);
    expect(viewOff.getByTestId('genre-preset-youtube')).toBeTruthy();
    expect(viewOff.getByTestId('genre-preset-course')).toBeTruthy();
    expect(viewOff.queryByTestId('genre-preset-series')).toBeNull();
    expect(viewOff.queryByTestId('genre-preset-cinema')).toBeNull();

    cleanup();
    useJobStore.setState((s) => ({
      systemInfo: s.systemInfo ? { ...s.systemInfo, dual_source_pivot_enabled: true } : null,
    }));
    const viewOn = render(<PipelineConfig />);
    expect(viewOn.getByTestId('genre-preset-series')).toBeTruthy();
    expect(viewOn.getByTestId('genre-preset-cinema')).toBeTruthy();
    expect(viewOn.getByTestId('genre-preset-action')).toBeTruthy();
    expect(viewOn.queryByTestId('genre-preset-youtube')).toBeNull();
    expect(viewOn.queryByTestId('genre-preset-course')).toBeNull();

    fireEvent.click(viewOn.getByTestId('genre-preset-series'));
    expect(useJobStore.getState().pipelineConfig.enable_dual_source_pivot).toBe(true);
    expect(useJobStore.getState().pipelineConfig.use_embedded_subtitles).toBe(true);
  });
});
