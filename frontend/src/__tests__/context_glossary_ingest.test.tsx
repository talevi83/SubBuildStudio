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

  it('supports Smart Audio Map auto modes without enabling global comms_boost in Cinema preset', async () => {
    const view = render(<PipelineConfig />);
    const cinemaBtn = view.getByText(/Cinema & Narrative|קולנוע ועלילתי/i).closest('button');
    expect(cinemaBtn).toBeTruthy();
    fireEvent.click(cinemaBtn!);

    const cfg = useJobStore.getState().pipelineConfig;
    expect(cfg.comms_boost).toBe(false);
    expect(cfg.comms_boost_mode).toBe('auto');
    expect(cfg.vocal_isolation).toBe(false);
    expect(cfg.vocal_isolation_mode).toBe('off');

    const vocalModeSelect = view.getByTestId('vocal-isolation-mode-select') as HTMLSelectElement;
    fireEvent.change(vocalModeSelect, { target: { value: 'auto' } });
    expect(useJobStore.getState().pipelineConfig.vocal_isolation_mode).toBe('auto');
    expect(useJobStore.getState().pipelineConfig.vocal_isolation).toBe(false);
  });
});
