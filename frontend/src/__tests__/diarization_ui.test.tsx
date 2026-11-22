// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { PipelineConfig } from '../components/ingest/PipelineConfig';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { GlossaryDialog } from '../components/studio/GlossaryDialog';
import { EmbedVideoDialog } from '../components/studio/EmbedVideoDialog';
import { api } from '../services/api';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { Job, ProbeResponse, SystemInfo } from '../types';

function makeDiarizedJob(): Job {
  return {
    id: 'job_diar_ui',
    video_path: 'C:\\Videos\\Podcast.mp4',
    file_name: 'Podcast.mp4',
    file_size: 4096,
    is_upload: false,
    status: 'COMPLETED',
    current_stage: 'COMPLETED',
    progress_percent: 100,
    video_duration: 60,
    audio_streams: [],
    selected_audio_stream: 0,
    whisper_model: 'large-v3-turbo',
    source_language: 'en',
    detected_language: 'en',
    vad_filter: true,
    comms_boost: false,
    enable_diarization: true,
    speakers: {
      SPEAKER_01: { label: 'Speaker 1', gender: 'unknown', color: '#38BDF8' },
      SPEAKER_02: { label: 'Sarah', gender: 'female', color: '#F472B6' },
    },
    detect_onscreen_text: false,
    enable_translation: true,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3-flash-preview',
    segment_count: 2,
    failed_translation_count: 0,
    created_at: '2026-11-22T10:00:00Z',
    updated_at: '2026-11-22T10:00:00Z',
    segments: [
      {
        id: 11,
        job_id: 'job_diar_ui',
        index: 1,
        start_time: 0,
        end_time: 2,
        original_text: 'Welcome back everyone.',
        translated_text: 'ברוכים השבים.',
        segment_type: 'DIALOGUE',
        speaker: 'SPEAKER_01',
        cps: 8,
      },
      {
        id: 12,
        job_id: 'job_diar_ui',
        index: 2,
        start_time: 2.2,
        end_time: 4.5,
        original_text: 'Happy to be here!',
        translated_text: 'שמחה להיות כאן!',
        segment_type: 'DIALOGUE',
        speaker: 'SPEAKER_02',
        cps: 8,
      },
    ],
  };
}

const probe: ProbeResponse = {
  file_path: 'C:\\Videos\\Podcast.mp4',
  file_name: 'Podcast.mp4',
  file_size: 4096,
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
  whisper_models: ['medium', 'large-v3-turbo', 'large-v3'],
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

describe('Speaker Diarization Studio UI', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.clear();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    useJobStore.setState((s) => ({
      systemInfo: sysInfo,
      probeResult: probe,
      ingestMode: 'transcribe',
      activeJob: null,
      speakerFilter: 'ALL',
      problemFilter: 'NONE',
      pipelineConfig: { ...s.pipelineConfig, enable_diarization: false },
    }));
  });

  it('toggles acoustic diarization in PipelineConfig and enables it automatically on podcast preset', () => {
    const { getByTestId } = render(<PipelineConfig />);
    const toggle = getByTestId('toggle-diarization') as HTMLInputElement;
    expect(toggle.checked).toBe(false);

    fireEvent.click(toggle);
    expect(useJobStore.getState().pipelineConfig.enable_diarization).toBe(true);

    fireEvent.click(getByTestId('genre-preset-youtube'));
    expect(useJobStore.getState().pipelineConfig.enable_diarization).toBe(false);

    fireEvent.click(getByTestId('genre-preset-podcast'));
    expect(useJobStore.getState().pipelineConfig.enable_diarization).toBe(true);
  });

  it('renders speaker pills, filters by speaker, and supports inline & bulk speaker assignment', async () => {
    const job = makeDiarizedJob();
    useJobStore.setState({ activeJob: job });
    vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, segId, patch) => {
      const seg = job.segments.find((s) => s.id === segId)!;
      return { ...seg, ...patch };
    });
    vi.spyOn(api, 'updateSegments').mockImplementation(async (_jobId, patches) =>
      patches.map((p) => {
        const seg = job.segments.find((s) => s.id === p.id)!;
        return { ...seg, ...p };
      })
    );

    const { getAllByTestId, getByTestId } = render(<SubtitleTable />);
    expect(getAllByTestId('segment-speaker-pill')).toHaveLength(2);

    // Filter down to SPEAKER_02
    fireEvent.change(getByTestId('speaker-filter'), { target: { value: 'SPEAKER_02' } });
    expect(getAllByTestId('segment-speaker-pill')).toHaveLength(1);
    expect((getAllByTestId('segment-speaker-pill')[0] as HTMLSelectElement).value).toBe('SPEAKER_02');

    // Clear filter and reassign row 1 speaker
    fireEvent.change(getByTestId('speaker-filter'), { target: { value: 'ALL' } });
    fireEvent.change(getAllByTestId('segment-speaker-pill')[0], { target: { value: 'SPEAKER_02' } });
    await waitFor(() => expect(api.updateSegment).toHaveBeenCalledWith('job_diar_ui', 11, { speaker: 'SPEAKER_02' }));

    // Bulk assign first row via checkbox
    const checkboxes = document.querySelectorAll('tbody input[type="checkbox"]');
    fireEvent.click(checkboxes[0]);
    fireEvent.change(getByTestId('bulk-speaker-select'), { target: { value: 'SPEAKER_01' } });
    await waitFor(() => expect(api.updateSegments).toHaveBeenCalledWith('job_diar_ui', [{ id: 11, speaker: 'SPEAKER_01' }]));
  });

  it('edits speaker label & gender, detects speakers from audio, and saves in GlossaryDialog', async () => {
    const job = makeDiarizedJob();
    const onDiarize = vi.fn().mockResolvedValue(true);
    const onSaveSpeakers = vi.fn().mockResolvedValue(true);
    const onSave = vi.fn().mockResolvedValue(undefined);

    const { getByTestId } = render(
      <GlossaryDialog
        glossary={job.glossary}
        profile={job.glossary_profile}
        canTranslate={true}
        speakers={job.speakers}
        segments={job.segments}
        onDiarize={onDiarize}
        onSaveSpeakers={onSaveSpeakers}
        onSave={onSave}
        onClose={() => undefined}
      />
    );

    fireEvent.click(getByTestId('diarize-audio-btn'));
    await waitFor(() => expect(onDiarize).toHaveBeenCalledTimes(1));

    fireEvent.change(getByTestId('speaker-label-SPEAKER_01'), { target: { value: 'Avi' } });
    fireEvent.change(getByTestId('speaker-gender-SPEAKER_01'), { target: { value: 'male' } });
    fireEvent.click(getByTestId('glossary-save'));

    await waitFor(() =>
      expect(onSaveSpeakers).toHaveBeenCalledWith(
        expect.objectContaining({
          SPEAKER_01: expect.objectContaining({ label: 'Avi', gender: 'male' }),
        })
      )
    );
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it('offers Color-code speakers checkbox in EmbedVideoDialog when speakers exist', () => {
    useJobStore.setState({ activeJob: makeDiarizedJob() });
    const { getByTestId } = render(
      <EmbedVideoDialog
        jobId="job_diar_ui"
        isUpload={false}
        hasTranslation={true}
        applyRtl={true}
        hasSpeakers={true}
        onClose={() => undefined}
        onDownload={() => undefined}
      />
    );
    fireEvent.click(getByTestId('embed-mode-burn'));
    const checkbox = getByTestId('embed-color-speakers') as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });
});
