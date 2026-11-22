// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { VideoSyncPlayer } from '../components/studio/VideoSyncPlayer';
import { EmbedVideoDialog } from '../components/studio/EmbedVideoDialog';
import { api } from '../services/api';
import { useJobStore } from '../store/useJobStore';
import {
  DEFAULT_SUBTITLE_STYLE,
  MAX_SUBTITLE_FONT_SIZE,
  MIN_SUBTITLE_FONT_SIZE,
  sanitizeSubtitleStyle,
  useUiStore,
} from '../store/useUiStore';
import type { Job } from '../types';

function makeJob(): Job {
  return {
    id: 'job_style',
    video_path: 'C:\\Videos\\Sample.mp4',
    file_name: 'Sample.mp4',
    file_size: 1024,
    is_upload: false,
    status: 'COMPLETED',
    current_stage: 'COMPLETED',
    progress_percent: 100,
    video_duration: 30,
    audio_streams: [],
    selected_audio_stream: 0,
    whisper_model: 'medium',
    source_language: 'en',
    detected_language: 'en',
    vad_filter: true,
    comms_boost: false,
    detect_onscreen_text: false,
    enable_translation: true,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3.8-flash',
    segment_count: 1,
    failed_translation_count: 0,
    created_at: '2026-11-22T10:00:00Z',
    updated_at: '2026-11-22T10:00:00Z',
    segments: [
      {
        id: 1,
        job_id: 'job_style',
        index: 1,
        start_time: 0,
        end_time: 4,
        original_text: 'Hello world',
        translated_text: 'שלום עולם',
        translation_failed: false,
        segment_type: 'DIALOGUE',
        confidence: 1,
        cps: 5,
      },
    ],
  };
}

describe('subtitle style customization', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.clear();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    useUiStore.getState().resetSubtitleStyle();
    useJobStore.setState({
      activeJob: makeJob(),
      subtitleOverlayMode: 'dual',
      selectedSegmentIndex: null,
      seekRequest: null,
    });
  });

  it('clamps font size (12..36px) and background opacity (0..100%) and persists to localStorage', () => {
    useUiStore.getState().setSubtitleStyle({
      fontFamily: 'Rubik',
      fontSizePx: 99,
      backgroundOpacity: 150,
      textColor: '#67e8f9',
      backgroundColor: '#1e1b4b',
    });

    const state = useUiStore.getState().subtitleStyle;
    expect(state).toEqual({
      fontFamily: 'Rubik',
      fontSizePx: MAX_SUBTITLE_FONT_SIZE,
      backgroundOpacity: 100,
      textColor: '#67E8F9',
      backgroundColor: '#1E1B4B',
    });

    useUiStore.getState().setSubtitleStyle({ fontSizePx: 4, backgroundOpacity: -20 });
    expect(useUiStore.getState().subtitleStyle.fontSizePx).toBe(MIN_SUBTITLE_FONT_SIZE);
    expect(useUiStore.getState().subtitleStyle.backgroundOpacity).toBe(0);

    const saved = JSON.parse(localStorage.getItem('subbuild.ui') || '{}');
    expect(saved.subtitleStyle.fontFamily).toBe('Rubik');
    expect(saved.subtitleStyle.fontSizePx).toBe(MIN_SUBTITLE_FONT_SIZE);
    expect(saved.subtitleStyle.backgroundOpacity).toBe(0);

    expect(
      sanitizeSubtitleStyle({
        fontFamily: 'InvalidFont' as never,
        textColor: 'not-a-color',
        backgroundColor: '#123',
      })
    ).toEqual(DEFAULT_SUBTITLE_STYLE);

    useUiStore.getState().resetSubtitleStyle();
    expect(useUiStore.getState().subtitleStyle).toEqual(DEFAULT_SUBTITLE_STYLE);
  });

  it('updates live subtitle overlay styling in VideoSyncPlayer', () => {
    const view = render(<VideoSyncPlayer />);
    fireEvent.click(view.getByTestId('open-subtitle-style'));

    const popover = view.getByTestId('subtitle-style-popover');
    expect(popover).toBeTruthy();

    fireEvent.change(view.getByTestId('subtitle-font-family'), { target: { value: 'Rubik' } });
    fireEvent.change(view.getByTestId('subtitle-font-size'), { target: { value: '28' } });
    fireEvent.change(view.getByTestId('subtitle-text-color'), { target: { value: '#67E8F9' } });
    fireEvent.change(view.getByTestId('subtitle-bg-color'), { target: { value: '#1E1B4B' } });
    fireEvent.change(view.getByTestId('subtitle-bg-opacity'), { target: { value: '40' } });

    const box = view.getByTestId('player-bottom-subtitle-box');
    expect(box.style.fontFamily).toContain('Rubik');
    expect(box.style.fontSize).toBe('28px');
    expect(box.style.backgroundColor).toBe('rgba(30, 27, 75, 0.4)');

    const translatedLine = view.getByText('שלום עולם');
    expect(translatedLine.style.color).toBe('rgb(103, 232, 249)');

    // 0% opacity turns off the box background and enables crisp outline shadow
    fireEvent.click(view.getByTestId('subtitle-opacity-preset-0'));
    expect(box.style.backgroundColor).toBe('transparent');
    expect(translatedLine.style.textShadow).toContain('rgba(0, 0, 0, 0.95)');

    fireEvent.click(view.getByTestId('subtitle-style-reset'));
    expect(useUiStore.getState().subtitleStyle).toEqual(DEFAULT_SUBTITLE_STYLE);
  });

  it('sends studio subtitle styling to burn-in embed API from EmbedVideoDialog', async () => {
    vi.spyOn(api, 'getLatestEmbedVideo').mockResolvedValue(null);
    const startSpy = vi.spyOn(api, 'startEmbedVideo').mockResolvedValue({
      task_id: 'task_1',
      state: 'done',
      mode: 'burn',
      percent: 100,
      path: 'C:\\Videos\\Sample.he.burned.mp4',
      file_name: 'Sample.he.burned.mp4',
      next_to_video: true,
      download_url: null,
      error: null,
      encoder: 'libx264',
    });

    useUiStore.getState().setSubtitleStyle({
      fontFamily: 'Assistant',
      fontSizePx: 26,
      textColor: '#FACC15',
      backgroundColor: '#000000',
      backgroundOpacity: 60,
    });

    const view = render(
      <EmbedVideoDialog
        jobId="job_style"
        isUpload={false}
        hasTranslation={true}
        applyRtl={true}
        onClose={vi.fn()}
        onDownload={vi.fn()}
      />
    );

    fireEvent.click(view.getByTestId('embed-mode-burn'));
    expect(view.getByTestId('embed-burn-style')).toBeTruthy();

    fireEvent.change(view.getByTestId('embed-bg-opacity'), { target: { value: '50' } });
    fireEvent.click(view.getByTestId('embed-start'));

    await waitFor(() => expect(startSpy).toHaveBeenCalledTimes(1));
    expect(startSpy).toHaveBeenCalledWith(
      'job_style',
      expect.objectContaining({
        mode: 'burn',
        font_size: 'large',
        font_family: 'Assistant',
        font_size_px: 26,
        text_color: '#FACC15',
        background_color: '#000000',
        background_opacity: 50,
      })
    );
  });
});
