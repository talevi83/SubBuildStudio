// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { ExportToolbar } from '../components/studio/ExportToolbar';
import { VideoSyncPlayer } from '../components/studio/VideoSyncPlayer';
import { SubtitleStylePopover } from '../components/studio/SubtitleStylePopover';
import { resolveDisplayWords } from '../utils/wordTiming';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { Job } from '../types';

function makeJob(): Job {
  return {
    id: 'job_creator',
    video_path: 'C:\\Videos\\Creator.mp4',
    file_name: 'Creator.mp4',
    file_size: 2048,
    is_upload: false,
    status: 'COMPLETED',
    current_stage: 'COMPLETED',
    progress_percent: 100,
    video_duration: 20,
    audio_streams: [],
    selected_audio_stream: 0,
    whisper_model: 'large-v3-turbo',
    source_language: 'en',
    detected_language: 'en',
    vad_filter: true,
    comms_boost: false,
    detect_onscreen_text: false,
    enable_translation: true,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3-flash-preview',
    segment_count: 1,
    failed_translation_count: 0,
    created_at: '2026-11-22T10:00:00Z',
    updated_at: '2026-11-22T10:00:00Z',
    segments: [
      {
        id: 1,
        job_id: 'job_creator',
        index: 1,
        start_time: 0.0,
        end_time: 3.0,
        original_text: 'One Two Three',
        translated_text: 'אחת שתיים שלוש',
        translation_failed: false,
        segment_type: 'DIALOGUE',
        confidence: 1,
        cps: 5,
        words: [
          { w: 'One', s: 0.0, e: 1.0 },
          { w: 'Two', s: 1.0, e: 2.0 },
          { w: 'Three', s: 2.0, e: 3.0 },
        ],
      },
    ],
  };
}

describe('Creator word-by-word highlight & More Formats exports', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.clear();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    useUiStore.getState().resetSubtitleStyle();
    useJobStore.setState({
      activeJob: makeJob(),
      subtitleOverlayMode: 'original',
      applyRtlExport: true,
      selectedSegmentIndex: null,
      seekRequest: null,
    });
  });

  it('resolveDisplayWords reuses aligned stored word timings and falls back proportionally when mismatched', () => {
    const stored = [
      { w: 'One', s: 0.1, e: 0.9 },
      { w: 'Two', s: 0.9, e: 2.1 },
    ];
    const reused = resolveDisplayWords('One Two', 0, 3, stored);
    expect(reused[0]).toEqual({ w: 'One', s: 0.1, e: 0.9 });
    expect(reused[1].e).toBe(3);

    const fallback = resolveDisplayWords('A LongerWord', 0, 10, stored.slice(0, 1));
    expect(fallback).toHaveLength(2);
    expect(fallback[0].s).toBe(0);
    expect(fallback[1].e).toBe(10);
    expect(fallback[1].s).toBeLessThan(fallback[1].e);
  });

  it('highlights active word and karaoke sweep in VideoSyncPlayer', () => {
    useUiStore.getState().setSubtitleStyle({ highlightMode: 'word', highlightColor: '#FACC15' });
    const { container, rerender } = render(<VideoSyncPlayer />);

    // At t = 0.0s -> first word active ("One")
    const activeWord = container.querySelectorAll('[data-active-word="true"]');
    expect(activeWord).toHaveLength(1);
    expect(activeWord[0].textContent).toBe('One');

    // Switch to karaoke at t = 1.5 via seek
    useUiStore.getState().setSubtitleStyle({ highlightMode: 'karaoke', highlightColor: '#22D3EE' });
    rerender(<VideoSyncPlayer />);
    const videoEl = container.querySelector('video')!;
    Object.defineProperty(videoEl, 'currentTime', { value: 1.5, configurable: true });
    fireEvent.seeked(videoEl);

    const sweptWords = [...container.querySelectorAll('[data-active-word="true"]')].map((n) => n.textContent);
    expect(sweptWords).toEqual(['One', 'Two']);
  });

  it('allows changing highlight mode in SubtitleStylePopover', () => {
    const anchorRef = { current: null };
    const { getByTestId } = render(
      <SubtitleStylePopover open={true} onClose={() => undefined} anchorRef={anchorRef} />
    );
    fireEvent.change(getByTestId('style-highlight-mode'), { target: { value: 'karaoke' } });
    expect(useUiStore.getState().subtitleStyle.highlightMode).toBe('karaoke');
    expect(getByTestId('subtitle-highlight-color')).toBeTruthy();
  });

  it('keeps SRT/VTT buttons visible and downloads ASS/FCPXML/MD/DOCX via More Formats menu', async () => {
    const urls: string[] = [];
    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag);
      if (tag === 'a') {
        vi.spyOn(el, 'click').mockImplementation(() => {
          urls.push((el as HTMLAnchorElement).getAttribute('href') || '');
        });
      }
      return el;
    });

    const { getByTestId } = render(<ExportToolbar />);
    expect(getByTestId('export-translated-srt')).toBeTruthy();
    expect(getByTestId('export-original-srt')).toBeTruthy();
    expect(getByTestId('export-vtt')).toBeTruthy();

    for (const testId of ['export-ass', 'export-fcpxml', 'export-markdown', 'export-docx']) {
      fireEvent.click(getByTestId('more-exports-toggle'));
      expect(getByTestId('more-exports-menu')).toBeTruthy();
      fireEvent.click(getByTestId(testId));
    }

    await waitFor(() => expect(urls).toHaveLength(4));
    expect(urls[0]).toContain('/export/ass?');
    expect(urls[1]).toContain('/export/fcpxml?');
    expect(urls[2]).toContain('/export/markdown?');
    expect(urls[3]).toContain('/export/docx?');
  });
});
