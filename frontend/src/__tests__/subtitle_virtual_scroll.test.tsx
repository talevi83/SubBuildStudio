// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { VideoSyncPlayer } from '../components/studio/VideoSyncPlayer';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { Job, SubtitleSegment, SystemInfo } from '../types';

function makeSegments(count: number): SubtitleSegment[] {
  return Array.from({ length: count }, (_, idx) => {
    const start = idx * 3;
    return {
      id: idx + 1,
      job_id: 'job_vscroll',
      index: idx + 1,
      start_time: start,
      end_time: start + 2.5,
      original_text: `Original line ${idx + 1}`,
      translated_text: `שורה מתורגמת ${idx + 1}`,
      segment_type: 'DIALOGUE',
      speaker: idx % 2 === 0 ? 'SPEAKER_01' : 'SPEAKER_02',
      cps: 10,
    };
  });
}

function makeLargeJob(count = 200, overrides: Partial<Job> = {}): Job {
  return {
    id: 'job_vscroll',
    video_path: 'C:\\Videos\\Episode.mkv',
    file_name: 'Episode.mkv',
    file_size: 4096,
    is_upload: false,
    status: 'COMPLETED',
    current_stage: 'COMPLETED',
    progress_percent: 100,
    stage_progress_percent: null,
    video_duration: count * 3,
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
      SPEAKER_02: { label: 'Speaker 2', gender: 'unknown', color: '#F472B6' },
    },
    detect_onscreen_text: false,
    enable_translation: true,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3-flash-preview',
    segment_count: count,
    failed_translation_count: 0,
    created_at: '2026-11-23T01:00:00Z',
    updated_at: '2026-11-23T01:00:00Z',
    segments: makeSegments(count),
    ...overrides,
  };
}

const sysInfo: SystemInfo = {
  is_gemini_configured: true,
  is_ai_configured: true,
  gemini_model: 'gemini-3-flash-preview',
  gemini_models: ['gemini-3-flash-preview'],
  whisper_models: ['medium', 'large-v3-turbo'],
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

describe('SubtitleTable anti-flicker & measured virtual scroll', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useUiStore.setState({ language: 'he', shortcutsOpen: false });
    useJobStore.setState({
      systemInfo: sysInfo,
      activeJob: null,
      selectedSegmentIndex: null,
      problemFilter: 'NONE',
      speakerFilter: 'ALL',
      videoCurrentTime: 0,
      savingCount: 0,
      lastSavedAt: null,
      timingBusy: false,
      insertDraft: null,
    });
  });

  it('disables CSS scroll anchoring and uses fixed table column layout', () => {
    useJobStore.setState({ activeJob: makeLargeJob(10) });
    const { getByTestId, container } = render(<SubtitleTable />);

    const scrollContainer = getByTestId('subtitle-scroll-container');
    expect(scrollContainer.style.overflowAnchor).toBe('none');

    const table = container.querySelector('table');
    expect(table?.className).toContain('table-fixed');
  });

  it('uses measured row heights (104px) for spacer sizing and scroll offsets instead of hardcoded 64px', async () => {
    const origGetRect = HTMLElement.prototype.getBoundingClientRect;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      if (this.tagName === 'TR' && this.hasAttribute('data-row-key')) {
        return {
          width: 800,
          height: 104,
          top: 0,
          left: 0,
          bottom: 104,
          right: 800,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      }
      return origGetRect.call(this);
    });

    useJobStore.setState({ activeJob: makeLargeJob(200) });
    const { getByTestId, container } = render(<SubtitleTable />);

    const mountedInitial = container.querySelectorAll('tr[data-row-key]');
    expect(mountedInitial.length).toBeGreaterThan(0);
    expect(mountedInitial.length).toBeLessThan(200);

    const scrollContainer = getByTestId('subtitle-scroll-container');
    scrollContainer.scrollTop = 5200; // 5200 / 104 = row index 50; minus overscan 12 -> startIndex = 38
    fireEvent.scroll(scrollContainer);

    await waitFor(() => {
      const firstRow = container.querySelector('tr[data-row-key]');
      expect(firstRow?.getAttribute('data-row-key')).toBe('39');
    });

    const spacerTd = container.querySelector('tbody tr[aria-hidden="true"] td') as HTMLTableCellElement | null;
    expect(spacerTd?.style.height).toBe(`${38 * 104}px`);
  });

  it('does not re-render SubtitleTable when live job progress ticks arrive', () => {
    useJobStore.setState({
      activeJob: makeLargeJob(20, {
        status: 'DIARIZING',
        current_stage: 'DIARIZING',
        progress_percent: 64,
        stage_progress_percent: 10,
      }),
    });

    const onRender = vi.fn();
    render(
      <React.Profiler id="subtitle-table" onRender={onRender}>
        <SubtitleTable />
      </React.Profiler>
    );

    const initialRenderCount = onRender.mock.calls.length;
    expect(initialRenderCount).toBeGreaterThanOrEqual(1);

    useJobStore.setState((s) => ({
      activeJob: s.activeJob
        ? {
            ...s.activeJob,
            progress_percent: 68.5,
            stage_progress_percent: 55,
            speed_factor: 9.1,
            eta_seconds: 7,
          }
        : null,
    }));

    expect(onRender.mock.calls.length).toBe(initialRenderCount);
  });

  it('auto-scrolls below sticky thead for out-of-view rows, skips comfortably visible rows, and respects autoScroll toggle', async () => {
    useJobStore.setState({ activeJob: makeLargeJob(30) });
    const { getByTestId, getByLabelText } = render(<SubtitleTable />);

    const scrollContainer = getByTestId('subtitle-scroll-container');
    Object.defineProperty(scrollContainer, 'clientHeight', { configurable: true, value: 500 });
    Object.defineProperty(scrollContainer, 'scrollHeight', { configurable: true, value: 3036 });

    const origGetRect = HTMLElement.prototype.getBoundingClientRect;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      if (this === scrollContainer) {
        return {
          top: 100,
          bottom: 600,
          height: 500,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: 100,
          toJSON: () => ({}),
        } as DOMRect;
      }
      if (this.tagName === 'THEAD') {
        return {
          top: 100,
          bottom: 136,
          height: 36,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: 100,
          toJSON: () => ({}),
        } as DOMRect;
      }
      if (this.tagName === 'TR' && this.hasAttribute('data-row-key')) {
        const idx = Number(this.getAttribute('data-row-key')) - 1;
        const top = 100 + 36 + idx * 100 - scrollContainer.scrollTop;
        return {
          top,
          bottom: top + 100,
          height: 100,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: top,
          toJSON: () => ({}),
        } as DOMRect;
      }
      return origGetRect.call(this);
    });

    // Row #12 (idx=11, 33s..35.5s) is below the viewport -> scrolls to ~28% below sticky thead
    // visTop=136, visHeight=464, desiredTopY = 136 + floor(464*0.28) = 265; rowTop at 0 = 1236 -> scrollTop = 971
    act(() => {
      useJobStore.setState({ videoCurrentTime: 34 });
    });
    await waitFor(() => {
      expect(scrollContainer.scrollTop).toBe(971);
    });

    // Row #11 (idx=10, 30s..32.5s) tucked under sticky thead when scrollTop=1020 -> uncovers it
    scrollContainer.scrollTop = 1020;
    act(() => {
      useJobStore.setState({ videoCurrentTime: 31 });
    });
    await waitFor(() => {
      expect(scrollContainer.scrollTop).toBe(871);
    });

    // Row #12 (idx=11) is now comfortably inside [144, 584] (top=365, bottom=465) -> does not jump
    act(() => {
      useJobStore.setState({ videoCurrentTime: 34 });
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(scrollContainer.scrollTop).toBe(871);

    // Turning off auto-scroll stops following playback
    const toggle = getByLabelText('גלילה אוטומטית עם הווידאו') as HTMLInputElement;
    fireEvent.click(toggle);
    expect(toggle.checked).toBe(false);

    act(() => {
      useJobStore.setState({ videoCurrentTime: 76 }); // Row #26
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(scrollContainer.scrollTop).toBe(871);
  });

  it('auto-scrolls and mounts target row #120 when jumping far ahead in a virtualized 200-row table', async () => {
    const origGetRect = HTMLElement.prototype.getBoundingClientRect;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      if (this.getAttribute('data-testid') === 'subtitle-scroll-container') {
        return {
          top: 100,
          bottom: 600,
          height: 500,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: 100,
          toJSON: () => ({}),
        } as DOMRect;
      }
      if (this.tagName === 'THEAD') {
        return {
          top: 100,
          bottom: 136,
          height: 36,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: 100,
          toJSON: () => ({}),
        } as DOMRect;
      }
      if (this.tagName === 'TR' && this.hasAttribute('data-row-key')) {
        const idx = Number(this.getAttribute('data-row-key')) - 1;
        const scroller = this.closest('[data-testid="subtitle-scroll-container"]') as HTMLElement | null;
        const top = 100 + 36 + idx * 104 - (scroller?.scrollTop ?? 0);
        return {
          top,
          bottom: top + 104,
          height: 104,
          left: 0,
          right: 800,
          width: 800,
          x: 0,
          y: top,
          toJSON: () => ({}),
        } as DOMRect;
      }
      return origGetRect.call(this);
    });

    useJobStore.setState({ activeJob: makeLargeJob(200) });
    const { getByTestId, container } = render(<SubtitleTable />);

    const scrollContainer = getByTestId('subtitle-scroll-container');
    Object.defineProperty(scrollContainer, 'clientHeight', { configurable: true, value: 500 });
    Object.defineProperty(scrollContainer, 'scrollHeight', { configurable: true, value: 20836 });

    expect(container.querySelector('tr[data-row-key="120"]')).toBeNull();

    // Jump playback to row #120 (idx=119 -> start_time=357s)
    act(() => {
      useJobStore.setState({ videoCurrentTime: 358 });
    });

    await waitFor(() => {
      expect(container.querySelector('tr[data-row-key="120"]')).not.toBeNull();
    });
    expect(scrollContainer.scrollTop).toBeGreaterThan(11000);
  });

  it('syncs store videoCurrentTime immediately when scrubbing the VideoSyncPlayer seek bar', () => {
    useJobStore.setState({ activeJob: makeLargeJob(10) });
    const { container } = render(<VideoSyncPlayer />);

    const slider = container.querySelector('input[type="range"]') as HTMLInputElement | null;
    expect(slider).not.toBeNull();
    fireEvent.change(slider!, { target: { value: '18.5' } });

    expect(useJobStore.getState().videoCurrentTime).toBe(18.5);
  });
});
