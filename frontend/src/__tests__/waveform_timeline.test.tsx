// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { WaveformTimeline } from '../components/studio/WaveformTimeline';
import {
  applySegmentDrag,
  computeSameLaneBounds,
  LANE_GAP,
  MIN_SEGMENT_DURATION,
} from '../components/studio/waveformMath';
import { useJobStore } from '../store/useJobStore';
import { api } from '../services/api';
import type { Job, SubtitleSegment } from '../types';

const makeSegments = (): SubtitleSegment[] => [
  {
    id: 1,
    job_id: 'job_wf_1',
    index: 1,
    start_time: 1.0,
    end_time: 3.0,
    original_text: 'First dialogue line',
    translated_text: 'שורה ראשונה',
    segment_type: 'DIALOGUE',
    cps: 4.0,
  },
  {
    id: 2,
    job_id: 'job_wf_1',
    index: 2,
    start_time: 5.0,
    end_time: 7.0,
    original_text: 'Second dialogue line',
    translated_text: 'שורה שנייה',
    segment_type: 'DIALOGUE',
    cps: 4.5,
  },
  {
    id: 3,
    job_id: 'job_wf_1',
    index: 3,
    start_time: 1.5,
    end_time: 6.0,
    original_text: '[HEADQUARTERS]',
    translated_text: '[מפקדה]',
    segment_type: 'ON_SCREEN',
    cps: 2.0,
  },
];

const makeJob = (): Job => ({
  id: 'job_wf_1',
  video_path: 'C:\\Movies\\Ep1.mkv',
  file_name: 'Ep1.mkv',
  file_size: 2048000,
  is_upload: false,
  status: 'COMPLETED',
  current_stage: 'COMPLETED',
  progress_percent: 100,
  video_duration: 20,
  audio_streams: [],
  selected_audio_stream: 0,
  whisper_model: 'medium',
  source_language: 'en',
  vad_filter: true,
  comms_boost: false,
  detect_onscreen_text: true,
  allow_onscreen_overlap: true,
  enable_translation: true,
  target_language: 'Hebrew',
  gemini_model: 'gemini-3.8-flash',
  segment_count: 3,
  failed_translation_count: 0,
  created_at: '2026-11-22T12:00:00Z',
  updated_at: '2026-11-22T12:00:00Z',
  segments: makeSegments(),
});

describe('Interactive Dual-Lane WaveformTimeline', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.removeItem('subbuild.waveformVisible');
    useJobStore.setState({
      activeJob: makeJob(),
      jobs: [makeJob()],
      selectedSegmentIndex: null,
      savingCount: 0,
      error: null,
    });
  });

  it('clamps drags to same-lane neighbours while allowing ON_SCREEN overlap on separate lane', () => {
    const segs = makeSegments();
    const bounds = computeSameLaneBounds(segs[0], segs, 20, true);
    expect(bounds.minBound).toBe(0);
    // Bounded by dialogue seg #2 (starts at 5.0), NOT ON_SCREEN seg #3 (starts at 1.5)
    expect(bounds.maxBound).toBeCloseTo(5.0 - LANE_GAP, 3);

    // Extending end_time past neighbour is clamped cleanly
    const resized = applySegmentDrag('resize-end', 1.0, 3.0, 10.0, bounds.minBound, bounds.maxBound);
    expect(resized.start_time).toBe(1.0);
    expect(resized.end_time).toBeCloseTo(5.0 - LANE_GAP, 3);

    // Shrinking end_time below start + MIN_SEGMENT_DURATION clamps to minimum duration
    const shrunk = applySegmentDrag('resize-end', 1.0, 3.0, -5.0, bounds.minBound, bounds.maxBound);
    expect(shrunk.end_time).toBeCloseTo(1.0 + MIN_SEGMENT_DURATION, 3);

    // Moving whole block preserves duration and stops before neighbour
    const moved = applySegmentDrag('move', 1.0, 3.0, 10.0, bounds.minBound, bounds.maxBound);
    expect(moved.end_time - moved.start_time).toBeCloseTo(2.0, 3);
    expect(moved.end_time).toBeCloseTo(5.0 - LANE_GAP, 3);
  });

  it('toggles visibility, changes zoom levels, shows dual-lane badge, and commits one PATCH only on pointerup', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue({
      job_id: 'job_wf_1',
      peaks_per_second: 50,
      duration: 20,
      peaks: Array.from({ length: 1000 }, (_, i) => (i % 50 < 25 ? 0.8 : 0.1)),
    });
    const updateSpy = vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, segId, patch) => ({
      ...makeSegments().find((s) => s.id === segId)!,
      ...patch,
    }));
    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: 1,
      can_undo: true,
      can_redo: false,
      entries: [],
    });

    const onSeek = vi.fn();
    const view = render(<WaveformTimeline jobId="job_wf_1" currentTime={2.0} duration={20} onSeek={onSeek} />);

    await waitFor(() => {
      expect(api.getWaveform).toHaveBeenCalledWith('job_wf_1');
    });

    expect(view.getByTestId('waveform-dual-lane-badge')).toBeTruthy();
    expect(view.getByTestId('waveform-zoom-label').textContent).toBe('15s');

    fireEvent.click(view.getByTestId('waveform-zoom-in'));
    expect(view.getByTestId('waveform-zoom-label').textContent).toBe('10s');
    fireEvent.click(view.getByTestId('waveform-zoom-out'));
    fireEvent.click(view.getByTestId('waveform-zoom-out'));
    expect(view.getByTestId('waveform-zoom-label').textContent).toBe('20s');

    // Reset zoom back to 15s -> viewStart = 0, viewEnd = 15 (since 2.0 - 15*0.4 = -4 -> clamped to 0)
    fireEvent.click(view.getByTestId('waveform-zoom-in'));
    expect(view.getByTestId('waveform-zoom-label').textContent).toBe('15s');

    const canvas = view.getByTestId('waveform-canvas') as HTMLCanvasElement;
    // Mock 600px wide canvas box -> 40 px/sec (0..15s)
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 600,
      height: 108,
      right: 600,
      bottom: 108,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    // Segment #1 is DIALOGUE (1.0s..3.0s) -> x=40..120, main lane y=47..104 (center y=70)
    // Click empty area at t=10.0s (x=400, y=70) -> seeks to 10.0s
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 400, clientY: 70 });
    fireEvent.pointerUp(canvas, { button: 0, pointerId: 1, clientX: 400, clientY: 70 });
    expect(onSeek).toHaveBeenCalledWith(10);
    expect(updateSpy).not.toHaveBeenCalled();

    // Drag right edge of segment #1 (x=120, y=70) right by +40px (+1.0s) across multiple pointermove events
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, clientX: 120, clientY: 70 });
    fireEvent.pointerMove(canvas, { button: 0, pointerId: 2, clientX: 140, clientY: 70 });
    fireEvent.pointerMove(canvas, { button: 0, pointerId: 2, clientX: 160, clientY: 70 });

    // During drag (before pointerup): local preview updated in store, ZERO API calls sent!
    expect(useJobStore.getState().activeJob?.segments[0].end_time).toBeCloseTo(4.0, 2);
    expect(updateSpy).not.toHaveBeenCalled();

    // Release mouse -> single PATCH sent for clean Undo/Redo history step
    fireEvent.pointerUp(canvas, { button: 0, pointerId: 2, clientX: 160, clientY: 70 });
    await waitFor(() => {
      expect(updateSpy).toHaveBeenCalledTimes(1);
    });
    expect(updateSpy).toHaveBeenCalledWith('job_wf_1', 1, { start_time: 1.0, end_time: 4.0 });

    // Toggle hide/show waveform
    fireEvent.click(view.getByTestId('waveform-toggle'));
    expect(view.queryByTestId('waveform-canvas')).toBeNull();
    expect(localStorage.getItem('subbuild.waveformVisible')).toBe('false');
  });
});
