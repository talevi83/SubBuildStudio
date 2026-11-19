// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { areNeighbours } from '../components/studio/table/mergeRules';
import { api, errorDetail, UploadCancelled } from '../services/api';
import { hasUnsavedEdits, mergeSegments, useJobStore } from '../store/useJobStore';
import { hashBlobSha256 } from '../utils/sha256';
import type { Job, SubtitleSegment } from '../types';

const seg = (id: number, index: number, start: number, extra: Partial<SubtitleSegment> = {}): SubtitleSegment => ({
  id,
  job_id: 'job_fix',
  index,
  start_time: start,
  end_time: start + 1.5,
  original_text: `line ${index}`,
  translated_text: null,
  translation_failed: false,
  segment_type: 'DIALOGUE',
  confidence: 1,
  cps: 5,
  ...extra,
});

const makeJob = (segments: SubtitleSegment[]): Job => ({
  id: 'job_fix',
  video_path: 'C:\\Movies\\Episode02.mkv',
  file_name: 'Episode02.mkv',
  file_size: 1000,
  is_upload: false,
  status: 'COMPLETED',
  current_stage: 'COMPLETED',
  progress_percent: 100,
  video_duration: 60,
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
  segment_count: segments.length,
  failed_translation_count: 0,
  created_at: '2026-11-19T08:00:00Z',
  updated_at: '2026-11-19T08:00:00Z',
  segments,
});

describe('code review fixes', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({ activeJob: null, jobs: [], error: null });
  });

  it('shows FastAPI validation errors (HTTP 422 lists) instead of a generic message', () => {
    const body = {
      detail: [
        { loc: ['query', 'max_chars_per_line'], msg: 'Input should be greater than or equal to 10' },
        { loc: ['body', 'start_time'], msg: 'Input should be a finite number' },
      ],
    };
    expect(errorDetail(body, 'Request failed')).toBe(
      'max_chars_per_line: Input should be greater than or equal to 10; start_time: Input should be a finite number'
    );
    expect(errorDetail({ detail: 'Job is still running' }, 'x')).toBe('Job is still running');
    expect(errorDetail(null, 'Request failed')).toBe('Request failed');
    expect(errorDetail({ detail: [{}] }, 'Request failed')).toBe('Request failed');
  });

  it('applies translated batches by row id and keeps each row\u2019s own number and times', () => {
    const rows = [seg(11, 1, 1), seg(12, 2, 3), seg(13, 3, 5)];
    // The batch was built when row 12 was still number 3 and started at 9s
    const batch = [{ ...seg(12, 3, 9), translated_text: 'שתיים' }];
    const merged = mergeSegments(rows, batch);
    expect(merged.map((s) => s.translated_text)).toEqual([null, 'שתיים', null]);
    expect(merged[1]).toMatchObject({ index: 2, start_time: 3 });
    // Live preview rows (no id yet) are still matched by number
    expect(mergeSegments([{ ...rows[0], id: undefined }], [{ ...seg(0, 1, 1), id: undefined, translated_text: 'אחת' }])[0]
      .translated_text).toBe('אחת');
  });

  it('allows merging only neighbouring lines of the same lane', () => {
    const rows = [seg(1, 1, 1), seg(2, 2, 3), seg(9, 3, 3.6, { segment_type: 'ON_SCREEN' }), seg(3, 4, 5)];
    expect(areNeighbours(rows, new Set([1, 2]))).toBe(true);
    expect(areNeighbours(rows, new Set([2, 3]))).toBe(true); // the on-screen title in between doesn't count
    expect(areNeighbours(rows, new Set([1, 3]))).toBe(false);
    expect(areNeighbours(rows, new Set([3, 9]))).toBe(false); // dialogue + on-screen text
    expect(areNeighbours(rows, new Set([1]))).toBe(false);
  });

  it('disables the merge button for lines that are not neighbours', async () => {
    useJobStore.setState({ activeJob: makeJob([seg(1, 1, 1), seg(2, 2, 3), seg(3, 3, 5)]) });
    vi.spyOn(api, 'getHistory').mockResolvedValue({ cursor: null, can_undo: false, can_redo: false, entries: [] });
    const mergeSpy = vi.spyOn(api, 'mergeSegments');
    const view = render(<SubtitleTable />);
    const boxes = view.getAllByLabelText('Select this subtitle') as HTMLInputElement[];
    fireEvent.click(boxes[0]);
    fireEvent.click(boxes[2]);
    const button = view.getByTestId('rows-merge') as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.title).toContain('neighbouring');
    fireEvent.click(button);
    expect(mergeSpy).not.toHaveBeenCalled();
  });

  it('reports a failed row delete instead of silently doing nothing', async () => {
    useJobStore.setState({ activeJob: makeJob([seg(1, 1, 1), seg(2, 2, 3)]) });
    vi.spyOn(api, 'deleteSegment').mockRejectedValue(new Error('The job is still running; delete lines when it has finished'));
    vi.spyOn(api, 'deleteSegments').mockRejectedValue(new Error('Delete failed'));
    await useJobStore.getState().deleteSegment(1);
    expect(useJobStore.getState().error).toContain('still running');
    useJobStore.setState({ error: null });
    await useJobStore.getState().deleteSegments([1, 2]);
    expect(useJobStore.getState().error).toBe('Delete failed');
  });

  it('cancels an upload and the duplicate check before it', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(api.uploadFile(new File(['x'], 'a.mkv'), undefined, controller.signal)).rejects.toBeInstanceOf(
      UploadCancelled
    );
    expect(await hashBlobSha256(new Blob(['abc']), { signal: controller.signal })).toBeNull();
    const progress: number[] = [];
    expect(await hashBlobSha256(new Blob(['abc']), { onProgress: (p) => progress.push(p) })).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    );
    expect(progress.at(-1)).toBe(1);
  });

  it('knows when an inline edit has not been saved yet (for the close-window warning)', async () => {
    useJobStore.setState({ activeJob: makeJob([seg(1, 1, 1)]) });
    vi.spyOn(api, 'getHistory').mockResolvedValue({ cursor: null, can_undo: false, can_redo: false, entries: [] });
    vi.spyOn(api, 'updateSegment').mockResolvedValue(seg(1, 1, 1, { original_text: 'edited' }));
    const view = render(<SubtitleTable />);
    expect(hasUnsavedEdits()).toBe(false);
    const cell = view.getAllByRole('textbox').find((el) => (el as HTMLTextAreaElement).value === 'line 1')!;
    fireEvent.change(cell, { target: { value: 'edited' } });
    expect(hasUnsavedEdits()).toBe(true);
    fireEvent.blur(cell);
    await waitFor(() => expect(hasUnsavedEdits()).toBe(false));
  });
});
