// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { ErrorToast, NoticeToast } from '../App';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { ShortcutsDialog } from '../components/common/ShortcutsDialog';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { translate } from '../i18n';
import { localizeError } from '../i18n/serverErrors';
import { api } from '../services/api';
import { confirmDialog } from '../store/useDialogStore';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import { buildSearchPattern, planReplacements } from '../utils/textSearch';
import type { Job, SubtitleSegment } from '../types';

const seg = (id: number, index: number, text: string, translated: string | null = null): SubtitleSegment => ({
  id,
  job_id: 'job_ux',
  index,
  start_time: index * 2,
  end_time: index * 2 + 1.5,
  original_text: text,
  translated_text: translated,
  translation_failed: false,
  segment_type: 'DIALOGUE',
  confidence: 1,
  cps: 5,
});

const makeJob = (segments: SubtitleSegment[]): Job => ({
  id: 'job_ux',
  video_path: 'C:\\Movies\\Ep.mkv',
  file_name: 'Ep.mkv',
  file_size: 1,
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

const en = (key: Parameters<typeof translate>[1], vars?: Record<string, string | number>) => translate('en', key, vars);
const he = (key: Parameters<typeof translate>[1], vars?: Record<string, string | number>) => translate('he', key, vars);

describe('search options', () => {
  const rows = [seg(1, 1, 'Tom met tom'), seg(2, 2, 'Tomato soup'), seg(3, 3, 'שלום לכולם, שלומית')];

  it('is case-insensitive by default and honours "match case"', () => {
    const loose = buildSearchPattern('tom', { useRegex: false, matchCase: false, wholeWord: false }).regex!;
    expect(planReplacements(rows, 'original_text', loose, 'X', false).map((p) => p.after)).toEqual([
      'X met X',
      'Xato soup',
    ]);
    const strict = buildSearchPattern('tom', { useRegex: false, matchCase: true, wholeWord: false }).regex!;
    expect(planReplacements(rows, 'original_text', strict, 'X', false).map((p) => p.after)).toEqual(['Tom met X']);
  });

  it('matches whole words, Hebrew included', () => {
    const word = buildSearchPattern('tom', { useRegex: false, matchCase: false, wholeWord: true }).regex!;
    expect(planReplacements(rows, 'original_text', word, 'X', false).map((p) => p.id)).toEqual([1]);
    const hebrew = buildSearchPattern('שלום', { useRegex: false, matchCase: false, wholeWord: true }).regex!;
    expect(planReplacements(rows, 'original_text', hebrew, 'היי', false)[0].after).toBe('היי לכולם, שלומית');
  });

  it('takes "$1" literally in plain text and as a group in regex mode', () => {
    const plain = buildSearchPattern('soup', { useRegex: false, matchCase: false, wholeWord: false }).regex!;
    expect(planReplacements(rows, 'original_text', plain, '$1', false)[0].after).toBe('Tomato $1');
    const re = buildSearchPattern('(\\w+) soup', { useRegex: true, matchCase: false, wholeWord: false }).regex!;
    expect(planReplacements(rows, 'original_text', re, 'hot $1', true)[0].after).toBe('hot Tomato');
    expect(buildSearchPattern('(', { useRegex: true, matchCase: false, wholeWord: false }).error).toBeTruthy();
  });
});

describe('UX improvements', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({ activeJob: null, jobs: [], error: null, notice: null });
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    vi.spyOn(api, 'getHistory').mockResolvedValue({ cursor: null, can_undo: false, can_redo: false, entries: [] });
  });

  it('previews Replace All and saves only after confirming', async () => {
    useJobStore.setState({ activeJob: makeJob([seg(1, 1, 'a', 'Tom'), seg(2, 2, 'b', 'tom'), seg(3, 3, 'c', 'Tim')]) });
    const save = vi.spyOn(api, 'updateSegments').mockResolvedValue([]);
    const view = render(<SubtitleTable />);
    fireEvent.change(view.getByPlaceholderText('Search subtitle text...'), { target: { value: 'Tom' } });
    fireEvent.change(view.getByPlaceholderText(en('table.replace')), { target: { value: 'Ron' } });
    fireEvent.click(view.getByTestId('search-match-case'));
    fireEvent.click(view.getByTestId('replace-all'));
    expect(view.getByTestId('replace-preview').textContent).toContain(en('ux.replacePreview', { count: 1 }));
    expect(save).not.toHaveBeenCalled();
    fireEvent.click(view.getByTestId('replace-confirm'));
    await waitFor(() => expect(save).toHaveBeenCalledWith('job_ux', [{ id: 1, translated_text: 'Ron' }]));
  });

  it('asks with the app\u2019s own dialog and resolves with the answer', async () => {
    const view = render(<ConfirmDialog />);
    let answer: Promise<boolean>;
    act(() => {
      answer = confirmDialog({ title: 'Delete project "Ep.mkv"?', danger: true });
    });
    expect(view.getByTestId('confirm-dialog').textContent).toContain('Delete project');
    expect(document.activeElement).toBe(view.getByTestId('confirm-dialog-cancel')); // the safe choice
    fireEvent.click(view.getByTestId('confirm-dialog-ok'));
    await expect(answer!).resolves.toBe(true);

    act(() => {
      answer = confirmDialog({ title: 'Again?' });
    });
    fireEvent.keyDown(window, { key: 'Escape' });
    await expect(answer!).resolves.toBe(false);
    expect(view.queryByTestId('confirm-dialog')).toBeNull();
  });

  it('deletes a row without asking and offers Undo', async () => {
    useJobStore.setState({ activeJob: makeJob([seg(1, 1, 'a'), seg(2, 2, 'b')]) });
    vi.spyOn(api, 'deleteSegment').mockResolvedValue({ changed: 1, retimed: false, segments: [seg(2, 1, 'b')] });
    const undo = vi.spyOn(api, 'undo').mockResolvedValue({
      segments: [seg(1, 1, 'a'), seg(2, 2, 'b')],
      history: { cursor: 1, can_undo: false, can_redo: true, entries: [] },
    });
    const confirmSpy = vi.spyOn(window, 'confirm');
    const view = render(
      <>
        <SubtitleTable />
        <NoticeToast />
      </>
    );
    fireEvent.click(view.getByTestId('delete-row-1'));
    await waitFor(() => expect(view.getByTestId('notice-toast').textContent).toContain(en('ux.rowDeleted', { index: 1 })));
    expect(confirmSpy).not.toHaveBeenCalled();
    fireEvent.click(view.getByTestId('notice-undo'));
    await waitFor(() => expect(undo).toHaveBeenCalledWith('job_ux'));
    await waitFor(() => expect(useJobStore.getState().activeJob?.segments).toHaveLength(2));
  });

  it('keeps several errors on screen, in the UI language', async () => {
    useUiStore.setState({ language: 'he' });
    const view = render(<ErrorToast />);
    act(() => useJobStore.setState({ error: 'The job is still running; delete lines when it has finished' }));
    act(() => useJobStore.setState({ error: 'Failed to fetch' }));
    const alerts = await view.findAllByRole('alert');
    expect(alerts.map((a) => a.textContent)).toEqual([
      expect.stringContaining(he('ux.err.stillRunning')),
      expect.stringContaining(he('ux.err.serverDown')),
    ]);
    fireEvent.click(view.getByText(he('ux.dismissAll')));
    expect(view.queryAllByRole('alert')).toHaveLength(0);
    expect(useJobStore.getState().error).toBeNull();
  });

  it('translates known server errors and leaves unknown ones as they are', () => {
    expect(localizeError('Overlaps line #4 (00:00:08,000 - 00:00:09,500)', he)).toBe(
      he('ux.err.overlaps', { index: '4', range: '00:00:08,000 - 00:00:09,500' })
    );
    expect(localizeError('Could not save subtitle: Failed to fetch', he)).toBe(
      he('ux.err.saveFailed', { message: he('ux.err.serverDown') })
    );
    expect(localizeError('Something new went wrong', he)).toBe('Something new went wrong');
  });

  it('opens the shortcuts window with "?" but not while typing', () => {
    const view = render(
      <>
        <input data-testid="field" />
        <ShortcutsDialog />
      </>
    );
    fireEvent.keyDown(view.getByTestId('field'), { key: '?' });
    expect(view.queryByTestId('shortcuts-dialog')).toBeNull();
    fireEvent.keyDown(window, { key: '?' });
    expect(view.getByTestId('shortcuts-dialog').textContent).toContain(en('ux.kSplit'));
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(view.queryByTestId('shortcuts-dialog')).toBeNull();
  });
});

describe('player shortcuts', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  });

  it('moves to the next / previous line and splits at the playhead', async () => {
    const { VideoSyncPlayer } = await import('../components/studio/VideoSyncPlayer');
    useJobStore.setState({
      activeJob: makeJob([seg(1, 1, 'one'), seg(2, 2, 'two'), seg(3, 3, 'three')]),
      selectedSegmentIndex: null,
    });
    const split = vi.spyOn(useJobStore.getState(), 'splitSegment').mockResolvedValue(null);
    const view = render(<VideoSyncPlayer />);
    const video = view.container.querySelector('video')!;
    video.currentTime = 4.2; // inside line 2 (4.0 - 5.5)

    fireEvent.keyDown(window, { key: 'ArrowDown' });
    expect(useJobStore.getState().selectedSegmentIndex).toBe(3);
    expect(video.currentTime).toBe(6);

    fireEvent.keyDown(window, { key: 'ArrowUp' });
    fireEvent.keyDown(window, { key: 'ArrowUp' });
    expect(useJobStore.getState().selectedSegmentIndex).toBe(1);

    useJobStore.setState({ selectedSegmentIndex: 2 });
    video.currentTime = 4.8;
    fireEvent.keyDown(window, { key: 's' });
    expect(split).toHaveBeenCalledWith(2, 4.8);
  });
});
