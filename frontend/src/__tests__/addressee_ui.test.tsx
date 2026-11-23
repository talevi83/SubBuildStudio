// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { addresseeChoice, isUncertainAddressee, manualAddressee } from '../components/studio/table/addressee';
import { api } from '../services/api';
import { mergeSegments, useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { AddresseeInfo, Job, SubtitleSegment } from '../types';

const sure: AddresseeInfo = {
  gender: 'female',
  number: 'singular',
  target: 'SPEAKER_02',
  label: 'שרה',
  confidence: 0.95,
  source: 'vocative',
  used: 'f',
};
const unresolved: AddresseeInfo = { gender: 'unknown', number: 'unknown', confidence: 0, source: 'unresolved' };

function makeJob(): Job {
  const row = (id: number, text: string, addressee?: AddresseeInfo | null): SubtitleSegment => ({
    id,
    job_id: 'job_addr_ui',
    index: id - 10,
    start_time: (id - 11) * 2,
    end_time: (id - 11) * 2 + 1.5,
    original_text: text,
    translated_text: 'תרגום',
    segment_type: 'DIALOGUE',
    cps: 8,
    addressee,
  });
  return {
    id: 'job_addr_ui',
    video_path: 'C:\\Videos\\Film.mp4',
    file_name: 'Film.mp4',
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
    detect_onscreen_text: false,
    enable_translation: true,
    target_language: 'Hebrew',
    gemini_model: 'gemini-3-flash-preview',
    segment_count: 3,
    failed_translation_count: 0,
    created_at: '2026-11-23T10:00:00Z',
    updated_at: '2026-11-23T10:00:00Z',
    segments: [
      row(11, 'Sarah, are you ready?', sure),
      row(12, 'Come here.', unresolved),
      row(13, 'The door is open.', null),
    ],
  };
}

describe('Addressee tag on subtitle rows', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    localStorage.clear();
    useUiStore.setState({ language: 'en', shortcutsOpen: false });
    useJobStore.setState({ activeJob: null, speakerFilter: 'ALL', problemFilter: 'NONE' });
  });

  it('maps a record to the choice the user sees and knows which ones are unsettled', () => {
    expect(addresseeChoice(sure)).toBe('female');
    expect(addresseeChoice({ ...sure, gender: 'mixed', number: 'plural' })).toBe('group');
    expect(addresseeChoice(unresolved)).toBe('');
    expect(addresseeChoice(null)).toBe('');
    expect(isUncertainAddressee(sure)).toBe(false);
    expect(isUncertainAddressee({ ...sure, confidence: 0.6 })).toBe(true);
    expect(isUncertainAddressee(unresolved)).toBe(true);
    expect(isUncertainAddressee({ ...unresolved, locked: true })).toBe(false);
    expect(isUncertainAddressee(undefined)).toBe(false);
    expect(manualAddressee('group')).toEqual({
      gender: 'mixed',
      number: 'plural',
      confidence: 1,
      source: 'manual',
      locked: true,
    });
    expect(manualAddressee(null)).toBeNull();
  });

  it('shows the tag only on rows that have one, filters the unsettled ones, and saves a manual pick', async () => {
    const job = makeJob();
    useJobStore.setState({ activeJob: job });
    vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, segId, patch) => {
      const seg = job.segments.find((s) => s.id === segId)!;
      return { ...seg, addressee: manualAddressee(patch.addressee_gender) };
    });

    const { getAllByTestId, getByTestId } = render(<SubtitleTable />);
    const pills = getAllByTestId('segment-addressee-pill') as HTMLSelectElement[];
    expect(pills).toHaveLength(2); // the line with nothing gendered has no tag
    expect(pills[0].value).toBe('female');
    expect(pills[0].dataset.uncertain).toBeUndefined();
    expect(pills[1].value).toBe('');
    expect(pills[1].dataset.uncertain).toBe('true');

    // Only the unsettled listener is a "problem"; the filter is offered because one exists
    fireEvent.change(getByTestId('problem-filter-select'), { target: { value: 'UNCERTAIN_ADDRESSEE' } });
    expect(getAllByTestId('segment-addressee-pill')).toHaveLength(1);
    fireEvent.change(getByTestId('problem-filter-select'), { target: { value: 'NONE' } });

    // Picking the listener by hand saves it, and the row shows the locked tag
    fireEvent.change(getAllByTestId('segment-addressee-pill')[1], { target: { value: 'male' } });
    await waitFor(() =>
      expect(api.updateSegment).toHaveBeenCalledWith('job_addr_ui', 12, { addressee_gender: 'male' })
    );
    await waitFor(() => {
      const row = useJobStore.getState().activeJob!.segments.find((s) => s.id === 12)!;
      expect(row.addressee).toEqual(manualAddressee('male'));
      expect('addressee_gender' in row).toBe(false);
    });
    expect((getAllByTestId('segment-addressee-pill')[1] as HTMLSelectElement).value).toBe('male');
  });

  it('takes the listener of a translated batch from the server', () => {
    const job = makeJob();
    const merged = mergeSegments(job.segments, [
      { ...job.segments[1], translated_text: 'בוא הנה.', addressee: { ...sure, gender: 'male', used: 'm' } },
    ]);
    expect(merged[1].addressee?.gender).toBe('male');
    expect(merged[0].addressee).toEqual(sure);
  });
});
