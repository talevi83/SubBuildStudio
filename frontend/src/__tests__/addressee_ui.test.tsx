// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { addresseeChoice, isUncertainAddressee } from '../components/studio/table/addressee';
import { api } from '../services/api';
import { mergeSegments, useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { AddresseeInfo, Job, SubtitleSegment, SystemInfo } from '../types';

const sure: AddresseeInfo = {
  gender: 'female',
  number: 'singular',
  target: 'SPEAKER_02',
  label: 'שרה',
  confidence: 0.95,
  source: 'vocative',
  used: 'f',
};
const sysInfo: SystemInfo = {
  is_gemini_configured: true,
  is_ai_configured: true,
  gemini_model: 'gemini-3-flash-preview',
  gemini_models: ['gemini-3-flash-preview'],
  whisper_models: ['medium'],
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
    useJobStore.setState({ systemInfo: sysInfo, activeJob: null, speakerFilter: 'ALL', problemFilter: 'NONE' });
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
  });

  it('shows a read-only tag on rows that have one and filters the unsettled ones', () => {
    const job = makeJob();
    job.segments[0].addressee = { ...sure, locked: true, source: 'manual', confidence: 1 };
    useJobStore.setState({ activeJob: job });
    const update = vi.spyOn(api, 'updateSegment');

    const { getAllByTestId, getByTestId } = render(<SubtitleTable />);
    const tags = getAllByTestId('segment-addressee-pill');
    expect(tags).toHaveLength(2); // the line with nothing gendered has no tag
    expect(tags.every((tag) => tag.tagName === 'SPAN')).toBe(true); // nothing to pick from here
    expect(tags[0].dataset.choice).toBe('female');
    expect(tags[0].dataset.locked).toBe('true');
    expect(tags[0].textContent).toContain('Woman');
    expect(tags[1].dataset.choice).toBe('unknown');
    expect(tags[1].dataset.uncertain).toBe('true');
    expect(tags[1].textContent).toContain('Unclear');
    expect(tags[1].getAttribute('title')).toContain('Speaking to a woman / man / group');

    // Only the unsettled listener is a "problem"; the filter is offered because one exists
    fireEvent.change(getByTestId('problem-filter-select'), { target: { value: 'UNCERTAIN_ADDRESSEE' } });
    expect(getAllByTestId('segment-addressee-pill')).toHaveLength(1);
    fireEvent.click(getAllByTestId('segment-addressee-pill')[0]);
    expect(update).not.toHaveBeenCalled();
  });

  it('fixes the listener through the selected rows: re-translates them as spoken to a woman', async () => {
    const job = makeJob();
    useJobStore.setState({ activeJob: job });
    const retranslate = vi.spyOn(api, 'retranslate').mockResolvedValue({ ...job, status: 'TRANSLATING' });

    const { getByTestId } = render(<SubtitleTable />);
    const checkboxes = document.querySelectorAll('tbody input[type="checkbox"]');
    fireEvent.click(checkboxes[1]);
    fireEvent.click(getByTestId('rows-to-woman'));
    await waitFor(() =>
      expect(retranslate).toHaveBeenCalledWith(
        'job_addr_ui',
        expect.objectContaining({ segment_ids: [12], addressee_gender: 'female' })
      )
    );
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
