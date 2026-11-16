// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SubtitleTable } from '../components/studio/SubtitleTable';
import { ExportToolbar } from '../components/studio/ExportToolbar';
import { SubtitleFilePicker } from '../components/ingest/SubtitleFilePicker';
import { MediaDropzone } from '../components/ingest/MediaDropzone';
import { PipelineConfig } from '../components/ingest/PipelineConfig';
import { useJobStore } from '../store/useJobStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { api } from '../services/api';
import { Sha256, hashBlobSha256 } from '../utils/sha256';
import type { Job, ParsedSubtitles, ProbeResponse } from '../types';

const makeJob = (): Job => ({
  id: 'job_roundtrip_1',
  video_path: 'C:\\Movies\\Episode01.mkv',
  file_name: 'Episode01.mkv',
  file_size: 1024000,
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
  segment_count: 1,
  failed_translation_count: 0,
  created_at: '2026-11-16T08:00:00Z',
  updated_at: '2026-11-16T08:00:00Z',
  segments: [
    {
      id: 101,
      job_id: 'job_roundtrip_1',
      index: 1,
      start_time: 1.0,
      end_time: 3.5,
      original_text: 'Hello world',
      translated_text: 'שלום עולם',
      translation_failed: false,
      segment_type: 'DIALOGUE',
      confidence: 0.99,
      cps: 3.6,
    },
  ],
});

describe('SubBuild Studio UI reliability & editor round-trip', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({
      activeJob: null,
      jobs: [],
      probeResult: null,
      syncFile: null,
      syncRawFile: null,
      syncLanguage: null,
      selectedSegmentIndex: null,
      savingCount: 0,
      lastSavedAt: null,
      error: null,
    });
  });

  it('flushes a Hebrew cue edit inside its 600ms autosave window and waits for in-flight save before exporting', async () => {
    const job = makeJob();
    useJobStore.setState({ activeJob: job, jobs: [job], applyRtlExport: false });

    // Server-side state simulated deterministically across PATCH /segments/{id} and POST /export/save-next-to-video
    const serverSegment = { ...job.segments[0] };
    const callOrder: string[] = [];
    let exportedSrtSnapshot = '';

    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: 1,
      can_undo: false,
      can_redo: false,
      entries: [],
    });

    vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, segId, patch) => {
      callOrder.push(`patch:start:${ JSON.stringify(patch) }`);
      // Simulate network latency so Export would race if it didn't await inflightSaves
      await new Promise((r) => setTimeout(r, 40));
      Object.assign(serverSegment, patch);
      callOrder.push(`patch:done:${segId}`);
      return { ...serverSegment };
    });

    vi.spyOn(api, 'saveNextToVideo').mockImplementation(async () => {
      callOrder.push('export:saveNextToVideo');
      const pad = (n: number, w = 2) => String(n).padStart(w, '0');
      const fmtTime = (sec: number) => {
        const ms = Math.round(sec * 1000);
        return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor((ms % 3600000) / 60000))}:${pad(
          Math.floor((ms % 60000) / 1000)
        )},${pad(ms % 1000, 3)}`;
      };
      exportedSrtSnapshot = `1\n${fmtTime(serverSegment.start_time)} --> ${fmtTime(serverSegment.end_time)}\n${
        serverSegment.translated_text
      }\n`;
      return { saved: true, path: 'C:\\Movies\\Episode01.he.srt', exists: false };
    });

    const view = render(
      <div>
        <SubtitleTable />
        <ExportToolbar />
      </div>
    );

    // 1. Nudge start timestamp (+100ms: 1.000 -> 1.100)
    const plusButtons = view.getAllByTitle('+100ms');
    fireEvent.click(plusButtons[0]);

    // 2. Edit the Hebrew cue text and DO NOT blur or wait 600ms
    const hebrewTextarea = view.getByDisplayValue('שלום עולם') as HTMLTextAreaElement;
    fireEvent.change(hebrewTextarea, { target: { value: 'שלום עולם מתוקן!' } });

    // 3. Immediately click Save Next to Video while still inside the 600ms debounce window
    const saveBtn = view.getByTestId('save-next-to-video');
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(exportedSrtSnapshot).toContain('שלום עולם מתוקן!');
    });

    // Verify both the timestamp edit and the final keystroke reached the exported SRT before export executed
    expect(exportedSrtSnapshot).toContain('00:00:01,100 --> 00:00:03,500');
    expect(exportedSrtSnapshot).toContain('שלום עולם מתוקן!');
    const exportIdx = callOrder.indexOf('export:saveNextToVideo');
    const textDoneIdx = callOrder.lastIndexOf('patch:done:101');
    expect(textDoneIdx).toBeGreaterThan(-1);
    expect(exportIdx).toBeGreaterThan(textDoneIdx);
  });

  it('verifies SHA-256 content hash before reusing an uploaded video with the same name and byte size', async () => {
    const storedByHash = new Map<string, ProbeResponse>();
    const uploadedFiles: string[] = [];
    const existingChecks: string[] = [];

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/media/upload/candidates') {
        const prefix = `${url.searchParams.get('name')}:${url.searchParams.get('size')}:`;
        const count = [...storedByHash.keys()].filter((k) => k.startsWith(prefix)).length;
        return new Response(JSON.stringify({ count }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (url.pathname === '/api/media/upload/existing') {
        existingChecks.push(url.searchParams.get('sha256') || '');
        const sha256 = url.searchParams.get('sha256') || '';
        const match = storedByHash.get(`${url.searchParams.get('name')}:${url.searchParams.get('size')}:${sha256}`) || null;
        return new Response(JSON.stringify(match), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    vi.spyOn(api, 'uploadFile').mockImplementation(async (file: File) => {
      const text = await file.text();
      uploadedFiles.push(text);
      const buf = await file.arrayBuffer();
      const digest = await globalThis.crypto.subtle.digest('SHA-256', buf);
      const sha256 = Array.from(new Uint8Array(digest))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
      const probe: ProbeResponse = {
        file_path: `/uploads/${uploadedFiles.length}_${file.name}`,
        file_name: file.name,
        file_size: file.size,
        duration: 10 + uploadedFiles.length,
        format_name: 'mp4',
        is_upload: true,
        video_streams: [],
        audio_streams: [],
      };
      storedByHash.set(`${file.name}:${file.size}:${sha256}`, probe);
      return probe;
    });

    const view = render(<MediaDropzone />);
    const dropzone = view.getByTestId('media-dropzone-area');

    const cut1 = new File(['AAAA_SAME_LENGTH_CUT'], 'final.mp4', { type: 'video/mp4' });
    const cut2 = new File(['BBBB_SAME_LENGTH_CUT'], 'final.mp4', { type: 'video/mp4' });
    expect(cut1.size).toBe(cut2.size);

    // Drop cut 1 -> uploaded
    fireEvent.drop(dropzone, { dataTransfer: { files: [cut1], getData: () => '' } });
    await waitFor(() => expect(useJobStore.getState().probeResult?.file_path).toBe('/uploads/1_final.mp4'));
    expect(uploadedFiles).toEqual(['AAAA_SAME_LENGTH_CUT']);
    // Nothing with this name and size existed yet, so the file was not hashed or looked up
    expect(existingChecks).toHaveLength(0);

    // Drop cut 2 (same name and byte count, different content) -> must NOT reuse cut 1; uploads cut 2
    fireEvent.drop(dropzone, { dataTransfer: { files: [cut2], getData: () => '' } });
    await waitFor(() => expect(useJobStore.getState().probeResult?.file_path).toBe('/uploads/2_final.mp4'));
    expect(uploadedFiles).toEqual(['AAAA_SAME_LENGTH_CUT', 'BBBB_SAME_LENGTH_CUT']);
    expect(existingChecks).toHaveLength(1);

    // Drop cut 1 again -> content hash matches cut 1, so it reuses /uploads/1_final.mp4 without re-uploading
    fireEvent.drop(dropzone, { dataTransfer: { files: [cut1], getData: () => '' } });
    await waitFor(() => expect(useJobStore.getState().probeResult?.file_path).toBe('/uploads/1_final.mp4'));
    expect(uploadedFiles).toHaveLength(2);
  });

  it('previews decoded subtitle cues on import and re-decodes original file bytes when user changes encoding', async () => {
    const parseCalls: { name: string; encoding?: string }[] = [];

    vi.spyOn(api, 'parseSubtitles').mockImplementation(async (file: File, encoding?: string) => {
      parseCalls.push({ name: file.name, encoding });
      const chosenEncoding = encoding || 'windows-1255';
      const decodedText = chosenEncoding === 'windows-1256' ? 'مرحبا بالعالم' : 'דַָֽׁ ַָבַבד';
      const language = chosenEncoding === 'windows-1256' ? 'Arabic' : 'Hebrew';
      const res: ParsedSubtitles = {
        file_name: file.name,
        format: 'srt',
        encoding: chosenEncoding,
        language,
        cue_count: 1,
        first_start: 1.0,
        last_end: 2.5,
        top_count: 0,
        sound_count: 0,
        cues: [{ start: 1.0, end: 2.5, text: decodedText, position: 'bottom', kind: 'speech' }],
      };
      return res;
    });

    const view = render(<SubtitleFilePicker />);
    const input = view.container.querySelector('input[type="file"]') as HTMLInputElement;
    const rawFile = new File([new Uint8Array([0xe3, 0xd1, 0xcd, 0xc8, 0xc7])], 'arabic.srt', { type: 'text/plain' });

    fireEvent.change(input, { target: { files: [rawFile] } });

    await waitFor(() => {
      expect(view.getByText('דַָֽׁ ַָבַבד')).toBeTruthy();
    });

    const encodingSelect = view.getByTestId('sync-encoding') as HTMLSelectElement;
    expect(encodingSelect.value).toBe('windows-1255');

    // User switches encoding to windows-1256 -> re-parses the original File bytes and updates preview + language
    fireEvent.change(encodingSelect, { target: { value: 'windows-1256' } });

    await waitFor(() => {
      expect(view.getByText('مرحبا بالعالم')).toBeTruthy();
    });
    expect(parseCalls).toEqual([
      { name: 'arabic.srt', encoding: undefined },
      { name: 'arabic.srt', encoding: 'windows-1256' },
    ]);
    expect(useJobStore.getState().syncLanguage).toBe('Arabic');
  });

  it('sends explicit false for metric conversion and SDH speaker removal when unchecked against a global true', async () => {
    vi.spyOn(api, 'listGlossaryProfiles').mockResolvedValue([]);
    const createdPayloads: Record<string, unknown>[] = [];
    vi.spyOn(api, 'createJob').mockImplementation(async (params) => {
      createdPayloads.push({ ...params });
      return makeJob();
    });
    vi.spyOn(useJobStore.getState(), 'connectWebSocket').mockImplementation(() => {});

    // Global settings have convert_to_metric and remove_sdh_speakers enabled
    useJobStore.getState().applySettingsDefaults(
      {
        convert_to_metric: true,
        remove_sdh_speakers: true,
        skip_end_credits: true,
        onscreen_position: 'top_center',
      },
      true
    );
    useSettingsStore.setState({
      loaded: true,
      catalog: [],
      settings: {
        translation: { model: 'gemini-3.8-flash', thinking: 'low', flex: false },
        glossary: { model: 'gemini-3.8-flash', thinking: 'low', flex: false },
        vision: { model: 'gemini-3.8-flash', thinking: 'minimal', flex: false },
        vision_media_resolution: 'medium',
        onscreen_position: 'top_center',
        convert_to_metric: true,
        remove_sdh_speakers: true,
        skip_end_credits: true,
        cost_saver_mode: false,
      },
    });
    useJobStore.setState({
      probeResult: {
        file_path: 'C:\\Movies\\Episode01.mkv',
        file_name: 'Episode01.mkv',
        file_size: 1024000,
        duration: 60,
        format_name: 'matroska',
        video_streams: [],
        audio_streams: [],
      },
    });

    const view = render(<PipelineConfig />);

    // Both checkboxes start checked from the global default; user explicitly unchecks them for this job
    const checkboxes = Array.from(view.container.querySelectorAll('input[type="checkbox"]')) as HTMLInputElement[];
    const metricCheckbox = checkboxes.find((cb) => cb.parentElement?.textContent?.includes('📏'))!;
    const sdhCheckbox = checkboxes.find((cb) => cb.parentElement?.textContent?.includes('🔇'))!;
    expect(metricCheckbox.checked).toBe(true);
    expect(sdhCheckbox.checked).toBe(true);

    fireEvent.click(metricCheckbox);
    fireEvent.click(sdhCheckbox);
    expect(metricCheckbox.checked).toBe(false);
    expect(sdhCheckbox.checked).toBe(false);

    fireEvent.click(view.getByTestId('launch'));

    await waitFor(() => {
      expect(createdPayloads).toHaveLength(1);
    });
    expect(createdPayloads[0].convert_to_metric).toBe(false);
    expect(createdPayloads[0].remove_sdh_speakers).toBe(false);
  });

  it('computes identical SHA-256 digests when streaming arbitrary chunk boundaries vs Web Crypto', async () => {
    const sample = new Uint8Array(500);
    for (let i = 0; i < sample.length; i++) sample[i] = (i * 31 + 17) & 0xff;

    const streaming = new Sha256();
    for (let pos = 0; pos < sample.length; pos += 13) {
      streaming.update(sample.subarray(pos, Math.min(pos + 13, sample.length)));
    }
    const streamHex = streaming.digestHex();
    const blobHex = await hashBlobSha256(new Blob([sample]));
    expect(streamHex).toBe(blobHex);
  });

  it('flushes pending inline text edits before replaceAll runs', async () => {
    const job = makeJob();
    useJobStore.setState({ activeJob: job, jobs: [job] });

    const serverSegment = { ...job.segments[0] };
    const callOrder: string[] = [];

    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: 1,
      can_undo: false,
      can_redo: false,
      entries: [],
    });
    vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, _segId, patch) => {
      callOrder.push('patch:single');
      Object.assign(serverSegment, patch);
      return { ...serverSegment };
    });
    vi.spyOn(api, 'updateSegments').mockImplementation(async (_jobId, patches) => {
      callOrder.push('patch:bulk');
      return patches.map((p) => ({ ...serverSegment, ...p }));
    });

    const view = render(<SubtitleTable />);
    const hebrewTextarea = view.getByDisplayValue('שלום עולם') as HTMLTextAreaElement;
    // Edit text to include a word we'll immediately replace without blurring or waiting 600ms
    fireEvent.change(hebrewTextarea, { target: { value: 'שלום חבר יקר' } });

    const count = await useJobStore.getState().replaceAll('חבר', 'ידיד', 'translated', false);
    expect(count).toBe(1);
    expect(callOrder).toEqual(['patch:single', 'patch:bulk']);
    expect(useJobStore.getState().activeJob?.segments[0].translated_text).toBe('שלום ידיד יקר');
  });

  it('allows direct inline timecode editing and flags overlapping adjacent cues', async () => {
    const job = makeJob();
    job.segments = [
      { ...job.segments[0], id: 101, index: 1, start_time: 1.0, end_time: 3.5 },
      { ...job.segments[0], id: 102, index: 2, start_time: 4.0, end_time: 6.0, original_text: 'Next', translated_text: 'הבא' },
    ];
    useJobStore.setState({ activeJob: job, jobs: [job] });

    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: 1,
      can_undo: false,
      can_redo: false,
      entries: [],
    });
    vi.spyOn(api, 'updateSegment').mockImplementation(async (_jobId, segId, patch) => {
      const seg = useJobStore.getState().activeJob!.segments.find((s) => s.id === segId)!;
      return { ...seg, ...patch };
    });

    const view = render(<SubtitleTable />);

    // Click end-time of cue #1 and edit it to overlap cue #2 (4.5s > 4.0s)
    fireEvent.click(view.getByTestId('end-time-1'));
    const input = view.getByTestId('end-time-1-input') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '00:00:04,500' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => {
      expect(useJobStore.getState().activeJob?.segments[0].end_time).toBe(4.5);
    });

    // Both overlapping cues should now show the overlap warning icon
    const warnings = view.getAllByLabelText('Time overlaps an adjacent subtitle');
    expect(warnings).toHaveLength(2);
  });

  it('supports autosave status, split/merge/shift controls, search navigation, and bilingual export URL', async () => {
    const job = makeJob();
    job.segments = [
      {
        ...job.segments[0],
        id: 101,
        index: 1,
        start_time: 1.0,
        end_time: 5.0,
        original_text: 'Hello world friend',
        translated_text: 'שלום עולם חבר',
      },
      {
        ...job.segments[0],
        id: 102,
        index: 2,
        start_time: 6.0,
        end_time: 8.0,
        original_text: 'Goodbye world',
        translated_text: 'להתראות עולם',
      },
    ];
    useJobStore.setState({ activeJob: job, jobs: [job], applyRtlExport: true });

    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: 1,
      can_undo: false,
      can_redo: false,
      entries: [],
    });

    let resolvePatch!: (seg: Job['segments'][0]) => void;
    vi.spyOn(api, 'updateSegment').mockImplementation(
      () =>
        new Promise((res) => {
          resolvePatch = res;
        })
    );

    const splitSpy = vi.spyOn(api, 'splitSegment').mockResolvedValue({
      changed: 2,
      retimed: true,
      created_id: 103,
      segments: [
        { ...job.segments[0], id: 101, index: 1, end_time: 3.0, original_text: 'Hello', translated_text: 'שלום' },
        { ...job.segments[0], id: 103, index: 2, start_time: 3.0, end_time: 5.0, original_text: 'world friend', translated_text: 'עולם חבר' },
        { ...job.segments[1], id: 102, index: 3 },
      ],
    });
    const mergeSpy = vi.spyOn(api, 'mergeSegments').mockResolvedValue({
      changed: 2,
      retimed: true,
      created_id: 101,
      segments: [
        { ...job.segments[0], id: 101, index: 1, start_time: 1.0, end_time: 5.0, original_text: 'Hello\nworld friend' },
        { ...job.segments[1], id: 102, index: 2 },
      ],
    });
    const shiftSpy = vi.spyOn(api, 'shiftSegments').mockResolvedValue({
      changed: 2,
      retimed: true,
      segments: [
        { ...job.segments[0], id: 101, index: 1, start_time: 2.0, end_time: 6.0 },
        { ...job.segments[1], id: 102, index: 2, start_time: 7.0, end_time: 9.0 },
      ],
    });

    const view = render(
      <div>
        <SubtitleTable />
        <ExportToolbar />
      </div>
    );

    // 1. Autosave status indicator transitions from Saving… -> Saved
    fireEvent.click(view.getAllByTitle('+100ms')[0]);
    expect(view.getByTestId('autosave-status').textContent).toContain('Saving');
    resolvePatch({ ...job.segments[0], start_time: 1.1 });
    await waitFor(() => {
      expect(view.getByTestId('autosave-status').textContent).toContain('Saved');
    });

    // 2. Bilingual SRT export button builds URL with bilingual=true
    const clickedHrefs: string[] = [];
    const origClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      clickedHrefs.push(this.getAttribute('href') || '');
    };
    try {
      fireEvent.click(view.getByTestId('export-bilingual-srt'));
      await waitFor(() => expect(clickedHrefs.some((h) => h.includes('bilingual=true'))).toBe(true));
    } finally {
      HTMLAnchorElement.prototype.click = origClick;
    }

    // 3. Split cue #1
    fireEvent.click(view.getByTestId('split-row-1'));
    await waitFor(() => expect(splitSpy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(useJobStore.getState().activeJob?.segments).toHaveLength(3));

    // 4. Select rows #1 and #2 via checkboxes and click Merge
    const rowCheckboxes = view.getAllByLabelText('Select this subtitle') as HTMLInputElement[];
    fireEvent.click(rowCheckboxes[0]);
    fireEvent.click(rowCheckboxes[1]);
    fireEvent.click(view.getByTestId('rows-merge'));
    await waitFor(() => expect(mergeSpy).toHaveBeenCalledWith('job_roundtrip_1', [101, 103]));
    await waitFor(() => expect(useJobStore.getState().activeJob?.segments).toHaveLength(2));

    // 5. Shift times by +1.0s
    fireEvent.click(view.getByTestId('shift-time-toggle'));
    fireEvent.change(view.getByTestId('shift-time-input'), { target: { value: '1.0' } });
    fireEvent.click(view.getByTestId('shift-time-apply'));
    await waitFor(() => expect(shiftSpy).toHaveBeenCalledWith('job_roundtrip_1', 1, undefined));

    // 6. Search navigation Next / Prev match
    useJobStore.setState({ selectedSegmentIndex: null });
    const searchInput = view.getByPlaceholderText('Search subtitle text...') as HTMLInputElement;
    fireEvent.change(searchInput, { target: { value: 'world' } });
    await waitFor(() => expect(view.getByTestId('search-next-match')).toBeTruthy());
    fireEvent.click(view.getByTestId('search-next-match'));
    expect(useJobStore.getState().selectedSegmentIndex).toBe(1);
    fireEvent.click(view.getByTestId('search-next-match'));
    expect(useJobStore.getState().selectedSegmentIndex).toBe(2);
    fireEvent.click(view.getByTestId('search-prev-match'));
    expect(useJobStore.getState().selectedSegmentIndex).toBe(1);
  });

  it('preserves live Whisper transcribed segments when switching to another project and returning', async () => {
    const liveJob: Job = {
      ...makeJob(),
      id: 'job_live_1',
      status: 'TRANSCRIBING',
      current_stage: 'TRANSCRIBING',
      progress_percent: 35,
      segments: [
        {
          job_id: 'job_live_1',
          index: 1,
          start_time: 1.0,
          end_time: 2.5,
          original_text: 'Live transcribed line one',
          translated_text: null,
          translation_failed: false,
          segment_type: 'DIALOGUE',
          confidence: 0.96,
          cps: 16.0,
        },
        {
          job_id: 'job_live_1',
          index: 2,
          start_time: 3.0,
          end_time: 5.0,
          original_text: 'Live transcribed line two',
          translated_text: null,
          translation_failed: false,
          segment_type: 'DIALOGUE',
          confidence: 0.92,
          cps: 12.5,
        },
      ],
    };
    const otherJob: Job = {
      ...makeJob(),
      id: 'job_other_2',
      file_name: 'OtherEpisode.mkv',
      segments: [
        {
          id: 501,
          job_id: 'job_other_2',
          index: 1,
          start_time: 0.5,
          end_time: 2.0,
          original_text: 'Other completed line',
          translated_text: 'שורה מפרויקט אחר',
          translation_failed: false,
          segment_type: 'DIALOGUE',
          confidence: 0.99,
          cps: 10.0,
        },
      ],
    };

    useJobStore.setState({ activeJob: liveJob, jobs: [liveJob, otherJob] });
    vi.spyOn(useJobStore.getState(), 'connectWebSocket').mockImplementation(() => {});
    vi.spyOn(api, 'getHistory').mockResolvedValue({
      cursor: null,
      can_undo: false,
      can_redo: false,
      entries: [],
    });
    vi.spyOn(api, 'getJob').mockImplementation(async (jobId: string) => {
      if (jobId === 'job_other_2') return otherJob;
      if (jobId === 'job_live_1') return liveJob;
      throw new Error(`Unexpected job id ${jobId}`);
    });

    const view = render(<SubtitleTable />);
    expect(view.getByDisplayValue('Live transcribed line one')).toBeTruthy();

    // Switch to another project
    await useJobStore.getState().selectJob('job_other_2');
    await waitFor(() => expect(view.getByDisplayValue('Other completed line')).toBeTruthy());

    // Switch back to the transcribing project
    await useJobStore.getState().selectJob('job_live_1');
    await waitFor(() => expect(view.getByDisplayValue('Live transcribed line one')).toBeTruthy());
    expect(view.getByDisplayValue('Live transcribed line two')).toBeTruthy();

    // Also ensure refreshActiveJob with a slightly lagging response never drops newer live WS segments
    const newerWsSeg = {
      job_id: 'job_live_1',
      index: 3,
      start_time: 5.5,
      end_time: 7.0,
      original_text: 'Third live WS cue',
      translated_text: null,
      translation_failed: false,
      segment_type: 'DIALOGUE' as const,
      confidence: 0.94,
      cps: 11.3,
    };
    useJobStore.setState({
      activeJob: { ...liveJob, segments: [...liveJob.segments, newerWsSeg] },
    });
    await useJobStore.getState().refreshActiveJob();
    expect(useJobStore.getState().activeJob?.segments.map((s) => s.index)).toEqual([1, 2, 3]);
  });
});
