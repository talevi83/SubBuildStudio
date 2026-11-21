// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, cleanup, act } from '@testing-library/react';
import {
  UpdateAvailableDialog,
  SKIPPED_VERSION_KEY,
  STARTUP_CHECK_DELAY_MS,
  WEBSITE_URL,
} from '../components/common/UpdateAvailableDialog';
import { api } from '../services/api';
import { useUiStore } from '../store/useUiStore';
import type { UpdateStatus } from '../types';

const status = (overrides: Partial<UpdateStatus> = {}): UpdateStatus => ({
  current_version: '2.2.3',
  current_commit: '',
  branch: '',
  update_available: true,
  update_kind: 'release',
  latest_version: '2.3.0',
  latest_commit: 'abc1234',
  commits_behind: 0,
  target_ref: 'v2.3.0',
  release_url: 'https://github.com/talevi83/SubBuildStudio/releases/tag/v2.3.0',
  release_notes: 'Faster translation',
  installer: 'windows',
  download_url: 'https://github.com/talevi83/SubBuildStudio/releases/download/v2.3.0/SubBuild-Studio-Setup.exe',
  error: null,
  ...overrides,
});

async function renderAfterStartup(res: UpdateStatus | Error) {
  const spy = vi.spyOn(api, 'getUpdateStatus');
  if (res instanceof Error) spy.mockRejectedValue(res);
  else spy.mockResolvedValue(res);
  const view = render(<UpdateAvailableDialog />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STARTUP_CHECK_DELAY_MS + 10);
  });
  return { ...view, spy };
}

describe('UpdateAvailableDialog', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    useUiStore.setState({ language: 'en' });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('offers the new installer to a Windows install, with steps and notes', async () => {
    const { getByTestId, spy } = await renderAfterStartup(status());
    expect(spy).toHaveBeenCalledTimes(1);
    expect(getByTestId('update-dialog-versions').textContent).toContain('v2.3.0');
    expect(getByTestId('update-dialog-notes').textContent).toBe('Faster translation');
    expect(getByTestId('update-dialog-steps').textContent).toContain('SubBuild-Studio-Setup.exe');
    expect(getByTestId('update-dialog-download').getAttribute('href')).toBe(status().download_url);
  });

  it('shows the macOS steps for the Mac app', async () => {
    const { getByTestId } = await renderAfterStartup(status({ installer: 'macos' }));
    expect(getByTestId('update-dialog-steps').textContent).toContain('Applications');
  });

  it('falls back to the website before the installer is attached to the release', async () => {
    const { getByTestId } = await renderAfterStartup(status({ download_url: null }));
    expect(getByTestId('update-dialog-download').getAttribute('href')).toBe(WEBSITE_URL);
  });

  it('stays hidden for a Git clone, when up to date, or when offline', async () => {
    for (const res of [status({ installer: null }), status({ update_available: false }), new Error('offline')]) {
      const { queryByTestId, unmount } = await renderAfterStartup(res);
      expect(queryByTestId('update-available-dialog')).toBeNull();
      unmount();
      vi.restoreAllMocks();
    }
  });

  it('"Skip this version" hides it for that version only', async () => {
    const first = await renderAfterStartup(status());
    fireEvent.click(first.getByTestId('update-dialog-skip'));
    expect(first.queryByTestId('update-available-dialog')).toBeNull();
    expect(localStorage.getItem(SKIPPED_VERSION_KEY)).toBe('2.3.0');
    first.unmount();
    vi.restoreAllMocks();

    const again = await renderAfterStartup(status());
    expect(again.queryByTestId('update-available-dialog')).toBeNull();
    again.unmount();
    vi.restoreAllMocks();

    const newer = await renderAfterStartup(status({ latest_version: '2.4.0' }));
    expect(newer.getByTestId('update-available-dialog')).toBeTruthy();
  });

  it('"Remind me later" closes it without remembering anything', async () => {
    const { getByTestId, queryByTestId } = await renderAfterStartup(status());
    fireEvent.click(getByTestId('update-dialog-later'));
    expect(queryByTestId('update-available-dialog')).toBeNull();
    expect(localStorage.getItem(SKIPPED_VERSION_KEY)).toBeNull();
  });
});
