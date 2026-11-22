// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, cleanup, act } from '@testing-library/react';
import {
  GpuLibsDialog,
  GPU_LIBS_DISMISSED_KEY,
  GPU_OFFER_DELAY_MS,
  GPU_POLL_MS,
} from '../components/common/GpuLibsDialog';
import { openGpuLibsDialog } from '../utils/gpuLibs';
import { api } from '../services/api';
import { useJobStore } from '../store/useJobStore';
import { useUiStore } from '../store/useUiStore';
import type { GpuLibsStatus, SystemInfo } from '../types';

const gpuStatus = (overrides: Partial<GpuLibsStatus> = {}): GpuLibsStatus => ({
  supported: true,
  installed: false,
  installing: false,
  error: null,
  download_size_mb: 1500,
  ...overrides,
});

function setHardware(gpuLibsMissing: boolean) {
  useJobStore.setState({
    systemInfo: {
      hardware: {
        cuda_available: false,
        device_count: 0,
        gpu_name: 'NVIDIA GeForce RTX 2070 SUPER',
        cpu_count: 8,
        recommended_device: 'cpu',
        recommended_compute_type: 'int8',
        gpu_libs_missing: gpuLibsMissing,
      },
    } as unknown as SystemInfo,
  });
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe('GpuLibsDialog', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    useUiStore.setState({ language: 'en' });
    useJobStore.setState({ fetchSystemInfo: vi.fn(async () => undefined) });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('offers the download when an NVIDIA card has no CUDA libraries, then follows it to the end', async () => {
    setHardware(true);
    const statusSpy = vi.spyOn(api, 'getGpuLibsStatus').mockResolvedValue(gpuStatus());
    const installSpy = vi.spyOn(api, 'installGpuLibs').mockResolvedValue(gpuStatus({ installing: true }));
    const { getByTestId, queryByTestId } = render(<GpuLibsDialog />);
    expect(queryByTestId('gpu-libs-dialog')).toBeNull();
    await advance(GPU_OFFER_DELAY_MS + 10);
    expect(getByTestId('gpu-libs-body').textContent).toContain('RTX 2070 SUPER');
    expect(getByTestId('gpu-libs-install').textContent).toContain('1.5 GB');

    fireEvent.click(getByTestId('gpu-libs-install'));
    await advance(0);
    expect(installSpy).toHaveBeenCalledTimes(1);
    expect(getByTestId('gpu-libs-installing')).toBeTruthy();

    statusSpy.mockResolvedValue(gpuStatus({ installed: true }));
    await advance(GPU_POLL_MS + 10);
    expect(getByTestId('gpu-libs-done')).toBeTruthy();
    expect(useJobStore.getState().fetchSystemInfo).toHaveBeenCalled();
  });

  it('shows the error and lets the user try again', async () => {
    setHardware(true);
    vi.spyOn(api, 'getGpuLibsStatus').mockResolvedValue(gpuStatus());
    vi.spyOn(api, 'installGpuLibs').mockResolvedValue(gpuStatus({ error: 'Failed to fetch' }));
    const { getByTestId } = render(<GpuLibsDialog />);
    await advance(GPU_OFFER_DELAY_MS + 10);
    fireEvent.click(getByTestId('gpu-libs-install'));
    await advance(0);
    expect(getByTestId('gpu-libs-failed').textContent).toContain('Failed to fetch');
    expect(getByTestId('gpu-libs-install').textContent).toBe('Try again');
  });

  it('stays quiet without an NVIDIA card, outside the AppImage, or after "Don\'t ask again"', async () => {
    const statusSpy = vi.spyOn(api, 'getGpuLibsStatus').mockResolvedValue(gpuStatus());
    setHardware(false);
    const first = render(<GpuLibsDialog />);
    await advance(GPU_OFFER_DELAY_MS + 10);
    expect(first.queryByTestId('gpu-libs-dialog')).toBeNull();
    expect(statusSpy).not.toHaveBeenCalled();
    cleanup();

    setHardware(true);
    statusSpy.mockResolvedValue(gpuStatus({ supported: false }));
    const second = render(<GpuLibsDialog />);
    await advance(GPU_OFFER_DELAY_MS + 10);
    expect(second.queryByTestId('gpu-libs-dialog')).toBeNull();
    cleanup();

    statusSpy.mockResolvedValue(gpuStatus());
    const third = render(<GpuLibsDialog />);
    await advance(GPU_OFFER_DELAY_MS + 10);
    fireEvent.click(third.getByTestId('gpu-libs-never'));
    expect(localStorage.getItem(GPU_LIBS_DISMISSED_KEY)).toBe('1');
    cleanup();

    statusSpy.mockClear();
    const fourth = render(<GpuLibsDialog />);
    await advance(GPU_OFFER_DELAY_MS + 10);
    expect(fourth.queryByTestId('gpu-libs-dialog')).toBeNull();
    expect(statusSpy).not.toHaveBeenCalled();

    // Settings' button still opens it
    act(() => openGpuLibsDialog());
    await advance(0);
    expect(fourth.getByTestId('gpu-libs-dialog')).toBeTruthy();
  });
});
