// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { FirstRunApiKeyModal } from '../components/settings/ApiKeySection';
import { useJobStore } from '../store/useJobStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { api } from '../services/api';

/** First start with no AI provider configured: the modal offers all four providers. */
describe('FirstRunApiKeyModal', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({
      systemInfo: {
        is_gemini_configured: false,
        is_ai_configured: false,
        masked_api_key: null,
        gemini_model: 'gemini-3.8-flash',
        gemini_models: [],
        providers: {},
        whisper_default_model: 'medium',
        whisper_models: ['medium'],
        max_cps_warning: 20,
        hardware: {
          cuda_available: false,
          device_count: 0,
          cpu_count: 8,
          recommended_device: 'cpu',
          recommended_compute_type: 'int8',
        },
      } as any,
      fetchSystemInfo: vi.fn().mockResolvedValue(undefined),
    });
    useSettingsStore.setState({ load: vi.fn().mockResolvedValue(undefined) } as any);
  });

  it('offers every provider and adapts the key link and placeholder to the chosen one', () => {
    const { getByTestId, queryByTestId } = render(<FirstRunApiKeyModal />);
    for (const prov of ['google', 'openai', 'anthropic', 'ollama']) {
      expect(getByTestId(`modal-provider-tab-${prov}`)).toBeTruthy();
    }
    expect(getByTestId('modal-apikey-link').getAttribute('href')).toContain('aistudio.google.com');

    fireEvent.click(getByTestId('modal-provider-tab-openai'));
    expect(getByTestId('modal-apikey-link').getAttribute('href')).toContain('platform.openai.com');
    expect((getByTestId('modal-apikey-input') as HTMLInputElement).placeholder).toBe('sk-proj-...');

    fireEvent.click(getByTestId('modal-provider-tab-anthropic'));
    expect(getByTestId('modal-apikey-link').getAttribute('href')).toContain('console.anthropic.com');
    expect(queryByTestId('modal-apikey-baseurl')).toBeNull();
  });

  it('saves a key for the chosen provider through the multi-provider key vault', async () => {
    const addSpy = vi.spyOn(api, 'addProviderKey').mockResolvedValue({ keys: [], providers: {} } as any);
    const { getByTestId, queryByTestId } = render(<FirstRunApiKeyModal />);

    fireEvent.click(getByTestId('modal-provider-tab-anthropic'));
    const save = getByTestId('modal-apikey-save') as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(getByTestId('modal-apikey-input'), { target: { value: ' sk-ant-test ' } });
    expect(save.disabled).toBe(false);
    fireEvent.click(save);

    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    expect(addSpy.mock.calls[0][0]).toMatchObject({ provider: 'anthropic', api_key: 'sk-ant-test', verify: true });
    expect(addSpy.mock.calls[0][0].base_url).toBeUndefined();
    await waitFor(() => expect(queryByTestId('first-run-apikey-modal')).toBeNull());
  });

  it('connects local Ollama without a key, and Ollama Cloud only with one', async () => {
    const addSpy = vi.spyOn(api, 'addProviderKey').mockResolvedValue({ keys: [], providers: {} } as any);
    const { getByTestId, queryByTestId } = render(<FirstRunApiKeyModal />);

    fireEvent.click(getByTestId('modal-provider-tab-ollama'));
    expect(queryByTestId('modal-apikey-input')).toBeNull();
    expect(getByTestId('modal-apikey-link').getAttribute('href')).toContain('ollama.com/download');

    fireEvent.click(getByTestId('modal-ollama-cloud'));
    expect((getByTestId('modal-apikey-save') as HTMLButtonElement).disabled).toBe(true);
    expect(getByTestId('modal-apikey-input')).toBeTruthy();

    fireEvent.click(getByTestId('modal-ollama-local'));
    fireEvent.click(getByTestId('modal-apikey-save'));
    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    expect(addSpy.mock.calls[0][0]).toMatchObject({
      provider: 'ollama',
      api_key: '',
      base_url: 'http://localhost:11434',
      label: 'Local Ollama',
    });
  });

  it('keeps the dialog open to show a provider check problem until the user continues', async () => {
    vi.spyOn(api, 'addProviderKey').mockResolvedValue({
      keys: [],
      providers: {},
      discovery_error: 'Ollama is not running at http://localhost:11434',
    } as any);
    const { getByTestId, queryByTestId } = render(<FirstRunApiKeyModal />);

    fireEvent.click(getByTestId('modal-provider-tab-ollama'));
    fireEvent.click(getByTestId('modal-apikey-save'));

    await waitFor(() => expect(getByTestId('modal-apikey-notice').textContent).toContain('Ollama is not running'));
    fireEvent.click(getByTestId('modal-apikey-continue'));
    expect(queryByTestId('first-run-apikey-modal')).toBeNull();
  });
});
