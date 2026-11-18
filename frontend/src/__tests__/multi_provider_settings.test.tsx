// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { ApiKeySection } from '../components/settings/ApiKeySection';
import { SettingsScreen } from '../components/settings/SettingsScreen';
import { RowTranslateDialog } from '../components/studio/RowTranslateDialog';
import { ExportToolbar } from '../components/studio/ExportToolbar';
import { Sidebar } from '../components/layout/Sidebar';
import { useJobStore } from '../store/useJobStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { api } from '../services/api';
import type { AppSettings, CatalogModel, ProviderKeyRecord, ProviderSummary } from '../types';

const sampleCatalog: CatalogModel[] = [
  {
    id: 'gemini-3.8-flash',
    provider: 'google',
    label: 'Gemini 3.8 Flash',
    family: '3',
    preview: false,
    available: true,
    key_ids: ['env_google'],
    is_local: false,
    thinking: ['default', 'low', 'medium', 'high'],
    default_level: 'medium',
    input_per_m: 0.75,
    output_per_m: 3.75,
    episode_cost_usd: 0.145,
    movie_cost_usd: 0.34,
    episode_vision_cost_usd: 0.32,
    movie_vision_cost_usd: 0.32,
    supports_media_resolution: true,
    tasks: ['translation', 'glossary', 'vision'],
    recommended_for: ['translation', 'glossary', 'vision'],
    warnings_by_task: {},
    supports_flex: true,
    flex_price_factor: 0.5,
  },
  {
    id: 'gpt-5-mini',
    provider: 'openai',
    label: 'GPT-5 mini',
    family: 'openai',
    preview: false,
    available: true,
    key_ids: ['key_oai_1'],
    is_local: false,
    thinking: ['default'],
    default_level: 'default',
    input_per_m: 0.25,
    output_per_m: 2.0,
    episode_cost_usd: 0.0617,
    movie_cost_usd: 0.1406,
    episode_vision_cost_usd: 0.1151,
    movie_vision_cost_usd: 0.1151,
    supports_media_resolution: false,
    tasks: ['translation', 'glossary', 'vision'],
    recommended_for: ['translation', 'glossary', 'vision'],
    warnings_by_task: {},
    supports_flex: false,
  },
  {
    id: 'gpt-5-nano',
    provider: 'openai',
    label: 'GPT-5 nano',
    family: 'openai',
    preview: false,
    available: true,
    key_ids: ['key_oai_1'],
    is_local: false,
    thinking: ['default'],
    default_level: 'default',
    input_per_m: 0.05,
    output_per_m: 0.4,
    episode_cost_usd: 0.0123,
    movie_cost_usd: 0.0281,
    episode_vision_cost_usd: 0.023,
    movie_vision_cost_usd: 0.023,
    supports_media_resolution: false,
    tasks: ['vision'],
    recommended_for: ['vision'],
    warnings_by_task: {},
    supports_flex: false,
  },
  {
    id: 'qwen2.5:14b',
    provider: 'ollama',
    label: 'qwen2.5:14b',
    family: 'ollama',
    preview: false,
    available: true,
    key_ids: ['local_ollama_default'],
    is_local: true,
    thinking: ['default'],
    default_level: 'default',
    input_per_m: 0,
    output_per_m: 0,
    episode_cost_usd: 0,
    movie_cost_usd: 0,
    episode_vision_cost_usd: null,
    movie_vision_cost_usd: null,
    supports_media_resolution: false,
    tasks: ['translation', 'glossary'],
    recommended_for: ['translation', 'glossary'],
    warnings_by_task: {},
    supports_flex: false,
  },
  {
    id: 'qwen2.5vl:7b',
    provider: 'ollama',
    label: 'qwen2.5vl:7b',
    family: 'ollama',
    preview: false,
    available: true,
    key_ids: ['local_ollama_default'],
    is_local: true,
    thinking: ['default'],
    default_level: 'default',
    input_per_m: 0,
    output_per_m: 0,
    episode_cost_usd: 0,
    movie_cost_usd: 0,
    episode_vision_cost_usd: 0,
    movie_vision_cost_usd: 0,
    supports_media_resolution: false,
    tasks: ['translation', 'glossary', 'vision'],
    recommended_for: ['translation', 'glossary', 'vision'],
    warnings_by_task: {},
    supports_flex: false,
  },
  {
    id: 'gpt-unverified-seed',
    provider: 'openai',
    label: 'GPT Unverified Seed',
    family: 'openai',
    preview: false,
    available: false,
    key_ids: [],
    is_local: false,
    thinking: ['default'],
    default_level: 'default',
    input_per_m: 1.0,
    output_per_m: 4.0,
    episode_cost_usd: 0.2,
    movie_cost_usd: 0.4,
    supports_media_resolution: false,
    tasks: ['translation', 'glossary', 'vision'],
    recommended_for: [],
    warnings_by_task: {},
    supports_flex: false,
  },
];

const initialSettings: AppSettings = {
  translation: { provider: 'google', model: 'gemini-3.8-flash', key_id: null, thinking: 'low', flex: false },
  glossary: { provider: 'google', model: 'gemini-3.8-flash', key_id: null, thinking: 'low', flex: false },
  vision: { provider: 'google', model: 'gemini-3.8-flash', key_id: null, thinking: 'low', flex: false },
  vision_media_resolution: 'medium',
  onscreen_position: 'top_center',
  convert_to_metric: false,
  remove_sdh_speakers: false,
  skip_end_credits: true,
  cost_saver_mode: false,
};

const sampleProviders: Record<string, ProviderSummary> = {
  google: { provider: 'google', configured: true, key_count: 1, default_key_id: 'env_google', masked_default_key: 'AIza••••••••1234', discovered: true },
  openai: { provider: 'openai', configured: true, key_count: 2, default_key_id: 'key_oai_1', masked_default_key: 'sk-p••••••••aaaa', discovered: true },
  anthropic: { provider: 'anthropic', configured: false, key_count: 0, discovered: false },
  ollama: { provider: 'ollama', configured: true, key_count: 1, default_key_id: 'local_ollama_default', discovered: true },
};

const sampleKeys: ProviderKeyRecord[] = [
  { id: 'env_google', provider: 'google', label: 'Environment (.env)', masked_key: 'AIza••••••••1234', has_key: true, is_default: true, source: 'env', discovered_count: 1 },
  { id: 'key_oai_1', provider: 'openai', label: 'Personal', masked_key: 'sk-p••••••••aaaa', has_key: true, is_default: true, source: 'vault', discovered_count: 2 },
  { id: 'key_oai_2', provider: 'openai', label: 'Vision Only Key', masked_key: 'sk-p••••••••cccc', has_key: true, is_default: false, source: 'vault', discovered_count: 0 },
];

describe('Multi-Provider AI Settings & Key Vault UI', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({
      systemInfo: {
        is_gemini_configured: true,
        is_ai_configured: true,
        masked_api_key: 'AIza••••••••1234',
        gemini_model: 'gemini-3.8-flash',
        gemini_models: ['gemini-3.8-flash'],
        providers: sampleProviders,
        whisper_default_model: 'medium',
        whisper_models: ['tiny', 'base', 'small', 'medium'],
        max_cps_warning: 20,
        hardware: {
          cuda_available: false,
          device_count: 0,
          cpu_count: 8,
          recommended_device: 'cpu',
          recommended_compute_type: 'int8',
        },
      },
      fetchSystemInfo: vi.fn().mockResolvedValue(undefined),
    });
    useSettingsStore.setState({
      settings: initialSettings,
      catalog: sampleCatalog,
      providers: sampleProviders,
      keys: sampleKeys,
      loaded: true,
      refreshing: false,
      error: null,
      pricingChecked: '2026-10',
      pricingSource: 'https://ai.google.dev/gemini-api/docs/pricing',
    });
  });

  it('adds a second OpenAI key via ApiKeySection and supports Ollama Cloud preset', async () => {
    const addSpy = vi.spyOn(api, 'addProviderKey').mockResolvedValue({
      key: { id: 'key_oai_2', provider: 'openai', label: 'Work', masked_key: 'sk-p••••••••bbbb', has_key: true, is_default: false, source: 'vault' },
      keys: [
        ...sampleKeys,
        { id: 'key_oai_2', provider: 'openai', label: 'Work', masked_key: 'sk-p••••••••bbbb', has_key: true, is_default: false, source: 'vault' },
      ],
      providers: {
        ...sampleProviders,
        openai: { provider: 'openai', configured: true, key_count: 2, default_key_id: 'key_oai_1', masked_default_key: 'sk-p••••••••aaaa' },
      },
      catalog: sampleCatalog,
      settings: initialSettings,
      discovery_error: null,
    });

    const view = render(<ApiKeySection />);
    fireEvent.click(view.getByTestId('settings-apikey-toggle'));
    expect((view.getByTestId('settings-apikey-link') as HTMLAnchorElement).href).toContain('aistudio.google.com');

    fireEvent.click(view.getByTestId('provider-tab-openai'));
    expect((view.getByTestId('settings-apikey-link') as HTMLAnchorElement).href).toContain('platform.openai.com/api-keys');

    expect(view.getByTestId('provider-key-item-key_oai_1')).toBeTruthy();

    fireEvent.change(view.getByTestId('provider-key-label-input'), { target: { value: 'Work' } });
    fireEvent.change(view.getByTestId('settings-apikey-input'), { target: { value: 'sk-proj-work-bbbb' } });
    fireEvent.click(view.getByTestId('settings-apikey-save'));

    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    expect(addSpy).toHaveBeenCalledWith(expect.objectContaining({
      provider: 'openai',
      label: 'Work',
      api_key: 'sk-proj-work-bbbb',
    }));

    fireEvent.click(view.getByTestId('provider-tab-anthropic'));
    expect((view.getByTestId('settings-apikey-link') as HTMLAnchorElement).href).toContain('console.anthropic.com/settings/keys');

    // Switch to Ollama tab and verify Cloud preset updates endpoint URL & key issuance link
    fireEvent.click(view.getByTestId('provider-tab-ollama'));
    expect((view.getByTestId('settings-apikey-link') as HTMLAnchorElement).href).toContain('ollama.com/settings/keys');
    fireEvent.click(view.getByTestId('ollama-preset-cloud'));
    expect((view.getByTestId('provider-key-baseurl-input') as HTMLInputElement).value).toBe('https://ollama.com');
  });

  it('strictly filters models by API key availability and pipeline step capability without symbol clutter', async () => {
    vi.spyOn(api, 'getOllamaStatus').mockResolvedValue({
      installed: false, running: false, models: [], selected_model: null, recommended_model: 'qwen2.5:7b',
      download_url: 'https://ollama.com/download', installing: false, step: null, error: null,
    });
    vi.spyOn(api, 'estimateCost').mockResolvedValue({
      duration_seconds: 2700,
      segments: 630,
      stages: {},
      total_cost_usd: 0.06,
    });
    const saveSpy = vi.spyOn(api, 'saveSettings').mockImplementation(async (s) => ({
      settings: s,
      catalog: sampleCatalog,
      providers: sampleProviders,
      keys: sampleKeys,
      pricing_checked: '2026-10',
      pricing_source: 'https://ai.google.dev/gemini-api/docs/pricing',
    }));
    vi.spyOn(api, 'checkModels').mockResolvedValue({ results: {}, steps: {} });

    const view = render(<SettingsScreen />);

    // Duplicated content toggles belong to PipelineConfig, not SettingsScreen; Vision resolution & collapsibles work
    expect(view.queryByTestId('metric-conversion-toggle')).toBeNull();
    expect(view.queryByTestId('remove-sdh-speakers-toggle')).toBeNull();
    fireEvent.click(view.getByTestId('vision-resolution-high'));
    expect(view.getByTestId('vision-resolution-high').getAttribute('aria-checked')).toBe('true');

    expect(view.queryByTestId('ollama-download-link')).toBeNull();
    fireEvent.click(view.getByTestId('ollama-details-toggle'));
    expect(view.getByTestId('ollama-download-link')).toBeTruthy();

    await waitFor(() => expect(view.getByTestId('estimate-table')).toBeTruthy());
    fireEvent.click(view.getByTestId('estimate-collapse-toggle'));
    expect(view.queryByTestId('estimate-table')).toBeNull();
    fireEvent.click(view.getByTestId('estimate-collapse-toggle'));
    expect(view.getByTestId('estimate-table')).toBeTruthy();

    // Switch translation step to OpenAI -> only capable & live models appear (no gpt-5-nano, no unverified seed, no ★/✓/⚠)
    fireEvent.change(view.getByTestId('model-card-translation-provider'), { target: { value: 'openai' } });
    const trModelSelect = view.getByTestId('model-card-translation-model') as HTMLSelectElement;
    const trValues = Array.from(trModelSelect.options).map((o) => o.value);
    const trLabels = Array.from(trModelSelect.options).map((o) => o.textContent || '');
    expect(trValues).toContain('gpt-5-mini');
    expect(trValues).not.toContain('gpt-5-nano');
    expect(trValues).not.toContain('gpt-unverified-seed');
    expect(trLabels.join('|')).not.toMatch(/[★✓⚠]/);

    // Episode & Movie benchmarks displayed for selected OpenAI model
    expect(view.getByTestId('model-card-translation-benchmarks').textContent).toContain('45m');
    expect(view.getByTestId('model-card-translation-benchmarks').textContent).toContain('120m');

    // Switching translation to second OpenAI key (which discovered 0 models) empties options and shows no-capable hint
    fireEvent.change(view.getByTestId('model-card-translation-key'), { target: { value: 'key_oai_2' } });
    expect(view.getByTestId('model-card-translation-no-models')).toBeTruthy();

    // Switch back to key_oai_1 so translation works
    fireEvent.change(view.getByTestId('model-card-translation-key'), { target: { value: 'key_oai_1' } });
    expect((view.getByTestId('model-card-translation-model') as HTMLSelectElement).value).toBe('gpt-5-mini');

    // Ollama in Vision only shows vision-capable qwen2.5vl:7b (qwen2.5:14b is text-only so filtered out!)
    fireEvent.change(view.getByTestId('model-card-vision-provider'), { target: { value: 'ollama' } });
    const viSelect = view.getByTestId('model-card-vision-model') as HTMLSelectElement;
    const viValues = Array.from(viSelect.options).map((o) => o.value);
    expect(viValues).toEqual(['qwen2.5vl:7b']);
    expect(viSelect.value).toBe('qwen2.5vl:7b');

    // Unconfigured Anthropic shows add-key hint and no fake unverified options
    fireEvent.change(view.getByTestId('model-card-glossary-provider'), { target: { value: 'anthropic' } });
    expect(view.getByTestId('model-card-glossary-no-models')).toBeTruthy();

    // Click "Apply provider to all steps" from Translation (OpenAI) -> updates Glossary & Vision too
    fireEvent.click(view.getByTestId('apply-provider-all-btn'));
    expect((view.getByTestId('model-card-glossary-provider') as HTMLSelectElement).value).toBe('openai');
    expect((view.getByTestId('model-card-vision-provider') as HTMLSelectElement).value).toBe('openai');
    expect(Array.from((view.getByTestId('model-card-vision-model') as HTMLSelectElement).options).map((o) => o.value)).toEqual(['gpt-5-mini', 'gpt-5-nano']);

    // Save works cleanly
    const saveBtn = view.container.querySelector('button svg.lucide-save')?.closest('button') as HTMLButtonElement;
    fireEvent.click(saveBtn);
    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
  });

  it('filters RowTranslateDialog models, hides Thinking for non-Google, and unlocks non-Gemini studio actions', () => {
    vi.spyOn(api, 'listJobs').mockResolvedValue([]);
    // Configure OpenAI as active provider WITHOUT any Google Gemini key
    const nonGoogleSettings: AppSettings = {
      ...initialSettings,
      translation: { provider: 'openai', model: 'gpt-5-mini', key_id: 'key_oai_1', thinking: 'default', flex: false },
      glossary: { provider: 'openai', model: 'gpt-5-mini', key_id: 'key_oai_1', thinking: 'default', flex: false },
      vision: { provider: 'ollama', model: 'qwen2.5vl:7b', key_id: 'local_ollama_default', thinking: 'default', flex: false },
    };
    useSettingsStore.setState({ settings: nonGoogleSettings, catalog: sampleCatalog });
    useJobStore.setState({
      systemInfo: {
        is_gemini_configured: false,
        is_ai_configured: true,
        gemini_model: 'gpt-5-mini',
        gemini_models: [],
        providers: {
          ...sampleProviders,
          google: { provider: 'google', configured: false, key_count: 0, discovered: false },
        },
        whisper_default_model: 'medium',
        whisper_models: ['medium'],
        max_cps_warning: 20,
        hardware: { cuda_available: false, device_count: 0, cpu_count: 8, recommended_device: 'cpu', recommended_compute_type: 'int8' },
      },
      activeJob: {
        id: 'job_non_google',
        video_path: 'ep.mkv',
        file_name: 'ep.mkv',
        status: 'COMPLETED',
        progress_percent: 100,
        audio_streams: [],
        selected_audio_stream: 0,
        whisper_model: 'medium',
        vad_filter: true,
        comms_boost: false,
        detect_onscreen_text: false,
        enable_translation: true,
        target_language: 'he',
        gemini_model: 'gpt-5-mini',
        segment_count: 1,
        failed_translation_count: 0,
        segments: [
          {
            id: 1,
            job_id: 'job_non_google',
            index: 1,
            start_time: 0,
            end_time: 2,
            original_text: 'Hello',
            translated_text: null,
            translation_failed: false,
            segment_type: 'DIALOGUE',
          },
        ],
        created_at: '2026-11-18T10:00:00Z',
        updated_at: '2026-11-18T10:00:00Z',
      },
    });

    // 1) RowTranslateDialog filters out unverified & vision-only models, hides Thinking for OpenAI, shows Thinking for Google
    const dlg = render(
      <RowTranslateDialog
        count={1}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />
    );
    const modelSel = dlg.getByTestId('row-model') as HTMLSelectElement;
    const optionValues = Array.from(modelSel.options).map((o) => o.value);
    expect(optionValues).not.toContain('gpt-unverified-seed');
    expect(optionValues).not.toContain('gpt-5-nano');
    expect(optionValues).toContain('gpt-5-mini');
    expect(dlg.queryByTestId('row-thinking')).toBeNull();
    expect(Array.from(modelSel.options).find((o) => o.value === 'gpt-5-mini')?.textContent).toContain('OpenAI');

    // Switching to Google model reveals Thinking level selector
    fireEvent.change(modelSel, { target: { value: 'gemini-3.8-flash' } });
    expect(dlg.getByTestId('row-thinking')).toBeTruthy();

    // Switching to Ollama hides Thinking selector again
    fireEvent.change(modelSel, { target: { value: 'qwen2.5:14b' } });
    expect(dlg.queryByTestId('row-thinking')).toBeNull();

    // 2) ExportToolbar Translate button is enabled even without Google Gemini key
    const tb = render(<ExportToolbar />);
    const translateBtn = Array.from(tb.container.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('Translate')
    ) as HTMLButtonElement | undefined;
    expect(translateBtn).toBeTruthy();
    expect(translateBtn!.disabled).toBe(false);

    // 3) Sidebar summarizes providers across all stages on the Settings button & warns if any stage lacks a key
    const sb = render(<Sidebar />);
    expect(sb.getByTestId('sidebar-ai-status').textContent).toContain('OpenAI · Ollama');
    expect(sb.container.textContent).not.toContain('No key (.env)');

    act(() => {
      useSettingsStore.setState({
        settings: {
          ...nonGoogleSettings,
          vision: { provider: 'anthropic', model: 'claude-sonnet-4-5', key_id: null, thinking: 'default', flex: false },
        },
      });
    });
    expect(sb.getByTestId('sidebar-ai-status').textContent).toContain('No key');
  });

  it('auto-selects best cost/quality models across keys, enables Google Flex when optimal, and respects GPU VRAM for local AI', () => {
    vi.spyOn(api, 'getOllamaStatus').mockResolvedValue({
      installed: true, running: true, models: ['qwen2.5:14b', 'qwen2.5vl:7b', 'gpt-oss:20b'],
      selected_model: 'qwen2.5vl:7b', recommended_model: 'qwen2.5:7b', download_url: 'https://ollama.com/download',
      installing: false, step: null, error: null,
    });
    vi.spyOn(api, 'estimateCost').mockResolvedValue({
      duration_seconds: 2700, segments: 630, stages: {}, total_cost_usd: 0.05,
    });

    const gpuCatalog: CatalogModel[] = [
      ...sampleCatalog,
      {
        id: 'gpt-oss:20b',
        provider: 'ollama',
        label: 'gpt-oss:20b',
        family: 'ollama',
        preview: false,
        available: true,
        key_ids: ['local_ollama_default'],
        is_local: true,
        param_billions: 20.9,
        thinking: ['default'],
        default_level: 'default',
        input_per_m: 0,
        output_per_m: 0,
        episode_cost_usd: 0,
        movie_cost_usd: 0,
        supports_media_resolution: false,
        tasks: ['translation', 'glossary'],
        recommended_for: ['translation', 'glossary'],
        warnings_by_task: {},
        supports_flex: false,
      },
    ];

    useJobStore.setState((prev) => ({
      systemInfo: {
        ...prev.systemInfo!,
        hardware: {
          cuda_available: true,
          device_count: 1,
          gpu_name: 'NVIDIA GeForce RTX 2070 SUPER',
          vram_total_mb: 8192,
          vram_free_mb: 7500,
          cpu_count: 8,
          recommended_device: 'cuda',
          recommended_compute_type: 'float16',
        },
      },
    }));
    useSettingsStore.setState({ catalog: gpuCatalog });

    const view = render(<SettingsScreen />);

    // 1) Cloud-only optimization (Include Local unchecked): picks cheapest recommended cloud models per step
    fireEvent.click(view.getByTestId('auto-recommend-models-btn'));
    expect((view.getByTestId('model-card-translation-provider') as HTMLSelectElement).value).toBe('openai');
    expect((view.getByTestId('model-card-translation-model') as HTMLSelectElement).value).toBe('gpt-5-mini');
    expect((view.getByTestId('model-card-glossary-provider') as HTMLSelectElement).value).toBe('openai');
    expect((view.getByTestId('model-card-glossary-model') as HTMLSelectElement).value).toBe('gpt-5-mini');
    expect((view.getByTestId('model-card-vision-provider') as HTMLSelectElement).value).toBe('openai');
    expect((view.getByTestId('model-card-vision-model') as HTMLSelectElement).value).toBe('gpt-5-nano');
    expect(view.getByTestId('auto-recommend-summary')).toBeTruthy();

    // 2) Include Local checked on 8 GB VRAM GPU -> selects 7B local vision/chat model and skips 14B/20B heavy models
    fireEvent.click(view.getByTestId('auto-recommend-include-local'));
    fireEvent.click(view.getByTestId('auto-recommend-models-btn'));
    expect((view.getByTestId('model-card-translation-provider') as HTMLSelectElement).value).toBe('ollama');
    expect((view.getByTestId('model-card-translation-model') as HTMLSelectElement).value).toBe('qwen2.5vl:7b');
    expect((view.getByTestId('model-card-vision-model') as HTMLSelectElement).value).toBe('qwen2.5vl:7b');
    const summaryText = view.getByTestId('auto-recommend-summary').textContent || '';
    expect(summaryText).toContain('qwen2.5:14b');
    expect(summaryText).toContain('gpt-oss:20b');

    // 3) When Google Flex is cheapest among available cloud models, selects Google and enables Flex mode
    act(() => {
      useSettingsStore.setState({
        catalog: gpuCatalog.filter((m) => m.provider === 'google'),
      });
    });
    fireEvent.click(view.getByTestId('auto-recommend-include-local')); // uncheck local
    fireEvent.click(view.getByTestId('auto-recommend-models-btn'));
    expect((view.getByTestId('model-card-translation-provider') as HTMLSelectElement).value).toBe('google');
    expect((view.getByTestId('model-card-translation-model') as HTMLSelectElement).value).toBe('gemini-3.8-flash');
    expect((view.getByTestId('model-card-translation-flex') as HTMLInputElement).checked).toBe(true);
  });
});
