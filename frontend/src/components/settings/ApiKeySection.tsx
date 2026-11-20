import React, { useEffect, useState } from 'react';
import { useJobStore } from '../../store/useJobStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { useT } from '../../i18n';
import type { MessageKey } from '../../i18n';
import { api } from '../../services/api';
import type { AIProvider } from '../../types';
import {
  Key,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  Eye,
  EyeOff,
  Save,
  Loader2,
  Check,
  Trash2,
  RefreshCw,
  Star,
  Plus,
  Server,
} from 'lucide-react';

const PROVIDERS: AIProvider[] = ['google', 'openai', 'anthropic', 'ollama'];

const PROVIDER_KEY_URLS: Record<AIProvider, string> = {
  google: 'https://aistudio.google.com/app/apikey',
  openai: 'https://platform.openai.com/api-keys',
  anthropic: 'https://console.anthropic.com/settings/keys',
  ollama: 'https://ollama.com/settings/keys',
};

const PROVIDER_PLACEHOLDERS: Record<AIProvider, string> = {
  google: 'AIzaSy...',
  openai: 'sk-proj-...',
  anthropic: 'sk-ant-...',
  ollama: 'Optional for local (required for Ollama Cloud)',
};

export const ApiKeySection: React.FC = () => {
  const t = useT();
  const systemInfo = useJobStore((s) => s.systemInfo);
  const fetchSystemInfo = useJobStore((s) => s.fetchSystemInfo);
  const { keys, providers, catalog, refreshing, loaded, load, refreshProviders, syncVaultState } = useSettingsStore();

  const totalKeys = keys.length;
  const isConfigured = !!systemInfo?.is_gemini_configured || !!systemInfo?.is_ai_configured || totalKeys > 0;
  const liveModelCount = catalog.filter((m) => m.available).length;

  const [expanded, setExpanded] = useState(!isConfigured);
  const [initializedCollapse, setInitializedCollapse] = useState(false);

  useEffect(() => {
    if (!loaded) {
      load();
    }
  }, [loaded, load]);

  useEffect(() => {
    if (systemInfo && !initializedCollapse) {
      setExpanded(!systemInfo.is_gemini_configured && !systemInfo.is_ai_configured);
      setInitializedCollapse(true);
    }
  }, [systemInfo, initializedCollapse]);

  const [activeTab, setActiveTab] = useState<AIProvider>('google');
  const [labelInput, setLabelInput] = useState('');
  const [keyInput, setKeyInput] = useState('');
  const [baseUrlInput, setBaseUrlInput] = useState('http://localhost:11434');
  const [showSecret, setShowSecret] = useState(false);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const tabKeys = keys.filter((k) => k.provider === activeTab);
  const canAdd = activeTab === 'ollama' ? (baseUrlInput.trim() !== 'https://ollama.com' || !!keyInput.trim()) : !!keyInput.trim();

  const handleAddKey = async () => {
    if (!canAdd) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.addProviderKey({
        provider: activeTab,
        api_key: keyInput.trim(),
        label: labelInput.trim() || undefined,
        base_url: activeTab === 'ollama' ? baseUrlInput.trim() || undefined : undefined,
        verify: true,
      });
      syncVaultState(res);
      setKeyInput('');
      setLabelInput('');
      if (res.discovery_error) {
        setNotice(res.discovery_error);
      }
      setJustSaved(true);
      setTimeout(() => setJustSaved(false), 2500);
    } catch (e: any) {
      setError(e.message || 'Failed to save API key');
    } finally {
      setSaving(false);
    }
  };

  const handleSetDefault = async (keyId: string) => {
    setError(null);
    try {
      const res = await api.updateProviderKey(keyId, { is_default: true });
      syncVaultState(res);
    } catch (e: any) {
      setError(e.message || 'Failed to update default key');
    }
  };

  const handleDeleteKey = async (keyId: string) => {
    setError(null);
    try {
      const res = await api.deleteProviderKey(keyId);
      syncVaultState(res);
      await fetchSystemInfo();
    } catch (e: any) {
      setError(e.message || 'Failed to delete key');
    }
  };

  const handleRefresh = async () => {
    setError(null);
    setNotice(null);
    const errs = await refreshProviders();
    const firstErr = Object.values(errs)[0];
    if (firstErr) setNotice(firstErr);
  };

  return (
    <section
      className="bg-surface border border-line rounded-xl shadow-lg overflow-hidden"
      data-testid="settings-apikey-section"
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        data-testid="settings-apikey-toggle"
        className="w-full p-5 flex items-center justify-between gap-3 text-start hover:bg-zinc-800/30 transition-colors cursor-pointer"
      >
        <div className="flex flex-wrap items-center gap-2.5 min-w-0">
          <Key className="w-4 h-4 text-indigo-400 flex-shrink-0" />
          <h2 className="text-sm font-semibold text-zinc-100">{t('settings.apiKey.title')}</h2>
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${
              isConfigured
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
            }`}
          >
            {isConfigured ? (
              <>
                <CheckCircle2 className="w-3 h-3" />
                <span>
                  {totalKeys > 0
                    ? t('settings.apiKey.statusActive', { count: totalKeys })
                    : t('settings.apiKey.statusActiveSingle', { masked: systemInfo?.masked_api_key || '••••••••' })}
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3 h-3" />
                <span>{t('settings.apiKey.statusMissing')}</span>
              </>
            )}
          </span>
          {liveModelCount > 0 && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
              {t('settings.apiKey.discoveredCount', { count: liveModelCount })}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-zinc-400">
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {expanded && (
        <div className="px-5 pb-5 pt-2 border-t border-line space-y-4 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <p className="text-[11px] text-zinc-400 leading-relaxed">{t('settings.apiKey.subtitle')}</p>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleRefresh}
                disabled={refreshing}
                data-testid="providers-refresh-btn"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-line text-zinc-200 font-medium text-[11px] cursor-pointer transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
                <span>{refreshing ? t('settings.apiKey.refreshing') : t('settings.apiKey.refreshModels')}</span>
              </button>
              <a
                href={PROVIDER_KEY_URLS[activeTab]}
                target="_blank"
                rel="noreferrer"
                data-testid="settings-apikey-link"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 font-medium text-[11px] transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                <span>{t(`settings.apiKey.getLinkByProvider.${activeTab}` as MessageKey)}</span>
              </a>
            </div>
          </div>

          {/* Provider Tabs */}
          <div className="flex flex-wrap gap-1.5" role="tablist" data-testid="provider-tabs">
            {PROVIDERS.map((prov) => {
              const count = providers[prov]?.key_count ?? keys.filter((k) => k.provider === prov).length;
              const active = activeTab === prov;
              return (
                <button
                  key={prov}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  data-testid={`provider-tab-${prov}`}
                  onClick={() => {
                    setActiveTab(prov);
                    setError(null);
                    setNotice(null);
                  }}
                  className={`px-3 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 cursor-pointer transition-colors ${
                    active
                      ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
                      : 'bg-sunken border-line text-zinc-400 hover:text-zinc-200 hover:border-zinc-600'
                  }`}
                >
                  <span>{t(`settings.providers.${prov}` as MessageKey)}</span>
                  <span
                    className={`text-[10px] px-1.5 rounded-full font-mono ${
                      active ? 'bg-white/20 text-white' : count > 0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-zinc-800 text-zinc-500'
                    }`}
                  >
                    {count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Saved keys list for selected provider */}
          <div className="space-y-1.5" data-testid="provider-key-list">
            {tabKeys.length === 0 ? (
              <p className="text-[11px] text-zinc-500 italic">
                {t('settings.apiKey.noKeysForProvider', {
                  provider: t(`settings.providers.${activeTab}` as MessageKey),
                })}
              </p>
            ) : (
              tabKeys.map((k) => (
                <div
                  key={k.id}
                  data-testid={`provider-key-item-${k.id}`}
                  className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-lg bg-sunken border border-line"
                >
                  <div className="flex flex-wrap items-center gap-2 min-w-0">
                    <span className="font-semibold text-zinc-200">{k.label}</span>
                    {k.masked_key && (
                      <code dir="ltr" className="text-[11px] font-mono text-zinc-400">
                        {k.masked_key}
                      </code>
                    )}
                    {k.base_url && (
                      <span dir="ltr" className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded">
                        {k.base_url}
                      </span>
                    )}
                    {k.is_default && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                        <Star className="w-2.5 h-2.5 fill-current" />
                        {t('settings.apiKey.defaultBadge')}
                      </span>
                    )}
                    {k.last_error && (
                      <span className="text-[10px] text-amber-300 truncate max-w-[260px]" title={k.last_error}>
                        ({k.last_error})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    {!k.is_default && k.source === 'vault' && (
                      <button
                        type="button"
                        onClick={() => handleSetDefault(k.id)}
                        data-testid={`provider-key-default-${k.id}`}
                        className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-[11px] text-zinc-300 cursor-pointer"
                      >
                        {t('settings.apiKey.setDefaultBtn')}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => handleDeleteKey(k.id)}
                      data-testid={k.id === 'env_google' ? 'settings-apikey-clear' : `provider-key-delete-${k.id}`}
                      title={t('settings.apiKey.clearBtn')}
                      className="p-1.5 rounded bg-zinc-800 hover:bg-rose-950/60 hover:text-rose-300 text-zinc-400 cursor-pointer transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Add Key Form */}
          <div className="p-3 rounded-xl bg-sunken/70 border border-line space-y-2.5">
            {activeTab === 'ollama' && (
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="text-[11px] font-medium text-zinc-300 flex items-center gap-1.5">
                    <Server className="w-3.5 h-3.5 text-indigo-400" />
                    {t('settings.apiKey.baseUrlInput')}
                  </label>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setBaseUrlInput('http://localhost:11434');
                        if (!labelInput) setLabelInput('Local Ollama');
                      }}
                      data-testid="ollama-preset-local"
                      className="px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-[10px] text-zinc-200 cursor-pointer"
                    >
                      {t('settings.apiKey.ollamaLocalPreset')}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setBaseUrlInput('https://ollama.com');
                        if (!labelInput) setLabelInput('Ollama Cloud');
                      }}
                      data-testid="ollama-preset-cloud"
                      className="px-2 py-0.5 rounded bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/30 text-[10px] text-indigo-300 cursor-pointer"
                    >
                      {t('settings.apiKey.ollamaCloudPreset')}
                    </button>
                  </div>
                </div>
                <input
                  type="text"
                  dir="ltr"
                  value={baseUrlInput}
                  onChange={(e) => setBaseUrlInput(e.target.value)}
                  data-testid="provider-key-baseurl-input"
                  className="w-full bg-surface border border-line rounded-lg px-3 py-1.5 text-xs text-zinc-100 font-mono focus:outline-none focus:border-indigo-500"
                />
                <p className="text-[10px] text-zinc-500">{t('settings.apiKey.localZeroCostNote')}</p>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 items-end">
              <label className="sm:col-span-3 space-y-1">
                <span className="block text-[11px] font-medium text-zinc-300">{t('settings.apiKey.labelInput')}</span>
                <input
                  type="text"
                  value={labelInput}
                  onChange={(e) => setLabelInput(e.target.value)}
                  placeholder={t('settings.apiKey.labelPlaceholder')}
                  data-testid="provider-key-label-input"
                  className="w-full bg-surface border border-line rounded-lg px-2.5 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-indigo-500"
                />
              </label>

              <label className="sm:col-span-6 space-y-1">
                <span className="block text-[11px] font-medium text-zinc-300">{t('settings.apiKey.inputLabel')}</span>
                <div className="relative">
                  <input
                    type={showSecret ? 'text' : 'password'}
                    dir="ltr"
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && canAdd) {
                        handleAddKey();
                      }
                    }}
                    placeholder={PROVIDER_PLACEHOLDERS[activeTab]}
                    data-testid="settings-apikey-input"
                    className="w-full bg-surface border border-line rounded-lg ps-3 pe-9 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSecret((s) => !s)}
                    title={showSecret ? t('settings.apiKey.hideKey') : t('settings.apiKey.showKey')}
                    className="absolute inset-y-0 end-2 flex items-center text-zinc-400 hover:text-zinc-200 cursor-pointer"
                  >
                    {showSecret ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </label>

              <div className="sm:col-span-3">
                <button
                  type="button"
                  onClick={handleAddKey}
                  disabled={saving || !canAdd}
                  data-testid="settings-apikey-save"
                  className="w-full px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white flex items-center justify-center gap-1.5 cursor-pointer transition-colors whitespace-nowrap"
                >
                  {saving ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : justSaved ? (
                    <Check className="w-3.5 h-3.5" />
                  ) : (
                    <Plus className="w-3.5 h-3.5" />
                  )}
                  <span>{saving ? t('settings.apiKey.saving') : justSaved ? t('settings.apiKey.saved') : t('settings.apiKey.addBtn')}</span>
                </button>
              </div>
            </div>

            {error && <p className="text-[11px] text-rose-400">{error}</p>}
            {notice && <p className="text-[11px] text-amber-300">{notice}</p>}
          </div>
        </div>
      )}
    </section>
  );
};

const OLLAMA_LOCAL_URL = 'http://localhost:11434';
const OLLAMA_CLOUD_URL = 'https://ollama.com';
const OLLAMA_DOWNLOAD_URL = 'https://ollama.com/download';

/**
 * First start without any AI provider: connect one (Gemini, OpenAI, Claude, or Ollama local / cloud)
 * or skip and use local Whisper transcription only. More providers and keys are added in Settings.
 */
export const FirstRunApiKeyModal: React.FC = () => {
  const t = useT();
  const systemInfo = useJobStore((s) => s.systemInfo);
  const fetchSystemInfo = useJobStore((s) => s.fetchSystemInfo);
  const loadSettings = useSettingsStore((s) => s.load);

  const [dismissed, setDismissed] = useState(false);
  const [provider, setProvider] = useState<AIProvider>('google');
  const [ollamaUrl, setOllamaUrl] = useState(OLLAMA_LOCAL_URL);
  const [keyInput, setKeyInput] = useState('');
  const [showSecret, setShowSecret] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Saved, but the provider check reported a problem (e.g. local Ollama is not running): keep the
  // dialog open to show it until the user continues.
  const [notice, setNotice] = useState<string | null>(null);

  if (!systemInfo || dismissed) return null;
  if ((systemInfo.is_gemini_configured || systemInfo.is_ai_configured) && !notice) return null;

  const isOllama = provider === 'ollama';
  const ollamaLocal = isOllama && ollamaUrl.trim() !== OLLAMA_CLOUD_URL;
  const canSave = ollamaLocal ? !!ollamaUrl.trim() : !!keyInput.trim();
  const keyLink = ollamaLocal ? OLLAMA_DOWNLOAD_URL : PROVIDER_KEY_URLS[provider];
  const keyLinkText = ollamaLocal
    ? t('settings.apiKey.modalOllamaDownload')
    : t(`settings.apiKey.getLinkByProvider.${provider}` as MessageKey);

  const chooseProvider = (prov: AIProvider) => {
    setProvider(prov);
    setKeyInput('');
    setError(null);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.addProviderKey({
        provider,
        api_key: keyInput.trim(),
        label: isOllama ? (ollamaLocal ? 'Local Ollama' : 'Ollama Cloud') : undefined,
        base_url: isOllama ? ollamaUrl.trim() : undefined,
        verify: true,
      });
      await fetchSystemInfo();
      await loadSettings();
      if (res.discovery_error) {
        setNotice(res.discovery_error);
      } else {
        setDismissed(true);
      }
    } catch (err: any) {
      setError(err.message || 'Could not save API key');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4"
      data-testid="first-run-apikey-modal"
    >
      <div className="bg-surface border border-line rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 max-h-[calc(100vh-2rem)] overflow-y-auto">
        <div className="flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center flex-shrink-0">
            <Key className="w-5 h-5 text-indigo-400" />
          </div>
          <div className="space-y-1">
            <h2 className="text-base font-bold text-zinc-50">{t('settings.apiKey.modalTitle')}</h2>
            <p className="text-xs text-zinc-400 leading-relaxed">{t('settings.apiKey.modalSubtitle')}</p>
          </div>
        </div>

        {notice ? (
          <div className="space-y-4" data-testid="modal-apikey-notice">
            <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-1.5 text-xs">
              <div className="flex items-center gap-1.5 font-semibold text-amber-200">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>{t('settings.apiKey.modalSavedWithNotice')}</span>
              </div>
              <p className="text-[11px] text-amber-100/80 leading-relaxed break-words">{notice}</p>
            </div>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setDismissed(true)}
                data-testid="modal-apikey-continue"
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white cursor-pointer transition-colors shadow-lg shadow-indigo-600/25"
              >
                {t('settings.apiKey.modalContinue')}
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSave} className="space-y-4">
            {/* Which provider */}
            <div className="space-y-1.5">
              <div className="text-xs font-medium text-zinc-300">{t('settings.providerLabel')}</div>
              <div className="grid grid-cols-2 gap-1.5" role="tablist" aria-label={t('settings.providerLabel')}>
                {PROVIDERS.map((prov) => {
                  const active = provider === prov;
                  return (
                    <button
                      key={prov}
                      type="button"
                      role="tab"
                      aria-selected={active}
                      onClick={() => chooseProvider(prov)}
                      data-testid={`modal-provider-tab-${prov}`}
                      className={`px-3 py-2 rounded-xl border text-xs font-medium text-start cursor-pointer transition-colors ${
                        active
                          ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
                          : 'bg-sunken border-line text-zinc-300 hover:text-zinc-100 hover:border-zinc-600'
                      }`}
                    >
                      {t(`settings.providers.${prov}` as MessageKey)}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Where to get a key / Ollama */}
            <div className="p-3.5 rounded-xl bg-sunken border border-line space-y-2.5 text-xs">
              {isOllama ? (
                <>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => setOllamaUrl(OLLAMA_LOCAL_URL)}
                      aria-pressed={ollamaLocal}
                      data-testid="modal-ollama-local"
                      className={`px-2.5 py-1 rounded-lg border text-[11px] font-medium cursor-pointer transition-colors ${
                        ollamaLocal ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-200' : 'bg-zinc-800 border-line text-zinc-300 hover:bg-zinc-700'
                      }`}
                    >
                      {t('settings.apiKey.ollamaLocalPreset')}
                    </button>
                    <button
                      type="button"
                      onClick={() => setOllamaUrl(OLLAMA_CLOUD_URL)}
                      aria-pressed={!ollamaLocal}
                      data-testid="modal-ollama-cloud"
                      className={`px-2.5 py-1 rounded-lg border text-[11px] font-medium cursor-pointer transition-colors ${
                        !ollamaLocal ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-200' : 'bg-zinc-800 border-line text-zinc-300 hover:bg-zinc-700'
                      }`}
                    >
                      {t('settings.apiKey.ollamaCloudPreset')}
                    </button>
                  </div>
                  <p className="text-[11px] text-zinc-400 leading-relaxed">
                    {ollamaLocal ? t('settings.apiKey.modalOllamaLocalText') : t('settings.apiKey.modalOllamaCloudText')}
                  </p>
                </>
              ) : (
                <>
                  <div className="font-semibold text-zinc-200">{t('settings.apiKey.modalWhereTitle')}</div>
                  <p className="text-[11px] text-zinc-400 leading-relaxed">{t('settings.apiKey.modalWhereText')}</p>
                </>
              )}
              <a
                href={keyLink}
                target="_blank"
                rel="noreferrer"
                data-testid="modal-apikey-link"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 font-semibold text-xs transition-colors"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                <span>{keyLinkText}</span>
              </a>
            </div>

            {isOllama && (
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-medium text-zinc-300" htmlFor="modal-ollama-url">
                  <Server className="w-3.5 h-3.5 text-indigo-400" />
                  {t('settings.apiKey.baseUrlInput')}
                </label>
                <input
                  id="modal-ollama-url"
                  type="text"
                  dir="ltr"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  data-testid="modal-apikey-baseurl"
                  className="w-full bg-sunken border border-line rounded-xl px-3.5 py-2 text-xs text-zinc-100 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>
            )}

            {!ollamaLocal && (
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-zinc-300" htmlFor="modal-apikey-input">
                  {t('settings.apiKey.inputLabel')}
                </label>
                <div className="relative">
                  <input
                    id="modal-apikey-input"
                    type={showSecret ? 'text' : 'password'}
                    dir="ltr"
                    autoFocus
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    placeholder={PROVIDER_PLACEHOLDERS[provider]}
                    data-testid="modal-apikey-input"
                    className="w-full bg-sunken border border-line rounded-xl ps-3.5 pe-10 py-2.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSecret((v) => !v)}
                    title={showSecret ? t('settings.apiKey.hideKey') : t('settings.apiKey.showKey')}
                    className="absolute inset-y-0 end-3 flex items-center text-zinc-400 hover:text-zinc-200 cursor-pointer"
                  >
                    {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            )}

            <p className="text-[11px] text-zinc-500">{t('settings.apiKey.modalMoreLater')}</p>
            {error && <p className="text-[11px] text-rose-400">{error}</p>}

            <div className="flex flex-wrap items-center justify-end gap-2.5 pt-1">
              <button
                type="button"
                onClick={() => setDismissed(true)}
                data-testid="modal-apikey-skip"
                className="px-3.5 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-300 cursor-pointer transition-colors"
              >
                {t('settings.apiKey.modalSkip')}
              </button>
              <button
                type="submit"
                disabled={saving || !canSave}
                data-testid="modal-apikey-save"
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white flex items-center gap-1.5 cursor-pointer transition-colors shadow-lg shadow-indigo-600/25"
              >
                {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                <span>
                  {saving
                    ? t('settings.apiKey.saving')
                    : ollamaLocal
                      ? t('settings.apiKey.modalConnectLocal')
                      : t('settings.apiKey.modalSaveAndContinue')}
                </span>
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
