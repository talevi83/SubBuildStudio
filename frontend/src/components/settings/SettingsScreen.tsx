import React, { useEffect, useMemo, useState } from 'react';
import { useSettingsStore } from '../../store/useSettingsStore';
import { useUiStore } from '../../store/useUiStore';
import type { Language, Theme } from '../../store/useUiStore';
import { useJobStore } from '../../store/useJobStore';
import { useT, formatUsd, formatNumber, formatRate } from '../../i18n';
import type { MessageKey } from '../../i18n';
import { useCostEstimate } from '../../hooks/useCostEstimate';
import { thinkingLabel, modelLabel, providerDisplayName } from './labels';
import type {
  AIProvider, AppSettings, CatalogModel, GeminiCallSettings, HardwareInfo, MediaResolution, ModelCheckResponse, ModelCheckResult, OllamaStatus, PipelineTask, ProviderKeyRecord, ProviderSummary, ThinkingLevel, UpdateApplyResponse, UpdateStatus,
} from '../../types';
import { api } from '../../services/api';
import { StorageSection } from './StorageSection';
import { ApiKeySection } from './ApiKeySection';
import {
  ArrowLeft, AlertTriangle, Brain, Bug, ChevronDown, ChevronUp, Coins, Lightbulb, Moon, RefreshCw, Sun, Save, Loader2, Check, ExternalLink, PlugZap, CheckCircle2, XCircle,
  Hourglass, ScrollText, Stethoscope, Download, Cpu, Terminal, Sparkles,
} from 'lucide-react';


const card = 'bg-surface border border-line rounded-xl p-4 sm:p-5 shadow-lg space-y-4';
const selectClass =
  'w-full bg-sunken border border-line rounded-lg px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500';

type Step = PipelineTask;
const STEPS: Step[] = ['translation', 'glossary', 'vision'];
/** Why the updater did not update (UpdateApplyResponse.skipped_reason), in the UI's language */
const UPDATE_SKIP_MESSAGES: Record<string, MessageKey> = {
  dirty_worktree: 'settings.updater.skippedDirty',
  tag_not_on_main: 'settings.updater.skippedNotOnMain',
  unsigned_tag: 'settings.updater.skippedUnsigned',
  dependency_install_failed: 'settings.updater.rolledBackDeps',
  frontend_build_failed: 'settings.updater.rolledBackFrontend',
  installer_managed: 'settings.updater.installerManaged',
};
const AI_PROVIDERS: AIProvider[] = ['google', 'openai', 'anthropic', 'ollama'];

function filterModelsForStepAndKey(
  catalog: CatalogModel[],
  providers: Partial<Record<AIProvider, ProviderSummary>>,
  keys: ProviderKeyRecord[],
  provider: AIProvider,
  step: Step,
  keyId?: string | null
): CatalogModel[] {
  const stepCapable = catalog.filter(
    (m) => (m.provider || 'google') === provider && (!m.tasks || m.tasks.includes(step))
  );
  const providerSummary = providers[provider];
  const providerKeys = keys.filter((k) => k.provider === provider);
  const liveForProvider = catalog.some((m) => (m.provider || 'google') === provider && m.available);
  const providerConfigured = !!providerSummary?.configured || providerKeys.length > 0;

  if (!liveForProvider && !providerConfigured) {
    return [];
  }

  if (liveForProvider || providerSummary?.discovered) {
    const effectiveKeyId = keyId || providerSummary?.default_key_id || '';
    const keyObj = effectiveKeyId ? providerKeys.find((k) => k.id === effectiveKeyId) : undefined;
    const keyChecked = keyObj ? keyObj.discovered_count !== null && keyObj.discovered_count !== undefined : false;
    const keyHasModels =
      !!effectiveKeyId &&
      catalog.some((m) => (m.provider || 'google') === provider && m.available && m.key_ids?.includes(effectiveKeyId));

    return stepCapable.filter((m) => {
      if (!m.available) return false;
      if (effectiveKeyId && (keyChecked || keyHasModels)) {
        return !!m.key_ids?.includes(effectiveKeyId);
      }
      return true;
    });
  }

  // Offline mode fallback when live discovery did not run
  return stepCapable;
}

function pickBestModelForProvider(
  catalog: CatalogModel[],
  providers: Partial<Record<AIProvider, ProviderSummary>>,
  keys: ProviderKeyRecord[],
  provider: AIProvider,
  step: Step,
  keyId?: string | null
): CatalogModel | undefined {
  const pool = filterModelsForStepAndKey(catalog, providers, keys, provider, step, keyId);
  return pool.find((m) => m.recommended_for?.includes(step)) || pool[0];
}

function parseModelBillions(m: CatalogModel): number | null {
  if (typeof m.param_billions === 'number' && Number.isFinite(m.param_billions)) {
    return m.param_billions;
  }
  const match = /(?:^|[:\-_])(\d+(?:\.\d+)?)b(?:$|[:\-_])/i.exec(m.id || '');
  if (!match) return null;
  const val = Number.parseFloat(match[1]);
  return Number.isFinite(val) ? val : null;
}

function maxLocalBillionsForHardware(hw?: HardwareInfo | null): number {
  if (!hw?.cuda_available) return 0;
  const vramMb = hw.vram_total_mb ?? 8192;
  if (vramMb < 5500) return 0;
  if (vramMb < 7800) return 8.5;
  if (vramMb < 11500) return 12.5;
  if (vramMb < 15500) return 16.0;
  if (vramMb < 23000) return 22.0;
  return 35.0;
}

function isLocalModelEntry(m: CatalogModel): boolean {
  return (
    !!m.is_local ||
    (m.provider === 'ollama' && m.input_per_m === 0 && !m.id.endsWith('-cloud') && !m.id.includes(':cloud'))
  );
}

function effectiveEpisodeStepCost(m: CatalogModel, step: Step): number {
  if (isLocalModelEntry(m)) return 0;
  const prov = m.provider || 'google';
  const factor = prov === 'google' && m.supports_flex && m.flex_price_factor ? m.flex_price_factor : 1;
  if (step === 'vision') {
    const base = m.episode_vision_cost_usd ?? (m.input_per_m ?? 999) * 0.04 + (m.output_per_m ?? 999) * 0.005;
    return base * factor;
  }
  if (step === 'glossary') {
    const inP = m.input_per_m ?? 999;
    const outP = m.output_per_m ?? 999;
    return ((9200 * inP + 2000 * outP) / 1e6) * factor;
  }
  const base = m.episode_cost_usd ?? (m.input_per_m ?? 999) * 0.04 + (m.output_per_m ?? 999) * 0.025;
  return base * factor;
}

function pickThinkingForAuto(m: CatalogModel, step: Step): ThinkingLevel {
  const prov = m.provider || 'google';
  if (prov !== 'google') return 'default';
  const preferred: ThinkingLevel[] =
    step === 'vision' ? ['minimal', 'low', 'default'] : ['low', 'medium', 'default'];
  return preferred.find((lv) => m.thinking.includes(lv)) || 'default';
}

function pickBestValueForStep(
  catalog: CatalogModel[],
  providers: Partial<Record<AIProvider, ProviderSummary>>,
  keys: ProviderKeyRecord[],
  step: Step,
  includeLocal: boolean,
  hw?: HardwareInfo | null,
): { call: GeminiCallSettings; chosen: CatalogModel; skippedHeavyLocal: CatalogModel[] } | null {
  const maxBillions = maxLocalBillionsForHardware(hw);
  const poolMap = new Map<string, CatalogModel>();
  for (const prov of AI_PROVIDERS) {
    const provKeys = keys.filter((k) => k.provider === prov);
    const keyIdsToCheck: (string | null)[] = provKeys.length > 0 ? provKeys.map((k) => k.id) : [null];
    for (const kid of keyIdsToCheck) {
      for (const m of filterModelsForStepAndKey(catalog, providers, keys, prov, step, kid)) {
        const key = `${m.provider || prov}:${m.id}`;
        if (!poolMap.has(key)) poolMap.set(key, m);
      }
    }
  }
  const allCapable = Array.from(poolMap.values());
  if (allCapable.length === 0) return null;

  const skippedHeavyLocal: CatalogModel[] = [];
  const eligible = allCapable.filter((m) => {
    if (!isLocalModelEntry(m)) return true;
    if (!includeLocal) return false;
    if (maxBillions <= 0) {
      skippedHeavyLocal.push(m);
      return false;
    }
    const isRecLocal =
      m.suitability_by_task?.[step] === 'recommended' || !!m.recommended_for?.includes(step);
    if (!isRecLocal) return false;
    const paramsB = parseModelBillions(m) ?? 8;
    if (paramsB < 6.5) return false;
    if (paramsB > maxBillions) {
      skippedHeavyLocal.push(m);
      return false;
    }
    return true;
  });

  if (eligible.length === 0) return null;

  const recommended = eligible.filter(
    (m) => m.suitability_by_task?.[step] === 'recommended' || !!m.recommended_for?.includes(step),
  );
  const suitable = eligible.filter(
    (m) => m.suitability_by_task?.[step] !== 'warning' && m.suitability_by_task?.[step] !== 'unsuitable',
  );
  const candidates = recommended.length > 0 ? recommended : suitable.length > 0 ? suitable : eligible;

  const sorted = [...candidates].sort((a, b) => {
    const costDiff = effectiveEpisodeStepCost(a, step) - effectiveEpisodeStepCost(b, step);
    if (Math.abs(costDiff) > 1e-6) return costDiff;
    if (isLocalModelEntry(a) && isLocalModelEntry(b)) {
      const pA = parseModelBillions(a) ?? 8;
      const pB = parseModelBillions(b) ?? 8;
      if (pA !== pB) return pB - pA;
    }
    return (a.rank ?? 50) - (b.rank ?? 50);
  });

  const best = sorted[0];
  const prov: AIProvider = best.provider || 'google';
  const defaultKeyId = providers[prov]?.default_key_id;
  const keyId =
    !best.key_ids?.length || (defaultKeyId && best.key_ids.includes(defaultKeyId))
      ? null
      : best.key_ids[0];
  return {
    call: {
      provider: prov,
      model: best.id,
      key_id: keyId,
      thinking: pickThinkingForAuto(best, step),
      flex: prov === 'google' && !!best.supports_flex,
    },
    chosen: best,
    skippedHeavyLocal,
  };
}

/** Per-step check results; older servers only report per model. */
function checksByStep(res: ModelCheckResponse | undefined, target: AppSettings): Partial<Record<Step, ModelCheckResult>> {
  const out: Partial<Record<Step, ModelCheckResult>> = {};
  for (const step of STEPS) {
    const r = res?.steps?.[step] ?? res?.results?.[target[step].model];
    if (r) out[step] = { ok: r.ok, error: r.error };
  }
  return out;
}

const Segmented = <V extends string>({
  value,
  options,
  onChange,
  itemTestIdPrefix,
}: {
  value: V;
  options: { value: V; label: React.ReactNode }[];
  onChange: (v: V) => void;
  itemTestIdPrefix?: string;
}) => (
  <div className="inline-flex bg-sunken p-1 rounded-lg border border-line" role="radiogroup">
    {options.map((o) => (
      <button
        key={o.value}
        type="button"
        role="radio"
        aria-checked={value === o.value}
        data-testid={itemTestIdPrefix ? `${itemTestIdPrefix}-${o.value}` : undefined}
        onClick={() => onChange(o.value)}
        className={`px-2.5 py-1 rounded text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
          value === o.value ? 'bg-indigo-600 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
        }`}
      >
        {o.label}
      </button>
    ))}
  </div>
);

const PriceLine: React.FC<{ model?: CatalogModel; flex?: boolean; step: Step }> = ({ model, flex, step }) => {
  const t = useT();
  const language = useUiStore((s) => s.language);
  if (!model) return <p className="text-[11px] text-amber-300">{t('settings.priceUnknown')}</p>;
  // Flex bills every token at a fixed share of the Standard price
  const factor = flex && model.flex_price_factor ? model.flex_price_factor : 1;
  const usd = (v: number) => formatRate(v * factor, language);
  const epRaw = step === 'vision' ? model.episode_vision_cost_usd : model.episode_cost_usd;
  const mvRaw = step === 'vision' ? model.movie_vision_cost_usd : model.movie_cost_usd;
  const epCost = epRaw != null ? epRaw * factor : null;
  const mvCost = mvRaw != null ? mvRaw * factor : null;

  return (
    <div className="space-y-1 text-[11px]" data-testid={`model-card-${step}-pricing`}>
      {model.is_local ? (
        <p className="text-emerald-300 font-medium">{t('settings.localFreeTag')}</p>
      ) : model.input_per_m != null && model.output_per_m != null ? (
        <p className="text-zinc-300">
          {factor !== 1 && <span className="text-emerald-400 font-medium">{t('settings.flexTag')}: </span>}
          {t('settings.price', { input: usd(model.input_per_m), output: usd(model.output_per_m) })}
        </p>
      ) : (
        <p className="text-amber-300">{t('settings.priceUnknown')}</p>
      )}
      {epCost != null && mvCost != null && (
        <p className="text-indigo-300/90 font-medium" data-testid={`model-card-${step}-benchmarks`}>
          {t('settings.benchmarkEstimate', {
            episode: formatUsd(epCost, language),
            movie: formatUsd(mvCost, language),
          })}
        </p>
      )}
      {model.input_long_per_m && model.output_long_per_m ? (
        <p className="text-zinc-500">
          {t('settings.priceLong', { input: usd(model.input_long_per_m), output: usd(model.output_long_per_m) })}
        </p>
      ) : null}
      {model.price_change && (
        <p className="text-amber-300">
          {t('settings.priceChange', {
            input: usd(model.price_change.input_per_m),
            output: usd(model.price_change.output_per_m),
            date: new Date(model.price_change.after + 'T12:00:00').toLocaleDateString(language === 'he' ? 'he-IL' : 'en-US', {
              day: 'numeric',
              month: 'short',
              year: 'numeric',
            }),
          })}
        </p>
      )}
      {model.note && <p className="text-zinc-500">{model.note}</p>}
    </div>
  );
};

const FlexToggle: React.FC<{
  step: Step;
  value: GeminiCallSettings;
  info?: CatalogModel;
  onChange: (v: GeminiCallSettings) => void;
}> = ({ step, value, info, onChange }) => {
  const t = useT();
  const supported = info?.supports_flex !== false;
  const on = !!value.flex && supported;
  return (
    <div className="space-y-1">
      <label
        className={`inline-flex items-center gap-2 text-[11px] font-medium ${
          supported ? 'text-zinc-300 cursor-pointer' : 'text-zinc-600 cursor-not-allowed'
        }`}
      >
        <input
          type="checkbox"
          role="switch"
          data-testid={`model-card-${step}-flex`}
          checked={on}
          disabled={!supported}
          onChange={(e) => onChange({ ...value, flex: e.target.checked })}
          className="accent-emerald-500"
        />
        <Hourglass className="w-3 h-3" />
        {t('settings.flex')}
        <span className="text-[10px] px-1.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
          {t('settings.flexSaving')}
        </span>
      </label>
      {!supported ? (
        <p className="text-[11px] text-zinc-500">{t('settings.flexUnsupported')}</p>
      ) : on ? (
        <p className="text-[11px] text-amber-300">
          {t(step === 'translation' ? 'settings.flexHintTranslation' : 'settings.flexHint')}
        </p>
      ) : null}
    </div>
  );
};

const ModelCard: React.FC<{
  step: Step;
  title: string;
  hint: string;
  value: GeminiCallSettings;
  catalog: CatalogModel[];
  providers: Partial<Record<AIProvider, ProviderSummary>>;
  keys: ProviderKeyRecord[];
  onChange: (v: GeminiCallSettings) => void;
  onApplyProviderAll?: (provider: AIProvider) => void;
  children?: React.ReactNode;
  check?: ModelCheckResult;
}> = ({ step, title, hint, value, catalog, providers, keys, onChange, onApplyProviderAll, children, check }) => {
  const t = useT();
  const testId = `model-card-${step}`;
  const activeProvider: AIProvider = value.provider || 'google';
  const providerKeys = keys.filter((k) => k.provider === activeProvider);
  // Strict filtering: only models available to the active key AND capable of this pipeline step
  const options = filterModelsForStepAndKey(catalog, providers, keys, activeProvider, step, value.key_id);
  const info =
    options.find((m) => m.id === value.model) ||
    (value.model
      ? catalog.find((m) => (m.provider || 'google') === activeProvider && m.id === value.model)
      : undefined);
  const levels: ThinkingLevel[] = info ? info.thinking : ['default'];
  const warningText = info?.warnings_by_task?.[step];
  const isRecommended = !!info?.recommended_for?.includes(step);
  const hasAnyProviderKey =
    providerKeys.length > 0 ||
    !!providers[activeProvider]?.configured ||
    catalog.some((m) => (m.provider || 'google') === activeProvider && m.available);

  const handleKeyChange = (nextKeyIdRaw: string) => {
    const nextKeyId = nextKeyIdRaw || null;
    const nextOptions = filterModelsForStepAndKey(catalog, providers, keys, activeProvider, step, nextKeyId);
    const keepCurrent = !!value.model && nextOptions.some((m) => m.id === value.model);
    const picked = nextOptions.find((m) => m.recommended_for?.includes(step)) || nextOptions[0];
    onChange({
      ...value,
      provider: activeProvider,
      key_id: nextKeyId,
      model: keepCurrent ? value.model : (picked?.id || ''),
    });
  };

  return (
    <div className="p-3.5 rounded-xl bg-sunken border border-line flex flex-col justify-between gap-3" data-testid={testId}>
      <div className="space-y-2.5">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center justify-between gap-1.5">
            <h3 className="text-xs font-semibold text-zinc-100 flex flex-wrap items-center gap-1.5">
              <span>{title}</span>
              {isRecommended && (
                <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  <Sparkles className="w-2.5 h-2.5" />
                  {t('settings.recommendedTag')}
                </span>
              )}
              {info?.available && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                  {t('settings.availableTag')}
                </span>
              )}
              {info?.preview && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
                  {t('settings.preview')}
                </span>
              )}
            </h3>
            {onApplyProviderAll && (
              <button
                type="button"
                onClick={() => onApplyProviderAll(activeProvider)}
                data-testid="apply-provider-all-btn"
                className="px-2 py-0.5 rounded-md bg-zinc-800 hover:bg-zinc-700 border border-line text-[10px] text-zinc-200 cursor-pointer transition-colors"
              >
                {t('settings.applyProviderAll')}
              </button>
            )}
          </div>
          <p className="text-[11px] text-zinc-500 leading-snug">{hint}</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <label className="text-[11px] text-zinc-400 space-y-1">
            <span className="block font-medium">{t('settings.providerLabel')}</span>
            <select
              value={activeProvider}
              aria-label={`${title} — ${t('settings.providerLabel')}`}
              data-testid={`${testId}-provider`}
              onChange={(e) => {
                const nextProv = e.target.value as AIProvider;
                const best = pickBestModelForProvider(catalog, providers, keys, nextProv, step);
                const nextModel = best?.id || '';
                const nextThinking =
                  nextProv === 'google' && best?.thinking.includes(value.thinking)
                    ? value.thinking
                    : nextProv === 'google' && best?.default_level && best.thinking.includes(best.default_level as ThinkingLevel)
                      ? (best.default_level as ThinkingLevel)
                      : 'default';
                onChange({
                  provider: nextProv,
                  model: nextModel,
                  key_id: null,
                  thinking: nextThinking,
                  flex: nextProv === 'google' && !!value.flex && best?.supports_flex !== false,
                });
              }}
              className={selectClass}
            >
              {AI_PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {t(`settings.providers.${p}` as MessageKey)}
                </option>
              ))}
            </select>
          </label>

          {activeProvider === 'google' ? (
            <label className="text-[11px] text-zinc-400 space-y-1">
              <span className="block font-medium flex items-center gap-1">
                <Brain className="w-3 h-3" />
                {t('settings.thinking')}
              </span>
              <select
                value={value.thinking}
                aria-label={`${title} — ${t('settings.thinking')}`}
                onChange={(e) => onChange({ ...value, thinking: e.target.value as ThinkingLevel })}
                className={selectClass}
                disabled={levels.length === 1}
              >
                {levels.map((l) => (
                  <option key={l} value={l}>
                    {thinkingLabel(t, l, info)}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <label className="text-[11px] text-zinc-400 space-y-1">
              <span className="block font-medium">{t('settings.keyLabel')}</span>
              <select
                value={value.key_id || ''}
                aria-label={`${title} — ${t('settings.keyLabel')}`}
                data-testid={`${testId}-key`}
                onChange={(e) => handleKeyChange(e.target.value)}
                className={selectClass}
                disabled={providerKeys.length === 0}
              >
                <option value="">{t('settings.keyDefaultAuto')}</option>
                {providerKeys.map((k) => (
                  <option key={k.id} value={k.id}>
                    {k.label} {k.masked_key ? `(${k.masked_key})` : ''}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="text-[11px] text-zinc-400 space-y-1 sm:col-span-2">
            <span className="block font-medium">{t('settings.model')}</span>
            <select
              value={options.some((m) => m.id === value.model) ? value.model : (options.length === 0 ? value.model : '')}
              aria-label={`${title} — ${t('settings.model')}`}
              data-testid={`${testId}-model`}
              disabled={options.length === 0 && !value.model}
              onChange={(e) => {
                const next = options.find((m) => m.id === e.target.value);
                const thinking = next?.thinking.includes(value.thinking) ? value.thinking : 'default';
                onChange({
                  ...value,
                  provider: activeProvider,
                  model: e.target.value,
                  thinking,
                  flex: activeProvider === 'google' && !!value.flex && next?.supports_flex !== false,
                });
              }}
              className={selectClass}
            >
              {options.map((m) => (
                <option key={`${activeProvider}:${m.id}`} value={m.id}>
                  {m.label}
                </option>
              ))}
              {options.length > 0 && !options.some((m) => m.id === value.model) && (
                <option value="">—</option>
              )}
              {options.length === 0 && value.model && (
                <option value={value.model}>{value.model}</option>
              )}
              {options.length === 0 && !value.model && (
                <option value="">—</option>
              )}
            </select>
          </label>
        </div>

        {activeProvider === 'google' && providerKeys.length > 1 && (
          <label className="block text-[11px] text-zinc-400 space-y-1">
            <span className="block font-medium">{t('settings.keyLabel')}</span>
            <select
              value={value.key_id || ''}
              aria-label={`${title} — ${t('settings.keyLabel')}`}
              data-testid={`${testId}-key`}
              onChange={(e) => handleKeyChange(e.target.value)}
              className={selectClass}
            >
              <option value="">{t('settings.keyDefaultAuto')}</option>
              {providerKeys.map((k) => (
                <option key={k.id} value={k.id}>
                  {k.label} {k.masked_key ? `(${k.masked_key})` : ''}
                </option>
              ))}
            </select>
          </label>
        )}

        {options.length === 0 && (
          <p data-testid={`${testId}-no-models`} className="text-[11px] text-amber-300">
            {hasAnyProviderKey ? t('settings.noCapableModelsHint') : t('settings.noKeyModelsHint')}
          </p>
        )}

        {warningText && (
          <div
            data-testid={`${testId}-warning`}
            className="flex items-start gap-2 p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-[11px] text-rose-300"
          >
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-rose-400" />
            <span>{warningText}</span>
          </div>
        )}

        {activeProvider === 'google' && (
          <FlexToggle step={step} value={value} info={info} onChange={onChange} />
        )}
        {children}
      </div>

      {(info || value.model || check) && (
        <div className="pt-2 border-t border-line/50 space-y-1.5">
          {(info || value.model) && (
            <PriceLine model={info} flex={activeProvider === 'google' && !!value.flex} step={step} />
          )}
          {check && (
            <p
              className={`text-[11px] flex items-start gap-1.5 ${check.ok ? 'text-emerald-400' : 'text-rose-400'}`}
              data-testid={`${testId}-check`}
            >
              {check.ok ? <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" /> : <XCircle className="w-3.5 h-3.5 flex-shrink-0" />}
              <span className="break-all">{check.ok ? t('settings.modelOk') : t('settings.modelFailed', { error: check.error || '' })}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export const SettingsScreen: React.FC = () => {
  const t = useT();
  const language = useUiStore((s) => s.language);
  const theme = useUiStore((s) => s.theme);
  const setLanguage = useUiStore((s) => s.setLanguage);
  const setTheme = useUiStore((s) => s.setTheme);
  const setView = useUiStore((s) => s.setView);
  const { settings, catalog, providers, keys, loaded, error, load, save, pricingChecked, pricingSource } = useSettingsStore();

  const [draft, setDraft] = useState<AppSettings | null>(settings);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [minutes, setMinutes] = useState(45);
  const [withTranslation, setWithTranslation] = useState(true);
  const [withOnscreen, setWithOnscreen] = useState(true);
  const [checks, setChecks] = useState<Partial<Record<Step, ModelCheckResult>>>({});
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [ollamaBusy, setOllamaBusy] = useState(false);
  const keyOverride = useJobStore((s) => s.pipelineConfig.gemini_api_key_override);
  const systemInfo = useJobStore((s) => s.systemInfo);
  const fetchSystemInfo = useJobStore((s) => s.fetchSystemInfo);
  const [includeLocalForAuto, setIncludeLocalForAuto] = useState(false);
  const [autoNotice, setAutoNotice] = useState<{ summary: string; detail?: string } | null>(null);
  const [ollamaDetailsOpen, setOllamaDetailsOpen] = useState(false);
  const [estimateOpen, setEstimateOpen] = useState(true);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [applyingUpdate, setApplyingUpdate] = useState(false);
  const [updateResult, setUpdateResult] = useState<UpdateApplyResponse | null>(null);

  const effectiveUpdate = updateStatus ?? systemInfo?.update ?? null;

  const onCheckUpdates = async () => {
    setCheckingUpdate(true);
    setUpdateResult(null);
    try {
      const res = await api.checkForUpdates();
      setUpdateStatus(res);
      await fetchSystemInfo();
    } catch (e: any) {
      setUpdateStatus({
        current_version: systemInfo?.version || '1.5.0',
        current_commit: '',
        branch: '',
        update_available: false,
        latest_version: systemInfo?.version || '1.5.0',
        latest_commit: '',
        commits_behind: 0,
        error: e.message || 'Check failed',
      });
    } finally {
      setCheckingUpdate(false);
    }
  };

  const onApplyUpdate = async () => {
    setApplyingUpdate(true);
    setUpdateResult(null);
    try {
      const res = await api.applyUpdate();
      setUpdateResult(res);
      if (res.updated) {
        const refreshed = await api.checkForUpdates().catch(() => null);
        if (refreshed) setUpdateStatus(refreshed);
        await fetchSystemInfo();
      }
    } catch (e: any) {
      setUpdateResult({
        updated: false,
        previous_version: systemInfo?.version || '1.5.0',
        new_version: systemInfo?.version || '1.5.0',
        new_commit: '',
        requirements_updated: false,
        frontend_rebuilt: false,
        restart_required: false,
        message: e.message || 'Update failed',
      });
    } finally {
      setApplyingUpdate(false);
    }
  };

  const runCheck = async (target: AppSettings | null = draft) => {
    if (!target) return;
    setChecking(true);
    setCheckError(null);
    try {
      setChecks(checksByStep(await api.checkModels(target, keyOverride), target));
    } catch (e: any) {
      setCheckError(e.message);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    if (!loaded) load();
  }, [loaded, load]);

  useEffect(() => {
    let active = true;
    api.getOllamaStatus()
      .then((res) => {
        if (active) setOllamaStatus(res);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!ollamaStatus?.installing) return;
    const timer = window.setInterval(async () => {
      try {
        const res = await api.getOllamaStatus();
        setOllamaStatus(res);
        if (!res.installing) {
          setOllamaBusy(false);
        }
      } catch {
        // Ignore transient polling errors
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [ollamaStatus?.installing]);

  const onSetupOllama = async () => {
    setOllamaBusy(true);
    try {
      const res = await api.setupOllama();
      setOllamaStatus(res);
      if (!res.installing) {
        setOllamaBusy(false);
      }
    } catch (e: any) {
      setOllamaBusy(false);
      setOllamaStatus((prev) =>
        prev
          ? { ...prev, installing: false, error: e.message }
          : {
              installed: false,
              running: false,
              models: [],
              selected_model: null,
              recommended_model: 'qwen2.5:7b',
              installing: false,
              step: 'error',
              error: e.message,
              download_url: 'https://ollama.com/download',
            }
      );
    }
  };

  // Adopt server settings when they (re)load, unless the user is editing identical server state
  const [syncedFrom, setSyncedFrom] = useState<AppSettings | null>(null);
  if (settings && settings !== syncedFrom) {
    const serverChanged = !syncedFrom || JSON.stringify(settings) !== JSON.stringify(syncedFrom);
    setSyncedFrom(settings);
    if (!draft || serverChanged) {
      setDraft(settings);
    }
  }

  const dirty = useMemo(() => JSON.stringify(draft) !== JSON.stringify(settings), [draft, settings]);

  const estimate = useCostEstimate(
    draft
      ? {
          duration_seconds: Math.max(1, minutes) * 60,
          enable_translation: withTranslation,
          detect_onscreen_text: withOnscreen,
          cost_saver_mode: !!draft.cost_saver_mode,
          settings: draft,
        }
      : null
  );

  const update = (step: Step, value: GeminiCallSettings) => {
    if (!draft) return;
    setDraft({ ...draft, [step]: value });
    setAutoNotice(null);
    // A result for the previous model would be misleading
    setChecks(({ [step]: _old, ...rest }) => rest);
  };
  const applyProviderToAll = (provider: AIProvider) => {
    if (!draft) return;
    const sharedKeyId = draft.translation.provider === provider ? draft.translation.key_id : null;
    const trBest = pickBestModelForProvider(catalog, providers, keys, provider, 'translation', sharedKeyId);
    const glBest = pickBestModelForProvider(catalog, providers, keys, provider, 'glossary', sharedKeyId);
    const viBest = pickBestModelForProvider(catalog, providers, keys, provider, 'vision', sharedKeyId);
    setDraft({
      ...draft,
      translation: {
        provider,
        model: trBest?.id || '',
        key_id: sharedKeyId,
        thinking: provider === 'google' ? 'low' : 'default',
        flex: false,
      },
      glossary: {
        provider,
        model: glBest?.id || trBest?.id || '',
        key_id: sharedKeyId,
        thinking: provider === 'google' ? 'low' : 'default',
        flex: false,
      },
      ...(viBest
        ? {
            vision: {
              provider,
              model: viBest.id,
              key_id: sharedKeyId,
              thinking: provider === 'google' ? 'minimal' : 'default',
              flex: false,
            },
          }
        : {}),
    });
    setAutoNotice(null);
    setChecks({});
  };

  const hw = systemInfo?.hardware;
  const maxLocalBillions = maxLocalBillionsForHardware(hw);
  const vramGb = hw?.vram_total_mb ? Math.round(hw.vram_total_mb / 1024) : 0;

  const autoRecommendBestValue = () => {
    if (!draft) return;
    const nextCalls: Partial<Record<Step, GeminiCallSettings>> = {};
    const summaryParts: string[] = [];
    const skippedMap = new Map<string, CatalogModel>();
    let choseLocal = false;

    for (const step of STEPS) {
      const res = pickBestValueForStep(catalog, providers, keys, step, includeLocalForAuto, hw);
      if (!res) continue;
      nextCalls[step] = res.call;
      const stepName =
        step === 'translation'
          ? t('settings.stepTranslation')
          : step === 'glossary'
            ? t('settings.stepGlossary')
            : t('settings.stepVision');
      const flexNote = res.call.flex ? ` (${t('settings.flexTag')})` : '';
      summaryParts.push(`${stepName}: ${providerDisplayName(res.call.provider)} · ${res.chosen.label}${flexNote}`);
      if (isLocalModelEntry(res.chosen)) choseLocal = true;
      for (const s of res.skippedHeavyLocal) {
        skippedMap.set(s.id, s);
      }
    }

    if (summaryParts.length === 0) return;
    setDraft({
      ...draft,
      ...nextCalls,
    });
    setChecks({});

    let detail: string | undefined;
    if (includeLocalForAuto) {
      if (maxLocalBillions <= 0) {
        detail = t('settings.autoRecommendNoGpu');
      } else if (skippedMap.size > 0) {
        detail = t('settings.autoRecommendSkippedHeavy', {
          gpu: hw?.gpu_name || 'GPU',
          vram: vramGb || '?',
          models: Array.from(skippedMap.values()).map((m) => m.label).join(', '),
        });
      } else if (!choseLocal) {
        detail = t('settings.autoRecommendNoLocalFit', {
          gpu: hw?.gpu_name || 'GPU',
          vram: vramGb || '?',
        });
      }
    }

    setAutoNotice({
      summary: t('settings.autoRecommendApplied', { summary: summaryParts.join(' · ') }),
      detail,
    });
  };
  const visionInfo = catalog.find(
    (m) => (m.provider || 'google') === (draft?.vision.provider || 'google') && m.id === draft?.vision.model
  );

  const onSave = async () => {
    if (!draft) return;
    setSaving(true);
    const ok = await save(draft);
    setSaving(false);
    if (ok) {
      setJustSaved(true);
      setTimeout(() => setJustSaved(false), 2000);
      runCheck(draft); // confirm right away that the saved models actually work with this key
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-4 pb-10">
      {/* Compact Header & Appearance Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-surface border border-line rounded-xl px-4 py-3 shadow-sm">
        <h1 className="text-xl font-bold tracking-tight text-zinc-50">{t('settings.title')}</h1>
        <div className="flex flex-wrap items-center gap-3">
          <Segmented<Language>
            value={language}
            onChange={setLanguage}
            options={[
              { value: 'he', label: 'עברית' },
              { value: 'en', label: 'English' },
            ]}
          />
          <Segmented<Theme>
            value={theme}
            onChange={setTheme}
            options={[
              { value: 'dark', label: <><Moon className="w-3.5 h-3.5" />{t('settings.dark')}</> },
              { value: 'light', label: <><Sun className="w-3.5 h-3.5" />{t('settings.light')}</> },
            ]}
          />
          <button
            onClick={() => setView('studio')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sunken hover:bg-zinc-800 border border-line text-xs text-zinc-300 hover:text-zinc-100 cursor-pointer transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5 rtl:rotate-180" />
            <span>{t('settings.back')}</span>
          </button>
        </div>
      </div>

      {/* Row 1: Multi-Provider API Key Vault (auto-collapsed once configured) */}
      <ApiKeySection />

      {/* Row 2: AI Models per Pipeline Step (3-Column Grid + Optimizer + Action Footer) */}
      <section className={card}>
        <div className="border-b border-line pb-3 space-y-3">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <Brain className="w-4 h-4 text-indigo-400" />
              {t('settings.aiModels')}
            </h2>
            <p className="text-[11px] text-zinc-400 mt-0.5">{t('settings.aiModelsHint')}</p>
          </div>

          <div className="p-2.5 rounded-lg bg-indigo-500/5 border border-indigo-500/20 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <button
                type="button"
                onClick={autoRecommendBestValue}
                disabled={!draft}
                data-testid="auto-recommend-models-btn"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600/90 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white cursor-pointer transition-colors shadow-sm"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>{t('settings.autoRecommendBtn')}</span>
              </button>

              <label className="inline-flex flex-wrap items-center gap-2 text-[11px] text-zinc-300 cursor-pointer select-none">
                <input
                  type="checkbox"
                  data-testid="auto-recommend-include-local"
                  checked={includeLocalForAuto}
                  onChange={(e) => setIncludeLocalForAuto(e.target.checked)}
                  className="rounded border-zinc-600 bg-sunken text-indigo-500 focus:ring-indigo-500"
                />
                <span>{t('settings.autoRecommendIncludeLocal')}</span>
                {hw && (
                  <span
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-sunken border border-line text-[10px] text-zinc-400 font-mono"
                    dir="ltr"
                  >
                    <Cpu className={`w-3 h-3 ${hw.cuda_available ? 'text-emerald-400' : 'text-zinc-500'}`} />
                    {hw.cuda_available
                      ? `${hw.gpu_name || 'CUDA'}${vramGb ? ` · ${vramGb} GB` : ''}`
                      : 'CPU'}
                  </span>
                )}
              </label>
            </div>

            {autoNotice && (
              <div
                data-testid="auto-recommend-summary"
                className="pt-2 border-t border-indigo-500/15 text-[11px] space-y-1"
              >
                <div className="flex items-start gap-1.5 text-emerald-300">
                  <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                  <span>{autoNotice.summary}</span>
                </div>
                {autoNotice.detail && (
                  <div className="flex items-start gap-1.5 text-amber-300">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>{autoNotice.detail}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {!draft ? (
          <div className="text-xs text-zinc-400 flex items-center gap-2">
            {error ? t('settings.loadFailed') : <Loader2 className="w-4 h-4 animate-spin" />}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 items-stretch">
              <ModelCard
                step="translation"
                title={t('settings.stepTranslation')}
                hint={t('settings.stepTranslationHint')}
                value={draft.translation}
                catalog={catalog}
                providers={providers}
                keys={keys}
                onChange={(v) => update('translation', v)}
                onApplyProviderAll={applyProviderToAll}
                check={checks.translation}
              />
              <ModelCard
                step="glossary"
                title={t('settings.stepGlossary')}
                hint={t('settings.stepGlossaryHint')}
                value={draft.glossary}
                catalog={catalog}
                providers={providers}
                keys={keys}
                onChange={(v) => update('glossary', v)}
                check={checks.glossary}
              />
              <ModelCard
                step="vision"
                title={t('settings.stepVision')}
                hint={t('settings.stepVisionHint')}
                value={draft.vision}
                catalog={catalog}
                providers={providers}
                keys={keys}
                onChange={(v) => update('vision', v)}
                check={checks.vision}
              >
                {visionInfo && !visionInfo.supports_media_resolution ? null : (
                  <div className="space-y-1.5 pt-2 border-t border-line/40" data-testid="vision-media-resolution">
                    <span className="block text-[11px] text-zinc-400 font-medium">{t('settings.frameDetail')}</span>
                    <Segmented<MediaResolution>
                      value={draft.vision_media_resolution}
                      itemTestIdPrefix="vision-resolution"
                      onChange={(v) => setDraft({ ...draft, vision_media_resolution: v })}
                      options={(['low', 'medium', 'high'] as MediaResolution[]).map((r) => ({
                        value: r,
                        label: t(`settings.resolution.${r}` as MessageKey),
                      }))}
                    />
                    <p className="text-[11px] text-zinc-500">{t('settings.frameDetailHint')}</p>
                  </div>
                )}
              </ModelCard>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-3 pt-3 border-t border-line/60">
              {checkError && <span className="text-[11px] text-rose-400">{checkError}</span>}
              <button
                onClick={() => runCheck()}
                disabled={checking}
                title={t('settings.testHint')}
                className="me-auto px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-xs text-zinc-200 flex items-center gap-1.5 cursor-pointer"
              >
                {checking ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <PlugZap className="w-3.5 h-3.5" />}
                {checking ? t('settings.testing') : t('settings.testModels')}
              </button>
              {dirty && <span className="text-[11px] text-amber-300">{t('settings.unsaved')}</span>}
              {error && <span className="text-[11px] text-rose-400">{error}</span>}
              <button
                onClick={() => setDraft(settings)}
                disabled={!dirty || saving}
                className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-xs text-zinc-200 cursor-pointer"
              >
                {t('settings.reset')}
              </button>
              <button
                onClick={onSave}
                disabled={!dirty || saving}
                className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white flex items-center gap-1.5 cursor-pointer"
              >
                {saving ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : justSaved ? (
                  <Check className="w-3.5 h-3.5" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
                {justSaved ? t('settings.saved') : t('settings.save')}
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Row 3: 2-Column Bento Grid — Cost-Saver / Local Ollama (left) & Cost Estimator (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        <section className={`${card} lg:col-span-5`}>
          {draft ? (
            <div
              className={`rounded-xl border transition-colors ${
                draft.cost_saver_mode
                  ? 'bg-emerald-500/10 border-emerald-500/40'
                  : 'bg-sunken border-line hover:border-zinc-600'
              }`}
            >
              <label
                className="flex items-start gap-3 p-3.5 cursor-pointer"
                data-testid="cost-saver-toggle"
              >
                <span className="flex-1 min-w-0">
                  <span className="block text-xs font-semibold text-zinc-100">
                    {t('settings.costSaver.label')}
                  </span>
                  <span className="block text-[11px] text-zinc-400 mt-0.5 leading-relaxed">
                    {t('settings.costSaver.description')}
                  </span>
                </span>
                <input
                  type="checkbox"
                  role="switch"
                  data-testid="cost-saver-switch"
                  aria-checked={!!draft.cost_saver_mode}
                  checked={!!draft.cost_saver_mode}
                  onChange={(e) => setDraft({ ...draft, cost_saver_mode: e.target.checked })}
                  className="sr-only peer"
                />
                <span
                  className={`relative mt-0.5 inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-emerald-500 ${
                    draft.cost_saver_mode ? 'bg-emerald-600' : 'bg-zinc-700'
                  }`}
                  aria-hidden
                >
                  <span
                    className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                      draft.cost_saver_mode ? 'start-4.5' : 'start-0.5'
                    }`}
                  />
                </span>
              </label>
            </div>
          ) : (
            <div className="text-xs text-zinc-400 flex items-center gap-2">
              {error ? t('settings.loadFailed') : <Loader2 className="w-4 h-4 animate-spin" />}
            </div>
          )}

          {/* Live Ollama status, quick setup action & collapsible guide */}
          <div className="space-y-2.5 text-[11px]" data-testid="ollama-setup-panel">
            <div className="flex flex-wrap items-center justify-between gap-2 bg-sunken border border-line rounded-lg p-2.5">
              <div className="flex items-center gap-2 min-w-0">
                {ollamaStatus?.running && ollamaStatus?.selected_model ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                    <span className="text-emerald-300 font-medium truncate">
                      {t('settings.costSaver.statusReady', { model: ollamaStatus.selected_model })}
                    </span>
                  </>
                ) : ollamaStatus?.installed ? (
                  <>
                    <Hourglass className="w-4 h-4 text-amber-400 flex-shrink-0" />
                    <span className="text-amber-300">
                      {t('settings.costSaver.statusInstalledNoModel')}
                    </span>
                  </>
                ) : (
                  <>
                    <Cpu className="w-4 h-4 text-zinc-400 flex-shrink-0" />
                    <span className="text-zinc-300">
                      {t('settings.costSaver.statusNotInstalled')}
                    </span>
                  </>
                )}
              </div>

              {!ollamaStatus?.installed && !ollamaStatus?.installing && !ollamaBusy && (
                // Installing Ollama is the user's step: the app never downloads and runs an installer itself
                <a
                  href={ollamaStatus?.download_url || 'https://ollama.com/download'}
                  target="_blank"
                  rel="noreferrer"
                  data-testid="ollama-install-link"
                  className="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white flex items-center gap-1.5 transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>{t('settings.costSaver.installButton')}</span>
                </a>
              )}

              {(ollamaStatus?.installed || ollamaStatus?.installing || ollamaBusy) &&
                (!ollamaStatus?.running || !ollamaStatus?.selected_model || ollamaStatus?.installing || ollamaBusy) && (
                <button
                  type="button"
                  onClick={onSetupOllama}
                  disabled={ollamaBusy || !!ollamaStatus?.installing}
                  data-testid="ollama-setup-btn"
                  className="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-xs font-semibold text-white flex items-center gap-1.5 cursor-pointer transition-colors"
                >
                  {ollamaBusy || ollamaStatus?.installing ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>
                        {t('settings.costSaver.installingStep', {
                          step:
                            ollamaStatus?.step === 'starting' || ollamaStatus?.step === 'starting_server'
                              ? t('settings.costSaver.stepStarting')
                              : t('settings.costSaver.stepPulling'),
                        })}
                      </span>
                    </>
                  ) : (
                    <>
                      <Download className="w-3.5 h-3.5" />
                      <span>
                        {ollamaStatus?.running
                          ? t('settings.costSaver.pullModelButton', {
                              model: ollamaStatus?.recommended_model || 'qwen2.5:7b',
                            })
                          : t('settings.costSaver.startServerButton')}
                      </span>
                    </>
                  )}
                </button>
              )}
            </div>

            {ollamaStatus?.error && (
              <p className="text-rose-400 text-[11px]">{ollamaStatus.error}</p>
            )}

            <button
              type="button"
              onClick={() => setOllamaDetailsOpen((v) => !v)}
              data-testid="ollama-details-toggle"
              aria-expanded={ollamaDetailsOpen}
              className="w-full flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg bg-sunken/70 hover:bg-sunken border border-line/70 text-[11px] font-medium text-zinc-300 cursor-pointer transition-colors"
            >
              <span>{t('settings.costSaver.detailsToggle')}</span>
              {ollamaDetailsOpen ? <ChevronUp className="w-3.5 h-3.5 text-zinc-400" /> : <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />}
            </button>

            {ollamaDetailsOpen && (
              <div className="space-y-2.5 pt-1">
                <div className="space-y-1 bg-sunken/50 border border-line/60 rounded-lg p-2.5">
                  <div className="font-semibold text-zinc-200 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                    <span>{t('settings.costSaver.whyTitle')}</span>
                  </div>
                  <p className="text-zinc-400 leading-relaxed">
                    {t('settings.costSaver.whyText')}
                  </p>
                </div>

                {/* Advanced Users manual link & CLI instructions */}
                <div className="bg-sunken/50 border border-line/60 rounded-lg p-2.5 space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
                      <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                      {t('settings.costSaver.advancedTitle')}
                    </span>
                    <a
                      href={ollamaStatus?.download_url || 'https://ollama.com/download'}
                      target="_blank"
                      rel="noreferrer"
                      data-testid="ollama-download-link"
                      className="inline-flex items-center gap-1 text-indigo-400 hover:text-indigo-300 font-medium"
                    >
                      <ExternalLink className="w-3 h-3" />
                      {t('settings.costSaver.downloadLink')}
                    </a>
                  </div>
                  <p className="text-zinc-400 leading-relaxed">
                    {t('settings.costSaver.advancedText')}
                  </p>
                  <code
                    dir="ltr"
                    className="block bg-sunken border border-line rounded px-2.5 py-1 text-[11px] font-mono text-emerald-300 select-all"
                  >
                    ollama pull qwen2.5:7b
                  </code>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Cost estimate */}
        <section className={`${card} lg:col-span-7`}>
          <div className="flex items-start justify-between gap-3 border-b border-line pb-3">
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <Coins className="w-4 h-4 text-amber-400" />
                {t('settings.estimate')}
              </h2>
              <p className="text-[11px] text-zinc-400 mt-0.5">{t('settings.estimateHint')}</p>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              {estimate && (
                <span className="px-2 py-0.5 rounded-full bg-sunken border border-line text-[11px] font-mono font-semibold text-zinc-200" dir="ltr">
                  ≈ {formatUsd(estimate.total_cost_usd, language)}
                </span>
              )}
              <button
                type="button"
                onClick={() => setEstimateOpen((v) => !v)}
                data-testid="estimate-collapse-toggle"
                aria-expanded={estimateOpen}
                aria-label={t('settings.estimate')}
                className="p-1 rounded-lg bg-sunken hover:bg-zinc-800 border border-line text-zinc-400 hover:text-zinc-200 cursor-pointer transition-colors"
              >
                {estimateOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {estimateOpen && (
            <>
              <div className="flex flex-wrap items-end gap-4 text-xs">
                <label className="space-y-1">
                  <span className="block text-zinc-400 font-medium">{t('settings.episodeLength')}</span>
                  <input
                    type="number"
                    min={1}
                    max={600}
                    value={minutes}
                    onChange={(e) => setMinutes(Number(e.target.value) || 1)}
                    className="w-24 bg-sunken border border-line rounded-lg px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-indigo-500"
                  />
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-zinc-300">
                  <input type="checkbox" checked={withTranslation} onChange={(e) => setWithTranslation(e.target.checked)} />
                  {t('settings.includeTranslation')}
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-zinc-300">
                  <input type="checkbox" checked={withOnscreen} onChange={(e) => setWithOnscreen(e.target.checked)} />
                  {t('settings.includeOnscreen')}
                </label>
              </div>

              {estimate && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs" data-testid="estimate-table">
                    <thead className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-line">
                      <tr>
                        <th className="py-2 text-start font-semibold">{t('settings.colStep')}</th>
                        <th className="py-2 text-start font-semibold">{t('settings.colModel')}</th>
                        <th className="py-2 text-end font-semibold">{t('settings.colRequests')}</th>
                        <th className="py-2 text-end font-semibold">{t('settings.colInput')}</th>
                        <th className="py-2 text-end font-semibold">{t('settings.colOutput')}</th>
                        <th className="py-2 text-end font-semibold">{t('settings.colCost')}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {Object.entries(estimate.stages).map(([stage, s]) => (
                        <tr key={stage} className="text-zinc-300">
                          <td className="py-2">{t(`stages.${stage}` as MessageKey)}</td>
                          <td className="py-2 text-zinc-400">
                            {modelLabel(catalog, s!.model)}
                            <span className="text-zinc-500">
                              {' · '}
                              {thinkingLabel(t, s!.thinking, catalog.find((m) => m.id === s!.model))}
                            </span>
                            {s!.flex && <span className="text-emerald-400">{' · '}{t('settings.flexTag')}</span>}
                          </td>
                          <td className="py-2 text-end font-mono">{formatNumber(s!.requests, language)}</td>
                          <td className="py-2 text-end font-mono">{formatNumber(s!.input_tokens, language)}</td>
                          <td className="py-2 text-end font-mono">{formatNumber(s!.output_tokens, language)}</td>
                          <td className="py-2 text-end font-mono" dir="ltr">
                            {formatUsd(s!.cost_usd, language)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                    <tfoot>
                      <tr className="border-t border-line text-zinc-50 font-semibold">
                        <td className="py-2" colSpan={5}>
                          {t('settings.total')}
                        </td>
                        <td className="py-2 text-end font-mono" dir="ltr" data-testid="estimate-total">
                          ≈ {formatUsd(estimate.total_cost_usd, language)}
                        </td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              )}
              <p className="text-[11px] text-zinc-500">{t('settings.estimateFootnote')}</p>
              {pricingSource && (
                <a
                  href={pricingSource}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300"
                >
                  <ExternalLink className="w-3 h-3" />
                  {t('settings.source', { date: pricingChecked })}
                </a>
              )}
            </>
          )}
        </section>
      </div>

      {/* Row 4: 2-Column Bento Grid — Storage (left) & Diagnostics + Feedback (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
        <div className="lg:col-span-8">
          <StorageSection />
        </div>

        <section className={`${card} lg:col-span-4 flex flex-col justify-between`}>
          <div className="space-y-4">
            <div className="space-y-3">
              <div className="border-b border-line pb-3">
                <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                  <Stethoscope className="w-4 h-4 text-indigo-400" />
                  {t('settings.diagnostics')}
                </h2>
                <p className="text-[11px] text-zinc-400 mt-1">{t('settings.diagnosticsHint')}</p>
              </div>
              <a
                href={api.getServerLogUrl()}
                download
                data-testid="server-log"
                className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs text-zinc-200 transition-colors"
              >
                <ScrollText className="w-3.5 h-3.5" />
                {t('settings.serverLog')}
              </a>
            </div>

            {/* Software Version & Git Updates */}
            <div className="pt-3 border-t border-line/60 space-y-2.5 text-xs" data-testid="settings-update-section">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-zinc-200">{t('settings.updater.title')}</span>
                <span className="px-2 py-0.5 rounded bg-sunken border border-line text-[11px] font-mono text-zinc-300" dir="ltr">
                  v{effectiveUpdate?.current_version || systemInfo?.version || '1.5.0'}
                  {effectiveUpdate?.current_commit ? ` (${effectiveUpdate.current_commit})` : ''}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={onCheckUpdates}
                  disabled={checkingUpdate || applyingUpdate}
                  data-testid="check-updates-btn"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sunken hover:bg-zinc-800 border border-line disabled:opacity-40 text-[11px] font-medium text-zinc-200 cursor-pointer transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${checkingUpdate ? 'animate-spin' : ''}`} />
                  <span>{checkingUpdate ? t('settings.updater.checking') : t('settings.updater.checkBtn')}</span>
                </button>

                {effectiveUpdate?.update_available && !updateResult?.updated && (
                  <button
                    type="button"
                    onClick={onApplyUpdate}
                    disabled={applyingUpdate || checkingUpdate}
                    data-testid="apply-update-btn"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-[11px] font-semibold text-white cursor-pointer transition-colors"
                  >
                    {applyingUpdate ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                    <span>{applyingUpdate ? t('settings.updater.applying') : t('settings.updater.applyBtn')}</span>
                  </button>
                )}
              </div>

              {(updateResult || effectiveUpdate) && (
                <div className="text-[11px]" data-testid="update-status-text">
                  {updateResult ? (
                    <p className={updateResult.updated ? 'text-emerald-300' : 'text-amber-300'}>
                      {updateResult.updated
                        ? t('settings.updater.appliedRestart', {
                            version: updateResult.new_version,
                            commit: updateResult.new_commit,
                          })
                        : (updateResult.skipped_reason && UPDATE_SKIP_MESSAGES[updateResult.skipped_reason]
                          ? t(UPDATE_SKIP_MESSAGES[updateResult.skipped_reason])
                          : updateResult.message)}
                    </p>
                  ) : effectiveUpdate?.error ? (
                    <p className="text-amber-300">
                      {t('settings.updater.checkFailed', { error: effectiveUpdate.error })}
                    </p>
                  ) : effectiveUpdate?.update_available ? (
                    <p className="text-emerald-300 flex flex-wrap items-center gap-1.5">
                      <span>
                        {effectiveUpdate.update_kind === 'release'
                          ? t('settings.updater.availableRelease', { version: effectiveUpdate.latest_version })
                          : t('settings.updater.availableCommits', { count: effectiveUpdate.commits_behind })}
                      </span>
                      {effectiveUpdate.release_url && (
                        <a
                          href={effectiveUpdate.release_url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-0.5 text-indigo-400 hover:text-indigo-300 underline"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </p>
                  ) : (
                    <p className="text-zinc-400">{t('settings.updater.upToDate')}</p>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="pt-3 border-t border-line/60 flex flex-wrap gap-2">
            <a
              href="https://github.com/talevi83/SubBuildStudio/issues/new?template=bug_report.md"
              target="_blank"
              rel="noopener noreferrer"
              data-testid="settings-report-bug"
              className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-sunken border border-line hover:border-rose-500/40 hover:bg-rose-500/10 text-xs font-medium text-zinc-300 hover:text-rose-300 transition-colors"
            >
              <Bug className="w-3.5 h-3.5 text-rose-400" />
              <span>{t('settings.reportBug')}</span>
              <ExternalLink className="w-3 h-3 opacity-60" />
            </a>
            <a
              href="https://github.com/talevi83/SubBuildStudio/issues/new?template=feature_request.md"
              target="_blank"
              rel="noopener noreferrer"
              data-testid="settings-request-feature"
              className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-sunken border border-line hover:border-indigo-500/40 hover:bg-indigo-500/10 text-xs font-medium text-zinc-300 hover:text-indigo-300 transition-colors"
            >
              <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
              <span>{t('settings.requestFeature')}</span>
              <ExternalLink className="w-3 h-3 opacity-60" />
            </a>
          </div>
        </section>
      </div>
    </div>
  );
};
