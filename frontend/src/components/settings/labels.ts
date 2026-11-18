import type { MessageKey } from '../../i18n';
import type { AIProvider, CatalogModel, SystemInfo, ThinkingLevel } from '../../types';

type T = (key: MessageKey, vars?: Record<string, string | number>) => string;

export function inferProviderFromModelId(modelId?: string | null): AIProvider | undefined {
  const mid = (modelId || '').trim().toLowerCase().replace(/^models\//, '');
  if (!mid) return undefined;
  if (mid.includes(':') || mid.endsWith('-cloud') || mid.startsWith('gpt-oss')) return 'ollama';
  if (mid.startsWith('gemini')) return 'google';
  if (
    mid.startsWith('gpt-') ||
    mid.startsWith('o1') ||
    mid.startsWith('o3') ||
    mid.startsWith('o4') ||
    mid.startsWith('chatgpt-')
  ) {
    return 'openai';
  }
  if (mid.startsWith('claude-')) return 'anthropic';
  return undefined;
}

export function providerDisplayName(provider?: string | null): string {
  switch (provider) {
    case 'google':
      return 'Gemini';
    case 'openai':
      return 'OpenAI';
    case 'anthropic':
      return 'Claude';
    case 'ollama':
      return 'Ollama';
    default:
      return 'AI';
  }
}

export function providerKeyEnvLabel(provider?: string | null): string {
  switch (provider) {
    case 'google':
      return 'GEMINI_API_KEY:';
    case 'openai':
      return 'OpenAI API:';
    case 'anthropic':
      return 'Anthropic API:';
    case 'ollama':
      return 'Ollama:';
    default:
      return 'AI API:';
  }
}

export function isProviderReady(
  systemInfo?: SystemInfo | null,
  provider?: string | null,
  keyOverride?: string | null,
): boolean {
  if (keyOverride && (!provider || provider === 'google')) return true;
  if (provider === 'ollama') return true;
  if (provider) {
    const provSummary = systemInfo?.providers?.[provider];
    if (provSummary !== undefined) return !!provSummary.configured;
    if (provider === 'google') return !!systemInfo?.is_gemini_configured;
    return false;
  }
  return !!systemInfo?.is_ai_configured || !!systemInfo?.is_gemini_configured;
}

export function isAnyAiReady(systemInfo?: SystemInfo | null, keyOverride?: string | null): boolean {
  return !!systemInfo?.is_ai_configured || !!systemInfo?.is_gemini_configured || !!keyOverride;
}

/** "Low", or "Model default (dynamic)" for the default option. */
export function thinkingLabel(t: T, level: ThinkingLevel | string, model?: CatalogModel): string {
  if (level === 'default') {
    const def = model?.default_level;
    const isDynamic = model?.family === '2.5' && def !== 'off';
    const shown = isDynamic ? 'dynamic' : def || 'dynamic';
    return t('settings.thinkingDefault', { level: t(`settings.thinkingLevels.${shown}` as MessageKey) });
  }
  return t(`settings.thinkingLevels.${level}` as MessageKey);
}

export function modelLabel(catalog: CatalogModel[], id: string): string {
  return catalog.find((m) => m.id === id)?.label || id;
}
