import React, { useEffect, useState } from 'react';
import type { OnscreenPosition, TranscriptionProvider } from '../../types';
import { api } from '../../services/api';
import { useJobStore } from '../../store/useJobStore';
import { useUiStore } from '../../store/useUiStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { useT, formatUsd } from '../../i18n';
import type { MessageKey } from '../../i18n';
import { useCostEstimate } from '../../hooks/useCostEstimate';
import { SOURCE_LANGUAGES, TARGET_LANGUAGES, sameLanguage, isSameExplicitLanguage } from './languages';
import {
  inferProviderFromModelId,
  isProviderReady,
  modelLabel,
  providerKeyEnvLabel,
  thinkingLabel,
} from '../settings/labels';
import { SyncOptions } from './SyncOptions';
import { GenrePresets } from './GenrePresets';
import { AlertTriangle, Sparkles, Settings, Mic, Key, Play, Loader2, FileText, Coins, Users, Layers } from 'lucide-react';

/** Accessible on/off switch with a title and a one-line description. */
export const FeatureToggle: React.FC<{
  icon: string;
  title: string;
  description: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  note?: React.ReactNode;
  children?: React.ReactNode;
  testId?: string;
}> = ({ icon, title, description, checked, onChange, note, children, testId }) => (
  <div
    className={`p-3 rounded-lg border transition-colors ${
      checked ? 'bg-indigo-500/10 border-indigo-500/40' : 'bg-surface border-line hover:border-zinc-600'
    }`}
  >
    <label className="flex items-start gap-3 cursor-pointer">
      <span className="text-lg leading-none mt-0.5" aria-hidden>
        {icon}
      </span>
      <span className="flex-1 min-w-0">
        <span className="block text-xs font-semibold text-zinc-200">{title}</span>
        <span className="block text-[11px] text-zinc-400 mt-0.5 leading-relaxed">{description}</span>
        {note && <span className="block text-[11px] mt-1">{note}</span>}
      </span>
      <input
        type="checkbox"
        role="switch"
        aria-checked={checked}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        data-testid={testId}
        className="sr-only peer"
      />
      <span
        className={`relative mt-0.5 inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500 ${
          checked ? 'bg-indigo-600' : 'bg-zinc-700'
        }`}
        aria-hidden
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'start-4.5' : 'start-0.5'}`}
        />
      </span>
    </label>
    {children}
  </div>
);

const selectClass =
  'w-full bg-surface border border-line rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-indigo-500';

export const PipelineConfig: React.FC = () => {
  const pipelineConfig = useJobStore((s) => s.pipelineConfig);
  const updatePipelineConfig = useJobStore((s) => s.updatePipelineConfig);
  const createAndStartJob = useJobStore((s) => s.createAndStartJob);
  const probeResult = useJobStore((s) => s.probeResult);
  const batchProbes = useJobStore((s) => s.batchProbes);
  const systemInfo = useJobStore((s) => s.systemInfo);
  const isLoading = useJobStore((s) => s.isLoading);
  const syncing = useJobStore((s) => s.ingestMode === 'sync');
  const syncFile = useJobStore((s) => s.syncFile);
  const syncLanguage = useJobStore((s) => s.syncLanguage);
  const syncUseGemini = useJobStore((s) => s.syncUseGemini);
  const [seriesProfiles, setSeriesProfiles] = useState<string[]>([]);
  useEffect(() => {
    api.listGlossaryProfiles().then(setSeriesProfiles).catch(() => setSeriesProfiles([]));
  }, []);
  const appSettings = useSettingsStore((s) => s.settings);
  const catalog = useSettingsStore((s) => s.catalog);
  const setView = useUiStore((s) => s.setView);
  const language = useUiStore((s) => s.language);
  const t = useT();

  const [showKeyOverride, setShowKeyOverride] = useState(false);
  const activeTranslationProvider =
    appSettings?.translation.provider || inferProviderFromModelId(appSettings?.translation.model) || 'google';
  const activeVisionProvider =
    appSettings?.vision.provider || inferProviderFromModelId(appSettings?.vision.model) || activeTranslationProvider;
  const isTranslationReady = isProviderReady(
    systemInfo,
    activeTranslationProvider,
    pipelineConfig.gemini_api_key_override,
  );
  const isVisionReady = isProviderReady(
    systemInfo,
    activeVisionProvider,
    pipelineConfig.gemini_api_key_override,
  );
  const whisperModels = systemInfo?.whisper_models?.length
    ? systemInfo.whisper_models
    : ['tiny', 'base', 'small', 'medium', 'large-v3-turbo', 'ivrit-ai/whisper-large-v3-turbo-ct2', 'large-v3'];
  // Mirrors the backend's choice: a text-based forced/"signs" track beats scanning frames
  const forcedTrack = (probeResult?.subtitle_streams || []).find(
    (s) => s.text_based && (s.forced || /forced|signs/i.test(s.title || ''))
  );
  // Embedded text-based full dialogue subtitle tracks that can be extracted directly
  const dialogueTracks = (probeResult?.subtitle_streams || []).filter(
    (s) => s.text_based && !s.forced && !/forced|signs|commentary|director/i.test(s.title || '')
  );
  const dialogueTrack =
    dialogueTracks.find((s) => s.index === pipelineConfig.selected_subtitle_stream) || dialogueTracks[0];
  const translationWithoutKey = pipelineConfig.enable_translation && !isTranslationReady;

  const activeProbes = batchProbes.length > 0 ? batchProbes : probeResult ? [probeResult] : [];
  const isBatch = activeProbes.length > 1 && !syncing;
  const totalDuration = activeProbes.reduce((sum, p) => sum + (p.duration || 0), 0);
  const hasLocalPath = activeProbes.some((p) => !p.is_upload);

  // Syncing a file in another language than the speech asks Gemini to match its lines
  const syncGeminiLines =
    syncing && syncFile && syncUseGemini && !sameLanguage(syncLanguage || syncFile.language, pipelineConfig.source_language)
      ? syncFile.cue_count
      : 0;
  const estimate = useCostEstimate(
    probeResult
      ? {
          duration_seconds: totalDuration,
          enable_translation: syncing ? false : pipelineConfig.enable_translation,
          detect_onscreen_text: pipelineConfig.detect_onscreen_text,
          has_forced_track: !!forcedTrack,
          sync_lines: syncGeminiLines,
          cost_saver_mode: appSettings?.cost_saver_mode,
          settings: appSettings,
        }
      : null
  );
  const cannotLaunch = syncing && !syncFile;

  if (!probeResult) return null;

  const translationModel = appSettings?.translation.model;
  const translationInfo = catalog.find(
    (m) => (m.provider || 'google') === activeTranslationProvider && m.id === translationModel
  );
  const estimateStages = estimate ? Object.entries(estimate.stages) : [];

  return (
    <div className="bg-surface border border-line rounded-xl p-5 shadow-lg space-y-6">
      <div className="flex items-center justify-between border-b border-line pb-3">
        <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          <Settings className="w-4 h-4 text-indigo-400" />
          {t('config.title')}
        </h2>
        <span className="text-xs text-zinc-400">{t('config.subtitle')}</span>
      </div>

      {isBatch && (
        <div
          data-testid="pipeline-batch-banner"
          className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/35 flex items-start gap-3 text-xs"
        >
          <Layers className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5 min-w-0">
            <div className="font-semibold text-indigo-200">
              {t('config.batchBannerTitle', { count: activeProbes.length })}
            </div>
            <p className="text-[11px] text-zinc-300 leading-relaxed">{t('config.batchBannerDesc')}</p>
          </div>
        </div>
      )}

      <GenrePresets
        pipelineConfig={pipelineConfig}
        whisperModels={whisperModels}
        onApply={updatePipelineConfig}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Whisper */}
        <div className="space-y-4 p-4 rounded-xl bg-sunken border border-line">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200 border-b border-line pb-2">
            <Mic className="w-4 h-4 text-indigo-400" />
            {pipelineConfig.transcription_provider === 'whisperx' ? t('config.whisperxTitle') : t('config.whisperTitle')}
          </div>
          <div className="space-y-3 text-xs">
            <div>
              <label htmlFor="transcription-provider-select" className="block text-zinc-400 mb-1 font-medium">
                {t('config.transcriptionEngineLabel')}
              </label>
              <select
                id="transcription-provider-select"
                data-testid="transcription-provider-select"
                value={pipelineConfig.transcription_provider ?? 'faster_whisper'}
                onChange={(e) =>
                  updatePipelineConfig({ transcription_provider: e.target.value as TranscriptionProvider })
                }
                className={selectClass}
              >
                <option value="faster_whisper">{t('config.engineFasterWhisper')}</option>
                <option value="whisperx">{t('config.engineWhisperX')}</option>
              </select>
              {pipelineConfig.transcription_provider === 'whisperx' && (
                <div className="mt-1.5 space-y-1 text-[11px] leading-snug">
                  <p className="text-indigo-300/90" dir="auto">{t('config.engineWhisperXHint')}</p>
                  {systemInfo?.whisperx_available === false && (
                    <p className="text-amber-300" dir="auto">{t('config.engineWhisperXFallbackNote')}</p>
                  )}
                </div>
              )}
            </div>
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">{t('config.whisperModel')}</label>
              <select
                value={pipelineConfig.whisper_model}
                onChange={(e) => updatePipelineConfig({ whisper_model: e.target.value })}
                className={selectClass}
              >
                {whisperModels.map((m) => (
                  <option key={m} value={m}>
                    {m in
                    {
                      tiny: 1,
                      base: 1,
                      small: 1,
                      medium: 1,
                      'large-v3-turbo': 1,
                      'ivrit-ai/whisper-large-v3-turbo-ct2': 1,
                      'large-v3': 1,
                    }
                      ? t(`whisperModels.${m}` as MessageKey)
                      : m}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">{t('config.sourceLanguage')}</label>
              <select
                value={pipelineConfig.source_language}
                onChange={(e) => updatePipelineConfig({ source_language: e.target.value })}
                className={selectClass}
              >
                {SOURCE_LANGUAGES.map((l) => (
                  <option key={l.value} value={l.value}>
                    {l.value === 'auto' ? t('languages.auto') : l.label}
                  </option>
                ))}
              </select>
              {pipelineConfig.source_language === 'he' &&
                pipelineConfig.whisper_model !== 'ivrit-ai/whisper-large-v3-turbo-ct2' &&
                whisperModels.includes('ivrit-ai/whisper-large-v3-turbo-ct2') && (
                  <div
                    data-testid="hebrew-model-banner"
                    className="mt-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between gap-2 text-[11px] text-amber-200"
                  >
                    <span>{t('config.hebrewModelHint')}</span>
                    <button
                      type="button"
                      data-testid="use-ivrit-model"
                      onClick={() => updatePipelineConfig({ whisper_model: 'ivrit-ai/whisper-large-v3-turbo-ct2' })}
                      className="px-2 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/40 text-amber-100 font-semibold whitespace-nowrap cursor-pointer"
                    >
                      {t('config.hebrewModelSwitch')}
                    </button>
                  </div>
                )}
            </div>
            <label className="pt-1 flex items-center justify-between gap-3 cursor-pointer">
              <span>
                <span className="text-zinc-300 font-medium block">{t('config.vad')}</span>
                <span className="text-[11px] text-zinc-500 block">{t('config.vadHint')}</span>
              </span>
              <input
                type="checkbox"
                checked={pipelineConfig.vad_filter}
                onChange={(e) => updatePipelineConfig({ vad_filter: e.target.checked })}
                className="w-4 h-4 rounded border-zinc-700 text-indigo-600 focus:ring-indigo-500 bg-zinc-800 cursor-pointer"
              />
            </label>
            <label className="pt-1 flex items-center justify-between gap-3 cursor-pointer">
              <span>
                <span className="text-zinc-300 font-medium block">{t('config.highSpeechSensitivityTitle')}</span>
                <span className="text-[11px] text-zinc-500 block">{t('config.highSpeechSensitivityDesc')}</span>
              </span>
              <input
                type="checkbox"
                checked={pipelineConfig.high_speech_sensitivity ?? false}
                onChange={(e) => updatePipelineConfig({ high_speech_sensitivity: e.target.checked })}
                className="w-4 h-4 rounded border-zinc-700 text-indigo-600 focus:ring-indigo-500 bg-zinc-800 cursor-pointer"
              />
            </label>

            {(() => {
              const vocalMode =
                pipelineConfig.vocal_isolation_mode ?? (pipelineConfig.vocal_isolation ? 'on' : 'off');
              const isActive = vocalMode !== 'off';
              return (
                <div
                  className={`p-2.5 rounded-lg border transition-colors space-y-2 ${
                    isActive
                      ? 'bg-indigo-500/10 border-indigo-500/40'
                      : 'bg-surface border-line hover:border-zinc-600'
                  }`}
                >
                  <label className="flex items-start justify-between gap-3 cursor-pointer">
                    <span className="min-w-0">
                      <span className="text-zinc-200 font-semibold block">{t('config.vocalIsolationTitle')}</span>
                      <span className="text-[11px] text-zinc-400 block mt-0.5 leading-relaxed">
                        {t('config.vocalIsolationDesc')}
                      </span>
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      data-testid="vocal-isolation-toggle"
                      aria-checked={isActive}
                      checked={isActive}
                      onChange={(e) => {
                        const nextMode = e.target.checked ? 'on' : 'off';
                        updatePipelineConfig({
                          vocal_isolation: nextMode === 'on',
                          vocal_isolation_mode: nextMode,
                        });
                      }}
                      className="sr-only peer"
                    />
                    <span
                      className={`relative mt-0.5 inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500 ${
                        isActive ? 'bg-indigo-600' : 'bg-zinc-700'
                      }`}
                      aria-hidden
                    >
                      <span
                        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                          isActive ? 'start-4.5' : 'start-0.5'
                        }`}
                      />
                    </span>
                  </label>
                  <div className="flex items-center justify-between gap-2 pt-1.5 border-t border-line/60">
                    <span className="text-[11px] text-zinc-400">{t('config.audioModeLabel')}</span>
                    <select
                      data-testid="vocal-isolation-mode-select"
                      value={vocalMode}
                      onChange={(e) => {
                        const m = e.target.value as 'off' | 'auto' | 'on';
                        updatePipelineConfig({
                          vocal_isolation: m === 'on',
                          vocal_isolation_mode: m,
                        });
                      }}
                      className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-[11px] text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                    >
                      <option value="off">{t('config.audioModeOff')}</option>
                      <option value="auto">{t('config.vocalModeAuto')}</option>
                      <option value="on">{t('config.audioModeOn')}</option>
                    </select>
                  </div>
                  {isActive && systemInfo?.vocal_isolation_available === false && (
                    <p className="text-[11px] text-amber-300 leading-snug" dir="auto">
                      {t('config.vocalIsolationFallbackNote')}
                    </p>
                  )}
                </div>
              );
            })()}

            <div className="pt-2 border-t border-line/60">
              <label
                htmlFor="transcription-context-glossary"
                className="text-zinc-300 mb-1 font-medium flex items-center gap-1.5"
              >
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                {t('config.contextGlossaryLabel')}
              </label>
              <textarea
                id="transcription-context-glossary"
                data-testid="transcription-context-glossary"
                rows={2}
                dir="auto"
                value={pipelineConfig.initial_prompt ?? pipelineConfig.context_notes ?? ''}
                onChange={(e) =>
                  updatePipelineConfig({
                    initial_prompt: e.target.value,
                    context_notes: e.target.value,
                  })
                }
                placeholder={t('config.contextGlossaryPlaceholder')}
                className="w-full bg-surface border border-line rounded-lg p-2 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 resize-none"
              />
              <p className="mt-1 text-[11px] text-zinc-500 leading-snug">{t('config.contextGlossaryHint')}</p>
            </div>

            {dialogueTrack && (
              <div
                className={`mt-2 p-3 rounded-lg border text-[11px] space-y-2 transition-colors ${
                  pipelineConfig.use_embedded_subtitles !== false
                    ? 'bg-emerald-500/10 border-emerald-500/40'
                    : 'bg-surface border-line'
                }`}
              >
                <label className="flex items-start justify-between gap-3 cursor-pointer select-none">
                  <span className="flex items-start gap-2 min-w-0">
                    <FileText
                      className={`w-4 h-4 mt-0.5 shrink-0 ${
                        pipelineConfig.use_embedded_subtitles !== false ? 'text-emerald-400' : 'text-zinc-400'
                      }`}
                    />
                    <span className="space-y-0.5">
                      <span className="block font-semibold text-zinc-200">{t('config.useEmbeddedTitle')}</span>
                      <span className="block text-zinc-400 leading-relaxed">{t('config.useEmbeddedDescription')}</span>
                    </span>
                  </span>
                  <input
                    type="checkbox"
                    role="switch"
                    data-testid="use-embedded-subtitles-toggle"
                    aria-checked={pipelineConfig.use_embedded_subtitles !== false}
                    checked={pipelineConfig.use_embedded_subtitles !== false}
                    onChange={(e) => updatePipelineConfig({ use_embedded_subtitles: e.target.checked })}
                    className="sr-only peer"
                  />
                  <span
                    className={`relative mt-0.5 inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-emerald-500 ${
                      pipelineConfig.use_embedded_subtitles !== false ? 'bg-emerald-600' : 'bg-zinc-700'
                    }`}
                    aria-hidden
                  >
                    <span
                      className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                        pipelineConfig.use_embedded_subtitles !== false ? 'start-4.5' : 'start-0.5'
                      }`}
                    />
                  </span>
                </label>

                {pipelineConfig.use_embedded_subtitles !== false ? (
                  <div className="pt-2 border-t border-emerald-500/20 space-y-2">
                    <p className="text-emerald-300 font-medium">
                      {t('config.embeddedDialogueFound', {
                        name:
                          (dialogueTrack.title || dialogueTrack.language?.toUpperCase() || `#${dialogueTrack.index}`) +
                          (dialogueTrack.language &&
                          dialogueTrack.title &&
                          !dialogueTrack.title.toLowerCase().includes(dialogueTrack.language.toLowerCase())
                            ? ` (${dialogueTrack.language.toUpperCase()})`
                            : ''),
                      })}
                    </p>
                    {dialogueTracks.length > 1 && (
                      <div>
                        <label className="block text-zinc-400 mb-1 font-medium">{t('config.selectEmbeddedTrack')}</label>
                        <select
                          value={
                            pipelineConfig.selected_subtitle_stream == null
                              ? 'auto'
                              : String(pipelineConfig.selected_subtitle_stream)
                          }
                          onChange={(e) =>
                            updatePipelineConfig({
                              selected_subtitle_stream: e.target.value === 'auto' ? null : Number(e.target.value),
                            })
                          }
                          className={selectClass}
                        >
                          <option value="auto">
                            {t('config.embeddedAutoTrack', {
                              name:
                                dialogueTracks[0].title ||
                                dialogueTracks[0].language?.toUpperCase() ||
                                `#${dialogueTracks[0].index}`,
                            })}
                          </option>
                          {dialogueTracks.map((tr) => (
                            <option key={tr.index} value={String(tr.index)}>
                              #{tr.index} {tr.title || tr.language?.toUpperCase() || t('streams.subtitles')}
                              {tr.language ? ` (${tr.language.toUpperCase()})` : ''} · {tr.codec_name}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    <label className="flex items-start justify-between gap-3 pt-2 border-t border-emerald-500/20 cursor-pointer select-none">
                      <span className="space-y-0.5">
                        <span className="block font-semibold text-zinc-200">{t('config.supplementWithWhisperTitle')}</span>
                        <span className="block text-zinc-400 text-[10px] leading-relaxed">
                          {t('config.supplementWithWhisperDesc')}
                        </span>
                      </span>
                      <input
                        type="checkbox"
                        checked={pipelineConfig.supplement_with_whisper ?? false}
                        onChange={(e) => updatePipelineConfig({ supplement_with_whisper: e.target.checked })}
                        className="mt-0.5 w-4 h-4 rounded border-zinc-700 text-emerald-600 focus:ring-emerald-500 bg-zinc-800 cursor-pointer"
                      />
                    </label>
                  </div>
                ) : (
                  <p className="pt-2 border-t border-line text-amber-300/90 font-medium">
                    {t('config.embeddedWhipserActive')}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Translation (or, when syncing a file, how its lines are matched) */}
        {syncing ? (
          <SyncOptions />
        ) : (
        <div className="space-y-4 p-4 rounded-xl bg-sunken border border-line">
          <div className="flex items-center justify-between border-b border-line pb-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              {t('config.translationTitle')}
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <span className="text-[11px] text-zinc-400">{t('config.enable')}</span>
              <input
                type="checkbox"
                checked={pipelineConfig.enable_translation}
                onChange={(e) => updatePipelineConfig({ enable_translation: e.target.checked })}
                className="w-4 h-4 rounded border-zinc-700 text-indigo-600 focus:ring-indigo-500 bg-zinc-800 cursor-pointer"
              />
            </label>
          </div>

          {pipelineConfig.enable_translation ? (
            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-zinc-400 mb-1 font-medium">{t('config.targetLanguage')}</label>
                <select
                  value={pipelineConfig.target_language}
                  onChange={(e) => updatePipelineConfig({ target_language: e.target.value })}
                  className={selectClass}
                >
                  {TARGET_LANGUAGES.map((l) => (
                    <option key={l.value} value={l.value}>
                      {l.label}
                    </option>
                  ))}
                </select>
                {isSameExplicitLanguage(pipelineConfig.source_language, pipelineConfig.target_language) && (
                  <div
                    data-testid="same-lang-polish-banner"
                    className="mt-2 p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30 space-y-1.5 text-[11px]"
                  >
                    <div className="font-semibold text-cyan-200">{t('config.sameLangPolishTitle')}</div>
                    <p className="text-cyan-300/90 leading-relaxed">{t('config.sameLangPolishDesc')}</p>
                    <button
                      type="button"
                      data-testid="disable-translation-btn"
                      onClick={() => updatePipelineConfig({ enable_translation: false })}
                      className="px-2 py-1 rounded bg-surface hover:bg-zinc-800 border border-line text-zinc-200 font-medium cursor-pointer"
                    >
                      {t('config.sameLangDisableAi')}
                    </button>
                  </div>
                )}
              </div>

              <div className="p-2.5 rounded-lg bg-surface border border-line space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[11px] text-zinc-300 min-w-0 truncate">
                    {appSettings
                      ? t('config.modelSummary', {
                          model: modelLabel(catalog, appSettings.translation.model),
                          thinking: thinkingLabel(t, appSettings.translation.thinking, translationInfo),
                        }) + (appSettings.translation.flex ? ` · ${t('settings.flexTag')}` : '')
                      : t('config.models')}
                  </span>
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                        isTranslationReady ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
                      }`}
                      title={providerKeyEnvLabel(activeTranslationProvider)}
                    >
                      <Key className="w-2.5 h-2.5" />
                      {isTranslationReady ? t('config.keyLoaded') : t('config.keyMissing')}
                    </span>
                    <button
                      type="button"
                      onClick={() => setView('settings')}
                      className="text-[11px] text-indigo-400 hover:text-indigo-300 underline cursor-pointer whitespace-nowrap"
                    >
                      {t('config.changeInSettings')}
                    </button>
                  </div>
                </div>
                {translationWithoutKey && (
                  <div className="flex items-start gap-1.5 text-[11px] text-amber-300 pt-1 border-t border-line/60">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>{t('config.noKeyWarning')}</span>
                  </div>
                )}
                {activeTranslationProvider === 'google' && (
                  <div className="pt-1 border-t border-line/60 space-y-1.5">
                    <div className="flex items-center justify-between text-[10px] text-zinc-400">
                      <span dir="ltr">{providerKeyEnvLabel(activeTranslationProvider)}</span>
                      <button
                        type="button"
                        onClick={() => setShowKeyOverride(!showKeyOverride)}
                        className="text-indigo-400 hover:text-indigo-300 underline cursor-pointer"
                      >
                        {showKeyOverride ? t('config.hideOverride') : t('config.overrideKey')}
                      </button>
                    </div>
                    {(showKeyOverride || translationWithoutKey) && (
                      <input
                        type="password"
                        dir="ltr"
                        placeholder={t('config.keyPlaceholder')}
                        value={pipelineConfig.gemini_api_key_override || ''}
                        onChange={(e) => updatePipelineConfig({ gemini_api_key_override: e.target.value })}
                        className="w-full bg-sunken border border-line rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 font-mono"
                      />
                    )}
                  </div>
                )}
              </div>

              <div>
                <label className="text-zinc-400 mb-1 font-medium flex items-center gap-1.5">
                  <Users className="w-3.5 h-3.5 text-zinc-400" />
                  {t('config.series')}
                </label>
                <input
                  dir="auto"
                  list="series-profiles"
                  value={pipelineConfig.glossary_profile || ''}
                  onChange={(e) => updatePipelineConfig({ glossary_profile: e.target.value })}
                  placeholder={t('config.seriesPlaceholder')}
                  data-testid="series-profile"
                  className="w-full bg-surface border border-line rounded-lg px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500"
                />
                <datalist id="series-profiles">
                  {seriesProfiles.map((p) => (
                    <option key={p} value={p} />
                  ))}
                </datalist>
                <p className="mt-1 text-[11px] text-zinc-500">{t('config.seriesHelp')}</p>
              </div>

              <div>
                <label className="text-zinc-400 mb-1 font-medium flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-zinc-400" />
                  {t('config.notes')}
                </label>
                <textarea
                  rows={2}
                  dir="auto"
                  value={pipelineConfig.context_notes ?? pipelineConfig.initial_prompt ?? ''}
                  onChange={(e) =>
                    updatePipelineConfig({
                      context_notes: e.target.value,
                      initial_prompt: e.target.value,
                    })
                  }
                  placeholder={t('config.notesPlaceholder')}
                  className="w-full bg-surface border border-line rounded-lg p-2 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 resize-none"
                />
              </div>

              <label className="flex items-start gap-2.5 pt-2.5 border-t border-line/40 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={pipelineConfig.convert_to_metric ?? false}
                  onChange={(e) => updatePipelineConfig({ convert_to_metric: e.target.checked })}
                  className="mt-0.5 rounded border-zinc-700 text-indigo-500 focus:ring-indigo-400 bg-zinc-800 cursor-pointer"
                />
                <div className="space-y-0.5">
                  <span className="text-zinc-200 font-medium text-xs flex items-center gap-1.5">
                    <span>📏</span>
                    <span>{t('config.metricTitle')}</span>
                  </span>
                  <span className="text-zinc-400 text-[11px] leading-relaxed block">
                    {t('config.metricDescription')}
                  </span>
                </div>
              </label>

              <label className="flex items-start gap-2.5 pt-2.5 border-t border-line/40 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={pipelineConfig.remove_sdh_speakers ?? false}
                  onChange={(e) => updatePipelineConfig({ remove_sdh_speakers: e.target.checked })}
                  className="mt-0.5 rounded border-zinc-700 text-indigo-500 focus:ring-indigo-400 bg-zinc-800 cursor-pointer"
                />
                <div className="space-y-0.5">
                  <span className="text-zinc-200 font-medium text-xs flex items-center gap-1.5">
                    <span>🔇</span>
                    <span>{t('config.removeSdhSpeakersTitle')}</span>
                  </span>
                  <span className="text-zinc-400 text-[11px] leading-relaxed block">
                    {t('config.removeSdhSpeakersDescription')}
                  </span>
                </div>
              </label>

              <label className="flex items-start gap-2.5 pt-2.5 border-t border-line/40 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={pipelineConfig.skip_end_credits ?? true}
                  onChange={(e) => updatePipelineConfig({ skip_end_credits: e.target.checked })}
                  className="mt-0.5 rounded border-zinc-700 text-indigo-500 focus:ring-indigo-400 bg-zinc-800 cursor-pointer"
                />
                <div className="space-y-0.5">
                  <span className="text-zinc-200 font-medium text-xs flex items-center gap-1.5">
                    <span>🎬</span>
                    <span>{t('config.skipEndCreditsTitle')}</span>
                  </span>
                  <span className="text-zinc-400 text-[11px] leading-relaxed block">
                    {t('config.skipEndCreditsDescription')}
                  </span>
                </div>
              </label>
            </div>
          ) : (
            <div className="py-6 text-center text-zinc-500 text-xs">{t('config.translationDisabled')}</div>
          )}
        </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {(() => {
          const commsMode = pipelineConfig.comms_boost_mode ?? (pipelineConfig.comms_boost ? 'on' : 'off');
          const commsActive = commsMode !== 'off';
          return (
            <FeatureToggle
              icon="📻"
              title={t('config.commsTitle')}
              description={t('config.commsDescription')}
              checked={commsActive}
              onChange={(v) => {
                const nextMode = v ? 'auto' : 'off';
                updatePipelineConfig({
                  comms_boost: false,
                  comms_boost_mode: nextMode,
                });
              }}
            >
              {commsActive && (
                <div className="mt-3 pt-2.5 border-t border-indigo-500/20 flex items-center justify-between gap-2 text-xs">
                  <span className="text-zinc-300 font-medium">{t('config.audioModeLabel')}</span>
                  <select
                    data-testid="comms-boost-mode-select"
                    value={commsMode}
                    onChange={(e) => {
                      const m = e.target.value as 'off' | 'auto' | 'on';
                      updatePipelineConfig({
                        comms_boost: m === 'on',
                        comms_boost_mode: m,
                      });
                    }}
                    className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                  >
                    <option value="auto">{t('config.commsModeAuto')}</option>
                    <option value="on">{t('config.commsModeOn')}</option>
                    <option value="off">{t('config.audioModeOff')}</option>
                  </select>
                </div>
              )}
            </FeatureToggle>
          );
        })()}
        <FeatureToggle
          icon="🗣️"
          title={t('config.diarization')}
          description={t('config.diarizationDesc')}
          checked={!!pipelineConfig.enable_diarization}
          onChange={(v) => updatePipelineConfig({ enable_diarization: v })}
          testId="toggle-diarization"
        />
        <FeatureToggle
          icon="📺"
          title={syncing ? t('config.syncOnscreenTitle') : t('config.onscreenTitle')}
          description={syncing ? t('config.syncOnscreenDescription') : t('config.onscreenDescription')}
          checked={pipelineConfig.detect_onscreen_text}
          onChange={(v) => updatePipelineConfig({ detect_onscreen_text: v })}
          note={
            pipelineConfig.detect_onscreen_text &&
            (forcedTrack ? (
              <span className="text-emerald-400">
                {t('config.forcedFound', { name: forcedTrack.title || forcedTrack.language || `#${forcedTrack.index}` })}
              </span>
            ) : !isVisionReady ? (
              <span className="text-amber-300">{t('config.onscreenNeedsKey')}</span>
            ) : null)
          }
        >
          {pipelineConfig.detect_onscreen_text && !forcedTrack && (
            <div className="mt-3 pt-3 border-t border-indigo-500/20 space-y-2 text-xs">
              <div className="flex items-center justify-between text-zinc-300 font-medium">
                <span>{t('config.onscreenSensitivityLabel')}</span>
                <span className="text-indigo-400 font-semibold text-[11px]">
                  {(pipelineConfig.onscreen_sensitivity ?? 2) === 1 && t('config.onscreenSensitivityStrict')}
                  {(pipelineConfig.onscreen_sensitivity ?? 2) === 2 && t('config.onscreenSensitivityBalanced')}
                  {(pipelineConfig.onscreen_sensitivity ?? 2) === 3 && t('config.onscreenSensitivityBroad')}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={3}
                step={1}
                value={pipelineConfig.onscreen_sensitivity ?? 2}
                onChange={(e) => updatePipelineConfig({ onscreen_sensitivity: Number(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-zinc-400">
                <button
                  type="button"
                  onClick={() => updatePipelineConfig({ onscreen_sensitivity: 1 })}
                  className={`hover:text-zinc-200 cursor-pointer ${(pipelineConfig.onscreen_sensitivity ?? 2) === 1 ? 'text-indigo-300 font-bold' : ''}`}
                >
                  1 · {t('config.onscreenSensitivityStrict')}
                </button>
                <button
                  type="button"
                  onClick={() => updatePipelineConfig({ onscreen_sensitivity: 2 })}
                  className={`hover:text-zinc-200 cursor-pointer ${(pipelineConfig.onscreen_sensitivity ?? 2) === 2 ? 'text-indigo-300 font-bold' : ''}`}
                >
                  2 · {t('config.onscreenSensitivityBalanced')}
                </button>
                <button
                  type="button"
                  onClick={() => updatePipelineConfig({ onscreen_sensitivity: 3 })}
                  className={`hover:text-zinc-200 cursor-pointer ${(pipelineConfig.onscreen_sensitivity ?? 2) === 3 ? 'text-indigo-300 font-bold' : ''}`}
                >
                  3 · {t('config.onscreenSensitivityBroad')}
                </button>
              </div>
              <p className="text-[11px] text-zinc-400 leading-relaxed">
                {(pipelineConfig.onscreen_sensitivity ?? 2) === 1 && t('config.onscreenSensitivityDescStrict')}
                {(pipelineConfig.onscreen_sensitivity ?? 2) === 2 && t('config.onscreenSensitivityDescBalanced')}
                {(pipelineConfig.onscreen_sensitivity ?? 2) === 3 && t('config.onscreenSensitivityDescBroad')}
              </p>
            </div>
          )}
          {pipelineConfig.detect_onscreen_text && (
            <>
              <label className="flex items-start gap-2.5 mt-3 pt-3 border-t border-indigo-500/20 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={pipelineConfig.allow_onscreen_overlap ?? true}
                  onChange={(e) => updatePipelineConfig({ allow_onscreen_overlap: e.target.checked })}
                  className="mt-0.5 rounded border-zinc-700 text-indigo-500 focus:ring-indigo-400 bg-zinc-800 cursor-pointer"
                />
                <div className="space-y-0.5">
                  <span className="text-zinc-200 font-medium text-xs block">
                    {t('config.onscreenOverlapTitle')}
                  </span>
                  <span className="text-zinc-400 text-[11px] leading-relaxed block">
                    {t('config.onscreenOverlapDescription')}
                  </span>
                </div>
              </label>

              <div className="pt-2.5 mt-2.5 border-t border-indigo-500/20 flex items-center justify-between text-xs">
                <span className="text-zinc-300 font-medium">
                  {t('config.onscreenPositionLabel')}
                </span>
                <select
                  value={pipelineConfig.onscreen_position || 'top_center'}
                  onChange={(e) => updatePipelineConfig({ onscreen_position: e.target.value as OnscreenPosition })}
                  className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                >
                  <option value="top_center">{t('settings.pos.top_center')}</option>
                  <option value="middle_right">{t('settings.pos.middle_right')}</option>
                  <option value="middle_left">{t('settings.pos.middle_left')}</option>
                  <option value="top_right">{t('settings.pos.top_right')}</option>
                  <option value="top_left">{t('settings.pos.top_left')}</option>
                </select>
              </div>
            </>
          )}
        </FeatureToggle>
      </div>

      {!syncing && hasLocalPath && (
        <div data-testid="auto-save-srt-toggle-wrapper">
          <FeatureToggle
            icon="💾"
            title={t('config.autoSaveSrtTitle')}
            description={t('config.autoSaveSrtDescription')}
            checked={Boolean(pipelineConfig.auto_save_srt)}
            onChange={(v) => updatePipelineConfig({ auto_save_srt: v })}
          />
        </div>
      )}

      {/* Cost estimate for this file */}
      {estimate && (
        <div className="p-3 rounded-lg bg-sunken border border-line text-xs flex items-start gap-3" data-testid="file-estimate">
          <Coins className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            {estimateStages.length === 0 ? (
              <span className="text-zinc-400">{t('config.estimateNone')}</span>
            ) : (
              <>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-zinc-300 font-medium">
                    {isBatch ? t('config.estimateBatchTitle', { count: activeProbes.length }) : t('config.estimateTitle')}
                  </span>
                  <span className="text-zinc-50 font-semibold" dir="ltr">
                    ≈ {formatUsd(estimate.total_cost_usd, language)}
                  </span>
                </div>
                <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-zinc-500">
                  {estimateStages.map(([stage, s]) => (
                    <span key={stage}>
                      {t(`stages.${stage}` as MessageKey)}: {modelLabel(catalog, s!.model)}
                      {s!.flex ? ` (${t('settings.flexTag')})` : ''} ·{' '}
                      <span dir="ltr">{formatUsd(s!.cost_usd, language)}</span>
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="pt-2 space-y-2">
        {cannotLaunch && <p className="text-center text-[11px] text-amber-300">{t('config.syncNeedFile')}</p>}
        <button
          onClick={createAndStartJob}
          disabled={isLoading || cannotLaunch}
          data-testid="launch"
          className="w-full py-3 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-60 text-white font-bold text-sm shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 cursor-pointer transition-all active:scale-[0.99]"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {t('config.launching')}
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              {syncing
                ? t('config.launchSync')
                : isBatch
                ? t('config.launchBatch', { count: activeProbes.length })
                : t('config.launch')}
            </>
          )}
        </button>
      </div>
    </div>
  );
};
