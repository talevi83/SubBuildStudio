import React, { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../services/api';
import { confirmDialog } from '../../store/useDialogStore';
import { useJobStore } from '../../store/useJobStore';
import {
  MAX_MARGIN_V,
  MAX_OUTLINE_WIDTH,
  MAX_SUBTITLE_FONT_SIZE,
  MIN_MARGIN_V,
  MIN_OUTLINE_WIDTH,
  MIN_SUBTITLE_FONT_SIZE,
  SUBTITLE_FONTS,
  SUBTITLE_STYLE_PRESETS,
  type SubtitleStylePresetId,
  buildSubtitleTextShadow,
  fontSizePxToEmbedTier,
  hexToRgba,
  subtitleFontStack,
  useUiStore,
} from '../../store/useUiStore';
import { isRtlLanguage, type EmbedMode, type EmbedQuality, type EmbedVideoOptions, type EmbedVideoStatus, type SubtitleFontFamily, type SubtitleHighlightMode } from '../../types';
import { useT } from '../../i18n';
import { CheckCircle2, Download, Film, Flame, Layers, Loader2, Palette, X, XCircle } from 'lucide-react';

const PRESET_KEY_MAP: Record<
  SubtitleStylePresetId,
  'presetCinema' | 'presetYoutube' | 'presetReels' | 'presetPodcast' | 'presetBox'
> = {
  cinema: 'presetCinema',
  youtube: 'presetYoutube',
  reels: 'presetReels',
  podcast: 'presetPodcast',
  box: 'presetBox',
};

interface Props {
  jobId: string;
  /** Uploaded through the browser: the result is downloaded instead of saved next to the video */
  isUpload: boolean;
  hasTranslation: boolean;
  /** The toolbar's RTL punctuation choice, applied to the embedded text too */
  applyRtl: boolean;
  hasSpeakers?: boolean;
  onClose: () => void;
  onDownload: (url: string) => void;
}

const selectClass =
  'w-full bg-sunken border border-line rounded-lg px-2 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500';

/** Make a copy of the video with the subtitles inside: a switchable track, or burned into the picture. */
export const EmbedVideoDialog: React.FC<Props> = ({ jobId, isUpload, hasTranslation, applyRtl, hasSpeakers: propHasSpeakers, onClose, onDownload }) => {
  const t = useT();
  const activeJob = useJobStore((s) => s.activeJob);
  const subtitleStyle = useUiStore((s) => s.subtitleStyle);
  const setSubtitleStyle = useUiStore((s) => s.setSubtitleStyle);
  const hasSpeakers = propHasSpeakers ?? Boolean(activeJob?.speakers && Object.keys(activeJob.speakers).length > 0);
  const [mode, setMode] = useState<EmbedMode>('soft');
  const [useTranslated, setUseTranslated] = useState(hasTranslation);
  const [bilingual, setBilingual] = useState(false);
  const [colorSpeakers, setColorSpeakers] = useState(true);
  const [setDefault, setSetDefault] = useState(true);
  const [quality, setQuality] = useState<EmbedQuality>('standard');
  const [status, setStatus] = useState<EmbedVideoStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [failedPreviewUrl, setFailedPreviewUrl] = useState<string | null>(null);
  const downloaded = useRef<string | null>(null);

  const previewSample = useMemo(() => {
    const segs = activeJob?.segments || [];
    const sample = segs.find(
      (s) => s.segment_type !== 'ON_SCREEN' && ((s.translated_text || s.original_text || '').trim().length > 0)
    );
    if (!sample) {
      return { time: undefined as number | undefined, orig: t('embed.previewSample'), trans: t('embed.previewSample') };
    }
    return {
      time: (sample.start_time + sample.end_time) / 2,
      orig: sample.original_text || t('embed.previewSample'),
      trans: sample.translated_text || sample.original_text || t('embed.previewSample'),
    };
  }, [activeJob?.segments, t]);

  const previewFrameUrl = useMemo(() => api.getPreviewFrameUrl(jobId, previewSample.time), [jobId, previewSample.time]);
  const frameFailed = failedPreviewUrl === previewFrameUrl;

  const previewIsRtl =
    useTranslated && hasTranslation
      ? isRtlLanguage(activeJob?.target_language)
      : isRtlLanguage(activeJob?.detected_language || activeJob?.source_language);

  const running = status?.state === 'running';

  // A run started earlier (dialog closed meanwhile) is picked up again
  useEffect(() => {
    let alive = true;
    api
      .getLatestEmbedVideo(jobId)
      .then((s) => {
        if (alive && s?.state === 'running') {
          setMode(s.mode);
          setStatus(s);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [jobId]);

  useEffect(() => {
    if (!running || !status?.task_id) return;
    const taskId = status.task_id;
    const timer = setInterval(() => {
      api
        .getEmbedVideoStatus(jobId, taskId)
        .then(setStatus)
        .catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setStatus(null);
        });
    }, 700);
    return () => clearInterval(timer);
  }, [jobId, running, status?.task_id]);

  // An uploaded video's copy is handed over as a download as soon as it is ready
  useEffect(() => {
    if (status?.state === 'done' && status.download_url && downloaded.current !== status.task_id) {
      downloaded.current = status.task_id;
      onDownload(status.download_url);
    }
  }, [status, onDownload]);

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      const options: EmbedVideoOptions = {
        mode,
        use_translated: useTranslated && hasTranslation,
        apply_rtl: applyRtl,
        set_default: setDefault,
        font_size: fontSizePxToEmbedTier(subtitleStyle.fontSizePx),
        quality,
        ...(mode === 'burn'
          ? {
              font_family: subtitleStyle.fontFamily,
              font_size_px: subtitleStyle.fontSizePx,
              text_color: subtitleStyle.textColor,
              outline_color: subtitleStyle.outlineColor,
              outline_width: subtitleStyle.outlineWidth,
              background_color: subtitleStyle.backgroundColor,
              background_opacity: subtitleStyle.backgroundOpacity,
              margin_v: subtitleStyle.marginV,
              bold: subtitleStyle.bold,
              highlight_mode: subtitleStyle.highlightMode,
              highlight_color: subtitleStyle.highlightColor,
              color_speakers: hasSpeakers ? colorSpeakers : false,
              bilingual: bilingual && hasTranslation && useTranslated,
            }
          : {}),
      };
      let result = await api.startEmbedVideo(jobId, options);
      if (result.state === 'exists') {
        const replace = await confirmDialog({
          title: t('toolbar.confirmOverwrite', { path: result.path }),
          confirmLabel: t('ux.replaceFile'),
          danger: true,
        });
        if (!replace) return;
        result = await api.startEmbedVideo(jobId, { ...options, overwrite: true });
      }
      setStatus(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!status?.task_id) return;
    try {
      setStatus(await api.cancelEmbedVideo(jobId, status.task_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const modeCard = (value: EmbedMode, icon: React.ReactNode, title: string, help: string) => (
    <button
      type="button"
      onClick={() => setMode(value)}
      disabled={running}
      data-testid={`embed-mode-${value}`}
      className={`text-start p-3 rounded-lg border transition-colors cursor-pointer disabled:cursor-not-allowed ${
        mode === value ? 'border-indigo-500 bg-indigo-500/10' : 'border-line hover:border-zinc-500'
      }`}
    >
      <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-100">
        {icon}
        {title}
      </div>
      <p className="mt-1 text-[11px] text-zinc-400 leading-relaxed">{help}</p>
    </button>
  );

  const finished = status && status.state !== 'running';
  const percent = Math.max(0, Math.min(100, status?.percent ?? 0));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-surface border border-line rounded-xl p-5 space-y-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        data-testid="embed-dialog"
      >
        <div className="flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            <Film className="w-4 h-4 text-indigo-400" />
            {t('embed.title')}
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-200" aria-label={t('embed.close')}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {modeCard('soft', <Layers className="w-3.5 h-3.5 text-cyan-400" />, t('embed.softTitle'), t('embed.softHelp'))}
          {modeCard('burn', <Flame className="w-3.5 h-3.5 text-amber-400" />, t('embed.burnTitle'), t('embed.burnHelp'))}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1 text-[11px] text-zinc-400">
            <span>{t('embed.subtitles')}</span>
            <select
              value={useTranslated && hasTranslation ? 'translated' : 'original'}
              onChange={(e) => setUseTranslated(e.target.value === 'translated')}
              disabled={running}
              className={selectClass}
              data-testid="embed-language"
            >
              {hasTranslation && <option value="translated">{t('embed.translated')}</option>}
              <option value="original">{t('embed.original')}</option>
            </select>
          </label>

          {mode === 'soft' ? (
            <label className="flex items-end gap-2 pb-1.5 text-xs text-zinc-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={setDefault}
                onChange={(e) => setSetDefault(e.target.checked)}
                disabled={running}
                className="w-3.5 h-3.5 rounded border-zinc-700 text-indigo-600 bg-zinc-800"
              />
              {t('embed.setDefault')}
            </label>
          ) : (
            <label className="space-y-1 text-[11px] text-zinc-400">
              <span>{t('embed.quality')}</span>
              <select
                value={quality}
                onChange={(e) => setQuality(e.target.value as EmbedQuality)}
                disabled={running}
                className={selectClass}
                data-testid="embed-quality"
              >
                <option value="standard">{t('embed.qualityStandard')}</option>
                <option value="high">{t('embed.qualityHigh')}</option>
              </select>
            </label>
          )}
        </div>

        {mode === 'burn' && (
          <div
            data-testid="embed-burn-style"
            className="bg-sunken border border-line rounded-lg p-3 space-y-3"
          >
            {/* Live Burn-in Frame Preview */}
            <div
              data-testid="embed-live-preview"
              className="relative w-full aspect-video rounded-lg overflow-hidden bg-gradient-to-br from-zinc-900 via-zinc-950 to-slate-900 border border-line/80 flex items-center justify-center"
            >
              {!frameFailed && (
                <img
                  key={previewFrameUrl}
                  src={previewFrameUrl}
                  alt={t('embed.livePreview')}
                  onError={() => setFailedPreviewUrl(previewFrameUrl)}
                  className="w-full h-full object-cover"
                />
              )}
              <span className="absolute top-2 start-2 px-1.5 py-0.5 rounded bg-black/60 text-[10px] text-zinc-300 border border-white/10">
                {t('embed.livePreview')}
              </span>
              <div
                className="absolute inset-x-4 flex flex-col items-center pointer-events-none"
                style={{ bottom: `${Math.min(42, Math.max(4, Math.round(subtitleStyle.marginV * 0.28)))}%` }}
              >
                <div
                  className="max-w-[92%] px-3 py-1 rounded text-center space-y-0.5"
                  style={{
                    fontFamily: subtitleFontStack(subtitleStyle.fontFamily),
                    fontSize: `${Math.max(11, Math.round(subtitleStyle.fontSizePx * 0.85))}px`,
                    fontWeight: subtitleStyle.bold ? 700 : 500,
                    backgroundColor:
                      subtitleStyle.backgroundOpacity > 0
                        ? hexToRgba(subtitleStyle.backgroundColor, subtitleStyle.backgroundOpacity)
                        : 'transparent',
                    textShadow: buildSubtitleTextShadow(subtitleStyle),
                  }}
                >
                  {bilingual && hasTranslation && useTranslated && previewSample.orig !== previewSample.trans && (
                    <p className="text-zinc-200/90 text-[0.82em] leading-snug whitespace-pre-line">
                      {previewSample.orig}
                    </p>
                  )}
                  <p
                    dir={previewIsRtl ? 'rtl' : 'ltr'}
                    className="leading-snug whitespace-pre-line"
                    style={{ color: subtitleStyle.textColor }}
                  >
                    {subtitleStyle.highlightMode !== 'none'
                      ? (useTranslated && hasTranslation ? previewSample.trans : previewSample.orig)
                          .split(/(\s+)/)
                          .map((part, idx) =>
                            idx === 0 ? (
                              <span key={idx} style={{ color: subtitleStyle.highlightColor, fontWeight: 800 }}>
                                {part}
                              </span>
                            ) : (
                              <React.Fragment key={idx}>{part}</React.Fragment>
                            )
                          )
                      : useTranslated && hasTranslation
                      ? previewSample.trans
                      : previewSample.orig}
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
              <span className="flex items-center gap-1.5 text-zinc-300 font-medium">
                <Palette className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                {t('embed.styleSummary', {
                  font: subtitleStyle.fontFamily,
                  size: subtitleStyle.fontSizePx,
                  opacity: subtitleStyle.backgroundOpacity,
                })}
              </span>
              <div className="flex flex-wrap items-center gap-1">
                {SUBTITLE_STYLE_PRESETS.map((preset) => {
                  const active =
                    subtitleStyle.fontFamily === preset.style.fontFamily &&
                    subtitleStyle.fontSizePx === preset.style.fontSizePx &&
                    subtitleStyle.backgroundOpacity === preset.style.backgroundOpacity &&
                    subtitleStyle.bold === preset.style.bold;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => setSubtitleStyle(preset.style)}
                      disabled={running}
                      data-testid={`embed-style-preset-${preset.id}`}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-colors cursor-pointer ${
                        active
                          ? 'bg-indigo-600/25 border-indigo-500 text-indigo-200'
                          : 'bg-surface border-line text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {t(`player.style.${PRESET_KEY_MAP[preset.id]}`)}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1 text-[11px] text-zinc-400">
                <div className="flex items-center justify-between">
                  <span>{t('player.style.fontFamily')}</span>
                  <label className="flex items-center gap-1 text-[11px] text-zinc-300 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={subtitleStyle.bold}
                      onChange={(e) => setSubtitleStyle({ bold: e.target.checked })}
                      disabled={running}
                      data-testid="embed-bold"
                      className="w-3.5 h-3.5 rounded border-zinc-700 text-indigo-600 bg-zinc-800"
                    />
                    <span>{t('player.style.bold')}</span>
                  </label>
                </div>
                <select
                  value={subtitleStyle.fontFamily}
                  onChange={(e) => setSubtitleStyle({ fontFamily: e.target.value as SubtitleFontFamily })}
                  disabled={running}
                  className={selectClass}
                  data-testid="embed-font-family"
                >
                  {SUBTITLE_FONTS.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.id}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1 text-[11px] text-zinc-400">
                <div className="flex items-center justify-between">
                  <span>{t('embed.fontSize')}</span>
                  <span className="font-mono text-indigo-300" dir="ltr">
                    {subtitleStyle.fontSizePx}px
                  </span>
                </div>
                <input
                  type="range"
                  min={MIN_SUBTITLE_FONT_SIZE}
                  max={MAX_SUBTITLE_FONT_SIZE}
                  step={1}
                  value={subtitleStyle.fontSizePx}
                  onChange={(e) => setSubtitleStyle({ fontSizePx: Number(e.target.value) })}
                  disabled={running}
                  aria-label={t('embed.fontSize')}
                  className="w-full accent-indigo-500 cursor-pointer mt-1"
                  data-testid="embed-font-size"
                />
              </div>

              <div className="flex items-center justify-between gap-2 text-[11px] text-zinc-400">
                <span>{t('player.style.textColor')}</span>
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] text-zinc-400" dir="ltr">
                    {subtitleStyle.textColor}
                  </span>
                  <input
                    type="color"
                    value={subtitleStyle.textColor}
                    onChange={(e) => setSubtitleStyle({ textColor: e.target.value })}
                    disabled={running}
                    aria-label={t('player.style.textColor')}
                    data-testid="embed-text-color"
                    className="w-6 h-6 rounded border border-line bg-transparent cursor-pointer p-0"
                  />
                </div>
              </div>

              <div className="space-y-1 text-[11px] text-zinc-400">
                <div className="flex items-center justify-between gap-2">
                  <span>{t('player.style.bgOpacity')}</span>
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-indigo-300" dir="ltr">
                      {subtitleStyle.backgroundOpacity}%
                    </span>
                    <input
                      type="color"
                      value={subtitleStyle.backgroundColor}
                      onChange={(e) => setSubtitleStyle({ backgroundColor: e.target.value })}
                      disabled={running}
                      aria-label={t('player.style.bgColor')}
                      data-testid="embed-bg-color"
                      className="w-5 h-5 rounded border border-line bg-transparent cursor-pointer p-0"
                    />
                  </div>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={subtitleStyle.backgroundOpacity}
                  onChange={(e) => setSubtitleStyle({ backgroundOpacity: Number(e.target.value) })}
                  disabled={running}
                  aria-label={t('player.style.bgOpacity')}
                  className="w-full accent-indigo-500 cursor-pointer"
                  data-testid="embed-bg-opacity"
                />
              </div>

              {subtitleStyle.backgroundOpacity < 100 && (
                <div className="space-y-1 text-[11px] text-zinc-400">
                  <div className="flex items-center justify-between gap-2">
                    <span>{t('player.style.outlineWidth')}</span>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-indigo-300" dir="ltr">
                        {subtitleStyle.outlineWidth.toFixed(1)}
                      </span>
                      <input
                        type="color"
                        value={subtitleStyle.outlineColor}
                        onChange={(e) => setSubtitleStyle({ outlineColor: e.target.value })}
                        disabled={running}
                        aria-label={t('player.style.outlineColor')}
                        data-testid="embed-outline-color"
                        className="w-5 h-5 rounded border border-line bg-transparent cursor-pointer p-0"
                      />
                    </div>
                  </div>
                  <input
                    type="range"
                    min={MIN_OUTLINE_WIDTH}
                    max={MAX_OUTLINE_WIDTH}
                    step={0.2}
                    value={subtitleStyle.outlineWidth}
                    onChange={(e) => setSubtitleStyle({ outlineWidth: Number(e.target.value) })}
                    disabled={running}
                    aria-label={t('player.style.outlineWidth')}
                    data-testid="embed-outline-width"
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              )}

              <div className="space-y-1 text-[11px] text-zinc-400">
                <div className="flex items-center justify-between">
                  <span>{t('player.style.bottomHeight')}</span>
                  <span className="font-mono text-indigo-300" dir="ltr">
                    {subtitleStyle.marginV}px
                  </span>
                </div>
                <input
                  type="range"
                  min={MIN_MARGIN_V}
                  max={MAX_MARGIN_V}
                  step={2}
                  value={subtitleStyle.marginV}
                  onChange={(e) => setSubtitleStyle({ marginV: Number(e.target.value) })}
                  disabled={running}
                  aria-label={t('player.style.bottomHeight')}
                  data-testid="embed-margin-v"
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div className="space-y-1 text-[11px] text-zinc-400">
                <div className="flex items-center justify-between gap-2">
                  <span>{t('player.style.highlightMode')}</span>
                  {subtitleStyle.highlightMode !== 'none' && (
                    <input
                      type="color"
                      value={subtitleStyle.highlightColor}
                      onChange={(e) => setSubtitleStyle({ highlightColor: e.target.value })}
                      disabled={running}
                      aria-label={t('player.style.highlightColor')}
                      data-testid="embed-highlight-color"
                      className="w-5 h-5 rounded border border-line bg-transparent cursor-pointer p-0"
                    />
                  )}
                </div>
                <select
                  value={subtitleStyle.highlightMode}
                  onChange={(e) => setSubtitleStyle({ highlightMode: e.target.value as SubtitleHighlightMode })}
                  disabled={running}
                  aria-label={t('player.style.highlightMode')}
                  data-testid="embed-highlight-mode"
                  className={selectClass}
                >
                  <option value="none">{t('player.style.highlightNone')}</option>
                  <option value="word">{t('player.style.highlightWord')}</option>
                  <option value="karaoke">{t('player.style.highlightKaraoke')}</option>
                </select>
              </div>
            </div>

            {(hasSpeakers || (hasTranslation && useTranslated)) && (
              <div className="flex flex-wrap items-center gap-4 pt-1 border-t border-line/60">
                {hasTranslation && useTranslated && (
                  <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={bilingual}
                      onChange={(e) => setBilingual(e.target.checked)}
                      disabled={running}
                      data-testid="embed-bilingual"
                      className="w-3.5 h-3.5 rounded border-zinc-700 text-indigo-600 bg-zinc-800"
                    />
                    <span>{t('embed.bilingual')}</span>
                  </label>
                )}
                {hasSpeakers && (
                  <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={colorSpeakers}
                      onChange={(e) => setColorSpeakers(e.target.checked)}
                      disabled={running}
                      data-testid="embed-color-speakers"
                      className="w-3.5 h-3.5 rounded border-zinc-700 text-indigo-600 bg-zinc-800"
                    />
                    <span>{t('embed.colorSpeakers')}</span>
                  </label>
                )}
              </div>
            )}
          </div>
        )}

        <p className="text-[11px] text-zinc-500 leading-relaxed">
          {isUpload ? t('embed.whereUpload') : mode === 'soft' ? t('embed.whereSoft') : t('embed.whereBurn')}
        </p>

        {running && (
          <div className="space-y-1.5" data-testid="embed-progress">
            <div className="flex justify-between text-[11px] text-zinc-400">
              <span className="flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin" />
                {status.mode === 'burn'
                  ? t('embed.burning', { encoder: status.encoder === 'h264_nvenc' ? 'NVIDIA' : 'CPU' })
                  : t('embed.muxing')}
              </span>
              <span>{percent.toFixed(0)}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-sunken overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all" style={{ width: `${percent}%` }} />
            </div>
            <p className="text-[11px] text-zinc-500">{t('embed.keepsRunning')}</p>
          </div>
        )}

        {finished && status.state === 'done' && (
          <div className="flex items-start gap-2 text-[11px] text-emerald-300" role="status" data-testid="embed-done">
            <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-px" />
            <span className="break-all">
              {status.next_to_video ? t('embed.savedTo') : t('embed.ready')}{' '}
              <bdi dir="ltr">{status.next_to_video ? status.path : status.file_name}</bdi>
            </span>
          </div>
        )}
        {finished && status.state === 'cancelled' && (
          <p className="text-[11px] text-zinc-400" role="status">{t('embed.cancelled')}</p>
        )}
        {(error || (finished && status.state === 'error')) && (
          <div className="flex items-start gap-2 text-[11px] text-rose-300" role="alert" data-testid="embed-error">
            <XCircle className="w-3.5 h-3.5 flex-shrink-0 mt-px" />
            <span dir="auto" className="break-all whitespace-pre-wrap max-h-28 overflow-auto">
              {error || status?.error}
            </span>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          {finished && status.state === 'done' && status.download_url && (
            <button
              onClick={() => onDownload(status.download_url!)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-zinc-200 border border-line hover:text-white"
            >
              <Download className="w-3.5 h-3.5" />
              {t('embed.downloadAgain')}
            </button>
          )}
          {running ? (
            <button
              onClick={() => void cancel()}
              className="px-3 py-1.5 rounded-lg text-xs text-rose-300 border border-rose-500/40 hover:bg-rose-500/10"
              data-testid="embed-cancel"
            >
              {t('embed.stop')}
            </button>
          ) : (
            <>
              <button
                onClick={onClose}
                className="px-3 py-1.5 rounded-lg text-xs text-zinc-300 border border-line hover:text-white"
              >
                {t('embed.close')}
              </button>
              <button
                onClick={() => void start()}
                disabled={starting}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white"
                data-testid="embed-start"
              >
                {starting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Film className="w-3.5 h-3.5" />}
                {finished ? t('embed.again') : t('embed.start')}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
