import React, { useEffect, useRef, useState } from 'react';
import { api } from '../../services/api';
import { confirmDialog } from '../../store/useDialogStore';
import {
  MAX_SUBTITLE_FONT_SIZE,
  MIN_SUBTITLE_FONT_SIZE,
  SUBTITLE_FONTS,
  fontSizePxToEmbedTier,
  hexToRgba,
  subtitleFontStack,
  useUiStore,
} from '../../store/useUiStore';
import type { EmbedMode, EmbedQuality, EmbedVideoOptions, EmbedVideoStatus, SubtitleFontFamily } from '../../types';
import { useT } from '../../i18n';
import { CheckCircle2, Download, Film, Flame, Layers, Loader2, Palette, X, XCircle } from 'lucide-react';

interface Props {
  jobId: string;
  /** Uploaded through the browser: the result is downloaded instead of saved next to the video */
  isUpload: boolean;
  hasTranslation: boolean;
  /** The toolbar's RTL punctuation choice, applied to the embedded text too */
  applyRtl: boolean;
  onClose: () => void;
  onDownload: (url: string) => void;
}

const selectClass =
  'w-full bg-sunken border border-line rounded-lg px-2 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500';

/** Make a copy of the video with the subtitles inside: a switchable track, or burned into the picture. */
export const EmbedVideoDialog: React.FC<Props> = ({ jobId, isUpload, hasTranslation, applyRtl, onClose, onDownload }) => {
  const t = useT();
  const subtitleStyle = useUiStore((s) => s.subtitleStyle);
  const setSubtitleStyle = useUiStore((s) => s.setSubtitleStyle);
  const [mode, setMode] = useState<EmbedMode>('soft');
  const [useTranslated, setUseTranslated] = useState(hasTranslation);
  const [setDefault, setSetDefault] = useState(true);
  const [quality, setQuality] = useState<EmbedQuality>('standard');
  const [status, setStatus] = useState<EmbedVideoStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const downloaded = useRef<string | null>(null);

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
              background_color: subtitleStyle.backgroundColor,
              background_opacity: subtitleStyle.backgroundOpacity,
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
        className="w-full max-w-lg bg-surface border border-line rounded-xl p-5 space-y-4 shadow-2xl"
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
            className="bg-sunken border border-line rounded-lg p-3 space-y-2.5"
          >
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="flex items-center gap-1.5 text-zinc-300 font-medium">
                <Palette className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                {t('embed.styleSummary', {
                  font: subtitleStyle.fontFamily,
                  size: subtitleStyle.fontSizePx,
                  opacity: subtitleStyle.backgroundOpacity,
                })}
              </span>
              <span
                className="px-2 py-0.5 rounded text-xs font-semibold border border-zinc-700/60"
                style={{
                  fontFamily: subtitleFontStack(subtitleStyle.fontFamily),
                  color: subtitleStyle.textColor,
                  backgroundColor:
                    subtitleStyle.backgroundOpacity > 0
                      ? hexToRgba(subtitleStyle.backgroundColor, subtitleStyle.backgroundOpacity)
                      : 'transparent',
                  textShadow:
                    subtitleStyle.backgroundOpacity < 30
                      ? '0 1px 2px rgba(0,0,0,0.95), 0 0 4px rgba(0,0,0,0.9)'
                      : undefined,
                }}
              >
                Aa אב
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1 text-[11px] text-zinc-400">
                <span>{t('player.style.fontFamily')}</span>
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
              </label>

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
            </div>
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
