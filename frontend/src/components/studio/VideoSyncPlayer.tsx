import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { desktopApi } from '../../utils/desktop';
import { useJobStore } from '../../store/useJobStore';
import { buildSubtitleTextShadow, hexToRgba, subtitleFontStack, useUiStore } from '../../store/useUiStore';
import { api } from '../../services/api';
import { isRtlLanguage } from '../../types';
import { findSegmentAt, formatClock, isTypingTarget } from '../../utils/time';
import { useT } from '../../i18n';
import { SubtitleStylePopover } from './SubtitleStylePopover';
import { Film, Play, Pause, RotateCcw, Maximize, Minimize, AlertTriangle, Keyboard, Palette } from 'lucide-react';

export const VideoSyncPlayer: React.FC = () => {
  const activeJob = useJobStore((s) => s.activeJob);
  const subtitleOverlayMode = useJobStore((s) => s.subtitleOverlayMode);
  const setSubtitleOverlayMode = useJobStore((s) => s.setSubtitleOverlayMode);
  const setVideoCurrentTime = useJobStore((s) => s.setVideoCurrentTime);
  const seekRequest = useJobStore((s) => s.seekRequest);
  const subtitleStyle = useUiStore((s) => s.subtitleStyle);
  const t = useT();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const styleButtonRef = useRef<HTMLButtonElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  // Component is keyed by job id (see App), so this initial state is per job
  const [duration, setDuration] = useState(activeJob?.video_duration || 0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [styleOpen, setStyleOpen] = useState(false);

  const onscreenPos = activeJob?.onscreen_position || 'top_center';

  const getOnscreenContainerClass = (pos: string) => {
    switch (pos) {
      case 'top_left':
        return 'absolute top-6 left-6 max-w-[48%] flex flex-col items-start gap-1.5 pointer-events-none z-10';
      case 'top_right':
        return 'absolute top-6 right-6 max-w-[48%] flex flex-col items-end gap-1.5 pointer-events-none z-10';
      case 'middle_left':
        return 'absolute top-1/2 -translate-y-1/2 left-6 max-w-[48%] flex flex-col items-start gap-1.5 pointer-events-none z-10';
      case 'middle_right':
        return 'absolute top-1/2 -translate-y-1/2 right-6 max-w-[48%] flex flex-col items-end gap-1.5 pointer-events-none z-10';
      case 'top_center':
      default:
        return 'absolute top-6 inset-x-4 flex flex-col items-center gap-1.5 pointer-events-none z-10';
    }
  };

  const getOnscreenTextClass = (pos: string) => {
    switch (pos) {
      case 'top_left':
      case 'middle_left':
        return 'text-left';
      case 'top_right':
      case 'middle_right':
        return 'text-right';
      case 'top_center':
      default:
        return 'text-center';
    }
  };

  const jobId = activeJob?.id;
  const videoSrc = useMemo(() => (jobId ? api.getVideoStreamUrl(jobId) : ''), [jobId]);

  const segments = useMemo(
    () => [...(activeJob?.segments || [])].sort((a, b) => a.start_time - b.start_time),
    [activeJob?.segments]
  );
  const { topOverlays, bottomOverlays } = useMemo(() => {
    const active = segments.filter((s) => currentTime >= s.start_time && currentTime <= s.end_time);
    const top: Array<{ key: string; orig?: string; trans?: string }> = [];
    const bottom: Array<{ key: string; orig?: string; trans?: string }> = [];

    const isBracketed = (line: string) => {
      const s = line.trim();
      return s.startsWith('[') && s.endsWith(']');
    };
    const isSpeakerTagOnly = (line: string) => {
      const s = line.trim();
      if (!isBracketed(s)) return false;
      const inner = s.slice(1, -1).trim();
      // A single name without spaces/dashes/digits (e.g. "[Travis]", "[טראוויס]") is a speaker label, not a screen title
      return inner.length > 0 && inner.length <= 18 && !/[\s—–\-0-9:,]/.test(inner);
    };
    const ensureBrackets = (text: string) => {
      const s = text.trim();
      if (!s) return s;
      return s.startsWith('[') && s.endsWith(']') ? s : `[${s}]`;
    };

    for (const seg of active) {
      const orig = (seg.original_text || '').trim();
      const trans = (seg.translated_text || '').trim();
      const checkText = trans || orig;
      const lines = checkText.split('\n').map((l) => l.trim()).filter(Boolean);

      if (seg.segment_type === 'ON_SCREEN' || (lines.length > 0 && lines.every((l) => isBracketed(l) && !isSpeakerTagOnly(l)))) {
        top.push({
          key: `top-${seg.index}`,
          orig: orig ? ensureBrackets(orig) : undefined,
          trans: trans ? ensureBrackets(trans) : undefined,
        });
      } else if (lines.length >= 2 && isBracketed(lines[0]) && !isSpeakerTagOnly(lines[0])) {
        const origLines = orig.split('\n').map((l) => l.trim()).filter(Boolean);
        const transLines = trans.split('\n').map((l) => l.trim()).filter(Boolean);
        const origFirstIsTitle = origLines.length >= 2 && isBracketed(origLines[0]) && !isSpeakerTagOnly(origLines[0]);
        const transFirstIsTitle = transLines.length >= 2 && isBracketed(transLines[0]) && !isSpeakerTagOnly(transLines[0]);
        if (origFirstIsTitle && (transFirstIsTitle || !trans)) {
          const topOrig = origLines[0];
          const botOrig = origLines.slice(1).join('\n');
          const topTrans = transFirstIsTitle ? transLines[0] : undefined;
          const botTrans = transFirstIsTitle ? transLines.slice(1).join('\n') : trans || undefined;
          top.push({ key: `top-${seg.index}`, orig: topOrig, trans: topTrans });
          bottom.push({ key: `bot-${seg.index}`, orig: botOrig, trans: botTrans });
        } else {
          bottom.push({ key: `bot-${seg.index}`, orig: orig || undefined, trans: trans || undefined });
        }
      } else {
        bottom.push({ key: `bot-${seg.index}`, orig: orig || undefined, trans: trans || undefined });
      }
    }
    return { topOverlays: top, bottomOverlays: bottom };
  }, [segments, currentTime]);

  // Smooth time tracking while playing (timeupdate only fires ~4x per second).
  // The store is updated less often — it drives the table highlight, not the overlay.
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    let lastStoreUpdate = 0;
    const tick = (now: number) => {
      const v = videoRef.current;
      if (v) {
        setCurrentTime(v.currentTime);
        if (now - lastStoreUpdate > 200) {
          lastStoreUpdate = now;
          setVideoCurrentTime(v.currentTime);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, setVideoCurrentTime]);

  // Seek only when explicitly requested (row click, shortcuts) — not on every edit.
  useEffect(() => {
    const v = videoRef.current;
    if (!seekRequest || !v) return;
    v.currentTime = seekRequest.time;
    setCurrentTime(seekRequest.time);
    setVideoCurrentTime(seekRequest.time);
    if (seekRequest.play) v.play().catch(() => {});
  }, [seekRequest, setVideoCurrentTime]);

  const syncTime = () => {
    const v = videoRef.current;
    if (!v) return;
    setCurrentTime(v.currentTime);
    setVideoCurrentTime(v.currentTime);
  };

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  }, []);

  const seekBy = useCallback((delta: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(v.duration || Infinity, v.currentTime + delta));
    setCurrentTime(v.currentTime);
    setVideoCurrentTime(v.currentTime);
  }, [setVideoCurrentTime]);

  const toggleFullscreen = useCallback(() => {
    // Fullscreen the container, not the <video>, so the subtitle overlay stays visible
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else containerRef.current?.requestFullscreen().catch(() => {});
  }, []);

  useEffect(() => {
    const onChange = () => {
      const full = !!document.fullscreenElement;
      setIsFullscreen(full);
      // In the desktop window the player only fills the window; make the window itself follow
      desktopApi()?.set_fullscreen?.(full)?.catch(() => {});
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  // Keyboard shortcuts (ignored while typing in the editor)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      const v = videoRef.current;
      if (!v) return;
      const state = useJobStore.getState();
      if (state.insertDraft) return; // the "add a line" dialog is open
      if (document.querySelector('[aria-modal="true"]')) return; // a dialog has the keyboard
      const job = state.activeJob;

      // Target for [ and ]: the selected subtitle, else the one under the playhead
      const target = () => {
        if (!job) return undefined;
        if (state.selectedSegmentIndex !== null) {
          return job.segments.find((s) => s.index === state.selectedSegmentIndex);
        }
        const sorted = [...job.segments].sort((a, b) => a.start_time - b.start_time);
        const i = findSegmentAt(sorted, v.currentTime);
        return i >= 0 ? sorted[i] : undefined;
      };

      switch (e.key) {
        case ' ':
        case 'k':
          e.preventDefault();
          // A focused button would also "click" on Space keyup (e.g. Retranslate) — drop focus
          if (e.target instanceof HTMLButtonElement) e.target.blur();
          togglePlay();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          seekBy(e.shiftKey ? -0.1 : -2);
          break;
        case 'ArrowRight':
          e.preventDefault();
          seekBy(e.shiftKey ? 0.1 : 2);
          break;
        case 'f':
          toggleFullscreen();
          break;
        case 'ArrowUp':
        case 'ArrowDown': {
          // Previous / next subtitle line: select it and jump to its start
          if (!job || job.segments.length === 0) break;
          e.preventDefault();
          const sorted = [...job.segments].sort((a, b) => a.start_time - b.start_time || a.index - b.index);
          const current = target();
          let pos = current ? sorted.indexOf(current) : -1;
          if (pos < 0) {
            // Between lines: "next" is the first line starting after the playhead, "previous" the one before it
            const found = sorted.findIndex((s) => s.start_time > v.currentTime);
            const after = found < 0 ? sorted.length : found;
            pos = e.key === 'ArrowDown' ? after - 1 : after;
          }
          const next = sorted[Math.max(0, Math.min(sorted.length - 1, pos + (e.key === 'ArrowDown' ? 1 : -1)))];
          state.setSelectedSegmentIndex(next.index);
          v.currentTime = next.start_time;
          break;
        }
        case 's':
        case 'S': {
          // Split the line under the playhead (or the selected one) at the playhead
          const seg = target();
          const tNow = v.currentTime;
          if (seg?.id && tNow > seg.start_time + 0.15 && tNow < seg.end_time - 0.15) {
            e.preventDefault();
            v.pause();
            void state.splitSegment(seg.id, Number(tNow.toFixed(3)));
          }
          break;
        }
        case 'n':
        case 'N':
          e.preventDefault();
          v.pause();
          state.openInsert({ time: v.currentTime });
          break;
        case '[': {
          const seg = target();
          if (seg?.id && v.currentTime < seg.end_time) {
            state.updateSegment(seg.id, { start_time: Number(v.currentTime.toFixed(3)) });
          }
          break;
        }
        case ']': {
          const seg = target();
          if (seg?.id && v.currentTime > seg.start_time) {
            state.updateSegment(seg.id, { end_time: Number(v.currentTime.toFixed(3)) });
          }
          break;
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [togglePlay, seekBy, toggleFullscreen]);

  const onVideoError = () => {
    const err = videoRef.current?.error;
    if (!err) return;
    setVideoError(
      err.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED || err.code === MediaError.MEDIA_ERR_DECODE
        ? t('player.codecError')
        : t('player.loadError', { message: err.message || `error ${err.code}` })
    );
  };

  const containsRtl = (text: string) => /[\u0590-\u05FF\u0600-\u06FF]/.test(text);
  const isRtlSource = isRtlLanguage(activeJob?.detected_language || activeJob?.source_language);
  const isRtlTarget = isRtlLanguage(activeJob?.target_language);
  const showOriginal = subtitleOverlayMode === 'dual' || subtitleOverlayMode === 'original';
  const showTranslated = subtitleOverlayMode === 'dual' || subtitleOverlayMode === 'translated';

  const overlayFontFamily = subtitleFontStack(subtitleStyle.fontFamily);
  const overlayFontSize = isFullscreen
    ? Math.round(subtitleStyle.fontSizePx * 1.45)
    : subtitleStyle.fontSizePx;
  const hasOverlayBg = subtitleStyle.backgroundOpacity > 0;
  const overlayBg = hasOverlayBg
    ? hexToRgba(subtitleStyle.backgroundColor, subtitleStyle.backgroundOpacity)
    : 'transparent';
  const overlayTextShadow = buildSubtitleTextShadow(subtitleStyle);
  const overlayFontWeight = subtitleStyle.bold ? 700 : 500;
  const overlayBoxStyle: React.CSSProperties = {
    backgroundColor: overlayBg,
    fontFamily: overlayFontFamily,
    fontSize: `${overlayFontSize}px`,
    fontWeight: overlayFontWeight,
    lineHeight: 1.4,
  };

  const modeButton = (mode: 'dual' | 'translated' | 'original', label: string, title: string) => (
    <button
      onClick={() => setSubtitleOverlayMode(mode)}
      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
        subtitleOverlayMode === mode ? 'bg-indigo-600 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
      }`}
      title={title}
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-col h-full bg-surface border border-line rounded-xl overflow-hidden shadow-xl">
      {/* Top Controls Bar */}
      <div className="relative z-20 px-4 py-2.5 bg-sunken border-b border-line flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-zinc-200">{t('player.title')}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <button
              ref={styleButtonRef}
              type="button"
              onClick={() => setStyleOpen((o) => !o)}
              data-testid="open-subtitle-style"
              title={t('player.style.buttonTitle')}
              aria-expanded={styleOpen}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer ${
                styleOpen
                  ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                  : 'bg-surface border-line text-zinc-300 hover:border-zinc-500'
              }`}
            >
              <Palette className="w-3.5 h-3.5 text-indigo-400" />
              <span>{t('player.style.button')}</span>
            </button>
            <SubtitleStylePopover
              open={styleOpen}
              onClose={() => setStyleOpen(false)}
              anchorRef={styleButtonRef}
            />
          </div>
          <div className="flex bg-surface p-0.5 rounded-lg border border-line text-xs">
            {modeButton('dual', t('player.dual'), t('player.dualTitle'))}
            {modeButton('translated', t('player.translated'), t('player.translatedTitle'))}
            {modeButton('original', t('player.original'), t('player.originalTitle'))}
          </div>
        </div>
      </div>

      {/* Video + overlay (this whole box goes fullscreen) */}
      <div
        ref={containerRef}
        className="relative flex-1 bg-black flex items-center justify-center overflow-hidden min-h-[300px]"
      >
        {videoSrc ? (
          <video
            ref={videoRef}
            src={videoSrc}
            onTimeUpdate={() => !isPlaying && syncTime()}
            onSeeked={syncTime}
            onLoadedMetadata={() => setDuration(videoRef.current?.duration || activeJob?.video_duration || 0)}
            onPlay={() => setIsPlaying(true)}
            onPause={() => {
              setIsPlaying(false);
              syncTime();
            }}
            onError={onVideoError}
            controls={false}
            playsInline
            className="w-full h-full object-contain cursor-pointer"
            onClick={togglePlay}
          />
        ) : (
          <div className="text-zinc-600 text-xs flex flex-col items-center gap-2">
            <Film className="w-8 h-8 opacity-40" />
            <span>{t('player.noVideo')}</span>
          </div>
        )}

        {videoError && (
          <div className="absolute inset-x-4 top-4 p-3 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{videoError}</span>
          </div>
        )}

        {topOverlays.length > 0 && (
          <div className={getOnscreenContainerClass(onscreenPos)} data-testid="player-top-overlay">
            {topOverlays.map((item) => {
              const displayOrig = showOriginal ? item.orig : (!item.trans ? item.orig : undefined);
              const displayTrans = showTranslated ? item.trans : undefined;
              if (!displayOrig && !displayTrans) return null;
              return (
                <div
                  key={item.key}
                  data-testid="player-top-subtitle-box"
                  style={overlayBoxStyle}
                  className={`max-w-[90%] px-4 py-1.5 rounded-lg space-y-1 ${
                    hasOverlayBg ? 'border border-cyan-500/25' : ''
                  } ${getOnscreenTextClass(onscreenPos)}`}
                >
                  {displayOrig && (
                    <p
                      dir={isRtlSource || containsRtl(displayOrig) ? 'rtl' : 'ltr'}
                      className={`font-medium whitespace-pre-line italic ${
                        displayTrans ? 'text-cyan-200' : ''
                      }`}
                      style={{
                        color: displayTrans ? undefined : subtitleStyle.textColor,
                        fontSize: displayTrans ? '0.88em' : undefined,
                        textShadow: overlayTextShadow,
                      }}
                    >
                      {displayOrig}
                    </p>
                  )}
                  {displayTrans && (
                    <p
                      dir={isRtlTarget ? 'rtl' : 'ltr'}
                      className="font-medium whitespace-pre-line"
                      style={{ color: subtitleStyle.textColor, textShadow: overlayTextShadow }}
                    >
                      {displayTrans}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {bottomOverlays.length > 0 && (
          <div
            className="absolute inset-x-4 flex flex-col items-center gap-1.5 pointer-events-none z-10"
            style={{ bottom: `${subtitleStyle.marginV}px` }}
            data-testid="player-bottom-overlay"
          >
            {bottomOverlays.map((item) => {
              const displayOrig = showOriginal ? item.orig : (!item.trans ? item.orig : undefined);
              const displayTrans = showTranslated ? item.trans : undefined;
              if (!displayOrig && !displayTrans) return null;
              return (
                <div
                  key={item.key}
                  data-testid="player-bottom-subtitle-box"
                  style={overlayBoxStyle}
                  className="max-w-[90%] px-4 py-2 rounded-lg text-center space-y-1"
                >
                  {displayOrig && (
                    <p
                      dir={isRtlSource || containsRtl(displayOrig) ? 'rtl' : 'ltr'}
                      className="font-medium whitespace-pre-line"
                      style={{
                        color: displayTrans ? 'rgba(255, 255, 255, 0.88)' : subtitleStyle.textColor,
                        fontSize: displayTrans ? '0.88em' : undefined,
                        textShadow: overlayTextShadow,
                      }}
                    >
                      {displayOrig}
                    </p>
                  )}
                  {displayTrans && (
                    <p
                      dir={isRtlTarget ? 'rtl' : 'ltr'}
                      className="font-medium whitespace-pre-line"
                      style={{ color: subtitleStyle.textColor, textShadow: overlayTextShadow }}
                    >
                      {displayTrans}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Transport: play, seek bar, time, fullscreen. Timelines stay left-to-right in RTL too. */}
      <div className="px-4 py-2.5 bg-sunken border-t border-line flex items-center gap-3" dir="ltr">
        <button
          onClick={togglePlay}
          className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors cursor-pointer"
          title={isPlaying ? t('player.pause') : t('player.play')}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-zinc-200" />}
        </button>
        <button
          onClick={() => seekBy(-3)}
          className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors cursor-pointer text-xs flex items-center gap-1 px-2"
          title={t('player.back3')}
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="text-[10px] font-mono">-3s</span>
        </button>

        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.1}
          value={Math.min(currentTime, duration || 0)}
          onChange={(e) => {
            const v = videoRef.current;
            if (!v) return;
            v.currentTime = Number(e.target.value);
            setCurrentTime(v.currentTime);
          }}
          className="flex-1 accent-indigo-500 cursor-pointer"
          aria-label={t('player.seek')}
        />
        <span className="text-[11px] font-mono-time text-zinc-400 whitespace-nowrap">
          {formatClock(currentTime)} / {formatClock(duration)}
        </span>

        <button
          onClick={() => useUiStore.getState().setShortcutsOpen(true)}
          className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors cursor-pointer"
          title={`${t('player.shortcuts')} (?)`}
          aria-label={t('player.shortcuts')}
          data-testid="open-shortcuts"
        >
          <Keyboard className="w-4 h-4" />
        </button>
        <button
          onClick={toggleFullscreen}
          className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors cursor-pointer"
          title={t('player.fullscreen')}
        >
          {isFullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
};
