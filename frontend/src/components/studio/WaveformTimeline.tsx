import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AudioWaveform, ChevronDown, ChevronUp, Loader2, RefreshCw, ZoomIn, ZoomOut } from 'lucide-react';
import { api } from '../../services/api';
import { useJobStore } from '../../store/useJobStore';
import { useT } from '../../i18n';
import { isActiveStatus } from '../../types';
import type { SubtitleSegment, WaveformData } from '../../types';
import {
  ZOOM_LEVELS,
  applySegmentDrag,
  clampTime,
  computeSameLaneBounds,
  computeTimeWindow,
  getLaneLayout,
  hitTestWaveformSegment,
  roundMs,
  type DragMode,
} from './waveformMath';

const WAVEFORM_VISIBLE_KEY = 'subbuild.waveformVisible';

function formatRulerTime(sec: number): string {
  const whole = Math.max(0, Math.floor(sec));
  const m = Math.floor(whole / 60);
  const s = whole % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatPreciseTime(sec: number): string {
  const safe = Math.max(0, sec);
  const totalHundredths = Math.round(safe * 100);
  const totalSeconds = Math.floor(totalHundredths / 100);
  const hundredths = totalHundredths % 100;
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const frac = hundredths % 10 === 0 ? String(hundredths / 10) : String(hundredths).padStart(2, '0');
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${frac}`;
  }
  return `${m}:${String(s).padStart(2, '0')}.${frac}`;
}

function formatSignedDelta(deltaSec: number): string {
  const absVal = Math.abs(deltaSec);
  if (absVal < 0.005) return '+0.0s';
  const sign = deltaSec >= 0 ? '+' : '-';
  const isExactTenth = Math.abs(Math.round(absVal * 10) / 10 - absVal) < 0.005;
  const roundedHundredths = Math.round(absVal * 100) / 100;
  const formatted = isExactTenth ? absVal.toFixed(1) : roundedHundredths.toFixed(2);
  return `${sign}${formatted}s`;
}

interface DragState {
  kind: 'segment' | 'seek';
  segId?: number;
  segIndex?: number;
  mode?: DragMode;
  startClientX: number;
  startX: number;
  currentX: number;
  origStart: number;
  origEnd: number;
  previewStart: number;
  previewEnd: number;
  minBound: number;
  maxBound: number;
  viewStart: number;
  viewEnd: number;
  width: number;
  moved: boolean;
}

interface DragOverlayInfo {
  kind: 'seek' | 'segment';
  mode?: DragMode;
  segIndex?: number;
  x: number;
  startX: number;
  width: number;
  targetTime?: number;
  deltaSec: number;
  startTime?: number;
  endTime?: number;
}

interface WaveformTimelineProps {
  jobId: string;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
}

export const WaveformTimeline: React.FC<WaveformTimelineProps> = ({ jobId, currentTime, duration, onSeek }) => {
  const activeJob = useJobStore((s) => s.activeJob);
  const selectedSegmentIndex = useJobStore((s) => s.selectedSegmentIndex);
  const setSelectedSegmentIndex = useJobStore((s) => s.setSelectedSegmentIndex);
  const previewSegmentTiming = useJobStore((s) => s.previewSegmentTiming);
  const updateSegment = useJobStore((s) => s.updateSegment);
  const subtitleOverlayMode = useJobStore((s) => s.subtitleOverlayMode);
  const t = useT();

  const [visible, setVisible] = useState<boolean>(() => {
    try {
      return localStorage.getItem(WAVEFORM_VISIBLE_KEY) !== 'false';
    } catch {
      return true;
    }
  });
  const [windowSec, setWindowSec] = useState<number>(15);
  const [retryNonce, setRetryNonce] = useState(0);
  const [cursorStyle, setCursorStyle] = useState<string>('pointer');
  const [waveformState, setWaveformState] = useState<{
    jobId: string;
    data: WaveformData | null;
    error: boolean;
  }>({ jobId: '', data: null, error: false });

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [canvasWidth, setCanvasWidth] = useState<number>(640);
  const [dragOverlay, setDragOverlay] = useState<DragOverlayInfo | null>(null);

  const jobStatus = activeJob?.status;
  const isRunning = isActiveStatus(jobStatus);
  const audioReady =
    !!jobStatus &&
    jobStatus !== 'IDLE' &&
    jobStatus !== 'QUEUED' &&
    jobStatus !== 'PROBING' &&
    jobStatus !== 'EXTRACTING_AUDIO';

  useEffect(() => {
    if (!visible || !jobId || !audioReady) return;
    let cancelled = false;
    api
      .getWaveform(jobId)
      .then((data) => {
        if (!cancelled) setWaveformState({ jobId, data, error: false });
      })
      .catch(() => {
        if (!cancelled) setWaveformState({ jobId, data: null, error: true });
      });
    return () => {
      cancelled = true;
    };
  }, [visible, jobId, audioReady, retryNonce]);

  const waveform = waveformState.jobId === jobId ? waveformState.data : null;
  const waveformError = waveformState.jobId === jobId ? waveformState.error : false;
  const loadingWaveform = visible && audioReady && waveformState.jobId !== jobId;

  const segments = useMemo(
    () =>
      [...(activeJob?.segments || [])].sort(
        (a, b) => a.start_time - b.start_time || a.end_time - b.end_time || a.index - b.index
      ),
    [activeJob?.segments]
  );

  const allowOnscreenOverlap = activeJob?.allow_onscreen_overlap !== false;
  const hasOnscreen = useMemo(
    () => allowOnscreenOverlap && segments.some((s) => s.segment_type === 'ON_SCREEN'),
    [allowOnscreenOverlap, segments]
  );

  const effectiveDuration = useMemo(() => {
    const maxSeg = segments.reduce((m, s) => Math.max(m, s.end_time), 0);
    return Math.max(duration || 0, waveform?.duration || 0, maxSeg, 10);
  }, [duration, waveform?.duration, segments]);

  const { viewStart, viewEnd } = useMemo(
    () => computeTimeWindow(currentTime, windowSec, effectiveDuration),
    [currentTime, windowSec, effectiveDuration]
  );

  const canvasHeight = hasOnscreen ? 108 : 88;
  const layout = useMemo(() => getLaneLayout(canvasHeight, hasOnscreen), [canvasHeight, hasOnscreen]);

  const toggleVisible = () => {
    const next = !visible;
    setVisible(next);
    try {
      localStorage.setItem(WAVEFORM_VISIBLE_KEY, String(next));
    } catch {
      /* ignore */
    }
  };

  const stepZoom = useCallback((direction: -1 | 1) => {
    setWindowSec((prev) => {
      const idx = ZOOM_LEVELS.indexOf(prev as (typeof ZOOM_LEVELS)[number]);
      const baseIdx = idx >= 0 ? idx : 2;
      const nextIdx = clampTime(baseIdx + direction, 0, ZOOM_LEVELS.length - 1);
      return ZOOM_LEVELS[nextIdx];
    });
  }, []);

  // Measure container width so canvas stays sharp at devicePixelRatio
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const updateWidth = () => {
      const w = el.clientWidth;
      if (w > 0) setCanvasWidth(w);
    };
    updateWidth();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(updateWidth);
      ro.observe(el);
      return () => ro.disconnect();
    }
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, [visible]);

  // Mouse wheel zooms the timeline window without scrolling the outer page
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !visible) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      stepZoom(e.deltaY < 0 ? -1 : 1);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [visible, stepZoom]);

  // Draw waveform peaks, dual-lane subtitle blocks, ruler, and playhead onto the 2D canvas
  useEffect(() => {
    if (!visible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const w = Math.max(200, canvasWidth);
    const h = canvasHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#09090d';
    ctx.fillRect(0, 0, w, h);

    // Ruler strip
    ctx.fillStyle = '#111118';
    ctx.fillRect(0, 0, w, layout.rulerHeight);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(0, layout.rulerHeight);
    ctx.lineTo(w, layout.rulerHeight);
    ctx.stroke();

    const span = Math.max(0.1, viewEnd - viewStart);
    const toX = (tSec: number) => ((tSec - viewStart) / span) * w;

    // Time ticks
    const tickStep = span <= 10 ? 1 : span <= 25 ? 2 : 5;
    const firstTick = Math.ceil(viewStart / tickStep) * tickStep;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    for (let ts = firstTick; ts <= viewEnd; ts += tickStep) {
      const tx = toX(ts);
      ctx.strokeStyle = 'rgba(161, 161, 170, 0.25)';
      ctx.beginPath();
      ctx.moveTo(tx, layout.rulerHeight - 5);
      ctx.lineTo(tx, layout.rulerHeight);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.beginPath();
      ctx.moveTo(tx, layout.rulerHeight);
      ctx.lineTo(tx, h);
      ctx.stroke();

      ctx.fillStyle = '#a1a1aa';
      ctx.fillText(formatRulerTime(ts), tx + 3, layout.rulerHeight / 2);
    }

    // Upper ON_SCREEN lane divider
    if (hasOnscreen) {
      ctx.fillStyle = 'rgba(8, 51, 68, 0.18)';
      ctx.fillRect(0, layout.onscreenTop - 2, w, layout.onscreenHeight + 4);
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.18)';
      ctx.beginPath();
      ctx.moveTo(0, layout.onscreenTop + layout.onscreenHeight + 2);
      ctx.lineTo(w, layout.onscreenTop + layout.onscreenHeight + 2);
      ctx.stroke();
    }

    // Waveform amplitude bars inside the main lane
    const midY = layout.mainTop + layout.mainHeight / 2;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.16)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    if (waveform && waveform.peaks.length > 0) {
      const pps = waveform.peaks_per_second || 50;
      const barStep = 2;
      const maxAmp = layout.mainHeight * 0.46;
      ctx.fillStyle = 'rgba(129, 140, 248, 0.42)';
      for (let px = 0; px < w; px += barStep) {
        const t0 = viewStart + (px / w) * span;
        const t1 = viewStart + ((px + barStep) / w) * span;
        const i0 = Math.max(0, Math.floor(t0 * pps));
        const i1 = Math.min(waveform.peaks.length, Math.max(i0 + 1, Math.ceil(t1 * pps)));
        let peak = 0;
        for (let k = i0; k < i1; k++) {
          if (waveform.peaks[k] > peak) peak = waveform.peaks[k];
        }
        const barH = Math.max(1.5, peak * maxAmp * 2);
        ctx.fillRect(px, midY - barH / 2, Math.max(1, barStep - 0.5), barH);
      }
    }

    // Subtitle segment blocks (both lanes)
    const drawBlock = (seg: SubtitleSegment, top: number, height: number, isOnscreen: boolean) => {
      if (seg.end_time < viewStart || seg.start_time > viewEnd) return;
      const rawX1 = toX(seg.start_time);
      const rawX2 = toX(seg.end_time);
      const x1 = Math.max(0, rawX1);
      const x2 = Math.min(w, rawX2);
      const bw = Math.max(2, x2 - x1);

      const isSelected = selectedSegmentIndex === seg.index;
      const isAtPlayhead = currentTime >= seg.start_time && currentTime <= seg.end_time;

      let fill = 'rgba(79, 70, 229, 0.36)';
      let border = 'rgba(129, 140, 248, 0.85)';
      if (isOnscreen) {
        fill = 'rgba(8, 145, 178, 0.34)';
        border = 'rgba(34, 211, 238, 0.9)';
      } else if (seg.segment_type === 'COMMS') {
        fill = 'rgba(217, 119, 6, 0.32)';
        border = 'rgba(251, 191, 36, 0.85)';
      } else if (seg.segment_type === 'FOREIGN') {
        fill = 'rgba(5, 150, 105, 0.32)';
        border = 'rgba(52, 211, 153, 0.85)';
      }
      if (isAtPlayhead || isSelected) {
        fill = isOnscreen ? 'rgba(6, 182, 212, 0.52)' : 'rgba(99, 102, 241, 0.55)';
        border = '#f8fafc';
      }

      ctx.fillStyle = fill;
      ctx.fillRect(x1, top, bw, height);
      ctx.strokeStyle = border;
      ctx.lineWidth = isSelected ? 1.5 : 1;
      ctx.strokeRect(x1 + 0.5, top + 0.5, Math.max(1, bw - 1), Math.max(1, height - 1));

      // Edge grab handles when block is wide enough
      if (rawX2 - rawX1 >= 12) {
        ctx.fillStyle = border;
        if (rawX1 >= -2) ctx.fillRect(x1, top, 3, height);
        if (rawX2 <= w + 2) ctx.fillRect(x2 - 3, top, 3, height);
      }

      // Label inside block
      if (bw >= 24) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1 + 4, top, bw - 8, height);
        ctx.clip();
        const textSource =
          subtitleOverlayMode === 'original'
            ? seg.original_text
            : seg.translated_text || seg.original_text;
        const preview = (textSource || '').replace(/\s+/g, ' ').trim();
        ctx.fillStyle = '#f4f4f5';
        ctx.font = '10px system-ui, -apple-system, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(`#${seg.index} ${preview}`, x1 + 5, top + height / 2);
        ctx.restore();
      }
    };

    for (const seg of segments) {
      const isTopLane = hasOnscreen && seg.segment_type === 'ON_SCREEN';
      if (isTopLane) {
        drawBlock(seg, layout.onscreenTop, layout.onscreenHeight, true);
      } else {
        drawBlock(seg, layout.mainTop + 2, layout.mainHeight - 4, false);
      }
    }

    // Seek drag range band and preview line
    if (dragOverlay?.kind === 'seek' && dragOverlay.targetTime != null) {
      const originX = clampTime(toX(currentTime), 0, w);
      const targetX = clampTime(toX(dragOverlay.targetTime), 0, w);
      const bandLeft = Math.min(originX, targetX);
      const bandWidth = Math.abs(targetX - originX);
      if (bandWidth >= 1) {
        ctx.fillStyle = 'rgba(99, 102, 241, 0.18)';
        ctx.fillRect(bandLeft, layout.rulerHeight, bandWidth, h - layout.rulerHeight);
      }

      ctx.save();
      ctx.strokeStyle = '#818cf8';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(targetX, 0);
      ctx.lineTo(targetX, h);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = '#818cf8';
      ctx.beginPath();
      ctx.moveTo(targetX - 4, 0);
      ctx.lineTo(targetX + 4, 0);
      ctx.lineTo(targetX, 6);
      ctx.closePath();
      ctx.fill();
    }

    // Playhead cursor
    if (currentTime >= viewStart && currentTime <= viewEnd) {
      const px = toX(currentTime);
      ctx.strokeStyle = '#fb7185';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();

      ctx.fillStyle = '#fb7185';
      ctx.beginPath();
      ctx.moveTo(px - 5, 0);
      ctx.lineTo(px + 5, 0);
      ctx.lineTo(px, 7);
      ctx.closePath();
      ctx.fill();
    }

    ctx.restore();
  }, [
    visible,
    canvasWidth,
    canvasHeight,
    layout,
    hasOnscreen,
    viewStart,
    viewEnd,
    waveform,
    segments,
    currentTime,
    selectedSegmentIndex,
    subtitleOverlayMode,
    dragOverlay,
  ]);

  const getCanvasCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const rect = canvas?.getBoundingClientRect();
    const width = rect && rect.width > 0 ? rect.width : canvasWidth || 640;
    const x = rect ? e.clientX - rect.left : e.clientX;
    const y = rect ? e.clientY - rect.top : e.clientY;
    return { x, y, width };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const { x, y, width } = getCanvasCoords(e);
    const hit = hitTestWaveformSegment(x, y, width, layout, hasOnscreen, viewStart, viewEnd, segments);

    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch {
      /* ignore in jsdom */
    }

    if (hit && hit.segment.id != null && !isRunning) {
      const { minBound, maxBound } = computeSameLaneBounds(
        hit.segment,
        segments,
        effectiveDuration,
        allowOnscreenOverlap
      );
      dragRef.current = {
        kind: 'segment',
        segId: hit.segment.id,
        segIndex: hit.segment.index,
        mode: hit.mode,
        startClientX: e.clientX,
        startX: x,
        currentX: x,
        origStart: hit.segment.start_time,
        origEnd: hit.segment.end_time,
        previewStart: hit.segment.start_time,
        previewEnd: hit.segment.end_time,
        minBound,
        maxBound,
        viewStart,
        viewEnd,
        width,
        moved: false,
      };
      return;
    }

    const initialSeekTime = clampTime(
      viewStart + (x / Math.max(1, width)) * (viewEnd - viewStart),
      0,
      effectiveDuration
    );
    dragRef.current = {
      kind: 'seek',
      startClientX: e.clientX,
      startX: x,
      currentX: x,
      origStart: initialSeekTime,
      origEnd: initialSeekTime,
      previewStart: initialSeekTime,
      previewEnd: initialSeekTime,
      minBound: 0,
      maxBound: effectiveDuration,
      viewStart,
      viewEnd,
      width,
      moved: false,
    };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      const { x, y, width } = getCanvasCoords(e);
      const hit = !isRunning
        ? hitTestWaveformSegment(x, y, width, layout, hasOnscreen, viewStart, viewEnd, segments)
        : null;
      if (!hit || hit.segment.id == null) setCursorStyle('pointer');
      else if (hit.mode === 'move') setCursorStyle('grab');
      else setCursorStyle('ew-resize');
      return;
    }

    const dx = e.clientX - drag.startClientX;
    if (!drag.moved && Math.abs(dx) < 3) return;
    drag.moved = true;

    const { x } = getCanvasCoords(e);
    const clampedX = clampTime(x, 0, drag.width);
    drag.currentX = clampedX;

    if (drag.kind === 'seek') {
      const targetTime = roundMs(
        clampTime(
          drag.viewStart + (clampedX / Math.max(1, drag.width)) * (drag.viewEnd - drag.viewStart),
          0,
          effectiveDuration
        )
      );
      drag.previewStart = targetTime;
      const deltaSec = roundMs(targetTime - currentTime);
      setDragOverlay({
        kind: 'seek',
        x: clampedX,
        startX: clampTime(drag.startX, 0, drag.width),
        width: drag.width,
        targetTime,
        deltaSec,
      });
      return;
    }

    if (drag.kind !== 'segment' || drag.segId == null || !drag.mode) return;

    const deltaSec = (dx / Math.max(1, drag.width)) * (drag.viewEnd - drag.viewStart);
    const next = applySegmentDrag(
      drag.mode,
      drag.origStart,
      drag.origEnd,
      deltaSec,
      drag.minBound,
      drag.maxBound
    );
    drag.previewStart = next.start_time;
    drag.previewEnd = next.end_time;
    // Live visual preview only during pointermove; persisted as a single Undo/Redo step on pointerup
    previewSegmentTiming(drag.segId, next);

    const appliedDeltaSec =
      drag.mode === 'resize-end'
        ? roundMs(next.end_time - drag.origEnd)
        : roundMs(next.start_time - drag.origStart);
    setDragOverlay({
      kind: 'segment',
      mode: drag.mode,
      segIndex: drag.segIndex,
      x: clampedX,
      startX: clampTime(drag.startX, 0, drag.width),
      width: drag.width,
      deltaSec: appliedDeltaSec,
      startTime: next.start_time,
      endTime: next.end_time,
    });
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    setDragOverlay(null);
    try {
      if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
        e.currentTarget.releasePointerCapture?.(e.pointerId);
      }
    } catch {
      /* ignore */
    }
    if (!drag) return;

    const { x, width } = getCanvasCoords(e);
    if (drag.kind === 'seek') {
      const targetTime = clampTime(
        drag.viewStart + (x / Math.max(1, width)) * (drag.viewEnd - drag.viewStart),
        0,
        effectiveDuration
      );
      onSeek(roundMs(targetTime));
      return;
    }

    if (drag.segIndex != null) setSelectedSegmentIndex(drag.segIndex);

    if (!drag.moved) {
      const targetTime = clampTime(
        drag.viewStart + (x / Math.max(1, width)) * (drag.viewEnd - drag.viewStart),
        0,
        effectiveDuration
      );
      onSeek(roundMs(targetTime));
      return;
    }

    if (
      drag.segId != null &&
      (Math.abs(drag.previewStart - drag.origStart) >= 0.001 ||
        Math.abs(drag.previewEnd - drag.origEnd) >= 0.001)
    ) {
      void updateSegment(drag.segId, {
        start_time: drag.previewStart,
        end_time: drag.previewEnd,
      });
    }
  };

  const handlePointerCancel = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    setDragOverlay(null);
    if (drag?.kind === 'segment' && drag.moved && drag.segId != null) {
      previewSegmentTiming(drag.segId, { start_time: drag.origStart, end_time: drag.origEnd });
    }
  };

  return (
    <div className="bg-sunken border-t border-line select-none" data-testid="waveform-panel">
      <div className="px-3 py-1.5 flex items-center justify-between gap-2 border-b border-line/60">
        <button
          type="button"
          onClick={toggleVisible}
          data-testid="waveform-toggle"
          className="inline-flex items-center gap-1.5 text-[11px] font-medium text-zinc-300 hover:text-white cursor-pointer"
        >
          <AudioWaveform className="w-3.5 h-3.5 text-indigo-400" />
          <span>{t('waveform.title')}</span>
          {visible ? <ChevronUp className="w-3.5 h-3.5 text-zinc-500" /> : <ChevronDown className="w-3.5 h-3.5 text-zinc-500" />}
        </button>

        {visible && (
          <div className="flex items-center gap-2">
            {hasOnscreen && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/25"
                data-testid="waveform-dual-lane-badge"
              >
                {t('waveform.dualLane')}
              </span>
            )}
            {loadingWaveform && (
              <span className="inline-flex items-center gap-1 text-[10px] text-indigo-300">
                <Loader2 className="w-3 h-3 animate-spin" />
                {t('waveform.loading')}
              </span>
            )}
            {waveformError && !loadingWaveform && (
              <button
                type="button"
                onClick={() => setRetryNonce((n) => n + 1)}
                className="inline-flex items-center gap-1 text-[10px] text-amber-300 hover:text-amber-200 cursor-pointer"
                title={t('waveform.retry')}
              >
                <RefreshCw className="w-3 h-3" />
                {t('waveform.unavailable')}
              </button>
            )}
            <div className="flex items-center gap-1 bg-surface border border-line rounded-md px-1 py-0.5" dir="ltr">
              <button
                type="button"
                onClick={() => stepZoom(-1)}
                disabled={windowSec <= ZOOM_LEVELS[0]}
                data-testid="waveform-zoom-in"
                title={t('waveform.zoomIn')}
                aria-label={t('waveform.zoomIn')}
                className="p-0.5 rounded text-zinc-400 hover:text-white disabled:opacity-35 cursor-pointer"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] font-mono text-zinc-300 px-1 min-w-[28px] text-center" data-testid="waveform-zoom-label">
                {windowSec}s
              </span>
              <button
                type="button"
                onClick={() => stepZoom(1)}
                disabled={windowSec >= ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
                data-testid="waveform-zoom-out"
                title={t('waveform.zoomOut')}
                aria-label={t('waveform.zoomOut')}
                className="p-0.5 rounded text-zinc-400 hover:text-white disabled:opacity-35 cursor-pointer"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {visible && (
        <div ref={containerRef} className="relative w-full overflow-hidden" dir="ltr">
          <canvas
            ref={canvasRef}
            data-testid="waveform-canvas"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
            style={{ width: '100%', height: `${canvasHeight}px`, cursor: cursorStyle, display: 'block', touchAction: 'none' }}
          />
          {dragOverlay && (() => {
            const containerW = Math.max(200, dragOverlay.width || canvasWidth);
            const halfPad = dragOverlay.kind === 'segment' ? Math.min(96, containerW / 2) : Math.min(64, containerW / 2);
            const leftPx = clampTime(dragOverlay.x, halfPad, Math.max(halfPad, containerW - halfPad));
            return (
              <div
                data-testid="waveform-drag-overlay"
                className="pointer-events-none absolute top-5 -translate-x-1/2 z-20 px-2 py-0.5 rounded-md bg-zinc-900/95 border border-indigo-400/60 text-[11px] font-mono text-zinc-100 shadow-lg whitespace-nowrap flex items-center gap-1.5"
                style={{ left: `${leftPx}px` }}
              >
                {dragOverlay.kind === 'seek' ? (
                  <>
                    <span>{formatPreciseTime(dragOverlay.targetTime ?? 0)}</span>
                    {' '}
                    <span className="text-indigo-300 font-semibold">
                      ({formatSignedDelta(dragOverlay.deltaSec)})
                    </span>
                  </>
                ) : (
                  <>
                    {dragOverlay.segIndex != null && (
                      <span className="text-zinc-400">#{dragOverlay.segIndex} · </span>
                    )}
                    <span>
                      {formatPreciseTime(dragOverlay.startTime ?? 0)} – {formatPreciseTime(dragOverlay.endTime ?? 0)}
                    </span>
                    {' '}
                    <span className="text-indigo-300 font-semibold">
                      ({formatSignedDelta(dragOverlay.deltaSec)})
                    </span>
                  </>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};
