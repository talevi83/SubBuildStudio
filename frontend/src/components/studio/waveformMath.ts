import type { SubtitleSegment } from '../../types';

export const ZOOM_LEVELS = [6, 10, 15, 20, 30, 45, 60] as const;
export const MIN_SEGMENT_DURATION = 0.2;
export const LANE_GAP = 0.02;
export const HANDLE_HIT_PX = 7;

export type DragMode = 'resize-start' | 'resize-end' | 'move';

export interface LaneLayout {
  rulerHeight: number;
  onscreenTop: number;
  onscreenHeight: number;
  mainTop: number;
  mainHeight: number;
}

export function getLaneLayout(height: number, hasOnscreen: boolean): LaneLayout {
  const rulerHeight = 18;
  if (hasOnscreen) {
    const onscreenTop = rulerHeight + 3;
    const onscreenHeight = 22;
    const mainTop = onscreenTop + onscreenHeight + 4;
    const mainHeight = Math.max(36, height - mainTop - 4);
    return { rulerHeight, onscreenTop, onscreenHeight, mainTop, mainHeight };
  }
  const mainTop = rulerHeight + 4;
  const mainHeight = Math.max(44, height - mainTop - 4);
  return { rulerHeight, onscreenTop: 0, onscreenHeight: 0, mainTop, mainHeight };
}

export function computeTimeWindow(currentTime: number, windowSec: number, effectiveDuration: number): {
  viewStart: number;
  viewEnd: number;
} {
  const dur = Math.max(4, effectiveDuration);
  const win = Math.min(windowSec, dur);
  let viewStart = currentTime - win * 0.4;
  if (viewStart + win > dur) viewStart = Math.max(0, dur - win);
  if (viewStart < 0) viewStart = 0;
  return { viewStart, viewEnd: viewStart + win };
}

export function isSameTimelineLane(a: SubtitleSegment, b: SubtitleSegment, allowOnscreenOverlap: boolean): boolean {
  if (!allowOnscreenOverlap) return true;
  return (a.segment_type === 'ON_SCREEN') === (b.segment_type === 'ON_SCREEN');
}

export function computeSameLaneBounds(
  seg: SubtitleSegment,
  segments: readonly SubtitleSegment[],
  effectiveDuration: number,
  allowOnscreenOverlap: boolean
): { minBound: number; maxBound: number } {
  const sameLane = [...segments]
    .filter((s) => isSameTimelineLane(s, seg, allowOnscreenOverlap))
    .sort((a, b) => a.start_time - b.start_time || a.end_time - b.end_time || a.index - b.index);
  const pos = sameLane.findIndex((s) => (seg.id != null ? s.id === seg.id : s.index === seg.index));
  const prev = pos > 0 ? sameLane[pos - 1] : null;
  const next = pos >= 0 && pos < sameLane.length - 1 ? sameLane[pos + 1] : null;
  const minBound = prev ? Math.min(seg.start_time, prev.end_time + LANE_GAP) : 0;
  const maxBound = next ? Math.max(seg.end_time, next.start_time - LANE_GAP) : Math.max(effectiveDuration, seg.end_time);
  return { minBound: Math.max(0, minBound), maxBound };
}

export const clampTime = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
export const roundMs = (v: number) => Math.round(v * 1000) / 1000;

export function applySegmentDrag(
  mode: DragMode,
  origStart: number,
  origEnd: number,
  deltaSec: number,
  minBound: number,
  maxBound: number
): { start_time: number; end_time: number } {
  const minDur = Math.min(MIN_SEGMENT_DURATION, Math.max(0.05, origEnd - origStart));
  if (mode === 'resize-start') {
    const start = clampTime(origStart + deltaSec, minBound, origEnd - minDur);
    return { start_time: roundMs(start), end_time: roundMs(origEnd) };
  }
  if (mode === 'resize-end') {
    const end = clampTime(origEnd + deltaSec, origStart + minDur, maxBound);
    return { start_time: roundMs(origStart), end_time: roundMs(end) };
  }
  const dur = Math.max(minDur, origEnd - origStart);
  const maxStart = Math.max(minBound, maxBound - dur);
  const start = clampTime(origStart + deltaSec, minBound, maxStart);
  return { start_time: roundMs(start), end_time: roundMs(start + dur) };
}

export function hitTestWaveformSegment(
  x: number,
  y: number,
  width: number,
  layout: LaneLayout,
  hasOnscreen: boolean,
  viewStart: number,
  viewEnd: number,
  segments: readonly SubtitleSegment[]
): { segment: SubtitleSegment; mode: DragMode } | null {
  const span = Math.max(0.1, viewEnd - viewStart);
  const toX = (t: number) => ((t - viewStart) / span) * width;

  const inOnscreen =
    hasOnscreen && y >= layout.onscreenTop && y <= layout.onscreenTop + layout.onscreenHeight;
  const inMain = y >= layout.mainTop && y <= layout.mainTop + layout.mainHeight;
  if (!inOnscreen && !inMain) return null;

  const candidates = segments.filter((s) =>
    inOnscreen ? s.segment_type === 'ON_SCREEN' : !hasOnscreen || s.segment_type !== 'ON_SCREEN'
  );

  for (let i = candidates.length - 1; i >= 0; i--) {
    const seg = candidates[i];
    if (seg.end_time < viewStart || seg.start_time > viewEnd) continue;
    const x1 = toX(seg.start_time);
    const x2 = toX(seg.end_time);
    if (x < x1 - 5 || x > x2 + 5) continue;

    const blockW = x2 - x1;
    const edgeThreshold = Math.min(HANDLE_HIT_PX, Math.max(3, blockW * 0.25));
    if (x <= x1 + edgeThreshold) {
      return { segment: seg, mode: 'resize-start' };
    }
    if (x >= x2 - edgeThreshold) {
      return { segment: seg, mode: 'resize-end' };
    }
    if (x >= x1 && x <= x2) {
      return { segment: seg, mode: 'move' };
    }
  }
  return null;
}
