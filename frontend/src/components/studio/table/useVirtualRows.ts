import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { SubtitleSegment } from '../../../types';

/** Virtual table constants: only rows near the viewport are mounted in the DOM */
const DEFAULT_ROW_HEIGHT = 96;
const OVERSCAN_ROWS = 12;
const VIRTUALIZE_THRESHOLD = 80;
const EMPTY_HEIGHTS: ReadonlyMap<string, number> = new Map();

interface LayoutMetrics {
  layoutKey: string;
  heights: Map<string, number>;
  avgHeight: number;
  calibrated: boolean;
}

export const virtualRowKey = (seg: SubtitleSegment): string =>
  seg.id != null ? String(seg.id) : `live-${seg.index}`;

function findRowAtOffset(offsets: Float64Array, pos: number): number {
  const count = offsets.length - 1;
  if (count <= 0 || pos <= 0) return 0;
  let lo = 0;
  let hi = count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= pos) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function findMountedRow(tbody: HTMLTableSectionElement | null, key: string): HTMLTableRowElement | null {
  if (!tbody) return null;
  const rows = tbody.querySelectorAll<HTMLTableRowElement>('tr[data-row-key]');
  for (const tr of rows) {
    if (tr.dataset.rowKey === key) return tr;
  }
  return null;
}

function getStickyHeaderHeight(container: HTMLDivElement): number {
  const thead = container.querySelector('thead');
  if (!thead) return 0;
  const h = Math.round(thead.getBoundingClientRect().height);
  return h > 0 ? h : 0;
}

/**
 * Windowed rendering of the subtitle table: tracks the scroll position, viewport height, and
 * measured per-row heights (with one-time fallback calibration), returns only the rows near the
 * viewport (plus spacer heights), scroll helpers, and follows playback when the active subtitle
 * changes.
 */
export function useVirtualRows(
  filteredSegments: SubtitleSegment[],
  activeIndex: number | null,
  autoScroll: boolean,
  layoutKey = ''
) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const tbodyRef = useRef<HTMLTableSectionElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(640);
  const [metrics, setMetrics] = useState<LayoutMetrics>(() => ({
    layoutKey,
    heights: new Map(),
    avgHeight: DEFAULT_ROW_HEIGHT,
    calibrated: false,
  }));
  const scrollRafRef = useRef<number | null>(null);
  const prevLayoutKeyRef = useRef<string>(layoutKey);
  const avgHeightRef = useRef<number>(DEFAULT_ROW_HEIGHT);
  const pendingAlignRef = useRef<string | null>(null);

  const filteredSegmentsRef = useRef<SubtitleSegment[]>(filteredSegments);
  useEffect(() => {
    filteredSegmentsRef.current = filteredSegments;
  }, [filteredSegments]);

  // Track scroll container height for virtualization
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const initial = Math.round(el.clientHeight || 640);
    setViewportHeight((prev) => (prev === initial ? prev : initial));
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const next = Math.round(entry.contentRect.height);
        if (next > 0) setViewportHeight((prev) => (Math.abs(prev - next) >= 1 ? next : prev));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      if (scrollRafRef.current !== null) cancelAnimationFrame(scrollRafRef.current);
    };
  }, []);

  const totalRows = filteredSegments.length;
  const isVirtualized = totalRows > VIRTUALIZE_THRESHOLD;
  const activeHeights = metrics.layoutKey === layoutKey ? metrics.heights : EMPTY_HEIGHTS;
  const activeAvgHeight = metrics.layoutKey === layoutKey ? metrics.avgHeight : DEFAULT_ROW_HEIGHT;

  const { offsets, totalHeight } = useMemo(() => {
    const arr = new Float64Array(totalRows + 1);
    if (!isVirtualized) return { offsets: arr, totalHeight: 0 };
    let acc = 0;
    for (let i = 0; i < totalRows; i++) {
      arr[i] = acc;
      acc += activeHeights.get(virtualRowKey(filteredSegments[i])) ?? activeAvgHeight;
    }
    arr[totalRows] = acc;
    return { offsets: arr, totalHeight: acc };
  }, [filteredSegments, isVirtualized, totalRows, activeHeights, activeAvgHeight]);

  const offsetsRef = useRef<Float64Array>(offsets);
  const totalHeightRef = useRef<number>(totalHeight);
  useEffect(() => {
    offsetsRef.current = offsets;
    totalHeightRef.current = totalHeight;
    avgHeightRef.current = activeAvgHeight;
  }, [offsets, totalHeight, activeAvgHeight]);

  const effectiveScrollTop = isVirtualized
    ? Math.max(0, Math.min(scrollTop, Math.max(0, totalHeight - viewportHeight)))
    : 0;
  const firstInView = isVirtualized ? findRowAtOffset(offsets, effectiveScrollTop) : 0;
  const lastInView = isVirtualized ? findRowAtOffset(offsets, effectiveScrollTop + viewportHeight) : totalRows - 1;
  const startIndex = isVirtualized ? Math.max(0, firstInView - OVERSCAN_ROWS) : 0;
  const endIndex = isVirtualized ? Math.min(totalRows, lastInView + 1 + OVERSCAN_ROWS) : totalRows;
  const topSpacerHeight = isVirtualized ? Math.round(offsets[startIndex]) : 0;
  const bottomSpacerHeight = isVirtualized ? Math.round(Math.max(0, totalHeight - offsets[endIndex])) : 0;
  const visibleSegments = isVirtualized ? filteredSegments.slice(startIndex, endIndex) : filteredSegments;

  // Measure rendered row heights once per render window; calibrate fallback once per layoutKey so
  // off-screen prefix sums above the viewport never shift mid-scroll. Also run one-shot Phase 2
  // DOM alignment after a large jump into a previously unmounted region.
  useLayoutEffect(() => {
    const layoutChanged = prevLayoutKeyRef.current !== layoutKey;
    prevLayoutKeyRef.current = layoutKey;
    if (!isVirtualized || !tbodyRef.current) {
      pendingAlignRef.current = null;
      if (layoutChanged) {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setMetrics((prev) =>
          prev.layoutKey === layoutKey && prev.heights.size === 0 && !prev.calibrated
            ? prev
            : { layoutKey, heights: new Map(), avgHeight: DEFAULT_ROW_HEIGHT, calibrated: false }
        );
      }
      return;
    }

    const rows = tbodyRef.current.querySelectorAll<HTMLTableRowElement>('tr[data-row-key]');
    const measuredEntries: Array<[string, number]> = [];
    for (const tr of rows) {
      const key = tr.dataset.rowKey;
      if (!key) continue;
      const measured = Math.round(tr.getBoundingClientRect().height);
      if (measured > 0) measuredEntries.push([key, measured]);
    }

    if (layoutChanged || measuredEntries.length > 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setMetrics((prev) => {
        const baseHeights = prev.layoutKey === layoutKey ? prev.heights : new Map<string, number>();
        let nextHeights: Map<string, number> | null = null;
        for (const [key, measured] of measuredEntries) {
          const cur = (nextHeights ?? baseHeights).get(key);
          if (cur === undefined || Math.abs(cur - measured) >= 1) {
            if (!nextHeights) nextHeights = new Map(baseHeights);
            nextHeights.set(key, measured);
          }
        }

        const wasCalibrated = prev.layoutKey === layoutKey ? prev.calibrated : false;
        const curHeights = nextHeights ?? baseHeights;
        let nextAvg = prev.layoutKey === layoutKey ? prev.avgHeight : DEFAULT_ROW_HEIGHT;
        let nextCalibrated = wasCalibrated;
        if (!wasCalibrated && curHeights.size > 0) {
          let sum = 0;
          for (const h of curHeights.values()) sum += h;
          nextAvg = Math.max(40, Math.round(sum / curHeights.size));
          nextCalibrated = true;
        }

        if (
          prev.layoutKey === layoutKey &&
          !nextHeights &&
          nextAvg === prev.avgHeight &&
          nextCalibrated === prev.calibrated
        ) {
          return prev;
        }
        return {
          layoutKey,
          heights: curHeights,
          avgHeight: nextAvg,
          calibrated: nextCalibrated,
        };
      });
    }

    const pendingKey = pendingAlignRef.current;
    if (pendingKey) {
      pendingAlignRef.current = null;
      const el = scrollContainerRef.current;
      const tr = findMountedRow(tbodyRef.current, pendingKey);
      if (el && tr && el.clientHeight > 0) {
        const rRect = tr.getBoundingClientRect();
        if (rRect.height > 0) {
          const cRect = el.getBoundingClientRect();
          const headerH = getStickyHeaderHeight(el);
          const visTop = cRect.top + headerH;
          const visHeight = Math.max(1, cRect.bottom - visTop);
          const desiredTopY = visTop + Math.floor(visHeight * 0.28);
          const delta = rRect.top - desiredTopY;
          if (Math.abs(delta) >= 4) {
            const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
            const nextTop = Math.max(0, Math.min(maxScroll, Math.round(el.scrollTop + delta)));
            if (Math.abs(nextTop - el.scrollTop) >= 1) {
              el.scrollTop = nextTop;
              // eslint-disable-next-line react-hooks/set-state-in-effect
              setScrollTop((prev) => (prev === nextTop ? prev : nextTop));
            }
          }
        }
      }
    }
  }, [isVirtualized, layoutKey, visibleSegments]);

  const handleTableScroll = useCallback(() => {
    if (scrollRafRef.current !== null) return;
    scrollRafRef.current = requestAnimationFrame(() => {
      scrollRafRef.current = null;
      const el = scrollContainerRef.current;
      if (!el) return;
      const next = Math.round(el.scrollTop);
      setScrollTop((prev) => (prev === next ? prev : next));
    });
  }, []);

  const scrollToRowIndex = useCallback((listIdx: number, mode: 'auto' | 'jump') => {
    const el = scrollContainerRef.current;
    const seg = filteredSegmentsRef.current[listIdx];
    if (!el || !seg) return;
    const key = virtualRowKey(seg);
    const mounted = findMountedRow(tbodyRef.current, key);
    if (mounted && el.clientHeight > 0) {
      const rRect = mounted.getBoundingClientRect();
      if (rRect.height > 0) {
        pendingAlignRef.current = null;
        const cRect = el.getBoundingClientRect();
        const headerH = getStickyHeaderHeight(el);
        const visTop = cRect.top + headerH;
        const visBottom = cRect.bottom;
        const visHeight = Math.max(1, visBottom - visTop);
        if (mode === 'auto' && rRect.top >= visTop + 8 && rRect.bottom <= visBottom - 16) {
          return;
        }
        const desiredTopY = visTop + Math.floor(visHeight * 0.28);
        const delta = rRect.top - desiredTopY;
        const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
        const nextTop = Math.max(0, Math.min(maxScroll, Math.round(el.scrollTop + delta)));
        el.scrollTop = nextTop;
        setScrollTop((prev) => (prev === nextTop ? prev : nextTop));
        return;
      }
    }

    const offsetsArr = offsetsRef.current;
    const useMeasured = offsetsArr.length > listIdx + 1 && totalHeightRef.current > 0;
    const headerH = getStickyHeaderHeight(el);
    const rowTop = headerH + (useMeasured ? offsetsArr[listIdx] : listIdx * avgHeightRef.current);
    const rowBottom = headerH + (useMeasured ? offsetsArr[listIdx + 1] : rowTop - headerH + avgHeightRef.current);
    const viewH = el.clientHeight || 640;
    const visHeight = Math.max(1, viewH - headerH);
    if (mode === 'auto' && rowTop >= el.scrollTop + headerH + 8 && rowBottom <= el.scrollTop + viewH - 16) {
      return;
    }
    const desired = Math.max(0, Math.round(rowTop - headerH - Math.floor(visHeight * 0.28)));
    if (!mounted) {
      pendingAlignRef.current = key;
    }
    el.scrollTop = desired;
    setScrollTop((prev) => (prev === desired ? prev : desired));
  }, []);

  const scrollToListIndex = useCallback(
    (listIdx: number) => {
      scrollToRowIndex(listIdx, 'jump');
    },
    [scrollToRowIndex]
  );

  const scrollToTop = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTop = 0;
    setScrollTop(0);
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const maxTop =
      el.scrollHeight > 0
        ? Math.max(0, el.scrollHeight - el.clientHeight)
        : Math.max(0, Math.round(totalHeightRef.current - (el.clientHeight || 640)));
    el.scrollTop = maxTop;
    setScrollTop(maxTop);
  }, []);

  // Follow playback — only when the active subtitle changes, not on every time tick
  useEffect(() => {
    if (!autoScroll || activeIndex === null) return;
    const idx = filteredSegmentsRef.current.findIndex((s) => s.index === activeIndex);
    if (idx >= 0) scrollToRowIndex(idx, 'auto');
  }, [activeIndex, autoScroll, scrollToRowIndex]);

  return {
    scrollContainerRef,
    tbodyRef,
    handleTableScroll,
    scrollToListIndex,
    scrollToTop,
    scrollToBottom,
    visibleSegments,
    topSpacerHeight,
    bottomSpacerHeight,
  };
}
