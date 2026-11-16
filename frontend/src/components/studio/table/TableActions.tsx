import React, { useEffect, useRef, useState } from 'react';
import { Clock, Loader2, Plus, ShieldCheck, Timer } from 'lucide-react';
import { useJobStore } from '../../../store/useJobStore';
import { useT } from '../../../i18n';
import { HistoryControls } from './HistoryControls';

interface TableActionsProps {
  /** A finished job with subtitles: re-check timing and the quality check are available */
  canRecheck: boolean;
  canInsert: boolean;
  qaBusy: boolean;
  onRunQa: () => void;
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
  checkedCount?: number;
  onShiftTimes?: (offsetSeconds: number) => void;
}

/** Right side of the second toolbar row: undo/redo/history, shift times, re-check timing, add line, quality check, auto-scroll. */
export const TableActions: React.FC<TableActionsProps> = ({
  canRecheck,
  canInsert,
  qaBusy,
  onRunQa,
  autoScroll,
  onAutoScrollChange,
  checkedCount = 0,
  onShiftTimes,
}) => {
  const retimeActiveJob = useJobStore((s) => s.retimeActiveJob);
  const timingBusy = useJobStore((s) => s.timingBusy);
  const timingNote = useJobStore((s) => s.timingNote);
  const openInsert = useJobStore((s) => s.openInsert);
  const t = useT();
  const [shiftOpen, setShiftOpen] = useState(false);
  const [shiftOffset, setShiftOffset] = useState('0.5');
  const shiftRootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!shiftOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (shiftRootRef.current && !shiftRootRef.current.contains(e.target as Node)) setShiftOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShiftOpen(false);
    };
    window.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('keydown', onKey);
    };
  }, [shiftOpen]);

  const applyShift = (offsetSeconds: number) => {
    onShiftTimes?.(offsetSeconds);
    setShiftOpen(false);
  };

  return (
    <>
      {canRecheck && <HistoryControls />}
      {canInsert && onShiftTimes && (
        <div ref={shiftRootRef} className="relative">
          <button
            type="button"
            onClick={() => setShiftOpen((v) => !v)}
            title={t('table.shiftTimesTitle')}
            data-testid="shift-time-toggle"
            aria-expanded={shiftOpen}
            className={`flex items-center gap-1.5 px-2 py-1 rounded border text-zinc-300 hover:text-white hover:border-indigo-500/60 cursor-pointer ${
              shiftOpen ? 'border-indigo-500/60 text-white' : 'border-line'
            }`}
          >
            <Clock className="w-3 h-3" />
            {t('table.shiftTimes')}
          </button>
          {shiftOpen && (
            <div
              role="dialog"
              data-testid="shift-time-popover"
              className="absolute top-full end-0 mt-1 z-30 w-64 rounded-lg border border-line bg-surface p-3 shadow-xl shadow-black/40 space-y-2 text-xs"
            >
              <div className="font-semibold text-zinc-200">{t('table.shiftTimes')}</div>
              <p className="text-[11px] text-zinc-400">
                {checkedCount > 0 ? t('table.shiftScopeSelected', { count: checkedCount }) : t('table.shiftScopeAll')}
              </p>
              <label className="block space-y-1">
                <span className="text-[11px] text-zinc-400">{t('table.shiftOffsetLabel')}</span>
                <input
                  type="number"
                  step="0.1"
                  dir="ltr"
                  value={shiftOffset}
                  onChange={(e) => setShiftOffset(e.target.value)}
                  data-testid="shift-time-input"
                  className="w-full bg-sunken border border-line rounded px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-indigo-500"
                />
              </label>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => applyShift(-0.5)}
                  className="px-2 py-1 rounded border border-line text-zinc-300 hover:text-white hover:border-indigo-500/60 cursor-pointer"
                >
                  -0.5s
                </button>
                <button
                  type="button"
                  onClick={() => applyShift(0.5)}
                  className="px-2 py-1 rounded border border-line text-zinc-300 hover:text-white hover:border-indigo-500/60 cursor-pointer"
                >
                  +0.5s
                </button>
                <button
                  type="button"
                  data-testid="shift-time-apply"
                  onClick={() => {
                    const val = Number(shiftOffset);
                    if (!Number.isNaN(val) && Math.abs(val) >= 0.001) applyShift(val);
                  }}
                  className="ms-auto px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium cursor-pointer"
                >
                  {t('table.shiftApply')}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      {canRecheck && (
        <span className="flex items-center gap-2">
          {timingNote !== null && !timingBusy && (
            <span className="text-indigo-300">
              {Number(timingNote) > 0 ? t('table.timingFixed', { count: Number(timingNote) }) : t('table.timingOk')}
            </span>
          )}
          <button
            onClick={() => void retimeActiveJob()}
            disabled={timingBusy}
            title={t('table.recheckTimingTitle')}
            className="flex items-center gap-1.5 px-2 py-1 rounded border border-line text-zinc-300 hover:text-white hover:border-indigo-500/60 disabled:opacity-60 disabled:cursor-wait"
          >
            {timingBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Timer className="w-3 h-3" />}
            {timingBusy ? t('table.recheckingTiming') : t('table.recheckTiming')}
          </button>
        </span>
      )}
      {canInsert && (
        <button
          onClick={() => openInsert({ time: useJobStore.getState().videoCurrentTime })}
          title={t('table.addLineTitle')}
          data-testid="add-line"
          className="flex items-center gap-1.5 px-2 py-1 rounded border border-line text-zinc-300 hover:text-white hover:border-indigo-500/60"
        >
          <Plus className="w-3 h-3" />
          {t('table.addLine')}
        </button>
      )}
      {canRecheck && (
        <button
          onClick={onRunQa}
          disabled={qaBusy}
          title={t('table.qaTitle')}
          data-testid="qa-check"
          className="flex items-center gap-1.5 px-2 py-1 rounded border border-line text-zinc-300 hover:text-white hover:border-indigo-500/60 disabled:opacity-60 disabled:cursor-wait"
        >
          {qaBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <ShieldCheck className="w-3 h-3" />}
          {t('table.qa')}
        </button>
      )}
      <label className="flex items-center gap-1.5 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={autoScroll}
          onChange={(e) => onAutoScrollChange(e.target.checked)}
          className="w-3.5 h-3.5 rounded border-zinc-700 text-indigo-600 bg-zinc-800"
        />
        <span>{t('table.autoScroll')}</span>
      </label>
    </>
  );
};
