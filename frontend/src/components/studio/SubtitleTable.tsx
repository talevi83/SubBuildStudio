import React, { useState, useRef, useMemo, useCallback, useDeferredValue } from 'react';
import { AlertTriangle, Check, Loader2 } from 'lucide-react';
import { useJobStore } from '../../store/useJobStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { inferProviderFromModelId, isAnyAiReady, isProviderReady } from '../settings/labels';
import { isActiveStatus, isRtlLanguage } from '../../types';
import type { SubtitleSegment } from '../../types';
import { SubtitleRow } from './SubtitleRow';
import { RowTranslateDialog } from './RowTranslateDialog';
import { InsertSegmentDialog } from './InsertSegmentDialog';
import { findSegmentAt } from '../../utils/time';
import { useT } from '../../i18n';
import { api } from '../../services/api';
import { SearchReplaceBar } from './table/SearchReplaceBar';
import { TagFilterSelect } from './table/TagFilterSelect';
import { JumpToRow } from './table/JumpToRow';
import { ProblemFilterSelect } from './table/ProblemFilterSelect';
import { TableActions } from './table/TableActions';
import { QaPanel } from './table/QaPanel';
import { SelectionBar, type RowTranslateOptions } from './table/SelectionBar';
import { useSegmentFilters } from './table/useSegmentFilters';
import { useRowSelection } from './table/useRowSelection';
import { useVirtualRows } from './table/useVirtualRows';
import { useRowDragDrop } from './table/useRowDragDrop';
import { useQualityCheck } from './table/useQualityCheck';

/** Clicking a row starts playback this many seconds before the subtitle, to hear the lead-in */
const SEEK_LEAD_IN_S = 1.5;

export const SubtitleTable: React.FC = () => {
  const activeJob = useJobStore((s) => s.activeJob);
  const selectedSegmentIndex = useJobStore((s) => s.selectedSegmentIndex);
  const setSelectedSegmentIndex = useJobStore((s) => s.setSelectedSegmentIndex);
  const problemFilter = useJobStore((s) => s.problemFilter);
  const seekTo = useJobStore((s) => s.seekTo);
  const updateSegment = useJobStore((s) => s.updateSegment);
  const deleteSegment = useJobStore((s) => s.deleteSegment);
  const deleteSegments = useJobStore((s) => s.deleteSegments);
  const splitSegment = useJobStore((s) => s.splitSegment);
  const mergeSegments = useJobStore((s) => s.mergeSegments);
  const shiftSegments = useJobStore((s) => s.shiftSegments);
  const moveSegment = useJobStore((s) => s.moveSegment);
  const savingCount = useJobStore((s) => s.savingCount);
  const lastSavedAt = useJobStore((s) => s.lastSavedAt);
  const retranslate = useJobStore((s) => s.retranslateActiveJob);
  const systemInfo = useJobStore((s) => s.systemInfo);
  const keyOverride = useJobStore((s) => s.pipelineConfig.gemini_api_key_override);
  const appSettings = useSettingsStore((s) => s.settings);
  const activeProvider =
    appSettings?.translation.provider || inferProviderFromModelId(activeJob?.gemini_model || appSettings?.translation.model);
  const canTranslate =
    isProviderReady(systemInfo, activeProvider, keyOverride) || isAnyAiReady(systemInfo, keyOverride);
  const timingBusy = useJobStore((s) => s.timingBusy);
  const maxCps = useJobStore((s) => s.systemInfo?.max_cps_warning ?? 20);
  const insertDraft = useJobStore((s) => s.insertDraft);
  const openInsert = useJobStore((s) => s.openInsert);
  const t = useT();

  const [searchTerm, setSearchTerm] = useState('');
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const [useRegex, setUseRegex] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [tagFilter, setTagFilter] = useState<string>('ALL');
  const [replaceResult, setReplaceResult] = useState<string | null>(null);
  const [rowsDialog, setRowsDialog] = useState(false);
  const activeRowRef = useRef<HTMLTableRowElement | null>(null);

  const segments = useMemo(
    () => [...(activeJob?.segments || [])].sort((a, b) => a.start_time - b.start_time),
    [activeJob?.segments]
  );
  const overlappedIds = useMemo(() => {
    const ids = new Set<number>();
    const scanLane = (lane: SubtitleSegment[]) => {
      let prev: SubtitleSegment | null = null;
      for (const cur of lane) {
        if (prev && cur.start_time < prev.end_time - 0.001) {
          if (prev.id) ids.add(prev.id);
          if (cur.id) ids.add(cur.id);
        }
        if (!prev || cur.end_time > prev.end_time) prev = cur;
      }
    };
    scanLane(segments.filter((s) => s.segment_type !== 'ON_SCREEN'));
    scanLane(segments.filter((s) => s.segment_type === 'ON_SCREEN'));
    return ids;
  }, [segments]);
  const isRtl = isRtlLanguage(activeJob?.target_language);
  const jobFinished = !!activeJob && !isActiveStatus(activeJob.status);
  // Only promise "Translating…" while a translation is actually running
  const translationEnabled = !!activeJob?.enable_translation && isActiveStatus(activeJob?.status);
  const hasTranslationExpectation =
    !!activeJob?.enable_translation || segments.some((s) => !!s.translated_text || !!s.translation_failed);
  // A synced file also flags lines that were placed with low confidence
  const isSyncJob = activeJob?.job_kind === 'sync';

  // Subscribe only to the primitive active subtitle index (not raw videoCurrentTime ticks)
  // so the table only re-renders when the active subtitle actually changes (~every 3-4s).
  const activeIndex = useJobStore(
    useCallback(
      (s) => {
        const i = findSegmentAt(segments, s.videoCurrentTime);
        return i >= 0 ? segments[i].index : null;
      },
      [segments]
    )
  );

  const { filteredSegments, problemStats, tagCounts, regexError } = useSegmentFilters({
    segments,
    isRtl,
    hasTranslationExpectation,
    translationEnabled,
    isSyncJob,
    maxCps,
    problemFilter,
    tagFilter,
    searchTerm: deferredSearchTerm,
    useRegex,
  });

  const isLive = segments.length > 0 && segments.some((s) => !s.id);
  const canSelect = jobFinished && !isLive;
  const canInsert = !!activeJob && jobFinished && !isLive;
  const canRecheck = !!activeJob && jobFinished && segments.length > 0;
  // Moving rows only makes sense in the full, time-ordered list of a finished job
  const canDrag =
    !!activeJob && jobFinished && !searchTerm && problemFilter === 'NONE' && tagFilter === 'ALL' && !timingBusy;

  const {
    checked,
    setChecked,
    selectableFilteredIds,
    allFilteredChecked,
    selectAllRef,
    toggleSelectAllFiltered,
    handleCheck,
    handleSelectToEnd,
    handleSelectFromStart,
    checkedRangeBounds,
  } = useRowSelection(filteredSegments, `${activeJob?.id ?? ''}:${jobFinished}`);
  const {
    scrollContainerRef,
    handleTableScroll,
    scrollToListIndex,
    scrollToTop,
    scrollToBottom,
    visibleSegments,
    topSpacerHeight,
    bottomSpacerHeight,
  } = useVirtualRows(filteredSegments, activeIndex, autoScroll);
  const { dragId, overId, dragFromPos, handleDragStart, handleDragOver, handleDragEnd, handleDrop } =
    useRowDragDrop(segments, moveSegment);
  const qa = useQualityCheck(activeJob?.id);

  const showReplaceResult = (message: string) => {
    setReplaceResult(message);
    setTimeout(() => setReplaceResult(null), 3000);
  };

  const translateRows = (opts: RowTranslateOptions) => {
    setRowsDialog(false);
    void retranslate({
      segment_ids: [...checked],
      speaker_gender: opts.speaker ?? null,
      addressee_gender: opts.addressee ?? null,
      hint: opts.hint || undefined,
      remember_hint: !!opts.remember,
      gemini_model: opts.model || undefined,
      thinking: opts.thinking,
    });
    setChecked(new Set());
  };

  const onSelect = useCallback(
    (seg: SubtitleSegment) => {
      setSelectedSegmentIndex(seg.index);
      seekTo(seg.start_time - SEEK_LEAD_IN_S, true); // seekTo clamps at 0
    },
    [setSelectedSegmentIndex, seekTo]
  );

  const jumpToIssue = (segmentId: number) => {
    const idx = filteredSegments.findIndex((s) => s.id === segmentId);
    if (idx >= 0) {
      scrollToListIndex(idx);
      onSelect(filteredSegments[idx]);
    } else {
      const seg = segments.find((s) => s.id === segmentId);
      if (seg) onSelect(seg);
    }
  };

  const insertAfter = useCallback(
    (seg: SubtitleSegment) => openInsert({ time: useJobStore.getState().videoCurrentTime, afterId: seg.id }),
    [openInsert]
  );
  const handleDeleteRow = useCallback(
    (seg: SubtitleSegment) => {
      if (seg.id) void deleteSegment(seg.id);
    },
    [deleteSegment]
  );
  const handleSplitRow = useCallback(
    (seg: SubtitleSegment) => {
      if (!seg.id) return;
      const playhead = useJobStore.getState().videoCurrentTime;
      const splitTime = playhead > seg.start_time + 0.15 && playhead < seg.end_time - 0.15 ? playhead : undefined;
      void splitSegment(seg.id, splitTime);
    },
    [splitSegment]
  );

  const handleDeleteSelected = async () => {
    if (checked.size === 0) return;
    if (!window.confirm(t('table.confirmDeleteSelectedRows', { count: checked.size }))) return;
    const ids = [...checked];
    setChecked(new Set());
    await deleteSegments(ids);
  };

  const handleMergeSelected = async () => {
    if (checked.size < 2) return;
    const ids = [...checked];
    setChecked(new Set());
    await mergeSegments(ids);
  };

  const handleShiftTimes = async (offsetSeconds: number) => {
    await shiftSegments(offsetSeconds, checked.size > 0 ? [...checked] : undefined);
  };

  const currentMatchPos = useMemo(() => {
    if (!searchTerm.trim() || filteredSegments.length === 0) return 0;
    const pos = filteredSegments.findIndex((s) => s.index === selectedSegmentIndex);
    return pos >= 0 ? pos : 0;
  }, [searchTerm, filteredSegments, selectedSegmentIndex]);

  const stepSearchMatch = useCallback(
    (delta: 1 | -1) => {
      if (filteredSegments.length === 0) return;
      const curPos = filteredSegments.findIndex((s) => s.index === selectedSegmentIndex);
      const nextPos =
        curPos >= 0
          ? (curPos + delta + filteredSegments.length) % filteredSegments.length
          : delta > 0
          ? 0
          : filteredSegments.length - 1;
      scrollToListIndex(nextPos);
      onSelect(filteredSegments[nextPos]);
    },
    [filteredSegments, selectedSegmentIndex, scrollToListIndex, onSelect]
  );

  const selectAllLabel = allFilteredChecked
    ? t('table.deselectAllShown')
    : t('table.selectAllShown', { count: selectableFilteredIds.length });

  return (
    <div className="flex flex-col h-full bg-surface border border-line rounded-xl overflow-hidden shadow-xl">
      <div className="p-3 bg-sunken border-b border-line space-y-2">
        <SearchReplaceBar
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          useRegex={useRegex}
          onToggleRegex={() => setUseRegex((v) => !v)}
          regexError={regexError}
          isLive={isLive}
          onResult={showReplaceResult}
          matchCount={searchTerm.trim() ? filteredSegments.length : 0}
          currentMatchIndex={currentMatchPos}
          onNextMatch={() => stepSearchMatch(1)}
          onPrevMatch={() => stepSearchMatch(-1)}
        />

        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-zinc-400 px-1">
          <div className="flex items-center gap-3 flex-wrap">
            <span>
              {useRegex && regexError ? (
                <span className="text-rose-400 font-medium flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3 inline" />
                  {t('table.invalidRegex')}: {regexError}
                </span>
              ) : replaceResult ? (
                <span className="text-indigo-300">{replaceResult}</span>
              ) : (
                <>
                  {t('table.showing', { shown: filteredSegments.length, total: segments.length })}
                  {isLive && <span className="text-zinc-500"> · {t('table.livePreview')}</span>}
                </>
              )}
            </span>

            {(savingCount > 0 || lastSavedAt !== null) && (
              <span
                data-testid="autosave-status"
                className={`flex items-center gap-1 text-[11px] ${
                  savingCount > 0 ? 'text-amber-300' : 'text-emerald-400'
                }`}
              >
                {savingCount > 0 ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    {t('table.saving')}
                  </>
                ) : (
                  <>
                    <Check className="w-3 h-3" />
                    {t('table.saved')}
                  </>
                )}
              </span>
            )}

            <TagFilterSelect value={tagFilter} onChange={setTagFilter} counts={tagCounts} />

            {filteredSegments.length > 20 && (
              <JumpToRow
                filteredSegments={filteredSegments}
                scrollToListIndex={scrollToListIndex}
                scrollToTop={scrollToTop}
                scrollToBottom={scrollToBottom}
              />
            )}
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <ProblemFilterSelect stats={problemStats} isRtl={isRtl} isSyncJob={isSyncJob} />
            <TableActions
              canRecheck={canRecheck}
              canInsert={canInsert}
              qaBusy={qa.busy}
              onRunQa={() => void qa.run()}
              autoScroll={autoScroll}
              onAutoScrollChange={setAutoScroll}
              checkedCount={checked.size}
              onShiftTimes={(offset) => void handleShiftTimes(offset)}
            />
          </div>
        </div>
      </div>

      {(qa.issues !== null || qa.error) && (
        <QaPanel
          issues={qa.issues}
          error={qa.error}
          canSelect={canSelect}
          onSelectRows={() => qa.issues && setChecked(new Set(qa.issues.map((i) => i.segment_id)))}
          onJump={jumpToIssue}
          onClose={qa.clear}
        />
      )}

      {checked.size > 0 && canSelect && (
        <SelectionBar
          count={checked.size}
          bounds={checkedRangeBounds}
          onSelectToEnd={handleSelectToEnd}
          onSelectFromStart={handleSelectFromStart}
          canTranslate={canTranslate}
          onTranslate={translateRows}
          onOpenNote={() => setRowsDialog(true)}
          canMerge={canInsert}
          onMerge={() => void handleMergeSelected()}
          canDelete={canInsert}
          onDelete={() => void handleDeleteSelected()}
          onClear={() => setChecked(new Set())}
        />
      )}
      {insertDraft && <InsertSegmentDialog draft={insertDraft} />}
      {rowsDialog && (
        <RowTranslateDialog
          count={checked.size}
          onCancel={() => setRowsDialog(false)}
          defaultModel={activeJob?.gemini_model}
          onDetect={
            activeJob && canRecheck
              ? (model, thinking) =>
                  api.detectSpeakers(
                    activeJob.id,
                    [...checked],
                    useJobStore.getState().pipelineConfig.gemini_api_key_override,
                    model,
                    thinking
                  )
              : undefined
          }
          onSubmit={(o) => translateRows(o)}
        />
      )}

      <div ref={scrollContainerRef} onScroll={handleTableScroll} className="flex-1 overflow-y-auto">
        <table className="w-full text-start border-collapse text-xs">
          <thead className="bg-sunken text-zinc-400 sticky top-0 z-10 border-b border-line uppercase font-semibold text-[10px] tracking-wider select-none">
            <tr>
              <th className="py-2.5 px-3 w-16 text-center">
                <div className="flex items-center justify-center gap-1.5">
                  {canSelect && selectableFilteredIds.length > 0 && (
                    <input
                      ref={selectAllRef}
                      type="checkbox"
                      checked={allFilteredChecked}
                      onChange={toggleSelectAllFiltered}
                      title={selectAllLabel}
                      aria-label={selectAllLabel}
                      data-testid="select-all-shown"
                      className="w-3.5 h-3.5 rounded border-zinc-600 text-indigo-600 bg-zinc-800 cursor-pointer"
                    />
                  )}
                  <span>#</span>
                </div>
              </th>
              <th className="py-2.5 px-3 w-48 text-start">{t('table.colTimecode')}</th>
              <th className="py-2.5 px-3 w-1/2 text-start">{t('table.colOriginal')}</th>
              <th className="py-2.5 px-3 w-1/2 text-start">{t('table.colTranslated')}</th>
              <th className="py-2.5 px-3 w-20 text-center">{t('table.colCps')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/60 font-sans">
            {filteredSegments.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-12 text-center text-zinc-500">
                  {segments.length === 0 ? t('table.empty') : t('table.noFilterMatch')}
                </td>
              </tr>
            ) : (
              <>
                {topSpacerHeight > 0 && (
                  <tr aria-hidden="true" className="border-0">
                    <td colSpan={5} style={{ height: `${topSpacerHeight}px`, padding: 0, border: 0 }} />
                  </tr>
                )}
                {visibleSegments.map((seg) => {
                  const isActive = seg.index === activeIndex;
                  const rowCanDrag = canDrag && !!seg.id;
                  const dropPosition =
                    rowCanDrag && dragId !== null && overId === seg.id && dragId !== seg.id
                      ? seg.index > (segments[dragFromPos]?.index ?? 0)
                        ? 'after'
                        : 'before'
                      : null;
                  return (
                    <SubtitleRow
                      key={seg.id ?? `live-${seg.index}`}
                      seg={seg}
                      isActive={isActive}
                      isSelected={selectedSegmentIndex === seg.index}
                      isRtl={isRtl}
                      translationEnabled={translationEnabled}
                      maxCps={maxCps}
                      onSelect={onSelect}
                      onSave={updateSegment}
                      rowRef={isActive ? activeRowRef : undefined}
                      checked={checked.has(seg.id ?? -1)}
                      onCheck={canSelect ? handleCheck : undefined}
                      onSelectToEnd={canSelect ? handleSelectToEnd : undefined}
                      onInsertAfter={canInsert ? insertAfter : undefined}
                      onSplit={canInsert ? handleSplitRow : undefined}
                      onDelete={canInsert ? handleDeleteRow : undefined}
                      onFilterTag={setTagFilter}
                      overlapsNeighbor={overlappedIds.has(seg.id ?? -1)}
                      canDrag={rowCanDrag}
                      isDragging={rowCanDrag && dragId === seg.id}
                      dropPosition={dropPosition}
                      onDragStart={rowCanDrag ? handleDragStart : undefined}
                      onDragOver={rowCanDrag ? handleDragOver : undefined}
                      onDrop={rowCanDrag ? handleDrop : undefined}
                      onDragEnd={rowCanDrag ? handleDragEnd : undefined}
                    />
                  );
                })}
                {bottomSpacerHeight > 0 && (
                  <tr aria-hidden="true" className="border-0">
                    <td colSpan={5} style={{ height: `${bottomSpacerHeight}px`, padding: 0, border: 0 }} />
                  </tr>
                )}
              </>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
