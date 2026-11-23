import React, { memo, useEffect, useRef, useState } from 'react';
import type { SegmentPatch, SpeakerProfile, SubtitleSegment } from '../../types';
import { addresseeChoice, isUncertainAddressee } from './table/addressee';
import { registerPendingFlush } from '../../store/useJobStore';
import { CpsBadge } from './CpsBadge';
import { virtualRowKey } from './table/useVirtualRows';
import { formatTimecode, parseTimecode } from '../../utils/time';
import { useT } from '../../i18n';
import { AlertTriangle, ChevronsDown, GripVertical, Minus, Plus, Scissors, Trash2 } from 'lucide-react';

const SAVE_DELAY_MS = 600;

const ADDRESSEE_NAME_KEY = {
  male: 'table.genderMale',
  female: 'table.genderFemale',
  group: 'table.genderGroup',
  unknown: 'table.addresseeUnknown',
} as const;

const ADDRESSEE_ICON = { male: '♂', female: '♀', group: '👥', unknown: '?' } as const;

interface SubtitleRowProps {
  seg: SubtitleSegment;
  isActive: boolean;
  isSelected: boolean;
  isRtl: boolean;
  isRtlSource?: boolean;
  translationEnabled: boolean;
  maxCps: number;
  speakers?: Record<string, SpeakerProfile> | null;
  onSelect: (seg: SubtitleSegment) => void;
  onSave: (segmentId: number, patch: SegmentPatch) => void;
  rowRef?: React.Ref<HTMLTableRowElement>;
  /** Row selection (for re-translating or deleting rows); omitted while the job is running */
  checked?: boolean;
  onCheck?: (segmentId: number, shift: boolean) => void;
  /** Select from this row to the end of the displayed list in one click */
  onSelectToEnd?: (segmentId: number) => void;
  /** "+" under the row number: add a line right after this one; omitted while the job runs */
  onInsertAfter?: (seg: SubtitleSegment) => void;
  /** Split this subtitle line into two parts; omitted while the job runs */
  onSplit?: (seg: SubtitleSegment) => void;
  /** Delete this subtitle line; omitted while the job runs */
  onDelete?: (seg: SubtitleSegment) => void;
  /** Drag-to-move support (passed as primitive/stable props so React.memo never invalidates) */
  canDrag?: boolean;
  isDragging?: boolean;
  dropPosition?: 'before' | 'after' | null;
  onDragStart?: (segmentId: number) => void;
  onDragOver?: (segmentId: number) => void;
  onDrop?: (segmentId: number) => void;
  onDragEnd?: () => void;
  /** Filter the table by this tag when clicking the badge */
  onFilterTag?: (tag: string) => void;
  /** True when this cue's time range overlaps an adjacent cue in the same lane */
  overlapsNeighbor?: boolean;
}

type TextField = 'original_text' | 'translated_text';

/**
 * A text cell that edits a local draft and saves after a short pause (or on blur),
 * instead of sending a request — and re-rendering the whole table — on every keystroke.
 */
const EditableText: React.FC<{
  value: string;
  disabled: boolean;
  dir: 'rtl' | 'ltr' | 'auto';
  placeholder?: string;
  className: string;
  onCommit: (value: string) => void;
}> = ({ value, disabled, dir, placeholder, className, onCommit }) => {
  const [draft, setDraft] = useState(value);
  const dirty = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latest = useRef(value);
  const draftRef = useRef(value);
  const commitRef = useRef(onCommit);
  useEffect(() => {
    commitRef.current = onCommit;
  });

  // Accept outside changes (Replace All, translation arriving) unless the user is mid-edit
  useEffect(() => {
    latest.current = value;
    if (!dirty.current && draftRef.current !== value) {
      draftRef.current = value;
      setDraft(value); // eslint-disable-line react-hooks/set-state-in-effect
    }
  }, [value]);

  const flush = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
    if (dirty.current) {
      dirty.current = false;
      if (draftRef.current !== latest.current) commitRef.current(draftRef.current);
    }
  };

  // Save pending edits if the row unmounts (e.g. search filter changes) or before Export / Undo
  useEffect(() => {
    const unregister = registerPendingFlush(flush, () => dirty.current);
    return () => {
      unregister();
      flush();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <textarea
      rows={2}
      dir={dir}
      value={draft}
      disabled={disabled}
      placeholder={placeholder}
      onChange={(e) => {
        dirty.current = true;
        draftRef.current = e.target.value;
        setDraft(e.target.value);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(flush, SAVE_DELAY_MS);
      }}
      onBlur={flush}
      className={className}
    />
  );
};

const EditableTimecode: React.FC<{
  value: number;
  disabled: boolean;
  isValid: (seconds: number) => boolean;
  onCommit: (seconds: number) => void;
  testId?: string;
}> = ({ value, disabled, isValid, onCommit, testId }) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  const startEdit = () => {
    if (disabled) return;
    setDraft(formatTimecode(value));
    setEditing(true);
    requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
  };

  const commitEdit = () => {
    setEditing(false);
    const parsed = parseTimecode(draft);
    if (parsed === null) return;
    const rounded = Math.round(parsed * 1000) / 1000;
    if (!isValid(rounded) || Math.abs(rounded - value) < 0.0005) return;
    onCommit(rounded);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        type="text"
        dir="ltr"
        value={draft}
        data-testid={testId ? `${testId}-input` : undefined}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commitEdit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitEdit();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            setEditing(false);
          }
        }}
        className="w-24 bg-sunken border border-indigo-500/70 rounded px-1 py-0.5 text-[11px] font-mono-time text-zinc-100 focus:outline-none"
      />
    );
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={startEdit}
      data-testid={testId}
      className={
        disabled
          ? 'cursor-default'
          : 'cursor-pointer hover:text-indigo-300 hover:underline underline-offset-2 transition-colors'
      }
    >
      {formatTimecode(value)}
    </button>
  );
};

const textareaClass =
  'w-full bg-transparent hover:bg-sunken focus:bg-sunken p-1.5 rounded border hover:border-line focus:border-indigo-500/60 focus:outline-none resize-none text-xs leading-relaxed transition-colors disabled:cursor-default';

const SubtitleRowInner: React.FC<SubtitleRowProps> = ({
  seg,
  isActive,
  isSelected,
  isRtl,
  isRtlSource,
  translationEnabled,
  maxCps,
  speakers,
  onSelect,
  onSave,
  rowRef,
  checked,
  onCheck,
  onSelectToEnd,
  onInsertAfter,
  onSplit,
  onDelete,
  canDrag,
  isDragging,
  dropPosition,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  onFilterTag,
  overlapsNeighbor,
}) => {
  const t = useT();
  const editable = !!seg.id;

  const commit = (field: TextField) => (value: string) => {
    if (seg.id) onSave(seg.id, { [field]: value });
  };

  const nudge = (field: 'start_time' | 'end_time', delta: number) => {
    if (!seg.id) return;
    const next = Math.max(0, Number((seg[field] + delta).toFixed(3)));
    if (field === 'start_time' && next >= seg.end_time) return;
    if (field === 'end_time' && next <= seg.start_time) return;
    onSave(seg.id, { [field]: next });
  };

  const nudgeButtons = (field: 'start_time' | 'end_time') =>
    editable && (
      <div className="opacity-0 group-hover:opacity-100 flex items-center ms-1">
        <button onClick={() => nudge(field, -0.1)} className="p-0.5 hover:text-white text-zinc-400" title="-100ms">
          <Minus className="w-2.5 h-2.5" />
        </button>
        <button onClick={() => nudge(field, 0.1)} className="p-0.5 hover:text-white text-zinc-400" title="+100ms">
          <Plus className="w-2.5 h-2.5" />
        </button>
      </div>
    );

  const failed = !!seg.translation_failed;

  return (
    <tr
      ref={rowRef}
      data-row-key={virtualRowKey(seg)}
      onClick={() => onSelect(seg)}
      onDragOver={
        canDrag && onDragOver && seg.id
          ? (e) => {
              e.preventDefault();
              onDragOver(seg.id!);
            }
          : undefined
      }
      onDrop={
        canDrag && onDrop && seg.id
          ? (e) => {
              e.preventDefault();
              onDrop(seg.id!);
            }
          : undefined
      }
      className={`transition-colors cursor-pointer group ${
        isDragging ? 'opacity-40 ' : ''
      }${dropPosition === 'before' ? 'shadow-[inset_0_2px_0_0_rgb(99,102,241)] ' : ''}${
        dropPosition === 'after' ? 'shadow-[inset_0_-2px_0_0_rgb(99,102,241)] ' : ''
      }${
        isActive
          ? 'bg-indigo-950/40 border-s-2 border-indigo-500'
          : isSelected
          ? 'bg-zinc-800/60'
          : 'hover:bg-zinc-800/30'
      }`}
    >
      <td className="py-2 px-2 text-center text-zinc-500 font-mono text-[11px] select-none align-top">
        {canDrag && onDragStart && seg.id && (
          <span
            draggable
            onDragStart={(e) => {
              e.dataTransfer.effectAllowed = 'move';
              e.dataTransfer.setData('text/plain', String(seg.id));
              const tr = (e.currentTarget as HTMLElement).closest('tr');
              if (tr) e.dataTransfer.setDragImage(tr, 16, 16);
              onDragStart(seg.id!);
            }}
            onDragEnd={onDragEnd}
            onClick={(e) => e.stopPropagation()}
            title={t('table.dragHandle')}
            className="mx-auto mb-0.5 block w-fit cursor-grab active:cursor-grabbing text-zinc-600 hover:text-zinc-300 opacity-0 group-hover:opacity-100"
          >
            <GripVertical className="w-3.5 h-3.5" />
          </span>
        )}
        {onCheck && seg.id && (
          <input
            type="checkbox"
            checked={!!checked}
            onChange={() => undefined}
            onClick={(e) => {
              e.stopPropagation();
              if (e.altKey && onSelectToEnd) {
                onSelectToEnd(seg.id!);
              } else {
                onCheck(seg.id!, e.shiftKey);
              }
            }}
            title={t('table.selectRowToEndTitle', { index: seg.index })}
            aria-label={t('table.selectRow')}
            className={`mx-auto mb-0.5 block w-3 h-3 rounded border-zinc-700 text-indigo-600 bg-zinc-800 cursor-pointer transition-opacity ${
              checked ? 'opacity-100' : 'opacity-60 group-hover:opacity-100'
            }`}
          />
        )}
        <div>{seg.index}</div>
        <div className="flex items-center justify-center gap-0.5 mt-0.5">
          {onInsertAfter && seg.id && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onInsertAfter(seg);
              }}
              title={t('table.insertAfter', { index: seg.index })}
              aria-label={t('table.insertAfter', { index: seg.index })}
              data-testid={`insert-after-${seg.index}`}
              className="flex h-4 w-4 items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-indigo-600 opacity-0 group-hover:opacity-100 focus:opacity-100 cursor-pointer transition-opacity"
            >
              <Plus className="w-3 h-3" />
            </button>
          )}
          {onSplit && seg.id && seg.end_time - seg.start_time >= 0.4 && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onSplit(seg);
              }}
              title={t('table.splitRow', { index: seg.index })}
              aria-label={t('table.splitRow', { index: seg.index })}
              data-testid={`split-row-${seg.index}`}
              className="flex h-4 w-4 items-center justify-center rounded text-zinc-500 hover:text-indigo-200 hover:bg-indigo-600/30 opacity-0 group-hover:opacity-100 focus:opacity-100 cursor-pointer transition-opacity"
            >
              <Scissors className="w-3 h-3" />
            </button>
          )}
          {onSelectToEnd && seg.id && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onSelectToEnd(seg.id!);
              }}
              title={t('table.selectRowToEndTitle', { index: seg.index })}
              aria-label={t('table.selectRowToEndTitle', { index: seg.index })}
              data-testid={`select-to-end-${seg.index}`}
              className="flex h-4 w-4 items-center justify-center rounded text-zinc-500 hover:text-indigo-200 hover:bg-indigo-600/30 opacity-0 group-hover:opacity-100 focus:opacity-100 cursor-pointer transition-opacity"
            >
              <ChevronsDown className="w-3 h-3" />
            </button>
          )}
          {onDelete && seg.id && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(seg); // no question asked: the toast that follows offers Undo
              }}
              title={t('table.deleteRow', { index: seg.index })}
              aria-label={t('table.deleteRow', { index: seg.index })}
              data-testid={`delete-row-${seg.index}`}
              className="flex h-4 w-4 items-center justify-center rounded text-zinc-500 hover:text-rose-300 hover:bg-rose-500/20 opacity-0 group-hover:opacity-100 focus:opacity-100 cursor-pointer transition-opacity"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>
        {seg.segment_type === 'ON_SCREEN' && (
          <span
            className={`mt-1 inline-block max-w-full truncate whitespace-nowrap px-1 py-0.5 rounded text-[9px] font-sans font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 ${
              onFilterTag ? 'cursor-pointer hover:bg-cyan-500/25 hover:border-cyan-400' : ''
            }`}
            title={onFilterTag ? t('table.filterByTag', { tag: t('table.onscreenBadge') }) : t('table.onscreenTitle')}
            onClick={(e) => {
              if (onFilterTag) {
                e.stopPropagation();
                onFilterTag('ON_SCREEN');
              }
            }}
          >
            📺 {t('table.onscreenBadge')}
          </span>
        )}
        {seg.segment_type === 'FOREIGN' && (
          <span
            className={`mt-1 inline-block max-w-full truncate whitespace-nowrap px-1 py-0.5 rounded text-[9px] font-sans font-semibold bg-violet-500/15 text-violet-300 border border-violet-500/30 ${
              onFilterTag ? 'cursor-pointer hover:bg-violet-500/25 hover:border-violet-400' : ''
            }`}
            title={onFilterTag ? t('table.filterByTag', { tag: t('table.foreignBadge') }) : t('table.foreignTitle')}
            onClick={(e) => {
              if (onFilterTag) {
                e.stopPropagation();
                onFilterTag('FOREIGN');
              }
            }}
          >
            🌐 {t('table.foreignBadge')}
          </span>
        )}
        {seg.segment_type === 'COMMS' && (
          <span
            className={`mt-1 inline-block max-w-full truncate whitespace-nowrap px-1 py-0.5 rounded text-[9px] font-sans font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 ${
              onFilterTag ? 'cursor-pointer hover:bg-emerald-500/25 hover:border-emerald-400' : ''
            }`}
            title={onFilterTag ? t('table.filterByTag', { tag: t('table.commsBadge') }) : t('table.commsTitle')}
            onClick={(e) => {
              if (onFilterTag) {
                e.stopPropagation();
                onFilterTag('COMMS');
              }
            }}
          >
            📻 {t('table.commsBadge')}
          </span>
        )}
        {((speakers && Object.keys(speakers).length > 0) || seg.speaker) && seg.segment_type !== 'ON_SCREEN' && (
          <div className="mt-1 flex items-center justify-center gap-1" onClick={(e) => e.stopPropagation()}>
            {seg.speaker && speakers?.[seg.speaker]?.color && (
              <span
                className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: speakers[seg.speaker].color }}
              />
            )}
            <select
              value={seg.speaker || ''}
              disabled={!editable}
              onChange={(e) => seg.id && onSave(seg.id, { speaker: e.target.value || null })}
              data-testid="segment-speaker-pill"
              aria-label={t('table.speaker')}
              className="max-w-[82px] truncate bg-zinc-800/80 border border-line rounded px-1 py-0.5 text-[9px] font-sans text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="">{t('speakers.unassigned')}</option>
              {speakers &&
                Object.entries(speakers).map(([id, sp]) => (
                  <option key={id} value={id}>
                    {sp.label || id}
                  </option>
                ))}
              {seg.speaker && (!speakers || !speakers[seg.speaker]) && (
                <option value={seg.speaker}>{seg.speaker}</option>
              )}
            </select>
          </div>
        )}
        {seg.addressee && seg.segment_type !== 'ON_SCREEN' && (
          <div className="mt-1 flex items-center justify-center">
            <span
              data-testid="segment-addressee-pill"
              data-choice={addresseeChoice(seg.addressee) || 'unknown'}
              data-uncertain={isUncertainAddressee(seg.addressee) ? 'true' : undefined}
              data-locked={seg.addressee.locked ? 'true' : undefined}
              aria-label={`${t('table.addressee')}: ${t(ADDRESSEE_NAME_KEY[addresseeChoice(seg.addressee) || 'unknown'])}`}
              title={
                seg.addressee.locked
                  ? t('table.addresseeTitleManual')
                  : isUncertainAddressee(seg.addressee)
                    ? t('table.addresseeTitleUnsure')
                    : t('table.addresseeTitleSure', {
                        who: seg.addressee.label || t(ADDRESSEE_NAME_KEY[addresseeChoice(seg.addressee) || 'unknown']),
                        percent: Math.round(seg.addressee.confidence * 100),
                      })
              }
              className={`inline-flex max-w-[82px] items-center gap-0.5 truncate rounded px-1 py-0.5 text-[9px] font-sans border cursor-help select-none ${
                seg.addressee.locked
                  ? 'bg-indigo-500/15 border-indigo-500/50 text-indigo-200'
                  : isUncertainAddressee(seg.addressee)
                    ? 'bg-amber-500/10 border-dashed border-amber-500/60 text-amber-200'
                    : 'bg-zinc-800/80 border-line text-zinc-300'
              }`}
            >
              {seg.addressee.locked && <span aria-hidden>🔒</span>}
              <span aria-hidden>{ADDRESSEE_ICON[addresseeChoice(seg.addressee) || 'unknown']}</span>
              <span className="truncate">{t(ADDRESSEE_NAME_KEY[addresseeChoice(seg.addressee) || 'unknown'])}</span>
            </span>
          </div>
        )}
      </td>

      <td
        className={`py-2 px-3 font-mono-time text-[11px] whitespace-nowrap ${
          overlapsNeighbor ? 'text-amber-300' : 'text-zinc-400'
        }`}
        title={overlapsNeighbor ? 'Time overlaps an adjacent subtitle' : undefined}
        onClick={(e) => e.stopPropagation()}
        dir="ltr"
      >
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            {overlapsNeighbor && (
              <AlertTriangle className="w-3 h-3 text-amber-400 flex-shrink-0" aria-label="Time overlaps an adjacent subtitle" />
            )}
            <span className="text-zinc-500 text-[9px]">{t('table.in')}</span>
            <EditableTimecode
              value={seg.start_time}
              disabled={!editable}
              isValid={(s) => s >= 0 && s < seg.end_time}
              onCommit={(s) => seg.id && onSave(seg.id, { start_time: s })}
              testId={`start-time-${seg.index}`}
            />
            {nudgeButtons('start_time')}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-zinc-500 text-[9px]">{t('table.out')}</span>
            <EditableTimecode
              value={seg.end_time}
              disabled={!editable}
              isValid={(e) => e > seg.start_time}
              onCommit={(e) => seg.id && onSave(seg.id, { end_time: e })}
              testId={`end-time-${seg.index}`}
            />
            {nudgeButtons('end_time')}
          </div>
        </div>
      </td>

      <td className="py-2 px-3" onClick={(e) => e.stopPropagation()}>
        <EditableText
          value={seg.original_text}
          disabled={!editable}
          dir="auto"
          className={`${textareaClass} border-transparent text-zinc-200${isRtlSource ? ' font-rtl-hebrew' : ''}`}
          onCommit={commit('original_text')}
        />
      </td>

      <td className="py-2 px-3 relative" onClick={(e) => e.stopPropagation()}>
        <EditableText
          value={seg.translated_text || ''}
          disabled={!editable}
          dir={isRtl ? 'rtl' : 'ltr'}
          placeholder={failed ? t('table.failedPlaceholder') : translationEnabled ? t('table.translatingPlaceholder') : '—'}
          className={`${textareaClass} ${
            failed ? 'border-rose-500/50 placeholder-rose-400/70' : 'border-transparent'
          } text-amber-300 font-rtl-hebrew`}
          onCommit={commit('translated_text')}
        />
        {failed && (
          <AlertTriangle className="w-3 h-3 text-rose-400 absolute top-3 end-4" aria-label={t('table.failedTitle')} />
        )}
      </td>

      <td className="py-2 px-3 text-center select-none">
        <CpsBadge cps={seg.cps} max={maxCps} />
      </td>
    </tr>
  );
};

// Re-render a row only when its own data or highlight changes
export const SubtitleRow = memo(SubtitleRowInner);
