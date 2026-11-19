import React, { useEffect } from 'react';
import { Keyboard, X } from 'lucide-react';
import { useUiStore } from '../../store/useUiStore';
import { useT, type MessageKey } from '../../i18n';
import { isTypingTarget } from '../../utils/time';

type Row = { keys: string[]; label: MessageKey };

const GROUPS: { title: MessageKey; rows: Row[] }[] = [
  {
    title: 'ux.groupPlayer',
    rows: [
      { keys: ['Space', 'K'], label: 'ux.kPlay' },
      { keys: ['←', '→'], label: 'ux.kSeek' },
      { keys: ['↑', '↓'], label: 'ux.kPrevNext' },
      { keys: ['F'], label: 'ux.kFullscreen' },
    ],
  },
  {
    title: 'ux.groupEditing',
    rows: [
      { keys: ['['], label: 'ux.kStart' },
      { keys: [']'], label: 'ux.kEnd' },
      { keys: ['S'], label: 'ux.kSplit' },
      { keys: ['N'], label: 'ux.kInsert' },
    ],
  },
  {
    title: 'ux.groupHistory',
    rows: [
      { keys: ['Ctrl+Z'], label: 'ux.kUndo' },
      { keys: ['Ctrl+Y', 'Ctrl+Shift+Z'], label: 'ux.kRedo' },
      { keys: ['?'], label: 'ux.kShortcuts' },
    ],
  },
];

/** Every keyboard shortcut in one window; opens with "?" anywhere outside a text field. */
export const ShortcutsDialog: React.FC = () => {
  const open = useUiStore((s) => s.shortcutsOpen);
  const setOpen = useUiStore((s) => s.setShortcutsOpen);
  const t = useT();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && useUiStore.getState().shortcutsOpen) {
        setOpen(false);
        return;
      }
      if (e.key !== '?' || e.ctrlKey || e.metaKey || e.altKey || isTypingTarget(e.target)) return;
      e.preventDefault();
      setOpen(!useUiStore.getState().shortcutsOpen);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setOpen]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setOpen(false)}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-title"
        data-testid="shortcuts-dialog"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg bg-surface border border-line rounded-xl shadow-2xl p-5 space-y-4"
      >
        <div className="flex items-center justify-between">
          <h2 id="shortcuts-title" className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-indigo-400" />
            {t('ux.shortcutsTitle')}
          </h2>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label={t('ux.close')}
            className="p-1 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {GROUPS.map((group) => (
            <section key={group.title} className="space-y-1.5">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{t(group.title)}</h3>
              {group.rows.map((row) => (
                <div key={row.label} className="flex items-center justify-between gap-3 text-xs text-zinc-300">
                  <span>{t(row.label)}</span>
                  <span className="flex gap-1 shrink-0" dir="ltr">
                    {row.keys.map((k) => (
                      <kbd key={k} className="px-1.5 py-0.5 rounded border border-line bg-sunken text-[10px] text-indigo-200 font-mono">
                        {k}
                      </kbd>
                    ))}
                  </span>
                </div>
              ))}
            </section>
          ))}
        </div>
        <p className="text-[11px] text-zinc-500">{t('ux.kTargetNote')}</p>
      </div>
    </div>
  );
};
