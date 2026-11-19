import React, { useState } from 'react';
import { CaseSensitive, ChevronDown, ChevronUp, Regex, Replace, Search, WholeWord } from 'lucide-react';
import { useJobStore } from '../../../store/useJobStore';
import { useT } from '../../../i18n';
import { buildSearchPattern, planReplacements, type ReplacePatch } from '../../../utils/textSearch';

interface SearchReplaceBarProps {
  searchTerm: string;
  onSearchChange: (value: string) => void;
  useRegex: boolean;
  onToggleRegex: () => void;
  matchCase: boolean;
  onToggleMatchCase: () => void;
  wholeWord: boolean;
  onToggleWholeWord: () => void;
  regexError: string | null;
  isLive: boolean;
  /** Shown for a few seconds in the status line after "Replace all" */
  onResult: (message: string) => void;
  matchCount?: number;
  currentMatchIndex?: number;
  onNextMatch?: () => void;
  onPrevMatch?: () => void;
}

/** Top toolbar row: search (plain or regex), replacement text, target column and "Replace all". */
export const SearchReplaceBar: React.FC<SearchReplaceBarProps> = ({
  searchTerm,
  onSearchChange,
  useRegex,
  onToggleRegex,
  matchCase,
  onToggleMatchCase,
  wholeWord,
  onToggleWholeWord,
  regexError,
  isLive,
  onResult,
  matchCount = 0,
  currentMatchIndex = 0,
  onNextMatch,
  onPrevMatch,
}) => {
  const replaceAll = useJobStore((s) => s.replaceAll);
  const t = useT();
  const [replaceTerm, setReplaceTerm] = useState('');
  const [targetColumn, setTargetColumn] = useState<'translated' | 'original'>('translated');
  // "Replace all" first shows what would change; nothing is saved until it is confirmed
  const [preview, setPreview] = useState<ReplacePatch[] | null>(null);
  const options = { useRegex, matchCase, wholeWord };

  const handleReplaceAll = () => {
    if (!searchTerm.trim() || (useRegex && !!regexError)) return;
    const { regex } = buildSearchPattern(searchTerm, options);
    const segments = useJobStore.getState().activeJob?.segments ?? [];
    const field = targetColumn === 'translated' ? 'translated_text' : 'original_text';
    const plan = regex ? planReplacements(segments, field, regex, replaceTerm, useRegex) : [];
    if (plan.length === 0) {
      onResult(t('table.noMatches'));
      return;
    }
    setPreview(plan);
  };

  const confirmReplaceAll = async () => {
    setPreview(null);
    const n = await replaceAll(searchTerm, replaceTerm, targetColumn, options);
    onResult(n ? t('table.replaced', { count: n }) : t('table.noMatches'));
  };

  const toggleClass = (on: boolean) =>
    `p-1 rounded transition-colors cursor-pointer flex items-center justify-center ${
      on
        ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/50 hover:bg-indigo-600/40'
        : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/60'
    }`;

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.altKey && e.key.toLowerCase() === 'r') {
      e.preventDefault();
      onToggleRegex();
    } else if (e.key === 'Enter' && searchTerm.trim()) {
      e.preventDefault();
      if (e.shiftKey) onPrevMatch?.();
      else onNextMatch?.();
    }
  };

  const showMatchNav = searchTerm.trim().length > 0 && matchCount > 0 && (onNextMatch || onPrevMatch);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <div
          className={`flex-1 flex items-center bg-surface border rounded-lg px-2 transition-colors ${
            useRegex && regexError
              ? 'border-rose-500/80 focus-within:border-rose-500'
              : 'border-line focus-within:border-indigo-500'
          }`}
        >
          <Search className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0 pointer-events-none" />
          <input
            type="text"
            placeholder={useRegex ? t('table.searchRegex') : t('table.search')}
            dir="auto"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            className={`flex-1 min-w-0 bg-transparent px-2 py-1.5 text-xs placeholder-zinc-500 focus:outline-none ${
              useRegex && regexError ? 'text-rose-200' : 'text-zinc-200'
            }`}
          />
          {showMatchNav && (
            <div className="flex items-center gap-0.5 me-1 text-[10px] text-zinc-400 tabular-nums" dir="ltr">
              <span className="px-1">{`${currentMatchIndex + 1}/${matchCount}`}</span>
              {onPrevMatch && (
                <button
                  type="button"
                  onClick={onPrevMatch}
                  title={t('table.prevMatch')}
                  aria-label={t('table.prevMatch')}
                  data-testid="search-prev-match"
                  className="p-0.5 rounded hover:bg-zinc-800 hover:text-white cursor-pointer"
                >
                  <ChevronUp className="w-3.5 h-3.5" />
                </button>
              )}
              {onNextMatch && (
                <button
                  type="button"
                  onClick={onNextMatch}
                  title={t('table.nextMatch')}
                  aria-label={t('table.nextMatch')}
                  data-testid="search-next-match"
                  className="p-0.5 rounded hover:bg-zinc-800 hover:text-white cursor-pointer"
                >
                  <ChevronDown className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          )}
          <button
            type="button"
            onClick={onToggleMatchCase}
            title={t('ux.matchCase')}
            aria-label={t('ux.matchCase')}
            aria-pressed={matchCase}
            data-testid="search-match-case"
            className={toggleClass(matchCase)}
          >
            <CaseSensitive className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onToggleWholeWord}
            title={t('ux.wholeWord')}
            aria-label={t('ux.wholeWord')}
            aria-pressed={wholeWord}
            data-testid="search-whole-word"
            className={toggleClass(wholeWord)}
          >
            <WholeWord className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onToggleRegex}
            title={t('table.regexToggle')}
            aria-label={t('table.regexToggle')}
            aria-pressed={useRegex}
            className={`p-1 rounded transition-colors cursor-pointer flex items-center justify-center ${
              useRegex
                ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/50 hover:bg-indigo-600/40'
                : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/60'
            }`}
          >
            <Regex className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="relative flex-1">
          <Replace className="w-3.5 h-3.5 absolute start-2.5 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
          <input
            type="text"
            placeholder={t('table.replace')}
            dir="auto"
            value={replaceTerm}
            onChange={(e) => {
              setReplaceTerm(e.target.value);
              setPreview(null);
            }}
            className="w-full bg-surface border border-line rounded-lg ps-8 pe-3 py-1.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-indigo-500"
          />
        </div>
        <select
          value={targetColumn}
          onChange={(e) => {
            setTargetColumn(e.target.value as 'translated' | 'original');
            setPreview(null);
          }}
          className="bg-surface border border-line rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-indigo-500"
        >
          <option value="translated">{t('table.inTranslated')}</option>
          <option value="original">{t('table.inOriginal')}</option>
        </select>
        <button
          onClick={handleReplaceAll}
          disabled={!searchTerm.trim() || (useRegex && !!regexError) || isLive}
          data-testid="replace-all"
          className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-xs font-semibold text-zinc-200 transition-colors cursor-pointer"
        >
          {t('table.replaceAll')}
        </button>
      </div>
      {preview && (
        <div
          data-testid="replace-preview"
          className="p-2.5 rounded-lg border border-indigo-500/40 bg-indigo-500/5 text-[11px] text-zinc-300 space-y-1.5"
        >
          <div className="font-medium text-indigo-200">{t('ux.replacePreview', { count: preview.length })}</div>
          <ul className="space-y-0.5" dir="auto">
            {preview.slice(0, 3).map((p) => (
              <li key={p.id} className="truncate">
                <span className="text-zinc-500 line-through">{p.before.replace(/\n/g, ' ')}</span>
                <span className="text-zinc-500 px-1">→</span>
                <span className="text-zinc-100">{p.after.replace(/\n/g, ' ')}</span>
              </li>
            ))}
            {preview.length > 3 && <li className="text-zinc-500">{t('ux.andMore', { count: preview.length - 3 })}</li>}
          </ul>
          <div className="flex gap-2 pt-0.5">
            <button
              type="button"
              data-testid="replace-confirm"
              onClick={() => void confirmReplaceAll()}
              className="px-3 py-1 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white font-semibold cursor-pointer"
            >
              {t('ux.replaceConfirm', { count: preview.length })}
            </button>
            <button
              type="button"
              onClick={() => setPreview(null)}
              className="px-3 py-1 rounded-md border border-line text-zinc-300 hover:border-zinc-500 cursor-pointer"
            >
              {t('ux.cancel')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
