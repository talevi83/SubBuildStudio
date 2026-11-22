import { useCallback, useMemo } from 'react';
import type { ProblemFilter, SubtitleSegment } from '../../../types';
import { buildSearchPattern, hasMatch } from '../../../utils/textSearch';

/** Matches 3+ consecutive ASCII Latin letters left inside an RTL translation */
const LATIN_WORD_RE = /[A-Za-z]{3,}/;

export interface ProblemStats {
  all: number;
  untranslated: number;
  latin: number;
  highCps: number;
}

export interface TagCounts {
  onscreen: number;
  comms: number;
  foreign: number;
  anyTagged: number;
}

export interface SegmentFilterOptions {
  segments: SubtitleSegment[];
  isRtl: boolean;
  /** The job translates (or did): an empty translation counts as a problem */
  hasTranslationExpectation: boolean;
  /** A translation is running right now, so empty lines are still on their way */
  translationEnabled: boolean;
  /** A synced file also flags lines that were placed with low confidence */
  isSyncJob: boolean;
  maxCps: number;
  problemFilter: ProblemFilter;
  tagFilter: string;
  speakerFilter?: string;
  /** Already deferred by the caller (useDeferredValue) */
  searchTerm: string;
  useRegex: boolean;
  matchCase?: boolean;
  wholeWord?: boolean;
}

/** Problem / tag / text filters of the subtitle table, plus the counts shown in the filter menus. */
export function useSegmentFilters({
  segments,
  isRtl,
  hasTranslationExpectation,
  translationEnabled,
  isSyncJob,
  maxCps,
  problemFilter,
  tagFilter,
  speakerFilter = 'ALL',
  searchTerm,
  useRegex,
  matchCase = false,
  wholeWord = false,
}: SegmentFilterOptions) {
  const isUntranslated = useCallback(
    (s: SubtitleSegment) =>
      !!s.translation_failed || (hasTranslationExpectation && !translationEnabled && !s.translated_text?.trim()),
    [hasTranslationExpectation, translationEnabled]
  );

  const hasLatinLeftover = useCallback(
    (s: SubtitleSegment) => isRtl && !!s.translated_text && LATIN_WORD_RE.test(s.translated_text),
    [isRtl]
  );

  const isHighCps = useCallback(
    (s: SubtitleSegment) => (s.cps ?? 0) > maxCps || (isSyncJob && (s.confidence ?? 1) < 0.5),
    [maxCps, isSyncJob]
  );

  const isAnyProblem = useCallback(
    (s: SubtitleSegment) => isUntranslated(s) || hasLatinLeftover(s) || isHighCps(s),
    [isUntranslated, hasLatinLeftover, isHighCps]
  );

  const problemStats = useMemo<ProblemStats>(() => {
    let all = 0;
    let untranslated = 0;
    let latin = 0;
    let highCps = 0;
    for (const s of segments) {
      const u = isUntranslated(s);
      const l = hasLatinLeftover(s);
      const c = isHighCps(s);
      if (u) untranslated++;
      if (l) latin++;
      if (c) highCps++;
      if (u || l || c) all++;
    }
    return { all, untranslated, latin, highCps };
  }, [segments, isUntranslated, hasLatinLeftover, isHighCps]);

  const tagCounts = useMemo<TagCounts>(() => {
    let onscreen = 0;
    let comms = 0;
    let foreign = 0;
    let anyTagged = 0;
    for (const s of segments) {
      if (s.segment_type && s.segment_type !== 'DIALOGUE') {
        anyTagged++;
        if (s.segment_type === 'ON_SCREEN') onscreen++;
        else if (s.segment_type === 'COMMS') comms++;
        else if (s.segment_type === 'FOREIGN') foreign++;
      }
    }
    return { onscreen, comms, foreign, anyTagged };
  }, [segments]);

  const { regex: searchRegex, error: regexError } = useMemo(
    () => buildSearchPattern(searchTerm.trim() ? searchTerm : '', { useRegex, matchCase, wholeWord }),
    [useRegex, matchCase, wholeWord, searchTerm]
  );

  const filteredSegments = useMemo(() => {
    let list = segments;
    if (problemFilter === 'ALL_PROBLEMS') list = list.filter(isAnyProblem);
    else if (problemFilter === 'UNTRANSLATED') list = list.filter(isUntranslated);
    else if (problemFilter === 'LATIN_LEFTOVER') list = list.filter(hasLatinLeftover);
    else if (problemFilter === 'HIGH_CPS') list = list.filter(isHighCps);

    if (tagFilter !== 'ALL') {
      if (tagFilter === 'TAGGED') {
        list = list.filter((s) => s.segment_type && s.segment_type !== 'DIALOGUE');
      } else {
        list = list.filter((s) => s.segment_type === tagFilter);
      }
    }
    if (speakerFilter && speakerFilter !== 'ALL') {
      list =
        speakerFilter === '__NONE__'
          ? list.filter((s) => !s.speaker)
          : list.filter((s) => s.speaker === speakerFilter);
    }
    if (!searchTerm.trim()) return list;
    if (regexError || !searchRegex) return [];
    return list.filter((s) => hasMatch(searchRegex, s.original_text) || hasMatch(searchRegex, s.translated_text));
  }, [
    segments,
    searchTerm,
    searchRegex,
    regexError,
    problemFilter,
    isAnyProblem,
    isUntranslated,
    hasLatinLeftover,
    isHighCps,
    tagFilter,
    speakerFilter,
  ]);

  return { filteredSegments, problemStats, tagCounts, regexError };
}
